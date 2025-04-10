#include "neural_mapping.h"
#include "utils/coordinates.h"

#ifdef ENABLE_ROS
#include <tf/transform_broadcaster.h>
#endif

#include <memory>
#include <opencv2/opencv.hpp>

#include "llog/llog.h"
#include "optimizer/loss.h"
#include "optimizer/loss_utils/loss_utils.h"
#include "params/params.h"
#include "utils/sensor_utils/cameras.hpp"
#include "utils/tqdm.hpp"

#include "kaolin_wisp_cpp/spc_ops/spc_ops.h"
#include "tracer/tracer.h"

#include "utils/ply_utils/ply_utils_torch.h"

using namespace std;

NeuralSLAM::NeuralSLAM(const int &mode,
                       const std::filesystem::path &_config_path,
                       const std::filesystem::path &_data_path) {
  cout << "TORCH_VERSION: " << TORCH_VERSION << '\n';

  read_params(_config_path, _data_path, mode);

  data_loader_ptr = std::make_unique<dataloader::DataLoader>(
      k_dataset_path, k_dataset_type, k_device, mode & k_preload, k_res_scale,
      k_sensor);
  if (k_export_colmap_format == 1) {
    data_loader_ptr->export_as_colmap_format(false, false);
  } else if (k_export_colmap_format == 2) {
    data_loader_ptr->export_as_colmap_format_for_nerfstudio(false);
  } else if (k_export_colmap_format == 3) {
    data_loader_ptr->export_as_colmap_format(true, false);
  }

  if (mode) {
    mapper_thread = std::thread(&NeuralSLAM::batch_train, this);
    keyboard_thread = std::thread(&NeuralSLAM::keyboard_loop, this);
    misc_thread = std::thread(&NeuralSLAM::misc_loop, this);
  } else {
    k_output_path = _config_path.parent_path().parent_path().parent_path();
    load_pretrained(k_output_path);
    train_callback(-1, RaySamples());
  }
}

void NeuralSLAM::load_pretrained(
    const std::filesystem::path &_pretrained_path) {
  load_checkpoint(_pretrained_path);

  auto as_prior_ply_file = _pretrained_path / "as_prior.ply";
  std::map<std::string, torch::Tensor> map_ply_tensor;
  if (ply_utils::read_ply_file_to_map_tensor(as_prior_ply_file, map_ply_tensor,
                                             torch::Device(torch::kCUDA))) {
    local_map_ptr->update_octree_as(map_ply_tensor["xyz"], true);

    auto dense_vox = local_map_ptr->p_acc_strcut_->get_quantized_points();
    auto normalized_fpts =
        spc_ops::quantized_points_to_fpoints(dense_vox, k_octree_level);
    auto fpts = local_map_ptr->m1p1_pts_to_xyz(normalized_fpts);
  } else {
    throw std::runtime_error("Failed to load " + as_prior_ply_file.string());
  }

#ifdef ENABLE_ROS
  mapper_thread = std::thread(&NeuralSLAM::pretrain_loop, this);
#endif
  keyboard_thread = std::thread(&NeuralSLAM::keyboard_loop, this);
}

DepthSamples NeuralSLAM::sample(DepthSamples _ray_samples, const int &iter,
                                const float &sample_std) {
  static auto p_t_utils_sample = llog::CreateTimer("   utils::sample");
  p_t_utils_sample->tic();
  // [n,num_samples]
  DepthSamples point_samples;
  _ray_samples.ridx = torch::arange(_ray_samples.size(0), k_device);
  _ray_samples.ray_sdf = torch::zeros({_ray_samples.size(0), 1}, k_device);
  point_samples = local_map_ptr->sample(_ray_samples, 1);

  DepthSamples surface_samples;
  utils::sample_surface_pts(_ray_samples, surface_samples, k_surface_sample_num,
                            sample_std);
  point_samples = point_samples.cat(surface_samples);

  auto trunc_idx = (point_samples.ray_sdf.abs() > k_truncated_dis)
                       .squeeze()
                       .nonzero()
                       .squeeze();
  point_samples.ray_sdf.index_put_(
      {trunc_idx},
      point_samples.ray_sdf.index({trunc_idx}).sign() * k_truncated_dis);

  point_samples = point_samples.cat(_ray_samples);

  llog::RecordValue("sdf_pt_n", point_samples.ray_sdf.size(0));

  p_t_utils_sample->toc_sum();
  return point_samples;
}

torch::Tensor NeuralSLAM::sdf_regularization(const torch::Tensor &xyz,
                                             const torch::Tensor &sdf,
                                             const bool &curvate_enable,
                                             const std::string &name) {
  auto grad_results =
      local_map_ptr->get_gradient(xyz, k_sample_std, sdf, curvate_enable);

  auto eik_loss = loss::eikonal_loss(grad_results[0], name);
  auto loss = k_eikonal_weight * eik_loss;
  llog::RecordValue(name + "_eik", eik_loss.item<float>());

  if (curvate_enable) {
    auto curv_loss = loss::curvate_loss(grad_results[1], name);
    loss += k_curvate_weight * curv_loss;
    llog::RecordValue(name + "_curv", curv_loss.item<float>());
  }
  return loss;
}

std::tuple<torch::Tensor, DepthSamples>
NeuralSLAM::sdf_train_batch_iter(const int &iter) {
  static auto p_t_sample = llog::CreateTimer("  sample");
  p_t_sample->tic();

  auto indices =
      (torch::rand({k_batch_num}) *
       data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(0))
          .to(torch::kLong)
          .clamp(0,
                 data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(
                     0) -
                     1);

  auto struct_ray_samples =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.index({indices}).to(
          k_device);
  auto point_samples = sample(struct_ray_samples, iter, k_sample_std);

  p_t_sample->toc_sum();

  static auto p_t_get_sdf = llog::CreateTimer("  get_sdf");
  p_t_get_sdf->tic();
  auto sdf_results = local_map_ptr->get_sdf(point_samples.xyz);
  point_samples.pred_sdf = sdf_results[0];
  point_samples.pred_isigma = sdf_results[1];

  auto sdf_loss = loss::sdf_loss(point_samples.pred_sdf, point_samples.ray_sdf,
                                 point_samples.pred_isigma);
  auto loss = k_sdf_weight * sdf_loss;
  llog::RecordValue("sdf", sdf_loss.item<float>());
  p_t_get_sdf->toc_sum();

  if (k_eikonal_weight > 0.0) {
    static bool curvate_enable = k_curvate_weight > 0.0;
    loss += sdf_regularization(point_samples.xyz, point_samples.pred_sdf,
                               curvate_enable, "pt");
  }

  return std::make_tuple(loss, point_samples);
}

torch::Tensor NeuralSLAM::color_train_batch_iter(const int &iter) {
  static auto p_t_get_rgb = llog::CreateTimer("  get_rgb");
  p_t_get_rgb->tic();

  ColorSamples color_ray_samples;
  torch::Tensor color_poses, h_idx, w_idx;
  static int train_num =
      data_loader_ptr->dataparser_ptr_->size(dataparser::DataType::TrainColor);
  if (k_preload) {
    static auto color_static = torch::tensor(
        {{train_num, data_loader_ptr->dataparser_ptr_->sensor_.camera.height,
          data_loader_ptr->dataparser_ptr_->sensor_.camera.width}},
        torch::kFloat);
    auto indices =
        (torch::rand({k_batch_num, 3}) * color_static).to(torch::kLong);
    auto img_idx = indices.select(1, 0);
    h_idx = indices.select(1, 1);
    w_idx = indices.select(1, 2);

    color_ray_samples.rgb = data_loader_ptr->dataparser_ptr_->train_color_
                                .index({img_idx, h_idx, w_idx})
                                .view({-1, 3})
                                .to(k_device)
                                .contiguous();
    // [N, 3, 4]
    color_poses = data_loader_ptr->dataparser_ptr_
                      ->get_pose(img_idx, dataparser::DataType::TrainColor)
                      .to(k_device);
  } else {
    static auto color_static = torch::tensor(
        {{data_loader_ptr->dataparser_ptr_->sensor_.camera.height,
          data_loader_ptr->dataparser_ptr_->sensor_.camera.width}},
        torch::kFloat);
    auto indices =
        (torch::rand({k_batch_num, 2}) * color_static).to(torch::kLong);
    auto img_idx = std::rand() % train_num;
    h_idx = indices.select(1, 0);
    w_idx = indices.select(1, 1);
    color_ray_samples.rgb =
        data_loader_ptr->dataparser_ptr_
            ->get_image(img_idx, dataparser::DataType::TrainColor)
            .index({h_idx, w_idx})
            .view({-1, 3})
            .to(k_device)
            .contiguous();
    // [N, 3, 4]
    color_poses = data_loader_ptr->dataparser_ptr_
                      ->get_pose(img_idx, dataparser::DataType::TrainColor)
                      .unsqueeze(0)
                      .repeat({k_batch_num, 1, 1})
                      .to(k_device);
  }

  // [N, 3, 1]
  color_ray_samples.origin =
      color_poses.slice(2, 3, 4).view({-1, 3}).contiguous();
  auto camera_ray_results_ =
      data_loader_ptr->dataparser_ptr_->sensor_.camera
          .generate_rays_from_coords(torch::stack({h_idx, w_idx}, 1) + 0.5f);
  color_ray_samples.direction =
      camera_ray_results_[0].view({-1, 3, 1}).to(k_device);

  // [N, 3, 3]
  auto rot = color_poses.slice(2, 0, 3).view({-1, 3, 3});
  color_ray_samples.direction =
      (rot.matmul(color_ray_samples.direction)).squeeze(-1).contiguous();

  auto trace_results =
      tracer::render_ray(local_map_ptr, color_ray_samples.origin,
                         color_ray_samples.direction, 1, k_trace_iter);

  auto loss = torch::tensor(0.0f, k_device);
  if (!trace_results.empty()) {
    auto color_loss = loss::rgb_loss(trace_results[0], color_ray_samples.rgb);
    loss += k_rgb_weight * color_loss;
    llog::RecordValue("color", color_loss.item<float>());

    float c_pt_n = trace_results[3].size(0);
    float sample_pts_per_ray = c_pt_n / (float)k_batch_num;
    k_sample_pts_per_ray =
        k_sample_pts_per_ray * 0.9 + sample_pts_per_ray * 0.1;
    llog::RecordValue("pts_per_ray", k_sample_pts_per_ray);
    static int max_batch_num = 5 * k_color_batch_pt_num / k_trace_iter;
    k_batch_num =
        min(k_color_batch_pt_num / k_sample_pts_per_ray, max_batch_num);

    llog::RecordValue("c_ray_n", k_batch_num);
    llog::RecordValue("c_pt_n", c_pt_n);

    if (k_eikonal_weight > 0.0) {
      loss +=
          sdf_regularization(trace_results[3], trace_results[4], false, "c");
    }

    if (k_dist_weight > 0.0f) {
      auto dist_loss = loss::distortion_loss(trace_results[11]);
      loss += k_dist_weight * dist_loss;
      llog::RecordValue("dist", dist_loss.item<float>());
    }
  }

  p_t_get_rgb->toc_sum();
  return loss;
}

void NeuralSLAM::train(int _opt_iter) {
  k_batch_num = k_batch_ray_num;
  k_sample_pts_per_ray = k_color_batch_pt_num / (float)k_batch_num;
  c10::cuda::CUDACachingAllocator::emptyCache();
  torch::GradMode::set_enabled(true);
  auto iter_bar = tq::trange(_opt_iter);
  for (int i : iter_bar) {
    torch::Tensor loss = torch::tensor(0.0, k_device);

    DepthSamples point_samples;
    if (k_sdf_weight > 0.0) {
      torch::Tensor sdf_loss = torch::tensor(0.0, k_device);
      std::tie(sdf_loss, point_samples) = sdf_train_batch_iter(i);
      loss += sdf_loss;
    }

    if (k_rgb_weight > 0) {
      auto color_loss = color_train_batch_iter(i);
      loss += color_loss;
    }

    static auto p_t_backward = llog::CreateTimer("  backward");
    p_t_backward->tic();
    static torch::autograd::AnomalyMode anomaly_mode;
    anomaly_mode.set_enabled(k_debug);
    p_optimizer_->zero_grad();
    loss.backward();
    p_optimizer_->step();
    p_t_backward->toc_sum();

    if (k_sdf_weight > 0) {
      train_callback(i, point_samples);
    } else {
      train_callback(i, RaySamples());
    }

    llog::RecordValue("loss", loss.item<float>());
    static std::string value_path = k_output_path / "loss_log.txt";
    iter_bar << llog::FlashValue(value_path, 3);

#ifdef ENABLE_ROS
    if ((i + 1) % k_vis_frame_step == 0) {
      visualization(point_samples.xyz);
    }
#endif
    if (i % k_export_interval == 0) {
      misc_trigger = i;
    }
  }
}

void NeuralSLAM::train_callback(const int &_iter,
                                const RaySamples &_point_samples) {
  torch::NoGradGuard no_grad;
  if (_iter == -1) {
    k_t = 1.0f;
    k_sample_std = 1e-2f;

    int dir_feat_dim = k_dir_embedding_degree * k_dir_embedding_degree;
    local_map_ptr->dir_mask =
        torch::ones({1, dir_feat_dim}, torch::kFloat32).to(k_device);
    return;
  } else {
    k_t = (float)_iter / k_iter_step;
  }

  float lr = k_lr * (1 - k_t) + k_lr_end * k_t;
  llog::RecordValue("lr", lr);

  for (auto &pg : p_optimizer_->param_groups()) {
    if (pg.has_options()) {
      pg.options().set_lr(lr);
    }
  }
  // extremely importance to avoid color degenerate structure in early stage
  k_rgb_weight = k_rgb_weight_init * (1 - k_t) + k_rgb_weight_end * k_t;

  if (_point_samples.pred_isigma.numel() > 0) {
    k_sample_std = (1.0 / _point_samples.pred_isigma).mean().item<float>();
    llog::RecordValue("sstd", k_sample_std);
    k_sample_std = max(k_sample_std, 1e-2f);
  }

  if (k_outlier_remove && (_iter > 0)) {
    if (_iter % k_outlier_removal_interval == 0) {
      static auto p_t_outlier_remove = llog::CreateTimer(" outlier_remove");
      p_t_outlier_remove->tic();
      // batch remove outlier
      std::vector<torch::Tensor> inlier_idx_vec;
      auto tmp_outlier_dist =
          exp(log(k_truncated_dis) * (1 - k_t) + log(k_outlier_dist) * k_t);
      auto train_pcl =
          data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});
      int point_num = train_pcl.size(0);
      for (int i = 0; i < point_num; i += k_vis_batch_pt_num) {
        auto end = min(i + k_vis_batch_pt_num, point_num - 1);
        auto xyz_pred_sdf = local_map_ptr->get_sdf(
            train_pcl.index_select(0, {torch::arange(i, end)}).to(k_device))[0];
        auto inlier_idx = ((xyz_pred_sdf.abs() < tmp_outlier_dist).squeeze())
                              .nonzero()
                              .squeeze()
                              .cpu();
        inlier_idx_vec.emplace_back(inlier_idx + i);
      }
      auto inlier_idx = torch::cat(inlier_idx_vec, 0);
      cout << "\nOutlier Removal(" << tmp_outlier_dist << "): "
           << data_loader_ptr->dataparser_ptr_->train_depth_pack_.size(0);

      data_loader_ptr->dataparser_ptr_->train_depth_pack_ =
          data_loader_ptr->dataparser_ptr_->train_depth_pack_.index_select(
              0, inlier_idx);
      cout << " -> "
           << data_loader_ptr->dataparser_ptr_->train_depth_pack_.size(0)
           << '\n';
      p_t_outlier_remove->toc_sum();
    }
  }

  if (_iter % 1000 == 0) {
    static int dir_degree = 0;
    dir_degree++;
    if (dir_degree <= k_dir_embedding_degree) {
      int dir_feat_dim = dir_degree * dir_degree;
      auto options = local_map_ptr->dir_mask.options();
      local_map_ptr->dir_mask =
          torch::cat({torch::ones({1, dir_feat_dim}, options),
                      torch::zeros(
                          {1, k_dir_embedding_degree * k_dir_embedding_degree -
                                  dir_feat_dim},
                          options)},
                     1)
              .to(k_device);

      std::cout << "\nExtend directional embedding from:"
                << ((dir_degree - 1) * (dir_degree - 1))
                << " to:" << dir_feat_dim << std::endl;
    }
  }

  if (_iter % k_export_interval == 0) {
    if (k_rgb_weight > 0) {
      static int test_idx = 0;
      auto psnr = export_test_image(test_idx, std::to_string(_iter) + "_");
      llog::RecordValue("psnr", psnr);
    }
  }
  if (_iter == k_export_ckp_interval) {
    export_checkpoint();
  }
}

void NeuralSLAM::prefilter_data(const bool &export_img) {
  std::vector<int> valid_ids_vec;
  std::vector<std::filesystem::path> filtered_train_color_filelists;
  int color_size =
      data_loader_ptr->dataparser_ptr_->size(dataparser::DataType::TrainColor);

  // [H, W, 3]
  auto pre_color = data_loader_ptr->dataparser_ptr_
                       ->get_image(0, dataparser::DataType::TrainColor)
                       .to(k_device);
  valid_ids_vec.emplace_back(0);
  filtered_train_color_filelists.emplace_back(
      data_loader_ptr->dataparser_ptr_->get_file(
          0, dataparser::DataType::TrainColor));

  std::filesystem::path fileter_color_path = k_output_path / "filter";
  if (export_img) {
    std::filesystem::create_directories(fileter_color_path);
    auto gt_color_file_name =
        data_loader_ptr->dataparser_ptr_
            ->get_file(0, dataparser::DataType::TrainColor)
            .filename()
            .replace_extension(".png");
    cv::imwrite(fileter_color_path / gt_color_file_name,
                utils::tensor_to_cv_mat(pre_color));
  }

  auto iter_bar = tq::trange(1, color_size);
  iter_bar.set_prefix("Prefilter Data");
  for (int i : iter_bar) {
    auto now_color = data_loader_ptr->dataparser_ptr_
                         ->get_image(i, dataparser::DataType::TrainColor)
                         .to(k_device);
    // auto metric = loss_utils::ssim(pre_color.permute({2, 0, 1}).unsqueeze(0),
    //                              now_color.permute({2, 0, 1}).unsqueeze(0))
    //                 .item<float>();

    auto metric = loss_utils::psnr(pre_color.permute({2, 0, 1}).unsqueeze(0),
                                   now_color.permute({2, 0, 1}).unsqueeze(0));
    if (metric < k_prefilter) {
      valid_ids_vec.emplace_back(i);
      filtered_train_color_filelists.emplace_back(
          data_loader_ptr->dataparser_ptr_->get_file(
              i, dataparser::DataType::TrainColor));
      pre_color = now_color;

      if (export_img) {
        auto gt_color_file_name =
            data_loader_ptr->dataparser_ptr_
                ->get_file(i, dataparser::DataType::TrainColor)
                .filename()
                .replace_extension(".png");
        cv::imwrite(fileter_color_path / gt_color_file_name,
                    utils::tensor_to_cv_mat(pre_color));
      }
    }
  }

  data_loader_ptr->dataparser_ptr_->train_color_filelists_ =
      filtered_train_color_filelists;

  auto valid_ids = torch::tensor(valid_ids_vec);
  auto reshape_color_pose_sizes =
      data_loader_ptr->dataparser_ptr_->train_color_poses_.sizes().vec();
  reshape_color_pose_sizes[0] = -1;
  data_loader_ptr->dataparser_ptr_->train_color_poses_ =
      data_loader_ptr->dataparser_ptr_->train_color_poses_
          .index_select(0, valid_ids)
          .reshape(reshape_color_pose_sizes);
  if (k_preload) {
    auto reshape_color_sizes =
        data_loader_ptr->dataparser_ptr_->train_color_.sizes().vec();
    reshape_color_sizes[0] = -1;
    data_loader_ptr->dataparser_ptr_->train_color_ =
        data_loader_ptr->dataparser_ptr_
            ->get_image(valid_ids, dataparser::DataType::TrainColor)
            .reshape(reshape_color_sizes);
  }

  std::cout << "Original color size: " << color_size << ", after filtering: "
            << data_loader_ptr->dataparser_ptr_->size(
                   dataparser::DataType::TrainColor)
            << std::endl;
}

/**
 * @brief Builds an occupancy map from depth and color data
 *
 * This function processes raw depth data to create an occupancy map
 * representation. If probability mapping is enabled, it uses ROG-Map for
 * enhanced occupancy mapping. The function supports two main operations:
 * 1. Depth raycasting: Creates points from depth measurements
 * 2. Color raycasting: Uses color camera rays to identify unknown areas (when
 * prob_map_en is true)
 *
 * @return bool Success status of the map building operation
 */
bool NeuralSLAM::build_occ_map() {
  std::cout << "Starting occupancy map building from raw data...\n";
  static auto timer_cal_prior = llog::CreateTimer("cal_prior");
  timer_cal_prior->tic();

  auto pcl_depth =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.view({-1});
  auto valid_mask = (pcl_depth > k_min_range) & (pcl_depth < k_max_range);
  auto valid_idx = valid_mask.nonzero().squeeze();

  auto pcl =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3})
          .index_select(0, valid_idx);
  // calculate pcl's center
  // and radius
  auto pcl_center = pcl.mean(0).squeeze();
  auto pcl_radius = (pcl - pcl_center).norm(2, 1).max().item<float>();

  std::cout << "PCL center: " << pcl_center << ", radius: " << pcl_radius
            << '\n';

  // set local map's center and radius
  {
    k_map_origin = pcl_center;
    if (k_inner_map_size < pcl_radius * 2.0f) {
      std::cout << "Warning: inner map size is smaller than pcl radius * 2.0\n";
    } else {
      k_inner_map_size = pcl_radius * 2.0f;
    }
    k_x_max = 0.5f * k_inner_map_size;
    k_y_max = k_x_max;
    k_z_max = k_x_max;
    k_x_min = -0.5f * k_inner_map_size;
    k_y_min = k_x_min;
    k_z_min = k_x_min;
    k_octree_level =
        ceil(log2((k_inner_map_size + 2 * k_leaf_size) * k_leaf_size_inv));
    k_map_resolution = std::pow(2, k_octree_level);
    k_map_size = k_map_resolution * k_leaf_size;
    k_map_size_inv = 1.0f / k_map_size;

    if (k_fill_level > k_octree_level) {
      k_fill_level = k_octree_level;
    }
  }

  local_map_ptr = std::make_shared<LocalMap>(
      k_map_origin, k_x_min, k_x_max, k_y_min, k_y_max, k_z_min, k_z_max);
  for (auto &p : local_map_ptr->named_parameters()) {
    cout << p.key() << p.value().sizes() << '\n';
  }

  auto inrange_idx =
      local_map_ptr->get_inrange_mask(pcl).squeeze().nonzero().squeeze();
  auto inrange_pt = pcl.index_select(0, inrange_idx).cuda();

  //----------------------------------------------------------------------
  // SECTION 1: Process depth data for occupancy mapping
  //----------------------------------------------------------------------
  if (k_prob_map_en) {
    // Initialize ROG-Map configuration
    auto rog_map_cfg_file = k_package_path / "config/ROG-Map/rog.yaml";
    std::cout << "Loading ROG-Map config from: " << rog_map_cfg_file
              << std::endl;

    cv::FileStorage fsSettings(rog_map_cfg_file, cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
      std::cerr << "ERROR: Could not open settings at: " << rog_map_cfg_file
                << std::endl;
      exit(-1);
    }

    // Create and configure ROG-Map settings
    rog_map::ROGMapConfig cfg;

    // Basic map settings
    cfg.resolution = k_leaf_size;
    cfg.inflation_resolution = k_leaf_size;
    cfg.safe_margin = 0.1f;
    cfg.block_inf_pt = true;
    cfg.map_sliding_en = false;
    cfg.frontier_extraction_en = false;
    cfg.inflation_en = false;
    cfg.inflation_step = 1;
    cfg.intensity_thresh = -1;
    cfg.point_filt_num = 1;

    // Map size and origin
    cfg.fix_map_origin = {k_map_origin[0].item<float>(),
                          k_map_origin[1].item<float>(),
                          k_map_origin[2].item<float>()};
    cfg.map_size_d = {k_inner_map_size, k_inner_map_size, k_inner_map_size};

    // Map sliding threshold
    fsSettings["fsm_node"]["rog_map"]["map_sliding"]["threshold"] >>
        cfg.map_sliding_thresh;

    // Raycasting configuration
    cfg.raycasting_en = true;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["batch_update_size"] >>
        cfg.batch_update_size;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]
              ["inf_map_known_free_thresh"] >>
        cfg.known_free_thresh;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_hit"] >> cfg.p_hit;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_miss"] >> cfg.p_miss;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_min"] >> cfg.p_min;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_max"] >> cfg.p_max;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_occ"] >> cfg.p_occ;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["p_free"] >> cfg.p_free;

    // Ray range settings
    cv::Mat cv_ray_range;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["ray_range"] >>
        cv_ray_range;
    cfg.raycast_range_min = cv_ray_range.at<double>(0);
    cfg.raycast_range_max = cv_ray_range.at<double>(1);

    // Local update box
    cv::Mat cv_local_update_box;
    fsSettings["fsm_node"]["rog_map"]["raycasting"]["local_update_box"] >>
        cv_local_update_box;
    cfg.local_update_box_d = {cv_local_update_box.at<double>(0),
                              cv_local_update_box.at<double>(1),
                              cv_local_update_box.at<double>(2)};

    // Height constraints
    fsSettings["fsm_node"]["rog_map"]["virtual_ground_height"] >>
        cfg.virtual_ground_height;
    fsSettings["fsm_node"]["rog_map"]["virtual_ceil_height"] >>
        cfg.virtual_ceil_height;

    // GPU settings
    fsSettings["fsm_node"]["rog_map"]["gpu"]["GPU_BLOCKSIZE"] >>
        cfg.GPU_BLOCKSIZE;
    fsSettings["fsm_node"]["rog_map"]["gpu"]["CLOUD_BUFFER_SIZE"] >>
        cfg.CLOUD_BUFFER_SIZE;

    // Initialize ROG-Map
    rog_map::ROSParamLoader ros_param_loader;
    ros_param_loader.initConfig(cfg);

    rog_map_ptr = std::make_shared<rog_map::ROGMap>(cfg);

    // Create progress bar for depth processing
    auto depth_iter_bar = tq::trange(
        data_loader_ptr->dataparser_ptr_->train_depth_poses_.size(0));
    depth_iter_bar.set_prefix("Depth RayCasting");

    for (auto i : depth_iter_bar) {
      // Get camera pose and corresponding XYZ points
      auto pose =
          data_loader_ptr->dataparser_ptr_->get_pose(i, dataparser::TrainDepth);
      auto xyz = data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz[i];

#ifdef ENABLE_ROS
      // Publish pose and pointcloud for visualization if ROS is enabled
      pub_pose(pose);
      pub_pointcloud(pointcloud_pub, xyz);
#endif

      // If probability mapping is enabled, update the ROG-Map

      // Note: If an illegal memory access error occurs, the pointcloud may be
      // too dense
      rog_map::PointCloudHost rog_cloud;
      rog_map::Pose rog_pose;

      // Convert pose to ROG-Map format
      auto pos = pose.slice(1, 3, 4).squeeze().cpu();
      rog_pose.first.x() = pos[0].item<float>();
      rog_pose.first.y() = pos[1].item<float>();
      rog_pose.first.z() = pos[2].item<float>();

      // Convert tensor to pointcloud and update ROG-Map
      rog_cloud = utils::tensor_to_pointcloud(xyz);
      rog_map_ptr->updateMap(rog_cloud, rog_pose);
    }

    //----------------------------------------------------------------------
    // SECTION 2: Process occupancy data from probability map (if enabled)
    //----------------------------------------------------------------------

    // Extract occupied cells from ROG-Map
    type_utils::vec_E<type_utils::Vec3f> prob_occ_map;
    rog_map_ptr->boxSearchBatch(rog_map_ptr->local_map_bound_min_d_,
                                rog_map_ptr->local_map_bound_max_d_,
                                rog_map::GridType::OCCUPIED, prob_occ_map);

    // Convert to tensor format
    auto occupancy_points = utils::vec_eigen_to_tensor(prob_occ_map);

#ifdef ENABLE_ROS
    // Publish occupied points for visualization
    pub_pointcloud(occ_pub, occupancy_points);
#endif

    // Combine ROG-Map occupied points with depth points
    occupancy_points = torch::cat(
        {occupancy_points,
         data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3})},
        0);

    // Update octree with combined points
    local_map_ptr->update_octree_as(occupancy_points.cuda());

    //----------------------------------------------------------------------
    // SECTION 3: Process color rays for unknown area discovery (prob_map only)
    //----------------------------------------------------------------------

    ColorSamples color_rays;
    ColorSamplesVec color_rays_vec;

    // Calculate batch processing parameters
    int color_width = data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
    int color_height = data_loader_ptr->dataparser_ptr_->sensor_.camera.height;
    int rays_per_frame = color_width * color_height;
    int frames_per_batch = k_vis_batch_pt_num / rays_per_frame;

    // Create progress bar for color processing
    auto color_iter_bar = tq::trange(data_loader_ptr->dataparser_ptr_->size(
        dataparser::DataType::TrainColor));
    color_iter_bar.set_prefix("Color Raycasting");

    for (int i : color_iter_bar) {
      // Get camera pose and generate rays
      auto pose = data_loader_ptr->dataparser_ptr_->get_pose(
          i, dataparser::DataType::TrainColor);

#ifdef ENABLE_ROS
      pub_pose(pose);
#endif

      // Generate camera rays for this frame
      auto camera_ray_results =
          data_loader_ptr->dataparser_ptr_->sensor_.camera.generate_rays(pose);
      color_rays.origin = camera_ray_results[0];
      color_rays.direction = camera_ray_results[1];
      color_rays_vec.emplace_back(color_rays.to(torch::kCPU));

      // Process rays in batches to manage memory usage
      bool is_last_frame = (i == data_loader_ptr->dataparser_ptr_->size(0) - 1);
      if ((i % frames_per_batch == 0) || is_last_frame) {
        // Combine collected rays
        auto batch_color_rays = color_rays_vec.cat();
        color_rays_vec.clear(); // Free memory

        // Prepare ray data for ROG-Map
        rog_map::PointCloudHost ray_endpoints, ray_origins;
        ray_endpoints = utils::tensor_to_pointcloud(batch_color_rays.origin +
                                                    batch_color_rays.direction);
        ray_origins = utils::tensor_to_pointcloud(batch_color_rays.origin);

        // Search for unknown cells along rays
        type_utils::vec_E<type_utils::Vec3f> unknown_cells;
        rog_map_ptr->raySearch(ray_endpoints, ray_origins,
                               rog_map::GridType::UNKNOWN, unknown_cells);
        auto unknown_points = utils::vec_eigen_to_tensor(unknown_cells);

        // Filter unknown points based on octree validity for efficiency, i
        if (occupancy_points.numel() > 0) {
          // Move unknown points to GPU for processing
          unknown_points = unknown_points.cuda();

          // Convert to quantized points and find corners
          auto quantized_points =
              (unknown_points * k_leaf_size_inv).floor().to(torch::kInt16);
          auto corner_points =
              spc_ops::points_to_corners(quantized_points).view({-1, 3});
          auto corner_vertices =
              corner_points.to(torch::kFloat32) * k_leaf_size;

          // Get valid mask based on octree structure
          auto valid_mask = local_map_ptr->get_valid_mask(corner_vertices, -1)
                                .view({-1, 8})
                                .any(1);

          // Filter points using the mask
          unknown_points = unknown_points.index({valid_mask}).cpu();
        }
        // Combine with existing occupancy points
        if (occupancy_points.numel() == 0) {
          occupancy_points = unknown_points;
        } else {
          occupancy_points = torch::cat({occupancy_points, unknown_points}, 0);
        }
      }
    }

    // Clean up ROG-Map resources
    rog_map_ptr.reset();

    // Update octree with final point set
    local_map_ptr->update_octree_as(occupancy_points.cuda());
  } else {
    // If probability map is disabled, just use depth points directly
    local_map_ptr->update_octree_as(
        data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3})
            .cuda());
  }

  //----------------------------------------------------------------------
  // SECTION 4: Extract and save final points from octree
  //----------------------------------------------------------------------

  // Get quantized points from acceleration structure
  auto dense_voxels = local_map_ptr->p_acc_strcut_->get_quantized_points();

  // Convert quantized points to normalized floating points
  auto normalized_points =
      spc_ops::quantized_points_to_fpoints(dense_voxels, k_octree_level);

  // Convert normalized points to world coordinates
  auto world_points = local_map_ptr->m1p1_pts_to_xyz(normalized_points);

  // Export points to PLY file
  std::string as_prior_ply_file = k_output_path / "as_prior.ply";
  ply_utils::export_to_ply(as_prior_ply_file, world_points.cpu());

  timer_cal_prior->toc_sum();

  //----------------------------------------------------------------------
  // SECTION 5: Reshape depth data for further processing
  //----------------------------------------------------------------------

  // Reshape origin data
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.origin =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.origin.unsqueeze(1)
          .repeat(
              {1,
               data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.size(
                   1),
               1})
          .view({-1, 3});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.depth.view({-1, 1});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.direction =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.direction.view(
          {-1, 3});
  data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz =
      data_loader_ptr->dataparser_ptr_->train_depth_pack_.xyz.view({-1, 3});
  return true;
}

void NeuralSLAM::batch_train() {
  if (k_prefilter > 0) {
    static auto p_prefilter_data_timer = llog::CreateTimer("prefilter_data");
    p_prefilter_data_timer->tic();
    prefilter_data(false);
    p_prefilter_data_timer->toc_sum();
  }
  static auto p_timer = llog::CreateTimer("build_occ_map");
  p_timer->tic();
  build_occ_map();

  torch::optim::AdamOptions adam_options;
  adam_options.lr(k_lr);
  adam_options.amsgrad(true);
  adam_options.eps(1e-15);
  p_optimizer_ = std::make_shared<torch::optim::Adam>(
      local_map_ptr->parameters(), adam_options);
  p_timer->toc_sum();

  static auto p_train = llog::CreateTimer("train");
  p_train->tic();

  train(k_iter_step);

  p_train->toc_sum();

  // save current cpu and gpu memory usage to file, it extremely slow
  static ofstream mem_file(k_output_path / "mem_usage.txt");
  mem_file << "frame_num\tcpu_mem_usage\tgpu_mem_usage\ttiming\n";
  mem_file << utils::get_cpu_mem_usage() << "\t" << utils::get_gpu_mem_usage()
           << "\t" << p_train->this_time() << '\n';

  llog::PrintLog();

  end();
}

/* Exporter */
std::vector<torch::Tensor> NeuralSLAM::render_image(const torch::Tensor &_pose,
                                                    const float &_scale,
                                                    const bool &training) {
  static auto p_timer_render = llog::CreateTimer("    render_image");
  p_timer_render->tic();
  static auto timer_generate_rays = llog::CreateTimer("     generate_rays");
  timer_generate_rays->tic();

  auto camera_ray_results =
      data_loader_ptr->dataparser_ptr_->sensor_.camera.generate_rays(_pose,
                                                                     _scale);
  RaySamples ray_samples;
  ray_samples.origin = camera_ray_results[0];
  ray_samples.direction = camera_ray_results[1];
  timer_generate_rays->toc_sum();

  static auto p_t_trace = llog::CreateTimer("     trace");
  p_t_trace->tic();

  int ray_num = ray_samples.origin.size(0);
  std::vector<torch::Tensor> render_colors_vec, render_depths_vec,
      render_accs_vec, render_scale_vec, render_num_vec, sdf_vec, alphas_vec;
  float batch_size = k_batch_ray_num;
  float sample_pts_per_ray = k_vis_batch_pt_num / batch_size;
  for (int i = 0; i < ray_num;) {
    int end = min(i + batch_size, ray_num);
    batch_size = end - i;
    auto batch_ray_samples = ray_samples.slice(0, i, end).to(k_device);
    static auto p_t_render_ray = llog::CreateTimer("      render_ray");
    p_t_render_ray->tic();
    auto trace_results =
        tracer::render_ray(local_map_ptr, batch_ray_samples.origin,
                           batch_ray_samples.direction, 0, k_trace_iter);
    p_t_render_ray->toc_sum();
    if (!trace_results.empty()) {
      render_colors_vec.emplace_back(trace_results[0].to(_pose.device()));
      render_depths_vec.emplace_back(trace_results[1].to(_pose.device()));
      render_accs_vec.emplace_back(trace_results[2].to(_pose.device()));
      render_scale_vec.emplace_back(trace_results[9].to(_pose.device()));
      render_num_vec.emplace_back(trace_results[10].to(_pose.device()));
    } else {
      return {};
    }
    i = end;
    float c_pt_n = trace_results[3].size(0);
    float tmp_sample_pts_per_ray = c_pt_n / (float)batch_size;
    sample_pts_per_ray =
        sample_pts_per_ray * 0.9f + tmp_sample_pts_per_ray * 0.1f;
    batch_size = k_vis_batch_pt_num / sample_pts_per_ray;
  }
  auto render_colors = torch::cat(render_colors_vec, 0);
  auto render_depths = torch::cat(render_depths_vec, 0);
  auto render_accs = torch::cat(render_accs_vec, 0);
  auto render_scales = torch::cat(render_scale_vec, 0);
  auto render_num = torch::cat(render_num_vec, 0);

  p_t_trace->toc_sum();
  p_timer_render->toc_sum();

  long scale_height =
      _scale * data_loader_ptr->dataparser_ptr_->sensor_.camera.height;
  long scale_width =
      _scale * data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
  return {render_colors.view({scale_height, scale_width, 3}),
          render_depths.view({scale_height, scale_width, 1}),
          render_accs.view({scale_height, scale_width, 1}),
          render_scales.view({scale_height, scale_width, 1}),
          render_num.view({scale_height, scale_width, 1})};
}

void NeuralSLAM::create_dir(const std::filesystem::path &base_path,
                            std::filesystem::path &color_path,
                            std::filesystem::path &depth_path,
                            std::filesystem::path &gt_color_path,
                            std::filesystem::path &render_color_path,
                            std::filesystem::path &gt_depth_path,
                            std::filesystem::path &render_depth_path,
                            std::filesystem::path &acc_path) {
  color_path = base_path / "color";
  depth_path = base_path / "depth";
  gt_color_path = color_path / "gt";
  std::filesystem::create_directories(gt_color_path);
  render_color_path = color_path / "renders";
  std::filesystem::create_directories(render_color_path);
  gt_depth_path = depth_path / "gt";
  std::filesystem::create_directories(gt_depth_path);
  render_depth_path = depth_path / "renders";
  std::filesystem::create_directories(render_depth_path);
  acc_path = base_path / "acc";
  std::filesystem::create_directories(acc_path);
}

void NeuralSLAM::render_path(bool eval, const int &fps, const bool &save) {
  // Early exit if evaluation requested with no evaluation data
  if (eval && (data_loader_ptr->dataparser_ptr_->size(
                   dataparser::DataType::EvalColor) == 0)) {
    return;
  }

  torch::GradMode::set_enabled(false);
  c10::cuda::CUDACachingAllocator::emptyCache();

  // Determine image type and create necessary directories once
  int image_type;
  std::string image_type_name;
  std::filesystem::path output_dir, gt_color_path, render_color_path,
      gt_depth_path, render_depth_path, acc_path;

  if (!eval) {
    image_type = dataparser::DataType::RawColor;
    image_type_name = "train";
    output_dir = k_output_path / "train";
  } else {
    image_type = dataparser::DataType::EvalColor;
    image_type_name = "eval";
    output_dir = k_output_path / "eval";
  }

  // Only create directories if saving is enabled
  if (save) {
    std::filesystem::path color_path = output_dir / "color";
    std::filesystem::path depth_path = output_dir / "depth";
    gt_color_path = color_path / "gt";
    render_color_path = color_path / "renders";
    gt_depth_path = depth_path / "gt";
    render_depth_path = depth_path / "renders";
    acc_path = output_dir / "acc";

    // Create all directories at once instead of in create_dir function
    std::vector<std::filesystem::path> paths = {
        gt_color_path, render_color_path, gt_depth_path, render_depth_path,
        acc_path};
    for (const auto &path : paths) {
      std::filesystem::create_directories(path);
    }

    // If not eval, create test directories too
    if (!eval) {
      std::filesystem::path test_dir = k_output_path / "test";
      for (const auto &subdir :
           {"color/gt", "color/renders", "depth/gt", "depth/renders", "acc"}) {
        std::filesystem::create_directories(test_dir / subdir);
      }
    }
  }

  // Prepare a queue for background image writing
  struct ImageSaveTask {
    unsigned long index;
    int image_type;
    std::filesystem::path base_dir;
    torch::Tensor render;
  };
  std::vector<ImageSaveTask> save_queue;
  int reserve_size = 16;
  save_queue.reserve(reserve_size); // Pre-allocate to avoid reallocations

  // Periodically flush save queue to avoid excessive memory usage
  auto flush_save_queue = [&save_queue, this]() {
#pragma omp parallel for
    for (const auto &task : save_queue) {
      int offset = 0;
      std::string task_type;
      if (task.image_type % 2 == 0) {
        task_type = "color";
      } else {
        offset = -1;
        task_type = "depth";
      }
      auto output_path = task.base_dir / task_type;
      auto gt_file = data_loader_ptr->dataparser_ptr_->get_file(
          task.index, task.image_type + offset);
      auto gt_file_name = gt_file.filename().replace_extension(".png");
      auto render_file = task.base_dir / task_type / "renders" / gt_file_name;
      auto cv_render = utils::tensor_to_cv_mat(task.render);
      auto gt = data_loader_ptr->dataparser_ptr_->get_image_cv_mat(
          task.index, task.image_type);
      gt_file = task.base_dir / task_type / "gt" / gt_file_name;
      if (offset == 0) {
        cv::imwrite(render_file, cv_render);
        cv::imwrite(gt_file, gt);
      } else {
        cv::imwrite(render_file, utils::apply_colormap_to_depth(cv_render));
        if (!gt.empty()) {
          auto gt_depth_colormap = utils::apply_colormap_to_depth(gt);
          cv::imwrite(gt_file, gt_depth_colormap);
        }
      }
    }
    save_queue.clear();
  };

  auto iter_bar =
      tq::trange(data_loader_ptr->dataparser_ptr_->size(image_type));
  iter_bar.set_prefix("Rendering");
  static auto p_timer_render = llog::CreateTimer("    render_train_image");

  // LLF skip flag - compute once rather than checking in loop
  bool llff_skip_enabled = (k_prefilter > 0) && k_llff;

  for (const auto &i : iter_bar) {
    // Skip frames if necessary (computed once for efficiency)
    if (llff_skip_enabled && (i % 8 != 0)) {
      continue;
    }

    // Render the image
    p_timer_render->tic();
    auto pose = data_loader_ptr->dataparser_ptr_->get_pose(i, image_type)
                    .slice(0, 0, 3);
    auto render_results = render_image(pose, 1.0, false);
    p_timer_render->toc_sum();

    // Skip processing if render failed or saving disabled
    if (render_results.empty() || !save) {
      continue;
    }

    // Queue images for saving - use test path for certain frames if needed
    bool is_test = (!eval) && k_llff && (i % 8 == 0);
    std::filesystem::path base_dir =
        is_test ? (k_output_path / "test") : output_dir;

    // Queue image saving tasks
    save_queue.push_back(
        {i, image_type, base_dir, render_results[0].clamp(0.0f, 1.0f)});
    save_queue.push_back({i, image_type + 1, base_dir, render_results[1]});

    // Flush save queue periodically to avoid excessive memory usage
    if (save_queue.size() >= reserve_size) {
      flush_save_queue();
    }
  }

  // Final flush of save queue
  flush_save_queue();
}

void NeuralSLAM::render_path(std::string pose_file, const int &fps) {
  torch::NoGradGuard no_grad;
  std::cout << pose_file << std::endl;
  auto poses =
      data_loader_ptr->dataparser_ptr_->load_poses(pose_file, false, 0)[0];
  if (poses.size(0) == 0) {
    std::cout << "poses is empty" << std::endl;
    return;
  }
  c10::cuda::CUDACachingAllocator::emptyCache();

  std::filesystem::path color_path, depth_path, gt_color_path,
      render_color_path, gt_depth_path, render_depth_path, acc_path;

  int image_type;
  std::string image_type_name;
  image_type = 0;
  image_type_name = "path";
  create_dir(k_output_path / "path", color_path, depth_path, gt_color_path,
             render_color_path, gt_depth_path, render_depth_path, acc_path);

  auto width = data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
  auto height = data_loader_ptr->dataparser_ptr_->sensor_.camera.height;

  auto video_format = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
  cv::VideoWriter video_color(k_output_path /
                                  (image_type_name + "/render_color.mp4"),
                              video_format, fps, cv::Size(width, height));
  cv::VideoWriter video_depth;

  auto depth_scale = 1.0 / data_loader_ptr->dataparser_ptr_->depth_scale_inv_;
  auto iter_bar = tq::trange(poses.size(0));
  iter_bar.set_prefix("Rendering");
  static auto p_timer_render = llog::CreateTimer("    render_train_image");
  for (const auto &i : iter_bar) {
    p_timer_render->tic();
    auto render_results = render_image(poses[i].slice(0, 0, 3), 1.0, false);
    p_timer_render->toc_sum();
    if (!render_results.empty()) {
      auto render_color = utils::tensor_to_cv_mat(render_results[0]);
      video_color.write(render_color);

      auto render_depth = utils::tensor_to_cv_mat(render_results[1]);
      cv::Mat depth_colormap = utils::apply_colormap_to_depth(render_depth);

      if (!video_depth.isOpened()) {
        video_depth = cv::VideoWriter(
            k_output_path / (image_type_name + "/render_depth.mp4"),
            video_format, fps, cv::Size(width, height));
      }
      video_depth.write(depth_colormap);

      std::filesystem::path color_file_name = to_string(i) + ".png";
      cv::imwrite(render_color_path / color_file_name, render_color);
      cv::imwrite(render_depth_path / color_file_name, depth_colormap);
    }
  }
  video_color.release();
  video_depth.release();
}

float NeuralSLAM::export_test_image(int idx, const std::string &prefix) {
  auto image_type = dataparser::DataType::TrainColor;
  torch::NoGradGuard no_grad;

  std::filesystem::path color_path = k_output_path / "mid/color";
  std::filesystem::path depth_path = k_output_path / "mid/depth";
  // std::filesystem::path acc_path = k_output_path / "mid/acc";
  // std::filesystem::create_directories(acc_path);
  std::filesystem::path scale_path = k_output_path / "mid/scale";
  std::filesystem::create_directories(scale_path);
  std::filesystem::path num_path = k_output_path / "mid/num";
  std::filesystem::create_directories(num_path);
  std::filesystem::path gt_color_path = color_path / "gt";
  std::filesystem::create_directories(gt_color_path);
  std::filesystem::path render_color_path = color_path / "renders";
  std::filesystem::create_directories(render_color_path);
  std::filesystem::path gt_depth_path = depth_path / "gt";
  std::filesystem::create_directories(gt_depth_path);
  std::filesystem::path render_depth_path = depth_path / "renders";
  std::filesystem::create_directories(render_depth_path);

  auto width = data_loader_ptr->dataparser_ptr_->sensor_.camera.width;
  auto height = data_loader_ptr->dataparser_ptr_->sensor_.camera.height;
  auto depth_scale = 1.0 / data_loader_ptr->dataparser_ptr_->depth_scale_inv_;

  if (idx < 0) {
    idx = std::rand() % data_loader_ptr->dataparser_ptr_->size(0);
  }
  auto render_results =
      render_image(data_loader_ptr->dataparser_ptr_->get_pose(idx, image_type)
                       .slice(0, 0, 3)
                       .to(k_device),
                   1.0, false);
  float psnr = 0.0f;
  if (!render_results.empty()) {
    auto render_color = render_results[0].clamp(0.0f, 1.0f);
    auto render_color_mat = utils::tensor_to_cv_mat(render_color);
    auto gt_color =
        data_loader_ptr->dataparser_ptr_->get_image(idx, image_type);
    auto gt_color_mat = utils::tensor_to_cv_mat(gt_color);

    auto render_depth = utils::tensor_to_cv_mat(render_results[1]);
    cv::Mat depth_colormap = utils::apply_colormap_to_depth(render_depth);

    // auto render_acc = utils::tensor_to_cv_mat(render_results[2]);
    // cv::Mat acc_colormap = utils::apply_colormap_to_depth(render_acc);

    auto render_scale = utils::tensor_to_cv_mat(render_results[3]);
    cv::Mat scale_colormap = utils::apply_colormap_to_depth(
        render_scale, 0.0f, 0.0f, cv::COLORMAP_PARULA);

    auto render_num = utils::tensor_to_cv_mat(render_results[4]);
    cv::Mat num_colormap = utils::apply_colormap_to_depth(
        render_num, 0.0f, 0.0f, cv::COLORMAP_HOT);

    auto gt_depth =
        data_loader_ptr->dataparser_ptr_->get_image_cv_mat(idx, image_type + 1);
    if (!gt_depth.empty()) {
      gt_depth = utils::apply_colormap_to_depth(gt_depth);
    }

    auto gt_color_file_name =
        prefix + data_loader_ptr->dataparser_ptr_->get_file(idx, image_type)
                     .filename()
                     .replace_extension(".png")
                     .string();
    cv::imwrite(gt_color_path / gt_color_file_name, gt_color_mat);
    cv::imwrite(render_color_path / gt_color_file_name, render_color_mat);

    std::filesystem::path gt_depth_file =
        data_loader_ptr->dataparser_ptr_->get_file(idx, image_type + 1);
    auto gt_depth_file_name = prefix + gt_depth_file.filename().string();

    if (!gt_depth.empty()) {
      cv::imwrite(gt_depth_path / gt_depth_file_name, gt_depth);
    }
    cv::imwrite(render_depth_path / gt_color_file_name, depth_colormap);
    // cv::imwrite(acc_path / gt_color_file_name, acc_colormap);
    cv::imwrite(scale_path / gt_color_file_name, scale_colormap);
    cv::imwrite(num_path / gt_color_file_name, num_colormap);

    // eval color
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto eval_python_cmd = "python " + k_package_path.string() +
                           "/eval/image_metrics/metrics_single.py --gt_color " +
                           (gt_color_path / gt_color_file_name).string() +
                           " --renders_color " +
                           (render_color_path / gt_color_file_name).string();
    std::cout << BLUE
              << "Conducting rendering evaluation command: " << eval_python_cmd
              << "\033[0m\n";
    int ret = std::system(eval_python_cmd.c_str());

    psnr = loss_utils::psnr(gt_color.to(render_color.device()).unsqueeze(0),
                            render_color.unsqueeze(0));
  }

  return psnr;
}

// https://pytorch.org/tutorials/advanced/cpp_frontend.html#checkpointing-and-recovering-the-training-state
void NeuralSLAM::export_checkpoint() {
  if (!k_output_path.empty()) {
    auto save_path = k_output_path / ("local_map_checkpoint.pt");
    torch::save(local_map_ptr, save_path);
    write_pt_params();
  }
}

void NeuralSLAM::load_checkpoint(
    const std::filesystem::path &_checkpoint_path) {
  read_pt_params();
  local_map_ptr = std::make_shared<LocalMap>(
      k_map_origin, k_x_min, k_x_max, k_y_min, k_y_max, k_z_min, k_z_max);
  torch::load(local_map_ptr, _checkpoint_path / "local_map_checkpoint.pt");
}

void NeuralSLAM::save_mesh(const bool &cull_mesh, const std::string &prefix) {
  cout << "\033[1;34m\nStart meshing map...\n\033[0m";

#ifdef ENABLE_ROS
  std_msgs::Header header;
  header.frame_id = "world";
  header.stamp = ros::Time::now();
  local_map_ptr->meshing_(mesh_pub, mesh_color_pub, header, k_export_res, true);
#else
  local_map_ptr->meshing_(k_export_res, true);
#endif
  if (cull_mesh) {
    local_map_ptr->p_mesher_->save_mesh(k_output_path, k_vis_attribute, prefix,
                                        data_loader_ptr->dataparser_ptr_);
  } else {
    local_map_ptr->p_mesher_->save_mesh(k_output_path, k_vis_attribute, prefix);
  }
}

void NeuralSLAM::eval_mesh() {
  auto gt_mesh_path = data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
  if (std::filesystem::exists(gt_mesh_path)) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto eval_python_cmd = "python " + k_package_path.string() +
                           "/eval/structure_metrics/evaluator.py --pred_mesh " +
                           k_output_path.string();
    if (k_cull_mesh) {
      eval_python_cmd += "/mesh_culled.ply --gt_pcd " +
                         data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
    } else {
      eval_python_cmd += "/mesh.ply --gt_pcd " +
                         data_loader_ptr->dataparser_ptr_->get_gt_mesh_path();
    }
    std::cout << BLUE
              << "Conducting structure evaluation command: " << eval_python_cmd
              << "\033[0m\n";
    int ret = std::system(eval_python_cmd.c_str());
    std::cout << GREEN
              << "Evaluation finished. Please check the results in the folder: "
              << k_output_path << "\033[0m\n";
  } else {
    std::cout << RED << "No ground truth mesh found, skip evaluation.\033[0m\n";
    return;
  }
}

void NeuralSLAM::eval_render() {
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto eval_python_cmd = "python " + k_package_path.string() +
                         "/eval/image_metrics/metrics.py -m " +
                         (k_output_path / "train/color").string();
  std::cout << BLUE
            << "Conducting rendering evaluation command: " << eval_python_cmd
            << "\033[0m\n";
  int ret = std::system(eval_python_cmd.c_str());

  eval_python_cmd = "python " + k_package_path.string() +
                    "/eval/image_metrics/metrics.py -m " +
                    (k_output_path / "eval/color").string();
  std::cout << BLUE
            << "Conducting rendering evaluation command: " << eval_python_cmd
            << "\033[0m\n";
  ret = std::system(eval_python_cmd.c_str());

  eval_python_cmd = "python " + k_package_path.string() +
                    "/eval/image_metrics/metrics.py -m " +
                    (k_output_path / "test/color").string();
  std::cout << BLUE
            << "Conducting rendering evaluation command: " << eval_python_cmd
            << "\033[0m\n";
  ret = std::system(eval_python_cmd.c_str());
  std::cout << GREEN
            << "Evaluation finished. Please check the results in the folder: "
            << k_output_path << "\033[0m\n";
}

void NeuralSLAM::export_timing(bool print) {
  if (print) {
    llog::PrintLog();
  }
  llog::SaveLog(k_output_path / ("timing.txt"));
}

void NeuralSLAM::plot_log(const std::string &log_file) {
  c10::cuda::CUDACachingAllocator::emptyCache();
  auto cmd = "python " + k_package_path.string() + "/eval/draw_loss.py -l " +
             (k_output_path / log_file).string();
  std::cout << BLUE << "Conducting draw_loss command: " << cmd << "\033[0m\n";
  auto ret = std::system(cmd.c_str());
}
/* /Exporter */

void NeuralSLAM::keyboard_loop() {
  while (true) {
    switch (getchar()) {
    case 'm': {
      save_mesh(k_cull_mesh);
      break;
    }
    case 'e': {
      eval_mesh();
      eval_render();
      break;
    }
    case 'q': {
      end();
      break;
    }
    case 'r': {
      c10::cuda::CUDACachingAllocator::emptyCache();
      render_path(false, k_fps);
      render_path(true, 2);
      eval_render();
      break;
    }
    case 'p': {
      save_image = true;
      break;
    }
    case 'o': {
      export_checkpoint();
      break;
    }
    case 'u': {
      std::string pose_file =
          "/home/chrisliu/Projects/rimv2_ws/src/RIM2/data/"
          "FAST_LIVO2_RIM_Datasets/culture01/inter_color_poses.txt";
      render_path(pose_file, k_fps);
      break;
    }
    case 'v': {
      render_path(false, k_fps, false);
      llog::PrintLog();
      break;
    }
    case 'd': {
      std::cout << "Double render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fx *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fy *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cy *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 's': {
      std::cout << "Dedouble render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fx /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.fy /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cy /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'f': {
      std::cout << "Double render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'g': {
      std::cout << "Dedouble render resolution! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.height /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width << "x"
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.height
                << "\n";
      break;
    }
    case 'k': {
      std::cout << "Double width! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width *= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx *= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << "\n";
      break;
    }
    case 'j': {
      std::cout << "DeDouble width! From "
                << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << " to ";
      data_loader_ptr->dataparser_ptr_->sensor_.camera.width /= 2;
      data_loader_ptr->dataparser_ptr_->sensor_.camera.cx /= 2;
      std::cout << data_loader_ptr->dataparser_ptr_->sensor_.camera.width
                << "\n";
      break;
    }
    case 'h': {
      std::cout << "Keyboard Controls:\n";
      std::cout << "  h: Display this help menu\n";
      std::cout << "  m: Save mesh\n";
      std::cout << "  e: Evaluate mesh and render\n";
      std::cout << "  q: End program\n";
      std::cout << "  r: Render path and evaluate\n";
      std::cout << "  p: Save current snapshot\n";
      std::cout << "  o: Save checkpoint\n";
      std::cout << "  u: Render path from file\n";
      std::cout << "  v: Test render speed (no saving)\n";
      std::cout << "\nResolution Controls:\n";
      std::cout << "  d: Double render resolution (width, height, focal)\n";
      std::cout << "  s: Halve render resolution (width, height, focal)\n";
      std::cout << "  f: Double render resolution (width, height only)\n";
      std::cout << "  g: Halve render resolution (width, height only)\n";
      std::cout << "  k: Double width only\n";
      std::cout << "  j: Halve width only\n";
      break;
    }
    }

    std::chrono::milliseconds dura(100);
    std::this_thread::sleep_for(dura);
  }
}

void NeuralSLAM::misc_loop() {
  while (true) {
    if (misc_trigger >= 0) {
      plot_log("loss_log.txt");
      export_timing(true);
      misc_trigger = -1;
    }
    std::chrono::milliseconds dura(100);
    std::this_thread::sleep_for(dura);
  }
}

bool NeuralSLAM::end() {
  export_timing(true);
  export_checkpoint();
  if (k_sdf_weight > 0) {
    // clear VRAM
    c10::cuda::CUDACachingAllocator::emptyCache();
    save_mesh(k_cull_mesh);
    eval_mesh();
  }
  if (k_rgb_weight > 0) {
    c10::cuda::CUDACachingAllocator::emptyCache();
    render_path(false, k_fps);
    render_path(true, 2);
    eval_render();
  }

  // exit(0); // weird memeory leak
  abort();
  return true;
}

#ifdef ENABLE_ROS
void NeuralSLAM::pretrain_loop() {
  while (true) {
    visualization();

    static std::chrono::milliseconds dura(10);
    std::this_thread::sleep_for(dura);
  }
}

NeuralSLAM::NeuralSLAM(ros::NodeHandle &_nh, const int &mode,
                       const std::filesystem::path &_config_path,
                       const std::filesystem::path &_data_path)
    : NeuralSLAM(mode, _config_path, _data_path) {
  register_subscriber(_nh);
  register_publisher(_nh);
}

void NeuralSLAM::register_subscriber(ros::NodeHandle &nh) {
  rviz_pose_sub = nh.subscribe("/rviz/current_camera_pose", 1,
                               &NeuralSLAM::rviz_pose_callback, this);
}

void NeuralSLAM::register_publisher(ros::NodeHandle &nh) {
  pose_pub = nh.advertise<geometry_msgs::PoseStamped>("pose", 1);
  path_pub = nh.advertise<nav_msgs::Path>("path", 1);
  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 1);

  mesh_pub = nh.advertise<mesh_msgs::MeshGeometryStamped>("mesh", 1);
  mesh_color_pub =
      nh.advertise<mesh_msgs::MeshVertexColorsStamped>("mesh_color", 1);
  voxel_pub = nh.advertise<visualization_msgs::Marker>("voxel", 1);
  vis_shift_map_pub =
      nh.advertise<visualization_msgs::Marker>("vis_shift_map", 1);
  pointcloud_pub = nh.advertise<sensor_msgs::PointCloud2>("pointcloud", 1);
  occ_pub = nh.advertise<sensor_msgs::PointCloud2>("occ", 1);
  unknown_pub = nh.advertise<sensor_msgs::PointCloud2>("unknown", 1);
  rgb_pub = nh.advertise<sensor_msgs::Image>("rgb", 1);
  depth_pub = nh.advertise<sensor_msgs::Image>("depth", 1);
  // wait all publisher to be ready
  ros::Duration(0.5).sleep();
}

void NeuralSLAM::rviz_pose_callback(
    const geometry_msgs::PoseConstPtr &_rviz_pose_ptr) {
  auto pos =
      torch::tensor({_rviz_pose_ptr->position.x, _rviz_pose_ptr->position.y,
                     _rviz_pose_ptr->position.z},
                    torch::kFloat);
  auto quat = torch::tensor(
      {_rviz_pose_ptr->orientation.w, _rviz_pose_ptr->orientation.x,
       _rviz_pose_ptr->orientation.y, _rviz_pose_ptr->orientation.z},
      torch::kFloat);
  auto rot = utils::quat_to_rot(quat).matmul(
      coords::opencv_to_opengl_camera_rotation());
  rviz_pose_ = torch::cat({rot, pos.view({3, 1})}, 1);
}

/* Publisher */
void NeuralSLAM::pub_pose(const torch::Tensor &_pose,
                          std_msgs::Header _header) {
  if (pose_pub.getNumSubscribers() > 0 || path_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    auto pos = _pose.slice(1, 3, 4).detach().cpu();
    auto quat = utils::rot_to_quat(_pose.slice(1, 0, 3).detach().cpu());
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header = _header;
    pose_stamped.pose.position.x = pos[0].item<float>();
    pose_stamped.pose.position.y = pos[1].item<float>();
    pose_stamped.pose.position.z = pos[2].item<float>();
    pose_stamped.pose.orientation.w = quat[0].item<float>();
    pose_stamped.pose.orientation.x = quat[1].item<float>();
    pose_stamped.pose.orientation.y = quat[2].item<float>();
    pose_stamped.pose.orientation.z = quat[3].item<float>();
    pose_pub.publish(pose_stamped);

    path_msg.header = _header;
    path_msg.poses.emplace_back(pose_stamped);
    path_pub.publish(path_msg);

    static tf::TransformBroadcaster br;
    tf::Transform transform;
    tf::Quaternion q;
    // body frame
    transform.setOrigin(tf::Vector3(pose_stamped.pose.position.x,
                                    pose_stamped.pose.position.y,
                                    pose_stamped.pose.position.z));
    q.setW(pose_stamped.pose.orientation.w);
    q.setX(pose_stamped.pose.orientation.x);
    q.setY(pose_stamped.pose.orientation.y);
    q.setZ(pose_stamped.pose.orientation.z);
    transform.setRotation(q);
    br.sendTransform(
        tf::StampedTransform(transform, _header.stamp, "world", "depth"));
    transform.setIdentity();
    br.sendTransform(tf::StampedTransform(transform, _header.stamp, "world",
                                          _header.frame_id));
  }
}

void NeuralSLAM::pub_path(std_msgs::Header _header) {
  if (path_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    for (int i = 0; i < data_loader_ptr->dataparser_ptr_->size(
                            dataparser::DataType::TrainColor);
         ++i) {
      auto _pose = data_loader_ptr->dataparser_ptr_->get_pose(
          i, dataparser::DataType::TrainColor);
      auto pos = _pose.slice(1, 3, 4).detach().cpu();
      auto quat = utils::rot_to_quat(_pose.slice(1, 0, 3).detach().cpu());
      geometry_msgs::PoseStamped pose_stamped;
      pose_stamped.header = _header;
      pose_stamped.pose.position.x = pos[0].item<float>();
      pose_stamped.pose.position.y = pos[1].item<float>();
      pose_stamped.pose.position.z = pos[2].item<float>();
      pose_stamped.pose.orientation.w = quat[0].item<float>();
      pose_stamped.pose.orientation.x = quat[1].item<float>();
      pose_stamped.pose.orientation.y = quat[2].item<float>();
      pose_stamped.pose.orientation.z = quat[3].item<float>();
      nav_msgs::Odometry odom_msg;
      odom_msg.header = _header;
      odom_msg.pose.pose = pose_stamped.pose;
      odom_pub.publish(odom_msg);

      path_msg.header = _header;
      path_msg.poses.emplace_back(pose_stamped);
    }
    path_pub.publish(path_msg);
  }
}

void NeuralSLAM::pub_voxel(const torch::Tensor &_xyz,
                           std_msgs::Header _header) {
  if (voxel_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    static std::vector<torch::Tensor> voxel_xyz;
    voxel_xyz.emplace_back(_xyz);
    auto vis_voxel_map =
        utils::get_vix_voxel_map(torch::cat(voxel_xyz, 0), k_leaf_size);
    // auto vis_voxel_map = utils::get_vix_voxel_map(_xyz, k_leaf_size);
    if (!vis_voxel_map.points.empty()) {
      vis_voxel_map.header = _header;
      voxel_pub.publish(vis_voxel_map);
    }
  }
}

void NeuralSLAM::pub_pointcloud(const ros::Publisher &_pub,
                                const torch::Tensor &_xyz,
                                std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(utils::tensor_to_pointcloud(_xyz), pcl_msg);
    pcl_msg.header = _header;
    _pub.publish(pcl_msg);
  }
}

void NeuralSLAM::pub_pointcloud(const ros::Publisher &_pub,
                                const torch::Tensor &_xyz,
                                const torch::Tensor &_rgb,
                                std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::PointCloud2 pcl_msg =
        utils::tensor_to_pointcloud_msg(_xyz, _rgb);
    pcl_msg.header = _header;
    _pub.publish(pcl_msg);
  }
}

void NeuralSLAM::pub_image(const ros::Publisher &_pub,
                           const torch::Tensor &_image,
                           std_msgs::Header _header) {
  if (_pub.getNumSubscribers() > 0) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }
    sensor_msgs::Image img_msg = utils::tensor_to_img_msg(_image);
    img_msg.header = _header;
    _pub.publish(img_msg);
  }
}

void NeuralSLAM::pub_render_image(std_msgs::Header _header) {
  if ((rviz_pose_.numel() > 0) &&
      (rgb_pub.getNumSubscribers() > 0 || depth_pub.getNumSubscribers() > 0)) {
    if (_header.frame_id.empty()) {
      _header.frame_id = "world";
      _header.stamp = ros::Time::now();
    }

    torch::NoGradGuard no_grad;
    auto render_results = render_image(rviz_pose_, 1.0, false);
    if (!render_results.empty()) {
      if (rgb_pub.getNumSubscribers() > 0) {
        pub_image(rgb_pub, render_results[0], _header);
      }
      if (depth_pub.getNumSubscribers() > 0) {
        pub_image(depth_pub, render_results[1], _header);
      }
      if (save_image) {
        static std::filesystem::path snapshot_color_path =
            k_output_path / "snapshot/color";
        if (!std::filesystem::exists(snapshot_color_path)) {
          std::filesystem::create_directories(snapshot_color_path);
        }
        static std::filesystem::path snapshot_depth_path =
            k_output_path / "snapshot/depth";
        if (!std::filesystem::exists(snapshot_depth_path)) {
          std::filesystem::create_directories(snapshot_depth_path);
        }
        static int count = 0;
        cv::imwrite(snapshot_color_path / (to_string(count) + ".png"),
                    utils::tensor_to_cv_mat(render_results[0]));
        cv::imwrite(snapshot_depth_path / (to_string(count) + ".png"),
                    utils::apply_colormap_to_depth(
                        utils::tensor_to_cv_mat(render_results[1])));
        // save pose
        std::ofstream pose_file(snapshot_color_path /
                                (to_string(count) + ".txt"));
        pose_file << rviz_pose_ << std::endl;

        count++;
        save_image = false;
        cout << "save snapshot in " << snapshot_color_path << endl;
        cout << "Snapshot pose:\n" << rviz_pose_ << endl;
      }
    }
  }
}

void NeuralSLAM::visualization(const torch::Tensor &_xyz,
                               std_msgs::Header _header) {
  torch::NoGradGuard no_grad;
  if (_header.frame_id.empty()) {
    _header.frame_id = "world";
    _header.stamp = ros::Time::now();
  }
  static auto p_timer = llog::CreateTimer("visualization");
  p_timer->tic();

  pub_render_image(_header);
  pub_path(_header);

  if (mesh_pub.getNumSubscribers() > 0) {
    local_map_ptr->meshing_(mesh_pub, mesh_color_pub, _header, k_vis_res,
                            false);
  }

  if (vis_shift_map_pub.getNumSubscribers() > 0) {
    auto vis_shift_map =
        utils::get_vis_shift_map(local_map_ptr->pos_W_M_, k_x_min, k_x_max,
                                 k_y_min, k_y_max, k_z_min, k_z_max);
    if (!vis_shift_map.points.empty()) {
      vis_shift_map.header = _header;
      vis_shift_map_pub.publish(vis_shift_map);
    }
  }
  p_timer->toc_sum();
}
/* /Publisher */
#endif