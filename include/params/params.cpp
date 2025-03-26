#include "params.h"
#include <opencv2/opencv.hpp>

#ifdef ENABLE_ROS
#include "ros/package.h"
#endif

bool k_debug;
int k_dataset_type;
bool k_preload;

torch::Tensor k_map_origin;
float k_prefilter;
float k_max_time_diff_camera_and_pose, k_max_time_diff_lidar_and_pose;
bool k_prob_map_en;

std::filesystem::path k_dataset_path;

int k_ds_pt_num, k_max_pt_num;

std::filesystem::path k_output_path, k_package_path;

torch::Device k_device = torch::kCPU;
// parameter
float k_x_max, k_x_min, k_y_max, k_y_min, k_z_max, k_z_min, k_min_range,
    k_max_range;

float k_inner_map_size, k_map_size, k_map_size_inv, k_boundary_size;

float k_leaf_size, k_leaf_size_inv;
int k_octree_level, k_fill_level;
int k_map_resolution;

int k_iter_step, k_export_interval, k_export_ckp_interval;
int k_surface_sample_num, k_free_sample_num;
float k_color_batch_pt_num;
int k_batch_num;
float k_sample_pts_per_ray;
float k_sample_std;
bool k_outlier_remove;
double k_outlier_dist;
int k_outlier_removal_interval;

// torch decoder params
int k_hidden_dim, k_color_hidden_dim, k_geo_feat_dim;
int k_geo_num_layer, k_color_num_layer;
int k_dir_embedding_degree;
// tcnn decoder params
int k_n_levels, k_n_features_per_level, k_log2_hashmap_size;

// abalation parmaeter
bool mapper_init, mapper_update;
float k_bce_sigma, k_bce_isigma, k_truncated_dis, k_sphere_trace_thr;
float k_sdf_weight, k_eikonal_weight, k_curvate_weight, k_rgb_weight,
    k_dist_weight, k_sdf_weight_init, k_rgb_weight_init;
float k_res_scale;

int k_trace_iter;
bool k_trunc_sdf;

float k_t, k_lr, k_lr_end, k_weight_decay;

sensor::Sensors k_sensor;
torch::Tensor k_T_B_S;

// visualization
int k_vis_attribute;
int k_vis_frame_step;
int k_vis_batch_pt_num, k_batch_ray_num;
float k_vis_res, k_export_res;
int k_fps;

int k_export_colmap_format, k_export_train_pcl;
bool k_llff;
bool k_cull_mesh;

void print_files(const std::string &_file_path) {
  std::cout << "print_files: " << _file_path << '\n';
  std::ifstream file(_file_path);
  std::string str;
  while (std::getline(file, str)) {
    std::cout << str << '\n';
  }
  file.close();
  std::cout << "print_files end\n";
}

void read_params(const std::filesystem::path &_config_path,
                 const std::filesystem::path &_data_path,
                 const bool &_new_dir) {
#ifdef ENABLE_ROS
  k_package_path = ros::package::getPath("neural_mapping");
#else
  k_package_path = _config_path.parent_path().parent_path().parent_path();
  std::cerr << "Using path derived from config file: " << k_package_path
            << '\n';
#endif

  if (_new_dir) {
    // get now data and time
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    std::stringstream ss;
    ss << std::put_time(&tm, "%Y-%m-%d-%H-%M-%S");
    ss << "_" << _data_path.filename().string();
    ss << "_" << _config_path.filename().string();
    k_output_path = k_package_path / "output" / ss.str();
    std::filesystem::create_directories(k_output_path);
    auto ret = std::system(("ln -sfn " + k_output_path.string() + " " +
                            (k_package_path / "output" / "latest_run").string())
                               .c_str());

    auto config_output_dir = k_output_path / "config/scene";
    std::filesystem::create_directories(config_output_dir);
    std::string params_file_path = config_output_dir / "config.yaml";
    // copy _config_path to params_file_path
    ret = std::system(
        ("cp " + _config_path.string() + " " + params_file_path).c_str());
    // write data_path into config.yaml
    std::ofstream ofs(params_file_path, std::ios::app);
    ofs << "\ndata_path: " << _data_path;
    ofs.close();
  } else {
    k_output_path = _config_path.parent_path().parent_path();
  }

  std::cout << "output_path: " << k_output_path << '\n';

  cv::FileStorage fsSettings(_config_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings: " << _config_path << "\n";
    exit(-1);
  }

  if (fsSettings["data_path"].empty()) {
    k_dataset_path = _data_path;
  } else {
    k_dataset_path = fsSettings["data_path"];
  }
  std::cout << "data_path: " << k_dataset_path << '\n';

  auto scene_config_path =
      _config_path.parent_path() / std::string(fsSettings["scene_config"]);
  if (fsSettings["scene_config"].isNone()) {
    scene_config_path = _config_path;
  }
  read_scene_params(scene_config_path);

  auto parent_config_path =
      _config_path.parent_path() / std::string(fsSettings["base_config"]);
  read_base_params(parent_config_path);

  fsSettings["iter_step"] >> k_iter_step;

  fsSettings["fill_level"] >> k_fill_level;
  fsSettings["leaf_sizes"] >> k_leaf_size;
  k_leaf_size_inv = 1.0f / k_leaf_size;
  fsSettings["bce_sigma"] >> k_bce_sigma;

  k_bce_isigma = 1.0f / k_bce_sigma;
  k_sample_std = k_bce_sigma;
  k_truncated_dis = 3 * k_leaf_size;

  fsSettings["trace_iter"] >> k_trace_iter;
  k_batch_ray_num = k_color_batch_pt_num / k_trace_iter;
  fsSettings["sphere_trace_thr"] >> k_sphere_trace_thr;

  if (!fsSettings["camera"].isNone()) {
    k_sensor.camera.model = fsSettings["camera"]["model"];
    k_sensor.camera.width = fsSettings["camera"]["width"];
    k_sensor.camera.height = fsSettings["camera"]["height"];
    k_sensor.camera.fx = fsSettings["camera"]["fx"];
    k_sensor.camera.fy = fsSettings["camera"]["fy"];
    k_sensor.camera.cx = fsSettings["camera"]["cx"];
    k_sensor.camera.cy = fsSettings["camera"]["cy"];

    k_sensor.camera.set_distortion(
        fsSettings["camera"]["d0"], fsSettings["camera"]["d1"],
        fsSettings["camera"]["d2"], fsSettings["camera"]["d3"],
        fsSettings["camera"]["d4"]);
  }
  if (!fsSettings["extrinsic"].isNone()) {
    cv::Mat cv_T_C_L;
    fsSettings["extrinsic"]["T_C_L"] >> cv_T_C_L;
    cv_T_C_L.convertTo(cv_T_C_L, CV_32FC1);
    k_sensor.T_C_L =
        torch::from_blob(cv_T_C_L.data, {4, 4}, torch::kFloat32).clone();

    cv::Mat cv_T_B_L;
    fsSettings["extrinsic"]["T_B_L"] >> cv_T_B_L;
    cv_T_B_L.convertTo(cv_T_B_L, CV_32FC1);
    k_sensor.T_B_L =
        torch::from_blob(cv_T_B_L.data, {4, 4}, torch::kFloat32).clone();

    k_sensor.T_B_C = k_sensor.T_B_L.matmul(k_sensor.T_C_L.inverse());
  }
  if (!fsSettings["map"].isNone()) {
    cv::Mat cv_map_origin;
    fsSettings["map"]["map_origin"] >> cv_map_origin;
    cv_map_origin.convertTo(cv_map_origin, CV_32FC1);
    k_map_origin =
        torch::from_blob(cv_map_origin.data, {3}, torch::kFloat32).clone();

    fsSettings["map"]["map_size"] >> k_inner_map_size;
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

    k_boundary_size = k_leaf_size;

    if (k_fill_level > k_octree_level) {
      k_fill_level = k_octree_level;
    }
  } else {
    throw std::runtime_error("map is not set in the config file");
  }

  fsSettings.release();
  print_files(_config_path);
}

void read_scene_params(const std::filesystem::path &_scene_config_path) {
  cv::FileStorage fsSettings(_scene_config_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings: " << _scene_config_path
              << "\n";
    exit(-1);
  }

  auto config_output_dir = k_output_path / "config/scene";
  std::filesystem::create_directories(config_output_dir);
  std::string params_file_path =
      config_output_dir / _scene_config_path.filename();
  // copy _config_path to params_file_path
  auto ret = std::system(
      ("cp " + _scene_config_path.string() + " " + params_file_path).c_str());

  /* Start reading parameters */
  fsSettings["dataset_type"] >> k_dataset_type;

  bool device_param;
  fsSettings["device_param"] >> device_param;
  if (device_param) {
    k_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  }

  fsSettings["min_range"] >> k_min_range;
  fsSettings["max_range"] >> k_max_range;

  fsSettings["outlier_remove"] >> k_outlier_remove;
  fsSettings["outlier_dist"] >> k_outlier_dist;
  fsSettings["outlier_removal_interval"] >> k_outlier_removal_interval;

  fsSettings["dir_embedding_degree"] >> k_dir_embedding_degree;

  k_res_scale = 1.0f;

  fsSettings["ds_pt_num"] >> k_ds_pt_num;
  fsSettings["max_pt_num"] >> k_max_pt_num;

  fsSettings["vis_attribute"] >> k_vis_attribute;

  fsSettings["vis_resolution"] >> k_vis_res;

  fsSettings["export_resolution"] >> k_export_res;

  fsSettings["fps"] >> k_fps;
  if (k_fps <= 0) {
    k_fps = 10;
  }

  fsSettings["preload"] >> k_preload;
  fsSettings["llff"] >> k_llff;
  fsSettings["cull_mesh"] >> k_cull_mesh;
  fsSettings["prob_map_en"] >> k_prob_map_en;

  fsSettings["prefilter"] >> k_prefilter;
  fsSettings["max_time_diff_camera_and_pose"] >>
      k_max_time_diff_camera_and_pose;
  fsSettings["max_time_diff_lidar_and_pose"] >> k_max_time_diff_lidar_and_pose;

  fsSettings.release();
  print_files(params_file_path);
}

void read_base_params(const std::filesystem::path &_base_config_path) {
  cv::FileStorage fsSettings(_base_config_path, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings: " << _base_config_path << "\n";
    exit(-1);
  }

  auto config_output_dir = k_output_path / "config";
  std::filesystem::create_directories(config_output_dir);
  std::string params_file_path =
      config_output_dir / _base_config_path.filename();
  // copy _config_path to params_file_path
  auto ret = std::system(
      ("cp " + _base_config_path.string() + " " + params_file_path).c_str());

  /* Start reading parameters */
  fsSettings["debug"] >> k_debug;

  bool device_param;
  fsSettings["device_param"] >> device_param;
  if (device_param) {
    k_device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  }

  fsSettings["surface_sample_num"] >> k_surface_sample_num;
  fsSettings["free_sample_num"] >> k_free_sample_num;
  fsSettings["color_batch_pt_num"] >> k_color_batch_pt_num;
  k_vis_batch_pt_num = 10 * k_color_batch_pt_num;
  fsSettings["trunc_sdf"] >> k_trunc_sdf;

  fsSettings["hidden_dim"] >> k_hidden_dim;
  fsSettings["color_hidden_dim"] >> k_color_hidden_dim;
  fsSettings["geo_feat_dim"] >> k_geo_feat_dim;
  fsSettings["geo_num_layer"] >> k_geo_num_layer;
  fsSettings["color_num_layer"] >> k_color_num_layer;

  // torch encoding params
  fsSettings["n_levels"] >> k_n_levels;
  fsSettings["n_features_per_level"] >> k_n_features_per_level;
  fsSettings["log2_hashmap_size"] >> k_log2_hashmap_size;

  k_t = 0.0f;
  fsSettings["lr"] >> k_lr;
  fsSettings["lr_end"] >> k_lr_end;

  fsSettings["sdf_weight"] >> k_sdf_weight;
  fsSettings["rgb_weight"] >> k_rgb_weight;
  k_sdf_weight_init = k_rgb_weight;
  k_rgb_weight_init = k_sdf_weight;
  k_rgb_weight = k_rgb_weight > 0 ? 1e-4f : 0.0f;
  k_rgb_weight_init = k_rgb_weight > 0 ? 1e-4f : 0.0f;
  fsSettings["eikonal_weight"] >> k_eikonal_weight;
  fsSettings["curvate_weight"] >> k_curvate_weight;
  fsSettings["dist_weight"] >> k_dist_weight;

  fsSettings["vis_frame_step"] >> k_vis_frame_step;

  fsSettings["export_interval"] >> k_export_interval;
  fsSettings["export_ckp_interval"] >> k_export_ckp_interval;
  fsSettings["export_colmap_format"] >> k_export_colmap_format;
  fsSettings["export_train_pcl"] >> k_export_train_pcl;
  fsSettings.release();
  print_files(params_file_path);
}

void write_pt_params() {
  auto pt_config_file = k_output_path / "config/scene/pt.yaml";
  std::ofstream ofs(pt_config_file);
  ofs << "%YAML:1.0\n";
  cv::Mat cv_map_origin(1, 3, CV_32FC1, k_map_origin.data_ptr());
  ofs << "map_origin: !!opencv-matrix\n   rows: 1\n   cols: 3\n   dt: f\n"
         "   data: "
      << cv_map_origin << '\n';
  ofs << "inner_map_size: " << k_inner_map_size << '\n';
}

void read_pt_params() {
  auto pt_config_file = k_output_path / "config/scene/pt.yaml";
  cv::FileStorage fsSettings(pt_config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cerr << "ERROR: Wrong path to settings: " << pt_config_file << "\n";
    exit(-1);
  }
  cv::Mat cv_map_origin = cv::Mat::zeros(3, 1, CV_32FC1);
  fsSettings["map_origin"] >> cv_map_origin;
  k_map_origin = torch::from_blob(cv_map_origin.data, {3}, torch::kFloat32)
                     .clone()
                     .to(k_device);
  fsSettings["inner_map_size"] >> k_inner_map_size;
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