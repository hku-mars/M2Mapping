#include "tracer.h"
#include "llog/llog.h"
#include "nerfacc_cpp/renderers.hpp"
#include <kaolin/csrc/render/spc/raytrace.h>

#include "params/params.h"
#include "sphere_trace/sphere_trace.h"

namespace tracer {

torch::Tensor stratified_bin_sampling(torch::Tensor near, torch::Tensor far,
                                      const int &num_sample) {
  int num_valid_ray = near.size(0);
  int num_bin = num_sample + 1;
  float inv_num_bin = 1.0f / num_bin;
  // [num_bin]
  auto bins = torch::linspace(0.0, 1.0 - inv_num_bin, num_bin, near.options());

  // [num_valid_ray, num_bin]
  auto rand = torch::rand({num_valid_ray, num_bin}, near.options());
  bins = bins.unsqueeze(0) + rand * inv_num_bin;
  // [num_valid_ray, 1] + [num_valid_ray, num_bin]
  // -> [num_valid_ray, num_bin]
  auto bin_depths = near + bins * (far - near);

  return bin_depths;
}

std::vector<torch::Tensor> stratified_sampling(torch::Tensor near,
                                               torch::Tensor far,
                                               const int &num_sample) {
  auto bin_depths = stratified_bin_sampling(near, far, num_sample);

  // [num_valid_ray, num_sample]
  auto depth_upper = bin_depths.slice(1, 1);
  auto depth_lower = bin_depths.slice(1, 0, -1);
  auto depths = 0.5 * (depth_upper + depth_lower);
  auto deltas = depth_upper - depth_lower;
  return {depths, deltas};
}

std::vector<torch::Tensor> volsdf_alpha_step(torch::Tensor sdf,
                                             torch::Tensor isigma,
                                             torch::Tensor delta,
                                             torch::Tensor slope) {
  auto iter_cos = torch::relu(-slope);
  auto density = isigma * torch::sigmoid(-sdf * isigma) * iter_cos;
  // [1,num_sample]
  auto alpha = (1.0f - torch::exp(-density * delta.abs())).clip(0.0f, 1.0f);
  return {alpha, density, sdf, isigma};
}

std::vector<torch::Tensor> volsdf_alpha(const LocalMap::Ptr &_local_map_ptr,
                                        torch::Tensor pts, torch::Tensor delta,
                                        torch::Tensor ridx = torch::Tensor(),
                                        torch::Tensor ray_d = torch::Tensor(),
                                        torch::Tensor slope = torch::Tensor()) {
  // [num_valid_ray*num_sample, 1]
  auto sdf_results = _local_map_ptr->get_sdf(pts);
  auto sdf = sdf_results[0].contiguous();
  auto isigma = sdf_results[1];

  if (isigma.isnan().any().item<bool>()) {
    throw std::runtime_error("isigma.isnan().any()");
  }
  // [num_valid_ray, num_sample]

  if (slope.numel() == 0) {
    slope = -torch::ones_like(sdf);
  }

  return volsdf_alpha_step(sdf, isigma, delta, slope);
}

std::vector<torch::Tensor>
pts_to_alpha(const LocalMap::Ptr &_local_map_ptr, const torch::Tensor &pts,
             const torch::Tensor &ray_d, const torch::Tensor &delta,
             const torch::Tensor &ridx, torch::Tensor slope = torch::Tensor()) {
  auto volsdf_results =
      volsdf_alpha(_local_map_ptr, pts, delta, ridx, ray_d, slope);
  auto alpha = volsdf_results[0];
  auto density = volsdf_results[1];
  auto sdf = volsdf_results[2];
  auto isigma = volsdf_results[3];
  return {alpha, sdf, isigma};
}

std::vector<torch::Tensor>
render_from_pts(const LocalMap::Ptr &_local_map_ptr, torch::Tensor pts,
                torch::Tensor ray_d, torch::Tensor ridx, torch::Tensor depth,
                torch::Tensor delta, const int &num_rays,
                const int &render_type, torch::Tensor slope) {
  torch::Tensor render_colors, accu_weight_colors, render_depths, render_scales,
      render_nums;
  auto convert_results =
      pts_to_alpha(_local_map_ptr, pts, ray_d, delta, ridx, slope);
  auto alpha = convert_results[0];
  auto sdf = convert_results[1];
  auto isigma = convert_results[2];

  auto packed_info = nerfacc::pack_info(ridx, num_rays);
  auto render_results =
      nerfacc::render_weight_from_alpha(alpha.view({-1}), packed_info);
  auto weights = render_results[0].unsqueeze(-1);
  auto trans = render_results[1];

  auto vis_mask = trans.view({-1}) > 1e-3f;
  auto vis_idx = vis_mask.nonzero().squeeze();
  pts = pts.index_select(0, vis_idx);
  sdf = sdf.index_select(0, vis_idx);
  isigma = isigma.index_select(0, vis_idx);
  alpha = alpha.index_select(0, vis_idx);
  ray_d = ray_d.index_select(0, vis_idx);
  ridx = ridx.index_select(0, vis_idx);
  depth = depth.index_select(0, vis_idx);
  weights = weights.index_select(0, vis_idx);

  static auto rgb_renderer = nerfacc::RGBRenderer();
  if (render_type == 0 || render_type == 1) {
    static auto timer_rgb_renderer = llog::CreateTimer("        rgb_renderer");
    timer_rgb_renderer->tic();
    // [N,3]
    auto color = _local_map_ptr->get_color(pts, ray_d)[0];
    auto render_results = rgb_renderer.forward(color, weights, ridx, num_rays);
    render_colors = render_results[0];
    accu_weight_colors = render_results[1];

    timer_rgb_renderer->toc_sum();
  }

  static auto depth_renderer = nerfacc::DepthRenderer("expected");
  static auto timer_depth_renderer =
      llog::CreateTimer("        depth_renderer");
  timer_depth_renderer->tic();
  render_depths = depth_renderer.forward(weights, depth, depth, ridx, num_rays);
  timer_depth_renderer->toc_sum();

  // TODO:if not train not need to render disp
  auto inv_depth = 1.0f / depth;
  auto render_disp =
      depth_renderer.forward(weights, inv_depth, inv_depth, ridx, num_rays);

  auto accu_weight_colors_detach = accu_weight_colors.detach();
  auto sigma = 1.0f / isigma;
  render_scales =
      depth_renderer.forward(weights, sigma, sigma, ridx, num_rays) /
      accu_weight_colors_detach;
  render_nums =
      depth_renderer.forward(torch::ones_like(weights), torch::ones_like(depth),
                             torch::ones_like(depth), ridx, num_rays);

  auto render_depths_detach = render_depths.detach().index_select(0, ridx);
  auto depth_delta = (1.0f - depth / render_depths_detach)
                         .abs()
                         .nan_to_num(0.0f)
                         .clamp_max(0.99f);
  auto render_weight_depth_delta =
      (depth_renderer.forward(weights, depth_delta, depth_delta, ridx,
                              num_rays) /
       accu_weight_colors.clamp(0.01f));
  return {render_colors, render_depths, pts,
          sdf,           isigma,        accu_weight_colors,
          alpha,         depth,         render_disp,
          render_scales, render_nums,   render_weight_depth_delta};
}

std::vector<torch::Tensor>
render_ray(const LocalMap::Ptr &_local_map_ptr, const torch::Tensor &ray_o,
           const torch::Tensor &ray_d, const int &render_type,
           const int &iter_num, const bool training) {
  static auto timer_render_ray = llog::CreateTimer("       render_ray");
  timer_render_ray->tic();

  static auto timer_raytrace = llog::CreateTimer("        raytrace");
  timer_raytrace->tic();
  auto num_rays = ray_o.size(0);

  auto nray_o = _local_map_ptr->xyz_to_m1p1_pts(ray_o);
  auto rayas_results =
      _local_map_ptr->p_acc_strcut_->raytrace(nray_o, ray_d, -1, true);

  auto ridx = rayas_results.ridx.to(torch::kLong);

  auto depth_io = _local_map_ptr->scale_from_m1p1(rayas_results.depth);

  timer_raytrace->toc_sum();

  torch::Tensor render_colors, render_depths, render_acc, render_scales,
      render_num, pts, sdf, isigma, depth, delta, alpha, disp,
      render_weight_depth_delta;

  auto sample_results = sphere_trace_adaptive_sampling(
      _local_map_ptr, ray_o, ray_d, depth_io, ridx, iter_num,
      k_sphere_trace_thr, training);
  auto sample_ridx = sample_results[0];
  depth = sample_results[1];
  delta = sample_results[2];
  auto slope = sample_results[3];

  static auto timer_index_select = llog::CreateTimer("        index_select");
  timer_index_select->tic();
  auto sample_ray_d = ray_d.index_select(0, sample_ridx).view({-1, 3});
  auto sample_ray_o = ray_o.index_select(0, sample_ridx).view({-1, 3});
  timer_index_select->toc_sum();
  auto sample_pts = torch::addcmul(sample_ray_o, sample_ray_d, depth);

  auto render_results =
      render_from_pts(_local_map_ptr, sample_pts, sample_ray_d, sample_ridx,
                      depth, delta, num_rays, render_type, slope);
  if (render_type != 2) {
    render_colors = render_results[0];
    render_acc = render_results[5];
  }
  if (render_type != 1) {
    render_depths = render_results[1];
  }
  render_scales = render_results[9];
  render_num = render_results[10];
  render_weight_depth_delta = render_results[11];
  pts = render_results[2];
  sdf = render_results[3];
  isigma = render_results[4];
  alpha = render_results[6];
  depth = render_results[7];

  timer_render_ray->toc_sum();
  return {render_colors, render_depths, render_acc, pts,
          sdf,           isigma,        depth,      delta,
          alpha,         render_scales, render_num, render_weight_depth_delta};
}

std::vector<torch::Tensor> sphere_trace_adaptive_sampling(
    const LocalMap::Ptr &_local_map_ptr, torch::Tensor ray_o,
    torch::Tensor ray_d, const torch::Tensor &depth_io, torch::Tensor ridx,
    const int &iter_num, const float &surface_thr, bool training) {
  auto grad_mode = torch::GradMode::is_enabled();
  torch::GradMode::set_enabled(false);

  std::vector<torch::Tensor> sample_ridx_vec, sample_depths_vec,
      sample_deltas_vec, sample_slope_vec;
  if (ridx.numel() > 0) {
    auto first_hit = kaolin::mark_pack_boundaries_cuda(ridx);
    auto start_idxes =
        torch::nonzero(first_hit).view({-1}).to(torch::kInt).contiguous();
    auto curr_idxes = start_idxes.clone().contiguous();

    auto hit_ridx = ridx.index_select(0, curr_idxes);
    auto hit_ray_d = ray_d.index_select(0, hit_ridx).contiguous().view({-1, 3});
    auto hit_ray_o = ray_o.index_select(0, hit_ridx).contiguous().view({-1, 3});

    auto t = depth_io.slice(1, 0, 1)
                 .index_select(0, curr_idxes)
                 .contiguous()
                 .view({-1, 1}) +
             1e-6f;
    auto pts = torch::addcmul(hit_ray_o, hit_ray_d, t).contiguous();
    auto sdf_results = _local_map_ptr->get_sdf(pts);
    auto r = sdf_results[0].contiguous();
    auto is = sdf_results[1].contiguous();
    auto IS = is.contiguous();
    auto m = -torch::ones_like(r).contiguous();
    auto M = torch::zeros_like(r).contiguous();
    auto z = r.clone().contiguous();
    auto T = (t + z).contiguous();
    auto trans = torch::ones_like(t).contiguous();

    sample_ridx_vec.reserve(iter_num + 5);
    sample_depths_vec.reserve(iter_num + 5);
    sample_deltas_vec.reserve(iter_num + 5);
    sample_slope_vec.reserve(iter_num + 5);

    auto ray_state =
        torch::ones_like(t.view({-1})).to(torch::kInt).contiguous();
    int converge_num_thr = 0;
    int i;
    for (i = 0; i < iter_num; i++) {
      pts = torch::addcmul(hit_ray_o, hit_ray_d, T);
      auto tmp_sdf_results = _local_map_ptr->get_sdf(pts);
      auto R = tmp_sdf_results[0].contiguous();
      IS = tmp_sdf_results[1].contiguous();

      auto step_mask = sphere_trace::sphere_trace_cuda(
          ray_state, r, is, trans, m, t, T, z, R, IS, M, surface_thr,
          start_idxes, curr_idxes, depth_io);

      auto step_idx = torch::nonzero(step_mask.view({-1})).view({-1});
      sample_ridx_vec.emplace_back(
          hit_ridx.index_select(0, step_idx).view({-1}));
      auto tmp_delta = z.index_select(0, step_idx).view({-1, 1});
      sample_deltas_vec.emplace_back(tmp_delta);
      sample_depths_vec.emplace_back(t.index_select(0, step_idx).view({-1, 1}) -
                                     0.5f * tmp_delta);
      sample_slope_vec.emplace_back(M.index_select(0, step_idx).view({-1, 1}));

      // ~(converge || beyond depth bound)
      if (ray_state.sum().item<int>() <= converge_num_thr) {
        break;
      }
    }
    llog::RecordValue("st_iter", i);
  }
  torch::Tensor z_near, z_far, mask_intersect;
  _local_map_ptr->get_intersect_point(ray_o, ray_d, z_near, z_far,
                                      mask_intersect);
  static int background_sample_num = k_surface_sample_num;
  auto ray_t_near = z_far.view({-1, 1});
  auto ray_t_bck = ray_t_near + k_leaf_size;
  auto bck_sample_resutls =
      stratified_sampling(ray_t_near, ray_t_bck, background_sample_num);
  auto bck_sample = bck_sample_resutls[0].view({-1, 1});
  auto bck_delta = bck_sample_resutls[1].view({-1, 1});
  auto bck_ridx = torch::arange(0, ray_o.size(0), ray_o.device())
                      .view({-1, 1})
                      .expand({ray_o.size(0), background_sample_num})
                      .contiguous()
                      .view({-1});
  sample_ridx_vec.emplace_back(bck_ridx);
  sample_depths_vec.emplace_back(bck_sample);
  sample_deltas_vec.emplace_back(bck_delta);
  sample_slope_vec.emplace_back(-torch::ones_like(bck_sample));

  // N
  auto sample_ridx = torch::cat(sample_ridx_vec, 0);
  // N, 1
  auto sample_depths = torch::cat(sample_depths_vec, 0);
  auto sample_deltas = torch::cat(sample_deltas_vec, 0);
  // N, 1
  auto sample_slope = torch::cat(sample_slope_vec, 0);

  // sort sample_ridx and sort sample_depths using the order of sample_ridx
  auto sort_results = torch::sort(sample_ridx, true, -1, false);
  sample_ridx = std::get<0>(sort_results).contiguous();
  auto order_idx = std::get<1>(sort_results);
  sample_depths =
      sample_depths.index_select(0, order_idx).contiguous().view({-1, 1});
  sample_deltas =
      sample_deltas.index_select(0, order_idx).contiguous().view({-1, 1});
  sample_slope =
      sample_slope.index_select(0, order_idx).contiguous().view({-1, 1});

  torch::GradMode::set_enabled(grad_mode);
  return {sample_ridx, sample_depths, sample_deltas, sample_slope};
}

} // namespace tracer