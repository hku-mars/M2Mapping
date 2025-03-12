#pragma once

#include "neural_net/local_map.h"
namespace tracer {

std::vector<torch::Tensor>
render_ray(const LocalMap::Ptr &_local_map_ptr, const torch::Tensor &ray_o,
           const torch::Tensor &ray_d, const int &render_type = 0,
           const int &iter_num = 1, const bool training = true);

std::vector<torch::Tensor>
render_from_pts(const LocalMap::Ptr &_local_map_ptr, torch::Tensor pts,
                torch::Tensor ray_d, torch::Tensor ridx, torch::Tensor depth,
                torch::Tensor delta, const int &num_rays,
                const int &render_type = 0,
                torch::Tensor slope = torch::Tensor());

void plot_ray_analysis(const LocalMap::Ptr &_local_map_ptr, torch::Tensor pts,
                       torch::Tensor sdf, torch::Tensor depth,
                       torch::Tensor delta, torch::Tensor alpha,
                       std::string save_path = "");

std::vector<torch::Tensor> sphere_trace_adaptive_sampling(
    const LocalMap::Ptr &_local_map_ptr, torch::Tensor ray_o,
    torch::Tensor ray_d, const torch::Tensor &depth_io, torch::Tensor ridx,
    const int &iter_num = 100, const float &surface_thr = 0.001,
    bool training = true);
} // namespace tracer