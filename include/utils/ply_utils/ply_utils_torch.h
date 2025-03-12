#pragma once

#include "tinyply.h"

#include <filesystem>
#include <torch/torch.h>

namespace ply_utils {

tinyply::Type torch_type_to_ply_type(const torch::ScalarType &dtype);

void export_to_ply(const std::filesystem::path &output_path,
                   const torch::Tensor &_xyz = torch::Tensor(),
                   const torch::Tensor &_rgb = torch::Tensor(),
                   const torch::Tensor &_origin = torch::Tensor(),
                   const torch::Tensor &_dir = torch::Tensor(),
                   const torch::Tensor &_depth = torch::Tensor(),
                   const torch::Tensor &_normal = torch::Tensor());

torch::Tensor cal_face_normal(const torch::Tensor &face_xyz);

torch::Tensor cal_face_normal_color(const torch::Tensor &face_xyz);

void face_to_ply(const torch::Tensor &face_xyz, const std::string &filename,
                 bool vis_attribute = false);

void face_to_ply(const torch::Tensor &face_xyz, const torch::Tensor &face_attr,
                 const std::string &filename);

void face_indice_to_ply(const torch::Tensor &face_xyz,
                        const torch::Tensor &face_indices,
                        const std::string &filename,
                        bool vis_attribute = false);

torch::Tensor tinyply_floatdata_to_torch_tensor(
    const std::shared_ptr<tinyply::PlyData> &_p_plydata, int _dim);

/**
 * @description:
 * @return xyz, rgb, direction, depth
 */
bool read_ply_file_to_map_tensor(
    const std::string &filepath,
    std::map<std::string, torch::Tensor> &map_ply_tensor,
    const torch::Device &_device = torch::Device(torch::kCPU));

bool read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Device &_device);

bool read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Tensor &_colors,
                             torch::Device &_device);

} // namespace ply_utils