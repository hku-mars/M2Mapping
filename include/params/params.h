#pragma once

#include "utils/sensor_utils/sensors.hpp"
#include <filesystem>
#include <torch/torch.h>

extern bool k_debug;
extern int k_dataset_type;
extern bool k_preload;

extern torch::Tensor k_map_origin;
extern float k_prefilter;
extern float k_max_time_diff_camera_and_pose, k_max_time_diff_lidar_and_pose;
extern bool k_prob_map_en;

extern std::filesystem::path k_dataset_path;

extern int k_ds_pt_num, k_max_pt_num;

extern std::filesystem::path k_output_path, k_package_path;

extern torch::Device k_device;
// parameter
extern float k_x_max, k_x_min, k_y_max, k_y_min, k_z_max, k_z_min, k_min_range,
    k_max_range;
extern float k_inner_map_size, k_map_size, k_map_size_inv, k_boundary_size;
extern float k_leaf_size, k_leaf_size_inv;
extern int k_octree_level, k_fill_level;
extern int k_map_resolution;

extern int k_iter_step, k_export_interval, k_export_ckp_interval,
    k_surface_sample_num, k_free_sample_num;
extern float k_color_batch_pt_num;
extern int k_batch_num;
extern float k_sample_pts_per_ray;
extern float k_sample_std;
extern bool k_outlier_remove;
extern double k_outlier_dist;
extern int k_outlier_removal_interval;

extern int k_hidden_dim, k_color_hidden_dim, k_geo_feat_dim;
extern int k_geo_num_layer, k_color_num_layer, k_dir_embedding_degree;
extern int k_n_levels, k_n_features_per_level, k_log2_hashmap_size;

// abalation parmaeter
extern float k_bce_sigma, k_bce_isigma, k_truncated_dis, k_sphere_trace_thr;
extern float k_sdf_weight, k_eikonal_weight, k_curvate_weight, k_rgb_weight,
    k_dist_weight, k_rgb_weight_end, k_rgb_weight_init;
extern float k_res_scale;

extern int k_trace_iter;

extern float k_t, k_lr, k_lr_end, k_weight_decay;

extern sensor::Sensors k_sensor;
extern torch::Tensor k_T_B_S;

// visualization
extern int k_vis_attribute;
extern int k_vis_frame_step;
extern int k_vis_batch_pt_num, k_batch_ray_num;
extern float k_vis_res, k_export_res;
extern int k_fps;

extern int k_export_colmap_format, k_export_train_pcl;
extern bool k_llff;
extern bool k_cull_mesh;

void read_params(const std::filesystem::path &_config_path,
                 const std::filesystem::path &_data_path = "",
                 const bool &_new_dir = true);
void read_scene_params(const std::filesystem::path &_scene_config_path);
void read_base_params(const std::filesystem::path &_base_config_path);
void write_pt_params();
void read_pt_params();