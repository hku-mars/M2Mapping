#pragma once
#include "vector"
#include "Eigen/Dense"
#include "string"
#include "common_type_name.h"

#include <PointCloudCuda/point_types.h>
#include <PointCloudCuda/point_cloud.cuh>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>


namespace rog_map {
    using namespace type_utils;
    using namespace std;
    #define RM_UNKNOWN_FLAG (-99999)
    typedef pcl::cuda::PointXYZI PclPoint;
    typedef pcl::cuda::PointCloudSOA PointCloud; 
    typedef pcl::PointXYZI PclPointHost;
    typedef pcl::PointCloud<PclPointHost> PointCloudHost;        

    class ROGMapConfig {
    public:
        double safe_margin{0};
        double resolution, inflation_resolution;
        int inflation_step;
        bool visualization_en{false}, frontier_extraction_en{false},
                raycasting_en{true}, ros_callback_en{false}, pub_unknown_map_en{false};

        bool block_inf_pt;

        std::vector<Vec3i> spherical_neighbor;
        size_t spherical_neighbor_N = 0;

        /* intensity noise filter*/
        int intensity_thresh;
        /* aster properties */
        string frame_id;
        bool map_sliding_en{true};
        bool inflation_en{true};
        Vec3f fix_map_origin;
        Vec3f local_update_box_d, half_local_update_box_d;
        Vec3i local_update_box_i, half_local_update_box_i;
        string odom_topic, cloud_topic;
        /* probability update */
        double raycast_range_min, raycast_range_max;
        int point_filt_num, batch_update_size;
        float p_hit, p_miss, p_min, p_max, p_occ, p_free;
        float l_hit, l_miss, l_min, l_max, l_occ, l_free;

        double virtual_ceil_height, virtual_ground_height;

        double odom_timeout;
        Vec3f visualization_range;
        double viz_time_rate;
        int viz_frame_rate;

        double known_free_thresh;
        double map_sliding_thresh;
        Vec3f map_size_d, half_map_size_d;
        Vec3i inf_half_map_size_i, half_map_size_i;

        int GPU_BLOCKSIZE;
        int CLOUD_BUFFER_SIZE;
    };

    class ROSParamLoader {
    public:
        __host__ void resetMapSize(ROGMapConfig &cfg);

        struct less_than_key{
            inline __device__ bool operator() (const Vec3i &a, const Vec3i &b){
                return a.norm() < b.norm();
            }
        };

        __host__ void initConfig(ROGMapConfig &cfg);
    };

}

