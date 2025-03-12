#pragma once
#include <thread>
#include <mutex>
#include <Eigen/Eigen>
#include "common_type_name.h"
#include "rog_map_cuda/config.cuh"
#include "rog_map_cuda/cuda_macro.cuh"
#include <math.h>




#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "log/"+name))

#define TINYCOLORMAP_WITH_EIGEN

namespace rog_map {
    using namespace type_utils;
    using namespace std;
    using namespace Eigen;    
    template<typename T>
    __host__ std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
        out << "[";
        for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it) {
            out << *it;
            if (it != v.end() - 1) {
                out << ", ";
            }
        }
        out << "]";
        return out;
    }

    class SlidingMap {
    public:
        struct SlidingConfig {
            double resolution;
            double resolution_inv;
//            Vec3f map_size_d;
//            Vec3f half_map_size_d;
            double sliding_thresh;
            bool map_sliding_en;
            Vec3f fix_map_origin;

            Vec3i visualization_range_i;
            Vec3i map_size_i;
            Vec3i half_map_size_i;
            int virtual_ceil_height_id_g;
            int virtual_ground_height_id_g;
            int safe_margin_i;
        } sc_;

        Vec3f local_map_origin_d_, local_map_bound_min_d_, local_map_bound_max_d_;
        Vec3i local_map_origin_i_, local_map_bound_min_i_, local_map_bound_max_i_;

        bool gpu_warmup = false;

        __host__ SlidingMap(
//                const Vec3f &map_size,
                   const Vec3i &
                   half_map_size_i,
                   const double &resolution,
                   const bool &map_sliding_en,
                   const double &sliding_thresh,
                   const Vec3f &fix_map_origin);

        __host__ void printMapInformation();

        __host__ bool insideLocalMap(const Vec3f &pos) const;

        __host__ bool insideLocalMap(const Vec3i &id_g) const;

    protected:

        __host__ virtual void resetLocalMap() = 0;

        __host__ virtual void clearMemoryOutOfMap(const std::vector<int> &clear_id, const int &i) = 0;


    public:
        __host__ void mapSliding(const Vec3f &odom);

    protected:
        __host__ int getLocalIndexHash(const Vec3i &id_in) const;

        __host__ void posToGlobalIndex(const Vec3f &pos, Vec3i &id) const;

        __host__ void posToGlobalIndex(const double &pos, int &id) const;

        __host__ void globalIndexToPos(const Vec3i &id_g, Vec3f &pos) const;

        __host__ void globalIndexToLocalIndex(const Vec3i &id_g, Vec3i &id_l) const;

        __host__ void localIndexToGlobalIndex(const Vec3i &id_l, Vec3i &id_g) const;

        __host__ void localIndexToPos(const Vec3i &id_l, Vec3f &pos) const;

        __host__ void hashIdToLocalIndex(const int &hash_id, Vec3i &id) const;

        __host__ void hashIdToPos(const int &hash_id, Vec3f &pos) const;

        __host__ int getHashIndexFromPos(const Vec3f &pos) const;

        __host__ int getHashIndexFromGlobalIndex(const Vec3i &id_g) const;
    };
}