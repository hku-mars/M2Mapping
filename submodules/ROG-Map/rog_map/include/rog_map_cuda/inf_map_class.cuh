#pragma once

#include "rog_map_cuda/sliding_map_class.cuh"
#include <string.h>

#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "log/"+name))

namespace rog_map {
    using namespace type_utils;
    using namespace std;
    using namespace Eigen;    
    class InfMap : public SlidingMap {
    private:
        int inf_num_{0};
        double inf_t_{0.0};

    public:
        typedef std::shared_ptr<InfMap> Ptr;
        ROGMapConfig cfg_;

        struct MapData {
            // std::vector<uint16_t> occupied_cnt;
            // std::vector<uint16_t> known_free_cnt;
            // std::vector<uint16_t> inflate_cnt;
            bool initialized = false;
            uint32_t *map_data_; // {0}: known free, {1}: inflate, {2}: occupied
            int sub_grid_num;
            int map_size;
        } md_;

        /* Used for Box Search */

        cudaStream_t stream_update, stream_boxSearch, stream_query;

        int *flags_ptr;
        
        Vec3f *ans_ptr, *out_ptr;

        Vec3i *raw_cfg_ptr;

        std::mutex map_update_lock_;

        bool map_empty_{true};

        thrust::device_ptr<Vec3i> cfg_ptr;

//        InfMap(ROGMapConfig &cfg) : SlidingMap(cfg.map_size_d + 2 * Vec3f::Constant(cfg.inflation_resolution),
//                                               cfg.inflation_resolution,
//                                               cfg.map_sliding_thresh, cfg.map_sliding_en, cfg.fix_map_origin) {

        __host__ InfMap(ROGMapConfig &cfg);

        __host__ ~InfMap();

        __host__ void setStream(cudaStream_t &stream);

        __host__ bool isOccupied(const Vec3f &pos) const;

        __host__ void getInflationNumAndTime(double & inf_n, double & inf_t);

        // void updateInflation(const Vec3f &pos, const GridType &occ_type);

        __host__ void boxSearch(const Vec3f &box_min, const Vec3f &box_max,
                       const GridType &gt, vec_E<Vec3f> &out_points) const;

        struct is_result{
            __device__ bool operator()(const int &x){
                return x == 1;
            }
        };

        __host__ void boxSearchBatch(const Vec3f &box_min, const Vec3f &box_max,
                            const GridType &gt, vec_E<Vec3f> &out_points);                     

        __host__ GridType getGridType(const Vec3f &pos) const;

        __host__ void getGridTypeBatch(const std::vector<Vec3f> &pos, std::vector<GridType> &out);

        __host__ void resetLocalMap() override;

        __host__ void resetFlags();

        __host__ void isOccupiedBatch(const std::vector<Vec3f> &pos, std::vector<bool> &out);

    private:

        __host__ GridType getGridType(const Vec3i &id_g) const;

        __host__ void clearMemoryOutOfMap(const std::vector<int> &clear_id, const int &i);

        __host__ void clearMap();

    };
}

