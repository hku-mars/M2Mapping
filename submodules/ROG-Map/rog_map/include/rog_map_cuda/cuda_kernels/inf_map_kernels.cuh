#pragma once
#include <rog_map_cuda/inf_map_class.cuh>


namespace rog_map{
    using namespace type_utils;
    struct InfMapParams{
        Vec3i local_map_origin_i_;
        Vec3i map_size_i;
        Vec3i half_map_size_i;
        uint32_t *buffer;
        Vec3i *neighbor_ptr;
        int neighbor_N;
        int map_size;

        InfMapParams(rog_map::InfMap *inf_map_){
            local_map_origin_i_ = inf_map_->local_map_origin_i_;
            half_map_size_i = inf_map_->sc_.half_map_size_i;
            map_size_i = inf_map_->sc_.map_size_i;
            map_size = inf_map_->md_.map_size;
            buffer = inf_map_->md_.map_data_;
            neighbor_ptr = inf_map_->raw_cfg_ptr;
            neighbor_N = inf_map_->cfg_.spherical_neighbor_N;       
        }
    };

    __device__ void localIndexToGlobalIndex(const Vec3i &id_l, Vec3i &id_g, InfMapParams &params) {
        for (int i = 0; i < 3; ++i) {
            /// first conver to wold index
            int min_id_g = -params.half_map_size_i(i) + params.local_map_origin_i_(i);
            int min_id_l = min_id_g % params.map_size_i(i);
            min_id_l -= min_id_l > params.half_map_size_i(i) ? params.map_size_i(i) : 0;
            min_id_l += min_id_l < -params.half_map_size_i(i) ? params.map_size_i(i) : 0;
            int cur_dis_to_min_id = id_l(i) - min_id_l;
            cur_dis_to_min_id =
                    (cur_dis_to_min_id) < 0 ? (params.map_size_i(i) + cur_dis_to_min_id) : cur_dis_to_min_id;
            int cur_id = cur_dis_to_min_id + min_id_g;
            id_g(i) = cur_id;
        }
    }

    __device__ int getLocalIndexHash(const Vec3i &id_in, const Vec3i &map_size_i, const Vec3i &half_map_size_i) {
            Vec3i id = id_in + half_map_size_i;
            return id(0) * map_size_i(1) * map_size_i(2) +
                    id(1) * map_size_i(2) +
                    id(2);
    }

    __device__ int InfGlobalIndexToHash(const Vec3i &id_g, const Vec3i &map_size_i, const Vec3i &half_map_size_i) {
        Vec3i id;
        id(0) = id_g(0) % map_size_i(0);
        id(0) += id(0) > half_map_size_i(0) ? -map_size_i(0) : 0;
        id(0) += id(0) < -half_map_size_i(0) ? map_size_i(0) : 0;
        id(1) = id_g(1) % map_size_i(1);
        id(1) += id(1) > half_map_size_i(1) ? -map_size_i(1) : 0;
        id(1) += id(1) < -half_map_size_i(1) ? map_size_i(1) : 0;
        id(2) = id_g(2) % map_size_i(2);
        id(2) += id(2) > half_map_size_i(2) ? -map_size_i(2) : 0;
        id(2) += id(2) < -half_map_size_i(2) ? map_size_i(2) : 0;                    
        return (id(0) + half_map_size_i(0)) * map_size_i(1) * map_size_i(2) +
                (id(1) + half_map_size_i(1)) * map_size_i(2) +
                (id(2) + half_map_size_i(2));        
    }

    __device__ void UpdateInflation(const Vec3i &id_g, InfMapParams &params, uint32_t value){
        int hash_id;
        Vec3i id_shift;
        for (size_t i = 0; i < params.neighbor_N; i++) {
            id_shift = id_g + params.neighbor_ptr[i];
            if (((id_shift - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i).maxCoeff() > 0) {
                // Point outside map does not need to be inflated
                continue;
            }
            hash_id = InfGlobalIndexToHash(id_shift, params.map_size_i, params.half_map_size_i);
            atomicAdd(&params.buffer[hash_id + params.map_size], value); // update inflation count
        }
    }

    __global__ void clearMemoryOutOfInfMapKernel(const int* ids, const int* clear_id, const int clear_id_N, InfMapParams params){
                                    
        int half_x = params.half_map_size_i(ids[1]);
        int half_y = params.half_map_size_i(ids[2]);
        uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
        uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
        uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
        if (tid_x > 2 * half_x || tid_y > 2 * half_y || tid_z >= clear_id_N){
            return;
        }
        Vec3i tmp_id;
        tmp_id(ids[0]) = clear_id[tid_z];
        tmp_id(ids[1]) = tid_x - half_x;
        tmp_id(ids[2]) = tid_y - half_y;
        int addr = getLocalIndexHash(tmp_id, params.map_size_i, params.half_map_size_i);
        if (params.buffer[addr + 2 * params.map_size] > 0){
            Vec3i id_g;
            localIndexToGlobalIndex(tmp_id, id_g, params);
            UpdateInflation(id_g, params, -1);
        }
        params.buffer[addr] = 0;
        params.buffer[addr + params.map_size] = 0;
        params.buffer[addr + 2 * params.map_size] = 0;
        return;
    }

    struct InfMapQueryParams{
        Vec3i local_map_origin_i_;
        Vec3i half_map_size_i;
        Vec3i map_size_i;
        float resolution;
        float resolution_inv;
        float virtual_ceil_height;
        float virtual_ground_height;
        float safe_margin;
        float known_free_thresh;
        uint32_t* buffer_;
        int map_size;
        int sub_grid_num;

        InfMapQueryParams(rog_map::InfMap *inf_map_){
            local_map_origin_i_ = inf_map_->local_map_origin_i_;
            half_map_size_i = inf_map_->sc_.half_map_size_i;
            map_size_i = inf_map_->sc_.map_size_i;
            resolution = inf_map_->sc_.resolution;
            resolution_inv = inf_map_->sc_.resolution_inv;
            virtual_ceil_height = inf_map_->cfg_.virtual_ceil_height;
            virtual_ground_height = inf_map_->cfg_.virtual_ground_height;
            safe_margin = inf_map_->cfg_.safe_margin;
            known_free_thresh = inf_map_->cfg_.known_free_thresh;
            buffer_ = inf_map_->md_.map_data_;
            map_size = inf_map_->md_.map_size;
            sub_grid_num = inf_map_->md_.sub_grid_num;
        }
    };

    __global__ void QueryOccupied(const Vec3f* pos, bool *out, int query_size, InfMapQueryParams params){
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= query_size){
            return;
        }

        Vec3f p = pos[tid];
        Vec3i id_g = (params.resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();

        if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i).maxCoeff() > 0) {
            out[tid] = false;
            return;
        }    
        if (p.z() > params.virtual_ceil_height - params.safe_margin || p.z() < params.virtual_ground_height + params.safe_margin) {
            out[tid] = true;
            return;
        }
        int addr = InfGlobalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
        out[tid] = params.buffer_[addr + params.map_size] > 0;
        return;
    }

    __global__ void QueryInfGridType(const Vec3f *pos, GridType *out, int query_size, InfMapQueryParams params){
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if (tid >= query_size){
            return;
        }

        Vec3f p = pos[tid];

        if (p.z() > params.virtual_ceil_height - params.safe_margin || p.z() < params.virtual_ground_height + params.safe_margin) {
            out[tid] = GridType::OCCUPIED;
            return;
        }

        Vec3i id_g = (params.resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();

        if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i).maxCoeff() > 0) {
            out[tid] = GridType::OUT_OF_MAP;
            return;
        }    

        int addr = InfGlobalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
        int free_cnt = params.buffer_[addr];
        int inflate_cnt = params.buffer_[addr + params.map_size];
        out[tid] = inflate_cnt>0? GridType::OCCUPIED : ((static_cast<float>(free_cnt) / params.sub_grid_num) >= params.known_free_thresh? GridType::KNOWN_FREE : GridType::UNKNOWN);
        return;
    }

    __global__ void boxSearchInfKernel(InfMapQueryParams params, Vec3i box_min_id_g, Vec3i box_size, GridType gt, int *flag_ptr, Vec3f *ans_ptr){
        const uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
        const uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
        const uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
        if (tid_x >= box_size.x() || tid_y >= box_size.y() || tid_z >= box_size.z()){
            return;
        }
        int idx = tid_x + tid_y * box_size.x() + tid_z * box_size.x() * box_size.y();
        atomicExch(&flag_ptr[idx], 0);
        Vec3i id_g = box_min_id_g + Vec3i(tid_x, tid_y, tid_z);
        if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i).maxCoeff() > 0) {
            return;
        }      
        int addr = InfGlobalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
        int free_cnt = params.buffer_[addr];
        int inflate_cnt = params.buffer_[addr + params.map_size];
        GridType tmp_gt = inflate_cnt>0? GridType::OCCUPIED : ((static_cast<float>(free_cnt) / params.sub_grid_num) >= params.known_free_thresh? GridType::KNOWN_FREE : GridType::UNKNOWN);
        atomicExch(&flag_ptr[idx], int(tmp_gt == gt));
        ans_ptr[idx] = id_g.cast<double>() * params.resolution;
        return;
    }
}
