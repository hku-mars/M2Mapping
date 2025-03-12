#include <rog_map_cuda/prob_map_class.cuh>

// #define DEBUG_KERNEL

#define FULLMASK 0xFFFFFFFF

namespace rog_map {
using namespace type_utils;
struct RayCastParams {
  // Ray Cast Params
  bool raycasting_en;
  int intensity_thresh;
  bool block_inf_pt;
  float raycast_range_min, raycast_range_max;
  float l_hit, l_miss;
  float l_max, l_min;
  float l_occ, l_free;
  // Prob Map Params
  float prob_resolution_inv;
  float prob_resolution;
  Vec3i prob_local_map_origin_i;
  Vec3i prob_half_map_size_i;
  Vec3i prob_map_size_i;
  Vec3f local_update_box_min;
  Vec3f local_update_box_max;
  float *prob_buffer_;
  // Inf Map Params
  float inf_resolution_inv;
  float inf_resolution;
  Vec3i inf_local_map_origin_i;
  Vec3i inf_half_map_size_i;
  Vec3i inf_map_size_i;
  int inf_map_size;
  uint32_t *inf_buffer_;
  // Inflation Params
  Vec3i *neighbor_ptr;
  int neighbor_N;

  RayCastParams(rog_map::ProbMap *prob_map_, rog_map::InfMap *inf_map_) {
    // Ray Cast Params
    raycasting_en = prob_map_->cfg_.raycasting_en;
    intensity_thresh = prob_map_->cfg_.intensity_thresh;
    block_inf_pt = prob_map_->cfg_.block_inf_pt;
    raycast_range_min = prob_map_->cfg_.raycast_range_min;
    raycast_range_max = prob_map_->cfg_.raycast_range_max;
    l_hit = prob_map_->cfg_.l_hit;
    l_miss = prob_map_->cfg_.l_miss;
    l_min = prob_map_->cfg_.l_min;
    l_max = prob_map_->cfg_.l_max;
    l_occ = prob_map_->cfg_.l_occ;
    l_free = prob_map_->cfg_.l_free;
    // ProbMap Params
    prob_resolution = prob_map_->sc_.resolution;
    prob_resolution_inv = prob_map_->sc_.resolution_inv;
    prob_local_map_origin_i = prob_map_->local_map_origin_i_;
    prob_half_map_size_i = prob_map_->sc_.half_map_size_i;
    prob_map_size_i = prob_map_->sc_.map_size_i;
    local_update_box_min = prob_map_->local_update_box_min;
    local_update_box_max = prob_map_->local_update_box_max;
    prob_buffer_ = prob_map_->occupancy_buffer_;
    // InfMap Params
    inf_resolution = inf_map_->sc_.resolution;
    inf_resolution_inv = inf_map_->sc_.resolution_inv;
    inf_local_map_origin_i = inf_map_->local_map_origin_i_;
    inf_half_map_size_i = inf_map_->sc_.half_map_size_i;
    inf_map_size_i = inf_map_->sc_.map_size_i;
    inf_map_size = inf_map_->md_.map_size;
    inf_buffer_ = inf_map_->md_.map_data_;
    // Inflation Params
    neighbor_ptr = inf_map_->raw_cfg_ptr;
    neighbor_N = inf_map_->cfg_.spherical_neighbor_N;
  }
};

__device__ int globalIndexToHash(const Vec3i &id_g, const Vec3i &map_size_i,
                                 const Vec3i &half_map_size_i) {
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

__device__ Vec3i GlobalProbMaptoInfMap(const Vec3i &prob_id_g,
                                       float &prob_resolution,
                                       float &inf_resolution_inv) {
  Vec3f pos = prob_id_g.cast<double>() * prob_resolution;
  return (inf_resolution_inv * pos + pos.cwiseSign() * 0.5).cast<int>();
}

__device__ void UpdateInflation(const Vec3i &id_g, RayCastParams &params,
                                int value) {
  int hash_id;
  Vec3i id_shift;
  for (size_t i = 0; i < params.neighbor_N; i++) {
    id_shift = id_g + params.neighbor_ptr[i];
    if (((id_shift - params.inf_local_map_origin_i).cwiseAbs() -
         params.inf_half_map_size_i)
            .maxCoeff() > 0) {
      // Point outside map does not need to be inflated
      continue;
    }
    hash_id = globalIndexToHash(id_shift, params.inf_map_size_i,
                                params.inf_half_map_size_i);
    atomicAdd(&params.inf_buffer_[hash_id + params.inf_map_size],
              value); // update inflation count
  }
}

__device__ __forceinline__ float atomicMinFloat(float *addr, float value) {
  float old;
  old = !signbit(value)
            ? __int_as_float(atomicMin((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMax((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__device__ __forceinline__ float atomicMaxFloat(float *addr, float value) {
  float old;
  old = !signbit(value)
            ? __int_as_float(atomicMax((int *)addr, __float_as_int(value)))
            : __uint_as_float(
                  atomicMin((unsigned int *)addr, __float_as_uint(value)));

  return old;
}

__global__ void UpdateHitKernel(float *points, uint32_t *update_hit,
                                int arr_size_, const Vec3f cur_odom,
                                const uint32_t cloud_in_size,
                                RayCastParams params, bool inflation_en = true,
                                int *status_ptr = nullptr,
                                float *value_ptr = nullptr) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cloud_in_size) {
    return;
  }

  bool should_update = update_hit[tid] > 0;
  if (!should_update)
    return;

  Vec3f p, ray_pt;
  p.x() = points[tid];
  p.y() = points[tid + arr_size_];
  p.z() = points[tid + 2 * arr_size_];

  Vec3i pt_id_g =
      (params.prob_resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();
  // Update occupancy buffer for occupied;
  int hash_id = globalIndexToHash(pt_id_g, params.prob_map_size_i,
                                  params.prob_half_map_size_i);
  float old_val = atomicAdd(&params.prob_buffer_[hash_id], params.l_hit);
  atomicMinFloat(&params.prob_buffer_[hash_id], params.l_max);
  if (inflation_en) {
    bool flag_free =
        (old_val <= params.l_free && (old_val + params.l_hit) > params.l_free);
    bool flag_occ =
        (old_val < params.l_occ && (old_val + params.l_hit) >= params.l_occ);
    /** Frontiers and surface frontiers could also be updated here in prob map
     */
    /** Update the inflation map */
    Vec3i inf_id_g = GlobalProbMaptoInfMap(pt_id_g, params.prob_resolution,
                                           params.inf_resolution_inv);
    int addr = globalIndexToHash(inf_id_g, params.inf_map_size_i,
                                 params.inf_half_map_size_i);
    atomicSub(&params.inf_buffer_[addr], uint32_t(flag_free)); // free count
    atomicAdd(&params.inf_buffer_[addr + 2 * params.inf_map_size],
              uint32_t(flag_occ)); // occ count
    if (flag_occ) {
      UpdateInflation(inf_id_g, params, 1);
    }
  }

  update_hit[tid] = 0;
  return;
}

__global__ void
UpdateMissKernel(float *points, int arr_size_, const Vec3f cur_odom,
                 const uint32_t cloud_in_size, RayCastParams params,
                 uint32_t *update_hit, bool inflation_en = true,
                 int *status_ptr = nullptr, float *value_ptr = nullptr) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cloud_in_size) {
    return;
  }
  Vec3f p, ray_pt;
  p.x() = points[tid];
  p.y() = points[tid + arr_size_];
  p.z() = points[tid + 2 * arr_size_];

  float range_norm = (p - cur_odom).norm();

  if (range_norm <= params.raycast_range_min * params.raycast_range_min) {
    return;
  }

  bool should_update{true};
  // Local Map
  if ((p - params.local_update_box_min).minCoeff() < 0 ||
      (p - params.local_update_box_max).maxCoeff() > 0) {
    p = raycaster::lineBoxIntersectPoint(
        p, cur_odom, params.local_update_box_min, params.local_update_box_max);
    should_update = false;
  }

  float dis = (p - cur_odom).norm();
  if (dis > params.raycast_range_max) {
    p = (p - cur_odom) / dis * params.raycast_range_max + cur_odom;
    should_update = false;
  }

  __syncwarp(); // synchronization point for all threads in a warp.

  points[tid] = p.x();
  points[tid + arr_size_] = p.y();
  points[tid + 2 * +arr_size_] = p.z();
  update_hit[tid] = should_update ? 1 : 0;

  if (!params.raycasting_en)
    return;

  Vec3i pt_id_g =
      (params.prob_resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();
  Vec3f raycast_start_f =
      (p - cur_odom).normalized() * params.raycast_range_min + cur_odom;
  Vec3i raycast_start = (params.prob_resolution_inv * raycast_start_f +
                         raycast_start_f.cwiseSign() * 0.5)
                            .cast<int>();
  raycaster::RayCaster raycaster;
  bool flag =
      raycaster.setInput(raycast_start.cast<float>(), pt_id_g.cast<float>());
  while (flag && raycaster.step(ray_pt)) {
    Vec3i cur_ray_id_g = ray_pt.cast<int>();
    // %%%%%% Could be removed. Need further check.
    if (((cur_ray_id_g - params.prob_local_map_origin_i).cwiseAbs() -
         params.prob_half_map_size_i)
            .maxCoeff() > 0) {
      break;
    }
    // Update occupancy buffer for free;
    int hash_id = globalIndexToHash(cur_ray_id_g, params.prob_map_size_i,
                                    params.prob_half_map_size_i);
    float old_val = atomicAdd(&params.prob_buffer_[hash_id], params.l_miss);
    atomicMaxFloat(&params.prob_buffer_[hash_id], params.l_min);
    if (inflation_en) {
      bool flag_free = (old_val > params.l_free &&
                        (old_val + params.l_miss) <= params.l_free);
      bool flag_occ =
          (old_val >= params.l_occ && (old_val + params.l_miss) < params.l_occ);
      if (!flag_occ && !flag_free)
        continue;
      /** Frontiers and surface frontiers could also be updated here in prob map
       */
      /** Update the inflation map */
      Vec3i inf_id_g = GlobalProbMaptoInfMap(
          cur_ray_id_g, params.prob_resolution, params.inf_resolution_inv);
      int addr = globalIndexToHash(inf_id_g, params.inf_map_size_i,
                                   params.inf_half_map_size_i);
      atomicAdd(&params.inf_buffer_[addr], uint32_t(flag_free)); // free count
      atomicSub(&params.inf_buffer_[addr + 2 * params.inf_map_size],
                uint32_t(flag_occ)); // occ count
      if (!flag_occ)
        continue;
      UpdateInflation(inf_id_g, params, -1);
    }
  }
  return;
}

__global__ void clearMemoryOutOfProbMapKernel(float *buffer, const int *ids,
                                              const int *clear_id,
                                              const int clear_id_N,
                                              const Vec3i half_map_size,
                                              const Vec3i map_size) {
  auto getLocalIndexHash = [](const Vec3i &id_in, const Vec3i &map_size_i,
                              const Vec3i &half_map_size_i) {
    Vec3i id = id_in + half_map_size_i;
    return id(0) * map_size_i(1) * map_size_i(2) + id(1) * map_size_i(2) +
           id(2);
  };
  int half_x = half_map_size(ids[1]);
  int half_y = half_map_size(ids[2]);
  uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
  if (tid_x > 2 * half_x || tid_y > 2 * half_y || tid_z >= clear_id_N) {
    return;
  }
  Vec3i tmp_id;
  tmp_id(ids[0]) = clear_id[tid_z];
  tmp_id(ids[1]) = tid_x - half_x;
  tmp_id(ids[2]) = tid_y - half_y;
  int addr = getLocalIndexHash(tmp_id, map_size, half_map_size);
  buffer[addr] = 0;
  return;
}

struct ProbMapQueryParams {
  Vec3i local_map_origin_i_;
  Vec3i half_map_size_i;
  Vec3i map_size_i;
  float resolution;
  float resolution_inv;
  float virtual_ceil_height;
  float virtual_ground_height;
  int virtual_ceil_height_id_g;
  int virtual_ground_height_id_g;
  int safe_margin_i;
  float safe_margin;
  float l_occ;
  float l_free;
  float *buffer_;

  ProbMapQueryParams(rog_map::ProbMap *prob_map_) {
    local_map_origin_i_ = prob_map_->local_map_origin_i_;
    half_map_size_i = prob_map_->sc_.half_map_size_i;
    map_size_i = prob_map_->sc_.map_size_i;
    resolution = prob_map_->cfg_.resolution;
    resolution_inv = prob_map_->sc_.resolution_inv;
    virtual_ceil_height = prob_map_->cfg_.virtual_ceil_height;
    virtual_ceil_height_id_g = prob_map_->sc_.virtual_ceil_height_id_g;
    virtual_ground_height = prob_map_->cfg_.virtual_ground_height;
    virtual_ground_height_id_g = prob_map_->sc_.virtual_ground_height_id_g;
    safe_margin = prob_map_->cfg_.safe_margin;
    safe_margin_i = prob_map_->sc_.safe_margin_i;
    l_occ = prob_map_->cfg_.l_occ;
    l_free = prob_map_->cfg_.l_free;
    buffer_ = prob_map_->occupancy_buffer_;
  }
};

__global__ void QueryOccupied(const Vec3i *ids, bool *out, int query_size,
                              ProbMapQueryParams params) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= query_size) {
    return;
  }

  Vec3i id_g = ids[tid];
  if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i)
          .maxCoeff() > 0) {
    out[tid] = false;
    return;
  }
  if (id_g.z() > params.virtual_ceil_height_id_g - params.safe_margin_i ||
      id_g.z() < params.virtual_ground_height_id_g + params.safe_margin_i) {
    out[tid] = true;
    return;
  }
  out[tid] =
      params.buffer_[globalIndexToHash(id_g, params.map_size_i,
                                       params.half_map_size_i)] > params.l_occ;
  return;
}

__global__ void QueryOccupied(const Vec3f *pos, bool *out, int query_size,
                              ProbMapQueryParams params) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= query_size) {
    return;
  }

  Vec3f p = pos[tid];

  Vec3i id_g = (params.resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();
  if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i)
          .maxCoeff() > 0) {
    out[tid] = false;
    return;
  }
  if (p.z() > params.virtual_ceil_height - params.safe_margin ||
      p.z() < params.virtual_ground_height + params.safe_margin) {
    out[tid] = true;
    return;
  }
  int hash_id =
      globalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
  out[tid] = params.buffer_[hash_id] > params.l_occ;
  return;
}

__global__ void QueryGridType(const Vec3f *pos, GridType *out, int query_size,
                              ProbMapQueryParams params) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= query_size) {
    return;
  }

  Vec3f p = pos[tid];
  if (p.z() > params.virtual_ceil_height - params.safe_margin ||
      p.z() < params.virtual_ground_height + params.safe_margin) {
    out[tid] = GridType::OCCUPIED;
    return;
  }

  Vec3i id_g = (params.resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();
  if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i)
          .maxCoeff() > 0) {
    out[tid] = GridType::OUT_OF_MAP;
    return;
  }

  int hash_id =
      globalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
  float ret = params.buffer_[hash_id];
  out[tid] = ret <= params.l_free ? GridType::KNOWN_FREE
                                  : (ret >= params.l_occ ? GridType::OCCUPIED
                                                         : GridType::UNKNOWN);
  return;
}

__global__ void boxSearchProbKernel(ProbMapQueryParams params,
                                    Vec3i box_min_id_g, Vec3i box_size,
                                    GridType gt, int *flag_ptr,
                                    Vec3f *ans_ptr) {
  const uint32_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;
  if (tid_x >= box_size.x() || tid_y >= box_size.y() || tid_z >= box_size.z()) {
    return;
  }
  int idx = tid_x + tid_y * box_size.x() + tid_z * box_size.x() * box_size.y();
  atomicExch(&flag_ptr[idx],
             0); // replaces the value at a specified memory location with a new
                 // value, and returns the old value.
  Vec3i id_g = box_min_id_g + Vec3i(tid_x, tid_y, tid_z);
  if (((id_g - params.local_map_origin_i_).cwiseAbs() - params.half_map_size_i)
          .maxCoeff() > 0) {
    return;
  }
  int hash_id =
      globalIndexToHash(id_g, params.map_size_i, params.half_map_size_i);
  float ret = params.buffer_[hash_id];
  GridType tmp_gt =
      ret <= params.l_free
          ? GridType::KNOWN_FREE
          : (ret >= params.l_occ ? GridType::OCCUPIED : GridType::UNKNOWN);
  atomicExch(&flag_ptr[idx], int(tmp_gt == gt));
  ans_ptr[idx] = id_g.cast<double>() * params.resolution;
  return;
}

__global__ void raySearchKernel(float *points, int arr_size_, float *pos,
                                const uint32_t ray_size,
                                const uint32_t max_hit_per_ray,
                                RayCastParams params, Vec3i box_min_id_g,
                                Vec3i box_size, GridType gt, int *flag_ptr,
                                Vec3f *ans_ptr) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= ray_size) {
    return;
  }
  Vec3f p, ray_pt;
  p.x() = points[tid];
  p.y() = points[tid + arr_size_];
  p.z() = points[tid + 2 * arr_size_];

  Vec3f cur_odom;
  cur_odom.x() = pos[tid];
  cur_odom.y() = pos[tid + ray_size];
  cur_odom.z() = pos[tid + 2 * ray_size];

  float range_norm = (p - cur_odom).norm();

  // Local Map
  if ((p - params.local_update_box_min).minCoeff() < 0 ||
      (p - params.local_update_box_max).maxCoeff() > 0) {
    p = raycaster::lineBoxIntersectPoint(
        p, cur_odom, params.local_update_box_min, params.local_update_box_max);
  }

  float dis = (p - cur_odom).norm();
  p = (p - cur_odom) / dis * params.raycast_range_max + cur_odom;

  __syncwarp(); // synchronization point for all threads in a warp.

  Vec3i pt_id_g =
      (params.prob_resolution_inv * p + p.cwiseSign() * 0.5).cast<int>();
  Vec3f raycast_start_f =
      (p - cur_odom).normalized() * params.raycast_range_min + cur_odom;
  Vec3i raycast_start = (params.prob_resolution_inv * raycast_start_f +
                         raycast_start_f.cwiseSign() * 0.5)
                            .cast<int>();
  raycaster::RayCaster raycaster;
  bool flag =
      raycaster.setInput(raycast_start.cast<float>(), pt_id_g.cast<float>());
  int hit_num = 0;
  int start_idx = tid * max_hit_per_ray;
  while (flag && raycaster.step(ray_pt)) {
    Vec3i cur_ray_id_g = ray_pt.cast<int>();
    // %%%%%% Could be removed. Need further check.
    if (((cur_ray_id_g - params.prob_local_map_origin_i).cwiseAbs() -
         params.prob_half_map_size_i)
            .maxCoeff() > 0) {
      break;
    }
    // Update occupancy buffer for free;
    int hash_id = globalIndexToHash(cur_ray_id_g, params.prob_map_size_i,
                                    params.prob_half_map_size_i);
    float ret = params.prob_buffer_[hash_id];
    GridType tmp_gt =
        ret <= params.l_free
            ? GridType::KNOWN_FREE
            : (ret >= params.l_occ ? GridType::OCCUPIED : GridType::UNKNOWN);

    auto id_l = cur_ray_id_g - box_min_id_g;
    auto idx = id_l.x() + id_l.y() * box_size.x() +
               id_l.z() * box_size.x() * box_size.y();
    atomicExch(&flag_ptr[idx], int(tmp_gt == gt));
    ans_ptr[idx] = cur_ray_id_g.cast<double>() * params.prob_resolution;
    hit_num++;

    if (tmp_gt == GridType::OCCUPIED) {
      break;
    }
  }
  return;
}
} // namespace rog_map