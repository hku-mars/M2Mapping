#include "rog_map_cuda/cuda_kernels/prob_map_kernels.cuh"
#include "rog_map_cuda/prob_map_class.cuh"
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#ifdef DEBUG_KERNEL
#include "rog_map_cuda/cuda_kernels/kernel_debug.cuh"
#endif

namespace rog_map {

__host__ ProbMap::ProbMap(ROGMapConfig &cfg)
    : SlidingMap(cfg.half_map_size_i, cfg.resolution, cfg.map_sliding_thresh,
                 cfg.map_sliding_en, cfg.fix_map_origin) {

  time_consuming_.resize(7);
  cfg_ = cfg;
  posToGlobalIndex(cfg_.visualization_range, sc_.visualization_range_i);
  posToGlobalIndex(cfg_.virtual_ceil_height, sc_.virtual_ceil_height_id_g);
  posToGlobalIndex(cfg_.virtual_ground_height, sc_.virtual_ground_height_id_g);
  posToGlobalIndex(cfg_.safe_margin, sc_.safe_margin_i);

  int map_size = sc_.map_size_i.prod();

  /* Update Stream and Kernel Setup */
  CHECK_ERROR(cudaStreamCreate(&stream_update));
  CHECK_ERROR(
      cudaMallocManaged((void **)&occupancy_buffer_, map_size * sizeof(float)));
  CHECK_ERROR(cudaMalloc((void **)&update_hit,
                         cfg_.CLOUD_BUFFER_SIZE * sizeof(uint32_t)));
  CHECK_ERROR(cudaStreamAttachMemAsync(stream_update, occupancy_buffer_));
  CHECK_ERROR(cudaMemsetAsync(occupancy_buffer_, 0, map_size * sizeof(float),
                              stream_update));
  CHECK_ERROR(cudaMemsetAsync(
      update_hit, 0, cfg_.CLOUD_BUFFER_SIZE * sizeof(uint32_t), stream_update));
  CHECK_ERROR(cudaStreamSynchronize(stream_update));

  CHECK_ERROR(cudaStreamCreate(&stream_boxSearch));
  cudaMalloc((void **)&flags_ptr, map_size * sizeof(int));
  cudaMalloc((void **)&ans_ptr, map_size * sizeof(Vec3f));
  cudaMallocManaged((void **)&out_ptr, map_size * sizeof(Vec3f));
  CHECK_ERROR(cudaStreamAttachMemAsync(stream_boxSearch, out_ptr));
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));

  CHECK_ERROR(cudaStreamCreate(&stream_query));

  d_cloud_.resize(cfg_.CLOUD_BUFFER_SIZE);
  d_cloud_.setStream(stream_update);
  d_pos_.resize(cfg_.CLOUD_BUFFER_SIZE);
  d_pos_.setStream(stream_update);

  inf_map_ = new InfMap(cfg);
  inf_map_->setStream(stream_update);

  cfg = cfg_;

  mapSliding(cfg.fix_map_origin);
  inf_map_->mapSliding(cfg.fix_map_origin);
  
  initialized = true;

  cout << GREEN << " -- [ProbMap] Init successfully -- ." << RESET << endl;
  printMapInformation();
}

__host__ ProbMap::~ProbMap() { clearMap(); };

__host__ void ProbMap::clearMap() {
  if (initialized) {
    cudaStreamDestroy(stream_update);
    cudaStreamDestroy(stream_boxSearch);
    cudaStreamDestroy(stream_query);
    CHECK_ERROR(cudaFree(occupancy_buffer_));
    CHECK_ERROR(cudaFree(update_hit));
    cudaFree(flags_ptr);
    cudaFree(ans_ptr);
    cudaFree(out_ptr);
    delete (inf_map_);
    initialized = false;
  }
};

__host__ void ProbMap::updateProbMap(const PointCloudHost &cloud,
                                     const Pose &pose) {
  d_cloud_ << cloud;

  Vec3f pos = pose.first;
  if (!insideLocalMap(pos)) {
    cout << YELLOW
         << " -- [ROGMapCore] cur_pose out of map range, reset the map."
         << RESET << endl;
    mapSliding(pos);
    inf_map_->mapSliding(pos);
    return;
  }

  if (pos.z() > cfg_.virtual_ceil_height) {
    cout << RED
         << " -- [ROGMapCore] Odom above virtual ceil, please check map "
            "parameter -- ."
         << RESET << endl;
    return;
  } else if (pos.z() < cfg_.virtual_ground_height) {
    cout << RED
         << " -- [ROGMapCore] Odom below virtual ground, please check map "
            "parameter -- ."
         << RESET << endl;
    return;
  }

  if ((map_empty_ ||
       (cfg_.map_sliding_en &&
        (pos - local_map_origin_d_).norm() > cfg_.map_sliding_thresh))) {
    mapSliding(pos);
    inf_map_->mapSliding(pos);
  }
  updateLocalBox(pos);

  raycastProcess(d_cloud_, pos);

  inf_map_->getInflationNumAndTime(time_consuming_[6], time_consuming_[5]);
  if (cloud.size() > 0) {
    map_empty_ = false;
    inf_map_->map_empty_ = false;
  }
}

__host__ GridType ProbMap::getGridType(Vec3i &id_g) const {
  if (id_g.z() <= sc_.virtual_ground_height_id_g + sc_.safe_margin_i ||
      id_g.z() >= sc_.virtual_ceil_height_id_g - sc_.safe_margin_i) {
    return OCCUPIED;
  }
  if (!insideLocalMap(id_g)) {
    return OUT_OF_MAP;
  }
  Vec3i id_l;
  globalIndexToLocalIndex(id_g, id_l);
  int hash_id = getLocalIndexHash(id_l);
  double ret = occupancy_buffer_[hash_id];
#ifdef USE_UNKNOWN_FLAG
  if (ret == RM_UNKNOWN_FLAG) {
    return UNKNOWN;
  } else if (ret >= cfg_.l_occ) {
    return GridType::OCCUPIED;
  } else {
    return GridType::KNOWN_FREE;
  }
#else
  if (ret <= cfg_.l_free) {
    return GridType::KNOWN_FREE;
  } else if (ret >= cfg_.l_occ) {
    return GridType::OCCUPIED;
  } else {
    return GridType::UNKNOWN;
  }
#endif
}

__host__ GridType ProbMap::getGridType(const Vec3f &pos) const {
  if (pos.z() <= cfg_.virtual_ground_height ||
      pos.z() >= cfg_.virtual_ceil_height) {
    return OCCUPIED;
  }
  if (!insideLocalMap(pos)) {
    return OUT_OF_MAP;
  }
  Vec3i id_g, id_l;
  posToGlobalIndex(pos, id_g);
  return getGridType(id_g);
}

__host__ void ProbMap::getGridTypeBatch(const std::vector<Vec3f> &pos,
                                        std::vector<GridType> &out) {
  const uint32_t BLOCK_SIZE = cfg_.GPU_BLOCKSIZE;
  const uint32_t query_size = pos.size();
  const uint32_t num_blocks = (query_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Vec3f *d_ids_ptr;
  GridType *ans_ptr;
  CHECK_ERROR(cudaMalloc(&d_ids_ptr, sizeof(Vec3f) * query_size));
  CHECK_ERROR(cudaMalloc(&ans_ptr, sizeof(GridType) * query_size));
  CHECK_ERROR(cudaStreamAttachMemAsync(stream_query, ans_ptr));
  thrust::device_ptr<Vec3f> d_ids(d_ids_ptr);
  thrust::copy(pos.begin(), pos.end(), d_ids);

  ProbMapQueryParams params(this);
  QueryGridType<<<num_blocks, BLOCK_SIZE, 0, stream_query>>>(
      d_ids_ptr, ans_ptr, query_size, params);

  CHECK_ERROR(cudaStreamSynchronize(stream_query));
  for (int i = 0; i < query_size; ++i) {
    out[i] = ans_ptr[i];
  }
  CHECK_ERROR(cudaFree(d_ids_ptr));
  CHECK_ERROR(cudaFree(ans_ptr));
}

__host__ bool ProbMap::isOccupied(const Vec3f &pos) const {
  if (!insideLocalMap(pos)) {
    return false;
  }
  if (pos.z() > cfg_.virtual_ceil_height - cfg_.safe_margin ||
      pos.z() < cfg_.virtual_ground_height + cfg_.safe_margin) {
    return true;
  }
  return occupancy_buffer_[getHashIndexFromPos(pos)] > cfg_.l_occ;
}

__host__ bool ProbMap::isOccupied(const Vec3i &id_g) const {
  if (!insideLocalMap(id_g)) {
    return false;
  }
  if (id_g.z() > sc_.virtual_ceil_height_id_g - sc_.safe_margin_i ||
      id_g.z() < sc_.virtual_ground_height_id_g + sc_.safe_margin_i) {
    return true;
  }
  return occupancy_buffer_[getHashIndexFromGlobalIndex(id_g)] > cfg_.l_occ;
}

__host__ bool ProbMap::isOccupiedInflate(const Vec3f &pos) const {
  return inf_map_->isOccupied(pos);
}

__host__ void ProbMap::isOccupiedBatch(const std::vector<Vec3i> &ids,
                                       std::vector<bool> &out) {
  const uint32_t BLOCK_SIZE = cfg_.GPU_BLOCKSIZE;
  const uint32_t query_size = ids.size();
  const uint32_t num_blocks = (query_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Vec3i *d_ids_ptr;
  bool *ans_ptr;
  CHECK_ERROR(cudaMalloc(&d_ids_ptr, sizeof(Vec3i) * query_size));
  CHECK_ERROR(cudaMallocManaged(&ans_ptr, sizeof(bool) * query_size));
  CHECK_ERROR(cudaStreamAttachMemAsync(stream_query, ans_ptr));
  thrust::device_ptr<Vec3i> d_ids(d_ids_ptr);
  thrust::copy(ids.begin(), ids.end(), d_ids);

  ProbMapQueryParams params(this);
  QueryOccupied<<<num_blocks, BLOCK_SIZE, 0, stream_query>>>(
      d_ids_ptr, ans_ptr, query_size, params);

  CHECK_ERROR(cudaStreamSynchronize(stream_query));
  for (int i = 0; i < query_size; ++i) {
    out[i] = ans_ptr[i];
  }
  CHECK_ERROR(cudaFree(d_ids_ptr));
  CHECK_ERROR(cudaFree(ans_ptr));
}

__host__ void ProbMap::isOccupiedBatch(const std::vector<Vec3f> &pos,
                                       std::vector<bool> &out) {
  const uint32_t BLOCK_SIZE = cfg_.GPU_BLOCKSIZE;
  const uint32_t query_size = pos.size();
  const uint32_t num_blocks = (query_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  Vec3f *d_ids_ptr;
  bool *ans_ptr;
  CHECK_ERROR(cudaMalloc(&d_ids_ptr, sizeof(Vec3f) * query_size));
  CHECK_ERROR(cudaMalloc(&ans_ptr, sizeof(bool) * query_size));
  CHECK_ERROR(cudaStreamAttachMemAsync(stream_query, ans_ptr));
  thrust::device_ptr<Vec3f> d_ids(d_ids_ptr);
  thrust::copy(pos.begin(), pos.end(), d_ids);

  ProbMapQueryParams params(this);
  QueryOccupied<<<num_blocks, BLOCK_SIZE, 0, stream_query>>>(
      d_ids_ptr, ans_ptr, query_size, params);

  CHECK_ERROR(cudaStreamSynchronize(stream_query));
  for (int i = 0; i < query_size; ++i) {
    out[i] = ans_ptr[i];
  }
  CHECK_ERROR(cudaFree(d_ids_ptr));
  CHECK_ERROR(cudaFree(ans_ptr));
}

__host__ void ProbMap::isOccupiedInflateBatch(const std::vector<Vec3f> &pos,
                                              std::vector<bool> &out) {
  inf_map_->isOccupiedBatch(pos, out);
}

__host__ GridType ProbMap::getInfGridType(const Vec3f &pos) const {
  return inf_map_->getGridType(pos);
}

__host__ void ProbMap::getInfGridTypeBatch(const std::vector<Vec3f> &pos,
                                           std::vector<GridType> &out) {
  inf_map_->getGridTypeBatch(pos, out);
}

__host__ double ProbMap::getMapValue(const Vec3f &pos) const {
  if (!insideLocalMap(pos)) {
    return 0;
  }
  return occupancy_buffer_[getHashIndexFromPos(pos)];
}

// ***** Parallelize this part *******
__host__ void ProbMap::boxSearch(const Vec3f &_box_min, const Vec3f &_box_max,
                                 const GridType &gt,
                                 vec_E<Vec3f> &out_points) const {
  out_points.clear();
  if (map_empty_) {
    cout << YELLOW << " -- [ROG] Map is empty, cannot perform box search."
         << RESET << endl;
    return;
  }
  if ((_box_max - _box_min).minCoeff() <= 0) {
    cout << YELLOW << " -- [ROG] Box search failed, box size is zero." << RESET
         << endl;
    return;
  }
  Vec3f box_min_d = _box_min, box_max_d = _box_max;
  boundBoxByLocalMap(box_min_d, box_max_d);
  if ((box_max_d - box_min_d).minCoeff() <= 0) {
    cout << YELLOW << " -- [ROG] Box search failed, box size is zero." << RESET
         << endl;
    return;
  }
  Vec3i box_min_id_g, box_max_id_g;
  posToGlobalIndex(box_min_d, box_min_id_g);
  posToGlobalIndex(box_max_d, box_max_id_g);
  Vec3i box_size = box_max_id_g - box_min_id_g;
  if (gt == UNKNOWN) {
    out_points.reserve(box_size.prod());
  } else {
    out_points.reserve(box_size.prod() / 3);
  }
  for (int i = box_min_id_g.x(); i <= box_max_id_g.x(); i++) {
    for (int j = box_min_id_g.y(); j <= box_max_id_g.y(); j++) {
      for (int k = box_min_id_g.z(); k <= box_max_id_g.z(); k++) {
        Vec3i id_g(i, j, k);
        if (insideLocalMap(id_g) && getGridType(id_g) == gt) {
          Vec3f pos;
          globalIndexToPos(id_g, pos);
          out_points.push_back(pos);
        }
      }
    }
  }
}

__host__ void ProbMap::boxSearchBatch(const Vec3f &_box_min,
                                      const Vec3f &_box_max, const GridType &gt,
                                      vec_E<Vec3f> &out_points) {
  out_points.clear();
  if (map_empty_) {
    cout << YELLOW << " -- [ROG] Map is empty, cannot perform box search."
         << RESET << endl;
    return;
  }
  if ((_box_max - _box_min).minCoeff() <= 0) {
    cout << YELLOW << " -- [ROG] Box search failed, box size is zero." << RESET
         << endl;
    return;
  }
  Vec3f box_min_d = _box_min, box_max_d = _box_max;
  boundBoxByLocalMap(box_min_d, box_max_d);
  if ((box_max_d - box_min_d).minCoeff() <= 0) {
    cout << YELLOW << " -- [ROG] Box search failed, box size is zero." << RESET
         << endl;
    return;
  }
  Vec3i box_min_id_g, box_max_id_g;
  posToGlobalIndex(box_min_d, box_min_id_g);
  posToGlobalIndex(box_max_d, box_max_id_g);
  Vec3i box_size = box_max_id_g - box_min_id_g + Vec3i(1, 1, 1);

  const uint32_t BLOCK_SIZE_X = 32;
  const uint32_t BLOCK_SIZE_Y = 4;
  const uint32_t BLOCK_SIZE_Z = 1;
  const uint32_t NUM_BLOCKS_X = (box_size(0) + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
  const uint32_t NUM_BLOCKS_Y = (box_size(1) + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
  const uint32_t NUM_BLOCKS_Z = (box_size(2) + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
  dim3 block(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
  dim3 thread(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

  auto flags_dev = thrust::device_pointer_cast(flags_ptr);
  auto ans_dev = thrust::device_pointer_cast(ans_ptr);
  auto out_dev = thrust::device_pointer_cast(out_ptr);
  ProbMapQueryParams params(this);
  boxSearchProbKernel<<<block, thread, 0, stream_boxSearch>>>(
      params, box_min_id_g, box_size, gt, flags_ptr, ans_ptr);
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));
  int N_out = thrust::copy_if(thrust::cuda::par.on(stream_boxSearch), ans_dev,
                              ans_dev + box_size.prod(), flags_dev, out_dev,
                              is_result()) -
              out_dev;
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));
  out_points.resize(N_out);
  for (int i = 0; i < N_out; i++) {
    out_points[i] = out_ptr[i];
  }
  return;
}

__host__ void ProbMap::boxSearchInflate(const Vec3f &box_min,
                                        const Vec3f &box_max,
                                        const GridType &gt,
                                        vec_E<Vec3f> &out_points) const {
  inf_map_->boxSearch(box_min, box_max, gt, out_points);
}

__host__ void ProbMap::boxSearchInflateBatch(const Vec3f &box_min,
                                             const Vec3f &box_max,
                                             const GridType &gt,
                                             vec_E<Vec3f> &out_points) const {
  inf_map_->boxSearchBatch(box_min, box_max, gt, out_points);
}

__host__ void ProbMap::boundBoxByLocalMap(Vec3f &box_min,
                                          Vec3f &box_max) const {
  if ((box_max - box_min).minCoeff() <= 0) {
    box_min = box_max;
    cout << RED << "-- [ROG] Bound box is invalid." << RESET << endl;
    return;
  }
  box_min = box_min.cwiseMax(local_map_bound_min_d_);
  box_max = box_max.cwiseMin(local_map_bound_max_d_);
  box_max.z() = min(box_max.z(), cfg_.virtual_ceil_height);
  box_min.z() = max(box_min.z(), cfg_.virtual_ground_height);
}

__host__ void ProbMap::raySearch(const PointCloudHost &_ray,
                                 const PointCloudHost &_pos, const GridType &gt,
                                 vec_E<Vec3f> &out_points) {
  out_points.clear();
  if (map_empty_) {
    cout << YELLOW << " -- [ROG] Map is empty, cannot perform box search."
         << RESET << endl;
    return;
  }

  d_cloud_ << _ray;
  d_pos_ << _pos;

  const uint32_t BLOCK_SIZE = cfg_.GPU_BLOCKSIZE;
  const uint32_t ray_size = _ray.size();
  /* all rays max hit size */
  static int max_hit_per_ray = sqrt(3) * sc_.map_size_i.maxCoeff();

  const uint32_t num_blocks = (ray_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

  RayCastParams raycastParams_(this, inf_map_);
  Vec3i box_min_id_g, box_max_id_g;
  posToGlobalIndex(local_map_bound_min_d_, box_min_id_g);
  posToGlobalIndex(local_map_bound_max_d_, box_max_id_g);
  Vec3i box_size = box_max_id_g - box_min_id_g + Vec3i(1, 1, 1);

  // reset flags
  CHECK_ERROR(cudaMemsetAsync(flags_ptr, 0, box_size.prod() * sizeof(int),
                              stream_boxSearch));
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));

  raySearchKernel<<<num_blocks, BLOCK_SIZE, 0, stream_boxSearch>>>(
      d_cloud_.points, d_cloud_.arr_size_, d_pos_.points, ray_size,
      max_hit_per_ray, raycastParams_, box_min_id_g, box_size, gt, flags_ptr,
      ans_ptr);
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));

  auto flags_dev = thrust::device_pointer_cast(flags_ptr);
  auto ans_dev = thrust::device_pointer_cast(ans_ptr);
  auto out_dev = thrust::device_pointer_cast(out_ptr);
  // The copy_if() function returns a pointer to the end of the output range.
  // Subtracting out_dev (a pointer to the beginning of the output range) from
  // this gives the number of elements that were copied.
  int N_out = thrust::copy_if(thrust::cuda::par.on(stream_boxSearch), ans_dev,
                              ans_dev + box_size.prod(), flags_dev, out_dev,
                              is_result()) -
              out_dev;
  CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));
  out_points.resize(N_out);
  for (int i = 0; i < N_out; i++) {
    out_points[i] = out_ptr[i];
  }

  return;
}

__host__ void ProbMap::clearMemoryOutOfMap(const std::vector<int> &clear_id,
                                           const int &i) {
  std::vector<int> ids{i, (i + 1) % 3, (i + 2) % 3};
  const int half_x = sc_.half_map_size_i(ids[1]);
  const int half_y = sc_.half_map_size_i(ids[2]);
  const int clear_id_N = clear_id.size();
  const uint32_t BLOCK_SIZE_X = 32;
  const uint32_t BLOCK_SIZE_Y = 32;
  const uint32_t BLOCK_SIZE_Z = 1;
  const uint32_t NUM_BLOCKS_X =
      (half_x * 2 + 1 + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
  const uint32_t NUM_BLOCKS_Y =
      (half_y * 2 + 1 + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y;
  const uint32_t NUM_BLOCKS_Z = (clear_id_N + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
  dim3 block(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
  dim3 thread(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

  int *d_ids_ptr, *d_clear_id_ptr;
  CHECK_ERROR(cudaMalloc(&d_ids_ptr, sizeof(int) * ids.size()));
  CHECK_ERROR(cudaMalloc(&d_clear_id_ptr, sizeof(int) * clear_id.size()));
  thrust::device_ptr<int> d_ids(d_ids_ptr);
  thrust::device_ptr<int> d_clear_id(d_clear_id_ptr);
  thrust::copy(ids.begin(), ids.end(), d_ids);
  thrust::copy(clear_id.begin(), clear_id.end(), d_clear_id);

  clearMemoryOutOfProbMapKernel<<<block, thread, 0, stream_update>>>(
      occupancy_buffer_, d_ids_ptr, d_clear_id_ptr, clear_id_N,
      sc_.half_map_size_i, sc_.map_size_i);
  CHECK_ERROR(cudaStreamSynchronize(stream_update));
  CHECK_ERROR(cudaFree(d_ids_ptr));
  CHECK_ERROR(cudaFree(d_clear_id_ptr));
  return;
}

__host__ void ProbMap::raycastProcess(const PointCloud &input_cloud,
                                      const Vec3f &cur_odom) {
  const uint32_t BLOCK_SIZE = cfg_.GPU_BLOCKSIZE;
  const uint32_t cloud_in_size = input_cloud.size();
  int map_size = sc_.map_size_i.prod();
  const uint32_t num_blocks = (cloud_in_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  RayCastParams raycastParams_(this, inf_map_);
#ifdef DEBUG_KERNEL
  KernelDebugger<float> RayCastDebugger(num_blocks, BLOCK_SIZE, "raycastKernel",
                                        stream_update);
  UpdateMissKernel<<<num_blocks, BLOCK_SIZE, 0, stream_update>>>(
      input_cloud.points, input_cloud.arr_size_, cur_odom, cloud_in_size,
      raycastParams_, update_hit, cfg_.inflation_en,
      RayCastDebugger.d_status_ptr, RayCastDebugger.d_value_ptr);
  UpdateHitKernel<<<num_blocks, BLOCK_SIZE, 0, stream_update>>>(
      input_cloud.points, update_hit, input_cloud.arr_size_, cur_odom,
      cloud_in_size, raycastParams_);
  CHECK_ERROR(cudaStreamSynchronize(stream_update));
  std::vector<float> ret;
  std::vector<int> ret_id;
  RayCastDebugger.checkStatus();
  // RayCastDebugger.checkValue(ret, ret_id, 1);
  // for (size_t i = 0; i < ret.size(); i++){
  //     printf("ret1: %d - %0.3f\n", ret_id[i], ret[i]);
  // }
#else
  UpdateMissKernel<<<num_blocks, BLOCK_SIZE, 0, stream_update>>>(
      input_cloud.points, input_cloud.arr_size_, cur_odom, cloud_in_size,
      raycastParams_, update_hit, cfg_.inflation_en);
  UpdateHitKernel<<<num_blocks, BLOCK_SIZE, 0, stream_update>>>(
      input_cloud.points, update_hit, input_cloud.arr_size_, cur_odom,
      cloud_in_size, raycastParams_, cfg_.inflation_en);
  CHECK_ERROR(cudaStreamSynchronize(stream_update));
#endif
  return;
}

__host__ void ProbMap::updateLocalBox(const Vec3f &cur_odom) {
  // The local map should be inside in index wise
  // 1) floor and ceil height
  // 2) local map size
  // The update box should follow odom.
  // The local map should follow current map center.
  Vec3i cur_odom_i;
  posToGlobalIndex(cur_odom, cur_odom_i);
  Vec3i local_updatebox_min_i, local_updatebox_max_i;
  local_updatebox_max_i = cur_odom_i + cfg_.half_local_update_box_i;
  local_updatebox_min_i = cur_odom_i - cfg_.half_local_update_box_i;
  local_update_box_min = local_updatebox_min_i.cast<double>() * sc_.resolution;
  local_update_box_max = local_updatebox_max_i.cast<double>() * sc_.resolution;

  local_map_bound_max_i_ = local_map_origin_i_ + sc_.half_map_size_i;
  local_map_bound_min_i_ = local_map_origin_i_ - sc_.half_map_size_i;
  local_map_bound_min_d_ =
      local_map_bound_min_i_.cast<double>() * sc_.resolution;
  local_map_bound_max_d_ =
      local_map_bound_max_i_.cast<double>() * sc_.resolution;

  // restrict the local map bound by ceil and ground height
  local_map_bound_min_d_.z() =
      max(local_map_bound_min_d_.z(), cfg_.virtual_ground_height);
  local_map_bound_max_d_.z() =
      min(local_map_bound_max_d_.z(), cfg_.virtual_ceil_height);

  // the local update box must insde the local map
  local_update_box_max = local_update_box_max.cwiseMin(local_map_bound_max_d_);
  local_update_box_min = local_update_box_min.cwiseMax(local_map_bound_min_d_);
}

__host__ void ProbMap::resetLocalMap() {
  // Clear local map
  int map_size = sc_.map_size_i.prod();

  // occupancy_buffer_.resize(map_size, 0);
  CHECK_ERROR(cudaMemsetAsync(occupancy_buffer_, 0, map_size * sizeof(float),
                              stream_update));
  CHECK_ERROR(cudaStreamSynchronize(stream_update));

  inf_map_->resetLocalMap();
}

__host__ void ProbMap::OutputMap(const string &file_name) const {
  int map_size = sc_.map_size_i.prod();
  float *buffer = (float *)malloc(map_size * sizeof(float));
  cudaMemcpy(buffer, occupancy_buffer_, map_size * sizeof(float),
             cudaMemcpyDeviceToHost);
  FILE *fp_unknown, *fp_occupied;
  fp_unknown = fopen(DEBUG_FILE_DIR(file_name + "_unknown.txt").c_str(), "w");
  fp_occupied = fopen(DEBUG_FILE_DIR(file_name + "_occupied.txt").c_str(), "w");
  for (int idx = -sc_.half_map_size_i(0); idx <= sc_.half_map_size_i(0);
       idx++) {
    for (int idy = -sc_.half_map_size_i(1); idy <= sc_.half_map_size_i(1);
         idy++) {
      for (int idz = -sc_.half_map_size_i(2); idz <= sc_.half_map_size_i(2);
           idz++) {
        Vec3i id_l(idx, idy, idz), id_g;
        localIndexToGlobalIndex(id_l, id_g);
        int addr = getLocalIndexHash(id_l);
        Vec3f pos;
        globalIndexToPos(id_g, pos);
        if (buffer[addr] > cfg_.l_occ) {
          fprintf(fp_occupied, "%0.4f,%0.4f,%0.4f,%0.4f\n", pos.x(), pos.y(),
                  pos.z(), buffer[addr]);
        } else if (buffer[addr] > cfg_.l_free && buffer[addr] < cfg_.l_occ) {
          fprintf(fp_unknown, "%0.4f,%0.4f,%0.4f,%0.4f\n", pos.x(), pos.y(),
                  pos.z(), buffer[addr]);
        }
      }
    }
  }
  fclose(fp_unknown);
  fclose(fp_occupied);
}
}; // namespace rog_map