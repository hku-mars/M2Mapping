#include "rog_map_cuda/inf_map_class.cuh"
#include "rog_map_cuda/cuda_kernels/inf_map_kernels.cuh"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>

namespace rog_map {

        __host__ InfMap::InfMap(ROGMapConfig &cfg) : SlidingMap(cfg.inf_half_map_size_i,
                                       cfg.inflation_resolution,
                                       cfg.map_sliding_thresh, cfg.map_sliding_en, cfg.fix_map_origin) {
            cfg_ = cfg;
            posToGlobalIndex(cfg_.visualization_range, sc_.visualization_range_i);
            posToGlobalIndex(cfg_.virtual_ceil_height, sc_.virtual_ceil_height_id_g);
            posToGlobalIndex(cfg_.virtual_ground_height, sc_.virtual_ground_height_id_g);
            posToGlobalIndex(cfg_.safe_margin, sc_.safe_margin_i);


            md_.map_size = sc_.map_size_i.prod();
            md_.sub_grid_num = pow(static_cast<int>((cfg.inflation_resolution / cfg.resolution) + 0.5), 3);

            CHECK_ERROR(cudaMallocManaged((void**)&md_.map_data_, 3 * md_.map_size * sizeof(uint32_t)));
            CHECK_ERROR(cudaMemset(md_.map_data_, 0, 3 * md_.map_size * sizeof(uint32_t)));
            CHECK_ERROR(cudaDeviceSynchronize());

            CHECK_ERROR(cudaStreamCreate(&stream_boxSearch));
            cudaMalloc((void**)&flags_ptr, md_.map_size * sizeof(int));
            cudaMalloc((void**)&ans_ptr, md_.map_size * sizeof(Vec3f));
            cudaMallocManaged((void**)&out_ptr, md_.map_size * sizeof(Vec3f));            
            CHECK_ERROR(cudaStreamAttachMemAsync(stream_boxSearch, out_ptr));
            CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));

            CHECK_ERROR(cudaStreamCreate(&stream_query));

            md_.initialized = true;

            posToGlobalIndex(cfg_.visualization_range, sc_.visualization_range_i);
            posToGlobalIndex(cfg_.virtual_ceil_height, sc_.virtual_ceil_height_id_g);
            posToGlobalIndex(cfg_.virtual_ground_height, sc_.virtual_ground_height_id_g);

            // resetLocalMap();
            cfg = cfg_;
            CHECK_ERROR(cudaMalloc(&raw_cfg_ptr, sizeof(Vec3i) * cfg_.spherical_neighbor.size()));            
            cfg_ptr = thrust::device_ptr<Vec3i>(raw_cfg_ptr);
            thrust::copy(cfg_.spherical_neighbor.begin(), cfg_.spherical_neighbor.end(), cfg_ptr);

            std::cout << GREEN << " -- [InfMap] Init successfully -- ." << RESET << std::endl;
            printMapInformation();
        }

        __host__ InfMap::~InfMap() {
            clearMap();
        }

        __host__ void InfMap::setStream(cudaStream_t &stream){
            stream_update = stream;
            CHECK_ERROR(cudaStreamAttachMemAsync(stream_update, md_.map_data_));
        }

        __host__ void InfMap::clearMap(){
            if (md_.initialized){
                cudaStreamDestroy(stream_boxSearch);
                cudaStreamDestroy(stream_query);
                CHECK_ERROR(cudaFree(md_.map_data_));
                cudaFree(flags_ptr);
                cudaFree(ans_ptr);
                cudaFree(out_ptr);                
            }            
        }

        __host__ bool InfMap::isOccupied(const Vec3f &pos) const {
            if (!insideLocalMap(pos)) return false;
            if (pos.z() > cfg_.virtual_ceil_height - cfg_.safe_margin) return false;
            if (pos.z() < cfg_.virtual_ground_height + cfg_.safe_margin) return false;
            return md_.map_data_[getHashIndexFromPos(pos) + md_.map_size] > 0;
        }

        __host__ void InfMap::isOccupiedBatch(const std::vector<Vec3f> &pos, std::vector<bool> &out){
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

            InfMapQueryParams params(this);
            QueryOccupied<<<num_blocks, BLOCK_SIZE, 0, stream_query>>>(d_ids_ptr, ans_ptr, query_size, params);

            CHECK_ERROR(cudaStreamSynchronize(stream_query));
            for (int i = 0; i < query_size; ++i) {
                out[i] = ans_ptr[i];
            }
            CHECK_ERROR(cudaFree(d_ids_ptr));
            CHECK_ERROR(cudaFree(ans_ptr));
        }

        __host__ void InfMap::getInflationNumAndTime(double & inf_n, double & inf_t) {
            inf_n = inf_num_;
            inf_num_ = 0;
            inf_t = inf_t_;
            inf_t_ = 0;
        }

        // ***** Parallelize this part *******
        __host__ void InfMap::boxSearch(const Vec3f &box_min, const Vec3f &box_max,
                       const GridType &gt, vec_E<Vec3f> &out_points) const {
            out_points.clear();
            if (map_empty_) {
                std::cout << RED << " -- [ROG] Map is empty, cannot perform box search." << RESET << std::endl;
                return;
            }
            Vec3i box_min_id_g, box_max_id_g;
            posToGlobalIndex(box_min, box_min_id_g);
            posToGlobalIndex(box_max, box_max_id_g);
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

        __host__ void InfMap::boxSearchBatch(const Vec3f &box_min, const Vec3f &box_max,
                            const GridType &gt, vec_E<Vec3f> &out_points){
            out_points.clear();
            if (map_empty_) {
                std::cout << RED << " -- [ROG] Map is empty, cannot perform box search." << RESET << std::endl;
                return;
            }
            if ((box_max - box_min).minCoeff() <= 0) {
                std::cout << RED <<  " -- [ROG] Box search failed, box size is zero.\n"<< RESET << std::endl;
                return;
            }

            Vec3i box_min_id_g, box_max_id_g;
            posToGlobalIndex(box_min, box_min_id_g);
            posToGlobalIndex(box_max, box_max_id_g);
            Vec3i box_size = box_max_id_g - box_min_id_g;
            
            const uint32_t BLOCK_SIZE_X = 32;
            const uint32_t BLOCK_SIZE_Y = 4;
            const uint32_t BLOCK_SIZE_Z = 1;            
            const uint32_t NUM_BLOCKS_X = (box_size(0) + BLOCK_SIZE_X -1) / BLOCK_SIZE_X;
            const uint32_t NUM_BLOCKS_Y = (box_size(1) + BLOCK_SIZE_Y -1) / BLOCK_SIZE_Y;
            const uint32_t NUM_BLOCKS_Z = (box_size(2) + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z;
            dim3 block(NUM_BLOCKS_X, NUM_BLOCKS_Y, NUM_BLOCKS_Z);
            dim3 thread(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

            auto flags_dev = thrust::device_pointer_cast(flags_ptr);
            auto ans_dev = thrust::device_pointer_cast(ans_ptr);
            auto out_dev = thrust::device_pointer_cast(out_ptr);

            InfMapQueryParams params(this);            
            boxSearchInfKernel<<<block, thread, 0, stream_boxSearch>>>(params, box_min_id_g, box_size, gt, flags_ptr, ans_ptr);
            CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));
            int N_out = thrust::copy_if(thrust::cuda::par.on(stream_boxSearch),ans_dev, ans_dev + box_size.prod(), flags_dev, out_dev, is_result()) - out_dev;
            out_points.resize(N_out);
            CHECK_ERROR(cudaStreamSynchronize(stream_boxSearch));
            for (int i = 0; i < N_out; i++){
                out_points[i] = out_ptr[i];
            }
            return;
        }        

        __host__ GridType InfMap::getGridType(const Vec3f &pos) const {
            Vec3i id_g, id_l;
            if (pos.z() >= cfg_.virtual_ceil_height - cfg_.safe_margin || pos.z() <= cfg_.virtual_ground_height + cfg_.safe_margin) {
                return OCCUPIED;
            }
            posToGlobalIndex(pos, id_g);
            return getGridType(id_g);
        }

        __host__ void InfMap::getGridTypeBatch(const std::vector<Vec3f> &pos, std::vector<GridType> &out){
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

            InfMapQueryParams params(this);
            QueryInfGridType<<<num_blocks, BLOCK_SIZE, 0, stream_query>>>(d_ids_ptr, ans_ptr, query_size, params);

            CHECK_ERROR(cudaStreamSynchronize(stream_query));
            for (int i = 0; i < query_size; ++i) {
                out[i] = ans_ptr[i];
            }
            CHECK_ERROR(cudaFree(d_ids_ptr));
            CHECK_ERROR(cudaFree(ans_ptr));
        }


        __host__ void InfMap::resetLocalMap() {
            CHECK_ERROR(cudaMemsetAsync(md_.map_data_, 0, 3 * md_.map_size * sizeof(int), stream_update));
            CHECK_ERROR(cudaStreamSynchronize(stream_update));
        }

        __host__ GridType InfMap::getGridType(const Vec3i &id_g) const {
            if (!insideLocalMap(id_g)) {
                return OUT_OF_MAP;
            }
            Vec3i id_l;
            globalIndexToLocalIndex(id_g, id_l);
            int addr = getLocalIndexHash(id_l);
            int free_cnt, inflate_cnt;
            free_cnt = md_.map_data_[addr];
            inflate_cnt = md_.map_data_[addr + md_.map_size];
            // The Occupied is defined by inflation layer
            if (inflate_cnt > 0) {
                return OCCUPIED;
            } else if ((static_cast<double>(free_cnt) / md_.sub_grid_num) >= cfg_.known_free_thresh) {
                return KNOWN_FREE;
            } else {
                return UNKNOWN;
            }
        }

        __host__ void InfMap::clearMemoryOutOfMap(const std::vector<int> &clear_id, const int &i) {
            std::vector<int> ids{i, (i + 1) % 3, (i + 2) % 3};
            const int half_x = sc_.half_map_size_i(ids[1]);
            const int half_y = sc_.half_map_size_i(ids[2]);
            const int clear_id_N = clear_id.size();
            const uint32_t BLOCK_SIZE_X = 32;
            const uint32_t BLOCK_SIZE_Y = 32;
            const uint32_t BLOCK_SIZE_Z = 1;
            const uint32_t NUM_BLOCKS_X = (half_x * 2 + 1 + BLOCK_SIZE_X -1) / BLOCK_SIZE_X;
            const uint32_t NUM_BLOCKS_Y = (half_y * 2 + 1 + BLOCK_SIZE_Y -1) / BLOCK_SIZE_Y;
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
            InfMapParams params(this);
            clearMemoryOutOfInfMapKernel<<<block, thread, 0, stream_update>>>(d_ids_ptr, d_clear_id_ptr, clear_id_N, params);
            cudaStreamSynchronize(stream_update);
            CHECK_ERROR(cudaFree(d_ids_ptr));
            CHECK_ERROR(cudaFree(d_clear_id_ptr));
            return;
        }   

}
