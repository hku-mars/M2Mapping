#pragma once

#include "rog_map_cuda/inf_map_class.cuh"
#include "rog_map_cuda/ray_caster.cuh"

#ifndef DEBUG_FILE_DIR
#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "log/" + name))
#endif
// #define USE_UNKNOWN_FLAG
namespace rog_map {
using namespace type_utils;
using namespace std;
using namespace Eigen;
class ProbMap : public SlidingMap {
protected:
  InfMap *inf_map_;
  /// Spherical neighborhood lookup table
  // std::vector<float> occupancy_buffer_;
  bool map_empty_{true};

  vector<string> time_consuming_name_{
      "Total",          "CloudCopy",        "MapSliding",
      "RayCast",        "PointCloudNumber", "CacheNumber",
      "InflationNumber"};

  std::mutex map_update_lock_;

  cudaStream_t stream_update, stream_boxSearch, stream_query;

public:
  ROGMapConfig cfg_;
  vector<double> time_consuming_;

  Vec3f local_update_box_max, local_update_box_min;

  PointCloud d_cloud_, d_pos_;

  uint32_t *update_hit;

  float *occupancy_buffer_;

  bool initialized = false;

  /* Used for box search */

  int *flags_ptr;

  Vec3f *ans_ptr, *out_ptr;

  typedef std::shared_ptr<ProbMap> Ptr;

  //        ProbMap(ROGMapConfig &cfg) : SlidingMap(cfg.map_size_d,
  //        cfg.resolution,
  //                                                cfg.map_sliding_thresh,
  //                                                cfg.map_sliding_en,
  //                                                cfg.fix_map_origin) {
  __host__ ProbMap(ROGMapConfig &cfg);

  __host__ ~ProbMap();

  __host__ void updateProbMap(const PointCloudHost &cloud, const Pose &pose);

  __host__ GridType getGridType(Vec3i &id_g) const;

  __host__ GridType getGridType(const Vec3f &pos) const;

  __host__ void getGridTypeBatch(const std::vector<Vec3f> &pos,
                                 std::vector<GridType> &out);

  __host__ bool isOccupied(const Vec3f &pos) const;

  __host__ bool isOccupied(const Vec3i &id_g) const;

  __host__ bool isOccupiedInflate(const Vec3f &pos) const;

  __host__ void isOccupiedBatch(const std::vector<Vec3i> &ids,
                                std::vector<bool> &out);

  __host__ void isOccupiedBatch(const std::vector<Vec3f> &pos,
                                std::vector<bool> &out);

  __host__ void isOccupiedInflateBatch(const std::vector<Vec3f> &pos,
                                       std::vector<bool> &out);

  __host__ GridType getInfGridType(const Vec3f &pos) const;

  __host__ void getInfGridTypeBatch(const std::vector<Vec3f> &pos,
                                    std::vector<GridType> &out);

  __host__ double getMapValue(const Vec3f &pos) const;

  __host__ void boxSearch(const Vec3f &_box_min, const Vec3f &_box_max,
                          const GridType &gt, vec_E<Vec3f> &out_points) const;

  struct is_result {
    __device__ bool operator()(const int &x) { return x == 1; }
  };

  __host__ void boxSearchBatch(const Vec3f &_box_min, const Vec3f &_box_max,
                               const GridType &gt, vec_E<Vec3f> &out_points);

  __host__ void boxSearchInflate(const Vec3f &box_min, const Vec3f &box_max,
                                 const GridType &gt,
                                 vec_E<Vec3f> &out_points) const;

  __host__ void boxSearchInflateBatch(const Vec3f &box_min,
                                      const Vec3f &box_max, const GridType &gt,
                                      vec_E<Vec3f> &out_points) const;

  __host__ void boundBoxByLocalMap(Vec3f &box_min, Vec3f &box_max) const;

  __host__ void OutputMap(const string &file_name) const;

  __host__ void raySearch(const PointCloudHost &_ray,
                          const PointCloudHost &_pos, const GridType &gt,
                          vec_E<Vec3f> &out_points);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:
  __host__ void clearMemoryOutOfMap(const std::vector<int> &clear_id,
                                    const int &i);

  __host__ void raycastProcess(const PointCloud &input_cloud,
                               const Vec3f &cur_odom);

  __host__ void updateLocalBox(const Vec3f &cur_odom);

  __host__ void resetLocalMap();

  __host__ void clearMap();
};
} // namespace rog_map
