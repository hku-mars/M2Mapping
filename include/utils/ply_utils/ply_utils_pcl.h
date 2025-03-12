#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace ply_utils {

void read_ply_file(const std::string &filepath,
                   pcl::PointCloud<pcl::PointXYZ> &_points);

} // namespace ply_utils