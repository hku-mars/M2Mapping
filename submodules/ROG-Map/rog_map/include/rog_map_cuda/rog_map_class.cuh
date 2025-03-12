#pragma once

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "common_type_name.h"

#include "rog_map_cuda/config.cuh"
#include "rog_map_cuda/prob_map_class.cuh"

#include <fstream>


#define DEBUG_FILE_DIR(name) (string(string(ROOT_DIR) + "log/"+name))

#define TINYCOLORMAP_WITH_EIGEN

namespace rog_map {
    using namespace type_utils;
    using namespace std;
    using namespace Eigen;

    class ROGMap : public ProbMap {
    private:
        RobotState robot_state_;

        std::ofstream time_log_file_, map_info_log_file_;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        typedef shared_ptr<ROGMap> Ptr;

        __host__ ROGMap(ROGMapConfig &cfg);

        __host__ ~ROGMap();

        // For user update
    public:

        __host__ const ROGMapConfig &getCfg() const;

        __host__ void updateRobotState(const Pose &pose);

        __host__ void updateMap(const PointCloudHost &cloud, const Pose &pose);

        __host__ RobotState getRobotState() const;
    };
}

