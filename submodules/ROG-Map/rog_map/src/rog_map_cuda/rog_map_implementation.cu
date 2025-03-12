#include "rog_map_cuda/rog_map_class.cuh"

// using namespace rog_map;
namespace rog_map{    
        __host__ ROGMap::ROGMap(ROGMapConfig &cfg) : ProbMap(cfg){

            map_info_log_file_.open(DEBUG_FILE_DIR("rm_info_log.txt"), ios::out | ios::trunc);
            time_log_file_.open(DEBUG_FILE_DIR("rm_performance_log.csv"), ios::out | ios::trunc);
            
            map_info_log_file_.close();
            for (int i = 0; i < time_consuming_name_.size(); i++) {
                time_log_file_ << time_consuming_name_[i] << ", ";
            }
            time_log_file_ << endl;
        }

        __host__ ROGMap::~ROGMap() {
            clearMap();
        };

        __host__ const ROGMapConfig &ROGMap::getCfg() const {
            return cfg_;
        }

        __host__ void ROGMap::updateRobotState(const Pose &pose) {
            robot_state_.p = pose.first;
            robot_state_.q = pose.second;
            // robot_state_.rcv_time = ros::Time::now().toSec();
            robot_state_.rcv = true;
            // robot_state_.yaw = geometry_utils::get_yaw_from_quaternion<double>(pose.second);
        }

        __host__ void ROGMap::updateMap(const PointCloudHost &cloud, const Pose &pose) {
            if (cfg_.ros_callback_en) {
                std::cout << RED << "ROS callback is enabled, can not insert map from updateMap API." << RESET << std::endl;
                return;
            }
            updateRobotState(pose);
            updateProbMap(cloud, pose);
        }

        __host__ RobotState ROGMap::getRobotState() const {
            return robot_state_;
        }
}