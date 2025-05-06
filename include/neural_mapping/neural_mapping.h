#pragma once

#include <Eigen/Core>

#ifdef ENABLE_ROS
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <ros/ros.h>
#endif

#include "data_loader/data_loader.h"
#include "neural_net/local_map.h"
#include "rog_map_cuda/rog_map_class.cuh"

class NeuralSLAM {
public:
  typedef std::shared_ptr<NeuralSLAM> Ptr;
  NeuralSLAM(const int &mode, const std::filesystem::path &_config_path,
             const std::filesystem::path &_data_path = "");

#ifdef ENABLE_ROS
  NeuralSLAM(ros::NodeHandle &nh, const int &mode,
             const std::filesystem::path &_config_path,
             const std::filesystem::path &_data_path = "");
#endif

private:
  rog_map::ROGMap::Ptr rog_map_ptr;

  std::vector<torch::Tensor> vec_inrange_xyz;
  LocalMap::Ptr local_map_ptr;
  std::shared_ptr<torch::optim::Adam> p_optimizer_;

  dataloader::DataLoader::Ptr data_loader_ptr;

#ifdef ENABLE_ROS
  // ROS stuff
  ros::Subscriber rviz_pose_sub;
  ros::Publisher pose_pub, path_pub, odom_pub;
  ros::Publisher mesh_pub, mesh_color_pub, voxel_pub, vis_shift_map_pub,
      pointcloud_pub, occ_pub, unknown_pub, rgb_pub, depth_pub;

  nav_msgs::Path path_msg;

  torch::Tensor rviz_pose_ = torch::Tensor();

  void register_subscriber(ros::NodeHandle &nh);
  void register_publisher(ros::NodeHandle &nh);

  void rviz_pose_callback(const geometry_msgs::PoseConstPtr &_rviz_pose_ptr);

  void visualization(const torch::Tensor &_xyz = {},
                     std_msgs::Header _header = std_msgs::Header());
  void pub_pose(const torch::Tensor &_pose,
                std_msgs::Header _header = std_msgs::Header());
  void pub_path(std_msgs::Header _header = std_msgs::Header());
  void pub_voxel(const torch::Tensor &_xyz,
                 std_msgs::Header _header = std_msgs::Header());
  static void pub_pointcloud(const ros::Publisher &_pub,
                             const torch::Tensor &_xyz,
                             std_msgs::Header _header = std_msgs::Header());
  static void pub_pointcloud(const ros::Publisher &_pub,
                             const torch::Tensor &_xyz,
                             const torch::Tensor &_rgb,
                             std_msgs::Header _header = std_msgs::Header());
  static void pub_image(const ros::Publisher &_pub, const torch::Tensor &_image,
                        std_msgs::Header _header = std_msgs::Header());

  void pub_render_image(std_msgs::Header _header);
  void pretrain_loop();
#endif

  // thread, buffer stuff
  std::mutex mapper_buf_mutex, train_mutex;
  std::condition_variable mapperer_buf_cond;
  std::queue<torch::Tensor> mapper_pcl_buf, mapper_pose_buf;
  std::thread mapper_thread, keyboard_thread, misc_thread;

  bool save_image = false;

  void prefilter_data(const bool &export_img = false);

  bool build_occ_map();

  DepthSamples sample(DepthSamples batch_ray_samples, const int &iter,
                      const float &sample_std);

  torch::Tensor sdf_regularization(const torch::Tensor &xyz,
                                   const torch::Tensor &sdf = torch::Tensor(),
                                   const bool &curvate_enable = false,
                                   const std::string &name = "");
  std::tuple<torch::Tensor, DepthSamples> sdf_train_batch_iter(const int &iter);
  torch::Tensor color_train_batch_iter(const int &iter);
  void train(int _opt_iter);

  void train_callback(const int &_opt_iter, const RaySamples &point_samples);

  void keyboard_loop();
  void batch_train();
  void misc_loop();
  int misc_trigger = -1;

  void create_dir(const std::filesystem::path &base_path,
                  std::filesystem::path &color_path,
                  std::filesystem::path &depth_path,
                  std::filesystem::path &gt_color_path,
                  std::filesystem::path &render_color_path,
                  std::filesystem::path &gt_depth_path,
                  std::filesystem::path &render_depth_path,
                  std::filesystem::path &acc_path);
  void render_image_colormap(int idx);

  std::vector<torch::Tensor> render_image(const torch::Tensor &_pose,
                                          sensor::Cameras &camera,
                                          const float &_scale = 1.0,
                                          const bool &training = true);

  void render_path(bool eval, const int &fps = 30, const bool &save = true);
  void render_path(std::string pose_file, std::string camera_file = "",
                   const int &fps = 30);

  float export_test_image(int idx = -1, const std::string &prefix = "");
  void export_checkpoint();

  void load_pretrained(const std::filesystem::path &_pretrained_path);
  void load_checkpoint(const std::filesystem::path &_checkpoint_path);

  void save_mesh(const bool &cull_mesh = false, const std::string &prefix = "");
  void eval_mesh();
  static void eval_render();
  static void plot_log(const std::string &log_file);
  static void export_timing(bool print = false);

  bool end();
};