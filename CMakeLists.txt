cmake_minimum_required(VERSION 2.8.3)
project(neural_mapping LANGUAGES CXX CUDA)

# set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.8/)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "RelWithDebInfo")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # for clangd

# set(CMAKE_BUILD_TYPE "Release") set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-fPIC")

add_definitions(-O3 -DWITH_CUDA -DTHRUST_IGNORE_CUB_VERSION_CHECK)

option(ENABLE_ROS "Enable ROS support" OFF)

if(ENABLE_ROS)
  message("ROS ENABLED")
  add_definitions(-DENABLE_ROS)
  find_package(
    catkin REQUIRED
    COMPONENTS roscpp
               rosbag
               roslib
               std_msgs
               geometry_msgs
               nav_msgs
               mesh_msgs
               cv_bridge
               tf)
else()
  message("ROS DISABLED")
  find_package(catkin REQUIRED)
endif()

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
  message("OPENMP FOUND")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
message(Eigen: ${EIGEN3_INCLUDE_DIR})

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS
    1
    CACHE INTERNAL "No dev warnings") # 关闭pcl烦人的警告
find_package(PCL REQUIRED)

# llog
add_subdirectory(submodules/llog)
include_directories(submodules/llog/include)

# 指定libTorch位置
find_package(Torch REQUIRED)

# find_package(PythonLibs REQUIRED) include_directories(${PYTHON_INCLUDE_DIRS})

# ROG-Map
add_subdirectory(submodules/ROG-Map)
include_directories(submodules/ROG-Map/include)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS} include)

catkin_package(
  # CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime
  # DEPENDS EIGEN3 PCL INCLUDE_DIRS
)

# kaolin_wisp_cpp
add_subdirectory(submodules/kaolin_wisp_cpp)
include_directories(submodules/kaolin_wisp_cpp
                    submodules/kaolin_wisp_cpp/submodules/kaolin)

# tcnn_binding
add_subdirectory(submodules/tcnn_binding)
include_directories(
  submodules/tcnn_binding
  submodules/tcnn_binding/submodules/tiny-cuda-nn/include
  submodules/tcnn_binding/submodules/tiny-cuda-nn/dependencies)

# nerfacc_cpp
add_subdirectory(submodules/nerfacc_cpp)
include_directories(submodules/nerfacc_cpp
                    submodules/nerfacc_cpp/submodules/nerfacc)

add_library(ply_utils include/utils/ply_utils/ply_utils_pcl.cpp
                      include/utils/ply_utils/ply_utils_torch.cpp)
target_link_libraries(ply_utils ${catkin_LIBRARIES} ${PCL_LIBRARIES}
                      ${TORCH_LIBRARIES})

add_library(cumcubes include/mesher/cumcubes/src/cumcubes.cpp
                     include/mesher/cumcubes/src/cumcubes_kernel.cu)
target_link_libraries(cumcubes ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
target_include_directories(cumcubes PUBLIC include/mesher/cumcubes/include)

add_library(mesher include/utils/utils.cpp include/mesher/mesher.cpp)
target_link_libraries(mesher ${catkin_LIBRARIES} ply_utils cumcubes)

add_library(data_parser include/data_loader/data_parsers/base_parser.cpp)
target_link_libraries(data_parser ${catkin_LIBRARIES} ${TORCH_LIBRARIES}
                      ply_utils)

add_library(ray_utils include/utils/ray_utils/ray_utils.cpp)
target_link_libraries(ray_utils ${TORCH_LIBRARIES})

add_library(data_loader include/data_loader/data_loader.cpp
                        include/utils/coordinates.cpp)
target_link_libraries(data_loader data_parser ray_utils)

add_library(
  neural_net
  include/params/params.cpp include/neural_net/sub_map.cpp
  include/neural_net/encoding_map.cpp include/neural_net/local_map.cpp)
target_link_libraries(
  neural_net
  ${catkin_LIBRARIES}
  mesher
  kaolin_wisp_cpp
  tcnn_binding
  llog
  ray_utils)

add_library(
  tracer include/tracer/tracer.cpp include/tracer/sphere_trace/sphere_trace.cpp
         include/tracer/sphere_trace/sphere_trace_cuda.cu)
target_link_libraries(tracer ${catkin_LIBRARIES} ${TORCH_LIBRARIES} nerfacc
                      kaolin_wisp_cpp)

add_library(loss_utils include/optimizer/loss.cpp
                       include/optimizer/loss_utils/loss_utils.cpp)
target_link_libraries(loss_utils ${TORCH_LIBRARIES})

add_library(neural_mapping_lib include/neural_mapping/neural_mapping.cpp
                               include/params/params.cpp)
target_link_libraries(
  neural_mapping_lib
  ${OpenCV_LIBS}
  dw
  data_loader
  neural_net
  rog_map_cuda
  tracer
  loss_utils)

add_executable(neural_mapping_node src/neural_mapping_node.cpp)
target_link_libraries(neural_mapping_node neural_mapping_lib)
