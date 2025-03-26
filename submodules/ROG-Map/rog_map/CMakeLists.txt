cmake_minimum_required(VERSION 3.9.0)
project(rog_map LANGUAGES CXX CUDA)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -Wall -g  -fPIC")
# set(CMAKE_CUDA_FLAGS_RELEASE
#     "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -gencode arch=compute_75,code=sm_75 "
# ) # -gencode arch=compute_87,code=sm_87
# set(CMAKE_CUDA_FLAGS_DEBUG
#     "${CMAKE_CUDA_FLAGS} -G -rdc=true -lcudadevrt -gencode arch=compute_75,code=sm_75"
# ) # -gencode arch=compute_87,code=sm_87

add_compile_options(-DQT_NO_VERSION_TAGGING)
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

find_package(catkin REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# find_package(cuda REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)

catkin_package(
  INCLUDE_DIRS include LIBRARIES rog_map CATKIN_DEPENDS
  # roscpp rospy std_msgs DEPENDS system_lib
)

include_directories(
  SYSTEM
  include
  ../../../devel/include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_cudadevrt_LIBRARY})

set(THIRD_PARTY ${catkin_LIBRARIES} ${PCL_LIBRARIES} -ldw)

file(GLOB_RECURSE rogmap_cuda_srcs CONFIGURE_DEPENDS src/rog_map_cuda/*.cu
     include/rog_map_cuda/*.cuh)
add_library(rog_map_cuda ${rogmap_cuda_srcs})

set_target_properties(rog_map_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(rog_map_cuda ${THIRD_PARTY})
