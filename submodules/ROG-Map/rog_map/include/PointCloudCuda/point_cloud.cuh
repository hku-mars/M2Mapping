/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#pragma once

#include <PointCloudCuda/point_types.h>
#include <PointCloudCuda/thrust.h>
#include <pcl/make_shared.h>

namespace pcl
{
  namespace cuda
  {
    /** \brief misnamed class holding a 3x3 matrix */
    struct CovarianceMatrix
    {
      float3 data[3];
    };
  
    /** @b PointCloudSOA represents a SOA (Struct of Arrays) PointCloud
      * implementation for CUDA processing.
      */

    class PointCloudSOA
    {
      public:
        PointCloudSOA () : width (0), height (0), size_(0), arr_size_(0)
        {}

        PointCloudSOA (unsigned int N) : width (0), height (0), size_(0)
        {
          resize (N);
        }
        
        ~PointCloudSOA ()
        {
          clear ();
        }

        void clear(){
            if (!allocated) return;
            CHECK_ERROR(cudaFree(points));
            width = 0;
            height = 0;
            size_ = 0;
            arr_size_ = 0;
        }
        //////////////////////////////////////////////////////////////////////////////////////
        inline __host__  PointCloudSOA& operator = (const PointCloudSOA& rhs)
        {
          if (rhs.arr_size_ > arr_size_){
            resize(rhs.size());
          }
          CHECK_ERROR(cudaMemcpy(points, rhs.points, rhs.arr_size_ * sizeof(float), cudaMemcpyDeviceToDevice));
          width    = rhs.width;
          height   = rhs.height;
          is_dense = rhs.is_dense;
          size_ = rhs.size();
          return (*this);
        }

        inline __host__ float x(int id){
          return points[id];
        }

        inline __host__ float y(int id){
          return points[id + arr_size_];
        }

        inline __host__ float z(int id){
          return points[id + 2 * arr_size_];
        }

        inline __host__ float intensity(int id){
          return points[id + 3 * arr_size_];
        }

        inline __host__ const float x(int id) const{
          // printf("x: %d\n",id);
          return points[id];
        }        
  
        inline __host__ const float y(int id) const{
          // printf("y: %d\n",id);
          return points[id + arr_size_];
        }

        inline __host__ const float z(int id) const{
          // printf("z: %d\n",id);
          return points[id + 2 * arr_size_];
        }

        inline __host__ const float intensity(int id) const{
          return points[id + 3 * arr_size_];
        }
        //////////////////////////////////////////////////////////////////////////////////////
        template <typename OtherStorage>
        inline __host__ __device__ PointCloudSOA& operator << (const OtherStorage& rhs)
        {
          if (rhs.size() > arr_size_){
            resize(rhs.size());
          }
          for (int i = 0; i < rhs.size(); i++)
          {
            points[i] = rhs[i].x;
            points[i + arr_size_] = rhs[i].y;
            points[i + 2 * arr_size_] = rhs[i].z;
            points[i + 3 * arr_size_] = rhs[i].intensity;
          }
          width    = rhs.width;
          height   = rhs.height;
          is_dense = rhs.is_dense;
          size_ = rhs.size();
          return (*this);
        }

        //////////////////////////////////////////////////////////////////////////////////////

        inline __host__ PointXYZI operator [] (int id) const
        {
            return {this->x(id), this->y(id), this->z(id), this->intensity(id)};
        }
  
        /** \brief Resize the internal point data vectors.
          * \param newsize the new size
          */
        __host__ __device__ void
        resize (std::size_t newsize)
        {
          clear();
          CHECK_ERROR(cudaMallocManaged(&points, 4 * newsize * sizeof(float)));
          arr_size_ = newsize;
          allocated = true;
        }
  
        /** \brief Return the size of the internal vectors */
        __host__ __device__ std::size_t 
        size () const
        {
          return (size_);
        }

        /** \brief Bind Managed Memory to cudaStream */
        __host__ void setStream(cudaStream_t stream){
          CHECK_ERROR(cudaStreamAttachMemAsync(stream, points));
        }

        /** \brief The point data. */
        float *points;
  
        /** \brief The point cloud width (if organized as an image-structure). */
        unsigned int width;
        /** \brief The point cloud height (if organized as an image-structure). */
        unsigned int height;
        /** \brief The point cloud size.  */
        unsigned int size_;
        /** \brief The storage memory size */
        unsigned int arr_size_;        
  
        /** \brief True if no points are invalid (e.g., have NaN or Inf values). */
        bool is_dense;

        /** \brief True if no memory has been allocated. */
        bool allocated = false;
  
        using Ptr = shared_ptr<PointCloudSOA>;
        using ConstPtr = shared_ptr<const PointCloudSOA>;
    };          

  } // namespace
} // namespace
