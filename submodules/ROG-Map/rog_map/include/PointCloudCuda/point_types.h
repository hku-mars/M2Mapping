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

#include <cuda.h>
#include <PointCloudCuda/cutil_math.h>

#define CHECK_ERROR(call)                                     \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


namespace pcl
{
namespace cuda
{
  /** \brief Default point xyz-intensity structure. */
  struct /*__align__(16)*/ PointXYZI
  {
    inline __host__ __device__ PointXYZI () {}
    inline __host__ __device__ PointXYZI (float _x, float _y, float _z, float _intensity) : 
                                     x(_x), y(_y), z(_z), intensity(_intensity) {}

    // Declare a union for XYZ
    union
    {
      float3 xyz;
      struct
      {
        float x;
        float y;
        float z;
      };
    };
    float intensity;
    
    inline __host__ __device__ bool operator == (const PointXYZI &rhs)
    {
      return (x == rhs.x && y == rhs.y && z == rhs.z && intensity == rhs.intensity);
    }

    // this allows direct assignment of a PointXYZI to float3...
    inline __host__ __device__ operator float3 () const
    {
      return xyz;
    }

    const inline __host__ __device__ PointXYZI operator - (const PointXYZI &rhs) const
    {
      PointXYZI res = *this;
      res -= rhs;
      return (res);
//      xyz = -rhs.xyz;
//      intensity = -rhs.intensity;
//      return (*this -= rhs);
    }

    inline __host__ __device__ PointXYZI& operator += (const PointXYZI &rhs)
    {
      xyz += rhs.xyz;
      intensity += rhs.intensity;
      return (*this);
    }

    inline __host__ __device__ PointXYZI& operator -= (const PointXYZI &rhs)
    {
      xyz -= rhs.xyz;
      intensity -= rhs.intensity;
      return (*this);
    }

    inline __host__ __device__ PointXYZI& operator *= (const PointXYZI &rhs)
    {
      xyz *= rhs.xyz;
      intensity *= rhs.intensity;
      return (*this);
    }

    inline __host__ __device__ PointXYZI& operator /= (const PointXYZI &rhs)
    {
      xyz /= rhs.xyz;
      intensity /= rhs.intensity;
      return (*this);
    }
  };

  /** \brief Default point xyz-intensity structure. */
  struct __align__(16) PointXYZINormal
  {
    inline __host__ __device__ PointXYZINormal () {}
    inline __host__ __device__ PointXYZINormal (float _x, float _y, float _z, int _intensity) : 
                                     x(_x), y(_y), z(_z), intensity(_intensity) {}

    // Declare a union for XYZ
    union
    {
      float3 xyz;
      struct
      {
        float x;
        float y;
        float z;
      };
    };
    float intensity;
    union
    {
      float4 normal;
      struct
      {
        float normal_x;
        float normal_y;
        float normal_z;
        float curvature;
      };
    };
    
    inline __host__ __device__ bool operator == (const PointXYZINormal &rhs)
    {
      return (x == rhs.x && y == rhs.y && z == rhs.z && intensity == rhs.intensity && normal_x == rhs.normal_x && normal_y == rhs.normal_y && normal_z == rhs.normal_z);
    }

    // this allows direct assignment of a PointXYZINormal to float3...
    inline __host__ __device__ operator float3 () const
    {
      return xyz;
    }

    const inline __host__ __device__ PointXYZINormal operator - (const PointXYZINormal &rhs) const
    {
      PointXYZINormal res = *this;
      res -= rhs;
      return (res);
//      xyz = -rhs.xyz;
//      intensity = -rhs.intensity;
//      return (*this -= rhs);
    }

    inline __host__ __device__ PointXYZINormal& operator += (const PointXYZINormal &rhs)
    {
      xyz += rhs.xyz;
      intensity += rhs.intensity;
      normal += rhs.normal;
      return (*this);
    }

    inline __host__ __device__ PointXYZINormal& operator -= (const PointXYZINormal &rhs)
    {
      xyz -= rhs.xyz;
      intensity -= rhs.intensity;
      normal -= rhs.normal;
      return (*this);
    }

    inline __host__ __device__ PointXYZINormal& operator *= (const PointXYZINormal &rhs)
    {
      xyz *= rhs.xyz;
      intensity *= rhs.intensity;
      normal *= rhs.normal;
      return (*this);
    }

    inline __host__ __device__ PointXYZINormal& operator /= (const PointXYZINormal &rhs)
    {
      xyz /= rhs.xyz;
      intensity /= rhs.intensity;
      normal /= rhs.normal;
      return (*this);
    }
  };
} // namespace
} // namespace
