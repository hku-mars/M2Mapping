#pragma once

#include <Eigen/Eigen>
#include <vector>
#include <common_type_name.h>

namespace raycaster {
    using namespace type_utils;
    __device__ inline static int signum(int x) {
        return x == 0 ? 0 : x < 0 ? -1 : 1;
    }

    __device__ inline static float mod(float value, float modulus) {
        return fmod(fmod(value, modulus) + modulus, modulus);
    }

    __device__ inline static float intbound(float s, float ds) {
        // Find the smallest positive t such that s+t*ds is an integer.
        if (ds < 0) {
            // return intbound(-s, -ds);
            s = mod(-s, 1);
            return (1 - s) / (-ds);
        } else {
            s = mod(s, 1);
            // problem is now s+t*ds = 1
            return (1 - s) / ds;
        }
    }

    __device__ __host__ static Vec3f lineBoxIntersectPoint(const Vec3f &pt, const Vec3f &pos,
                                       const Vec3f &box_min, const Vec3f &box_max) {
        Eigen::Vector3f diff, max_tc, min_tc;
        diff(0) = pt(0) - pos(0);
        diff(1) = pt(1) - pos(1);
        diff(2) = pt(2) - pos(2);
        max_tc(0) = box_max(0) - pos(0);
        max_tc(1) = box_max(1) - pos(1);
        max_tc(2) = box_max(2) - pos(2);
        min_tc(0) = box_min(0) - pos(0);
        min_tc(1) = box_min(1) - pos(1);
        min_tc(2) = box_min(2) - pos(2);

        float min_t = 1000000;

        min_t = (fabs(diff[0])>0 && max_tc[0]/diff[0]>0 && max_tc[0]/diff[0]<min_t)? max_tc[0]/diff[0]:min_t;
        min_t = (fabs(diff[0])>0 && min_tc[0]/diff[0]>0 && min_tc[0]/diff[0]<min_t)? min_tc[0]/diff[0]:min_t;
        min_t = (fabs(diff[1])>0 && max_tc[1]/diff[1]>0 && max_tc[1]/diff[1]<min_t)? max_tc[1]/diff[1]:min_t;
        min_t = (fabs(diff[1])>0 && min_tc[1]/diff[1]>0 && min_tc[1]/diff[1]<min_t)? min_tc[1]/diff[1]:min_t;
        min_t = (fabs(diff[2])>0 && max_tc[2]/diff[2]>0 && max_tc[2]/diff[2]<min_t)? max_tc[2]/diff[2]:min_t;
        min_t = (fabs(diff[2])>0 && min_tc[2]/diff[2]>0 && min_tc[2]/diff[2]<min_t)? min_tc[2]/diff[2]:min_t;

        Vec3f ret;
        ret(0) = pos(0) + (min_t - 1e-3) * diff(0);
        ret(1) = pos(1) + (min_t - 1e-3) * diff(1);
        ret(2) = pos(2) + (min_t - 1e-3) * diff(2);

        return ret;
    }

    class RayCaster {
    private:
        /* data */
        Eigen::Vector3f start_;
        Eigen::Vector3f end_;
        Eigen::Vector3f direction_;
        Eigen::Vector3f min_;
        Eigen::Vector3f max_;
        int x_;
        int y_;
        int z_;
        int endX_;
        int endY_;
        int endZ_;
        float dx_;
        float dy_;
        float dz_;
        int stepX_;
        int stepY_;
        int stepZ_;
        float tMaxX_;
        float tMaxY_;
        float tMaxZ_;
        float tDeltaX_;
        float tDeltaY_;
        float tDeltaZ_;

        int step_num_;

    public:
        __device__ RayCaster();

        __device__ ~RayCaster();

        __device__ bool setInput(const Eigen::Vector3f &start, const Eigen::Vector3f &end);

        __device__ bool step(Vec3f &ray_pt);
    };
}

