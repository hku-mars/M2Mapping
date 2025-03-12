#include "rog_map_cuda/ray_caster.cuh"

namespace raycaster{

    __device__ RayCaster::RayCaster() {
    }

    __device__ RayCaster::~RayCaster() {
    }

    __device__ bool RayCaster::setInput(const Eigen::Vector3f &start, const Eigen::Vector3f &end) {
        start_ = start;
        end_ = end;

        x_ = (int) floor(start_.x());
        y_ = (int) floor(start_.y());
        z_ = (int) floor(start_.z());
        endX_ = (int) floor(end_.x());
        endY_ = (int) floor(end_.y());
        endZ_ = (int) floor(end_.z());

        // Break out direction vector.
        dx_ = endX_ - x_;
        dy_ = endY_ - y_;
        dz_ = endZ_ - z_;

        // Direction to increment x,y,z when stepping.
        stepX_ = (int) signum((int) dx_);
        stepY_ = (int) signum((int) dy_);
        stepZ_ = (int) signum((int) dz_);

        // See description above. The initial values depend on the fractional
        // part of the origin.
        tMaxX_ = intbound(start_.x(), dx_);
        tMaxY_ = intbound(start_.y(), dy_);
        tMaxZ_ = intbound(start_.z(), dz_);

        // The change in t when taking a step (always positive).
        tDeltaX_ = ((float) stepX_) / dx_;
        tDeltaY_ = ((float) stepY_) / dy_;
        tDeltaZ_ = ((float) stepZ_) / dz_;

        // Avoids an infinite loop.
        if (stepX_ == 0 && stepY_ == 0 && stepZ_ == 0)
            return false;
        else
            return true;
    }

    __device__ bool RayCaster::step(Vec3f &ray_pt) {
        ray_pt = Vec3f(x_, y_, z_);

        if (x_ == endX_ && y_ == endY_ && z_ == endZ_) {
            return false;
        }

        // tMaxX stores the t-value at which we cross a cube boundary along the
        // X axis, and similarly for Y and Z. Therefore, choosing the least tMax
        // chooses the closest cube boundary. Only the first case of the four
        // has been commented in detail.
        x_ += (tMaxX_ < tMaxY_ && tMaxX_ < tMaxZ_) ? stepX_ : 0;
        tMaxX_ += (tMaxX_ < tMaxY_ && tMaxX_ < tMaxZ_) ? tDeltaX_ : 0;
        y_ += (!(tMaxX_ < tMaxY_) && tMaxY_ < tMaxZ_) ? stepY_ : 0;
        tMaxY_ += (!(tMaxX_ < tMaxY_) && tMaxY_ < tMaxZ_) ? tDeltaY_ : 0;
        z_ += (tMaxX_ < tMaxY_ && !(tMaxX_ < tMaxZ_) || !(tMaxX_ < tMaxY_) && !(tMaxY_ < tMaxZ_)) ? stepZ_ : 0;
        tMaxZ_ += (tMaxX_ < tMaxY_ && !(tMaxX_ < tMaxZ_) || !(tMaxX_ < tMaxY_) && !(tMaxY_ < tMaxZ_)) ? tDeltaZ_ : 0;

        return true;
    }

}