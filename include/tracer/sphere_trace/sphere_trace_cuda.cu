#include <ATen/ATen.h>

namespace sphere_trace {
typedef unsigned int uint;

__global__ void sphere_trace_cuda_kernel(
    int64_t num_packs, int64_t num_nugs, int *ray_state, float *r, float *is,
    float *trans, float *m, float *t, float *T, float *z, float *R, float *IS,
    float *M, float surface_thr, const int *start_idxes_in, int *curr_idxes_in,
    const float2 *depth_io, bool *step_mask) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    if (ray_state[tidx] == 0) {
      return;
    }

    step_mask[tidx] = false;
    // as sdf is in training, we need more aggressive step
    auto tmp_z = T[tidx] - t[tidx];
    // take the scale into acount to slack to avoid unconverged sdf make too
    // much revert step and resutls in noisy pixel.
    float slack = max(3.0f / is[tidx], surface_thr);
    bool do_back_step = tmp_z > (abs(R[tidx]) + abs(r[tidx]) + slack);
    if ((ray_state[tidx] == 3) || (m[tidx] == -1.0f)) {
      do_back_step = false;
    }
    if (do_back_step) {
      m[tidx] = -1.0f;
    } else {
      if (ray_state[tidx] == 2) {
        // inner object is Isotropic
        M[tidx] = -1.0f;
      } else if (ray_state[tidx] == 3) {
        // Skip gap, need skip a round
        m[tidx] = -1.0f;
        M[tidx] = 0.0f;
        ray_state[tidx] = 1;
      } else {
        float tmp_M = (R[tidx] - r[tidx]) / tmp_z;
        // clip M to [-1, 1]
        tmp_M = min(0.99f, max(-1.0f, tmp_M)); // never shrink it to [-1,0]
        float step_relaxation = 0.8f;
        m[tidx] = step_relaxation * m[tidx] + (1.0f - step_relaxation) * tmp_M;
        M[tidx] = m[tidx];
      }
      if (M[tidx] < 0.0f) {
        float density = -M[tidx] * is[tidx] / (1 + exp(r[tidx] * is[tidx]));
        trans[tidx] *= exp(-density * tmp_z);
        step_mask[tidx] = true;
        z[tidx] = tmp_z;
      }

      is[tidx] = IS[tidx];
      r[tidx] = R[tidx];
      t[tidx] = T[tidx];
      if ((ray_state[tidx] == 2) && (trans[tidx] < 1e-3f)) {
        ray_state[tidx] = 0;
        return;
      }
      if (r[tidx] <= surface_thr) {
        ray_state[tidx] = 2;
      }
    }
    float omega = 2.0f / (1.0f - m[tidx]);
    // avoid stuck in 0
    float tmp_step = max(abs(r[tidx]) * omega, surface_thr);
    T[tidx] = t[tidx] + tmp_step;

    // find next depth bound
    uint max_iidx =
        (tidx == num_packs - 1) ? num_nugs : start_idxes_in[tidx + 1];

    int iidx = curr_idxes_in[tidx];
    float query = t[tidx];
    float entry = depth_io[iidx].x;
    float exit = depth_io[iidx].y;
    while (true) {
      if ((query >= entry && query <= exit) || (query < entry)) {
        break;
      }
      iidx++;
      if (iidx >= max_iidx) {
        // it matter: helps improve strcuture
        if (query > exit) {
          ray_state[tidx] = 0;
        }
        return;
      }

      entry = depth_io[iidx].x;
      if ((entry - exit) > 1e-6) {
        T[tidx] = entry + 1e-6f;
        ray_state[tidx] = 3;
      }
      exit = depth_io[iidx].y;
    }
    curr_idxes_in[tidx] = iidx;
  }
}
void sphere_trace_cuda_impl(int64_t num_packs, int64_t num_nugs,
                            at::Tensor ray_state, at::Tensor r, at::Tensor is,
                            at::Tensor trans, at::Tensor m, at::Tensor t,
                            at::Tensor T, at::Tensor z, at::Tensor R,
                            at::Tensor IS, at::Tensor M, float surface_thr,
                            at::Tensor start_idxes_in, at::Tensor curr_idxes_in,
                            at::Tensor depth_io, at::Tensor step_mask) {
  sphere_trace_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_nugs, ray_state.data_ptr<int>(), r.data_ptr<float>(),
      is.data_ptr<float>(), trans.data_ptr<float>(), m.data_ptr<float>(),
      t.data_ptr<float>(), T.data_ptr<float>(), z.data_ptr<float>(),
      R.data_ptr<float>(), IS.data_ptr<float>(), M.data_ptr<float>(),
      surface_thr, start_idxes_in.data_ptr<int>(),
      curr_idxes_in.data_ptr<int>(),
      reinterpret_cast<float2 *>(depth_io.data_ptr<float>()),
      step_mask.data_ptr<bool>());
}
} // namespace sphere_trace
