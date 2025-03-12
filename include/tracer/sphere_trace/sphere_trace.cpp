#include <ATen/ATen.h>

namespace sphere_trace {
void sphere_trace_cuda_impl(int64_t num_packs, int64_t num_nugs,
                            at::Tensor ray_state, at::Tensor r, at::Tensor is,
                            at::Tensor trans, at::Tensor m, at::Tensor t,
                            at::Tensor T, at::Tensor z, at::Tensor R,
                            at::Tensor IS, at::Tensor M, float surface_thr,
                            at::Tensor start_idxes_in, at::Tensor curr_idxes_in,
                            at::Tensor depth_io, at::Tensor step_mask);

at::Tensor sphere_trace_cuda(at::Tensor ray_state, at::Tensor r, at::Tensor is,
                             at::Tensor trans, at::Tensor m, at::Tensor t,
                             at::Tensor T, at::Tensor z, at::Tensor R,
                             at::Tensor IS, at::Tensor M, float surface_thr,
                             at::Tensor start_idxes_in,
                             at::Tensor curr_idxes_in, at::Tensor depth_io) {
#ifdef WITH_CUDA
  TORCH_CHECK(ray_state.is_contiguous() && r.is_contiguous() &&
              is.is_contiguous() && trans.is_contiguous() &&
              m.is_contiguous() && t.is_contiguous() && T.is_contiguous() &&
              z.is_contiguous() && R.is_contiguous() && IS.is_contiguous() &&
              M.is_contiguous());
  int64_t num_packs = ray_state.numel();
  int64_t num_nugs = depth_io.size(0);
  auto step_mask = at::zeros_like(ray_state).to(at::kBool);
  sphere_trace_cuda_impl(num_packs, num_nugs, ray_state, r, is, trans, m, t, T,
                         z, R, IS, M, surface_thr, start_idxes_in,
                         curr_idxes_in, depth_io, step_mask);
  return step_mask;
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}
} // namespace sphere_trace
