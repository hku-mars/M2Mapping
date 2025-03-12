#pragma once

#include <ATen/ATen.h>

namespace sphere_trace {

at::Tensor sphere_trace_cuda(at::Tensor ray_state, at::Tensor r, at::Tensor is,
                             at::Tensor trans, at::Tensor m, at::Tensor t,
                             at::Tensor T, at::Tensor z, at::Tensor R,
                             at::Tensor IS, at::Tensor M, float surface_thr,
                             at::Tensor start_idxes_in,
                             at::Tensor curr_idxes_in, at::Tensor depth_io);
} // namespace sphere_trace
