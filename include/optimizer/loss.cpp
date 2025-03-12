#include "loss.h"

#include "loss_utils/loss_utils.h"

namespace loss {

torch::Tensor rgb_loss(const torch::Tensor &rgb, const torch::Tensor &rgb_gt,
                       const std::string &name) {
  return torch::sqrt((rgb - rgb_gt).square() + 1e-4f).mean();
  // return torch::abs(rgb - rgb_gt).mean();
}

torch::Tensor distortion_loss(const torch::Tensor &render_dist,
                              const std::string &name) {
  return render_dist.square().mean();
}

torch::Tensor dssim_loss(const torch::Tensor &pred_image,
                         const torch::Tensor &gt_image,
                         const std::string &name) {
  auto ssim = loss_utils::ssim(pred_image.permute({2, 0, 1}).unsqueeze(0),
                               gt_image.permute({2, 0, 1}).unsqueeze(0));
  return 1.0f - ssim;
}

torch::Tensor sdf_loss(const torch::Tensor &pred_sdf,
                       const torch::Tensor &gt_sdf,
                       const torch::Tensor &pred_isigma) {
  auto isigma = pred_isigma.clamp_max(5e2f);

  if (isigma.isnan().any().item<bool>()) {
    // throw std::runtime_error("inv_bce_sigma.isnan().any()");
    isigma = isigma.nan_to_num(5e2f);
  }

  // better avoid nan
  torch::Tensor sdf_loss = torch::binary_cross_entropy_with_logits(
      (-pred_sdf * isigma).clamp(-15.0f, 15.0f),
      torch::sigmoid(-gt_sdf * isigma).clamp(1e-7f, 1 - 1e-7f));

  return sdf_loss;
}

torch::Tensor eikonal_loss(const torch::Tensor &grad, const std::string &name) {
  return (grad.norm(2, 1) - 1.0f).square().mean();
}

torch::Tensor curvate_loss(const torch::Tensor &hessian,
                           const std::string &name) {
  auto curvate_loss = hessian.sum(-1).abs().mean();
  curvate_loss.nan_to_num_(0.0f, 0.0f, 0.0f);
  return curvate_loss;
}

} // namespace loss
