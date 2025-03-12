#include "mesher.h"
#include "cumcubes.hpp"
#include "utils/tqdm.hpp"

using namespace std;

namespace mesher {

#ifdef ENABLE_ROS
void tensor_to_mesh(mesh_msgs::MeshGeometry &mesh,
                    mesh_msgs::MeshVertexColors &mesh_color,
                    const torch::Tensor &vertice, const torch::Tensor &face,
                    const torch::Tensor &color) {
  auto vertex_num = vertice.size(0);
  auto face_num = face.size(0);
  mesh.faces.resize(face_num);
  mesh.vertices.resize(vertex_num);
  mesh_color.vertex_colors.resize(vertex_num);

  auto vertice_float64 = vertice.to(torch::kFloat64).contiguous().cpu();
  torch::Tensor color_cat;
  TORCH_CHECK(color.size(-1) == 3)
  color_cat =
      torch::cat({color, torch::ones({color.size(0), 1}, color.options())}, 1)
          .contiguous()
          .cpu();

  memcpy(mesh.vertices.data(), vertice_float64.data_ptr(),
         vertice_float64.numel() * sizeof(double));
  memcpy(mesh_color.vertex_colors.data(), color_cat.data_ptr(),
         color_cat.numel() * sizeof(float));

  auto face_int32 = face.to(torch::kInt32).contiguous().cpu();
  memcpy(mesh.faces.data(), face_int32.data_ptr(),
         face_int32.numel() * sizeof(uint32_t));
}
#endif

Mesher::Mesher() {};

void Mesher::save_mesh(const std::string &output_path, bool vis_attribute,
                       const std::string &prefix,
                       dataparser::DataParser::Ptr dataparser_ptr_) {
  if (vec_face_.empty()) {
    cout << "\n"
         << "No mesh to save! Please first meshing using 'v' first!" << "\n";
    return;
  }

  auto vertice_attrs = torch::cat({vec_face_attr_}, 0).contiguous();
  auto face = torch::cat({vec_face_}, 0).contiguous();
  auto vertices = torch::cat({vec_vertice_}, 0).contiguous();
  std::string filename = output_path + "/" + prefix + "mesh.ply";
  if (vertice_attrs.numel() == 0) {
    vertice_attrs = torch::full_like(vertices, 127).to(torch::kUInt8);
  }
  mc::save_mesh_as_ply(filename, vertices, face, vertice_attrs);
  if (dataparser_ptr_) {
    std::string cull_filename = output_path + "/" + prefix + "mesh_culled.ply";
    auto cull_results =
        cull_mesh(dataparser_ptr_, vertices, face, vertice_attrs);
    mc::save_mesh_as_ply(cull_filename, cull_results[0], cull_results[1],
                         cull_results[2]);
  }
  vec_face_attr_.clear();
  vec_face_.clear();
  vec_vertice_.clear();
  printf("\033[1;32mExport mesh saved to %s\n\033[0m", filename.c_str());
}

std::vector<torch::Tensor>
Mesher::cull_mesh(dataparser::DataParser::Ptr dataparser_ptr_,
                  torch::Tensor vertices, torch::Tensor faces,
                  torch::Tensor vertice_attrs) {
  int n_imgs = dataparser_ptr_->raw_depth_filelists_.size();
  auto whole_mask =
      torch::ones(vertices.size(0), vertices.options()).to(torch::kBool);
  auto device = vertices.device();
  auto iter_bar = tq::trange(n_imgs);
  iter_bar.set_prefix("Culling mesh");
  for (const auto &i : iter_bar) {
    auto pose =
        dataparser_ptr_->get_pose(i, dataparser::DataType::RawDepth).to(device);
    auto depth = dataparser_ptr_->get_depth_image(i).to(device);

    auto w2c = torch::inverse(pose);

    static auto K = torch::tensor({{dataparser_ptr_->sensor_.camera.fx, 0.f,
                                    dataparser_ptr_->sensor_.camera.cx},
                                   {0.f, dataparser_ptr_->sensor_.camera.fy,
                                    dataparser_ptr_->sensor_.camera.cy},
                                   {0.f, 0.f, 1.f}},
                                  device);
    auto ones = torch::ones({vertices.size(0), 1}, torch::kFloat32).to(device);
    auto homo_points = torch::cat({vertices, ones}, 1).reshape({-1, 4, 1});
    auto cam_cord_homo = w2c.matmul(homo_points);
    // [N, 3, 1]
    auto cam_cord = cam_cord_homo.slice(1, 0, 3);
    // [N, 1, 1]
    auto z = cam_cord.slice(1, -1);

    auto uv = cam_cord / z.abs();
    // auto first_col = cam_cord.index_select(1, torch::tensor({0}));
    // // Multiply the first column by -1
    // first_col *= -1;
    // cam_cord.index_copy_(1, torch::tensor({0}), first_col);

    // [N, 3, 1]
    uv = K.matmul(uv);
    // [N,2]
    uv = uv.slice(1, 0, 2).squeeze(-1);

    // https://pytorch.org/docs/main/generated/torch.nn.functional.grid_sample.html#torch.nn.functional.grid_sample
    static auto W = dataparser_ptr_->sensor_.camera.width;
    static auto H = dataparser_ptr_->sensor_.camera.height;
    auto grid_x = uv.slice(1, 0, 1) / W;
    auto grid_y = uv.slice(1, 1) / H;

    auto grid = torch::cat({grid_x, grid_y}, 1);
    // need range in [-1,1]
    grid = 2 * grid - 1;
    // [1,1,H,W]
    auto input = depth.unsqueeze(0).unsqueeze(1).squeeze(-1);
    // [1,1,N,2(W+H)]
    auto flow_field = grid.unsqueeze(0).unsqueeze(1);
    auto depth_samples = torch::nn::functional::grid_sample(
                             input, flow_field,
                             torch::nn::functional::GridSampleFuncOptions()
                                 .padding_mode(torch::kZeros)
                                 .align_corners(true))
                             .squeeze();

    auto mask = (0 <= z.squeeze()) & (uv.select(1, 0) < W) &
                (uv.select(1, 0) > 0) & (uv.select(1, 1) < H) &
                (uv.select(1, 1) > 0);
    auto depth_mask = (depth_samples + 0.02f) > z.squeeze();
    mask = mask & depth_mask;
    whole_mask &= ~mask;
  }

  auto face_mask = ~(whole_mask.index({faces}).all(1));
  auto valid_face_idx = face_mask.nonzero().squeeze();
  // auto valid_vertices_idx = faces.index({valid_face_idx}).view(-1);
  // valid_vertices_idx = std::get<0>(torch::unique_dim(valid_vertices_idx, 0));
  // vertices = vertices.index({valid_vertices_idx});
  faces = faces.index({valid_face_idx});
  // vertice_attrs = vertice_attrs.index({valid_vertices_idx});
  return {vertices, faces, vertice_attrs};
}
} // namespace mesher