#include "ply_utils_torch.h"
#include <fstream>

#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

namespace ply_utils {

tinyply::Type torch_type_to_ply_type(const torch::ScalarType &dtype) {
  switch (dtype) {
  case torch::kFloat32:
    return tinyply::Type::FLOAT32;
  case torch::kFloat64:
    return tinyply::Type::FLOAT64;
  case torch::kUInt8:
    return tinyply::Type::UINT8;
  case torch::kInt16:
    return tinyply::Type::INT16;
  case torch::kInt32:
    return tinyply::Type::INT32;
  default:
    std::cerr << "Unsupported dtype: " << dtype;
    throw std::runtime_error("Unsupported dtype");
  }
}

void export_to_ply(const std::filesystem::path &output_path,
                   const torch::Tensor &_xyz, const torch::Tensor &_rgb,
                   const torch::Tensor &_origin, const torch::Tensor &_dir,
                   const torch::Tensor &_depth, const torch::Tensor &_normal) {
  // Make sure input tensor is on CPU
  std::string filename = output_path;

  tinyply::PlyFile export_ply;

  if (_xyz.numel() > 0) {
    auto xyz = _xyz.to(torch::kFloat32);
    export_ply.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        ply_utils::torch_type_to_ply_type(xyz.scalar_type()), xyz.size(0),
        reinterpret_cast<uint8_t *>(xyz.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  if (_rgb.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"red", "green", "blue"},
        ply_utils::torch_type_to_ply_type(_rgb.scalar_type()), _rgb.size(0),
        reinterpret_cast<uint8_t *>(_rgb.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  if (_origin.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"ox", "oy", "oz"},
        ply_utils::torch_type_to_ply_type(_origin.scalar_type()),
        _origin.size(0), reinterpret_cast<uint8_t *>(_origin.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  if (_dir.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"dx", "dy", "dz"},
        ply_utils::torch_type_to_ply_type(_dir.scalar_type()), _dir.size(0),
        reinterpret_cast<uint8_t *>(_dir.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  if (_depth.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"depth"},
        ply_utils::torch_type_to_ply_type(_depth.scalar_type()), _depth.size(0),
        reinterpret_cast<uint8_t *>(_depth.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  if (_normal.numel() > 0) {
    export_ply.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        ply_utils::torch_type_to_ply_type(_normal.scalar_type()),
        _normal.size(0), reinterpret_cast<uint8_t *>(_normal.cpu().data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  std::cout << "\033[1m\033[34m\nSaving ply to: " << filename << "\n\033[0m";
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  // create directory

  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  export_ply.write(outstream_ascii, false);
  std::cout << "\033[1m\033[32m" << filename << " saved!\n\033[0m";
}

torch::Tensor cal_face_normal(const torch::Tensor &face_xyz) {
  // cal normal
  /// [n,3]
  auto v1 = face_xyz.select(1, 1) - face_xyz.select(1, 0);
  auto v2 = face_xyz.select(1, 2) - face_xyz.select(1, 1);
  auto face_normal = v1.cross(v2, 1);
  return face_normal / face_normal.norm(2, 1, true);
}

torch::Tensor cal_face_normal_color(const torch::Tensor &face_xyz) {
  // https://zhuanlan.zhihu.com/p/575404558
  return cal_face_normal(face_xyz) / 2.0 + 0.5;
}

void face_to_ply(const torch::Tensor &face_xyz, const std::string &filename,
                 bool vis_attribute) {

  auto face_num = face_xyz.size(0);
  auto vertex_num = face_num * 3;

  torch::Tensor vertex_index;
  vertex_index = torch::arange(0, vertex_num).to(torch::kInt);

  tinyply::PlyFile mesh_ply;
  mesh_ply.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_num,
      reinterpret_cast<uint8_t *>(face_xyz.data_ptr()), tinyply::Type::INVALID,
      0);

  if (vis_attribute) {
    auto tmp_face_normal_color = cal_face_normal_color(face_xyz);
    tmp_face_normal_color =
        tmp_face_normal_color.unsqueeze(1).expand({face_num, 3, 3});
    tmp_face_normal_color = (tmp_face_normal_color * 255).to(torch::kUInt8);
    mesh_ply.add_properties_to_element(
        "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, vertex_num,
        reinterpret_cast<uint8_t *>(tmp_face_normal_color.data_ptr()),
        tinyply::Type::INVALID, 0);
  }

  mesh_ply.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32, face_num,
      reinterpret_cast<uint8_t *>(vertex_index.data_ptr()),
      tinyply::Type::UINT8, 3);
  printf("\033[1;34mSaving mesh to: %s\n\033[0m", filename.c_str());
  // make directory if not exist
  std::string dir = filename.substr(0, filename.find_last_of('/'));
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
  }
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  mesh_ply.write(outstream_ascii, false);
}

void face_to_ply(const torch::Tensor &face_xyz, const torch::Tensor &face_attr,
                 const std::string &filename) {
  auto face_num = face_xyz.size(0);
  auto vertex_num = face_num * 3;

  torch::Tensor vertex_index = torch::arange(0, vertex_num, torch::kInt);

  tinyply::PlyFile mesh_ply;
  mesh_ply.add_properties_to_element(
      "vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, vertex_num,
      reinterpret_cast<uint8_t *>(face_xyz.data_ptr()), tinyply::Type::INVALID,
      0);

  auto face_attr_color =
      (face_attr * 255).unsqueeze(1).expand({face_num, 3, 3}).to(torch::kUInt8);
  mesh_ply.add_properties_to_element(
      "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, vertex_num,
      reinterpret_cast<uint8_t *>(face_attr_color.data_ptr()),
      tinyply::Type::INVALID, 0);

  mesh_ply.add_properties_to_element(
      "face", {"vertex_indices"}, tinyply::Type::INT32, face_num,
      reinterpret_cast<uint8_t *>(vertex_index.data_ptr()),
      tinyply::Type::UINT8, 3);
  printf("\033[1;34mSaving mesh to: %s\n\033[0m", filename.c_str());
  // make directory if not exist
  std::string dir = filename.substr(0, filename.find_last_of('/'));
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
  }
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  mesh_ply.write(outstream_ascii, false);
}

void face_indice_to_ply(const torch::Tensor &face_xyz,
                        const torch::Tensor &face_indices,
                        const std::string &filename, bool vis_attribute) {

  auto face_xyz_cpu = face_xyz.cpu().contiguous();
  auto face_indices_cpu = face_indices.to(torch::kInt32).cpu().contiguous();

  tinyply::PlyFile mesh_ply;
  mesh_ply.add_properties_to_element(
      "vertex", {"x", "y", "z"},
      ply_utils::torch_type_to_ply_type(face_xyz_cpu.scalar_type()),
      face_xyz_cpu.size(0),
      reinterpret_cast<uint8_t *>(face_xyz_cpu.data_ptr()),
      tinyply::Type::INVALID, 0);

  // if (vis_attribute) {
  //   auto tmp_face_normal_color = cal_face_normal_color(face_xyz);
  //   tmp_face_normal_color =
  //       tmp_face_normal_color.unsqueeze(1).expand({face_num, 3, 3});
  //   tmp_face_normal_color = (tmp_face_normal_color * 255).to(torch::kUInt8);
  //   mesh_ply.add_properties_to_element(
  //       "vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, vertex_num,
  //       reinterpret_cast<uint8_t *>(tmp_face_normal_color.data_ptr()),
  //       tinyply::Type::INVALID, 0);
  // }

  mesh_ply.add_properties_to_element(
      "face", {"vertex_indices"},
      ply_utils::torch_type_to_ply_type(face_indices_cpu.scalar_type()),
      face_indices_cpu.size(0),
      reinterpret_cast<uint8_t *>(face_indices_cpu.data_ptr()),
      tinyply::Type::UINT8, 3);
  printf("\033[1;34mSaving mesh to: %s\n\033[0m", filename.c_str());
  // make directory if not exist
  std::string dir = filename.substr(0, filename.find_last_of('/'));
  if (!dir.empty()) {
    std::filesystem::create_directories(dir);
  }
  std::filebuf fb_ascii;
  fb_ascii.open(filename, std::ios::out);
  std::ostream outstream_ascii(&fb_ascii);
  if (outstream_ascii.fail())
    throw std::runtime_error("failed to open " + filename);
  // Write an ASCII file
  mesh_ply.write(outstream_ascii, false);
}

torch::Tensor tinyply_floatdata_to_torch_tensor(
    const std::shared_ptr<tinyply::PlyData> &_p_plydata, int _dim) {
  if (_p_plydata->t == tinyply::Type::FLOAT32) {
    return torch::from_blob(_p_plydata->buffer.get(),
                            {(long)_p_plydata->count, _dim}, torch::kFloat32)
        .clone();
  } else if (_p_plydata->t == tinyply::Type::FLOAT64) {
    return torch::from_blob(_p_plydata->buffer.get(),
                            {(long)_p_plydata->count, _dim}, torch::kFloat64)
        .clone()
        .to(torch::kFloat32);
  } else if (_p_plydata->t == tinyply::Type::UINT8) {
    return torch::from_blob(_p_plydata->buffer.get(),
                            {(long)_p_plydata->count, _dim}, torch::kUInt8)
        .clone();
  } else {
    throw std::runtime_error("Unsupported ply data type");
  }
}

/**
 * @description:
 * @return xyz, rgb, direction, depth
 */
bool read_ply_file_to_map_tensor(
    const std::string &filepath,
    std::map<std::string, torch::Tensor> &map_ply_tensor,
    const torch::Device &_device) {
  // std::cout << "Reading ply file: " << filepath << "\n";

  map_ply_tensor.clear();

  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail()) {
    std::cout << "file_stream failed to open " + filepath << "\n";
    return false;
  }

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> xyz, rgb, origin, direction, depth;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    xyz = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    rgb = file.request_properties_from_element("vertex",
                                               {"red", "green", "blue"});
  } catch (const std::exception &e) {
    // std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    origin = file.request_properties_from_element("vertex", {"ox", "oy", "oz"});
  } catch (const std::exception &e) {
    // std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    direction =
        file.request_properties_from_element("vertex", {"dx", "dy", "dz"});
  } catch (const std::exception &e) {
    // std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    depth = file.request_properties_from_element("vertex", {"depth"});
  } catch (const std::exception &e) {
    // std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);

  if (xyz) {
    map_ply_tensor.emplace(
        "xyz", tinyply_floatdata_to_torch_tensor(xyz, 3).to(_device));
  }

  if (rgb) {
    map_ply_tensor.emplace(
        "rgb", tinyply_floatdata_to_torch_tensor(rgb, 3).to(_device));
  }

  if (origin) {
    map_ply_tensor.emplace(
        "origin", tinyply_floatdata_to_torch_tensor(origin, 3).to(_device));
  }

  if (direction) {
    map_ply_tensor.emplace(
        "direction",
        tinyply_floatdata_to_torch_tensor(direction, 3).to(_device));
  }

  if (depth) {
    map_ply_tensor.emplace(
        "depth", tinyply_floatdata_to_torch_tensor(depth, 1).to(_device));
  }

  // std::cout << "Read ply file: " << filepath << " done.\n";
  return true;
}

bool read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Device &_device) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail()) {
    std::cout << "file_stream failed to open " + filepath << "\n";
    return false;
  }

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);

  if (vertices->t == tinyply::Type::FLOAT32) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat32)
                  .clone()
                  .to(_device);
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    _points = torch::from_blob(vertices->buffer.get(),
                               {(long)vertices->count, 3}, torch::kFloat64)
                  .clone()
                  .to(_device)
                  .to(torch::kFloat32);
  }
  return true;
}

bool read_ply_file_to_tensor(const std::string &filepath,
                             torch::Tensor &_points, torch::Tensor &_colors,
                             torch::Device &_device) {
  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail()) {
    std::cout << "file_stream failed to open " + filepath << "\n";
    return false;
  }

  file_stream->seekg(0, std::ios::end);
  const float size_mb = file_stream->tellg() * float(1e-6);
  file_stream->seekg(0, std::ios::beg);

  tinyply::PlyFile file;
  file.parse_header(*file_stream);

  // Because most people have their own mesh types, tinyply treats parsed data
  // as structured/typed byte buffers. See examples below on how to marry your
  // own application-specific data structures with this one.
  std::shared_ptr<tinyply::PlyData> vertices, colors;

  // The header information can be used to programmatically extract properties
  // on elements known to exist in the header prior to reading the data. For
  // brevity of this sample, properties like vertex position are hard-coded:
  try {
    vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
  } catch (const std::exception &e) {
    std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  try {
    colors = file.request_properties_from_element("vertex",
                                                  {"red", "green", "blue"});
  } catch (const std::exception &e) {
    // std::cerr << "tinyply exception: " << e.what() << "\n";
  }

  file.read(*file_stream);
  if (vertices) {

    if (vertices->t == tinyply::Type::FLOAT32) {
      _points = torch::from_blob(vertices->buffer.get(),
                                 {(long)vertices->count, 3}, torch::kFloat32)
                    .clone()
                    .to(_device);
    }
    if (vertices->t == tinyply::Type::FLOAT64) {
      _points = torch::from_blob(vertices->buffer.get(),
                                 {(long)vertices->count, 3}, torch::kFloat64)
                    .clone()
                    .to(_device)
                    .to(torch::kFloat32);
    }
  }
  if (colors) {
    _colors = torch::from_blob(colors->buffer.get(), {(long)colors->count, 3},
                               torch::kUInt8)
                  .clone()
                  .to(_device);
  }
  return true;
}

} // namespace ply_utils