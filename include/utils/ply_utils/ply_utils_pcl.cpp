#include "ply_utils_pcl.h"

#include "tinyply.h"
#define TINYPLY_IMPLEMENTATION
#include <fstream>

namespace ply_utils {

void read_ply_file(const std::string &filepath,
                   pcl::PointCloud<pcl::PointXYZ> &_points) {

  std::unique_ptr<std::istream> file_stream =
      std::make_unique<std::ifstream>(filepath, std::ios::binary);

  if (!file_stream || file_stream->fail())
    throw std::runtime_error("file_stream failed to open " + filepath);

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

  _points.clear();
  _points.resize(vertices->count);

  if (vertices->t == tinyply::Type::FLOAT32) {
    std::vector<Eigen::Vector3f> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());

#pragma omp parallel for
    for (size_t i = 0; i < vertices->count; i++) {
      _points[i].x = verts[i].x();
      _points[i].y = verts[i].y();
      _points[i].z = verts[i].z();
    }
  }
  if (vertices->t == tinyply::Type::FLOAT64) {
    std::vector<Eigen::Vector3d> verts(vertices->count);
    std::memcpy(verts.data(), vertices->buffer.get(),
                vertices->buffer.size_bytes());

#pragma omp parallel for
    for (size_t i = 0; i < vertices->count; i++) {
      _points[i].x = verts[i].x();
      _points[i].y = verts[i].y();
      _points[i].z = verts[i].z();
    }
  }
}

} // namespace ply_utils