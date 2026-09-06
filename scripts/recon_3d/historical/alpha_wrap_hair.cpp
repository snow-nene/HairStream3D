#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/alpha_wrap_3.h>
#include <CGAL/IO/polygon_soup_io.h>
#include <CGAL/boost/graph/IO/polygon_mesh_io.h>
#include <iostream>
#include <vector>
#include <array>

int main(int argc, char** argv) {
  if (argc != 5) { std::cerr << "input.off output.off alpha_m offset_m\n"; return 1; }
  using K = CGAL::Exact_predicates_inexact_constructions_kernel;
  std::vector<K::Point_3> points;
  std::vector<std::vector<std::size_t>> faces;
  if (!CGAL::IO::read_polygon_soup(argv[1], points, faces)) return 2;
  const double alpha = std::stod(argv[3]), offset = std::stod(argv[4]);
  if (!(alpha > 0 && offset > 0)) return 3;
  CGAL::Surface_mesh<K::Point_3> wrap;
  CGAL::alpha_wrap_3(points, faces, alpha, offset, wrap);
  std::cout << "vertices=" << wrap.number_of_vertices() << " faces=" << wrap.number_of_faces()
            << " closed=" << CGAL::is_closed(wrap) << std::endl;
  return CGAL::IO::write_polygon_mesh(argv[2], wrap, CGAL::parameters::stream_precision(17)) ? 0 : 4;
}
