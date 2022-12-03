#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/readPLY.h>
#include <igl/writeOBJ.h>
#include <iostream>
#include <ostream>
#include <igl/opengl/glfw/Viewer.h>

#include <igl/lscm.h>

#include "LSCMs.h"

Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXd V_uv;

inf574_project::LeastSquaresConformalMaps myLSCMs;

bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{

  if (key == '1')
  {
    // Plot the 3D mesh
    viewer.data().set_mesh(V,F);
    viewer.core().align_camera_center(V,F);
  }
  else if (key == '2')
  {
    // Plot the mesh in 2D using the UV coordinates as vertex coordinates
    viewer.data().set_mesh(V_uv,F);
    viewer.core().align_camera_center(V_uv,F);
  }
  else if (key == '3')
  {
      Eigen::MatrixXd V_uv2 = myLSCMs.get_uv();
      // Plot the mesh in 2D using the UV coordinates as vertex coordinates
      viewer.data().set_mesh(V_uv2, F);
      viewer.core().align_camera_center(V_uv2, F);
  }

  viewer.data().compute_normals();

  return false;
}

void iglLSCM()
{
    // Fix two points on the boundary
    VectorXi bnd, b(2, 1);
    igl::boundary_loop(F, bnd);
    b(0) = bnd(0);
    b(1) = bnd(bnd.size() / 2);
    MatrixXd bc(2, 2);
    bc << 0, 0, 1, 0;

    // LSCM parametrization
    igl::lscm(V, F, b, bc, V_uv);

    // Scale the uv
    V_uv *= 5;
}

void myLSCM()
{
    myLSCMs.init(V, F);
    myLSCMs.parameterization();
}

int main(int argc, char *argv[])
{
  using namespace Eigen;
  using namespace std;

  // Load a mesh in OFF format
  igl::readOFF("data/beetle.off", V, F);
  //igl::readOFF("data/camelhead.off", V, F);
  //igl::readOFF("data/hexagon.off", V, F);

  // LSCM from igl
  iglLSCM();

  // LSCM from my implementation
  myLSCM();

  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data().set_mesh(V, F);
  viewer.data().set_uv(V_uv);
  viewer.callback_key_down = &key_down;

  // Disable wireframe
  viewer.data().show_lines = false;

  // Draw checkerboard texture
  viewer.data().show_texture = true;

  // Launch the viewer
  viewer.launch();
}
