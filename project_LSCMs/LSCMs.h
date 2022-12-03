#pragma once

#define M_PI_2     1.57079632679489661923   // pi/2
#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>
#include <Eigen/Geometry>

using namespace Eigen;

namespace inf574_project
{
	class LeastSquaresConformalMaps
	{
	private:
		int build_xiyi_method_flag;
		std::function<void(void)> solver_callback;
		std::function<void(void)> compute_pinned_uv_callback;

		const double zero_threshold = 1e-6;
		// data from model, vertices, facets
		Eigen::MatrixXd V;
		Eigen::MatrixXi F;

		Eigen::MatrixXd uv; // result storing the uv coordinates

		// refer paper: "Least Squares Conformal Maps for Automatic Texture Atlas Generation"
		// A matrix, b_sparse(matrix)
		Eigen::SparseMatrix<double> A, b_sparse;
		// X vector 
		Eigen::VectorXd X;

		// U vector of pinned point: a block matrix which has u1p-real part, u2p-imaginary part
		Eigen::VectorXd Up;

		int vertices_len; // the length of vertices: n points
		const int pinned_len = 2; // length of pinned point in vector
		int free_len; // length of free point in vector(free point means they can move freely.
		int pinned_split_index; // "pinned_len" is 2, then I select last two points in V as pinned points.

		Eigen::VectorXi boundary; // index of vertex which is in boundary
		int pinned_p1, pinned_p2; // store the index of pinned points, for now only 2 pinned points.

		void swap_points(Eigen::MatrixXd& points, int index1, int index2);

		void compute_pinned_uv();

		void assign_pinned_uv();

		void build_pinned_point();

		// return the vertex index in current V
		int get_current_index(int refer_index);

		Vector3d get_point(int refer_index);

		void build_A_b();

		void build_uv();

		void lscg_method();

		void sim_LDLT_method();

		void sim_LDLT_improved_method();

		// before calling it, make sure set up all inputs
		void solve();

		void build_xiyi_method1(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, double& x1, double& y1, double& x2, double& y2, double& x3, double& y3, double& area);
		void build_xiyi_method2(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, const Eigen::Vector3d& p3, double& x1, double& y1, double& x2, double& y2, double& x3, double& y3, double& area);

	public:

		void init(Eigen::MatrixXd& v, Eigen::MatrixXi& f);

		void parameterization();

		Eigen::MatrixXd& get_uv();
	};
};
