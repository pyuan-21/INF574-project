#include <igl/boundary_loop.h>
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <vector>

using namespace Eigen;

namespace inf574_project
{
	class LeastSquaresConformalMaps
	{
	private:
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

		void swap_points(Eigen::MatrixXd& points, int index1, int index2)
		{
			Eigen::MatrixXd temp = points.row(index1);
			points.row(index1) = points.row(index2);
			points.row(index2) = temp;
		}

		void compute_pinned_uv()
		{
			// compute Up coording to pinned points by using bounding box
			// as paper defined: U1p, U2p has the same size, and they are "p" length vector.(It can be also considered as matrix with "p*1" size)
			Up.resize(2 * pinned_len, 1);

			// refer: https://libigl.github.io/tutorial/
			// igl coordinate, positive z direction is pointing inside the screen
			Eigen::Vector3d min = V.colwise().minCoeff(); // bounding box min point
			Eigen::Vector3d max = V.colwise().maxCoeff(); // bounding box max point
			Eigen::Vector3d range = max - min;
			// ignore the y direction, squash model into x-z plane to get uv
			Eigen::Vector3d point1_pinned(V(pinned_p1, 0), V(pinned_p1, 1), V(pinned_p1, 2));
			//std::cout << "point1_pinned: " << point1_pinned << std::endl;
			Up[0] = (point1_pinned.x() - min.x()) / range.x();
			Up[pinned_len + 0] = (point1_pinned.z() - min.z()) / range.z();

			Eigen::Vector3d point2_pinned(V(pinned_p2, 0), V(pinned_p2, 1), V(pinned_p2, 2));
			//std::cout << "point2_pinned: " << point2_pinned << std::endl;
			Up[1] = (point2_pinned.x() - min.x()) / range.x();
			Up[pinned_len + 1] = (point2_pinned.z() - min.z()) / range.z();

			//std::cout << "fisrt Up: " << Up[0] << ", " << Up[pinned_len] << std::endl;
			//std::cout << "second Up: " << Up[1] << ", " << Up[pinned_len + 1] << std::endl;
		}

		void assign_pinned_uv()
		{
			Up.resize(2 * pinned_len, 1);

			// assign pinned uv directly
			Up[0] = 0;
			Up[2] = 0.7;

			Up[1] = 0.7;
			Up[3] = 0;
		}

		void build_pinned_point()
		{
			igl::boundary_loop(F, boundary);
			// As Lecture-6-Page-16 said, pinned points matter a lot! In LSCMs paper they used two pinned point on the boundary.
			// Then the idea is to find two points, which have most distance, on the boundary.
			// Using brute-force: get boundary points, assume we have K boundary points, then On(K * K) loop to find two farest points
			pinned_p1 = V.rows() - 2; // by default
			pinned_p2 = V.rows() - 1; // by default
			double max_distance = 0; // using squaredNorm, save the sqrt() time
			for (int i = 0; i < boundary.rows(); i++)
			{
				int p1_index = boundary(i, 0);
				Vector3d p1(V(p1_index, 0), V(p1_index, 1), V(p1_index, 2));
				for (int j = 0; j < boundary.rows(); j++)
				{
					if (i == j)
						continue;
					int p2_index = boundary(j, 0);
					Vector3d p2(V(p2_index, 0), V(p2_index, 1), V(p2_index, 2));
					double cur_distance = (p1 - p2).squaredNorm();
					if (cur_distance > max_distance)
					{
						max_distance = cur_distance;
						pinned_p1 = p1_index;
						pinned_p2 = p2_index;
					}
				}
			}

			// compute uv for pinned points
			compute_pinned_uv_callback();

			
			// switch these two pinned points to the end of vertices
			// then last two points in V are pinned points.

			//std::cout << "pinned_p1: " << pinned_p1 << std::endl;
			//std::cout << "pinned_p2: " << pinned_p2 << std::endl;

			//std::cout << "------before swap------" << std::endl;
			//std::cout << V.row(pinned_p1) << std::endl;
			//std::cout << V.row(V.rows() - 2) << std::endl;

			// be careful, here we swap four points, but the refered indices in their facets(triangle) are still unchanged!!! 
			swap_points(V, pinned_p1, vertices_len - 2);
			swap_points(V, pinned_p2, vertices_len - 1);

			//std::cout << "------after swap------" << std::endl;
			//std::cout << V.row(pinned_p1) << std::endl;
			//std::cout << V.row(V.rows() - 2) << std::endl;
		}

		// return the vertex index in current V
		int get_current_index(int refer_index)
		{
			// because we switch two pinned points with last two points in V
			// but we haven't change its cooresponding facet(triangle)'s information(refered index inside it)
			// it is pinned point, its true information is at the last two points
			if (refer_index == pinned_p1)
				return vertices_len - 2;
			else if (refer_index == pinned_p2)
				return vertices_len - 1;
			else if (refer_index == vertices_len - 2)
				return pinned_p1;
			else if (refer_index == vertices_len - 1)
				return pinned_p2;
			else
				return refer_index;
		}

		Vector3d get_point(int refer_index)
		{
			// because we switch two pinned points with last two points in V
			// but we haven't change its cooresponding facet(triangle)'s information(refered index inside it)
			// it is pinned point, its true information is at the last two points
			int v_index = get_current_index(refer_index);
			return Vector3d(V(v_index, 0), V(v_index, 1), V(v_index, 2));
		}

		void build_A_b()
		{
			// Before building A, b matrix, we need to build (mij) from paper, actually by looping the facets can simply build each mij.
			// because mij only has value not 0 when j(vertex index) is inside the triangle i

			int A_block_width = vertices_len - pinned_len; // col offset
			int A_block_height = F.rows(); // row offset
			std::vector<Eigen::Triplet<double>> A_rs;

			int b_block_width = pinned_len; // col offset
			int b_block_height = F.rows(); // row offset
			std::vector<Eigen::Triplet<double>> b_sub_left_rs; // part of b matrix

			for (int i = 0; i < F.rows(); i++)
			{
				Vector3d p1 = get_point(F(i, 0));
				Vector3d p2 = get_point(F(i, 1));
				Vector3d p3 = get_point(F(i, 2));

				// [important] One thing the paper didn't explain well is how to build the 2D xi,yi in a triangle
				// they just start with assumption that we have already have xi,yi for each vertices in a triangle T
				// One way to do it is to use the "p1" as origin, and edge "p1p2" as the one basis, then its height of this edge is the second basis.
				// By using this 2D coordinate, it satisfies "two basis are orthogonal, two connected triangles are sharing one same basis. Normal of triangle is the 'z' axis" from paper.
				double x1, y1, x2, y2, x3, y3;
				x1 = 0;
				y1 = 0; // since we choose p1 as origin

				// before going ahead, we need the lengths of three edges
				Vector3d e12 = p2 - p1; // from p1 to p2
				Vector3d e13 = p3 - p1; // from p1 to p3
				double e12_len = e12.norm();
				double e13_len = e13.norm();
				//double e23_len = (p3 - p2).norm();
				Vector3d unit_e12 = e12.normalized();
				Vector3d unit_e13 = e13.normalized();

				// since we are using "p1p2" as basis, 
				x2 = e12_len;
				y2 = 0;

				// compute the angle between e12 and e13

				double cos_angle = unit_e12.dot(unit_e13);
				double sin_angle = (unit_e12.cross(unit_e13)).norm();
				x3 = e13_len * cos_angle;
				y3 = e13_len * sin_angle;

				// twice area of this triangle (from paper, they said they were using twice of area of triangle)
				double area = sin_angle * e12_len * e13_len; // it is actually " e12.cross(e13) "

				// compute the dominator of mij
				double dominator_mij = 1.0 / sqrt(area);

				double W1r, W1i, W2r, W2i, W3r, W3i; // complex number for each vertex. r~real part, i~imaginary part
				W1r = (x3 - x2) * dominator_mij;
				W1i = (y3 - y2) * dominator_mij;
				W2r = (x1 - x3) * dominator_mij;
				W2i = (y1 - y3) * dominator_mij;
				W3r = (x2 - x1) * dominator_mij;
				W3i = (y2 - y1) * dominator_mij;

				double Wr[] = { W1r,W2r,W3r };
				double Wi[] = { W1i,W2i,W3i };
				for (int k = 0; k < 3; k++)
				{
					// because we switch two pinned points with last two points in V
					// but we haven't change its cooresponding facet(triangle)'s information(refered index inside it)
					// it is pinned point, its true information is at the last two points
					int refer_index = F(i, k);
					int v_index = get_current_index(refer_index);
					if (v_index < pinned_split_index)
					{
						// free points: set to A, and A is a block matrix which can be splited into four parts.
						// top-left
						A_rs.push_back(Eigen::Triplet<double>(i, v_index, Wr[k]));
						// top-right
						A_rs.push_back(Eigen::Triplet<double>(i, v_index + A_block_width, -Wi[k]));
						// bottom-left
						A_rs.push_back(Eigen::Triplet<double>(i + A_block_height, v_index, Wi[k]));
						// bottom-right
						A_rs.push_back(Eigen::Triplet<double>(i + A_block_height, v_index + A_block_width, Wr[k]));
					}
					else
					{
						// pinned points: set to B and B is also a block matrix which can be splited into four parts.
						// be careful, I put the negative symbol into this expression
						// top-left
						b_sub_left_rs.push_back(Eigen::Triplet<double>(i, v_index - pinned_split_index, -Wr[k]));
						// top-right
						b_sub_left_rs.push_back(Eigen::Triplet<double>(i, v_index - pinned_split_index + b_block_width, Wi[k]));
						// bottom-left
						b_sub_left_rs.push_back(Eigen::Triplet<double>(i + b_block_height, v_index - pinned_split_index, -Wi[k]));
						// bottom-right
						b_sub_left_rs.push_back(Eigen::Triplet<double>(i + b_block_height, v_index - pinned_split_index + b_block_width, -Wr[k]));
					}
				}
			}

			// build A
			// using setFromTriplets() to initialize A.
			// "Block Operation" can not be applied on SparseMatrix
			// and "BlockSparseMatrix" is in the extra eigen header.(not including this project)
			A.resize(2 * F.rows(), 2 * (vertices_len - pinned_len));
			A.setFromTriplets(A_rs.begin(), A_rs.end());
			A.makeCompressed();

			// build b
			// split b matrix into two part: sub_left, Up
			Eigen::SparseMatrix<double> b_sub_left;
			b_sub_left.resize(2 * F.rows(), 2 * pinned_len);
			b_sub_left.setFromTriplets(b_sub_left_rs.begin(), b_sub_left_rs.end());
			b_sub_left.makeCompressed();

			Eigen::MatrixXd bMat = b_sub_left * Up;
			// bMat is 2n' * 1 matrix
			std::vector<Eigen::Triplet<double>> b_rs;
			for (int i = 0; i < bMat.rows(); i++)
			{
				for (int j = 0; j < bMat.cols(); j++)
				{
					double value = bMat(i, j);
					b_rs.push_back(Eigen::Triplet<double>(i, j, value));
				}
			}
			b_sparse.resize(2 * F.rows(), 1);
			b_sparse.setFromTriplets(b_rs.begin(), b_rs.end());
			b_sparse.makeCompressed();
		}

		void build_uv()
		{
			// build uv from X
			// I need to put "X" into "uv" because "X" represents only the points which can move freely
			// Pinned points are needed to be considered.

			// X is 2*(n-p) vector(It can be considered as matrix with row:2*(n-p), col:1)
			uv.resize(V.rows(), 2); // for each point, we generate (u,v) for it

			int row_offset = vertices_len - pinned_len;

			// put X into uv
			for (int i = 0; i < (vertices_len - pinned_len); i++)
			{
				uv(i, 0) = X(i, 0); // set u
				uv(i, 1) = X(row_offset + i, 0); // set v
			}
			// final two points
			for (int i = 0; i < 2; i++)
			{
				uv(vertices_len - 2 + i, 0) = Up[i];
				uv(vertices_len - 2 + i, 1) = Up[i + pinned_len];
			}
			
			// switch pinned points back to its original positions
			swap_points(uv, pinned_p1, vertices_len - 2);
			swap_points(uv, pinned_p2, vertices_len - 1);

			// switch back
			swap_points(V, pinned_p1, vertices_len - 2);
			swap_points(V, pinned_p2, vertices_len - 1);
		}

		void lscg_method()
		{
			// this method is extremely slow!!!! More than 5 mins to get the solution!
			Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;
			solver.compute(A);
			if (solver.info() != Success) {
				// decomposition failed
				std::cout << "Decomposition Failed" << std::endl;
				return;
			}
			//solver.setTolerance(zero_threshold);
			X = solver.solve(b_sparse);
			if (solver.info() != Success) {
				// solving failed
				std::cout << "solving failed" << std::endl;
				return;
			}
			std::cout << "Check solution : "
				<< (A * X - b_sparse).norm() << std::endl;
			std::cout << "N iterations : "
				<< solver.iterations() << std::endl;
		}

		void sim_LDLT_method()
		{
			// refer: https://stackoverflow.com/questions/46014719/eigen-lscg-solver-performance-issue
			// and also the lecture from INF584-lecture "Laplacian deformation least square solution"
			// ||Ax-b||^2  <---> (A^TA)x = A^Tb
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
			solver.compute(A.transpose() * A);
			if (solver.info() != Success) {
				// decomposition failed
				std::cout << "Decomposition Failed" << std::endl;
				return;
			}
			X = solver.solve(A.transpose() * b_sparse);
			if (solver.info() != Success) {
				// solving failed
				std::cout << "solving failed" << std::endl;
				return;
			}
			std::cout << "Check solution : "
				<< (A * X - b_sparse).norm() << std::endl;
		}

		void sim_LDLT_improved_method()
		{
			// refer: https://stackoverflow.com/questions/42116271/best-way-of-solving-sparse-linear-systems-in-c-gpu-possible
			Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Lower, Eigen::AMDOrdering<int>> solver;
			Eigen::SparseMatrix<double> At_A = A.transpose() * A;
			solver.analyzePattern(At_A);
			solver.compute(At_A);
			if (solver.info() != Eigen::Success)
			{
				// decomposition failed
				std::cout << "Decomposition Failed" << std::endl;
				return;
			}
			X= solver.solve(A.transpose() * b_sparse);
			if (solver.info() != Success) {
				// solving failed
				std::cout << "solving failed" << std::endl;
				return;
			}
			std::cout << "Check solution : "
				<< (A * X - b_sparse).norm() << std::endl;
		}

		// before calling it, make sure set up all inputs
		void solve()
		{
			std::cout << "-----------------------------------------------" << std::endl;
			std::cout << "Start to solve..." << std::endl;
			auto start = std::chrono::high_resolution_clock::now(); // for measuring time performance

			// solve the ||Ax-b||^2 problem using the solver.
			solver_callback();
			
			auto end = std::chrono::high_resolution_clock::now(); // for measuring time performances
			std::chrono::duration<double> elapsed_seconds = end - start;
			std::cout << "performance time: " << elapsed_seconds.count() << "s\n";
			std::cout << "-----------------------------------------------" << std::endl;
		}

	public:
		void init(Eigen::MatrixXd& v, Eigen::MatrixXi& f)
		{
			V = Eigen::MatrixXd(v); // deep copy iy, for later use in switch pinned points.
			F = f;

			vertices_len = V.rows();
			// for now I just pinned two point, as paper did.
			free_len = vertices_len - pinned_len;

			// I always put 2 pinned points at the last two vertex of V. (achieve this goal by switching them)
			pinned_split_index = vertices_len - pinned_len;

			// Allow user to change it
			//solver_callback = std::bind(&LeastSquaresConformalMaps::lscg_method, this);
			//solver_callback = std::bind(&LeastSquaresConformalMaps::sim_LDLT_method, this);
			solver_callback = std::bind(&LeastSquaresConformalMaps::sim_LDLT_improved_method, this);

			// Allow user to change it
			compute_pinned_uv_callback = std::bind(&LeastSquaresConformalMaps::compute_pinned_uv, this);
			//compute_pinned_uv_callback = std::bind(&LeastSquaresConformalMaps::assign_pinned_uv, this);
		}

		void parameterization()
		{
			build_pinned_point();
			build_A_b();
			solve();
			build_uv();
		}

		Eigen::MatrixXd& get_uv()
		{
			return uv;
		}
	};
};
