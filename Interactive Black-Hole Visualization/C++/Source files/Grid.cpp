#include "../Header files/Grid.h"

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <vector>
#include <fstream>
#include "../Header files/Camera.h"
#include "../Header files/BlackHole.h"
#include "../Header files/Code.h"
#include "../Header files/IntegrationDefines.h"
#include <chrono>
#include <numeric>
#include <algorithm>

#include "../../CUDA/Header files/Metric.cuh"
#include "../../CUDA/Header files/ImageDistorterCaller.cuh"
#include "../../CUDA/Header files/Constants.cuh"
#include "../../CUDA/Header files/Vector_operations.cuh"

#define PRECCELEST 0.015
#define DISK_PRECCELEST_RELAXATION 3
#define ERROR 0.001//1e-6

#define MIN_GPU_INTEGRATION 50
#define BLACK_HOLE_MAX_DIAGNAL 1e-10





bool Grid::check2PIcross(const std::vector<float2>& spl, float factor) {
	for (int i = 0; i < spl.size(); i++) {
		if (spl[i]_phi > PI2 * (1. - 1. / factor))
			return true;
	}
	return false;
};

bool Grid::correct2PIcross(std::vector<float2>& spl, float factor) {
		bool check = false;
		for (int i = 0; i < spl.size(); i++) {
			if (spl[i]_phi < PI2 * (1. / factor)) {
				spl[i]_phi += PI2;
				check = true;
			}
		}
		return check;
};

#pragma region post processing

/// <summary>
/// Returns if a location lies within the boundaries of the provided polygon.
/// </summary>
/// <param name="point">The point (theta, phi) to evaluate.</param>
/// <param name="thphivals">The theta-phi coordinates of the polygon corners.</param>
/// <param name="sgn">The winding order of the polygon (+ for CW, - for CCW).</param>
/// <returns></returns>
bool Grid::pointInPolygon(cv::Point2d& point, std::vector<cv::Point2d>& thphivals, int sgn) {
	for (int q = 0; q < 4; q++) {
		cv::Point2d vecLine = sgn * (thphivals[q] - thphivals[(q + 1) % 4]);
		cv::Point2d vecPoint = sgn ? (point - thphivals[(q + 1) % 4]) : (point - thphivals[q]);
		if (vecLine.cross(vecPoint) < 0) {
			return false;
		}
	}
	return true;
}

/// <summary>
/// Fixes the t-vertices in the grid.
/// </summary>
/// <param name="block">The block to check and fix.</param>
void Grid::fixTvertices(uint64_t ij, int level) {
	if (level == MAXLEVEL) return;
	uint32_t i = i_32;
	uint32_t j = j_32;
	if (isnan(grid_vector[i * M + j]_phi)) return;

	int gap = pow(2, MAXLEVEL - level);

	uint32_t k = i + gap;
	uint32_t l = (j + gap) % M;

	//If the change in radius is small enough between the vertices they are one the same surface fix the possible T vertices

		checkAdjacentBlock(ij, k_j, level, 1, gap);


		checkAdjacentBlock(ij, i_l, level, 0, gap);


		checkAdjacentBlock(i_l, k_l, level, 1, gap);


		checkAdjacentBlock(k_j, k_l, level, 0, gap);
}
	
/// <summary>
/// Recursively checks the edge of a block for adjacent smaller blocks causing t-vertices.
/// Adjusts the value of smaller block vertices positioned on the edge to be halfway
/// inbetween the values at the edges of the larger block.
/// </summary>
/// <param name="ij">The key for one of the corners of the block edge.</param>
/// <param name="ij2">The key for the other corner of the block edge.</param>
/// <param name="level">The level of the block.</param>
/// <param name="udlr">1=down, 0=right</param>
/// <param name="lr">1=right</param>
/// <param name="gap">The gap at the current level.</param>
void Grid::checkAdjacentBlock(uint64_t ij, uint64_t ij2, int level, int udlr, int gap) {
	uint32_t i = i_32 + udlr * gap / 2;
	uint32_t j = j_32 + (1 - udlr) * gap / 2;

	uint32_t i2 = (ij2 >> 32);
	uint32_t j2 = ij;

	if (grid_vector[i_32 * M + j_32].x != -2)
		return;
	else {
		uint32_t jprev = (j_32 - (1 - udlr) * gap + M) % M;
		uint32_t jnext = (j_32 + (1 - udlr) * 2 * gap) % M;
		uint32_t iprev = i_32 - udlr * gap;
		uint32_t inext = i_32 + 2 * udlr * gap;

		bool half = false;

		if (i_32 == 0) {
			jprev = (j_32 + M / 2) % M;
			iprev = gap;
		}
		else if (inext > N - 1) {
			inext = i_32;
			jnext = (j_32 + M / 2) % M;
		}
		uint64_t ijprev = (uint64_t)iprev << 32 | jprev;
		uint64_t ijnext = (uint64_t)inext << 32 | jnext;

		bool succes = false;
		if ((grid_vector[iprev * M + jprev].x != -2) && (grid_vector[inext * M + jnext].x != -2)) {
			std::vector<float2> check = { grid_vector[iprev * M + jprev], grid_vector[i_32 * M + j_32], grid_vector[i2 * M + j2], grid_vector[inext * M + jnext] };
			if (!isnan(grid_vector[iprev * M + jprev].x) && !(isnan(grid_vector[inext * M + jnext].x))
				//&& abs(CamToCel[ijprev].z- CamToCel[ij].z) < R_CHANGE_THRESHOLD
				//&& abs(CamToCel[ijnext].z - CamToCel[ij2].z) < R_CHANGE_THRESHOLD
				) {
				succes = true;
				if (half) check[3].x = PI - check[3].x;
				if (check2PIcross(check, 5.)) correct2PIcross(check, 5.);
				grid_vector[i * M + j] = hermite(0.5, check[0], check[1], check[2], check[3], 0., 0.);
			}
		}
		if (!succes) {
			std::vector<float2> check = { grid_vector[i_32 * M + j_32], grid_vector[i2 * M + j2] };
			if (check2PIcross(check, 5.)) correct2PIcross(check, 5.);
			grid_vector[i * M + j] = 1. / 2. * (check[1] + check[0]);
		
		}
		if (level + 1 == MAXLEVEL) return;
		checkAdjacentBlock(ij, i_j, level + 1, udlr, gap / 2);
		checkAdjacentBlock(i_j, ij2, level + 1, udlr, gap / 2);
	}
}

float2 const Grid::hermite(double aValue, float2 const& aX0, float2 const& aX1, float2 const& aX2, float2 const& aX3, double aTension, double aBias) {
	/* Source:
	* http://paulbourke.net/miscellaneous/interpolation/
	*/

	double const v = aValue;
	double const v2 = v * v;
	double const v3 = v * v2;

	double const aa = (double(1) + aBias) * (double(1) - aTension) / double(2);
	double const bb = (double(1) - aBias) * (double(1) - aTension) / double(2);

	float2 const m0 = aa * (aX1 - aX0) + bb * (aX2 - aX1);
	float2 const m1 = aa * (aX2 - aX1) + bb * (aX3 - aX2);

	double const u0 = double(2) * v3 - double(3) * v2 + double(1);
	double const u1 = v3 - double(2) * v2 + v;
	double const u2 = v3 - v2;
	double const u3 = double(-2) * v3 + double(3) * v2;

	return u0 * aX1 + u1 * m0 + u2 * m1 + u3 * aX2;
}




#pragma endregion

/** -------------------------------- RAY TRACING -------------------------------- **/

/// <summary>
/// Prints the provided level of the grid.
/// </summary>
/// <param name="level">The level.</param>
void Grid::printGridCam(int level) {
	std::cout.precision(2);
	std::cout << std::endl;

	int gap = (int)pow(2, MAXLEVEL - level);
	for (uint32_t i = 0; i < N; i += gap) {
		for (uint32_t j = 0; j < M; j += gap) {
			double val = grid_vector[i * M + j]_theta;
			if (val > 1e-10)
				std::cout << std::setw(4) << val / PI;
			else
				std::cout << std::setw(4) << 0.0;
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	for (uint32_t i = 0; i < N; i += gap) {
		for (uint32_t j = 0; j < M; j += gap) {
			double val = grid_vector[i * M + j]_phi;
			if (val > 1e-10)
				std::cout << std::setw(4) << val / PI;
			else
				std::cout << std::setw(4) << 0.0;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << std::endl;
	for (uint32_t i = 0; i < N; i += gap) {
		for (uint32_t j = 0; j < M; j += gap) {
			double val = steps[i * M + j];
			std::cout << std::setw(4) << val;
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	int sum = 0;
	int sumnotnull = 0;
	int countnotnull = 0;
	std::ofstream myfile;
	myfile.open("steps.txt");
	for (int i = 0; i < N * M; i++) {
		sum += steps[i];
		if (steps[i] > 0) {
			sumnotnull += steps[i];
			countnotnull++;
		}
		myfile << steps[i] << "\n";
	}
	myfile.close();
	std::cout << "steeeeps" << sum << std::endl;
	std::cout << "steeeepsnotnull" << sumnotnull << std::endl;

	std::cout << "ave" << (float)sum / (float)(M * (N + 1)) << std::endl;
	std::cout << "avenotnull" << (float)sumnotnull / (float)(countnotnull) << std::endl;

	//for (uint32_t i = 0; i < N; i += gap) {
	//	for (uint32_t j = 0; j < M; j += gap) {
	//		int val = CamToAD[i_j];
	//		cout << setw(4) << val;
	//	}
	//	cout << endl;
	//}
	//cout << endl;

	std::cout.precision(10);
}

/// <summary>
/// Configures the basis of the grid, then starts the adaptive raytracing of the whole grid.
/// </summary>
void Grid::raytrace() {
	int gap = (int)pow(2, MAXLEVEL - STARTLVL);
	int s = 2;

	std::vector<uint64_t> ijstart(s);

	ijstart[0] = 0;
	ijstart[1] = (uint64_t)(N - 1) << 32;

	if (print) std::cout << "Computing Level " << STARTLVL << "..." << std::endl;
	integrateCameraCoordinates(ijstart);

	for (uint32_t j = 0; j < M; j += gap) {
		uint32_t i, l, k;
		i = l = k = 0;
		grid_vector[i * M + j] = grid_vector[k * M + l];
		steps[i * M + j] = steps[0];
		checkblocks.insert(i_j);

		i = k = N - 1;
		grid_vector[i * M + j] = grid_vector[k * M + l];
		steps[i * M + j] = steps[0];


	}

	integrateFirst(gap);
	adaptiveBlockIntegration(STARTLVL);
}

/// <summary>
/// Integrates the first blocks.
/// </summary>
/// <param name="gap">The gap at the current trace level.</param>
void Grid::integrateFirst(const int gap) {
	std::vector<uint64_t> toIntIJ;

	for (uint32_t i = gap; i < N - 1; i += gap) {
		for (uint32_t j = 0; j < M; j += gap) {
			toIntIJ.push_back(i_j);
			if (i == N - 1);// && !equafactor);
			else if (MAXLEVEL == STARTLVL) blockLevels[i_j] = STARTLVL;
			else checkblocks.insert(i_j);
		}
	}
	integrateCameraCoordinates(toIntIJ);

}

/// <summary>
/// Fills the grid map with the just computed raytraced values.
/// </summary>
/// <param name="ijvals">The original keys for which rays where traced.</param>
/// <param name="s">The size of the vectors.</param>
/// <param name="thetavals">The computed theta values (celestial sky).</param>
/// <param name="phivals">The computed phi values (celestial sky).</param>
void Grid::fillGridCam(const std::vector<uint64_t>& ijvals, const size_t s, std::vector<double>& thetavals,
	std::vector<double>& phivals, std::vector<double>& disk_r, std::vector<double>& disk_phis, float3* disk_incidents, std::vector<int>& step) {
	for (int k = 0; k < s; k++) {
		uint64_t ij = ijvals[k];

		grid_vector[i_32 * M + j_32] = make_float2(thetavals[k], phivals[k]);
		disk_grid_vector[i_32 * M + j_32] = make_float2(disk_r[k], disk_phis[k]);
		disk_incident_vector[i_32 * M + j_32] = disk_incidents[k];
		
		steps[i_32 * M + j_32] = step[k];
	}
}

template <typename T, typename Compare>
std::vector<std::size_t> Grid::sort_permutation(
	const std::vector<T>& vec, const std::vector<T>& vec1,
	Compare& compare) {
	std::vector<std::size_t> p(vec.size());
	std::iota(p.begin(), p.end(), 0);
	std::sort(p.begin(), p.end(),
		[&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j], vec1[i], vec1[j]); });
	return p;
}
template <typename T>
std::vector<T> Grid::apply_permutation(
	const std::vector<T>& vec,
	const std::vector<std::size_t>& p) {
	std::vector<T> sorted_vec(vec.size());
	std::transform(p.begin(), p.end(), sorted_vec.begin(),
		[&](std::size_t i) { return vec[i]; });
	return sorted_vec;
}


/// <summary>
/// Calls the integrationWrapper.
/// </summary>
/// <param name="ijvec">The ijvec.</param>
void Grid::integrateCameraCoordinates(std::vector<uint64_t>& ijvec) {
	size_t s = ijvec.size();
	std::vector<double> theta(s), phi(s), disk_rs(s), disk_phis(s);
	//std::vector<float3> disk_incidents(s, make_float3(0,0,0));
	float3* disk_incidents;
	disk_incidents = reinterpret_cast<float3*>(malloc(s * sizeof(float3)));


	std::vector<int> step(s);
	for (int q = 0; q < s; q++) {
		uint64_t ij = ijvec[q];
		int i = i_32;
		int j = j_32;
		j = j % M;
		
	
		theta[q] = (double)i_32 / (N - 1) * PI;
		phi[q] = (double)j_32 / M * PI2;
		
	}

	

	auto start_time = std::chrono::high_resolution_clock::now();
	integration_wrapper(theta, phi, disk_incidents, disk_rs,disk_phis, s, step);
	fillGridCam(ijvec, s, theta, phi,  disk_rs, disk_phis,disk_incidents, step);
	auto end_time = std::chrono::high_resolution_clock::now();

	free(disk_incidents);
}

/// <summary>
/// Returns if a block needs to be refined.
/// </summary>
/// <param name="i">The i position of the block.</param>
/// <param name="j">The j position of the block.</param>
/// <param name="gap">The current block gap.</param>
/// <param name="level">The current block level.</param>
bool Grid::refineCheck(const uint32_t i, const uint32_t j, const int gap, const int level) {
	int k = i + gap;
	int l = (j + gap) % M;

	int k_low = std::max((int) i - gap, 0);
	int k_high = std::min(k + gap, N-1);
	int l_low = ((j - gap) + M) % M;
	int l_high = (l + gap) % M;

	//If the block level is still lower than the minimum required number return 
	if (level < param->gridMinLevel) return true;

	//Check if the points are well alligned
	float2 topLeft = grid_vector[i * M + j];
	float2 topRight = grid_vector[k * M + j];
	float2 bottomLeft = grid_vector[i * M + l];
	float2 bottomRight = grid_vector[k * M + l];

	float2 disk_topLeft = disk_grid_vector[i * M + j];
	float2 disk_topRight = disk_grid_vector[k * M + j];
	float2 disk_bottomLeft = disk_grid_vector[i * M + l ];
	float2 disk_bottomRight = disk_grid_vector[k * M + l];

	
	bool topLeft_on_disk = !isnan(disk_topLeft.x);

	//If all r coordinates are not either on accretion disk or off the accretion disk we need to refine (Edge of accretion disk)
	if (!((topLeft_on_disk == !isnan(disk_topRight.x)) && (topLeft_on_disk == !isnan(disk_bottomLeft.x)) && (topLeft_on_disk == !isnan(disk_bottomRight.x)))) {
		return true;
	}

	//If the r coordinate on the disk is close enough to the max
	/*
	if (disk_topLeft.x > DISK_EDGE_REFINE_FRAC * param->accretionDiskMaxRadius || disk_topRight.x > DISK_EDGE_REFINE_FRAC * param->accretionDiskMaxRadius || disk_bottomLeft.x > DISK_EDGE_REFINE_FRAC * param->accretionDiskMaxRadius || disk_bottomRight.x > DISK_EDGE_REFINE_FRAC * param->accretionDiskMaxRadius) {
		return true;
	}
	*/


	//If all vertices are not either in the blackhole or out we need to refine
	// Nan indicates 1 of the vertices was part of the black hole, meaning we want better resolution unless they were all BH.
	bool topLeftNan = isnan(topLeft.x);
	if (!((topLeftNan == isnan(topRight.x)) && (topLeftNan == isnan(bottomLeft.x)) && (topLeftNan == isnan(bottomRight.x)))) {
		return true;
	}

	float diag = vector_ops::dot((topLeft - bottomRight), (topLeft - bottomRight));
	bottomRight.y += PI2;
	diag = std::min(diag, vector_ops::dot((topLeft - bottomRight), (topLeft - bottomRight)));


	float diag2 = vector_ops::dot((topRight - bottomLeft), (topRight - bottomLeft));
	bottomRight.y += PI2;
	diag2 = std::min(diag2, vector_ops::dot((topRight - bottomLeft), (topRight - bottomLeft)));

	// If the maximum diagonal is not less than required precision split the block

	if (diag > PRECCELEST || diag2 > PRECCELEST) return true;


	//If we are one the accretion disk diagonal is in phi and r coordinates otherwise it is in phi and theta coordinates
	if (topLeft_on_disk) {
		float diag = abs(disk_topLeft.y - disk_bottomRight.y);
		bottomRight.y += PI2;
		diag = std::min(diag,abs(disk_topLeft.y - disk_bottomRight.y));

		float diag2 = abs(disk_topRight.y - disk_bottomLeft.y);
		bottomRight.y += PI2;
		diag2 = std::min(diag2, (float)abs(disk_topRight.y - disk_bottomLeft.y));


		// If phi change is too large refine
		if (diag > PRECCELEST * DISK_PRECCELEST_RELAXATION || diag2 > PRECCELEST * DISK_PRECCELEST_RELAXATION) return true;


		//if we are on the disk but a neighbour point is not refine
		float2 disk_topLeft_out = disk_grid_vector[k_low * M + l_low];
		float2 disk_topRight_out = disk_grid_vector[k_low * M + l_high];
		float2 disk_bottomLeft_out = disk_grid_vector[k_high * M + l_low];
		float2 disk_bottomRight_out = disk_grid_vector[k_high * M + l_high];

		if (isnan(disk_topLeft_out.x) || isnan(disk_topRight_out.x) || isnan(disk_bottomLeft_out.x) || isnan(disk_bottomRight_out.x)) {
			return true;
		}

		return true;
	}
	else {
		//if we are not on the disk but a neighbour point is refine
		float2 disk_topLeft_out = disk_grid_vector[k_low * M + l_low];
		float2 disk_topRight_out = disk_grid_vector[k_low * M + l_high];
		float2 disk_bottomLeft_out = disk_grid_vector[k_high * M + l_low];
		float2 disk_bottomRight_out = disk_grid_vector[k_high * M + l_high];

		//We only need to check for positive since we are not on the disk and values might be -2 and Nan / not on the disk returns false
		if (disk_topLeft_out.x > 0 || disk_topRight_out.x > 0 || disk_bottomLeft_out.x > 0 || disk_bottomRight_out.x > 0) {
			return true;
		}
	}

	


	// If no refinement necessary, save level at position.
	blockLevels[i_j] = level;
	return false;

};

/// <summary>
/// Fills the toIntIJ vector with unique instances of theta-phi combinations.
/// </summary>
/// <param name="toIntIJ">The vector to store the positions in.</param>
/// <param name="i">The i key - to be translated to theta.</param>
/// <param name="j">The j key - to be translated to phi.</param>
void Grid::fillVector(std::vector<uint64_t>& toIntIJ, uint32_t i, uint32_t j) {
	if (grid_vector[i * M + j].x == -2) {
		toIntIJ.push_back(i_j);
		grid_vector[i * M + j] = float2{ -10, -10};
		disk_grid_vector[i * M + j] = float2{-10,-10};
		disk_incident_vector[i * M + j] = float3{ -10,-10,-10 };
	}
}

/// <summary>
/// Adaptively raytraces the grid.
/// </summary>
/// <param name="level">The current level.</param>
void Grid::adaptiveBlockIntegration(int level) {

	while (level < MAXLEVEL) {
		if (level < 5 && print) printGridCam(level);
		if (print) std::cout << "Computing level " << level + 1 << "..." << std::endl;

		if (checkblocks.size() == 0) return;

		std::unordered_set<uint64_t, hashing_func2> todo;
		std::vector<uint64_t> toIntIJ;

		for (auto ij : checkblocks) {

			uint32_t gap = (uint32_t)pow(2, MAXLEVEL - level);
			uint32_t i = i_32;
			uint32_t j = j_32;
			uint32_t k = i + gap / 2;
			uint32_t l = j + gap / 2;
			j = j % M;
			
			

			if (refineCheck(i, j, gap, level)) {
				fillVector(toIntIJ, k, j);
				fillVector(toIntIJ, k, l);
				fillVector(toIntIJ, i, l);
				fillVector(toIntIJ, i + gap, l);
				fillVector(toIntIJ, k, (j + gap) % M);
				todo.insert(i_j);
				todo.insert(k_j);
				todo.insert(k_l);
				todo.insert(i_l);
			}

		}
		integrateCameraCoordinates(toIntIJ);
		level++;
		checkblocks = todo;
	}

	for (auto ij : checkblocks)
		blockLevels[ij] = level;
}

/// <summary>
/// Raytraces the rays starting in camera sky from the theta, phi positions defined
/// in the provided vectors.
/// </summary>
/// <param name="theta">The theta positions.</param>
/// <param name="phi">The phi positions.</param>
/// /// <param name="disk_redshift">redshift of the disk if hit.must be at least size n </param>
/// /// <param name="disk_distances">distance to disk if hit. must be at least size n</param>
/// /// <param name="disk_r">The r positions if on the disk otherwise nan. must be at least size n</param>
/// /// <param name="disk_phi">The phi postions if on the disk if it was hit otherwise nan.  must be at least size n</param>
/// <param name="n">The size of the vectors.</param>
void Grid::integration_wrapper(std::vector<double>& theta, std::vector<double>& phi, float3* disk_incidents, std::vector<double>& disk_r, std::vector<double>& disk_phi, const int n, std::vector<int>& step) {
	double thetaS = cam->theta;
	double phiS = cam->phi;
	double rS = cam->r;
	double sp = cam->speed;

	//Update the count of traced rays and batches
	ray_count += n;
	integration_batches += 1;

	std::vector<float>paths;

	//reserve space to save paths if required
	if (param->savePaths) {
		paths = std::vector<float>(n * 3 * (MAXSTP / STEP_SAVE_INTERVAL));
	}


#pragma loop(hint_parallel(8))
#pragma loop(ivdep)
	for (int i = 0; i < n; i++) {
		//Calculate starting paramaters
		double xCam = sin(theta[i]) * cos(phi[i]);
		double yCam = sin(theta[i]) * sin(phi[i]);
		double zCam = cos(theta[i]);

		double yFido = (-yCam + sp) / (1 - sp * yCam);
		double xFido = -sqrtf(1 - sp * sp) * xCam / (1 - sp * yCam);
		double zFido = -sqrtf(1 - sp * sp) * zCam / (1 - sp * yCam);

		double k = sqrt(1 - cam->btheta * cam->btheta);
		double rFido = xFido * cam->bphi / k + cam->br * yFido + cam->br * cam->btheta / k * zFido;
		double thetaFido = cam->btheta * yFido - k * zFido;
		double phiFido = -xFido * cam->br / k + cam->bphi * yFido + cam->bphi * cam->btheta / k * zFido;

		double eF = 1. / (cam->alpha + cam->w * cam->wbar * phiFido);

		double pR = eF * cam->ro * rFido / sqrtf(cam->Delta);
		double pTheta = eF * cam->ro * thetaFido;
		double pPhi = eF * cam->wbar * phiFido;

		double b = pPhi;
		double q = pTheta * pTheta + cos(thetaS) * cos(thetaS) * (b * b / (sin(thetaS) * sin(thetaS)) - metric::asq<double>);


		//Save the starting parameters in the places were they will be used for integration, the vectors are also reused as return values
		disk_r[i] = pR;
		theta[i] = b;
		phi[i] = q;
		disk_phi[i] = pTheta;
	}


	if (n < MIN_GPU_INTEGRATION) {
	#pragma loop(hint_parallel(8))
		//CPU integration simply loop over all the required gedesics and save them
		for (int i = 0; i < n; i++) {
			metric::rkckIntegrate1<double>(rS, thetaS, phiS, &disk_r[i], &theta[i], &phi[i], &disk_phi[i], &disk_incidents[i], param->savePaths, reinterpret_cast<float3*>(&(paths.data()[i * 3 * (MAXSTP / STEP_SAVE_INTERVAL)])));
		}
		if (param->savePaths) {
			geodesics.insert(geodesics.end(), paths.begin(), paths.end());
		}
		
	}
	else {
		//GPU integration delegate GPU integration to function doing the GPU stuff
		CUDA::integrateGrid<double>(rS, thetaS, phiS, disk_r, theta, phi, disk_phi,disk_incidents);
		GPU_batches += 1;
	}	

}

#pragma endregion private


/// <summary>
/// Initializes a new instance of the <see cref="Grid"/> class.
/// </summary>
/// <param name="maxLevelPrec">The maximum level for the grid.</param>
/// <param name="startLevel">The start level for the grid.</param>
/// <param name="angle">If the camera is not on the symmetry axis.</param>
/// <param name="camera">The camera.</param>
/// <param name="bh">The black hole.</param>
Grid::Grid(const Camera* camera, const BlackHole* bh, Parameters* _param) {
	param = _param;
	
	MAXLEVEL = param->gridMaxLevel;
	STARTLVL = param->gridMinLevel;
	cam = camera;
	black = bh;
	

	N = param->grid_N;
	STARTN = (uint32_t)round(pow(2, STARTLVL) + 1);
	M = param->grid_M;
	STARTM = 2 * (STARTN - 1);
	steps = std::vector<int>(M * N);

	grid_vector = std::vector<float2>(M * N, make_float2(-2,  -2));
	disk_grid_vector = std::vector<float2>(M * N, make_float2( -2, -2));
	disk_incident_vector = std::vector<float3>(M * N, make_float3(-2, -2,-2 ));
	raytrace();
	//printGridCam(5);
	


	for (auto block : blockLevels) {
		fixTvertices(block.first, block.second);
	}

	if (param->savePaths) {
		saveGeodesics(_param);
	}
	
};

void Grid::saveGeodesics(Parameters* param) {
	cv::FileStorage fs(param->getGeodesicsResultFileName(metric::a<double>,0), cv::FileStorage::WRITE);
	fs << "paths" << geodesics;
	fs.release();
}

void Grid::saveAsGpuHash() {

}


void Grid::drawBlocks(std::string filename) {
	std::vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);

	cv::Mat gridimg(1025, 2049, CV_8UC4);//, Scalar(255));
	//Mat gridimg((*grids)[g].N, (*grids)[g].M, CV_8UC1);// , Scalar(255));
	gridimg = cv::Scalar(255, 255, 255, 0);
	for (auto block : blockLevels) {
		uint64_t ij = block.first;
		int level = block.second;
		int gap = pow(2, MAXLEVEL - level);
		//int gap = pow(2, 10 - level);
		int gap2 = pow(2, 10 - MAXLEVEL);
		uint32_t i = i_32 * gap2;
		uint32_t j = j_32 * gap2;
		uint32_t k = (i_32 + gap) * gap2;
		uint32_t l = (j_32 + gap) * gap2;
		cv::line(gridimg, cv::Point2d(j, i), cv::Point2d(j, k), cv::Scalar(255, 255, 255, 255), 1);// Scalar(255), 1); 
		cv::line(gridimg, cv::Point2d(l, i), cv::Point2d(l, k), cv::Scalar(255, 255, 255, 255), 1);
		cv::line(gridimg, cv::Point2d(j, i), cv::Point2d(l, i), cv::Scalar(255, 255, 255, 255), 1);
		cv::line(gridimg, cv::Point2d(j, k), cv::Point2d(l, k), cv::Scalar(255, 255, 255, 255), 1);
	}
	cv::imwrite(filename, gridimg, compressionParams);
}


void Grid::makeHeatMapOfIntegrationSteps(std::string filename) {
	std::vector<cv::Point3i> imagemat(2048 * 1025);
	for (int q = 0; q < 2048 * 1025; q++) {
		if (steps[q] == 0) imagemat[q] = { 255, 255, 255 };
		else if (steps[q] <= 20) imagemat[q] = { 84, 1, 68 };
		else if (steps[q] <= 30) imagemat[q] = { 136, 69, 64 };
		else if (steps[q] <= 40) imagemat[q] = { 141, 96, 52 };
		else if (steps[q] <= 50) imagemat[q] = { 142, 121, 41 };
		else if (steps[q] <= 60) imagemat[q] = { 140, 148, 32 };
		else if (steps[q] <= 70) imagemat[q] = { 132, 168, 34 };
		else if (steps[q] <= 80) imagemat[q] = { 111, 192, 70 };
		else if (steps[q] <= 90) imagemat[q] = { 84, 208, 117 };
		else if (steps[q] <= 100) imagemat[q] = { 38, 223, 189 };
		else imagemat[q] = { 33, 231, 249 };
	}

	cv::Mat img = cv::Mat(1025, 2048, CV_8UC3, (void*)&imagemat[0]);
	cv::imwrite(filename, img);
}