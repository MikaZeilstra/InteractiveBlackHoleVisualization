#pragma once

#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
//#include <cereal/archives/binary.hpp>
//#include <cereal/types/vector.hpp>
//#include <cereal/types/memory.hpp>
//#include <cereal/types/unordered_map.hpp>
#include <vector>
#include <fstream>
#include <unordered_set>
#include "Camera.h"
#include "BlackHole.h"
#include "Code.h"
#include "Parameters.h"
#include <chrono>
#include <numeric>

#define PRECCELEST 0.015
#define ERROR 0.001//1e-6
#define MIN_GPU_INTEGRATION 1000


#ifndef GRID_CLASS
#define GRID_CLASS



class Grid
{
private:
	#pragma region private
	/** ------------------------------ VARIABLES ------------------------------ **/
	/// <summary>
	/// Checks if a polygon has a high chance of crossing the 2pi border.
	/// </summary>
	/// <param name="poss">The coordinates of the polygon corners.</param>
	/// <param name="factor">The factor to check whether a point is close to the border.</param>
	/// <returns>Boolean indicating if the polygon is a likely 2pi cross candidate.</returns>
	static bool check2PIcross(const std::vector<float2>& spl, float factor);

	/// <summary>
	/// Assumes the polygon crosses 2pi and adds 2pi to every corner value of a polygon
	/// that is close (within 2pi/factor) to 0.
	/// </summary>
	/// <param name="poss">The coordinates of the polygon corners.</param>
	/// <param name="factor">The factor to check whether a point is close to the border.</param>
	static bool correct2PIcross(std::vector<float2>& spl, float factor);

	///TODO FIX SERIALISATION
	// Cereal settings for serialization
	/*
	friend class cereal::access;
	template < class Archive >
	void serialize(Archive & ar)
	{
		ar(MAXLEVEL, N, M, grid_vector);
	}
	*/
	// Camera & Blackhole
	const Camera* cam;
	const BlackHole* black;

	// Hashing functions (2 options)
	struct hashing_func {
		uint64_t operator()(const uint64_t& key) const {
			uint64_t v = key * 3935559000370003845 + 2691343689449507681;

			v ^= v >> 21;
			v ^= v << 37;
			v ^= v >> 4;

			v *= 4768777513237032717;

			v ^= v << 20;
			v ^= v >> 41;
			v ^= v << 5;

			return v;
		}
	};
	struct hashing_func2 {
		uint64_t  operator()(const uint64_t& key) const{
			uint64_t x = key;
			x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
			x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
			x = x ^ (x >> 31);
			return x;
		}
	};

	// Set of blocks to be checked for division
	std::unordered_set<uint64_t, hashing_func2> checkblocks;

	bool disk = false;

	/** ------------------------------ POST PROCESSING ------------------------------ **/

	#pragma region post processing

	/// <summary>
	/// Returns if a location lies within the boundaries of the provided polygon.
	/// </summary>
	/// <param name="point">The point (theta, phi) to evaluate.</param>
	/// <param name="thphivals">The theta-phi coordinates of the polygon corners.</param>
	/// <param name="sgn">The winding order of the polygon (+ for CW, - for CCW).</param>
	/// <returns></returns>
	bool pointInPolygon(cv::Point2d& point, std::vector<cv::Point2d>& thphivals, int sgn);

	/// <summary>
	/// Fixes the t-vertices in the grid.
	/// </summary>
	/// <param name="block">The block to check and fix.</param>
	void fixTvertices(uint64_t ij, int level);

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
	void checkAdjacentBlock(uint64_t ij, uint64_t ij2, int level, int udlr, int gap);


	float2 const hermite(double aValue, float2 const& aX0, float2 const& aX1, float2 const& aX2, float2 const& aX3, double aTension, double aBias);




	#pragma endregion

	/** -------------------------------- RAY TRACING -------------------------------- **/

	/// <summary>
	/// Prints the provided level of the grid.
	/// </summary>
	/// <param name="level">The level.</param>
	void printGridCam(int level);

	/// <summary>
	/// Configures the basis of the grid, then starts the adaptive raytracing of the whole grid.
	/// </summary>
	void raytrace();

	void integrateFirst(const int gap);

	/// <summary>
	/// Fills the grid map with the just computed raytraced values.
	/// </summary>
	/// <param name="ijvals">The original keys for which rays where traced.</param>
	/// <param name="s">The size of the vectors.</param>
	/// <param name="thetavals">The computed theta values (celestial sky).</param>
	/// <param name="phivals">The computed phi values (celestial sky).</param>
	void fillGridCam(const std::vector<uint64_t>& ijvals, const size_t s, std::vector<double>& thetavals,
		std::vector<double>& phivals, std::vector<double>& disk_r, std::vector<double>& disk_phis, float3* disk_incidents, std::vector<int>& step);

	template <typename T, typename Compare>
	std::vector<std::size_t> sort_permutation(
		const std::vector<T>& vec, const std::vector<T>& vec1,
		Compare& compare);

	template <typename T>
	std::vector<T> apply_permutation(
		const std::vector<T>& vec,
		const std::vector<std::size_t>& p);


	/// <summary>
	/// Calls the integrationWrapper.
	/// </summary>
	/// <param name="ijvec">The ijvec.</param>
	void integrateCameraCoordinates(std::vector<uint64_t>& ijvec);

	/// <summary>
	/// Returns if a block needs to be refined.
	/// </summary>
	/// <param name="i">The i position of the block.</param>
	/// <param name="j">The j position of the block.</param>
	/// <param name="gap">The current block gap.</param>
	/// <param name="level">The current block level.</param>
	bool refineCheck(const uint32_t i, const uint32_t j, const int gap, const int level);

	/// <summary>
	/// Fills the toIntIJ vector with unique instances of theta-phi combinations.
	/// </summary>
	/// <param name="toIntIJ">The vector to store the positions in.</param>
	/// <param name="i">The i key - to be translated to theta.</param>
	/// <param name="j">The j key - to be translated to phi.</param>
	void fillVector(std::vector<uint64_t>& toIntIJ, uint32_t i, uint32_t j);
	

	/// <summary>
	/// Adaptively raytraces the grid.
	/// </summary>
	/// <param name="level">The current level.</param>
	void adaptiveBlockIntegration(int level);

	/// <summary>
	/// Raytraces the rays starting in camera sky from the theta, phi positions defined
	/// in the provided vectors.
	/// </summary>
	/// <param name="theta">The theta positions.</param>
	/// <param name="phi">The phi positions.</param>
	/// <param name="n">The size of the vectors.</param>
	void integration_wrapper(std::vector<double>& theta, std::vector<double>& phi, float3* disk_incidents, std::vector<double>& disk_r, std::vector<double>& disk_phi, const int n, std::vector<int>& step);

	#pragma endregion private

public:

	/// <summary>
	/// 1 if rotation axis != camera axis, 0 otherwise
	/// </summary>
	int equafactor;

	/// <summary>
	/// N = max vertical rays, M = max horizontal rays.
	/// </summary>
	int MAXLEVEL, N, M, STARTN, STARTM, STARTLVL;

	/// <summary>
	/// Mapping from camera sky position to celestial angle.
	/// </summary>
	//std::unordered_map <uint64_t, float2, hashing_func2> CamToCel;
	//std::unordered_map <uint64_t, float2, hashing_func2> CamToDisk;
	//std::unordered_map <uint64_t, float3, hashing_func2> CamToIncident;

	std::vector<int> steps;

	/// <summary>
	/// Stores paths the light takes
	/// </summary>
	std::vector<float> geodesics;

	std::vector<float2> grid_vector;
	std::vector<float2> disk_grid_vector;
	std::vector<float3> disk_incident_vector;

	Parameters* param;


	/// <summary>
	/// Mapping from block position to level at that point.
	/// </summary>
	std::unordered_map <uint64_t, int, hashing_func2> blockLevels;

	bool print = false;

	/// <summary>
	/// Initializes an empty new instance of the <see cref="Grid"/> class.
	/// </summary>
	Grid() {};

	/// <summary>
	/// Initializes a new instance of the <see cref="Grid"/> class.
	/// </summary>
	/// <param name="maxLevelPrec">The maximum level for the grid.</param>
	/// <param name="startLevel">The start level for the grid.</param>
	/// <param name="angle">If the camera is not on the symmetry axis.</param>
	/// <param name="camera">The camera.</param>
	/// <param name="bh">The black hole.</param>
	Grid(const int maxLevelPrec, const int startLevel, const bool angle, const Camera* camera, const BlackHole* bh, Parameters& param);

	void saveAsGpuHash();


	void drawBlocks(std::string filename);


	void makeHeatMapOfIntegrationSteps(std::string filename);

	void saveGeodesics(Parameters& param);

	/// <summary>
	/// Finalizes an instance of the <see cref="Grid"/> class.
	/// </summary>
	~Grid() {};
};

#endif // !GRID_CLASS