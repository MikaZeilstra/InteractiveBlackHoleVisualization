#pragma once
#include <cuda_runtime.h>

__global__ void camUpdate(const float alpha, const float* camParam, float* camParam2, float* cam);

__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* thphi, const float2* grid, const float2* grid_2,
	const int GM, const int GN, int* gapsave, int gridlvl,
	const float2* bhBorder_1, const float2* bhBorder_2, const float2 bh_center, const int angleNum, const float alpha);


__global__ void map_disk_vert(float2* disk_summary, float3* disk_incident_summary, float2* disk_mapped_vertexes, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments, const float2 bh_center, const int GM, const int GN, const int M, const int N,
	float pixelWidth, float screenDepthR, float screenDistance, float IOD, bool useIntegrationDistance, float* camera);

__device__ float3 is_point_in_quad(const float2* vertices1, float2* vertices2, float2& point);

__global__ void disk_pixInterpolationVR(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* disk_thphi, float3* disk_incident, const float2* disk_grid, const float3* disk_incident_grid,
	float2* disk_summary, float2* disk_summary_2, float3* disk_incident_summary, float3* disk_incident_summary_2, float2* disk_mapped_vertexes, float2* disk_mapped_vertexes_2, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments, const int GM, const int GN, int* gapsave, int gridlvl,
	const float2 bh_center, const int angleNum, float alpha);

__global__ void disk_pixInterpolation(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* disk_thphi, float3* disk_incident, const float2* disk_grid, const float3* disk_incident_grid,
	float2* disk_summary, float2* disk_summary_2, float3* disk_incident_summary, float3* disk_incident_summary_2, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments, const int GM, const int GN, int* gapsave, int gridlvl,
	const float2 bh_center, const int angleNum, float alpha);


template <class T, bool CheckPi> __device__ T interpolate_summary_angle(T* disk_summary, float segment_frac, float2 index_edges);
template <class T, bool CheckPi> __device__ T interpolate_summary(T* disk_summary_angle_1, T* disk_summary_angle_2, float2 index_edges_1, float2 index_edges_2, float angle_alpha, float segment_frac, float segment_frac2);
template <class T, bool CheckPi> __device__ T interpolate_vector_linearly(T* values, float alpha);


template <class T, bool B> __device__ T  interpolatePix(const float theta, const float phi, const int M, const int N, const int gridlvl,
	const T* grid, const int GM, const int GN, int* gapsave, const int i, const int j);
template <class T, bool CheckPi> __device__ T interpolateGridCoord(const int GM, const int GN, T* grid, float2 grid_coord);
template  __device__ float2 interpolateGridCoord<float2, true>(const int GM, const int GN, float2* grid, float2 grid_coord);
template  __device__ float3 interpolateGridCoord<float3, false>(const int GM, const int GN, float3* grid, float2 grid_coord);

/// <summary>
/// Interpolates the corners of a projected pixel on the celestial sky to find the position
/// of a star in the (normal, unprojected) pixel in the output image.
/// </summary>
/// <param name="t0 - t4">The theta values of the projected pixel.</param>
/// <param name="p0 - p4">The phi values of the projected pixel.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ void interpolate(float t0, float t1, float t2, float t3, float p0, float p1, float p2, float p3,
	float& start, float& starp, int sgn, int i, int j);

template <class T, bool CheckPi> __device__ T interpolateNeirestNeighbour(float percDown, float percRight, T* cornersCel);

template <class T, bool CheckPi> __device__ T interpolateLinear( float percDown, float percRight, T* cornersCel);

template <class T> __device__ T hermite(float aValue, T const& aX0, T const& aX1, T const& aX2, T const& aX3,
	float aTension, float aBias);

template <class T, bool CheckPi> __device__ T findPoint(const int i, const int j, const int GM, const int GN, 
	const int offver, const int offhor, const int gap, const T* grid, int count, T& r_check);

template <class T, bool CheckPi> __device__ T interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	 T* cornersCel, const T* grid, int count, T& r_check);

template <class T, bool CheckPi> __device__ T interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, float perc_down, float perc_right,
	T* cornersCel, const T* grid);