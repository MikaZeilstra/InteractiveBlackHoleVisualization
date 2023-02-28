#pragma once
#include <cuda_runtime.h>

__global__ void camUpdate(const float alpha, const int g, const float* camParam, float* cam);

__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const int Gr, float3* thphi, const float3* grid,
	const int GM, const int GN, const float hor, const float ver, int* gapsave, int gridlvl,
	const float2* bhBorder, const int angleNum, const float alpha);

__device__ float3 interpolatePix(const float theta, const float phi, const int M, const int N, const int g, const int gridlvl,
	const float3* grid, const int GM, const int GN, int* gapsave, const int i, const int j);

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

__device__ float3 interpolateNeirestNeighbour(float percDown, float percRight, float3* cornersCel);

__device__ float3 interpolateLinear(int i, int j, float percDown, float percRight, float3* cornersCel);

__device__ float3 hermite(float aValue, float3 const& aX0, float3 const& aX1, float3 const& aX2, float3 const& aX3,
	float aTension, float aBias);

__device__ float3 findPoint(const int i, const int j, const int GM, const int GN, const int g,
	const int offver, const int offhor, const int gap, const float3* grid, int count);


/// <summary>
/// Uses hermite interpolation to find the celestial sky coordinates for the give camera coordinates
/// </summary>
/// <param name="i">i coordinate of camera coordinate</param>
/// <param name="j">j coordinate of camera coordinate</param>
/// <param name="gap">The gapsize at the non-interpolated celesetion sky grid</param>
/// <param name="GM">Gridsize in j direction (horizontal)</param>
/// <param name="GN">Gridsize in i direction (vertical)</param>
/// <param name="percDown">decimal part of requested coordinate in i direction</param>
/// <param name="percRight">decimal part of requested coordinate in j direction</param>
/// <param name="g"></param>
/// <param name="cornersCel">array of size 12 with first 4 points filled with (i,j) to (i+1,j+1)</param>
/// <param name="grid">grid containing ray traced coordinates</param>
/// <param name="count"></param>
/// <param name="correct_pi">wheter the first 4 points needed to be corrected for phi = 2PI crossing </param>
/// <returns></returns>
__device__ float3 interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float3* cornersCel, const float3* grid, int count, bool no_phi_cross);

__device__ float3 interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, const float thetaCam, const float phiCam, const int g,
	float3* cornersCel, float* cornersCam, const float3* grid);