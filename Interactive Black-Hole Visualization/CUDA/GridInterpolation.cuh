#pragma once
#include "intellisense_cuda_intrinsics.cuh"
//#include "Constants.cuh"
#include "GridLookup.cuh"

__global__ void camUpdate(const float alpha, const int g, const float* camParam, float* cam);

__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const int Gr, float2* thphi, const float2* grid,
	const int GM, const int GN, const float hor, const float ver, int* gapsave, int gridlvl,
	const float2* bhBorder, const int angleNum, const float alpha);

__device__ float2 interpolatePix(const float theta, const float phi, const int M, const int N, const int g, const int gridlvl,
	const float2* grid, const int GM, const int GN, int* gapsave, const int i, const int j);

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


__device__ float2 intersection(const float ax, const float ay, const float bx, const float by, const float cx, const float cy, const float dx, const float dy);

__device__ float2 interpolateLinear(int i, int j, float percDown, float percRight, float2* cornersCel);

__device__ float2 hermite(float aValue, float2 const& aX0, float2 const& aX1, float2 const& aX2, float2 const& aX3,
	float aTension, float aBias);

__device__ float2 findPoint(const int i, const int j, const int GM, const int GN, const int g,
	const int offver, const int offhor, const int gap, const float2* grid, int count);

__device__ float2 interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float2* cornersCel, const float2* grid, int count);

__device__ float2 interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, const float thetaCam, const float phiCam, const int g,
	float2* cornersCel, float* cornersCam, const float2* grid);