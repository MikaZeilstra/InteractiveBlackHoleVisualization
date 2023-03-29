#pragma once
#include <cuda_runtime.h>

__global__ void camUpdate(const float alpha, const int g, const float* camParam, float* cam);

__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const int Gr, float4* thphi, const float4* grid,
	const int GM, const int GN, const float hor, const float ver, int* gapsave, int gridlvl,
	const float2* bhBorder, const int angleNum, const float alpha);

__device__ float4 interpolatePix(const float theta, const float phi, const int M, const int N, const int g, const int gridlvl,
	const float4* grid, const int GM, const int GN, int* gapsave, const int i, const int j);

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

__device__ float4 interpolateNeirestNeighbour(float percDown, float percRight, float4* cornersCel);

__device__ float4 interpolateLinear(int i, int j, float percDown, float percRight, float4* cornersCel);

__device__ float4 hermite(float aValue, float4 const& aX0, float4 const& aX1, float4 const& aX2, float4 const& aX3,
	float aTension, float aBias);

__device__ float4 findPoint(const int i, const int j, const int GM, const int GN, const int g,
	const int offver, const int offhor, const int gap, const float4* grid, int count, float4& r_check);

__device__ float4 interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	const int g, float4* cornersCel, const float4* grid, int count, float4& r_check);

__device__ float4 interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, const float thetaCam, const float phiCam, const int g,
	float4* cornersCel, float* cornersCam, const float4* grid);