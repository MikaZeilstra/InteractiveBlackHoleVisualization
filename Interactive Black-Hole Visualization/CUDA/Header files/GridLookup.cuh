#pragma once
#include "intellisense_cuda_intrinsics.cuh"

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>

__device__ bool piCheck(volatile float* p, float factor);
/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
template <class T, bool CheckPi> __device__  void  piCheckTot(T* tp, float factor, int size);


// Set values for projected pixel corners & update phi values in case of 2pi crossing.
__device__ void retrievePixelCorners(const float2* thphi, float* t, float* p, int& ind, const int M, bool& picheck, float offset);

// __device__ void wrapToPi(float& thetaW, float& phiW);

__device__ int2 hash1(int2 key, int ow);

__device__ int2 hash0(int2 key, int hw);

__device__ float3 hashLookup(int2 key, const float3* hashTable, const int2* hashPosTag, const int2* offsetTable, const int2* tableSize, const int g);


template <class T> __device__ void findBlock(const float theta, const float phi, const int g, const T* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level);
template __device__ void findBlock(const float theta, const float phi, const int g, const float2* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level);
template __device__ void findBlock(const float theta, const float phi, const int g, const float3* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level);
template __device__ void findBlock(const float theta, const float phi, const int g, const float4* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level);