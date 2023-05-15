#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "intellisense_cuda_intrinsics.cuh"
#include "Constants.cuh"
#include "ColorComputation.cuh"
#include "GridInterpolation.cuh"

__global__ void makeGradField(const float2* thphi, const int M, const int N, float2* grad);

__global__ void addDiffraction(float3* starLight, const int M, const int N, const uchar3* diffraction, const int filtersize);

__global__ void clearArrays(int* stnums, int2* stCache, const int frame, const int trailnum, const int starSize);

__global__ void sumStarLight(float3* starLight, float3* trail, float3* out, int step, int M, int N, int filterW);

__global__ void distortStarMap(float3* starLight, const float2* thphi, const unsigned char* bh, const float* stars, const int* tree,
								const int starSize, const float* camParam, const float* magnitude, const int treeLevel,
								const int M, const int N, const int step, float offset, int* search, int searchNr, int2* stCache,
								int* stnums, float3* trail, int trailnum, float2* grad, const int framenumber, const float2* viewthing, 
								bool redshiftOn, bool lensingOn, const float* area);

__global__ void addDiffraction(float3* starLight, const int M, const int N, const uchar3* diffraction, const int filtersize);

__device__ void searchTree(const int* tree, const float* thphiPixMin, const float* thphiPixMax, const int treeLevel,
							int* searchNrs, int startNr, int& pos, int picheck);

__device__ void addTrails(const int starsToCheck, const int starSize, const int framenumber, int* stnums, float3* trail,
							const float2* grad, int2* stCache, const int q, const int i, const int j, const int M, const int trailnum,
							const float part, const float frac, float3 hsp);


/// <summary>
/// Checks if the cross product between two vectors a and b is positive.
/// </summary>
/// <param name="t_a, p_a">Theta and phi of the a vector.</param>
/// <param name="t_b, p_b">Theta of the b vector.</param>
/// <param name="starTheta, starPhi">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ bool checkCrossProduct(float t_a, float t_b, float p_a, float p_b,
	float starTheta, float starPhi, int sgn);

/// <summary>
/// Returns if a (star) location lies within the boundaries of the provided polygon.
/// </summary>
/// <param name="t, p">The theta and phi values of the polygon corners.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ bool starInPolygon(const float* t, const float* p, float start, float starp, int sgn);

//__device__ int counter;

