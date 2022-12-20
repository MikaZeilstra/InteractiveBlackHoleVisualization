#include "GridLookup.cuh"
#include "intellisense_cuda_intrinsics.cuh"
#include "Constants.cuh"
#include "ColorComputation.cuh"

__global__ void distortEnvironmentMap(const float3* thphi, uchar4* out, const unsigned char* bh, const int2 imsize,
	const int M, const int N, float offset, float4* sumTable, const float* camParam,
	const float* solidangle, float2* viewthing,bool redshiftOn, bool lensingOn);

__global__ void makePix(float3* starLight, uchar4* out, int M, int N);

__global__ void addStarsAndBackground(uchar4* stars, uchar4* background, uchar4* output, int M);