#include "GridLookup.cuh"
#include "intellisense_cuda_intrinsics.cuh"
#include "Constants.cuh"

__global__ void findBhCenter(const int GM, const int GN, const float2* grid, float2* bhBorder);

__global__ void findBhBorders(const int GM, const int GN, const float2* grid, const int angleNum, float2* bhBorder);

__global__ void displayborders(const int angleNum, float2* bhBorder, uchar4* out, const int M);

__global__ void smoothBorder(const float2* bhBorder, float2* bhBorder2, const int angleNum);

__global__ void findBlackPixels(const float2* thphi, const int M, const int N, unsigned char* bh);