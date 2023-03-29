#include <cuda.h>
#include "cuda_runtime.h"

__device__ double getRealTemperature(const float& M, const float& Ma, const float& r);

__global__ void createTemperatureTable(const int size, double* table, const float step_size, float M, float Ma);
__global__ void addAccretionDisk(const float4* thphi, uchar4* out, double* temperature_table, const float temperature_table_step_size, const int temperature_table_size, const unsigned char* bh, const int M, const int N, const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask);
__global__ void addAccretionDiskTexture(const float4* thphi, const int M, const unsigned char* bh, uchar4* out, float3* summed_texture, float  maxAccretionRadius, int tex_width, int tex_height,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask);

__global__ void makeDiskCheck(const float4* thphi, unsigned char* disk,const int M, const int N);