#include <cuda.h>
#include "cuda_runtime.h"

__device__ double getRealTemperature(const float& M, const float& Ma, const float& r);

__global__ void createTemperatureTable(const int size, double* table, const float step_size, float M, float Ma);
__global__ void addAccretionDisk(const float2* thphi,const float3* disk_incident, uchar4* out, double* temperature_table, const float temperature_table_step_size, const int temperature_table_size, const unsigned char* bh, const int M, const int N, const float* camParam, const float* solidangle, const float* solidangle_disk, float2* viewthing, float max_disk_r, bool lensingOn, const unsigned char* diskMask);
__global__ void addAccretionDiskTexture(const float2* thphi, const int M, const unsigned char* bh, uchar4* out, float4* summed_texture, float  maxAccretionRadius, int tex_width, int tex_height,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask);
__global__ void makeDiskCheck(const float2* thphi, unsigned char* disk,const int M, const int N);

__global__ void CreateDiskSummary(const int GM, const int GN, float2* disk_grid, float3* disk_incident_grid, float2* disk_summary, float3* disk_incident_summary, float2* bhBorder, float max_r, int n_angles, int n_samples, int max_disk_segments);