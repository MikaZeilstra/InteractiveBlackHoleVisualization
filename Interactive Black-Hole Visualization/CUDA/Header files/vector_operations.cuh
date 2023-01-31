#pragma once
#include <cuda_runtime.h>


inline __host__ __device__  float3 operator+ (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__  float3 operator- (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * a.x, b * a.y, b * a.z);
}

namespace vector_ops {
	inline __host__ __device__  float dot(float3 a, float3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
}