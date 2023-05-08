#pragma once
#include <cuda_runtime.h>

inline __host__ __device__  float4 operator/ (uchar4 a, float b) {
	return make_float4(a.x / b, a.y /b, a.z /b , a.w / b);
}

inline __host__ __device__  float2 operator+ (float2 a, float2 b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__  float2 operator- (float2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * (float)a.x, b * (float)a.y);
}

inline __host__ __device__  float3 operator+ (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__  float3 operator- (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * (float) a.x, b * (float) a.y, b * (float) a.z);
}

inline __host__ __device__ float3 operator*(float b, uchar3 a)
{
	return make_float3(b * (float) a.x, b * (float) a.y, b * (float) a.z);
}


inline __host__ __device__ float4 operator*(float b, float4 a)
{
	return make_float4(b * a.x, b * a.y, b * a.z, b* a.w);
}

inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
	return make_float4(a.x-  b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

inline __host__ __device__ float4 operator+(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}


namespace vector_ops {
	inline __host__ __device__  float dot(float2 a, float2 b) {
		return a.x * b.x + a.y * b.y;
	}

	inline __host__ __device__  float dot(float3 a, float3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	inline __host__ __device__  float dot(float4 a, float4 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	inline __host__ __device__ float3 cross(float3 a, float3 b) {
		return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}
}