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

inline __host__ __device__  float2 operator- (int2 a, float2 b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

inline __host__ __device__ float2 operator*(float b, float2 a)
{
	return make_float2(b * (float)a.x, b * (float)a.y);
}


inline __host__ __device__  float3 operator+ (float3 a, float3 b) {
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}


inline __host__ __device__  double3 operator+ (double3 a, float3 b) {
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__  float3 operator- (float3 a, float3 b) {
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__  double3 operator- (double3 a, double3 b) {
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__  double3 operator- (double3 a, float3 b) {
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__  double3 operator/ (double3 a, float3 b) {
	return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__  double3 operator* (double3 a, float3 b) {
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__  float3 operator* (float3 a, float3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__  float3 operator* (float3 a, int3 b) {
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__  double3 operator* (double3 a, int3 b) {
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ double3 operator*(float b, double3 a)
{
	return make_double3(b * (float)a.x, b * (float)a.y, b * (float)a.z);
}

inline __host__ __device__ float3 operator*(float b, float3 a)
{
	return make_float3(b * (float) a.x, b * (float) a.y, b * (float) a.z);
}

inline __host__ __device__ float4 operator*(float b, uchar4 a)
{
	return make_float4(b * (float)a.x, b * (float)a.y, b * (float)a.z,b * (float)a.w);
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

	inline __host__ __device__  float dot(double3 a, double3 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}

	inline __host__ __device__  float dot(float4 a, float4 b) {
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	inline __host__ __device__ float3 cross(float3 a, float3 b) {
		return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
	}

	inline __host__ __device__ float sq_norm(float3 a) {
		return dot(a, a);
	}

	inline __host__ __device__ float sq_norm(double3 a) {
		return dot(a, a);
	}

	/// <summary>
	/// Performs the square norm operation but replaces nans in the argument with 0
	/// </summary>
	/// <param name="a">argument to find square norm on</param>
	/// <returns>square_norm</returns>
	inline __host__ __device__ float sq_norm_no_nan(double3 a) {
		if (isnan(a.x)) {
			a.x = 0;
		}
		if (isnan(a.y)) {
			a.y = 0;
		}
		if (isnan(a.z)) {
			a.z = 0;
		}

		return dot(a, a);
	}

	inline __host__ __device__ float sq_norm(float2 a) {
		return dot(a, a);
	}
}