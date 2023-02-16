#include "./../Header files/AccretionDiskColorComputation.cuh"
#include "./../Header files/Constants.cuh"
#include "./../Header files/Metric.cuh"
#include "./../Header files/Temperature_color_lookup.cuh"
#include "./../Header files/vector_operations.cuh"

#include "device_launch_parameters.h"


/// <summary>
/// Calculates the actual temperature of the disk at a given radius r in schwarschild radii and actual Mass M and accretion rate Ma.
/// Uses the formula descibed in thesis
/// </summary>
/// <param name="M"></param>
/// <param name="Ma"></param>
/// <param name="r"></param>
/// <returns></returns>
__device__ double getRealTemperature(const float& M, const float& Ma, const float& r) {
	return 4.2939e+9 * pow((Ma * (4.0 * sqrt(r) + 2.4495 * 
		log(
			((-SQRT6 + 2 * SQRT3) * (2 * sqrt(r) + SQRT6)) / 
			((SQRT6 + 2 * SQRT3) * (2 * sqrt(r) - SQRT6))
		) - 6.9282)) /
		(M * M * pow(r, 2.5) * (2.0 * r - 3.0)), 0.25);
}

//
__device__ double lookUpTemperature(double* table, const float step_size, const int size, const float r) {
	if (r < 3 || r >= (size * step_size + 3)) {
		return 0;
	}

	float r_low = floor(r-3);
	float r_high = ceil(r - 3);
	float mix = (r-3 - r_low);

	return (1-mix) * table[(int) (r_low/step_size)] + mix * table[(int) (r_high / step_size)];
}

__global__ void createTemperatureTable(const int size,double* table, const float step_size, float M, float Ma) {
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < size) {
		table[id] = getRealTemperature(M, Ma, 3 + step_size * id);
	}
}

__global__ void addAccretionDisk(const float3* thphi, uchar4* out, double*temperature_table,const float temperature_table_step_size, const int temperature_table_size, const unsigned char* bh, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float4 color = { 0.f, 0.f, 0.f, 0.f };
	int ind = i * M1 + j;
	// Only compute if pixel is not black hole and i j is in image

	if (i < N && j < M) {
		if (bh[ijc] == 0 && thphi[ind].z < INFINITY_CHECK) {
			if (thphi[ind].z > 3) {
				double temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, thphi[ind].z / 2);

				float grav_redshift = metric::calculate_gravitational_redshift<float>(thphi[ind].z, thphi[ind].z * thphi[ind].z, 1, cos(thphi[ind].x) * cos(thphi[ind].x), sin(thphi[ind].x) * sin(thphi[ind].x));
				float doppler_redshift = thphi[ind].x;

				float redshift = doppler_redshift * grav_redshift;

				//Apply redshift and clip temperature to [100,29000] outside this range barely any change 
				double observerd_temp = temp * redshift;
				float3 temperature_sRGB;
				if (observerd_temp < 10000) {
					float mix = (observerd_temp / 100) - floor(observerd_temp / 100);
					temperature_sRGB = (1-mix) * temperature_LUT[(int)(observerd_temp / 100)] + mix * temperature_LUT[(int)(observerd_temp / 100) + 1];
					//temperature_sRGB = temperature_LUT[(int)(observerd_temp / 100)];
				}
				else {
					temperature_sRGB = temperature_LUT[(int)(((observerd_temp - 10000) / 1000) + 99)];
				}


				//Out image in BGR format while table is RGB
				//out[ijc] = { (unsigned char)(temperature_sRGB.z*255),(unsigned char)(temperature_sRGB.y * 255),(unsigned char)(temperature_sRGB.x * 255),255 };
				out[ijc] = { 255,255,0,255 };
			}
			else {
				//out[ijc] = { 0,0,0,255 };
			}

			
		}
	}
}