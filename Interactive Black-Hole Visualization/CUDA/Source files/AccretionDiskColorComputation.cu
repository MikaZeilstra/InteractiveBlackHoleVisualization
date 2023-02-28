#include "./../Header files/AccretionDiskColorComputation.cuh"
#include "./../Header files/Constants.cuh"
#include "./../Header files/Metric.cuh"
#include "./../Header files/Temperature_color_lookup.cuh"
#include "./../Header files/vector_operations.cuh"
#include "../../C++/Header files/IntegrationDefines.h"

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
			if (thphi[ind].z > MIN_STABLE_ORBIT) {
				double temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, thphi[ind].z / 2);

				float grav_redshift = metric::calculate_gravitational_redshift<float>(thphi[ind].z, thphi[ind].z * thphi[ind].z);
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
				out[ijc] = { (unsigned char)(temperature_sRGB.z*255),(unsigned char)(temperature_sRGB.y * 255),(unsigned char)(temperature_sRGB.x * 255),255 };
				//out[ijc] = { 255,255,0,255 };
			}
			else {
				//out[ijc] = { 0,0,0,255 };
			}

			
		}
	}
}

//Max to create axis alligned bb for integration approximation
__device__ int2 coord_max(int2* coords) {
	return {
		max(max(coords[0].x,coords[1].x),max(coords[2].x,coords[3].x)),
		max(max(coords[0].y,coords[1].y),max(coords[2].y,coords[3].y))
	};
}

__device__ int2 coord_min(int2* coords) {
	return {
		min(min(coords[0].x,coords[1].x),min(coords[2].x,coords[3].x)),
		min(min(coords[0].y,coords[1].y),min(coords[2].y,coords[3].y))
	};
}

__global__ void addAccretionDiskTexture(const float3* thphi, const int M, const unsigned char* bh, uchar4* out, float3* summed_texture, float  maxAccretionRadius, int tex_width, int tex_height) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;
	


	bool r_check = false;
	r_check = (thphi[ind].z < INFINITY_CHECK) || (thphi[ind + 1].z < INFINITY_CHECK) || (thphi[ind + M1].z < INFINITY_CHECK) || (thphi[ind + M1 + 1].z < INFINITY_CHECK);

	if (bh[ijc] == 0 && r_check) {
		//Calculate texture coordinates of the pixel corners
		int2 tex_coord[4] = {
			{ (thphi[ind].y / PI2) * (tex_width - 1), fmaxf(fminf((thphi[ind].z - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT),1.0f),0.0f)* (tex_height - 1)},
			{ (thphi[ind + 1].y / PI2) * (tex_width - 1),fmaxf(fminf((thphi[ind + 1].z - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT),1.0f),0.0f) * (tex_height - 1)},
			{ (thphi[ind + M1].y / PI2) * (tex_width - 1), fmaxf(fminf((thphi[ind + M1].z - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT),1.0f),0.0f) * (tex_height - 1)},
			{ (thphi[ind + M1 + 1].y / PI2) * (tex_width - 1), fmaxf(fminf((thphi[ind + M1 + 1].z - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT),1.0f),0.0f) * (tex_height - 1)}
		};

		int2 max_coord = coord_max(tex_coord);
		int2 min_coord = coord_min(tex_coord);

		//If the difference in y coordinates is more than half the height 1 ore more corners were of the disk and we need to clamp them to 0. The 
		if (abs(max_coord.y - min_coord.y) > tex_height / 2) {
			max_coord.y = min_coord.y;
			min_coord.y = 0;
		}

		//If max and min coordinates are the same add 1 pixel to the area such that the pixel area is non-zero
		if ((max_coord.x) == (min_coord.x)) {
			max_coord.x += 1;
		}

		if (max_coord.y == min_coord.y) {
			if (max_coord.y != tex_height - 1) {
				max_coord.y += 1;
			}
			else {
				min_coord.y -= 1;
			}
				
		}

			
			


		float3 color = { 0.f, 0.f, 0.f };
		int pix_sum = 0;

		//If the area of the pixel is 1 block approximate it by a square and integrate it.
		if (abs(max_coord.x - min_coord.x)  < (tex_width / 2)) {
			pix_sum = ((max_coord.x  + 1) * (max_coord.y + 1)) +
				((min_coord.x + 1) * (min_coord.y + 1)) -
				((min_coord.x + 1) * (max_coord.y + 1)) -
				((max_coord.x + 1) * (min_coord.y + 1));
			color = summed_texture[max_coord.x  * tex_height + max_coord.y] +
				summed_texture[min_coord.x  * tex_height + min_coord.y] -
				summed_texture[min_coord.x  * tex_height + max_coord.y] -
				summed_texture[max_coord.x  * tex_height + min_coord.y];
					
				
		}
		else {
			//If the area is split on the 0 and 2pi areas of phi integrate both areas
			//Correct the min and max x coord if they are too close to 1.
			if (max_coord.x == (tex_width - 1)) {
				max_coord.x -= 1;
			}
			if (min_coord.x == (tex_width - 1)) {
				min_coord.x += 1;
			}

			//Integrate lower side of phi
			pix_sum += ((min_coord.x  + 1) * (max_coord.y + 1)) +
				(1 * (min_coord.y + 1)) -
				(1 * (max_coord.y + 1)) -
				((min_coord.x + 1) * (min_coord.y + 1));
			color = color +
				summed_texture[min_coord.x  * tex_height + max_coord.y] +
				summed_texture[0 * tex_height + min_coord.y] -
				summed_texture[0 * tex_height + max_coord.y] -
				summed_texture[min_coord.x  * tex_height + min_coord.y];


			//Integrate higher side of phi
			pix_sum += (tex_width * (max_coord.y + 1)) +
				((max_coord.x + 1) * (min_coord.y + 1)) -
				((max_coord.x + 1) * (max_coord.y + 1)) -
				(tex_width * (min_coord.y + 1));
			color = color + 
				summed_texture[(tex_width-1) * tex_height + max_coord.y] +
				summed_texture[max_coord.x  * tex_height + min_coord.y] -
				summed_texture[max_coord.x  * tex_height + max_coord.y] -
				summed_texture[(tex_width - 1) * tex_height + min_coord.y];

		}

		color = (1 / ((float)pix_sum)) * color;
		color = {
			fminf(1.0f,color.x),
			fminf(1.0f,color.y),
			fminf(1.0f,color.z)
		};


 		out[ijc] = {(unsigned char)(color.x * 255),(unsigned char)(color.y * 255),(unsigned char)(color.z * 255), 255};
	}
}