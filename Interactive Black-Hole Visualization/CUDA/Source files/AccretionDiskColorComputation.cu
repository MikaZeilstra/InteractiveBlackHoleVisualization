#include "./../Header files/AccretionDiskColorComputation.cuh"
#include "./../Header files/Constants.cuh"
#include "./../Header files/Metric.cuh"
#include "./../Header files/Temperature_color_lookup.cuh"
#include "./../Header files/vector_operations.cuh"
#include "../../C++/Header files/IntegrationDefines.h"
#include "../../CUDA/Header files/ColorComputation.cuh"

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

/// <summary>
/// Looks up temperature in provided lookup table
/// </summary>
/// <param name="table">Array containing lookup table</param>
/// <param name="step_size">Size of the steps between r values</param>
/// <param name="size">Ampount of entries in the table</param>
/// <param name="r">r in schwarschild radii (r/2 if r is in boyler-linqiust)</param>
/// <returns></returns>
__device__ double lookUpTemperature(double* table, const float step_size, const int size, const float r) {
	if (r < 3 || r >= (size * step_size + 3)) {
		return 0;
	}

	float r_low = floor(r-3);
	float r_high = ceil(r - 3);
	float mix = (r-3 - r_low);

	return (1-mix) * table[(int) (r_low/step_size)] + mix * table[(int) (r_high / step_size)];
}

/// <summary>
/// Creates a temperature look table
/// </summary>
/// <param name="size">Number of entries to generate</param>
/// <param name="table">Pointer to array to store the table</param>
/// <param name="step_size">Size of the steps between r values</param>
/// <param name="M">Mass of the black hole in solar masses</param>
/// <param name="Ma">Accretion rate of the black hole in solar masses per year</param>
/// <returns></returns>
__global__ void createTemperatureTable(const int size,double* table, const float step_size, float M, float Ma) {
	int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (id < size) {
		table[id] = getRealTemperature(M, Ma, 3 + step_size * id);
	}
}

__global__ void addAccretionDisk(const float4* thphi, uchar4* out, double*temperature_table,const float temperature_table_step_size, const int temperature_table_size, const unsigned char* bh, const int M, const int N,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;
	// Only compute if pixel is not black hole and i j is in image
	float4 color = { 0.f, 0.f, 0.f };

	

	if (i < N && j < M) {
		bool r_check = false;
		float4 corners[4] = {thphi[ind] ,thphi[ind + 1] , thphi[ind + M1] ,thphi[ind + M1 + 1]};
		

		if (diskMask[ijc] == 4) {
			
			float4 avg_thp = { 0,0,0,0 };
			int disk_count = 0;
			for (int k = 0; k < 4; k++) {
				if (!isnan(corners[k].x)) {
					avg_thp = avg_thp + corners[k];
					disk_count++;
				}
			}
			avg_thp = (1.0f / (float)disk_count) * avg_thp;

			double temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, fmaxf(avg_thp.x / 2,MIN_STABLE_ORBIT/2));
			double max_temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, 4.8);

			float grav_redshift = metric::calculate_gravitational_redshift<float>(avg_thp.x, avg_thp.x * avg_thp.x);
			float doppler_redshift = avg_thp.z;

			float redshift = doppler_redshift * grav_redshift;

			//Apply redshift and clip temperature to [100,29000] outside this range barely any change 
			double observerd_temp = temp * redshift;

				

			if (observerd_temp < TEMP_SPLIT) {
				float mix = (observerd_temp / TEMP_STEP_SMALL) - floor(observerd_temp / TEMP_STEP_SMALL);
				color = (1-mix) * temperature_LUT[(int)(observerd_temp / TEMP_STEP_SMALL)] + mix * temperature_LUT[(int)(observerd_temp / TEMP_STEP_SMALL) + 1];
				//temperature_sRGB = temperature_LUT[(int)(observerd_temp / 100)];
			}
			else if(observerd_temp < TEMP_MAX) {
				color = temperature_LUT[(int)(((observerd_temp - TEMP_SPLIT) / TEMP_STEP_LARGE) + 99)];
			}

			float max_intensity;
			if (max_temp < TEMP_SPLIT) {
				float mix = (max_temp / TEMP_STEP_SMALL) - floor(max_temp / TEMP_STEP_SMALL);
				max_intensity = ((1 - mix) * temperature_LUT[(int)(max_temp / TEMP_STEP_SMALL)] + mix * temperature_LUT[(int)(max_temp / TEMP_STEP_SMALL) + 1]).w;
				//temperature_sRGB = temperature_LUT[(int)(observerd_temp / 100)];
			}
			else if (max_intensity < TEMP_MAX) {
				max_intensity = temperature_LUT[(int)(((max_temp - TEMP_SPLIT) / TEMP_STEP_LARGE) + 99)].w;
			}

			float H, S, P;
			RGBtoHSP(color.x, color.y , color.z , H, S, P);
			float intensity_factor = fminf(color.w / max_intensity,1.f);
			float redshft = 1;
			float frac = 1;
			findLensingRedshift(M, ind, camParam, viewthing, frac, redshft, solidangle[ijc]);
			if (lensingOn) P *= frac;

 			P *= intensity_factor;

			HSPtoRGB(H, S, fminf(1.f, P), color.x, color.y, color.z);

			float4 out_color = {
				fminf(color.x, 1.0f),
				fminf(color.y, 1.0f),
				fminf(color.z, 1.0f),
				0
			};


			//Out image in BGR format while table is RGB
			out[ijc] = { (unsigned char)(out_color.z*255), (unsigned char)(out_color.y*255), (unsigned char)(out_color.x * 255),255 };
			//out[ijc] = { 255,255,0,255 };


			
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

__global__ void addAccretionDiskTexture(const float4* thphi, const int M, const unsigned char* bh, uchar4* out, float3* summed_texture, float  maxAccretionRadius, int tex_width, int tex_height,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;
	

	float3 color = { 0.f, 0.f, 0.f };
	
	if (bh[ijc] == 0 && diskMask[ijc]) {
		//Get the coordinates of the disk
		float4 corners[4] = {
			thphi[ind],
			thphi[ind + 1],
			thphi[ind + M1],
			thphi[ind + M1 + 1]
		};

		//Find a point on the disk we can set faulty points to
		int good_point_ind = -1;
		float min_r = INFINITY;
		for (int k = 0; k < 4; k++) {
			if (corners[k].x < min_r) {
				min_r = corners[k].x;
				good_point_ind = k;
			}
		}

		//Check if we are at the lower or higher edge of the disk
		bool lower_edge = corners[good_point_ind].x < (maxAccretionRadius / 2);


		int2 tex_coord[4] = {};
		//Calculate texture coordinates of the pixel corners
		for (int k = 0; k < 4; k++) {
			if (fabsf(corners[good_point_ind].x - corners[k].x) < 10) {
				tex_coord[k] = {
					((int)(((corners[k].y / PI2) * (tex_width - 1)) + tex_width) % tex_width),
					(int) (fmaxf(fminf((corners[k].x - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT), 1.0f), 0.0f) * (tex_height - 1))
				};
			}
			else {
				tex_coord[k] = {
					((int)(((corners[good_point_ind].y / PI2) * (tex_width - 1)) + tex_width) % tex_width),
					lower_edge ? 0 : (tex_width - 1)
				};
			}
		}
		
		int2 max_coord = coord_max(tex_coord);
		int2 min_coord = coord_min(tex_coord);

		//If max and min coordinates are the same add 1 pixel to the area such that the pixel area is non-zero
		if ((max_coord.x) == (min_coord.x)) {
			if (max_coord.x == (tex_width - 1)) {
				min_coord.x = 0;
			}
			else {
				max_coord.x += 1;
			}
	
		}


		

		if (max_coord.y == min_coord.y) {
			if (max_coord.y != tex_height - 1) {
				max_coord.y += 1;
			}
			else {
				min_coord.y -= 1;
			}
				
		}

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
		float4 out_color = {
			fminf(1.0f,color.x),
			fminf(1.0f,color.y),
			fminf(1.0f,color.z),
			1
		};

	

		float H, S, P;
		if (lensingOn) {
			RGBtoHSP(out_color.z, out_color.y, out_color.x , H, S, P);


			float redshft = 1;
			float frac = 1;
			findLensingRedshift(M, ind, camParam, viewthing, frac, redshft, solidangle[ijc]);
			if (lensingOn) P *= frac;
			HSPtoRGB(H, S, min(1.f, P), out_color.z, out_color.y, out_color.x);
		}


		out_color = {
				fminf(color.x,1.0f),
				fminf(color.y,1.0f),
				fminf(color.z,1.0f),
				1
		};

 		out[ijc] = {(unsigned char)(out_color.x * 255),(unsigned char)(out_color.y * 255),(unsigned char)(out_color.z * 255), 255};
	}
}

__global__ void makeDiskCheck(const float4* thphi, unsigned char* disk, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;

	if (i < N && j < M) {
		bool r_check = false;
		float4 corners[4] = { thphi[ind] ,thphi[ind + 1] , thphi[ind + M1] ,thphi[ind + M1 + 1] };
		disk[ijc] = !isnan(corners[0].x) + !isnan(corners[1].x) + !isnan(corners[2].x) + !isnan(corners[3].x);
	}
}