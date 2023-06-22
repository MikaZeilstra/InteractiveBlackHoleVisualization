#include "./../Header files/AccretionDiskColorComputation.cuh"
#include "./../Header files/Constants.cuh"
#include "./../Header files/Metric.cuh"
#include "./../Header files/Temperature_color_lookup.cuh"
#include "./../Header files/vector_operations.cuh"
#include "./../Header files/GridInterpolation.cuh"
#include "./../Header files/GridLookup.cuh"
#include "../../C++/Header files/IntegrationDefines.h"
#include "../../CUDA/Header files/ColorComputation.cuh"

#include <stdio.h>

#include "device_launch_parameters.h"


#define MAX_R_NEW_DISK_SEGMENT 0.5
#define MIN_R_CHANGE_SEGMENT 0.4
#define MAX_DISTANCE_JUMP_SEGMENT 5

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

__global__ void addAccretionDisk(const float2* thphi, const float3* disk_incident, uchar4* out, double*temperature_table,const float temperature_table_step_size, const int temperature_table_size, const unsigned char* bh, const int M, const int N,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;
	// Only compute if pixel is not black hole and i j is in image
	float4 color = { 0.f, 0.f, 0.f };



	if (i < N && j < M) {
		bool r_check = false;
		float2 corners[4] = {thphi[ind] ,thphi[ind + 1] , thphi[ind + M1] ,thphi[ind + M1 + 1]};
		

		if (diskMask[ijc] == 4) {
			float3 disk_incidents[4] = {disk_incident[ind] ,disk_incident[ind + 1] , disk_incident[ind + M1] ,disk_incident[ind + M1 + 1]};


			float2 avg_thp = { 0,0 };
			float3 avg_incident = { 0,0,0 };
			for (int k = 0; k < 4; k++) {
					avg_thp = avg_thp + corners[k];
					avg_incident = avg_incident + disk_incidents[k];
			}
			avg_thp = 0.25 * avg_thp;
			avg_incident = 0.25 * avg_incident;

			double temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, fmaxf(avg_thp.x / 2,MIN_STABLE_ORBIT/2));
			double max_temp = lookUpTemperature(temperature_table, temperature_table_step_size, temperature_table_size, 4.8);


			float grav_redshift = metric::calculate_gravitational_redshift<float>(avg_thp.x, avg_thp.x * avg_thp.x);

			float3 norm_incident = rsqrt(vector_ops::sq_norm(avg_incident)) * avg_incident;
			float orbit_speed = metric::calcSpeed(avg_thp.x, (float)PI1_2);
			float doppler_redshift = (1 + orbit_speed * vector_ops::dot(norm_incident, { 0,0,1 }))/ sqrt(1 - (orbit_speed * orbit_speed));

			

			float redshift = doppler_redshift * grav_redshift;
			redshift = 1;
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
			//if (lensingOn) P *= frac;

 			//P *= intensity_factor;

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

__global__ void addAccretionDiskTexture(const float2* thphi, const int M, const unsigned char* bh, uchar4* out, float4* summed_texture, float  maxAccretionRadius, int tex_width, int tex_height,
	const float* camParam, const float* solidangle, float2* viewthing, bool lensingOn, const unsigned char* diskMask) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;
	

	float4 color = { 0.f, 0.f, 0.f,0.f };
	
	if (diskMask[ijc] == 4) {
		//Get the coordinates of the disk
		float2 corners[4] = {
			thphi[ind],
			thphi[ind + 1],
			thphi[ind + M1],
			thphi[ind + M1 + 1]
		};


		int2 tex_coord[4] = {};
		//Calculate texture coordinates of the pixel corners
		for (int k = 0; k < 4; k++) {
			tex_coord[k] = {
				((int)(((corners[k].y / PI2) * (tex_width - 1)) + tex_width) % tex_width),
				(int)(fmaxf(fminf((corners[k].x - MIN_STABLE_ORBIT) / (maxAccretionRadius - MIN_STABLE_ORBIT), 1.0f), 0.0f) * (tex_height - 1))
			};
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
				fminf(out_color.x,1.0f), 
				fminf(out_color.y,1.0f),
				fminf(out_color.z,1.0f),
				fmaxf(fminf(color.w,1.0f),0.0f)
		};

		float4 prev_color = (1.0f / 255.0f) * out[ijc];

		out_color = (1 - out_color.w) * prev_color + (out_color.w) * out_color;
		out_color.w = 1;

 		out[ijc] = {(unsigned char)(out_color.x * 255),(unsigned char)(out_color.y * 255),(unsigned char)(out_color.z * 255), 255};
	}
}

/// <summary>
/// Creates a disk summary from the disk grid, summarizing the shape and structure of the disk in an easy to intepolate way
/// </summary>
/// <param name="GM">Grid size in horizontal direction</param>
/// <param name="GN">Grid size in vertical direction</param>
/// <param name="disk_grid">disk grid</param>
/// <param name="disk_summary">Summary for each angle the first max_disk_segments are the starting and ending distances from the center for each segment, the next max_disk_segments entries are the min indexes for the sample values per segment, lastly the remaining n_samples entries are the actual samples </param>
/// <param name="bhBorder">The black hole border with the center of the black hole filled</param>
/// <param name="max_r">maximum accretion disk radius</param>
/// <param name="n_angles">Number of angles to sample</param>
/// <param name="n_samples">Number of samples per angle</param>
/// <param name="max_disk_segments">Maximum number of segments to keep track of</param>
/// <returns></returns>
__global__ void CreateDiskSummary(const int GM, const int GN, float2* disk_grid, float3* disk_incident_grid, float2* disk_summary, float3* disk_incident_summary,  float2* bhBorder, float max_r, int n_angles, int n_samples, int max_disk_segments) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;


	if (i < n_angles) {
		//First zero the sections since it might retain some from previous frame.
		for (int x = 0; x < 2 * (1 + max_disk_segments) ;x++) {
			disk_summary[x + ((n_samples + (2 * (1 + max_disk_segments))) * i)] = { 0,0 };
		}
		
		float total_distance = 0;

		int disk_found = -1;
		bool between_disk = true;

		bool switched_directly = false;
		bool ray_length_jumped = false;
		float last_ray_length_disk = -1;


		float angle = PI2 / (1.f * n_angles) * 1.f * i;
		float2 mov_dir = { -sinf(angle) , cosf(angle) };
		float2 bh_pt = bhBorder[0];
		float2 pt = bh_pt;
		int2 gridpt = { int(pt.x), int(pt.y) };

		float2 gridB = disk_grid[gridpt.x * GM + gridpt.y];
		float2 gridA = disk_grid[gridpt.x * GM + gridpt.y];
		float ray_length_A = vector_ops::sq_norm(disk_incident_grid[gridpt.x * GM + gridpt.y]);
		float ray_length_B = vector_ops::sq_norm(disk_incident_grid[gridpt.x * GM + gridpt.y]);

		while (!(disk_found==max_disk_segments)) {

			//If the next point is nan we exited the disk
			if (gridB.x > 0 && isnan(gridA.x) && !between_disk) {

				//Calculate distance and save it in the high distance spot for the disk
				float dist = sqrtf(vector_ops::sq_norm(pt - bh_pt - mov_dir));
				disk_summary[disk_found + ((n_samples+ 2* (1 + max_disk_segments)) * i)].y = dist;

				//Now between disk
				between_disk = true;

				last_ray_length_disk = ray_length_B;
				//Update total distance
				total_distance += disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].y - disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x;

			}
			//If we are inbetween disk and the next point is on disk we note the distance
			//We also dont start a new segment if r is to close to the border. The disk is convcave and although small there is a chance it will clip the same segment twice messing up the ordering
			//This only happens near the edges where r is very large
			else if (between_disk && gridA.x > 0 && gridA.x < MAX_R_NEW_DISK_SEGMENT * max_r) {
				//We found another part of the disk
				disk_found++;

				//Calculate distance and save it in the low distance spot for the disk
				float dist = sqrtf(vector_ops::sq_norm(pt - bh_pt));
				disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x = dist;

				if (ray_length_A / last_ray_length_disk > MAX_DISTANCE_JUMP_SEGMENT) {
					ray_length_jumped = true;
				}

				//No longer between the disks
				between_disk = false;
				

			}
			//If we are on the disk but the change in r is too large we probably changed surface.
			else if (gridA.x > 0 && gridB.x > 0 && ((ray_length_A / ray_length_B) < (1 / MAX_DISTANCE_JUMP_SEGMENT) || (ray_length_A / ray_length_B) > MAX_DISTANCE_JUMP_SEGMENT) && !between_disk) {
				//Calculate distance and save it in the high distance spot for the disk
				if (ray_length_A / ray_length_B > MAX_DISTANCE_JUMP_SEGMENT) {
					ray_length_jumped = true;
				}

				float dist = sqrtf(vector_ops::sq_norm(pt - bh_pt - mov_dir));
				disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].y = dist;

				//Update total distance
				total_distance += disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].y - disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x;

				//Update disk count
				disk_found++;

				//Calculate distance and save it in the low distance spot for the disk
				disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments))* i)].x = dist;

			}

			//Walk over the grid while saving last step
			gridB = gridA;
			ray_length_B = ray_length_A;
			pt = pt + mov_dir;
			gridpt = { int(roundf(pt.x)), int(roundf(pt.y)) };
			ray_length_A = vector_ops::sq_norm(disk_incident_grid[gridpt.x * GM + gridpt.y]);

			//If either coord is oob break
			if (gridpt.x > GN|| gridpt.x < 0 || gridpt.y > GM || gridpt.y < 0) {
				break;
			}


			gridA = disk_grid[gridpt.x * GM + gridpt.y];

		}

		if (i == 5) {
			i;
		}

		// We trace a ray outwards over the image to detect disk segments, this means the most inner segment is inside, however the most inner segment varies per angle since it might be ocluded by the disk or the black hole.
		// We know that the outer segments can only oclude inner segments so if we store the segments inside-out it means that indexes ar guaranteed to correspond to the same disk-segment, if an inner segment is occluded it will be zero. 
		//We need to switch everything untill half rounded down of the outer disk segment index.		
		int half = (disk_found / 2);

		float2 temp;
		for (int y = 0; y <= half; y++) {
			//Swap inner and outer segments
			temp = disk_summary[y + ((n_samples + 2 * (1 + max_disk_segments)) * i)];
			disk_summary[y + ((n_samples + 2 * (1 + max_disk_segments)) * i)] = disk_summary[(disk_found - y) + ((n_samples + 2 * (1 + max_disk_segments)) * i)];
 			disk_summary[(disk_found - y) + ((n_samples + 2 * (1 + max_disk_segments)) * i)] = temp;

		}

		

		//We also need to set the 0 values if the disk segement was not found to the minimum of the lower disk segment such that the interpolation does not go into the black hole but into the other disk segment
		float2 value = { disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x, disk_summary[disk_found + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x };
		for (int y = disk_found + 1; y < max_disk_segments; y++) {
			disk_summary[y + ((n_samples + 2 * (1 + max_disk_segments)) * i)] = value;
		}

		//Now we need to sample each disk section based on the size;
		int current_sample_count = 0;


		
		for (int y = 0; y <= disk_found;y++) {
			

			//Find the number of samples for this segment
			float2 disk_edges = disk_summary[y + ((n_samples + 2 * (1 + max_disk_segments)) * i)];
			float size_fraction = (disk_edges.y - disk_edges.x) / total_distance;
			
			size_fraction = fminf(1, size_fraction);

			int n_segment_samples = size_fraction * n_samples;
			
		

			//Intialize the grid_point and how much to move for each sanple
			float2 grid_pt = bh_pt + disk_edges.x * mov_dir;
			float2 grid_mvmnt = ((disk_edges.y - disk_edges.x) / n_segment_samples) * mov_dir;

			int2 int_grid_pt = { (int)grid_pt.x, (int)grid_pt.y };

			//Save the starting index of these segments
			disk_summary[(1 + max_disk_segments) + y + ((n_samples + 2 * (1 + max_disk_segments)) * i)].x = current_sample_count;
					
			for (int z = 0; z < n_segment_samples; z++){
				int_grid_pt = { (int)grid_pt.x, (int)grid_pt.y };

				//Find the of the sample by interpolating the disk to the requested theta-phi values
				
				disk_summary[2 * (1 + max_disk_segments) + current_sample_count + ((n_samples + 2 * (1 + max_disk_segments)) * i)] =
					interpolateGridCoord<float2, true>(GM, GN, disk_grid, grid_pt);
					
				//Find the incident angle as well
				
				disk_incident_summary[current_sample_count + (n_samples * i)] = 
					interpolateGridCoord<float3, false>(GM, GN, disk_incident_grid, grid_pt);

				//Update grid point and sample counter
				grid_pt = grid_pt + grid_mvmnt;
				current_sample_count++;
			}

			//Save the last index of these segments
			disk_summary[(1 + max_disk_segments) + y + ((n_samples + 2 * (1 + max_disk_segments)) * i)].y = current_sample_count - 1;

		}

		if (!(ray_length_jumped)) {
			for (int x = 0; x < max_disk_segments; x++) {
				disk_summary[(max_disk_segments - x) + ((n_samples + 2 * (1 + max_disk_segments)) * i)] = disk_summary[(max_disk_segments - x - 1) + ((n_samples + 2 * (1 + max_disk_segments)) * i)];
				disk_summary[(max_disk_segments + 1) + (max_disk_segments - x) + ((n_samples + 2 * (1 + max_disk_segments)) * i)] = disk_summary[(max_disk_segments + 1) + (max_disk_segments - x - 1) + ((n_samples + 2 * (1 + max_disk_segments)) * i)];

			}

			float outer_edge_value = disk_summary[1 + ((n_samples + 2 * (1 + max_disk_segments)) * i)].y;
			disk_summary[((n_samples + 2 * (1 + max_disk_segments)) * i)] = { outer_edge_value ,outer_edge_value };
		}

	}
}

__global__ void makeDiskCheck(const float2* thphi, unsigned char* disk, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	int ind = i * M1 + j;

	if (i < N && j < M) {
		bool r_check = false;
		float2 corners[4] = { thphi[ind] ,thphi[ind + 1] , thphi[ind + M1] ,thphi[ind + M1 + 1] };
		disk[ijc] = !isnan(corners[0].x) + !isnan(corners[1].x) + !isnan(corners[2].x) + !isnan(corners[3].x);
	}
}