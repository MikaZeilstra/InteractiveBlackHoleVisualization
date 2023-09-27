#pragma once
#include "../Header files/GridInterpolation.cuh"
#include "../Header files/GridLookup.cuh"
#include "../Header files/vector_operations.cuh"
#include "../Header files/metric.cuh"
#include "../../C++/Header files/IntegrationDefines.h"
#include "../Header files/Constants.cuh"

#include "../Header files/ColorComputation.cuh"

#include <stdio.h>

#define MIN_DISK_FRACTION_FOR_CONTRIBUTION 0.2

/// <summary>
/// Interpolated the vector while interpolating the length seperatly, the calculate area function is very sensitive to the vector length and thus needs to be taken extra care off,
/// this is actually the exact oposite for when interpolating grid coordinates where we (obviously) not want to preserve length we can thus just use the CheckPi template parameter
/// </summary>
/// <param name="values">Array where the first two values are interpolated</param>
/// <param name="alpha">The mixing factor to use</param>
/// <returns></returns>
template<>  __device__ float3 interpolate_vector_linearly<float3, false>(float3* values, float alpha) {
	//Calculate the inverse length so we can interpolate it seperately
	float inv_length[] = {
		rsqrtf(vector_ops::sq_norm(values[0])),
		rsqrtf(vector_ops::sq_norm(values[1]))
	};

	float3 normalized_values[] = {
		inv_length[0] * values[0],
		inv_length[1] * values[1]
	};

	//Interpolate vector and length seperatly
	float3  interpolated = (1 - alpha) * normalized_values[0] + alpha * normalized_values[1];
	float interpolated_length = (1 - alpha) * inv_length[0] + alpha * inv_length[1];

	//Normalize interpolated vector and then rescale to interpolated length
	return rsqrtf(vector_ops::sq_norm(interpolated)) * (1 / interpolated_length) * interpolated;
}

template<>  __device__ float2 interpolate_vector_linearly<float2, true>(float2* values, float alpha) {
	return (1 - alpha) * values[0] + alpha * values[1];
}

__global__ void camUpdate(const float alpha, const float* camParam, float* camParam2, float* cam) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < 10) cam[i] = (1.f - alpha) * camParam[i] + alpha * camParam2[i];
}


__global__ void pixInterpolation(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* thphi, const float2* grid, const float2* grid_2,
	const int GM, const int GN, int* gapsave, int gridlvl,
	const float2* bhBorder_1, const float2* bhBorder_2, float2 bh_center, const int angleNum, const float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < N1 && j < M1) {
		float theta = viewthing[i * M1 + j].x;
		float phi = fmodf(viewthing[i * M1 + j].y + PI2, PI2);
		float2 grid_point = { (theta / (PI / ((float)GN))) ,  (phi / (PI2 / ((float)GM))) };


		if (should_interpolate_grids) {
			float2 A, B;
			float2 center = bh_center;
			//The longest stretch is to the "right" of the black hole at 0PI 
			float stretchRad = sqrtf(fmaxf(vector_ops::sq_norm(bhBorder_1[0]), vector_ops::sq_norm(bhBorder_2[0]))) * 1.25;
			float centerdist = vector_ops::sq_norm(grid_point - center);
			
			if (centerdist < (stretchRad * stretchRad)) {

				float angle = atan2(center.x - grid_point.x, grid_point.y - center.y);
				angle = fmodf(angle + PI2, PI2);
				int angleSlot = (angle / PI2) * angleNum;
				int angleSlot2 = (angleSlot + 1) % angleNum;

				float angle_alpha = ((((float) angle) / PI2) * angleNum) - angleSlot;


				float2 bhBorder_g1 = (1.f - angle_alpha) * bhBorder_1[angleSlot ] + angle_alpha * bhBorder_1[angleSlot2];
				float2 bhBorder_g2 = (1.f - angle_alpha) * bhBorder_2[angleSlot] + angle_alpha * bhBorder_2[angleSlot2];


				float2 bhBorderNew = (1 - alpha) * bhBorder_g1 + alpha * bhBorder_g2;

				if (centerdist <= vector_ops::sq_norm(bhBorderNew)) {
					thphi[i * M1 + j] = { nanf(""), nanf("")};
					return;
				}

				float StoB = stretchRad -  sqrtf(vector_ops::sq_norm(bhBorderNew));
	

				float Perc = fabsf(StoB) < 1E-5 ? 0 : (sqrtf(centerdist) - sqrtf(vector_ops::sq_norm(bhBorderNew))) / StoB;

				float thetaA = (center.x + ((stretchRad * -sinf(angle) - bhBorder_g1.x) * Perc + bhBorder_g1.x)) * (PI / ((float)GN));
				float phiA = (center.y + ((stretchRad * cosf(angle) - bhBorder_g1.y) * Perc + bhBorder_g1.y)) * (PI2 / ((float)GM));
				float thetaB = (center.x + ((stretchRad * -sinf(angle) - bhBorder_g2.x) * Perc + bhBorder_g2.x)) * (PI / ((float)GN));
				float phiB = (center.y + ((stretchRad * cosf(angle) - bhBorder_g2.y) * Perc + bhBorder_g2.y)) * (PI2 / ((float)GM));

				A = interpolatePix<float2, true>(thetaA, phiA, M, N, gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix<float2, true>(thetaB, phiB, M, N, gridlvl, grid_2, GM, GN, gapsave, i, j);
			}
			else {
				A = interpolatePix<float2, true>(theta, phi, M, N,  gridlvl, grid, GM, GN, gapsave, i, j);
				B = interpolatePix<float2, true>(theta, phi, M, N,  gridlvl, grid_2, GM, GN, gapsave, i, j);

			}
			if (isnan(A.x) || isnan(B.x)) thphi[i * M1 + j] = { nanf(""),  nanf("") };
			else {

				if (A.y < .2f * PI2 && B.y > .8f * PI2) A.y += PI2;
				if (B.y < .2f * PI2 && A.y > .8f * PI2) B.y += PI2;
				thphi[i * M1 + j] = { (1.f - alpha) * A.x + alpha * B.x, fmodf((1.f - alpha) * A.y + alpha * B.y, PI2)};
			}
		}
		else {
			float2 interpolated_tp = interpolatePix<float2,true>(theta, phi, M, N, gridlvl, grid, GM, GN, gapsave, i, j);
			thphi[i * M1 + j] = interpolated_tp;
		}
	}
}



__global__ void map_disk_vert(float2* disk_summary, float3* disk_incident_summary, float2* disk_mapped_vertexes, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments,const float2 bh_center, const int GM, const int GN, const int M, const int N,
	float pixelWidth, float screenDepthR, float screenDistance, float IOD, bool useIntegrationDistance, float* camera) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if (i < n_disk_angles) {
		float angle = PI2 / (1.f * n_disk_angles) * 1.f * i;
		float2 mov_dir = { -sinf(angle) , cosf(angle) };
		float2 gridToImage = { (float)M / (float)GM, (float)N / (float)GN };
		for (int seg = 0; seg < n_disk_segments; seg++) {
			float2 grid_pos_lower = bh_center + disk_summary[seg + ((n_disk_sample + 2 * (1 + n_disk_segments)) * i)].x * mov_dir;
			float2 grid_pos_upper = bh_center + disk_summary[seg + ((n_disk_sample + 2 * (1 + n_disk_segments)) * i)].y * mov_dir;

			float2 image_pos_lower = grid_pos_lower * gridToImage;
			float2 image_pos_upper = grid_pos_upper * gridToImage;
			
			//Correct for the half size
			image_pos_lower.y = image_pos_lower.y / 2 + (M / 4);
			image_pos_upper.y = image_pos_upper.y / 2 + (M / 4);

			float2 map_lower = { 0, (M / 4) };
			float2 map_upper = { 0, (M / 4) };

			float2 index_edges = disk_summary[(1 + n_disk_segments) + seg + i * (n_disk_sample + 2 * (1 + n_disk_segments))];

			float2 dist;

			if (useIntegrationDistance) {
				//Find coordinates of lower and higher point in world space
				float3 grid_coord_lower = disk_incident_summary[(int)index_edges.x + i * n_disk_sample];
				float3 grid_coord_higher = disk_incident_summary[(int)index_edges.y + i * n_disk_sample];

				dist = {
					sqrtf(vector_ops::sq_norm(grid_coord_lower)),
					sqrtf(vector_ops::sq_norm(grid_coord_higher))
				};

			}
			else {
				//Find coordinates of lower and higher point in world space
				float2 grid_coord_lower = disk_summary[(int)index_edges.x + 2 * (1 + n_disk_segments) + i * (n_disk_sample + 2 * (1 + n_disk_segments))];
				float2 grid_coord_higher = disk_summary[(int)index_edges.y + 2 * (1 + n_disk_segments) + i * (n_disk_sample + 2 * (1 + n_disk_segments))];

				//Convert all coordinates to cartesion
				float3 cam_pos = getCartesianCoordinates({ camera[9],camera[7],camera[8] });
				float3 lower_pos = getCartesianCoordinates({ grid_coord_lower.x, PI1_2, grid_coord_lower.y });
				float3 higher_pos = getCartesianCoordinates({ grid_coord_higher.x, PI1_2, grid_coord_higher.y });

				//Find distance
				dist = {
					sqrtf(vector_ops::sq_norm(lower_pos - cam_pos)),
					sqrtf(vector_ops::sq_norm(higher_pos - cam_pos))
				};

				
			}


			//Center such that black hole is at center of screen
			dist = -1 * (dist - camera[9]);

			//Find the actual depth for the screen
			dist = screenDepthR * dist;

			//Find the pixel disparity according to lecture 4 of CSE4365 Applied Image processing
			float2 offset = 0.5 * ((IOD / pixelWidth) * (dist / (dist + screenDistance)));

			map_lower.y += offset.x;
			map_upper.y += offset.y;

		

			float2 image_pos_lower_L = image_pos_lower - map_lower;
			float2 image_pos_upper_L = image_pos_upper - map_upper;

			float2 image_pos_lower_R = image_pos_lower + map_lower;
			float2 image_pos_upper_R = image_pos_upper + map_upper;


			//Left image mapped
			disk_mapped_vertexes[0 + (seg * 4) +  (i * 4 * n_disk_segments)] = image_pos_lower_L;
			disk_mapped_vertexes[1 + (seg * 4) +  (i * 4 * n_disk_segments)] = image_pos_upper_L;

			//Right image mapped
			disk_mapped_vertexes[2 + (seg * 4) + (i * 4 * n_disk_segments)] = image_pos_lower_R;
			disk_mapped_vertexes[3 + (seg * 4) + (i * 4 * n_disk_segments)] = image_pos_upper_R;
		}
	}
}


__device__ float3 is_point_in_quad(const float2* vertices1,float2* vertices2, float2& point) {
	float2 v01 = vertices1[1] - vertices1[0], v11 = vertices2[0] - vertices1[0], v21 = point - vertices1[0];
	float d001 = vector_ops::dot(v01, v01);
	float d011 = vector_ops::dot(v01, v11);
	float d111 = vector_ops::dot(v11, v11);
	float d201 = vector_ops::dot(v21, v01);
	float d211 = vector_ops::dot(v21, v11);
	float denom1 = d001 * d111 - d011 * d011;
	float v1 = (d111 * d201 - d011 * d211) / denom1;
	float w1 = (d001 * d211 - d011 * d201) / denom1;
	float u1 = 1.0f - v1 - w1;

	float2 v02 = vertices2[1] - vertices1[1], v12 = vertices2[0] - vertices1[1], v22 = point - vertices1[1];
	float d002 = vector_ops::dot(v02, v02);
	float d012 = vector_ops::dot(v02, v12);
	float d112 = vector_ops::dot(v12, v12);
	float d202 = vector_ops::dot(v22, v02);
	float d212 = vector_ops::dot(v22, v12);
	float denom2 = d002 * d112 - d012 * d012;
	float v2 = (d112 * d202 - d012 * d212) / denom2;
	float w2 = (d002 * d212 - d012 * d202) / denom2;
	float u2 = 1.0f - v2 - w2;


	float3 ret = { 0.5,0.5,0.5 }; 

	if (v1 < 1 && v1 > 0 && w1 < 1 && w1 > 0 && u1 < 1 && u1 > 0) {
		float2 tex_coord = v1 * make_float2(1,0) + w1 * make_float2(0, 1);

		ret.x = tex_coord.x;
		ret.y = tex_coord.x;
		ret.z = tex_coord.y;
	}
	else if (v2 < 1 && v2 > 0 && w2 < 1 && w2 > 0 && u2 < 1 && u2 > 0) {
		float2 tex_coord = w2 * make_float2(0,1) + u2 * make_float2(1, 0) + v2 * make_float2(1, 1);

		ret.x = tex_coord.x;
		ret.y = tex_coord.x;
		ret.z = tex_coord.y;
	}
	else {
		ret.x = -1;
	}

	return ret;
}

__global__ void disk_pixInterpolationVR(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* disk_thphi, float3* disk_incident, const float2* disk_grid, const float3* disk_incident_grid,
	float2* disk_summary, float2* disk_summary_2, float3* disk_incident_summary, float3* disk_incident_summary_2, float2* disk_mapped_vertexes, float2* disk_mapped_vertexes_2, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments, const int GM, const int GN, int* gapsave, int gridlvl,
	const float2 bh_center, const int angleNum, float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;


	float2 pix_pos = { i,j };
	if (i < N1 && j < M1) {
		disk_thphi[i * M1 + j] = { nanf(""),0 };


		int mapped_vertex_LR_offset = (j > (M1 / 2)) * 2;

		float theta = viewthing[i * M1 + j].x;
		float phi = fmodf(viewthing[i * M1 + j].y + PI2, PI2);

		float3 alphas = {  };
		if (should_interpolate_grids) {
			
		}
		else {
			for (int angleSlot = 0; angleSlot < n_disk_angles; angleSlot++) {
				for (int segment_slot = 1; segment_slot < n_disk_segments; segment_slot++) {
					int angleSlot2 = (angleSlot + 1) % n_disk_angles;
					alphas = is_point_in_quad(
						&disk_mapped_vertexes[segment_slot * 4 + mapped_vertex_LR_offset + (angleSlot * 4 * n_disk_segments)],
						&disk_mapped_vertexes[segment_slot * 4 + mapped_vertex_LR_offset + (angleSlot2 * 4 * n_disk_segments)],
						pix_pos);
					if (alphas.x > 0) {
						

						//Get index range for this segment on grid 1 and 2
						float2 index_edges_G1_A1 = disk_summary[(1 + n_disk_segments) + segment_slot + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))];
						float2 index_edges_G1_A2 = disk_summary[(1 + n_disk_segments) + segment_slot + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))];


						//Calculate grid values

						float2 summary_val = interpolate_summary<float2, true>(&disk_summary[2 * (1 + n_disk_segments) + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))],
							&disk_summary[2 * (1 + n_disk_segments) + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))],
							index_edges_G1_A1, index_edges_G1_A2, alphas.z, alphas.x, alphas.y);

						//Interpolate grids to get final value
						disk_thphi[i * M1 + j] = summary_val;


						//Calulate the incident grid values
						//Calculate grid values
						float3 incident_val = interpolate_summary<float3, false>(&disk_incident_summary[angleSlot * n_disk_sample],
							&disk_incident_summary[angleSlot2 * n_disk_sample],
							index_edges_G1_A1, index_edges_G1_A2, alphas.z, alphas.x, alphas.y);

						disk_incident[i * M1 + j] = incident_val;

						return;
					}
				}
			}
		}


	}

};

__global__ void disk_pixInterpolation(const float2* viewthing, const int M, const int N, const bool should_interpolate_grids, float2* disk_thphi, float3* disk_incident, const float2* disk_grid, const float3* disk_incident_grid,
	float2* disk_summary, float2* disk_summary_2, float3* disk_incident_summary, float3* disk_incident_summary_2, const int n_disk_angles, const int n_disk_sample, const int n_disk_segments, const int GM, const int GN, int* gapsave, int gridlvl,
	const float2 bh_center, const int angleNum, float alpha) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N1 && j < M1) {
		disk_thphi[i * M1 + j] = { nanf(""),0 };
		
		float theta = viewthing[i * M1 + j].x;
		float phi = fmodf(viewthing[i * M1 + j].y + PI2, PI2);
		if (should_interpolate_grids) {

			//Calculate angle and angleslot from blackhole center.
			float2 center = bh_center;
			float2 grid_point = { (theta / (PI / ((float)GN))) ,  (phi / (PI2 / ((float)GM))) };


			float angle = atan2(center.x - grid_point.x, grid_point.y - center.y);
			angle = fmodf(angle + PI2, PI2);
			

			float2 grid_coordinates = { theta / (float)PI2 * GM, phi / (float)PI2 * GM };
			float centerdist = sqrt(vector_ops::sq_norm(grid_coordinates- center));

			int angleSlot = angle / PI2 * n_disk_angles;
			int angleSlot2 = (angleSlot + 1) % n_disk_angles;

			float angle_alpha = ((angle / PI2) * n_disk_angles) - angleSlot;

			//Find which band in the angleslot the disk falls in
			for (int segment_slot = 0; segment_slot < n_disk_segments + 1; segment_slot++) {
				//Interpolate within each angle

				float2 interp_edges_gr_1 = (1 - angle_alpha) * disk_summary[angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments)) + segment_slot] +
					angle_alpha * disk_summary[angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments)) + segment_slot];


				float2 interp_edges_gr_2 = (1 - angle_alpha) * disk_summary_2[angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments)) + segment_slot] +
					angle_alpha * disk_summary_2[angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments)) + segment_slot];

				//Interpolate between the grids
				float2 interp_edges = (1 - alpha) * interp_edges_gr_1 + alpha * interp_edges_gr_2;

				if (centerdist > interp_edges.x && centerdist < interp_edges.y) {
					float segment_frac = (centerdist - interp_edges.x) / (interp_edges.y - interp_edges.x);

					//Get index range for this segment on grid 1 and 2
					float2 index_edges_G1_A1 = disk_summary[(1 + n_disk_segments) + segment_slot + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))];
					float2 index_edges_G1_A2 = disk_summary[(1 + n_disk_segments) + segment_slot + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))];

					float2 index_edges_G2_A1 = disk_summary_2[(1 + n_disk_segments) + segment_slot + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))];
					float2 index_edges_G2_A2 = disk_summary_2[(1 + n_disk_segments) + segment_slot + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))];

					//If either segment is zero on 1 grid we wont use its summary
					if ((interp_edges_gr_1.y - interp_edges_gr_1.x) / (interp_edges_gr_2.y - interp_edges_gr_2.x) < MIN_DISK_FRACTION_FOR_CONTRIBUTION) {
						alpha = 1;
					}
					else if ((interp_edges_gr_1.y - interp_edges_gr_1.x) / (interp_edges_gr_2.y - interp_edges_gr_2.x) > (1/ MIN_DISK_FRACTION_FOR_CONTRIBUTION)) {
						alpha = 0;
					}

					//Calculate grid values
					float2 grid_values[] = {
						interpolate_summary<float2, true>(&disk_summary[2 * (1 + n_disk_segments) + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))],
							&disk_summary[2 * (1 + n_disk_segments) + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))],
							index_edges_G1_A1, index_edges_G1_A2, angle_alpha, segment_frac, segment_frac),
						interpolate_summary<float2, true>(&disk_summary_2[2 * (1 + n_disk_segments) + angleSlot * (n_disk_sample + 2 * (1 + n_disk_segments))],
							&disk_summary_2[2 * (1 + n_disk_segments) + angleSlot2 * (n_disk_sample + 2 * (1 + n_disk_segments))],
							index_edges_G2_A1,index_edges_G2_A2, angle_alpha, segment_frac,segment_frac)
					};

					piCheckTot<float2, true>(grid_values, PI_CHECK_FACTOR, 2);

					//Interpolate grids to get final value
					disk_thphi[i * M1 + j] = interpolate_vector_linearly<float2, true>(grid_values, alpha);
					
					

					//Calulate the incident grid values
					//Calculate grid values
					float3 incident_grid_values[] = {
						interpolate_summary<float3, false>(&disk_incident_summary[angleSlot * n_disk_sample],
							&disk_incident_summary[angleSlot2 * n_disk_sample],
							index_edges_G1_A1, index_edges_G1_A2, angle_alpha, segment_frac, segment_frac),
						interpolate_summary<float3, false>(&disk_incident_summary_2[angleSlot * n_disk_sample],
							&disk_incident_summary_2[angleSlot2 * n_disk_sample],
							index_edges_G2_A1,index_edges_G2_A2, angle_alpha, segment_frac, segment_frac)
					};

					disk_incident[i * M1 + j] = interpolate_vector_linearly<float3, false>(incident_grid_values,alpha);

					return;
				}
			}
			
			
		}
		else {
			float2 interpolated_tp = interpolatePix<float2, true>(theta, phi, M, N, gridlvl, disk_grid, GM, GN, gapsave, i, j);
			disk_thphi[i * M1 + j] = interpolated_tp;
			float3 interpolated_incident = interpolatePix<float3, false>(theta, phi, M, N, gridlvl, disk_incident_grid, GM, GN, gapsave, i, j);
			disk_incident[i * M1 + j] = interpolated_incident;
		}


	}
}




/// <summary>
/// Interpolates the disk summary at a given angle
/// </summary>
/// <param name="disk_summary">Pointer in the summary to the angle</param>
/// <param name="segment_frac">The length along the disk segment we want to know</param>
/// <param name="index_edges">The min and max index for this disk segment in the summary</param>
/// <returns>interpolated value</returns>
template <class T, bool CheckPi> __device__ T interpolate_summary_angle(T* disk_summary, float segment_frac, float2 index_edges) {
	//Calculate the index segment, segment frac needs
	float index_fl = ((index_edges.y - index_edges.x) * segment_frac) + index_edges.x;

	//floor index for lower index and subtract to get the alpha
	int lower_index = index_fl;
	float index_alpha = index_fl - lower_index;

	//Get values from summary
	T values[] = {
		disk_summary[lower_index],
		disk_summary[lower_index + 1]
	};

	//Fix 2pi crossings
	piCheckTot<T, CheckPi>(values, PI_CHECK_FACTOR, 2);


	//Interpolate angles to get grid value
	return interpolate_vector_linearly<T,CheckPi>(values,index_alpha);
}

/// <summary>
/// Calculates the value for a given grid summary at the given position
/// </summary>
/// <param name="disk_summary_angle_1">Pointer in summary to the first angle</param>
/// <param name="disk_summary_angle_2">Pointer in summary to the second angle</param>
/// <param name="index_edges_1">min and max index in the summary for the disk segment in the first angle</param>
/// <param name="index_edges_2">min and max index in the summary for the disk segment in the second angle</param>
/// <param name="angle_alpha">Mixing factor for the angles</param>
/// <param name="segment_frac">How far along the segment the point is [0,1]</param>
/// <returns>interpolated value</returns>
template <class T, bool CheckPi> __device__ T interpolate_summary(T* disk_summary_angle_1, T* disk_summary_angle_2, float2 index_edges_1, float2 index_edges_2, float angle_alpha, float segment_frac, float segment_frac_2) {

	//Calculate the values according to both adjecent angles
	T values[] = {
		interpolate_summary_angle<T, CheckPi>(disk_summary_angle_1, segment_frac,index_edges_1),
		interpolate_summary_angle<T, CheckPi>(disk_summary_angle_2, segment_frac_2,index_edges_2)
	};

	//Fix 2 pi crossings
	piCheckTot<T, CheckPi>(values, PI_CHECK_FACTOR, 2);
	
	//Interpolate angles to get grid value
	return interpolate_vector_linearly<T, CheckPi>(values, angle_alpha);
}



template <class T, bool CheckPi> __device__ T interpolatePix(const float theta, const float phi, const int M, const int N, const int gridlvl,
	const T* grid, const int GM, const int GN, int* gapsave, const int i, const int j) {
	int half = (phi < PI) ? 0 : 1;
	int a = 0;
	int b = half * GM / 2;
	int gap = GM / 2;


	findBlock(theta, phi,grid, GM, GN, a, b, gap, gridlvl);
	gapsave[i * M1 + j] = gap;

	int k = a + gap;
	int l = b + gap;

	float factor = PI2 / (1.f * GM);
	l = l % GM;
	T nul = { -1, -1};
	T cornersCel[12] = { grid[a * GM + b], grid[a * GM + l], grid[k * GM + b], grid[k * GM + l],
									nul, nul, nul, nul, nul, nul, nul, nul };

	float thetaUp = factor * a;
	float phiLeft = factor * b;
	float thetaDown = factor * k;
	float phiRight = factor * l;


	float percDown = (theta - thetaUp) / (thetaDown - thetaUp);
	float percRight = (phi - phiLeft) / (phiRight - phiLeft);

	T thphiInter = interpolateSpline<T, CheckPi>(a, b, gap, GM, GN, percDown, percRight,  cornersCel,  grid);


	return thphiInter;
}

template <class T, bool CheckPi> __device__ T interpolateGridCoord(const int GM, const int GN, T* grid, float2 grid_coord) {
	int a = grid_coord.x;
	int b = grid_coord.y;
	
	int gap = 1;
	
	//if the grid value is not positive or nan find new a and b coordinates
	while (!(!(grid[a * GM + b].x == -2)
		&& !(grid[(a + gap) * GM + ((b) % GM)].x == -2)
		&& !(grid[(a) * GM + ((b + gap) % GM)].x == -2)
		&& !(grid[(a + gap) * GM + ((b + gap) % GM)].x == -2))) {
		gap = gap * 2;
		a = a - (a % gap);
		b = b - (b % gap);
	}
	

	int k = a + gap;
	int l = b + gap;


	l = l % GM;
	T nul = { -1, -1 };
	T cornersCel[12] = { grid[a * GM + b], grid[a * GM + l], grid[k * GM + b], grid[k * GM + l],
									nul, nul, nul, nul, nul, nul, nul, nul };

	T thphiInter = interpolateSpline<T, CheckPi>(a, b, gap, GM, GN, (grid_coord.x - a) / gap, (grid_coord.y - b) / gap, cornersCel, grid);


	return thphiInter;
}


/// <summary>
/// Interpolates the corners of a projected pixel on the celestial sky to find the position
/// of a star in the (normal, unprojected) pixel in the output image.
/// </summary>
/// <param name="t0 - t4">The theta values of the projected pixel.</param>
/// <param name="p0 - p4">The phi values of the projected pixel.</param>
/// <param name="start, starp">The star theta and phi.</param>
/// <param name="sgn">The winding order of the polygon + for CW, - for CCW.</param>
/// <returns></returns>
__device__ void interpolate(float t0, float t1, float t2, float t3, float p0, float p1, float p2, float p3,
	float& start, float& starp, int sgn, int i, int j) {
	float error = 0.00001f;

	float midT = (t0 + t1 + t2 + t3) * .25f;
	float midP = (p0 + p1 + p2 + p3) * .25f;

	float starInPixY = 0.5f;
	float starInPixX = 0.5f;

	float perc = 0.5f;
#pragma unroll
	for (int q = 0; q < 10; q++) {
		if ((fabs(start - midT) < error) && (fabs(starp - midP) < error)) break;

		float half01T = (t0 + t1) * .5f;
		float half23T = (t2 + t3) * .5f;
		float half12T = (t2 + t1) * .5f;
		float half03T = (t0 + t3) * .5f;
		float half01P = (p0 + p1) * .5f;
		float half23P = (p2 + p3) * .5f;
		float half12P = (p2 + p1) * .5f;
		float half03P = (p0 + p3) * .5f;

		float line01to23T = half23T - half01T;
		float line03to12T = half12T - half03T;
		float line01to23P = half23P - half01P;
		float line03to12P = half12P - half03P;

		float line01toStarT = start - half01T;
		float line03toStarT = start - half03T;
		float line01toStarP = starp - half01P;
		float line03toStarP = starp - half03P;

		int a = (((line03to12T * line03toStarP) - (line03toStarT * line03to12P)) > 0.f) ? 1 : -1;
		int b = (((line01to23T * line01toStarP) - (line01toStarT * line01to23P)) > 0.f) ? 1 : -1;

		perc *= 0.5f;

		if (sgn * a > 0) {
			if (sgn * b > 0) {
				t2 = half12T;
				t0 = half01T;
				t3 = midT;
				p2 = half12P;
				p0 = half01P;
				p3 = midP;
				starInPixX -= perc;
				starInPixY -= perc;
			}
			else {
				t2 = midT;
				t1 = half01T;
				t3 = half03T;
				p2 = midP;
				p1 = half01P;
				p3 = half03P;
				starInPixX -= perc;
				starInPixY += perc;
			}
		}
		else {
			if (sgn * b > 0) {
				t1 = half12T;
				t3 = half23T;
				t0 = midT;
				p1 = half12P;
				p3 = half23P;
				p0 = midP;
				starInPixX += perc;
				starInPixY -= perc;
			}
			else {
				t0 = half03T;
				t1 = midT;
				t2 = half23T;
				p0 = half03P;
				p1 = midP;
				p2 = half23P;
				starInPixX += perc;
				starInPixY += perc;
			}
		}
		midT = (t0 + t1 + t2 + t3) * .25f;
		midP = (p0 + p1 + p2 + p3) * .25f;
	}
	start = starInPixY;
	starp = starInPixX;
}

/// <summary>
/// Interpolates the location using neirest neighbour interpolation
/// </summary>
/// <param name="percDown">fraction down from top left to top right line</param>
/// <param name="percRight">fraction right from top left to bottom left line</param>
/// <param name="cornersCel">Values at top left, top right, bottom left and bottom right corners respectively</param>
/// <returns></returns>
template <class T, bool CheckPi> __device__ T interpolateNeirestNeighbour(float percDown, float percRight, T* cornersCel) {
	T corners[4] = { cornersCel[0], cornersCel[1], cornersCel[2], cornersCel[3] };

	piCheckTot<T, CheckPi>(corners, PI_CHECK_FACTOR,4);
	return corners[(int)(2 * roundf(percDown) + roundf(percRight))];
}

template <class T, bool CheckPi> __device__ T interpolateLinear(float percDown, float percRight, T* cornersCel) {
	T corners[4] = { cornersCel[0], cornersCel[1], cornersCel[2], cornersCel[3] };
	piCheckTot<T, CheckPi>(corners, PI_CHECK_FACTOR,4);

	return (1 - percRight) * ((1 - percDown) * corners[0] + percDown * corners[2]) + percRight * ((1 - percDown) * corners[1] + percDown * corners[3]);
}

template <class T> __device__ T hermite(float aValue, T& aX0, T& aX1, T& aX2, T& aX3,
	float aTension, float aBias) {
	/* Source:
	* http://paulbourke.net/miscellaneous/interpolation/
	*/

	float const v = aValue;
	float const v2 = v * v;
	float const v3 = v * v2;

	float const aa = (1.f + aBias) * (1.f - aTension) / 2.f;
	float const bb = (1.f - aBias) * (1.f - aTension) / 2.f;

	T m0 = aa * (aX1 - aX0) + bb * (aX2 - aX1);
	T m1 = aa * (aX2 - aX1) + bb * (aX3 - aX2);

	float const u0 = 2.f * v3 - 3.f * v2 + 1.f;
	float const u1 = v3 - 2.f * v2 + v;
	float const u2 = v3 - v2;
	float const u3 = -2.f * v3 + 3.f * v2;

	return u0 * aX1 + u1 * m0 + u2 * m1 + u3 * aX2;
}

template <class T, bool CheckPi> __device__ T findPoint(const int i, const int j, const int GM, const int GN, 
	const int offver, const int offhor, const int gap, const T* grid, int count, T& r_check) {
	T gridpt = grid[i * GM + j];
	
	if (gridpt.x == -2 && gridpt.y == -2) {
		//return{ -1, -1 };
		int j2 = (j + offhor * gap + GM) % GM;
		int i2 = i + offver * gap;
		T ij2 = grid[ i2 * GM + j2];
		if (ij2.x != -2 && ij2.y != -2) {

			int j0 = (j - offhor * gap + GM) % GM;
			int i0 = (i - offver * gap);

			T ij0 = grid[i0 * GM + j0];

			//If either ij0 or ij2 is not one the same surface directly return the point since there exist no visible point on the same surface 
			//if ((fabsf(ij0.z - r_check.z) < R_CHANGE_THRESHOLD) || (fabsf(ij2.z - r_check.z) < R_CHANGE_THRESHOLD)) return ij0;


			
			int jprev = (j - 3 * offhor * gap + GM) % GM;
			int jnext = (j + 3 * offhor * gap + GM) % GM;
			int iprev = i - offver * 3 * gap;
			int inext = i + offver * 3 * gap;
			if (offver != 0) {
				if (i2 == 0) {
					jnext = (j0 + GM / 2) % GM;
					inext = i0;
				}
				else if (i0 == GN - 1) {
					jprev = (j0 + GM / 2) % GM;
					iprev = i2;
				}
				else if (i2 == GN - 1) {
					inext = i0;
					jnext = (j0 + GM / 2) % GM;
				}
			}
			T ijprev = grid[iprev * GM + jprev];
			T ijnext = grid[inext * GM + jnext];

			//If the ijnext and prev are integrated and on the same surface we can use hermite interpolation.
			if (ijprev.x > -2 && ijnext.x > -2 
				//&& (fabsf(ijprev.z - r_check.z) < R_CHANGE_THRESHOLD)
				//&& (fabsf(ijnext.z - r_check.z) < R_CHANGE_THRESHOLD)
				) {
				T pt[4] = { ijprev, ij0, ij2, ijnext };
				if (pt[0].x != -1 && pt[3].x != -1) {
					piCheckTot<T, CheckPi>(pt, PI_CHECK_FACTOR, 4);
					return hermite(0.5f, pt[0], pt[1], pt[2], pt[3], 0.f, 0.f);
				}
			}

			T pt[2] = { ij2, ij0 };
			piCheckTot<T, CheckPi>(pt, PI_CHECK_FACTOR, 2);
			return  .5f * pt[0] +  .5f * pt[1];
		}
		else {
			//If we cant find points to interpolate with we give up on finding the point and eventually interpolated linearly.
			return { -1, -1 };
		}
	}
	//return { 0, 0 };
	return gridpt;
}

template <class T, bool CheckPi> __device__ T interpolateHermite(const int i, const int j, const int gap, const int GM, const int GN, const float percDown, const float percRight,
	 T* cornersCel, const T* grid, int count, T& r_check) {

	int k = i + gap;
	int l = (j + gap) % GM;
	int imin1 = i - gap;
	int kplus1 = k + gap;
	int jmin1 = (j - gap + GM) % GM;
	int lplus1 = (l + gap) % GM;
	int jx = j;
	int jy = j;
	int lx = l;
	int ly = l;

	if (i == 0) {
		jx = (j + GM / 2) % GM;
		lx = (jx + gap) % GM;
		imin1 = k;
	}
	else if (k == GN - 1) {
		jy = (j + GM / 2) % GM;
		ly = (jy + gap) % GM;
		kplus1 = i;
	}

	cornersCel[4] = findPoint<T,CheckPi>(i, jmin1, GM, GN,  0, -1, gap, grid, count,r_check);		//4 upleft
	cornersCel[5] = findPoint<T, CheckPi>(i, lplus1, GM, GN,  0, 1, gap, grid, count, r_check);		//5 upright
	cornersCel[6] = findPoint<T, CheckPi>(k, jmin1, GM, GN,  0, -1, gap, grid, count, r_check);		//6 downleft
	cornersCel[7] = findPoint<T, CheckPi>(k, lplus1, GM, GN,  0, 1, gap, grid, count, r_check);		//7 downright
	cornersCel[8] = findPoint<T, CheckPi>(imin1, jx, GM, GN,  -1, 0, gap, grid, count, r_check);		//8 lefthigh
	cornersCel[9] = findPoint<T, CheckPi>(imin1, lx, GM, GN,  -1, 0, gap, grid, count, r_check);		//9 righthigh
	cornersCel[10] = findPoint<T, CheckPi>(kplus1, jy, GM, GN,  1, 0, gap, grid, count, r_check);		//10 leftdown
	cornersCel[11] = findPoint<T, CheckPi>(kplus1, ly, GM, GN,  1, 0, gap, grid, count, r_check);		//11 rightdown

	piCheckTot<T, CheckPi>(cornersCel, PI_CHECK_FACTOR, 12);


	//If any of the extra points are in the black hole return a linear interpolation (we know the inner points are correct)
	//Or if the r coordinate differs too much meaning they are on different disk sections or in the background
	//Or if the Differnce in phi is larger than PI to indicate a change of surface of the disk
	for (int q = 4; q < 12; q++) {
		if (isnan(cornersCel[q].x) || cornersCel[q].x == -1) return interpolateLinear<T, CheckPi>(percDown, percRight, cornersCel);
		if (fabs(cornersCel[q - 1].y - cornersCel[q].y) > PI)  return interpolateLinear<T, CheckPi>(percDown, percRight, cornersCel);
	}


	T interpolateUp = hermite(percRight, cornersCel[4], cornersCel[0], cornersCel[1], cornersCel[5], 0.f, 0.f);
	T interpolateDown = hermite(percRight, cornersCel[6], cornersCel[2], cornersCel[3], cornersCel[7], 0.f, 0.f);
	T interpolateUpUp = cornersCel[8] + percRight * (cornersCel[9] - cornersCel[8]);
	T interpolateDownDown = cornersCel[10] + percRight * (cornersCel[11] - cornersCel[10]);
	//HERMITE FINITE

	T r = hermite(percDown, interpolateUpUp, interpolateUp, interpolateDown, interpolateDownDown, 0.f, 0.f);
	if(r.y < 0)  return interpolateLinear<T, CheckPi>(percDown, percRight, cornersCel);
	return r;
}

template <class T, bool CheckPi> __device__ T interpolateSpline(const int i, const int j, const int gap, const int GM, const int GN, float perc_down, float perc_right, 
	T* cornersCel, const T* grid) {


	T r_check = cornersCel[0];
	for (int q = 0; q < 4; q++) {
		if (isnan(cornersCel[q].x)) return interpolateNeirestNeighbour<T, CheckPi>(perc_down, perc_right,cornersCel);
	}

	return interpolateHermite<T, CheckPi>(i, j, gap, GM, GN, perc_down, perc_right, cornersCel, grid, 0, r_check);
	//return interpolateLinear<T, CheckPi>(perc_down, perc_right, cornersCel);
}