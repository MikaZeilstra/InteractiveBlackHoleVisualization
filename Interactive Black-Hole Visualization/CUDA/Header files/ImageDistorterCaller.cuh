#pragma once
#include <iostream>
#include <stdint.h>
#include <iomanip>
#include <algorithm>
//#include <GL/glew.h>
//#include <GL/freeglut.h>
//#include <cuda_gl_interop.h>
#include <chrono>
#include <vector>
#include <vector_types.h>

#include "GridLookup.cuh"
#include "intellisense_cuda_intrinsics.cuh"
#include "ColorComputation.cuh"
#include "GridLookup.cuh"
#include "GridInterpolation.cuh"
#include "StarDeformation.cuh"
#include "ShadowComputation.cuh"
#include "ImageDeformation.cuh"
#include "glad.h"

#include "../../C++/Header files/CelestialSkyProcessor.h"
#include "../../C++/Header files/Grid.h"
#include "../../C++/Header files/StarProcessor.h"
#include "../../C++/Header files/Camera.h"
#include "../../C++/Header files/Viewer.h"
#include "../../C++/Header files/ViewCamera.h"

#include "../../CUDA/Header files/vector_operations.cuh"


#ifndef CUDA_FUNCTIONS
#define CUDA_FUNCTIONS

namespace CUDA {

	struct CelestialSky {
		CelestialSky(CelestialSkyProcessor& celestialsky) {
			summedCelestialSky = (float4*)&(celestialsky.summedImageVec[0]);
			rows = celestialsky.rows;
			cols = celestialsky.cols;
			imsize = { rows, cols };
			minmaxnr = (int)(cols * 0.2f);
		}
		float4* summedCelestialSky;
		int rows;
		int cols;
		int2 imsize;
		int minmaxnr;
	};

	struct Stars {
		Stars(StarProcessor& starProc) {
			tree = &(starProc.binaryStarTree[0]);
			stars = &(starProc.starPos[0]);
			starSize = starProc.starSize;
			treeLevel = starProc.treeLevel;
			magnitude = &(starProc.starMag[0]);
		};

		float* stars;
		int* tree;
		int starSize;
		float* magnitude;
		int treeLevel;
	};

	struct Image {
		Image(Viewer& view) {
			M = view.pixelwidth;
			N = view.pixelheight;
			viewAngle = view.viewAngleWide;
			viewer = (float2*)&(view.viewMatrix[0]);
			compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
			compressionParams.push_back(0);
			result.resize(N * M* 4);
		}
		int M;
		int N;
		float viewAngle;
		float2* viewer;
		mutable std::vector<uchar> result;
		// Image and frame parameters
		std::vector<int> compressionParams;

	};

	struct Texture {
		cv::Mat texture;
		int width;
		int height;
		std::vector<float4> summed;

		Texture() {
			width = 0;
			height = 0;
		};

		Texture(cv::Mat texture_, int width_, int height_) {
			texture = texture_;
			width = width_;
			height = height_;

			uchar4* texture_pointer = reinterpret_cast<uchar4*>(texture.data);


			summed = std::vector<float4>(width * height);
			summed[0] = (1/255.f) * texture_pointer[0] ;


			for (int i = 1; i < width; i++) {
				summed[i * height] = summed[(i - 1) * height] + (1 / 255.f) * texture_pointer[i];
			}

			for (int i = 1; i < height; i++) {
				summed[i] = summed[i - 1] + (1 / 255.f) * texture_pointer[i * width];
			}

			for (int x = 1; x < width; x++) {
				for (int y = 1; y < height; y++) {
					summed[x * height+ y] = summed[(x) * height + (y-1)] + summed[(x-1)*height + (y)] + (1 / 255.f) * texture_pointer[y * width + x] - summed[(x-1)*height + (y - 1)];
				}
			}
		};


	};

	struct Grids {
		Grids(std::vector<Grid>& grids, std::vector<Camera>& cameras) {
			G = grids.size();
			GM = grids[0].M;
			GN = grids[0].N;
			for (int g = 0; g < G; g++) {
				grid_vectors.push_back(grids[g].grid_vector);
				grid_disk_vectors.push_back(grids[g].disk_grid_vector);
				grid_incident_vectors.push_back(grids[g].disk_incident_vector);
			}
			camParams.resize(10 * G);
			for (int g = 0; g < G; g++) {
				std::vector<float> camParamsG = cameras[g].getParamArray();
				for (int cp = 0; cp < 10; cp++) camParams[g * 10 + cp] = camParamsG[cp];
			}
			gridStart = cameras[0].r;
			gridStep = (cameras[G - 1].r - gridStart) / (1.f * G - 1.f);
			camParam = &(camParams[0]);
			level = grids[0].MAXLEVEL;
			sym = float(GM) / float(GN) > 3 ? 1 : 0;
			GN1 = (sym == 1) ? 2 * GN - 1 : GN;
			

		}
		std::vector<float> camParams;
		std::vector<std::vector<float2>>grid_vectors;
		std::vector<std::vector<float2>>grid_disk_vectors;
		std::vector<std::vector<float3>>grid_incident_vectors;
		int GM;
		int GN;
		int GN1;
		int level;
		int G;
		float gridStep;
		float gridStart;
		float* camParam;
		int sym;
	};

	struct StarVis {
		StarVis(Stars& stars, Image& img, Parameters& param) {
			gaussian = 1;
			diffSize = img.M / 16;
			cv::Mat diffImg = cv::imread(param.getStarDiffractionFile());
			cv::resize(diffImg, diffImgSmall, cv::Size(diffSize, diffSize), 0, 0, cv::INTER_LINEAR_EXACT);
			diffraction = (uchar3*)diffImgSmall.data;
			trailnum = 30;
			diffusionFilter = gaussian * 2 + 1;
			searchNr = (int)powf(2, stars.treeLevel / 3 * 2);

			starSize = stars.starSize;
			treeLevel = stars.treeLevel;
		}
		cv::Mat diffImgSmall;
		int gaussian;
		int diffusionFilter;
		int trailnum;
		int searchNr;
		uchar3* diffraction;
		int diffSize;

		int starSize;
		int treeLevel;

	};

	struct GPUBlocks {
		GPUBlocks(Parameters& param, const Stars& stars,const Image image) {
			numBlocks_starsize = stars.starSize / threadsPerBlock_32 + 1;
			numBlocks_bordersize = (param.n_black_hole_angles * 2) / threadsPerBlock_32 + 1;
			numBlocks_tempLUT = param.accretionTemperatureLUTSize / threadsPerBlock_32 + 1;
			numBlocks_disk_edges = (param.n_disk_angles) / threadsPerBlock_32 + 1;

			numBlocks_N_M_4_4 = { (image.N - 1) / threadsPerBlock4_4.x + 1, (image.M - 1) / threadsPerBlock4_4.y + 1 };
			numBlocks_N1_M1_4_4 = { image.N / threadsPerBlock4_4.x + 1, image.M / threadsPerBlock4_4.y + 1 };
			numBlocks_GN_GM_4_4 = { (param.grid_N - 1) / threadsPerBlock4_4.x + 1, (param.grid_M - 1) / threadsPerBlock4_4.y + 1 };

			numBlocks_GN_GM_5_25 = { (param.grid_N - 1) / threadsPerBlock5_25.x + 1, (param.grid_M - 1) / threadsPerBlock5_25.y + 1 };
			numBlocks_N_M_5_25 = { (image.N - 1) / threadsPerBlock5_25.x + 1, (image.M - 1) / threadsPerBlock5_25.y + 1 };
			numBlocks_N1_M1_5_25 = { image.N / threadsPerBlock5_25.x + 1, image.M / threadsPerBlock5_25.y + 1 };

			numBlocks_N_M_1_24 = { (image.N - 1) / threadsPerBlock1_24.x + 1, (image.M - 1) / threadsPerBlock1_24.y + 1 };

		}

		int threadsPerBlock_32 = 32;

		int numBlocks_starsize;
		int numBlocks_bordersize;
		int numBlocks_tempLUT;
		int numBlocks_disk_edges;

		dim3 threadsPerBlock4_4 = { 4, 4 };
		dim3 numBlocks_N_M_4_4;
		dim3 numBlocks_N1_M1_4_4;
		dim3 numBlocks_GN_GM_4_4;

		dim3 threadsPerBlock5_25 = { 5, 25 };
		dim3 numBlocks_GN_GM_5_25;
		dim3 numBlocks_N_M_5_25;
		dim3 numBlocks_N1_M1_5_25;

		dim3 threadsPerBlock1_24 = { 1, 24 };
		dim3 numBlocks_N_M_1_24;
	};

	cudaError_t cleanup();

	//void setDeviceVariables(const Grids& grids, const Image& image, const CelestialSky& celestialSky, const Stars& stars);

	void checkCudaStatus(cudaError_t cudaStatus, const char* message);

	void checkCudaErrors();

	void init();

	void call(BlackHole* bh, StarProcessor& stars, Viewer& view, CelestialSkyProcessor& celestialSky, Texture& accretionTexture, Parameters& param, ViewCamera* viewer);

	//unsigned char* getDiffractionImage(const int size);
	void allocateGridMemory(size_t size);


	void memoryAllocationAndCopy(const Image& image, const CelestialSky& celestialSky,
		const Stars& stars, const StarVis& starvis,const Texture accretionTexture, const Parameters& param);

	void runKernels(BlackHole* bh, const Image& image, const CelestialSky& celestialSky,
		const Stars& stars, const StarVis& starvis, const Texture& accretionDiskTexture, Parameters& param, ViewCamera* viewer);

	template <class T> void integrateGrid(const T rV, const T thetaV, const T phiV, std::vector <T>& pRV,
		std::vector <T>& bV, std::vector <T>& qV, std::vector <T>& pThetaV, float3* disk_incident);
	template void integrateGrid<double>(const double rV, const double thetaV, const double phiV, std::vector <double>& pRV,
		std::vector <double>& bV, std::vector <double>& qV, std::vector <double>& pThetaV, float3* disk_incident);
	template void integrateGrid<float>(const float rV, const float thetaV, const float phiV, std::vector <float>& pRV,
		std::vector <float>& bV, std::vector <float>& qV, std::vector <float>& pThetaV, float3* disk_incident);

	void glfw_setup(ViewCamera* camera);

	/// <summary>
	/// Requests a grid to be generated and send to the GPU
	/// </summary>
	/// <param name="cam_pos">Camera positon of the grid</param>
	/// <param name="cam_speed_dir">Direction of travel of the camera</param>
	/// <param name="speed">Magnitude of velocity of the grid</param>
	/// <param name="bh">The black hole data</param>
	/// <param name="param">Render parameters</param>
	/// <param name="dev_cam">GPU pointer to store camera</param>
	/// <param name="dev_grid">GPU pointer to store generated grid</param>
	/// <param name="dev_disk">GPU pointer to store generated disk grid</param>
	/// <param name="dev_inc">GPU pointer to store generated disk incident angles</param>
	/// <param name="dev_bh_border">GPU pointer to black hole borders</param>
	/// <param name="dev_disk_sum">GPU pointer to disk summary</param>
	/// <param name="dev_disk_sum_inc">GPU pointer to disk summary incident data</param>
	void requestGrid(double3 cam_pos, double3 cam_speed_dir, float speed, GPUBlocks& gpuBlocks, BlackHole* bh, Parameters* param, float* dev_cam, float2* dev_grid, float2* dev_disk, float3* dev_inc,
		float2* dev_bh_border, float2* dev_disk_sum, float3* dev_disk_sum_inc, float2* dev_mapped_disk_vert);


	void manageGrids(Parameters& param, ViewCamera* viewer, GPUBlocks gpuBlocks, BlackHole* bh, int q);

	void CreateTexture(Parameters& param, const Image& image, const StarVis& starvis, const Texture accretionDiskTexture, const CelestialSky& celestialsky, const GPUBlocks& gpublocks,
		float camera_phi_offset, int q, bool should_interpolate_grids, float grid_alpha, float vr_offset);

	void projectTextureToWindow(Parameters& param, const Image& image, ViewCamera* viewer, GLuint gl_PBO, GLuint gl_Tex, GLuint shaderProgram);


	void saveImages(Parameters& param, const Image& image, float grid_value, int q);

	/// <summary>
	/// Swaps grid and related variables with grid_2 and related
	/// </summary>
	void swap_grids();

	std::string readFile(const char* filePath);

	void gridLevelCount(Grid& grid);


}

#endif // !CUDA_FUNCTIONS