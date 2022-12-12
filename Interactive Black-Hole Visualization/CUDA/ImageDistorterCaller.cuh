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

#include "GridLookup.cuh"
#include "intellisense_cuda_intrinsics.cuh"
#include "ColorComputation.cuh"
#include "GridLookup.cuh"
#include "GridInterpolation.cuh"
#include "StarDeformation.cuh"
#include "ShadowComputation.cuh"
#include "ImageDeformation.cuh"

#include "Integration.cuh"

#include "../C++/CelestialSkyProcessor.h"
#include "../C++/Grid.h"
#include "../C++/StarProcessor.h"
#include "../C++/Camera.h"
#include "../C++/Viewer.h"

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
			result.resize(N * M);
		}
		int M;
		int N;
		float viewAngle;
		float2* viewer;
		mutable std::vector<uchar4> result;
		// Image and frame parameters
		std::vector<int> compressionParams;

	};

	struct Grids {
		Grids(std::vector<Grid>& grids, std::vector<Camera>& cameras) {
			G = grids.size();
			GM = grids[0].M;
			GN = grids[0].N;
			for (int g = 0; g < G; g++) {
				hashTable.insert(hashTable.end(), grids[g].hasher.hashTable.begin(), grids[g].hasher.hashTable.end());
				offsetTable.insert(offsetTable.end(), grids[g].hasher.offsetTable.begin(), grids[g].hasher.offsetTable.end());
				hashPosTag.insert(hashPosTag.end(), grids[g].hasher.hashPosTag.begin(), grids[g].hasher.hashPosTag.end());
				tableSize.push_back(grids[g].hasher.hashTableWidth);
				tableSize.push_back(grids[g].hasher.offsetTableWidth);
			}
			camParams.resize(7 * G);
			for (int g = 0; g < G; g++) {
				std::vector<float> camParamsG = cameras[g].getParamArray();
				for (int cp = 0; cp < 7; cp++) camParams[g * 7 + cp] = camParamsG[cp];
			}
			gridStart = cameras[0].r;
			gridStep = (cameras[G - 1].r - gridStart) / (1.f * G - 1.f);
			hashTableSize = hashTable.size() / 2;
			offsetTableSize = offsetTable.size() / 2;
			offsetTables = (int2*)&(offsetTable[0]);
			hashTables = (float2*)&(hashTable[0]);
			hashPosTags = (int2*)&(hashPosTag[0]);
			tableSizes = (int2*)&(tableSize[0]);
			camParam = &(camParams[0]);
			level = grids[0].MAXLEVEL;
			sym = float(GM) / float(GN) > 3 ? 1 : 0;
			GN1 = (sym == 1) ? 2 * GN - 1 : GN;

		}
		std::vector<float> camParams;
		std::vector<float> hashTable;
		std::vector<int> offsetTable, hashPosTag;
		std::vector<int> tableSize;
		int2* offsetTables;
		float2* hashTables;
		int2* hashPosTags;
		int2* tableSizes;
		int GM;
		int GN;
		int GN1;
		int level;
		int offsetTableSize;
		int hashTableSize;
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
		}
		cv::Mat diffImgSmall;
		int gaussian;
		int diffusionFilter;
		int trailnum;
		int searchNr;
		uchar3* diffraction;
		int diffSize;

	};

	struct BlackHoleProc {
		BlackHoleProc(int anglenum) {
			angleNum = anglenum;
			bh = std::vector<float2>((angleNum + 1) * 2);
			bh[0] = { 100, 0 };
			bh[1] = { 100, 0 };
			bhBorder = (float2*)&(bh[0]);
		}
		int angleNum;// = 1000;
		float2* bhBorder;
		std::vector<float2> bh;
	};

	cudaError_t cleanup();

	//void setDeviceVariables(const Grids& grids, const Image& image, const CelestialSky& celestialSky, const Stars& stars);

	void checkCudaStatus(cudaError_t cudaStatus, const char* message);

	void checkCudaErrors();

	void init();

	void call(std::vector<Grid>& grids, std::vector<Camera>& cameras, StarProcessor& stars, Viewer& view, CelestialSkyProcessor& celestialSky, Parameters& param);

	//unsigned char* getDiffractionImage(const int size);
	void allocateGridMemory(size_t size);


	void memoryAllocationAndCopy(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
		const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis);

	void runKernels(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
		const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis, const Parameters& param);

	template <class T> void integrateGrid(const T rV, const T thetaV, const T phiV, std::vector <T>& pRV,
		std::vector <T>& bV, std::vector <T>& qV, std::vector <T>& pThetaV);
	template void integrateGrid<double>(const double rV, const double thetaV, const double phiV, std::vector <double>& pRV,
		std::vector <double>& bV, std::vector <double>& qV, std::vector <double>& pThetaV);
	template void integrateGrid<float>(const float rV, const float thetaV, const float phiV, std::vector <float>& pRV,
		std::vector <float>& bV, std::vector <float>& qV, std::vector <float>& pThetaV);
}

#endif // !CUDA_FUNCTIONS