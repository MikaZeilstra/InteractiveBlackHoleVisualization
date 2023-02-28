#include <iostream>
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include <stdint.h> 
#include <sstream>
#include <string>
#include <stdlib.h>
#include <vector>

#include "../Header files/Parameters.h"
#include "../Header files/BlackHole.h"
#include "../Header files/Camera.h"
#include "../Header files/Grid.h"
#include "../Header files/Viewer.h"
#include "../../CUDA/Header files/ImageDistorterCaller.cuh"
#include "../Header files/StarProcessor.h"
#include "../Header files/CelestialSkyProcessor.h"
#include "../Header files/Archive.h"

/// <summary>
/// Prints the number of blocks for each level, and total rays of a grid.
/// </summary>
/// <param name="grid">The grid.</param>
/// <param name="maxlevel">The maxlevel.</param>
void gridLevelCount(Grid& grid, int maxlevel) {
	std::vector<int> check(maxlevel + 1);
	for (int p = 1; p < maxlevel + 1; p++)
		check[p] = 0;
	for (auto block : grid.blockLevels)
		check[block.second]++;
	for (int p = 1; p < maxlevel + 1; p++)
		std::cout << "lvl " << p << " blocks: " << check[p] << std::endl;
	std::cout << std::endl << "Total rays: " << grid.CamToCel.size() << std::endl << std::endl;
}

/// <summary>
/// Compares two images and gives the difference error.
/// Prints error info and writes difference image.
/// </summary>
/// <param name="filename1">First image.</param>
/// <param name="filename2">Second image.</param>
void compare(std::string filename1, std::string filename2, std::string writeFilename) {
	std::vector<int> compressionParams;
	compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
	compressionParams.push_back(0);
	cv::Mat compare = cv::imread(filename1);
	compare.convertTo(compare, CV_32F);
	cv::Mat compare2 = cv::imread(filename2);
	compare2.convertTo(compare2, CV_32F);
	cv::Mat imgMINUS = (compare - compare2);
	cv::Mat imgabs = cv::abs(imgMINUS);
	cv::Scalar sum = cv::sum(imgabs);

	double minVal;
	double maxVal;
	cv::Point minLoc;
	cv::Point maxLoc;
	cv::Mat m_out;
	cv::transform(imgabs, m_out, cv::Matx13f(1, 1, 1));
	cv::minMaxLoc(m_out, &minVal, &maxVal, &minLoc, &maxLoc);

	std::cout << 1.f * (sum[0] + sum[1] + sum[2]) / (255.f * 1920 * 960 * 3) << std::endl;
	std::cout << minVal << " " << maxVal / (255.f * 3.f) << std::endl;

	cv::Mat m_test;
	cv::transform(compare, m_test, cv::Matx13f(1, 1, 1));
	cv::minMaxLoc(m_test, &minVal, &maxVal, &minLoc, &maxLoc);
	std::cout << minVal << " " << maxVal / (255.f * 3.f) << std::endl;
	imgMINUS = 4 * imgMINUS;
	imgMINUS = cv::Scalar::all(255) - imgMINUS;
	cv::imwrite(writeFilename, imgMINUS, compressionParams);
}

void reportDuration(std::chrono::time_point<std::chrono::high_resolution_clock> start_time, std::string did, std::string something) {
	auto end_time = std::chrono::high_resolution_clock::now();
	std::cout << did << " " << something << " in " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms!" << 
		std::endl << std::endl;
}

int main()
{
	/* ----------------------- VARIABLE SETTINGS -------------------------- */

	Parameters param("parameters.txt");
	
	

	/* --------------------- INITIALIZATION BLACK HOLE -------------------- */

	BlackHole black = BlackHole(param.afactor);
	metric::setMetricParameters(param.afactor,param.accretionDiskMaxRadius,param.useAccretionDisk);
	std::cout << "Initialized Black Hole " << std::endl << std::endl;

	/* ----------------------- SETUP CUDA -------------------------- */
	CUDA::init();

	int max_grid_size = pow(2, param.gridMaxLevel) + 1;

	CUDA::allocateGridMemory(max_grid_size * max_grid_size);


	/* ------------------ INITIALIZATION CAMERAS & GRIDS ------------------ */

	std::vector<Camera> cams;
	std::vector<Grid> grids(param.gridNum);

	for (int q = 0; q < param.gridNum; q++) {
		double camRad = param.getRadius(q);
		double camInc = param.getInclination(q);
		double camSpeed = param.getSpeed(q);

		Camera cam;
		if (param.userSpeed) cam = Camera(camInc, 0, camRad, camSpeed);
		else cam = Camera(camInc, 0, camRad, param.br, param.btheta, param.bphi);
		cams.push_back(cam);

		std::cout << "Initialized Camera at Radius " << camRad;
		std::cout << " and Inclination " << camInc / PI << "pi" << std::endl;
		


		/* ------------------ GRID LOADING / COMPUTATION ------------------ */
		#pragma region loading grid from file or computing new grid
		
		std::string gridFilename = param.getGridFileName(camRad, camInc, camSpeed);

		//if (!Archive<Grid>::load(gridFilename, grids[q])) {

			std::cout << "Computing new grid file..." << std::endl << std::endl;
			auto start_time = std::chrono::high_resolution_clock::now();
			grids[q] = Grid(param.gridMaxLevel, param.gridStartLevel, param.angleView, &cam, &black, param);
			reportDuration(start_time, "Computed", "grid file");

			std::cout << "Writing to file..." << std::endl << std::endl;
			//Archive<Grid>::serialize(gridFilename, grids[q]);

			gridLevelCount(grids[q], param.gridMaxLevel);
			grids[q].drawBlocks(param.getGridBlocksFileName(camRad, camInc, camSpeed));

			//grids[q].makeHeatMapOfIntegrationSteps(heatMapFilename);
		//}
		std::cout << "Initialized Grid " << q + 1 << " of " << param.gridNum << std::endl;
	}
	std::cout << "Initialized all grids" << std::endl << std::endl;
	
	CUDA::Texture accretionTexture = {};

	/* ------------------- ACCRETION DISK TEXTURE --------------------- */
	if (param.useAccretionDiskTexture) {

		cv::Mat accretionTextureMat = cv::imread(param.getAccretionDiskTextureFolder() + param.accretionDiskTexture);

		accretionTexture = { accretionTextureMat ,accretionTextureMat.cols, accretionTextureMat.rows };
	}


	/* -------------------- INITIALIZATION STARS ---------------------- */


	StarProcessor starProcessor;
	std::string starFilename = param.getStarFileName();
	if (!Archive<StarProcessor>::load(starFilename, starProcessor)) {

		std::cout << "Computing new star file..." << std::endl;
		auto start_time = std::chrono::high_resolution_clock::now();
		starProcessor = StarProcessor(param);
		reportDuration(start_time, "Calculated", "star file");
		auto end_time = std::chrono::high_resolution_clock::now();

		std::cout << "Writing to file..." << std::endl << std::endl;
		Archive<StarProcessor>::serialize(starFilename, starProcessor);
	}
	std::cout << "Initialized " << starProcessor.starSize <<  " Stars" << std::endl << std::endl;
	
	/* ----------------------- INITIALIZATION CELESTIAL SKY ----------------------- */
	
	CelestialSkyProcessor celestialSkyProcessor;
	std::string celestialSkyFilename = param.getCelestialSum();
	if (!Archive<CelestialSkyProcessor>::load(celestialSkyFilename, celestialSkyProcessor)) {

		std::cout << "Computing new celestial sky file..." << std::endl;

		auto start_time = std::chrono::high_resolution_clock::now();
		celestialSkyProcessor = CelestialSkyProcessor(param);
		reportDuration(start_time, "Calculated", "celestial sky file");

		std::cout << "Writing to file..." << std::endl;
		Archive<CelestialSkyProcessor>::serialize(celestialSkyFilename, celestialSkyProcessor);
	}
	std::cout << "Initialized Celestial Sky " << param.celestialSkyImg << std::endl << std::endl;


	/* --------------------- INITIALIZATION VIEW ---------------------- */

	Viewer view = Viewer(param);
	std::cout << "Initialized Viewer " << std::endl << std::endl;

	/* ----------------------- CALL CUDA ----------------------- */
	//Distorter spacetime(&grids, &view, &starProcessor, &cams, &celestialProcessor);
	CUDA::call(grids, cams, starProcessor, view, celestialSkyProcessor, accretionTexture, param);

}