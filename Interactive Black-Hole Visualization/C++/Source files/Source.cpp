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
#include "../Header files/IntegrationDefines.h"

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
	metric::setMetricParameters<INTEGRATION_PRECISION_MODE>(param.afactor,param.accretionDiskMinRadius,param.accretionDiskMaxRadius,param.useAccretionDisk);
	std::cout << "Initialized Black Hole " << std::endl << std::endl;

	/* ----------------------- SETUP CUDA -------------------------- */
	CUDA::init();

	int max_grid_size = pow(2, param.gridMaxLevel + 1) + 1;

	CUDA::allocateGridMemory(max_grid_size * max_grid_size);


	/* ------------------ INITIALIZATION CAMERAS & GRIDS ------------------ */
	
	CUDA::Texture accretionTexture = {};

	/* ------------------- ACCRETION DISK TEXTURE --------------------- */
	if (param.useAccretionDiskTexture) {

		cv::Mat accretionTextureMat = cv::imread(param.getAccretionDiskTextureFolder() + param.accretionDiskTexture, cv::IMREAD_UNCHANGED);

		accretionTexture = { accretionTextureMat ,accretionTextureMat.cols, accretionTextureMat.rows };
	}


	/* -------------------- INITIALIZATION STARS ---------------------- */


	StarProcessor starProcessor;
	std::string starFilename = param.getStarFileName();
	if (!Archive<StarProcessor>::load(starFilename, starProcessor) || param.useRandomStars) {

		std::cout << "Computing new star file..." << std::endl;
		auto start_time = std::chrono::high_resolution_clock::now();
		starProcessor = StarProcessor(param);
		reportDuration(start_time, "Calculated", "star file");
		auto end_time = std::chrono::high_resolution_clock::now();

		std::cout << "Writing to file..." << std::endl << std::endl;
		if (!param.useRandomStars) {
			Archive<StarProcessor>::serialize(starFilename, starProcessor);
		}
		
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


	ViewCamera* camera = new ViewCamera(&param,{ -1,0,0 }, { 0,0,1 });
	/* ----------------------- CALL CUDA ----------------------- */
	//Distorter spacetime(&grids, &view, &starProcessor, &cams, &celestialProcessor);
	CUDA::call(&black, starProcessor, view, celestialSkyProcessor, accretionTexture, param, camera);

}