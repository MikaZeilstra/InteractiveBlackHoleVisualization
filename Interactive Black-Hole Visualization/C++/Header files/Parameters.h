#pragma once
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <iostream>
#include <libconfig.h++>
#include "../../CUDA/Header files/Constants.cuh"

struct Parameters {
	
	bool sphereView, angleView;
	int windowWidth, windowHeight = 1920;
	double viewAngle;
	cv::Point2i viewOffset;
	int nrOfFrames;

	std::string celestialSkyImg, starCatalogue, diffractionImg, accretionDiskTexture;
	int starTreeLevel, starMagnitudeCut;
	double br, bphi, btheta;
	bool userSpeed;
	bool useStars, useRedshift, useLensing,useAccretionDisk;
	bool savePaths;
	bool camSpeedChange, camInclinationChange, camRadiusChange;
	cv::Point2d camSpeedFromTo, camInclinationFromTo, camRadiusFromTo;
	double camRadiusStepsize, camSpeedStepsize, camInclinationStepsize;
	double afactor;
	double accretionDiskMaxRadius;
	double blackholeMass, blackholeAccretion;
	int accretionTemperatureLUTSize = 0;
	bool useAccretionDiskTexture = false;
	int gridStartLevel, gridMaxLevel, gridMinLevel;
	int gridNum = 1;

	std::string getResourceFolder() const {
		return "../Resources/";
	}

	std::string getGridFolder() const {
		return getResourceFolder() + "Grids/";
	}

	std::string getStarFolder() const {
		return getResourceFolder() + "Stars/";
	}

	std::string getAccretionDiskTextureFolder() const {
		return getResourceFolder() + "AccretionDiskTexture/";
	}

	std::string getCelestialSkyFolder() const {
		return getResourceFolder() + "CelestialBackgrounds/";
	}

	std::string getInitializationFolder() const {
		return getResourceFolder() + "Initialization/";
	}

	std::string getResultsFolder() const {
		return "../Results/";
	}

	std::string getGridResultFolder() const {
		return "Grids/";
	}

	std::string getInterpolatedGridResultFolder() const {
		return "Interpolated grids/";
	}

	std::string getCelestialSummedFolder() const {
		return getCelestialSkyFolder() + "Summed/";
	}


	std::string getCelestialSum() {
		return getCelestialSummedFolder() + celestialSkyImg + ".sum";
	}

	std::string getGridBlocksFolder() const {
		return getGridFolder() + "Blocks/";
	}

	std::string getDiffractionFolder() const {
		return getResourceFolder() + "Diffraction/";
	}

	std::string getGeodesicsResultFolder() const {
		return "Geodesics/";
	}
 
	std::string getStarDiffractionFile() const {
		return getDiffractionFolder() + diffractionImg;
	}

	std::string getGridDescription(float camRad, float camInc, float camSpeed) const {
		std::stringstream ss;

		ss << std::setprecision(3)
			<< "Grid_" << gridStartLevel << "_to_" << gridMaxLevel 
			<< "_Spin_" << afactor << "_Rad_" << camRad << "_Inc_" << camInc / PI << "pi";
		if (userSpeed) ss << "_Speed_" << camSpeed;

		return ss.str();
	}

	std::string getResultFileName(float alpha, int q, const char folder[] = "", const char name_extra[] = "", const char ext[] = "png") const {
		float camRad = camRadiusFromTo.x;
		float camInc = camInclinationFromTo.x;
		float camSpeed = camSpeedFromTo.x;
		if (camRadiusChange) camRad = camRadiusFromTo.x + alpha * (camRadiusFromTo.y - camRadiusFromTo.x);
		if (camInclinationChange) camInc = (camInclinationFromTo.x + alpha * (camInclinationFromTo.y - camInclinationFromTo.x)) / PI;
		if (camSpeedChange) camSpeed = camSpeedFromTo.x + alpha * (camSpeedFromTo.y - camSpeedFromTo.x);

		std::stringstream ss;
		ss << getResultsFolder() << folder << getGridDescription(camRad, camInc, camSpeed) << "_" << q << name_extra << "." << ext;
		return ss.str();
	}

	std::string getInterpolatedGridResultFileName(float alpha, int q, const char name_extra[] = "") const {
		return getResultFileName(alpha, q, getInterpolatedGridResultFolder().c_str(), name_extra);
	}

	std::string getGridResultFileName(float alpha, int q, const char name_extra[] = "") const {
		return getResultFileName(alpha, q, getGridResultFolder().c_str(), name_extra);
	}

	std::string getGeodesicsResultFileName(float alpha, int q, const char name_extra[] = "") const {
		return getResultFileName(alpha, q, getGeodesicsResultFolder().c_str(), name_extra,"geo");
	}
	

	std::string getGridFileName(float camRad, float camInc, float camSpeed) const {
		std::stringstream ss;
		ss << getGridFolder() << getGridDescription(camRad, camInc, camSpeed) << ".grid";
		return ss.str();
	}

	std::string getStarFileName() {
		// Filename for stars and image.
		std::stringstream ss;
		ss << getStarFolder() << "Stars_lvl_" << starTreeLevel << "_m" << starMagnitudeCut << ".star";
		return ss.str();
	}

	std::string getGridBlocksFileName(float camRad, float camInc, float camSpeed) const {
		std::stringstream ss;
		ss << getGridBlocksFolder() << getGridDescription(camRad, camInc, camSpeed) << ".png";
		return ss.str();
	}
	
	double getRadius(int step) {
		if (camRadiusChange) return camRadiusFromTo.x + step * ((camRadiusFromTo.y - camRadiusFromTo.x) / (gridNum - 1.0));
		return camRadiusFromTo.x;
	}

	double getInclination(int step) {
		if (camInclinationChange) return camInclinationFromTo.x + step * ((camInclinationFromTo.y - camInclinationFromTo.x) / (gridNum - 1.0));
		return camInclinationFromTo.x;
	}

	double getSpeed(int step) {
		if (camSpeedChange) return camSpeedFromTo.x + step * ((camSpeedFromTo.y - camSpeedFromTo.x) / (gridNum - 1.0));
		return camSpeedFromTo.x;
	}

	Parameters(std::string option_file) {
		libconfig::Config config;

		try {
			config.readFile(getInitializationFolder() + option_file);
		}
		catch (libconfig::FileIOException& e) {
			/*inform user about IOException*/
			std::cerr << "FileIOException occurred. Could not read the Initialization file!!\n";
			/*terminate program*/
			exit(EXIT_FAILURE);
		}
		catch (libconfig::ParseException& e) {
			/*inform user about the parse exception*/
			std::cerr << "Parse error at " << e.getFile() << ":" << e.getLine()
				<< " - " << e.getError() << std::endl;
			/*terminate program*/
			exit(EXIT_FAILURE);
		}

		try {
			sphereView = config.lookup("sphereView");
			windowWidth = config.lookup("windowWidth");
			windowHeight = config.lookup("windowHeight");



			viewAngle = config.lookup("viewAngle");
			viewAngle *= PI;
			viewOffset.x = config.lookup("offsetX");
			viewOffset.y = config.lookup("offsetY");
			viewOffset *= PI;
			nrOfFrames = config.lookup("nrOfFrames");

			std::string str1 = config.lookup("celestialSkyImg");
			celestialSkyImg = str1;

			useRedshift = config.lookup("useRedshift");
			useLensing = config.lookup("useLensing");
			savePaths = config.lookup("saveGeodesics");
			
			useStars = config.lookup("useStars");
			std::string str2 = config.lookup("starCatalogue");
			starCatalogue = str2;
			starTreeLevel = config.lookup("starTreeLevel");
			starMagnitudeCut = config.lookup("starMagnitudeCut");
			std::string str3 = config.lookup("diffractionImg");
			diffractionImg = str3;

			userSpeed = config.lookup("userSpeed");
			br = config.lookup("br");
			bphi = config.lookup("bphi");
			btheta = config.lookup("btheta");
			camSpeedFromTo.x = config.lookup("camSpeed");
			camRadiusFromTo.x = config.lookup("camRadius");
			camInclinationFromTo.x = config.lookup("camInclination");
			angleView = camInclinationFromTo.x != 0.5;
			camInclinationFromTo *= PI;

			afactor = config.lookup("afactor");

			gridStartLevel = config.lookup("gridStartLevel");
			gridMinLevel = config.lookup("gridMinLevel");
			gridMaxLevel = config.lookup("gridMaxLevel");

			useAccretionDisk = config.lookup("useAccretionDisk");
			if (useAccretionDisk) {
				useAccretionDiskTexture = config.lookup("useAccretionDiskTexture");
				if (useAccretionDiskTexture) {
					std::string str = config.lookup("accretionDiskTexture");
					accretionDiskTexture = str;
				}

				accretionDiskMaxRadius = config.lookup("accretionDiskRadius");
				accretionTemperatureLUTSize = config.lookup("temperatureLUTSize");
				blackholeMass = config.lookup("blackholeMass");
				blackholeAccretion = config.lookup("blackholeAccretionRate");
			}

			camSpeedChange = config.lookup("camSpeedChange");
			if (camSpeedChange) {
				camSpeedFromTo.x = config.lookup("camSpeedFrom");
				camSpeedFromTo.y = config.lookup("camSpeedTo");
				camSpeedStepsize = config.lookup("camSpeedStepsize");
			}

			camRadiusChange = config.lookup("camRadiusChange");
			if (camRadiusChange) {
				camRadiusFromTo.x = config.lookup("camRadiusFrom");
				camRadiusFromTo.y = config.lookup("camRadiusTo");
				camRadiusStepsize = config.lookup("camRadiusStepsize");
			}

			camInclinationChange = config.lookup("camInclinationChange");
			if (camInclinationChange) {
				camInclinationFromTo.x = config.lookup("camInclinationFrom");
				camInclinationFromTo.y = config.lookup("camInclinationTo");
				camInclinationFromTo *= PI;
				camInclinationStepsize = config.lookup("camInclinationStepsize");
				camInclinationStepsize *= PI;
			}
		}
		catch (libconfig::SettingNotFoundException& e) {
			std::cerr << "Incorrect setting(s) in configuration file." << std::endl;
		}

		if (camRadiusChange) gridNum = 1. + round(abs(camRadiusFromTo.y - camRadiusFromTo.x) / camRadiusStepsize);
		else if (camInclinationChange) gridNum = 1. + round(abs(camInclinationFromTo.y - camInclinationFromTo.x) / camInclinationStepsize);
		else if (camSpeedChange) gridNum = 1. + round(abs(camSpeedFromTo.y - camSpeedFromTo.x) / camSpeedStepsize);

		if (sphereView) windowHeight = (int)floor(windowWidth / 2);
	}
};