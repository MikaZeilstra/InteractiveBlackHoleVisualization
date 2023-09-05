#pragma once
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include <iostream>
#include <libconfig.h++>
#include <iomanip>
#include "../../CUDA/Header files/Constants.cuh"

struct Parameters {
	
	bool sphereView;
	int windowWidth, windowHeight = 1920;
	float fov = 70;
	int texWidth, texHeight;
	double viewAngle;
	cv::Point2i viewOffset;
	int nrOfFrames;

	std::string celestialSkyImg, starCatalogue, diffractionImg, accretionDiskTexture;
	int starTreeLevel;
	float starMagnitudeCutLow, starMagnitudeCutHigh;
	double br, bphi, btheta;
	bool userSpeed;
	bool useStars, useRedshift, useLensing,useAccretionDisk;
	bool savePaths;
	bool camSpeedChange, camInclinationChange, camRadiusChange, camPhiChange;
	cv::Point2d camSpeedFromTo, camInclinationFromTo, camRadiusFromTo, camPhiFromTo;
	double afactor;
	double accretionDiskMinRadius, accretionDiskMaxRadius;
	double blackholeMass, blackholeAccretion;
	int accretionTemperatureLUTSize = 0;
	bool useAccretionDiskTexture = false;
	int gridStartLevel, gridMaxLevel, gridMinLevel;
	int gridNum = 1;
	int grid_N, grid_M;
	int n_black_hole_angles;
	int n_disk_angles, n_disk_sample, max_disk_segments;
	bool useRandomStars;
	float randomStarSelectionChance;

	float gridDistanceR, gridDistanceTheta;

	float pixelWidth, screenDistance, screenDepthPerR, interOcularDistance;
	bool useIntegrationDistance;

	int movementMode, outputMode = -1;

	float2 bh_center = {};

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
		ss << getStarFolder() << "Stars_lvl_" << starTreeLevel << "_m" << starMagnitudeCutLow << "-"<< starMagnitudeCutHigh << ".star";
		return ss.str();
	}

	std::string getGridBlocksFileName(float camRad, float camInc, float camSpeed) const {
		std::stringstream ss;
		ss << getGridBlocksFolder() << getGridDescription(camRad, camInc, camSpeed) << ".png";
		return ss.str();
	}
	
	/// <summary>
	/// Returns Phi for given grid number
	/// </summary>
	/// <param name="step">Frame number</param>
	/// <returns></returns>
	double getPhi(int step) {
		if (camPhiChange) return camPhiFromTo.x + step * ((camPhiFromTo.y - camPhiFromTo.x) / (nrOfFrames - 1.0));
		return camPhiFromTo.x;
	}

	/// <summary>
	/// Returns Radius for given grid
	/// </summary>
	/// <param name="step"> Grid number</param>
	/// <returns></returns>
	double getRadius(int step) {
		if (camRadiusChange) return camRadiusFromTo.x + step * ((camRadiusFromTo.y - camRadiusFromTo.x) / (gridNum - 1.0));
		return camRadiusFromTo.x;
	}


	/// <summary>
	/// Returns Inclination for given grid
	/// </summary>
	/// <param name="step"> Grid number</param>
	/// <returns></returns>
	double getInclination(int step) {
		if (camInclinationChange) return camInclinationFromTo.x + step * ((camInclinationFromTo.y - camInclinationFromTo.x) / (gridNum - 1.0));
		return camInclinationFromTo.x;
	}

	/// <summary>
/// Returns speed for given grid
/// </summary>
/// <param name="step"> Grid number</param>
/// <returns></returns>
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
			//Width and height of window
			windowWidth = config.lookup("windowWidth");
			windowHeight = config.lookup("windowHeight");

			//Width and height of distorted sky texture
			texWidth = config.lookup("texWidth");
			texHeight = config.lookup("texHeight");

			viewAngle = config.lookup("viewAngle");
			viewAngle *= PI;
			viewOffset.x = config.lookup("offsetX");
			viewOffset.y = config.lookup("offsetY");
			viewOffset *= PI;
			nrOfFrames = config.lookup("nrOfFrames");
			gridNum = config.lookup("nrOfGrids");
			n_black_hole_angles = config.lookup("n_black_hole_angles");


			std::string str1 = config.lookup("celestialSkyImg");
			celestialSkyImg = str1;

			useRedshift = config.lookup("useRedshift");
			useLensing = config.lookup("useLensing");
			savePaths = config.lookup("saveGeodesics");
			
			useStars = config.lookup("useStars");
			std::string str2 = config.lookup("starCatalogue");
			starCatalogue = str2;
			starTreeLevel = config.lookup("starTreeLevel");
			starMagnitudeCutLow = config.lookup("starMagnitudeCutLow");
			starMagnitudeCutHigh = config.lookup("starMagnitudeCutHigh");
			std::string str3 = config.lookup("diffractionImg");
			diffractionImg = str3;

			useRandomStars = config.lookup("randomStarSelection");
			randomStarSelectionChance = config.lookup("randomSelectionChance");

			movementMode = config.lookup("movementMode");
			outputMode = config.lookup("outputMode");

			if (outputMode == 2) {
				pixelWidth = config.lookup("pixelWidth");
				screenDepthPerR = config.lookup("screenDepthPerR");
				screenDistance = config.lookup("screenDistance");
				interOcularDistance = config.lookup("interOcularDistance");

				useIntegrationDistance = config.lookup("useIntegrationDistance");
			}

			userSpeed = config.lookup("userSpeed");
			br = config.lookup("br");
			bphi = config.lookup("bphi");
			btheta = config.lookup("btheta");

			fov = config.lookup("cameraFov");

			//Lookup (initial) camera position
			camSpeedFromTo.x = config.lookup("camSpeed");
			camRadiusFromTo.x = config.lookup("camRadius");
			camInclinationFromTo.x = config.lookup("camInclination");
			camPhiFromTo.x = config.lookup("camPhi");
			camInclinationFromTo.x *= PI;
			camPhiFromTo.x *= PI;

			

			afactor = config.lookup("afactor");

			gridStartLevel = config.lookup("gridStartLevel");
			gridMinLevel = config.lookup("gridMinLevel");
			gridMaxLevel = config.lookup("gridMaxLevel");
			grid_N = round(pow(2, gridMaxLevel) + 1);
			grid_M = 2 * (grid_N - 1);

			bh_center = { (float) (grid_N - 1) / 2, (grid_N -1)  * 0.95f };

			useAccretionDisk = config.lookup("useAccretionDisk");
			if (useAccretionDisk) {
				useAccretionDiskTexture = config.lookup("useAccretionDiskTexture");
				if (useAccretionDiskTexture) {
					std::string str = config.lookup("accretionDiskTexture");
					accretionDiskTexture = str;
				}

				accretionDiskMaxRadius = config.lookup("accretionDiskMaxRadius");
				accretionDiskMinRadius = config.lookup("accretionDiskMinRadius");
				accretionTemperatureLUTSize = config.lookup("temperatureLUTSize");
				blackholeMass = config.lookup("blackholeMass");
				blackholeAccretion = config.lookup("blackholeAccretionRate");

				n_disk_angles = config.lookup("n_disk_angles");
				n_disk_sample = config.lookup("n_disk_sample");
				max_disk_segments = config.lookup("max_disk_segments");

			}
			
			//If we move allong a path lookup the ending locations
			if (movementMode == 1) {
				camSpeedChange = config.lookup("camSpeedChange");
				if (camSpeedChange) {
					camSpeedFromTo.y = config.lookup("camSpeedTo");
				}

				camRadiusChange = config.lookup("camRadiusChange");
				if (camRadiusChange) {
					camRadiusFromTo.y = config.lookup("camRadiusTo");
				}

				camInclinationChange = config.lookup("camInclinationChange");
				if (camInclinationChange) {
					camInclinationFromTo.y = config.lookup("camInclinationTo");
					camInclinationFromTo.y *= PI;
				}

				camPhiChange = config.lookup("camPhiChange");
				if (camPhiChange) {
					camPhiFromTo.y = config.lookup("camPhiTo");
					camPhiFromTo.y *= PI;
				}
			}
			//If we dont move allong the path set frames and grids to 1
			else {
				nrOfFrames = 1;
				gridNum = 1;
			}

			//If we do free movement lookup the distance required per grid
			if (movementMode == 2) {
				gridDistanceR = config.lookup("gridDistanceR");
				gridDistanceTheta = config.lookup("gridDistanceTheta");
				gridDistanceTheta *= PI;
			}

			
		}
		catch (libconfig::SettingNotFoundException& e) {
			std::cerr << "Incorrect setting(s) in configuration file for setting " << e.getPath() << "." << std::endl;
		}

		if (sphereView) texHeight = (int)floor(texWidth / 2);
	}
};