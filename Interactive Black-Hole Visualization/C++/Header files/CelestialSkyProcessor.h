#pragma once
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "Const.h"
#include "Parameters.h"
#include <fstream>
#include <iomanip>

class CelestialSkyProcessor {
public:
	template < class Archive >
	void serialize(Archive& ar) {
		ar(summedImageVec, cols, rows);
	}

	std::vector<float> summedImageVec;
	int rows;
	int cols;

	CelestialSkyProcessor() {};

	CelestialSkyProcessor(Parameters& param) {
		//size_t lastindex = param.celestialSkyImg.find_last_of(".");
		//std::string rawname = param.celestialSkyImg.substr(0, lastindex);
		//std::string searchSummed = param.getCelestialSummedFolder() + rawname + ".tiff";

		std::ifstream f(param.getCelestialSkyFolder() + param.celestialSkyImg);
		if (f.good()) {
			f.close();
			cv::Mat celestialSky = cv::imread(param.getCelestialSkyFolder() + param.celestialSkyImg); 
			rows = celestialSky.rows;
			cols = celestialSky.cols;

			summedImageVec.resize(4 * rows * cols);
			createSummedImage(celestialSky);
			//summedImageVec.assign(temp.data, temp.data + temp.total() * temp.channels());
		}
		else {
			std::cout << "Could not open image " << param.celestialSkyImg << std::endl;
			exit(1);
			//celestialSky = 
			//rows = celestialSky.rows;
			//cols = celestialSky.cols;

			//std::cout << "Computing summed celestial sky image..." << std::endl;
			//summedImageVec.resize(4 * rows * cols);
			//createSummedImage();

			//std::cout << "Writing summed celestial sky image..." << std::endl;
			//std::vector<int> compressionParams = { cv::IMWRITE_PNG_COMPRESSION, 0 };
			//cv::Mat temp = cv::Mat(rows, cols, CV_32FC4, (void*)&summedImageVec[0]);

			//cv::imwrite(searchSummed, temp);// , compressionParams);
		}

	}

	float calcArea(float t[4], float p[4]) {
		float x[4], y[4], z[4];
		#pragma unroll
		for (int q = 0; q < 4; q++) {
			float sint = sinf(t[q]);
			x[q] = sint * cosf(p[q]);
			y[q] = sint * sinf(p[q]);
			z[q] = cosf(t[q]);
		}
		float dotpr1 = 1.f;
		dotpr1 += x[0] * x[2] + y[0] * y[2] + z[0] * z[2];
		float dotpr2 = dotpr1;
		dotpr1 += x[2] * x[1] + y[2] * y[1] + z[2] * z[1];
		dotpr1 += x[0] * x[1] + y[0] * y[1] + z[0] * z[1];
		dotpr2 += x[0] * x[3] + y[0] * y[3] + z[0] * z[3];
		dotpr2 += x[2] * x[3] + y[2] * y[3] + z[2] * z[3];
		float triprod1 = fabsf(x[0] * (y[1] * z[2] - y[2] * z[1]) -
			y[0] * (x[1] * z[2] - x[2] * z[1]) +
			z[0] * (x[1] * y[2] - x[2] * y[1]));
		float triprod2 = fabsf(x[0] * (y[2] * z[3] - y[3] * z[2]) -
			y[0] * (x[2] * z[3] - x[3] * z[2]) +
			z[0] * (x[2] * y[3] - x[3] * y[2]));
		float area = 2.f * (atanf(triprod1 / dotpr1) + atanf(triprod2 / dotpr2));
		return area;
	}

	void createSummedImage(cv::Mat celestialSky) {
		//cv::Mat sum(celestialSky.rows, celestialSky.cols, CV_32FC4);
		unsigned char* imgvalues = (unsigned char*) celestialSky.data;
		// Summed image
		float pxsz = PI2 / (1.f * cols);
		float phi[4] = { 0, 0, pxsz, pxsz };

		//#pragma omp parallel for
		for (int q = 0; q < cols; q++) {
			cv::Vec4f col_sum = { 0.f, 0.f, 0.f, 0.f };

			for (int p = 0; p < rows; p++) {
				float theta[4] = { pxsz * (p + 1), pxsz * p, pxsz * p, pxsz * (p + 1) };
				float area = calcArea(theta, phi);

				for (int i = 0; i < 3; i++) {
					unsigned char pixval = imgvalues[3 * (p * cols + q) + i];// celestialSky.at<cv::Vec3b>(p, q)[i];
					col_sum.val[i] += powf(pixval, 2.2f) * area;
					summedImageVec[4 * (p * cols + q) + i] = col_sum.val[i];
				}
				col_sum.val[3] += area;
				summedImageVec[4 * (p * cols + q) + 3]  = col_sum.val[3];
			}
		}

	}


};