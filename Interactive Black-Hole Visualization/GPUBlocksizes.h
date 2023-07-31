#pragma once

#include "./C++/Header files/Parameters.h"

class gpuBlocksize {

	gpuBlocksize(Parameters& param) {
		numBlocks_starsize = stars.starSize / threadsPerBlock_32 + 1;
		numBlocks_bordersize = (param.n_black_hole_angles * 2) / threadsPerBlock_32 + 1;
		numBlocks_tempLUT = param.accretionTemperatureLUTSize / threadsPerBlock_32 + 1;
		numBlocks_disk_edges = (param.n_disk_angles) / threadsPerBlock_32 + 1;
	}

public:
	int threadsPerBlock_32 = 32;
	int numBlocks_starsize;
	int numBlocks_bordersize;
	int numBlocks_tempLUT;
	int numBlocks_disk_edges;

	dim3 threadsPerBlock4_4(4, 4);
	dim3 numBlocks_N_M_4_4((image.N - 1) / threadsPerBlock4_4.x + 1, (image.M - 1) / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_N1_M1_4_4(image.N / threadsPerBlock4_4.x + 1, image.M / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_GN_GM_4_4((param.grid_N - 1) / threadsPerBlock4_4.x + 1, (param.grid_M - 1) / threadsPerBlock4_4.y + 1);

	dim3 threadsPerBlock5_25(5, 25);
	dim3 numBlocks_GN_GM_5_25((param.grid_N - 1) / threadsPerBlock5_25.x + 1, (param.grid_M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N_M_5_25((image.N - 1) / threadsPerBlock5_25.x + 1, (image.M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N1_M1_5_25(image.N / threadsPerBlock5_25.x + 1, image.M / threadsPerBlock5_25.y + 1);

	dim3 threadsPerBlock1_24(1, 24);
	dim3 numBlocks_N_M_1_24((image.N - 1) / threadsPerBlock1_24.x + 1, (image.M - 1) / threadsPerBlock1_24.y + 1);

};