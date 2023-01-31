#pragma once
#include "../Header files/ImageDistorterCaller.cuh"
#include "../../C++/Header files/IntegrationDefines.h"
#include "./../Header files/AccretionDiskColorComputation.cuh"

#define copyHostToDevice(dev_pointer, host_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpy(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice), errtxt.c_str()); }

#define copyDeviceToHost(host_pointer, dev_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpy(host_pointer,dev_pointer , size, cudaMemcpyDeviceToHost), errtxt.c_str()); }

#define copyHostToDeviceAsync(dev_pointer, host_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpyAsync(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice,stream), errtxt.c_str()); }

#define copyDeviceToHostAsync(host_pointer, dev_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpyAsync(host_pointer, dev_pointer, size, cudaMemcpyDeviceToHost,stream), errtxt.c_str()); }

#define allocate(dev_pointer, size, txt);			  { std::string errtxt = ("Allocation Error " + std::string(txt)); \
														checkCudaStatus(cudaMalloc((void**)&dev_pointer, size), errtxt.c_str()); }

#define callKernel(txt, kernel, blocks, threads, ...);{ cudaEventRecord(start);							\
														kernel <<<blocks, threads>>>(__VA_ARGS__);		\
														cudaEventRecord(stop);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }

#define callKernelAsync(txt, kernel, blocks, threads,shared_mem_size, ...);{ cudaEventRecord(start);							\
														kernel <<<blocks, threads,shared_mem_size,stream>>>(__VA_ARGS__);		\
														cudaEventRecord(stop);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }

CUstream stream;
cudaEvent_t start, stop;

float milliseconds = 0.f;

float3* dev_hashTable = 0;
int2* dev_offsetTable = 0;
int2* dev_hashPosTag = 0;
int2* dev_tableSize = 0;

float3* dev_grid = 0;
float3* dev_grid_2 = 0;
static float3* dev_interpolatedGrid = 0;
int* dev_gridGap = 0;
static float* dev_cameras = 0;
float* dev_camera0 = 0;

float2* dev_viewer = 0;
float4* dev_summedCelestialSky = 0;

unsigned char* dev_blackHoleMask = 0;
float2* dev_blackHoleBorder0 = 0;
float2* dev_blackHoleBorder1 = 0;

float* dev_solidAngles0 = 0;
float* dev_solidAngles1 = 0;

static float* dev_starPositions = 0;
static int2* dev_starCache = 0;
static int* dev_nrOfImagesPerStar = 0;
static float3* dev_starTrails = 0;
static float* dev_starMagnitudes = 0;
float2* dev_gradient = 0;
int* dev_starTree = 0;
int* dev_treeSearch = 0;
uchar3* dev_diffraction = 0;
float3* dev_starLight0 = 0;
float3* dev_starLight1 = 0;

uchar4* dev_outputImage = 0;
uchar4* dev_starImage = 0;

double* pRvs_device;
double* bs_device;
double* qs_device;
double* pThetas_device;
double* theta_device;
double* phi_device;

double* temperatureLUT_device;

cudaError_t CUDA::cleanup() {
	cudaFree(dev_hashTable);
	cudaFree(dev_offsetTable);
	cudaFree(dev_hashPosTag);
	cudaFree(dev_tableSize);

	cudaFree(dev_grid);

	cudaFree(dev_interpolatedGrid);
	cudaFree(dev_gridGap);

	cudaFree(dev_cameras);
	cudaFree(dev_camera0);

	cudaFree(dev_viewer);
	cudaFree(dev_summedCelestialSky);

	cudaFree(dev_blackHoleMask);
	cudaFree(dev_blackHoleBorder0);
	cudaFree(dev_blackHoleBorder1);

	cudaFree(dev_solidAngles0);
	cudaFree(dev_solidAngles1);

	cudaFree(dev_starPositions);
	cudaFree(dev_starCache);
	cudaFree(dev_nrOfImagesPerStar);
	cudaFree(dev_starTrails);
	cudaFree(dev_starMagnitudes);
	cudaFree(dev_gradient);
	cudaFree(dev_starTree);
	cudaFree(dev_treeSearch);
	cudaFree(dev_diffraction);
	cudaFree(dev_starLight0);
	cudaFree(dev_starLight1);

	cudaFree(dev_outputImage);
	cudaFree(dev_starImage);

	cudaFree(pRvs_device);
	cudaFree(bs_device);
	cudaFree(qs_device);
	cudaFree(pThetas_device);
	cudaFree(theta_device);
	cudaFree(phi_device);

	cudaFree(temperatureLUT_device);

	cudaStreamDestroy(stream);

	cudaError_t cudaStatus = cudaDeviceReset();
	return cudaStatus;
}

void CUDA::checkCudaErrors() {
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		//cleanup();
		return;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		//cleanup();
		return;
	}
}

void CUDA::checkCudaStatus(cudaError_t cudaStatus, const char* message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, message);
		printf("\n");
		//cleanup();
	}
}

void CUDA::init() {
	checkCudaStatus(cudaSetDevice(0), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	cudaStreamCreate(&stream);

	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

}


void CUDA::call(std::vector<Grid>& grids_, std::vector<Camera>& cameras_, StarProcessor& stars_, Viewer& view, CelestialSkyProcessor& celestialsky, Parameters& param) {
	std::cout << "Preparing CUDA parameters..." << std::endl;

	CelestialSky celestSky(celestialsky);
	Stars stars(stars_);
	Image image(view);
	Grids grids(grids_, cameras_);
	StarVis starvis(stars, image, param);
	BlackHoleProc bhproc(1000);

	memoryAllocationAndCopy(grids, image, celestSky, stars, bhproc, starvis,param);
	runKernels(grids, image, celestSky, stars, bhproc, starvis, param);
}

void CUDA::allocateGridMemory(size_t size) {
	allocate(pRvs_device, sizeof(double) * size, "Momentum R");
	allocate(bs_device, sizeof(double) * size, "b param");
	allocate(qs_device, sizeof(double) * size, "q param");
	allocate(pThetas_device, sizeof(double) * size, "Momentum theta");
	allocate(theta_device, sizeof(double) * size, "Thetas");
	allocate(phi_device, sizeof(double) * size, "phis");
	checkCudaErrors();
};

template <class T> void CUDA::integrateGrid(const T rV, const T thetaV, const T phiV, std::vector <T>& pRV,
	std::vector <T>& bV, std::vector <T>& qV, std::vector <T>& pThetaV){



	copyHostToDevice(pRvs_device, pRV.data(), pRV.size() * sizeof(T), "pRs");
	copyHostToDevice(bs_device, bV.data(), bV.size() * sizeof(T), "bs");
	copyHostToDevice(qs_device, qV.data(), qV.size() * sizeof(T), "qs");
	copyHostToDevice(pThetas_device, pThetaV.data(), pThetaV.size() * sizeof(T), "pThetaVs");
	checkCudaErrors();


	int threads_per_block = 32;

	int block_size = ceil(pRV.size() / (float)threads_per_block);

	//We can reinterpret_cast since T is either double or float and we reserve space for the larger double type
	callKernel("integrate GPU", metric::integrate_kernel<T>, block_size, threads_per_block,
		rV, thetaV, phiV, reinterpret_cast<T*>(pRvs_device), reinterpret_cast<T*>(bs_device), reinterpret_cast<T*>(qs_device), reinterpret_cast<T*>(pThetas_device), pRV.size());

	copyDeviceToHost(bV.data(), bs_device, bV.size() * sizeof(T), "found theta");
	copyDeviceToHost(qV.data(), qs_device, qV.size() * sizeof(T), "found phi");
	copyDeviceToHost(pThetaV.data(), pThetas_device, pThetaV.size() * sizeof(T), "found r");
	checkCudaErrors();

}

void CUDA::memoryAllocationAndCopy(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis, const Parameters& param) {

	std::cout << "Allocating CUDA memory..." << std::endl;

	// Size parameters for malloc and memcopy
	int treeSize = (1 << (stars.treeLevel + 1)) - 1;

	int imageSize = image.M * image.N;
	int rastSize = (image.M + 1) * (image.N + 1);

	int gridsize = grids.GM * grids.GN1;
	int gridnum = (grids.G > 1) ? 2 : 1;

	int celestSize = celestialSky.rows * celestialSky.cols;

	//Increase memory limits
	size_t size_heap, size_stack;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 67108864);
	cudaDeviceSetLimit(cudaLimitStackSize, 16384);
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);
	//printf("Heap size found to be %d; Stack size found to be %d\n", (int)size_heap, (int)size_stack);

	//allocate(dev_hashTable, grids.hashTableSize * sizeof(float3), "hashTable");
	//allocate(dev_offsetTable, grids.offsetTableSize * sizeof(int2), "offsetTable");
	//allocate(dev_tableSize, grids.G * sizeof(int2), "tableSize");
	//allocate(dev_hashPosTag, grids.hashTableSize * sizeof(int2), "hashPosTag");

	allocate(dev_grid, gridnum * gridsize * sizeof(float3), "grid");
	dev_grid_2 = &dev_grid[gridsize];

	allocate(dev_interpolatedGrid, rastSize * sizeof(float3), "interpolatedGrid");
	allocate(dev_gridGap, rastSize * sizeof(int), "gridGap");

	allocate(dev_cameras, 7 * grids.G * sizeof(float), "cameras");
	allocate(dev_camera0, 7 * sizeof(float), "camera0");

	allocate(dev_blackHoleMask, imageSize * sizeof(unsigned char), "blackHoleMask");
	allocate(dev_blackHoleBorder0, (bhproc.angleNum + 1) * 2 * sizeof(float2), "blackHoleBorder0");
	allocate(dev_blackHoleBorder1, (bhproc.angleNum + 1) * 2 * sizeof(float2), "BlackHOleBorder1");

	allocate(dev_solidAngles0, imageSize * sizeof(float), "solidAngles0");
	allocate(dev_solidAngles1, imageSize * sizeof(float), "solidAngles1");

	allocate(dev_viewer, rastSize * sizeof(float2), "viewer");

	allocate(dev_summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");

	allocate(dev_outputImage, imageSize * sizeof(uchar4), "outputImage");
	allocate(dev_starImage, imageSize * sizeof(uchar4), "starImage");
	allocate(dev_starLight0, imageSize * starvis.diffusionFilter * starvis.diffusionFilter * sizeof(float3), "starLight0");
	allocate(dev_starLight1, imageSize * sizeof(float3), "starLight1");

	allocate(dev_starTrails, imageSize * sizeof(float3), "starTrails");
	allocate(dev_starPositions, stars.starSize * 2 * sizeof(float), "starPositions");
	allocate(dev_starMagnitudes, stars.starSize * 2 * sizeof(float), "starMagnitudes");
	allocate(dev_starCache, 2 * stars.starSize * starvis.trailnum * sizeof(int2), "starCache");
	allocate(dev_nrOfImagesPerStar, stars.starSize * sizeof(int), "nrOfImagesPerStar");
	allocate(dev_diffraction, starvis.diffSize * starvis.diffSize * sizeof(uchar3), "diffraction");
	allocate(dev_starTree, treeSize * sizeof(int), "starTree");
	allocate(dev_treeSearch, starvis.searchNr * imageSize * sizeof(int), "treeSearch");
	allocate(dev_gradient, rastSize * sizeof(float2), "gradient");

	allocate(temperatureLUT_device, param.accretionTemperatureLUTSize * sizeof(double),"temperature table");


	std::cout << "Copying variables into CUDA memory..." << std::endl;

	copyHostToDevice(dev_cameras, grids.camParam, 7 * grids.G * sizeof(float), "cameras");
	copyHostToDevice(dev_viewer, image.viewer, rastSize * sizeof(float2), "viewer");

	copyHostToDevice(dev_blackHoleBorder0, bhproc.bhBorder, (bhproc.angleNum + 1) * 2 * sizeof(float2), "blackHoleBorder0");

	copyHostToDevice(dev_starTree, stars.tree, treeSize * sizeof(int), "starTree");
	copyHostToDevice(dev_starPositions, stars.stars, stars.starSize * 2 * sizeof(float), "starPositions");
	copyHostToDevice(dev_starMagnitudes, stars.magnitude, stars.starSize * 2 * sizeof(float), "starMagnitudes");
	copyHostToDevice(dev_diffraction, starvis.diffraction, starvis.diffSize * starvis.diffSize * sizeof(uchar3), "diffraction");

	copyHostToDevice(dev_summedCelestialSky, celestialSky.summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");

	checkCudaErrors();

	//copyHostToDevice(dev_hit, grids.hit, grids.G * imageSize * sizeof(float2),"hit ");
	std::cout << "Completed CUDA preparation." << std::endl << std::endl;

}


bool map = true;
float hor = 0.0f;
float ver = 0.0f;
//bool redshiftOn = true;
//bool lensingOn = true;

void CUDA::runKernels(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis, const Parameters& param) {
	bool star = param.useStars;

	std::vector<float> cameraUsed(7);
	for (int q = 0; q < 7; q++) cameraUsed[q] = grids.camParam[q];

	

	int threadsPerBlock_32 = 32;
	int numBlocks_starsize = stars.starSize / threadsPerBlock_32 + 1;
	int numBlocks_bordersize = (bhproc.angleNum * 2) / threadsPerBlock_32 + 1;
	int numBlocks_tempLUT = param.accretionTemperatureLUTSize / threadsPerBlock_32 + 1;

	dim3 threadsPerBlock4_4(4, 4);
	dim3 numBlocks_N_M_4_4((image.N - 1) / threadsPerBlock4_4.x + 1, (image.M - 1) / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_N1_M1_4_4(image.N / threadsPerBlock4_4.x + 1, image.M / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_GN_GM_4_4((grids.GN - 1) / threadsPerBlock4_4.x + 1, (grids.GM - 1) / threadsPerBlock4_4.y + 1);

	dim3 threadsPerBlock5_25(5, 25);
	dim3 numBlocks_GN_GM_5_25((grids.GN - 1) / threadsPerBlock5_25.x + 1, (grids.GM - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N_M_5_25((image.N - 1) / threadsPerBlock5_25.x + 1, (image.M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N1_M1_5_25(image.N / threadsPerBlock5_25.x + 1, image.M / threadsPerBlock5_25.y + 1);

	dim3 threadsPerBlock1_24(1, 24);
	dim3 numBlocks_N_M_1_24((image.N - 1) / threadsPerBlock1_24.x + 1, (image.M - 1) / threadsPerBlock1_24.y + 1);


	std::cout << "Running Kernels" << std::endl << std::endl;

	if (grids.G == 1) {
		copyHostToDevice(dev_grid, grids.grid_vectors[0].data(), grids.grid_vectors[0].size() * sizeof(float3), "grid");
		checkCudaErrors();
	}

	float grid_value = 0.f;
	float alpha = 0.f;
	int grid_nr = -1;
	int startframe = 0;

	
	for (int q = 0 + startframe; q < param.nrOfFrames + startframe; q++) {
		float speed = 1.f / cameraUsed[0];
		float offset = PI2 * q / (.25f * speed * image.M);

		//TODO Fix grid_vector for more grids
		if (grids.G > 1) {
			grid_value = fmodf(grid_value, (float)grids.G - 1.f);
			std::cout << "Computing grid: " << grid_value << std::endl;
			alpha = fmodf(grid_value, 1.f);
			if (grid_nr != (int)grid_value) {
				grid_nr = (int)grid_value;
				copyHostToDevice(dev_grid, grids.grid_vectors[grid_nr].data(), grids.grid_vectors[grid_nr].size() * sizeof(float3), "grid");
				checkCudaErrors();
				copyHostToDevice(dev_grid, grids.grid_vectors[grid_nr+1].data(), grids.grid_vectors[grid_nr+1].size() * sizeof(float3), "grid");
				checkCudaErrors();
				callKernel("Find black-hole shadow center", findBhCenter, numBlocks_GN_GM_5_25, threadsPerBlock5_25,
					grids.GM, grids.GN1, dev_grid, dev_blackHoleBorder0);
				checkCudaErrors();
				callKernel("Find black-hole shadow border", findBhBorders, numBlocks_bordersize, threadsPerBlock_32,
					grids.GM, grids.GN1, dev_grid, bhproc.angleNum, dev_blackHoleBorder0);
				checkCudaErrors();
				callKernel("Smoothed shadow border 1/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
					dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);
				checkCudaErrors();
				callKernel("Smoothed shadow border 2/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
					dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);
				checkCudaErrors();
				callKernel("Smoothed shadow border 3/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
					dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);
				checkCudaErrors();
				callKernel("Smoothed shadow border 4/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
					dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);
				checkCudaErrors();
				//displayborders << <dev_angleNum * 2 / tpb + 1, tpb >> >(angleNum, dev_bhBorder, dev_img, image.M);
			}
			callKernel("Update Camera", camUpdate, 1, 8, alpha, grid_nr, dev_cameras, dev_camera0);
			checkCudaErrors();


			checkCudaStatus(cudaMemcpy(&cameraUsed[0], dev_camera0, 7 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host Cam");

			grid_value += .5f;
		}
		cudaEventRecord(start);
		callKernel("Interpolated grid", pixInterpolation, numBlocks_N1_M1_5_25, threadsPerBlock5_25,
			dev_viewer, image.M, image.N, grids.G, dev_interpolatedGrid, dev_grid, grids.GM, grids.GN1,
			hor, ver, dev_gridGap, grids.level, dev_blackHoleBorder0, bhproc.angleNum, alpha);
		checkCudaErrors();

		callKernel("Constructed black-hole shadow mask", findBlackPixels, numBlocks_N_M_5_25, threadsPerBlock5_25,
			dev_interpolatedGrid, image.M, image.N, dev_blackHoleMask);
		checkCudaErrors();

		callKernel("Calculated solid angles", findArea, numBlocks_N_M_5_25, threadsPerBlock5_25,
			dev_interpolatedGrid, image.M, image.N, dev_solidAngles0);
		checkCudaErrors();

		callKernel("Smoothed solid angles horizontally", smoothAreaH, numBlocks_N_M_5_25, threadsPerBlock5_25,
			dev_solidAngles1, dev_solidAngles0, dev_blackHoleMask, dev_gridGap, image.M, image.N);
		checkCudaErrors();


		callKernel("Smoothed solid angles vertically", smoothAreaV, numBlocks_N_M_5_25, threadsPerBlock5_25,
			dev_solidAngles0, dev_solidAngles1, dev_blackHoleMask, dev_gridGap, image.M, image.N);
		checkCudaErrors();



		if (star) {
			callKernel("Cleared star cache", clearArrays, numBlocks_starsize, threadsPerBlock_32,
				dev_nrOfImagesPerStar, dev_starCache, q, starvis.trailnum, stars.starSize);
			checkCudaErrors();

			callKernel("Calculated gradient field for star trails", makeGradField, numBlocks_N1_M1_4_4, threadsPerBlock4_4,
				dev_interpolatedGrid, image.M, image.N, dev_gradient);
			checkCudaErrors();

			callKernel("Distorted star map", distortStarMap, numBlocks_N_M_4_4, threadsPerBlock4_4,
				dev_starLight0, dev_interpolatedGrid, dev_blackHoleMask, dev_starPositions, dev_starTree, stars.starSize,
				dev_camera0, dev_starMagnitudes, stars.treeLevel,
				image.M, image.N, starvis.gaussian, offset, dev_treeSearch, starvis.searchNr, dev_starCache, dev_nrOfImagesPerStar,
				dev_starTrails, starvis.trailnum, dev_gradient, q, dev_viewer, param.useRedshift, param.useLensing, dev_solidAngles0);
			checkCudaErrors();

			callKernel("Summed all star light", sumStarLight, numBlocks_N_M_1_24, threadsPerBlock1_24,
				dev_starLight0, dev_starTrails, dev_starLight1, starvis.gaussian, image.M, image.N, starvis.diffusionFilter);
			checkCudaErrors();

			callKernel("Added diffraction", addDiffraction, numBlocks_N_M_4_4, threadsPerBlock4_4,
				dev_starLight1, image.M, image.N, dev_diffraction, starvis.diffSize);
			checkCudaErrors();

			if (!map) {
				callKernel("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,
					dev_starLight1, dev_outputImage, image.M, image.N);
				checkCudaErrors();

			}
		}

		if (map) {
			callKernel("Distorted celestial sky image", distortEnvironmentMap, numBlocks_N_M_4_4, threadsPerBlock4_4,
				dev_interpolatedGrid, dev_outputImage, dev_blackHoleMask, celestialSky.imsize, image.M, image.N, offset,
				dev_summedCelestialSky, dev_cameras, dev_solidAngles0, dev_viewer, param.useRedshift, param.useLensing);
			checkCudaErrors();

		}

		if (star && map) {
			callKernel("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,
				dev_starLight1, dev_starImage, image.M, image.N);
			checkCudaErrors();

			callKernel("Added distorted star and celestial sky image", addStarsAndBackground, numBlocks_N_M_5_25, threadsPerBlock5_25,
				dev_starImage, dev_outputImage, dev_outputImage, image.M);
		}
		std::cout << std::endl;

		if (param.useAccretionDisk) {
			callKernel("Calculate temperature LUT", createTemperatureTable, numBlocks_tempLUT, threadsPerBlock_32,
				param.accretionTemperatureLUTSize, temperatureLUT_device, (param.accretionDiskMaxRadius - 3) / (param.accretionTemperatureLUTSize-1), param.blackholeMass, param.blackholeAccretion);
			checkCudaErrors();

			callKernel("Add accretion Disk", addAccretionDisk, numBlocks_N_M_4_4, threadsPerBlock4_4,
				dev_interpolatedGrid, dev_outputImage, temperatureLUT_device,(param.accretionDiskMaxRadius/param.accretionTemperatureLUTSize), param.accretionTemperatureLUTSize, dev_blackHoleMask, image.M, image.N);
			checkCudaErrors();
		}

		checkCudaErrors();

		// Copy output vector from GPU buffer to host memory.
		checkCudaStatus(cudaMemcpy(&image.result[0], dev_outputImage, image.N * image.M * sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");
		cv::Mat img = cv::Mat(image.N, image.M, CV_8UC4, (void*)&image.result[0]);
		cv::imwrite(param.getResultFileName(grid_value, q), img, image.compressionParams);


		//Write grid results to file
		std::vector<float3> grid((grids.GM)* (grids.GN1));
		checkCudaStatus(cudaMemcpy(grid.data(), dev_grid, (grids.GN1)* (grids.GM) * sizeof(float3), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");

		for (int i = 0; i < grid.size(); i++) {
			if (grid[i].x != -2.0 && grid[i].x != -1) {
				if (grid[i].z > INFINITY_CHECK) {
					grid[i].z = 0;
					grid[i].x = (grid[i].x / PI);
				}
				else {
					grid[i].z = grid[i].z / (grids.gridStart + q * (param.camRadiusChange ? grids.gridStep : 0));
					grid[i].x = grid[i].x*0 / 2;
				}
				

				grid[i].y = (grid[i].y / PI2)*0;
				grid[i].z;
			}
			
		}

		cv::Mat gridmat = cv::Mat((grids.GN1), (grids.GM), CV_32FC3, (void*)grid.data());
		cv::Mat gridUchar;
		gridmat.convertTo(gridUchar, CV_8UC3, 255.0);
		cv::imwrite(param.getGridResultFileName(grid_value, q, "_grid"), gridUchar, image.compressionParams);
		
		std::vector<float3> interpolated_grid((image.N + 1)* (image.M + 1));
		checkCudaStatus(cudaMemcpy(interpolated_grid.data(), dev_interpolatedGrid, (image.N + 1)* (image.M + 1) * sizeof(float3), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");

		for (int i = 0; i < interpolated_grid.size(); i++) {
			interpolated_grid[i].x = (interpolated_grid[i].x / PI);
			interpolated_grid[i].y = (interpolated_grid[i].y / PI2);

			if (interpolated_grid[i].z == INFINITY) {
				interpolated_grid[i].z = 0;
			}
			else {
				interpolated_grid[i].z = interpolated_grid[i].z / (grids.gridStart + q * (param.camRadiusChange ? grids.gridStep : 0));
			}
		}

		cv::Mat gridIntermat = cv::Mat((image.N + 1), (image.M + 1), CV_32FC3, (void*)interpolated_grid.data());
		cv::Mat gridInterUchar;
		gridIntermat.convertTo(gridInterUchar, CV_8UC3, 255.0);
		cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_interpolated_grid"), gridInterUchar, image.compressionParams);
	}


}
