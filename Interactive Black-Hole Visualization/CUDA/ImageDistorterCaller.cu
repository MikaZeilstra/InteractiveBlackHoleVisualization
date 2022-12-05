#pragma once
#include "ImageDistorterCaller.cuh"
#define copyHostToDevice(dev_pointer, host_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpy(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice), errtxt.c_str()); }

#define allocate(dev_pointer, size, txt);			  { std::string errtxt = ("Allocation Error " + std::string(txt)); \
														checkCudaStatus(cudaMalloc((void**)&dev_pointer, size), errtxt.c_str()); }

#define callKernel(txt, kernel, blocks, threads, ...);{ cudaEventRecord(start);							\
														kernel <<<blocks, threads>>>(__VA_ARGS__);		\
														cudaEventRecord(stop);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }

#define callKernelWithSharedMemory(txt, kernel, blocks, threads, memory_size, ...);{ cudaEventRecord(start);							\
														kernel <<<blocks, threads, memory_size>>>(__VA_ARGS__);		\
														cudaEventRecord(stop);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }

float2* dev_hashTable = 0;
int2* dev_offsetTable = 0;
int2* dev_hashPosTag = 0;
int2* dev_tableSize = 0;

float2* dev_grid = 0;
static float2* dev_interpolatedGrid = 0;
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
	cudaError_t cudaStatus = cudaDeviceReset();
	return cudaStatus;
}

void CUDA::checkCudaErrors() {
	// Check for any errors launching the kernel
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cleanup();
		return;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Device synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
		cleanup();
		return;
	}
}

void CUDA::checkCudaStatus(cudaError_t cudaStatus, const char* message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, message);
		printf("\n");
		cleanup();
	}
}


void CUDA::call(std::vector<Grid>& grids_, std::vector<Camera>& cameras_, StarProcessor& stars_, Viewer& view, CelestialSkyProcessor& celestialsky, Parameters& param) {
	std::cout << "Preparing CUDA parameters..." << std::endl;

	CelestialSky celestSky(celestialsky);
	Stars stars(stars_);
	Image image(view);
	Grids grids(grids_, cameras_);
	StarVis starvis(stars, image, param);
	BlackHoleProc bhproc(1000);
	//setDeviceVariables(grids, image, celestSky, stars);
	checkCudaStatus(cudaSetDevice(0), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	memoryAllocationAndCopy(grids, image, celestSky, stars, bhproc, starvis);
	runKernels(grids, image, celestSky, stars, bhproc, starvis, param);
}

//bool play = false;
//float hor = 0.0f;
//float ver = 0.0f;
/*
#pragma region glutzooi
uchar *sprite = 0;
uchar4 *d_out = 0;
GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint vbo = 0;
GLuint vao = 0;
GLuint tex = 0;     // OpenGL texture object
GLuint diff = 0;
struct cudaGraphicsResource *cuda_pbo_resource;
int g_start_time;
int g_current_frame_number;
bool moving = false;

void render() {
	dim3 threadsPerBlock(TILE_H, TILE_W);
	dim3 numBlocks((dev_N - 1) / threadsPerBlock.x + 1, (dev_M - 1) / threadsPerBlock.y + 1);
	dim3 numBlocksM1(dev_N / threadsPerBlock.x + 1, dev_M / threadsPerBlock.y + 1);
	dim3 simplethreads(5, 25);
	dim3 simpleblocks((dev_N - 1) / simplethreads.x + 1, (dev_M - 1) / simplethreads.y + 1);
	dim3 numBlocks2((dev_GN - 1) / threadsPerBlock.x + 1, (dev_GM - 1) / threadsPerBlock.y + 1);
	dim3 simpleblocks2((dev_GN1 - 1) / simplethreads.x + 1, (dev_GM - 1) / simplethreads.y + 1);
	dim3 interthreads(5, 25);
	dim3 interblocks(dev_N / interthreads.x + 1, dev_M / interthreads.y + 1);
	dim3 testthreads(1, 24);
	dim3 testblocks((dev_N - 1) / testthreads.x + 1, (dev_M - 1) / testthreads.y + 1);

	float speed = 4.f;
	int tpb = 32;
	if (dev_G > 1) {
		prec = fmodf(prec, (float)dev_G - 1.f);
		dev_alpha = fmodf(prec, 1.f);
		//cout << q << " " << 0.2f*prec+5.0f << " " << dev_alpha << endl;
		if (gnr != (int)prec) {
			gnr = (int)prec;

			makeGrid << < numBlocks2, threadsPerBlock >> >(gnr, dev_GM, dev_GN, dev_GN1, dev_grid, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 0, dev_sym);
			makeGrid << < numBlocks2, threadsPerBlock >> >(gnr + 1, dev_GM, dev_GN, dev_GN1, dev_grid, dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 1, dev_sym);

			findBhCenter << < simpleblocks2, simplethreads >> >(dev_GM, dev_GN1, dev_grid, dev_bhBorder);
			findBhBorders << < dev_angleNum * 2 / tpb + 1, tpb >> >(dev_GM, dev_GN1, dev_grid, dev_angleNum, dev_bhBorder);
			smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
			smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
			smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder, dev_bhBorder2, dev_angleNum);
			smoothBorder << <dev_angleNum * 2 / tpb + 1, tpb >> >(dev_bhBorder2, dev_bhBorder, dev_angleNum);
		}
		camUpdate << <1, 8 >> >(dev_alpha, gnr, dev_cam, dev_camIn);
		//prec += .05f;
	}

	pixInterpolation << <interblocks, interthreads >> >(dev_viewthing, dev_M, dev_N, dev_G,
														   dev_in, dev_grid, dev_GM, dev_GN1, hor, ver, dev_gap, dev_gridlvl,
														   dev_bhBorder, dev_angleNum, dev_alpha);

	findBlackPixels << <simpleblocks, simplethreads >> >(dev_in, dev_M, dev_N, dev_bh);
	findArea << <simpleblocks, simplethreads >> > (dev_in, dev_M, dev_N, dev_area);
	smoothAreaH << <simpleblocks, simplethreads >> > (dev_areaSmooth, dev_area, dev_bh, dev_gap, dev_M, dev_N);
	smoothAreaV << <simpleblocks, simplethreads >> > (dev_areaSmooth, dev_area, dev_bh, dev_gap, dev_M, dev_N);

	offset += PI2 / (.25f*speed*dev_M);
	offset = fmodf(offset, PI2);
	if (star) {
		int nb = dev_starSize / tpb + 1;
		clearArrays << < nb, tpb >> > (dev_stnums, dev_stCache, g_current_frame_number, dev_trailnum, dev_starSize);
		makeGradField << <interblocks, interthreads >> > (dev_in, dev_M, dev_N, dev_grad);
		distortStarMap << <numBlocks, threadsPerBlock >> >(dev_temp, dev_in, dev_bh, dev_st, dev_tree, dev_starSize,
														   dev_camIn, dev_mag, dev_treelvl, dev_M, dev_N, dev_step, offset, dev_search,
														   dev_searchNr, dev_stCache, dev_stnums, dev_trail, dev_trailnum,
														   dev_grad, g_current_frame_number, dev_viewthing, dev_lr, dev_area);
		sumStarLight << <testblocks, testthreads >> >(dev_temp, dev_trail, dev_starlight, dev_step, dev_M, dev_N, dev_filterW);
		addDiffraction << <numBlocks, threadsPerBlock >> >(dev_starlight, dev_M, dev_N, dev_diff, dev_diffSize);
		makePix << <simpleblocks, simplethreads >> >(dev_starlight, dev_img2, dev_M, dev_N, dev_hit);

		distortEnvironmentMap << <numBlocks, threadsPerBlock >> >(dev_in, dev_img, dev_bh, dev_imsize, dev_M, dev_N,
																  offset, dev_sumTable, dev_camIn, dev_minmaxnr,
																  dev_minmax, dev_area, dev_viewthing, dev_lr);
		addStarsAndBackground << < simpleblocks, simplethreads >> > (dev_img2, dev_img, d_out, dev_M);
	}
	else {
		distortEnvironmentMap << < numBlocks, threadsPerBlock >> >(dev_in, d_out, dev_bh, dev_imsize, dev_M, dev_N, 
																   offset, dev_sumTable, dev_camIn, dev_minmaxnr, 
																   dev_minmax, dev_area, dev_viewthing, dev_lr);
	}

	cudaError_t cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		glutExit();
	}
	double end_frame_time, end_rendering_time, waste_time;

	// wait until it is time to draw the current frame
	end_frame_time = g_start_time + (g_current_frame_number + 1) * 15.f;
	end_rendering_time = glutGet(GLUT_ELAPSED_TIME);
	waste_time = end_frame_time - end_rendering_time;
	if (waste_time > 0.0) Sleep(waste_time / 1000.);    // sleep parameter should be in seconds
	// update frame number
	g_current_frame_number++;
}

void processNormalKeys(unsigned char key, int x, int y) {
	if (key == 27)
		exit(0);
	else if (key == '=') {
		prec += .1f;
		//cout << " " << dev_G << " " << prec << endl;
		if (prec >= dev_G) prec = 0;
	}
	else if (key == '-') {
		prec -= .1f;
		if (prec < 0) prec = 0;
	}
	else if (key == 'l') {
		dev_lr = !dev_lr;
	}
	else if (key == 's') {
		star = !star;
	}
}

void processSpecialKeys(int key, int x, int y) {
	float maxangle = (PI - dev_viewAngle / (1.f*dev_M) * dev_N) / 2.f;

	switch (key) {
	case GLUT_KEY_UP:
		if (ver + 0.01f <= maxangle) {
			ver += 0.01f;
		}
		break;
	case GLUT_KEY_DOWN:
		if (fabs(ver - 0.01f) <= maxangle) {
			ver -= 0.01f;
		}
		break;
	case GLUT_KEY_LEFT:
		hor += 0.01f;
		break;
	case GLUT_KEY_RIGHT:
		hor -= 0.01f;
		break;
	}
}

void drawTexture() {
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, dev_M, dev_N, 0, GL_RGBA,
		GL_UNSIGNED_BYTE, NULL);
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(0, dev_N);
	glTexCoord2f(1.0f, 1.0f); glVertex2f(dev_M, dev_N);
	glTexCoord2f(1.0f, 0.0f); glVertex2f(dev_M, 0);
	glEnd();

	//glBindVertexArray(vao);
	//glBindTexture(GL_TEXTURE_2D, diff);
	//glEnable(GL_BLEND);
	//glBlendFunc(GL_ONE, GL_ONE);

	//glPointSize(100.0);
	//glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	//glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
	//glEnable(GL_POINT_SPRITE);

	//int diffstars = 1;
	//glDrawArrays(GL_POINTS, 0, diffstars);

	//glBindVertexArray(0);
	//glDisable(GL_BLEND);
	//glDisable(GL_POINT_SPRITE);
	//glDisable(GL_TEXTURE_2D);

}

void display() {
	render();
	drawTexture();
	glutSwapBuffers();
	glutPostRedisplay();
}

void initGLUT(int *argc, char **argv) {
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(dev_M, dev_N);
	glutCreateWindow("whooptiedoe");
	g_start_time = glutGet(GLUT_ELAPSED_TIME);
	g_current_frame_number = 0;
	glewInit();
}

void initTexture() {
	glGenTextures(1, &diff);
	glBindTexture(GL_TEXTURE_2D, diff);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	//int x,y,n;
	//unsigned char *data = stbi_load("../pic/0.png", &x, &y, &n, STBI_rgb);

	//glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8, x, y, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
	//glGenerateMipmap(GL_TEXTURE_2D);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//float sprites[2] = { 700.f, 500.f };
	//glGenBuffers(1, &vbo);
	//glBindBuffer(GL_ARRAY_BUFFER, vbo);
	//glBufferData(GL_ARRAY_BUFFER, 2*sizeof(GLfloat), sprites, GL_STREAM_DRAW);

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);

	//stbi_image_free(data);
}

void initPixelBuffer() {
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*dev_M*dev_N*sizeof(GLubyte), 0, GL_STREAM_DRAW);
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);

	cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL,	cuda_pbo_resource);
	cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void exitfunc() {
	if (pbo) {
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
}

#pragma endregion
*/


/*void CUDA::setDeviceVariables(const Grids& grids, const Image& image, const CelestialSky& celestialSky, const Stars& stars) {
	printf("Setting cuda variables...\n");

	dev_M = image.M;
	dev_N = image.N;
	dev_viewAngle = image.viewAngle;
	dev_minmaxnr = (int)(celestialSky.cols * 0.2f);
	dev_imsize = { celestialSky.rows, celestialSky.cols };

	dev_G = grids.G;
	dev_GM = grids.GM;
	dev_GN = grids.GN;
	dev_gridlvl = grids.gridlvl;

	dev_treelvl = stars.treeLevel;
	dev_starSize = stars.starSize;
	dev_trailnum = 30;
	dev_searchNr = (int)powf(2, stars.treeLevel / 3 * 2);
	dev_step = image.starGaussian;
	dev_filterW = image.starGaussian * 2 + 1;

	dev_angleNum = 1000;
	dev_alpha = 0.f;
	dev_sym = float(grids.GM) / float(grids.GN) > 3 ? 1 : 0;
	dev_GN1 = (dev_sym == 1) ? 2 * grids.GN - 1 : grids.GN;
}*/


void CUDA::memoryAllocationAndCopy(const Grids& grids, const Image& image, const CelestialSky& celestialSky, 
								   const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis) {
	
	std::cout << "Allocating CUDA memory..." << std::endl;

	// Size parameters for malloc and memcopy
	int treeSize = (1 << (stars.treeLevel + 1)) - 1;

	int imageSize = image.M * image.N;
	int rastSize = (image.M+1) * (image.N+1);

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

	allocate(dev_hashTable, grids.hashTableSize * sizeof(float2), "hashTable");
	allocate(dev_offsetTable, grids.offsetTableSize * sizeof(int2), "offsetTable");
	allocate(dev_tableSize, grids.G * sizeof(int2), "tableSize");
	allocate(dev_hashPosTag, grids.hashTableSize * sizeof(int2), "hashPosTag");

	allocate(dev_grid, gridnum * gridsize * sizeof(float2), "grid");
	allocate(dev_interpolatedGrid, rastSize * sizeof(float2), "interpolatedGrid");
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

	std::cout << "Copying variables into CUDA memory..." << std::endl;

	copyHostToDevice(dev_hashTable, grids.hashTables, grids.hashTableSize * sizeof(float2), "hashTable");
	copyHostToDevice(dev_offsetTable, grids.offsetTables, grids.offsetTableSize * sizeof(int2), "offsetTable");
	copyHostToDevice(dev_hashPosTag, grids.hashPosTags, grids.hashTableSize * sizeof(int2), "hashPosTags");
	copyHostToDevice(dev_tableSize, grids.tableSizes, grids.G * sizeof(int2), "tableSizes");

	copyHostToDevice(dev_cameras, grids.camParam, 7 * grids.G * sizeof(float),"cameras");
	copyHostToDevice(dev_viewer, image.viewer, rastSize * sizeof(float2),"viewer");

	copyHostToDevice(dev_blackHoleBorder0, bhproc.bhBorder, (bhproc.angleNum + 1) * 2 * sizeof(float2), "blackHoleBorder0");

	copyHostToDevice(dev_starTree, stars.tree, treeSize * sizeof(int),"starTree");
	copyHostToDevice(dev_starPositions, stars.stars, stars.starSize * 2 * sizeof(float),"starPositions");
	copyHostToDevice(dev_starMagnitudes, stars.magnitude, stars.starSize * 2 * sizeof(float),"starMagnitudes");
	copyHostToDevice(dev_diffraction, starvis.diffraction, starvis.diffSize * starvis.diffSize * sizeof(uchar3),"diffraction");

	copyHostToDevice(dev_summedCelestialSky, celestialSky.summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");

	//copyHostToDevice(dev_hit, grids.hit, grids.G * imageSize * sizeof(float2),"hit ");
	std::cout << "Completed CUDA preparation." << std::endl << std::endl;

}


bool star = true;
bool map = true;
float hor = 0.0f;
float ver = 0.0f;
bool redshiftOn = true;
bool lensingOn = true;

void CUDA::runKernels(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
					  const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis, const Parameters& param) {


	std::vector<float> cameraUsed(7);
	for (int q = 0; q < 7; q++) cameraUsed[q] = grids.camParam[q];
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0.f;

	int threadsPerBlock_32 = 32;
	int numBlocks_starsize = stars.starSize / threadsPerBlock_32 + 1;
	int numBlocks_bordersize = (bhproc.angleNum * 2) / threadsPerBlock_32 + 1;

	dim3 threadsPerBlock4_4(32, 8);
	dim3 numBlocks_N_M_4_4((image.N - 1) / threadsPerBlock4_4.x + 1, (image.M - 1) / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_N1_M1_4_4(image.N / threadsPerBlock4_4.x + 1, image.M / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_GN_GM_4_4((grids.GN - 1) / threadsPerBlock4_4.x + 1, (grids.GM - 1) / threadsPerBlock4_4.y + 1);

	dim3 threadsPerBlock32_4(128, 8);
	dim3 numBlocks_N_M_32_4((image.N - 1) / threadsPerBlock32_4.x + 1, (image.M - 1) / threadsPerBlock32_4.y + 1);
	dim3 numBlocks_N1_M1_32_4(image.N / threadsPerBlock32_4.x + 1, image.M / threadsPerBlock32_4.y + 1);
	dim3 numBlocks_GN_GM_32_4((grids.GN - 1) / threadsPerBlock32_4.x + 1, (grids.GM - 1) / threadsPerBlock32_4.y + 1);

	dim3 threadsPerBlock5_25(5, 25);
	dim3 numBlocks_GN_GM_5_25((grids.GN - 1) / threadsPerBlock5_25.x + 1, (grids.GM - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N_M_5_25((image.N - 1) / threadsPerBlock5_25.x + 1, (image.M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N1_M1_5_25(image.N / threadsPerBlock5_25.x + 1, image.M / threadsPerBlock5_25.y + 1);

	dim3 threadsPerBlock1_24(1, 24);
	dim3 numBlocks_N_M_1_24((image.N - 1) / threadsPerBlock1_24.x + 1, (image.M - 1) / threadsPerBlock1_24.y + 1);


	std::cout << "Running Kernels" << std::endl << std::endl;

	if (grids.G == 1) {
		callKernel("Expanded grid", makeGrid, numBlocks_GN_GM_5_25, threadsPerBlock5_25,
					0, grids.GM, grids.GN, grids.GN1, dev_grid, 
					dev_hashTable, dev_hashPosTag, dev_offsetTable, dev_tableSize, 0, grids.sym);
		dev_camera0 = dev_cameras;
		checkCudaErrors();
	}

	float grid_value = 0.f;
	float alpha = 0.f;
	int grid_nr = -1;
	int startframe = 0;

	for (int q = 0 + startframe; q < param.nrOfFrames + startframe; q++) {
		float speed = 1.f/ cameraUsed[0];
		float offset = PI2*q / (.25f*speed*image.M);

		if (grids.G > 1) {
			grid_value = fmodf(grid_value, (float)grids.G - 1.f);
			std::cout << "Computing grid: " << grid_value << std::endl;
			alpha = fmodf(grid_value, 1.f);
			if (grid_nr != (int)grid_value) {
				grid_nr = (int)grid_value;
			
				callKernel("Made grid A", makeGrid, numBlocks_GN_GM_5_25, threadsPerBlock5_25,
							grid_nr, grids.GM, grids.GN, grids.GN1, dev_grid, dev_hashTable, dev_hashPosTag, 
							dev_offsetTable, dev_tableSize, 0, grids.sym);
				
				callKernel("Made grid B", makeGrid, numBlocks_GN_GM_5_25, threadsPerBlock5_25,
							grid_nr + 1, grids.GM, grids.GN, grids.GN1, dev_grid, dev_hashTable, dev_hashPosTag,
							dev_offsetTable, dev_tableSize, 1, grids.sym);

				callKernel("Find black-hole shadow center", findBhCenter, numBlocks_GN_GM_5_25, threadsPerBlock5_25,
							grids.GM, grids.GN1, dev_grid, dev_blackHoleBorder0);
				
				callKernel("Find black-hole shadow border", findBhBorders, numBlocks_bordersize, threadsPerBlock_32,
							grids.GM, grids.GN1, dev_grid, bhproc.angleNum, dev_blackHoleBorder0);
		
				callKernel("Smoothed shadow border 1/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
							dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);

				callKernel("Smoothed shadow border 2/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
							dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);

				callKernel("Smoothed shadow border 3/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
							dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);

				callKernel("Smoothed shadow border 4/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,
							dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);

				//displayborders << <dev_angleNum * 2 / tpb + 1, tpb >> >(angleNum, dev_bhBorder, dev_img, image.M);
			}
			callKernel("Update Camera", camUpdate, 1, 8, alpha, grid_nr, dev_cameras, dev_camera0);

			checkCudaStatus(cudaMemcpy(&cameraUsed[0], dev_camera0, 7 * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host Cam");

			grid_value += .5f;
		}

		cudaEventRecord(start);
		callKernel("Interpolated grid", pixInterpolation, numBlocks_N1_M1_5_25, threadsPerBlock5_25,
					dev_viewer, image.M, image.N, grids.G, dev_interpolatedGrid, dev_grid, grids.GM, grids.GN1, 
					hor, ver, dev_gridGap, grids.level, dev_blackHoleBorder0, bhproc.angleNum, alpha);

		callKernel("Constructed black-hole shadow mask", findBlackPixels, numBlocks_N_M_5_25, threadsPerBlock5_25,
					dev_interpolatedGrid, image.M, image.N, dev_blackHoleMask);

		callKernel("Calculated solid angles", findArea, numBlocks_N_M_5_25, threadsPerBlock5_25,
					dev_interpolatedGrid, image.M, image.N, dev_solidAngles0);

		callKernel("Smoothed solid angles horizontally", smoothAreaH, numBlocks_N_M_5_25, threadsPerBlock5_25,
					dev_solidAngles1, dev_solidAngles0, dev_blackHoleMask, dev_gridGap, image.M, image.N);

		callKernel("Smoothed solid angles vertically", smoothAreaV, numBlocks_N_M_5_25, threadsPerBlock5_25,
					dev_solidAngles0, dev_solidAngles1, dev_blackHoleMask, dev_gridGap, image.M, image.N);


		if (star) {
			callKernel("Cleared star cache", clearArrays, numBlocks_starsize, threadsPerBlock_32,
						dev_nrOfImagesPerStar, dev_starCache, q, starvis.trailnum, stars.starSize);

			callKernel("Calculated gradient field for star trails", makeGradField, numBlocks_N1_M1_4_4, threadsPerBlock4_4,
						dev_interpolatedGrid, image.M, image.N, dev_gradient);

			//Check if we can use shared memory for stars
			cudaMemset(dev_starLight0, 0, sizeof(float3)*(starvis.gaussian * 2 + 1) * (starvis.gaussian * 2 + 1) * image.M * image.N);

			if (stars.starSize * sizeof(float) * 2 < 49152) {
				callKernelWithSharedMemory("Distorted star map", distortStarMapSharedMem, numBlocks_N_M_4_4, threadsPerBlock4_4, stars.starSize * 2 * sizeof(float),
					dev_starLight0, dev_interpolatedGrid, dev_blackHoleMask, dev_starPositions, dev_starTree, stars.starSize,
					dev_camera0, dev_starMagnitudes, stars.treeLevel,
					image.M, image.N, starvis.gaussian, offset, dev_treeSearch, starvis.searchNr, dev_starCache, dev_nrOfImagesPerStar,
					dev_starTrails, starvis.trailnum, dev_gradient, q, dev_viewer, redshiftOn, lensingOn, dev_solidAngles0);
			}
			else {
				callKernel("Distorted star map", distortStarMap, numBlocks_N_M_4_4, threadsPerBlock4_4,
					dev_starLight0, dev_interpolatedGrid, dev_blackHoleMask, dev_starPositions, dev_starTree, stars.starSize,
					dev_camera0, dev_starMagnitudes, stars.treeLevel,
					image.M, image.N, starvis.gaussian, offset, dev_treeSearch, starvis.searchNr, dev_starCache, dev_nrOfImagesPerStar,
					dev_starTrails, starvis.trailnum, dev_gradient, q, dev_viewer, redshiftOn, lensingOn, dev_solidAngles0);
			}
			

			callKernel("Summed all star light", sumStarLight, numBlocks_N_M_1_24, threadsPerBlock1_24,
						dev_starLight0, dev_starTrails, dev_starLight1, starvis.gaussian, image.M, image.N, starvis.diffusionFilter);

			callKernel("Added diffraction", addDiffraction, numBlocks_GN_GM_32_4, threadsPerBlock32_4,
						dev_starLight1, dev_starLight0, image.M, image.N, dev_diffraction, starvis.diffSize);

			if (!map) {
				callKernel("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,
						dev_starLight0, dev_outputImage, image.M, image.N);
			}
		}

		if (map) {
			callKernel("Distorted celestial sky image", distortEnvironmentMap, numBlocks_N_M_4_4, threadsPerBlock4_4,
						dev_interpolatedGrid, dev_outputImage, dev_blackHoleMask, celestialSky.imsize, image.M, image.N, offset, 
						dev_summedCelestialSky, dev_cameras, dev_solidAngles0, dev_viewer, redshiftOn, lensingOn);
		}

		if (star && map) {
			callKernel("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,
						dev_starLight0, dev_starImage, image.M, image.N);

			callKernel("Added distorted star and celestial sky image", addStarsAndBackground, numBlocks_N_M_5_25, threadsPerBlock5_25,
						dev_starImage, dev_outputImage, dev_outputImage, image.M);
		}
		std::cout << std::endl;

		checkCudaErrors();

		// Copy output vector from GPU buffer to host memory.
		checkCudaStatus(cudaMemcpy(&image.result[0], dev_outputImage, image.N * image.M *  sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy failed! Dev to Host");
		cv::Mat img = cv::Mat(image.N, image.M, CV_8UC4, (void*)&image.result[0]);
		cv::imwrite(param.getResultFileName(grid_value, q), img, image.compressionParams);
	}

}
