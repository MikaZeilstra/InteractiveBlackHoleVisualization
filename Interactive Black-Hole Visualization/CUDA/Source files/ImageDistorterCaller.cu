#pragma once
#include "../Header files/ImageDistorterCaller.cuh"
#include "../../C++/Header files/IntegrationDefines.h"
#include "./../Header files/AccretionDiskColorComputation.cuh"

#define STB_IMAGE_IMPLEMENTATION

#include <fstream>

#include "glad.h"
#include <glfw3.h>

#include <cuda_gl_interop.h>

#include <glm/gtc/type_ptr.hpp>

#include <stbi_image.h>

/// <summary>
/// Checks if there was an error generated on any previous opengl call and prints the error. Also prints the statement this function was called with (not nececerally the cause of the error). 
/// </summary>
/// <param name="stmt">Statement to blame on error</param>
/// <param name="fname">Filename of the statement</param>
/// <param name="line">Line number of the statement</param>
void CheckOpenGLError(const char* stmt, const char* fname, int line)
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		printf("OpenGL error %03x, at %s:%i - for %s\n", err, fname, line, stmt);
		abort();
	}
}

//Macro to automatically check for opengl error after a call. Only does things in the debug build.
#ifdef _DEBUG
#define GL_CHECK(stmt) do { \
            stmt; \
            CheckOpenGLError(#stmt, __FILE__, __LINE__); \
        } while (0)
#else
#define GL_CHECK(stmt) stmt
#endif


/*
#define copyHostToDevice(dev_pointer, host_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpy(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice), errtxt.c_str()); }

#define copyDeviceToHost(host_pointer, dev_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpy(host_pointer,dev_pointer , size, cudaMemcpyDeviceToHost), errtxt.c_str()); }
																 */
#define copyHostToDeviceAsync(dev_pointer, host_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpyAsync(dev_pointer, host_pointer, size, cudaMemcpyHostToDevice,stream), errtxt.c_str()); }

#define copyDeviceToHostAsync(host_pointer, dev_pointer, size, txt) { std::string errtxt = ("Host to Device copy Error " + std::string(txt)); \
																 checkCudaStatus(cudaMemcpyAsync(host_pointer, dev_pointer, size, cudaMemcpyDeviceToHost,stream), errtxt.c_str()); }

#define allocate(dev_pointer, size, txt);			  { std::string errtxt = ("Allocation Error " + std::string(txt)); \
														checkCudaStatus(cudaMalloc((void**)&dev_pointer, size), errtxt.c_str()); }

/*
#define callKernel(txt, kernel, blocks, threads, ...);{ cudaEventRecord(start);							\
														kernel <<<blocks, threads>>>(__VA_ARGS__);		\
														cudaEventRecord(stop);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }
													  */
#ifdef _DEBUG
	#define callKernelAsync(txt, kernel, blocks, threads,shared_mem_size, ...);{ cudaEventRecord(start,stream);							\
														kernel <<<blocks, threads,shared_mem_size,stream>>>(__VA_ARGS__);		\
														cudaEventRecord(stop,stream);							\
														cudaEventSynchronize(stop);						\
														cudaEventElapsedTime(&milliseconds, start, stop); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }
#else
	#define callKernelAsync(txt, kernel, blocks, threads,shared_mem_size, ...);{ 							\
														kernel <<<blocks, threads,shared_mem_size,stream>>>(__VA_ARGS__);		\
													  }
#endif // DEBUG



CUstream stream;
cudaEvent_t start, stop;

float milliseconds = 0.f;

float3* dev_hashTable = 0;
int2* dev_offsetTable = 0;
int2* dev_hashPosTag = 0;
int2* dev_tableSize = 0;

float4* dev_grid = 0;
float4* dev_grid_2 = 0;
static float4* dev_interpolatedGrid = 0;
int* dev_gridGap = 0;
static float* dev_cameras = 0;
float* dev_camera0 = 0;

float2* dev_viewer = 0;
float4* dev_summedCelestialSky = 0;

unsigned char* dev_blackHoleMask = 0;
unsigned char* dev_diskMask = 0;
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
float3* dev_accretionDiskTexture = 0;

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
	cudaFree(dev_diskMask);
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

 	cudaFree(dev_starImage);

	cudaFree(pRvs_device);
	cudaFree(bs_device);
	cudaFree(qs_device);
	cudaFree(pThetas_device);
	cudaFree(theta_device);
	cudaFree(phi_device);

	cudaFree(temperatureLUT_device);
	cudaFree(dev_accretionDiskTexture);

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
		fprintf(stderr, "%s, %s", message, cudaGetErrorString(cudaStatus));
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


void CUDA::call(std::vector<Grid>& grids_, std::vector<Camera>& cameras_, StarProcessor& stars_, Viewer& view, CelestialSkyProcessor& celestialsky, Texture& accretionTexture, Parameters& param) {
	std::cout << "Preparing CUDA parameters..." << std::endl;

	CelestialSky celestSky(celestialsky);
	Stars stars(stars_);
	Image image(view);
	Grids grids(grids_, cameras_);
	StarVis starvis(stars, image, param);
	BlackHoleProc bhproc(1000);

	memoryAllocationAndCopy(grids, image, celestSky, stars, bhproc, starvis, accretionTexture,param);
	runKernels(grids, image, celestSky, stars, bhproc, starvis, accretionTexture, param);
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



	copyHostToDeviceAsync(pRvs_device, pRV.data(), pRV.size() * sizeof(T), "pRs");
	copyHostToDeviceAsync(bs_device, bV.data(), bV.size() * sizeof(T), "bs");
	copyHostToDeviceAsync(qs_device, qV.data(), qV.size() * sizeof(T), "qs");
	copyHostToDeviceAsync(pThetas_device, pThetaV.data(), pThetaV.size() * sizeof(T), "pThetaVs");


	int threads_per_block = 32;

	int block_size = ceil(pRV.size() / (float)threads_per_block);

	//We can reinterpret_cast since T is either double or float and we reserve space for the larger double type
	callKernelAsync("integrate GPU", metric::integrate_kernel<T>, block_size, threads_per_block, 0,
		rV, thetaV, phiV, reinterpret_cast<T*>(pRvs_device), reinterpret_cast<T*>(bs_device), reinterpret_cast<T*>(qs_device), reinterpret_cast<T*>(pThetas_device), pRV.size());

	copyDeviceToHostAsync(bV.data(), bs_device, bV.size() * sizeof(T), "found theta");
	copyDeviceToHostAsync(qV.data(), qs_device, qV.size() * sizeof(T), "found phi");
	copyDeviceToHostAsync(pThetaV.data(), pThetas_device, pThetaV.size() * sizeof(T), "found r");
	copyDeviceToHostAsync(pRV.data(), pRvs_device, pRV.size() * sizeof(T), "found r");
	
	cudaStreamSynchronize(stream);
}



void CUDA::memoryAllocationAndCopy(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis,const Texture accretionTexture, const Parameters& param) {

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


	allocate(dev_grid, gridnum * gridsize * sizeof(float4), "grid");
	dev_grid_2 = &dev_grid[gridsize];

	allocate(dev_interpolatedGrid, rastSize * sizeof(float4), "interpolatedGrid");
	allocate(dev_gridGap, rastSize * sizeof(int), "gridGap");

	allocate(dev_cameras, 10 * grids.G * sizeof(float), "cameras");
	allocate(dev_camera0, 10 * sizeof(float), "camera0");

	allocate(dev_blackHoleMask, imageSize * sizeof(unsigned char), "blackHoleMask");
	allocate(dev_diskMask, imageSize * sizeof(unsigned char), "blackHoleMask");
	allocate(dev_blackHoleBorder0, (bhproc.angleNum + 1) * 2 * sizeof(float2), "blackHoleBorder0");
	allocate(dev_blackHoleBorder1, (bhproc.angleNum + 1) * 2 * sizeof(float2), "BlackHOleBorder1");

	allocate(dev_solidAngles0, imageSize * sizeof(float), "solidAngles0");
	allocate(dev_solidAngles1, imageSize * sizeof(float), "solidAngles1");

	allocate(dev_viewer, rastSize * sizeof(float2), "viewer");

	allocate(dev_summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");

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
	allocate(dev_outputImage, image.N * image.M * sizeof(uchar4),"output")


	allocate(temperatureLUT_device, param.accretionTemperatureLUTSize * sizeof(double),"temperature table");
	allocate(dev_accretionDiskTexture, accretionTexture.width * accretionTexture.height * sizeof(float3), "AccretionTexture");

	std::cout << "Copying variables into CUDA memory..." << std::endl;

	copyHostToDeviceAsync(dev_cameras, grids.camParam, 10 * grids.G * sizeof(float), "cameras");
	copyHostToDeviceAsync(dev_viewer, image.viewer, rastSize * sizeof(float2), "viewer");

	copyHostToDeviceAsync(dev_blackHoleBorder0, bhproc.bhBorder, (bhproc.angleNum + 1) * 2 * sizeof(float2), "blackHoleBorder0");

	copyHostToDeviceAsync(dev_starTree, stars.tree, treeSize * sizeof(int), "starTree");
	copyHostToDeviceAsync(dev_starPositions, stars.stars, stars.starSize * 2 * sizeof(float), "starPositions");
	copyHostToDeviceAsync(dev_starMagnitudes, stars.magnitude, stars.starSize * 2 * sizeof(float), "starMagnitudes");
	copyHostToDeviceAsync(dev_diffraction, starvis.diffraction, starvis.diffSize * starvis.diffSize * sizeof(uchar3), "diffraction");

	copyHostToDeviceAsync(dev_summedCelestialSky, celestialSky.summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");
	copyHostToDeviceAsync(dev_accretionDiskTexture, accretionTexture.summed.data(), accretionTexture.height * accretionTexture.width * sizeof(float3), "accretionTexture");

	cudaStreamSynchronize(stream);

	//copyHostToDevice(dev_hit, grids.hit, grids.G * imageSize * sizeof(float2),"hit ");
	std::cout << "Completed CUDA preparation." << std::endl << std::endl;

}


bool map = true;
float hor = 0.0f;
float ver = 0.0f;
//bool redshiftOn = true;
//bool lensingOn = true;

void CUDA::runKernels(const Grids& grids, const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const BlackHoleProc& bhproc, const StarVis& starvis, const Texture& accretionDiskTexture,  const Parameters& param) {
	bool star = param.useStars;

	std::vector<float> cameraUsed(10);
	for (int q = 0; q < 10; q++) cameraUsed[q] = grids.camParam[q];

	

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
		copyHostToDeviceAsync(dev_grid, grids.grid_vectors[0].data(), grids.grid_vectors[0].size() * sizeof(float4), "grid");
	}

	float grid_value = 0.f;
	float alpha = 0.f;
	int grid_nr = -1;
	int startframe = 0;

	int q = startframe;
	
	std::vector<float4> grid((grids.GM) * (grids.GN1));
	std::vector<float4> depth((image.N + 1) * (image.M + 1), { 0,0,0,1 });
	std::vector<float4> interpolated_grid((image.N + 1) * (image.M + 1));
	std::vector<float> area((image.N) * (image.M));

	ViewCamera* viewer = glfw_setup(param.windowWidth, param.windowHeight);

	//Setup pointers for openGL objects
	GLuint gl_Tex;
	GLuint gl_PBO;
	GLuint VAO;

	//Cuda OpenGl interop object pointer
	cudaGraphicsResource* cuda_pbo_resource;


	//Reserve texture object in opengl to use in the shaders
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &gl_Tex);

	glBindTexture(GL_TEXTURE_2D, gl_Tex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.M, image.N, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0); // no upload, just reserve 
	glBindTexture(GL_TEXTURE_2D, 0); // unbind
	
	//Reserve PBO spaCe
	glGenBuffers(1, &gl_PBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, image.N *  image.M * sizeof(uchar4), 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); //unbind


	//Register the PBO to cuda so we can write to it
	checkCudaStatus(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, gl_PBO,
		cudaGraphicsMapFlagsWriteDiscard), "register");

	//Setup screen covering triangles into a VBO
	float vertices[] = {
	 -1.0f,  -1.0f, 
	 -1.0f, 1.0f, 
	 1.0f, -1.0f,
	 1.0f, 1.0f
	};
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);



	//Compile vertexShader
	std::string vertexShaderText = readFile("../Resources/Shaders/VertexShader.vert");
	const char* vertexPointer = vertexShaderText.c_str();
	GLuint vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexPointer, NULL);
	glCompileShader(vertexShader);

	//Compile fragment shader
	std::string fragShaderText = readFile("../Resources/Shaders/FragmentShader.frag");
	const char* fragPointer = fragShaderText.c_str();
	GLuint fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragPointer, NULL);
	glCompileShader(fragmentShader);


	//Create shader program
	GLuint shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glUseProgram(shaderProgram);

	//Setup placeholder for the number of bytes return 
	size_t num_bytes;

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0); // << default texture object

	std::chrono::steady_clock::time_point frame_start_time;


	while(q < param.nrOfFrames + startframe && !glfwWindowShouldClose(viewer->get_window())) {
		frame_start_time = std::chrono::high_resolution_clock::now();


		//Map the PBO to cuda and set the outputimage pointer to that location
		checkCudaStatus( cudaGraphicsMapResources(1, &cuda_pbo_resource, 0),"map_resource");
		checkCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_outputImage, &num_bytes, cuda_pbo_resource),"get_pointer");
		

		float speed = 1.f / cameraUsed[0];
		
		//Calculate phi offset due to camera movement
		float camera_phi_offset = PI2 * q / (.25f * speed * image.M);

		//TODO Fix grid_vector for more grids
		if (grids.G > 1) {
			grid_value = fmodf(grid_value, (float)grids.G - 1.f);
			std::cout << "Computing grid: " << grid_value << std::endl;
			alpha = fmodf(grid_value, 1.f);
			if (grid_nr != (int)grid_value) {
				grid_nr = (int)grid_value;
				copyHostToDeviceAsync(dev_grid, grids.grid_vectors[grid_nr].data(), grids.grid_vectors[grid_nr].size() * sizeof(float4), "grid");
				//checkCudaErrors();
				copyHostToDeviceAsync(dev_grid_2, grids.grid_vectors[grid_nr+1].data(), grids.grid_vectors[grid_nr+1].size() * sizeof(float4), "grid");
				
				callKernelAsync("Find black-hole shadow center", findBhCenter, numBlocks_GN_GM_5_25, threadsPerBlock5_25,0,
					grids.GM, grids.GN1, dev_grid, dev_blackHoleBorder0);

				callKernelAsync("Find black-hole shadow border", findBhBorders, numBlocks_bordersize, threadsPerBlock_32,0,
					grids.GM, grids.GN1, dev_grid, bhproc.angleNum, dev_blackHoleBorder0);
				callKernelAsync("Smoothed shadow border 1/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,0,
					dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);
				callKernelAsync("Smoothed shadow border 2/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,0,
					dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);
				callKernelAsync("Smoothed shadow border 3/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,0,
					dev_blackHoleBorder0, dev_blackHoleBorder1, bhproc.angleNum);
				callKernelAsync("Smoothed shadow border 4/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32,0,
					dev_blackHoleBorder1, dev_blackHoleBorder0, bhproc.angleNum);
				//displayborders << <dev_angleNum * 2 / tpb + 1, tpb >> >(angleNum, dev_bhBorder, dev_img, image.M);
			}
			callKernelAsync("Update Camera", camUpdate,0, 1, 8, alpha, grid_nr, dev_cameras, dev_camera0);


			cudaMemcpyAsync(&cameraUsed[0], dev_camera0, 10 * sizeof(float), cudaMemcpyDeviceToHost, stream);

			grid_value += .5f;
		}
		//cudaEventRecord(start);

		callKernelAsync("Interpolated grid", pixInterpolation,numBlocks_N1_M1_5_25, threadsPerBlock5_25, 0,
			dev_viewer, image.M, image.N, grids.G, dev_interpolatedGrid, dev_grid, grids.GM, grids.GN1,
			hor, ver, dev_gridGap, grids.level, dev_blackHoleBorder0, bhproc.angleNum, alpha);


		callKernelAsync("Constructed black-hole shadow mask", findBlackPixels, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_interpolatedGrid, image.M, image.N, dev_blackHoleMask);


		callKernelAsync("Constructed disk mask", makeDiskCheck, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_interpolatedGrid, dev_diskMask, image.M, image.N);


		
		callKernelAsync("Calculated solid angles", findArea, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_interpolatedGrid, image.M, image.N, dev_solidAngles0,dev_camera0, param.accretionDiskMaxRadius, dev_diskMask);

		
		

		
		callKernelAsync("Smoothed solid angles horizontally", smoothAreaH, numBlocks_N_M_5_25, threadsPerBlock5_25,0,
			dev_solidAngles1, dev_solidAngles0, dev_blackHoleMask, dev_gridGap, image.M, image.N, dev_diskMask);
#ifdef _DEBUG
		checkCudaErrors();
#endif // _DEBUG



		callKernelAsync("Smoothed solid angles vertically", smoothAreaV, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_solidAngles0, dev_solidAngles1, dev_blackHoleMask, dev_gridGap, image.M, image.N, dev_diskMask);
		
		
		


		if (star) {
			callKernelAsync("Cleared star cache", clearArrays, numBlocks_starsize, threadsPerBlock_32,0,
				dev_nrOfImagesPerStar, dev_starCache, q, starvis.trailnum, stars.starSize);

			callKernelAsync("Calculated gradient field for star trails", makeGradField, numBlocks_N1_M1_4_4, threadsPerBlock4_4,0,
				dev_interpolatedGrid, image.M, image.N, dev_gradient);

			callKernelAsync("Distorted star map", distortStarMap, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
				dev_starLight0, dev_interpolatedGrid, dev_blackHoleMask, dev_starPositions, dev_starTree, stars.starSize,
				dev_camera0, dev_starMagnitudes, stars.treeLevel,
				image.M, image.N, starvis.gaussian, camera_phi_offset, dev_treeSearch, starvis.searchNr, dev_starCache, dev_nrOfImagesPerStar,
				dev_starTrails, starvis.trailnum, dev_gradient, q, dev_viewer, param.useRedshift, param.useLensing, dev_solidAngles0);


			callKernelAsync("Summed all star light", sumStarLight, numBlocks_N_M_1_24, threadsPerBlock1_24,0,
				dev_starLight0, dev_starTrails, dev_starLight1, starvis.gaussian, image.M, image.N, starvis.diffusionFilter);


			callKernelAsync("Added diffraction", addDiffraction, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
				dev_starLight1, image.M, image.N, dev_diffraction, starvis.diffSize);


			if (!map) {
				callKernelAsync("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,0,
					dev_starLight1, dev_outputImage, image.M, image.N);

			}
		}

		if (map) {
			callKernelAsync("Distorted celestial sky image", distortEnvironmentMap, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
				dev_interpolatedGrid, dev_outputImage, dev_blackHoleMask, celestialSky.imsize, image.M, image.N, camera_phi_offset,
				dev_summedCelestialSky, dev_cameras, dev_solidAngles0, dev_viewer, param.useRedshift, param.useLensing, dev_diskMask);

		}

		if (star && map) {
			callKernelAsync("Created pixels from star light", makePix, numBlocks_N_M_5_25, threadsPerBlock5_25,0,
				dev_starLight1, dev_starImage, image.M, image.N);

			callKernelAsync("Added distorted star and celestial sky image", addStarsAndBackground, numBlocks_N_M_5_25, threadsPerBlock5_25,0,
				dev_starImage, dev_outputImage, dev_outputImage, image.M);
		}
		std::cout << std::endl;

		if (param.useAccretionDisk) {
			if (!param.useAccretionDiskTexture) {
				callKernelAsync("Calculate temperature LUT", createTemperatureTable, numBlocks_tempLUT, threadsPerBlock_32,0,
					param.accretionTemperatureLUTSize, temperatureLUT_device, (param.accretionDiskMaxRadius - 3) / (param.accretionTemperatureLUTSize - 1), param.blackholeMass, param.blackholeAccretion);

				callKernelAsync("Add accretion Disk", addAccretionDisk, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
					dev_interpolatedGrid, dev_outputImage, temperatureLUT_device, (param.accretionDiskMaxRadius / param.accretionTemperatureLUTSize), param.accretionTemperatureLUTSize,
					dev_blackHoleMask, image.M, image.N, dev_cameras, dev_solidAngles0, dev_viewer, param.useLensing, dev_diskMask);
			}
			else {
				callKernelAsync("Add accretion Disk Texture", addAccretionDiskTexture, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
					dev_interpolatedGrid, image.M, dev_blackHoleMask, dev_outputImage, dev_accretionDiskTexture, param.accretionDiskMaxRadius,
					accretionDiskTexture.width, accretionDiskTexture.height, dev_cameras, dev_solidAngles0, dev_viewer, param.useLensing, dev_diskMask
				)
			}
		}

		

		cudaMemcpyAsync(&image.result[0], dev_outputImage, image.N* image.M * 4 * sizeof(uchar), cudaMemcpyDeviceToHost,stream);
		cudaMemcpyAsync(grid.data(), dev_grid, (grids.GN1)* (grids.GM) * sizeof(float4), cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync(area.data(), dev_solidAngles0, (image.N)* (image.M) * sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync(interpolated_grid.data(), dev_interpolatedGrid, (image.N + 1)* (image.M + 1) * sizeof(float4), cudaMemcpyDeviceToHost, stream);
		
		cudaStreamSynchronize(stream);




		cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
		
		

		//checkCudaErrors();

		q++;

		glClearColor(0.0f, 1.0f, 1.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Copy the PBO to the texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, gl_PBO);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);
		GL_CHECK(glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image.M, image.N, GL_BGRA, GL_UNSIGNED_BYTE, 0)); // copy from pbo to texture
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);

		glUniform1i(1, param.windowWidth);
		glUniform1i(2, param.windowHeight);

		glUniform1f(3, viewer->m_CameraFov);
		glUniform3fv(4, 1, glm::value_ptr(viewer->m_CameraDirection));
		glUniform3fv(5, 1, glm::value_ptr(viewer->m_UpDirection));

		GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
		glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(posAttrib);

		GL_CHECK(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));




		glfwSwapBuffers(viewer->get_window());
		glfwPollEvents();

		checkCudaErrors();
		
		std::cout << "frame_duration " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start_time).count() << "ms!" <<
			std::endl << std::endl;
	}
	 
	//Save last output to file

	// Copy output vector from GPU buffer to host memory.
	
	cv::Mat img = cv::Mat(image.N, image.M, CV_8UC4, (void*)&image.result[0]);
	cv::imwrite(param.getResultFileName(grid_value, q), img, image.compressionParams);


	//Write grid results to file
	

	for (int i = 0; i < grid.size(); i++) {
		if (grid[i].x > -2.0) {
			if (grid[i].z > INFINITY_CHECK) {
				grid[i].z = 0;
				grid[i].x = (grid[i].x / PI);
			}
			else {
				grid[i].z = grid[i].z / param.accretionDiskMaxRadius;
				grid[i].x = grid[i].x / 2;
			}


			grid[i].y = (grid[i].y / PI2);
			grid[i].w = 1;

		}
		else {
			grid[i] = { 0,0,0,1 };
		}

	}

	cv::Mat gridmat = cv::Mat((grids.GN1), (grids.GM), CV_32FC4, (void*)grid.data());
	cv::Mat gridUchar;
	gridmat.convertTo(gridUchar, CV_8UC4, 255.0);
	cv::imwrite(param.getGridResultFileName(grid_value, q, "_grid"), gridUchar, image.compressionParams);



	
	
	float depth_max = 0;
	//for (int i = 0; i < area.size(); i++) area[i] = area[i] / 1e-2;

	



	
	for (int i = 0; i < interpolated_grid.size(); i++) {
		interpolated_grid[i].x = (interpolated_grid[i].x / PI);
		interpolated_grid[i].y = (interpolated_grid[i].y / PI2);

		if (interpolated_grid[i].z == INFINITY) {
			interpolated_grid[i].z = 0;
		}
		else {
			interpolated_grid[i].z = interpolated_grid[i].z / (grids.gridStart + q * (param.camRadiusChange ? grids.gridStep : 0));
		}
		//depth[i].y = interpolated_grid[i].w / -30;
		interpolated_grid[i].w = 1;
		if (i % (image.M) < image.M && i < image.M * image.N) {
			depth[i].x = abs(area[i - (i / image.M)] / 1e-7);
		}

	}

	cv::Mat depthmat = cv::Mat((image.N + 1), (image.M + 1), CV_32FC4, (void*)depth.data());
	cv::Mat depthUchar;
	depthmat.convertTo(depthUchar, CV_8U, 255.0);
	cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_depth"), depthUchar, image.compressionParams);



	cv::Mat gridIntermat = cv::Mat((image.N + 1), (image.M + 1), CV_32FC4, (void*)interpolated_grid.data());
	cv::Mat gridInterUchar;
	gridIntermat.convertTo(gridInterUchar, CV_8UC4, 255.0);
	cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_interpolated_grid"), gridInterUchar, image.compressionParams);
	
	while (!glfwWindowShouldClose(viewer->get_window())) {

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, gl_Tex);

		glUniform1i(1, param.windowWidth);
		glUniform1i(2, param.windowHeight);

		glUniform1f(3, viewer->m_CameraFov);
		glUniform3fv(4, 1, glm::value_ptr(viewer->m_CameraDirection));
		glUniform3fv(5, 1, glm::value_ptr(viewer->m_UpDirection));

		GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
		glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(posAttrib);

		GL_CHECK(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
		
		glfwSwapBuffers(viewer->get_window());
		glfwPollEvents();
	};

	//Cleanup
	CUDA::cleanup();
	glfwTerminate();
	delete viewer;


}

ViewCamera* CUDA::glfw_setup(int screen_width, int screen_height) {
	glfwInit();
	GLFWwindow* window = glfwCreateWindow(screen_width, screen_height, "Black-hole visualization", NULL, NULL);
	ViewCamera* camera = new ViewCamera(window, { -1,0,0 }, { 0,0,1 }, screen_width, screen_height, 70);

	glfwSetWindowUserPointer(window, camera);
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, mouse_cursor_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);

	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return nullptr;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return nullptr;
	}

	//Setup viewport after loading glad
	glViewport(0, 0, screen_width, screen_height);

	//Turn off vsync
	glfwSwapInterval(0);

	//Setup opengl
	gladLoadGL();
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

	return camera;
}

std::string CUDA::readFile(const char* filePath) {
	std::string content;
	std::ifstream fileStream(filePath, std::ios::in);


	if (!fileStream.is_open()) {
		std::cerr << "Could not read file " << filePath << ". File does not exist." << std::endl;
		return "";
	}

	std::string line = "";
	while (!fileStream.eof()) {
		std::getline(fileStream, line);
		content.append(line + "\n");
	}

	fileStream.close();
	return content;
}