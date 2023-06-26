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
														checkCudaErrors(); \
														std::cout << milliseconds << " ms\t " << txt << std::endl; \
													  }
#else
	#define callKernelAsync(txt, kernel, blocks, threads,shared_mem_size, ...);{ 							\
														kernel <<<blocks, threads,shared_mem_size,stream>>>(__VA_ARGS__);		\
													  }
#endif // DEBUG

/// <summary>
/// Trackers for the total time, total rays, integration batches and integration batches on the gpu over the entire program run.
/// </summary>
long total_time = 0;
long total_rays = 0;
long total_batches = 0;
long gpu_batches = 0;

CUstream stream;
cudaEvent_t start, stop;

float milliseconds = 0.f;


float2* dev_grid = 0;
float2* dev_grid_2 = 0;

float2* dev_disk_grid = 0;
float2* dev_disk_grid_2 = 0;

float3* dev_incident_grid = 0;
float3* dev_incident_grid_2 = 0;

float2* dev_disk_summary = 0;
float2* dev_disk_summary_2 = 0;

float3* dev_disk_incident_summary = 0;
float3* dev_disk_incident_summary_2 = 0;

static float2* dev_interpolatedGrid = 0;

static float2* dev_interpolatedDiskGrid = 0;

static float3* dev_interpolatedIncidentGrid = 0;

int* dev_gridGap = 0;

float* dev_cameras = 0;
float* dev_cameras_2 = 0;

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
float3* disk_incident_device;

double* temperatureLUT_device;
float4* dev_accretionDiskTexture = 0;

cudaError_t CUDA::cleanup() {

	cudaFree(dev_grid);
	cudaFree(dev_disk_grid);
	cudaFree(dev_incident_grid);

	cudaFree(dev_interpolatedGrid);
	cudaFree(dev_interpolatedDiskGrid);
	cudaFree(dev_interpolatedIncidentGrid);

	cudaFree(dev_disk_summary);
	cudaFree(dev_disk_incident_summary);

	cudaFree(dev_gridGap);

	cudaFree(dev_cameras);

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
	cudaFree(disk_incident_device);


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


void CUDA::call(BlackHole* bh, StarProcessor& stars_, Viewer& view, CelestialSkyProcessor& celestialsky, Texture& accretionTexture, Parameters& param) {
	std::cout << "Preparing CUDA parameters..." << std::endl;

	CelestialSky celestSky(celestialsky);
	Stars stars(stars_);
	Image image(view);
	StarVis starvis(stars, image, param);

	memoryAllocationAndCopy( image, celestSky, stars,  starvis, accretionTexture,param);
	runKernels(bh, image, celestSky, stars,  starvis, accretionTexture, param);
}

void CUDA::allocateGridMemory(size_t size) {
	allocate(pRvs_device, sizeof(double) * size, "Momentum R");
	allocate(bs_device, sizeof(double) * size, "b param");
	allocate(qs_device, sizeof(double) * size, "q param");
	allocate(pThetas_device, sizeof(double) * size, "Momentum theta");
	allocate(disk_incident_device, sizeof(float3) * size, "Thetas");
	checkCudaErrors();
};

template <class T> void CUDA::integrateGrid(const T rV, const T thetaV, const T phiV, std::vector <T>& pRV,
	std::vector <T>& bV, std::vector <T>& qV, std::vector <T>& pThetaV, float3* disk_incident){

	copyHostToDeviceAsync(pRvs_device, pRV.data(), pRV.size() * sizeof(T), "pRs");
	copyHostToDeviceAsync(bs_device, bV.data(), bV.size() * sizeof(T), "bs");
	copyHostToDeviceAsync(qs_device, qV.data(), qV.size() * sizeof(T), "qs");
	copyHostToDeviceAsync(pThetas_device, pThetaV.data(), pThetaV.size() * sizeof(T), "pThetaVs");


	int threads_per_block = 32;

	int block_size = ceil(pRV.size() / (float)threads_per_block);

	//We can reinterpret_cast since T is either double or float and we reserve space for the larger double type
	callKernelAsync("integrate GPU", metric::integrate_kernel<T>, block_size, threads_per_block, 0,
		rV, thetaV, phiV, reinterpret_cast<T*>(pRvs_device), reinterpret_cast<T*>(bs_device), reinterpret_cast<T*>(qs_device), reinterpret_cast<T*>(pThetas_device), reinterpret_cast<float3*>(disk_incident_device), pRV.size());

	copyDeviceToHostAsync(bV.data(), bs_device, bV.size() * sizeof(T), "found theta");
	copyDeviceToHostAsync(qV.data(), qs_device, qV.size() * sizeof(T), "found phi");
	copyDeviceToHostAsync(pThetaV.data(), pThetas_device, pThetaV.size() * sizeof(T), "found disk r");
	copyDeviceToHostAsync(pRV.data(), pRvs_device, pRV.size() * sizeof(T), "found disk phi");
	copyDeviceToHostAsync(disk_incident, disk_incident_device, pThetaV.size() * sizeof(float3), "found incident");

	
	cudaStreamSynchronize(stream);
}



void CUDA::memoryAllocationAndCopy(const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const StarVis& starvis,const Texture accretionTexture, const Parameters& param) {

	std::cout << "Allocating CUDA memory..." << std::endl;

	// Size parameters for malloc and memcopy
	int treeSize = (1 << (stars.treeLevel + 1)) - 1;

	int imageSize = image.M * image.N;
	int rastSize = (image.M + 1) * (image.N + 1);

	int gridsize = param.grid_M * param.grid_N;

	int celestSize = celestialSky.rows * celestialSky.cols;

	//Increase memory limits
	size_t size_heap, size_stack;
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 67108864);
	cudaDeviceSetLimit(cudaLimitStackSize, 16384);
	cudaDeviceGetLimit(&size_heap, cudaLimitMallocHeapSize);
	cudaDeviceGetLimit(&size_stack, cudaLimitStackSize);


	allocate(dev_grid, 2 * gridsize * sizeof(float2), "grid");
	dev_grid_2 = &dev_grid[gridsize];

	allocate(dev_disk_grid, 2 * gridsize * sizeof(float2), "grid");
	dev_disk_grid_2 = &dev_disk_grid[gridsize];

	allocate(dev_incident_grid, 2 * gridsize * sizeof(float3), "interpolatedGrid");
	dev_incident_grid_2 = &dev_incident_grid[gridsize];

	allocate(dev_interpolatedGrid, rastSize * sizeof(float2), "interpolatedGrid");
	
	allocate(dev_interpolatedDiskGrid, rastSize * sizeof(float4), "interpolatedGrid");


	if (param.useAccretionDisk) {
		allocate(dev_interpolatedIncidentGrid, rastSize * sizeof(float3), "interpolatedGrid");

		allocate(dev_disk_summary, 2 * param.n_disk_angles * (param.n_disk_sample + 2 * (1 + param.max_disk_segments)) * sizeof(float2), "grid_summary");
		dev_disk_summary_2 = &dev_disk_summary[param.n_disk_angles * (param.n_disk_sample + 2 * (1 + param.max_disk_segments))];

		allocate(dev_disk_incident_summary, 2 * param.n_disk_angles * (param.n_disk_sample) * sizeof(float3), "grid_incident_summary");
		dev_disk_incident_summary_2 = &dev_disk_incident_summary[param.n_disk_angles * (param.n_disk_sample)];

		allocate(temperatureLUT_device, param.accretionTemperatureLUTSize * sizeof(double), "temperature table");
		allocate(dev_accretionDiskTexture, accretionTexture.width * accretionTexture.height * sizeof(float4), "AccretionTexture");
	}

	allocate(dev_gridGap, rastSize * sizeof(int), "gridGap");

	allocate(dev_cameras, 10 * 2 * sizeof(float), "cameras");
	dev_cameras_2 = &dev_cameras[10];

	allocate(dev_blackHoleMask, imageSize * sizeof(unsigned char), "blackHoleMask");
	allocate(dev_diskMask, imageSize * sizeof(unsigned char), "blackHoleMask");
	allocate(dev_blackHoleBorder0, ((param.n_black_hole_angles * 2) +1)* sizeof(float2), "blackHoleBorder0");
	allocate(dev_blackHoleBorder1, ((param.n_black_hole_angles * 2) +1) * sizeof(float2), "BlackHOleBorder1");

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


	

	std::cout << "Copying variables into CUDA memory..." << std::endl;
	copyHostToDeviceAsync(dev_viewer, image.viewer, rastSize * sizeof(float2), "viewer");

	copyHostToDeviceAsync(dev_blackHoleBorder0, &param.bh_center, sizeof(float2), "blackHoleBorder0");

	copyHostToDeviceAsync(dev_starTree, stars.tree, treeSize * sizeof(int), "starTree");
	copyHostToDeviceAsync(dev_starPositions, stars.stars, stars.starSize * 2 * sizeof(float), "starPositions");
	copyHostToDeviceAsync(dev_starMagnitudes, stars.magnitude, stars.starSize * 2 * sizeof(float), "starMagnitudes");
	copyHostToDeviceAsync(dev_diffraction, starvis.diffraction, starvis.diffSize * starvis.diffSize * sizeof(uchar3), "diffraction");

	copyHostToDeviceAsync(dev_summedCelestialSky, celestialSky.summedCelestialSky, celestSize * sizeof(float4), "summedCelestialSky");
	copyHostToDeviceAsync(dev_accretionDiskTexture, accretionTexture.summed.data(), accretionTexture.height * accretionTexture.width * sizeof(float4), "accretionTexture");

	cudaStreamSynchronize(stream);

	//copyHostToDevice(dev_hit, grids.hit, grids.G * imageSize * sizeof(float2),"hit ");
	std::cout << "Completed CUDA preparation." << std::endl << std::endl;

}


bool map = true;
float hor = 0.0f;
float ver = 0.0f;
//bool redshiftOn = true;
//bool lensingOn = true;

void CUDA::runKernels(BlackHole* bh, const Image& image, const CelestialSky& celestialSky,
	const Stars& stars, const StarVis& starvis, const Texture& accretionDiskTexture, Parameters& param) {
	bool star = param.useStars;

	

	int threadsPerBlock_32 = 32;
	int numBlocks_starsize = stars.starSize / threadsPerBlock_32 + 1;
	int numBlocks_bordersize = (param.n_black_hole_angles * 2) / threadsPerBlock_32 + 1;
	int numBlocks_tempLUT = param.accretionTemperatureLUTSize / threadsPerBlock_32 + 1;
	int numBlocks_disk_edges = (param.n_disk_angles) / threadsPerBlock_32 + 1;

	dim3 threadsPerBlock4_4(4, 4);
	dim3 numBlocks_N_M_4_4((image.N - 1) / threadsPerBlock4_4.x + 1, (image.M - 1) / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_N1_M1_4_4(image.N / threadsPerBlock4_4.x + 1, image.M / threadsPerBlock4_4.y + 1);
	dim3 numBlocks_GN_GM_4_4((param.grid_N - 1) / threadsPerBlock4_4.x + 1, (param.grid_M- 1) / threadsPerBlock4_4.y + 1);

	dim3 threadsPerBlock5_25(5, 25);
	dim3 numBlocks_GN_GM_5_25((param.grid_N - 1) / threadsPerBlock5_25.x + 1, (param.grid_M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N_M_5_25((image.N - 1) / threadsPerBlock5_25.x + 1, (image.M - 1) / threadsPerBlock5_25.y + 1);
	dim3 numBlocks_N1_M1_5_25(image.N / threadsPerBlock5_25.x + 1, image.M / threadsPerBlock5_25.y + 1);

	dim3 threadsPerBlock1_24(1, 24);
	dim3 numBlocks_N_M_1_24((image.N - 1) / threadsPerBlock1_24.x + 1, (image.M - 1) / threadsPerBlock1_24.y + 1);



		
	CUDA::requestGrid(
		{
			param.getRadius(0),
			param.getInclination(0),
			0
		},
		{
			param.br,
			param.btheta,
			param.bphi
		},
		metric::calcSpeed(param.getRadius(0), param.getInclination(0)), bh, &param, dev_cameras_2, dev_grid_2, dev_disk_grid_2, dev_incident_grid_2
	);




	bool should_interpolate_grids = false;
	float grid_value = 0.f;
	float alpha = 0.f;
	int grid_nr = -1;
	int prev_grid_nr = -2;
	int startframe = 0;

	int q = startframe;

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

	//Calculate phi offset due to camera movement
	float camera_phi_offset = 0;

	while(q < param.nrOfFrames + startframe && !glfwWindowShouldClose(viewer->get_window())) {
 		frame_start_time = std::chrono::high_resolution_clock::now();


		//Map the PBO to cuda and set the outputimage pointer to that location
		checkCudaStatus( cudaGraphicsMapResources(1, &cuda_pbo_resource, 0),"map_resource");
		checkCudaStatus(cudaGraphicsResourceGetMappedPointer((void**)&dev_outputImage, &num_bytes, cuda_pbo_resource),"get_pointer");
		
		
		camera_phi_offset = param.getPhi(q);		

		//Calculate the value of the grid we need for this frame
		grid_value = q * (((float)param.gridNum -1) / (param.nrOfFrames-1));
		std::cout << "Computing grid: " << grid_value << " for frame " << q << std::endl;

		//Get the blending between lower and higher grid
		alpha = fmodf(grid_value, 1.f);

			

		//Get the lower grid nr
		grid_nr = (int)grid_value;
		should_interpolate_grids = alpha > FLT_EPSILON;
		//if it is the last frame we dont need to interpolate but do need to swap grids and not load a new grid
		if (q == (param.nrOfFrames - 1)) {
			should_interpolate_grids = false;
			grid_nr = prev_grid_nr;
			swap_grids();
		}
			


		//If a new grid is required and we need to interpolate this grid
		if (grid_nr != prev_grid_nr) {
			//Move the grids further
			swap_grids();
			//Request the new next grid
			requestGrid(
				{
					param.getRadius(grid_nr + 1),
					param.getInclination(grid_nr + 1),
					0
				},
				{
					param.br,
					param.btheta,
					param.bphi
				},
				metric::calcSpeed(param.getRadius(grid_nr + 1), param.getInclination(grid_nr + 1)), bh, &param, dev_cameras_2, dev_grid_2, dev_disk_grid_2, dev_incident_grid_2
			);

			if (q == 0) {
				if (param.useAccretionDisk) {
					callKernelAsync("disk_edges", CreateDiskSummary, numBlocks_disk_edges, threadsPerBlock_32, 0, param.grid_M, param.grid_N, dev_disk_grid, dev_incident_grid, dev_disk_summary, dev_disk_incident_summary, dev_blackHoleBorder0, param.accretionDiskMaxRadius, param.n_disk_angles, param.n_disk_sample, param.max_disk_segments);
				}
			}
			


			callKernelAsync("Find black-hole shadow border", findBhBorders, numBlocks_bordersize, threadsPerBlock_32, 0,
				param.grid_M, param.grid_N, dev_grid, dev_grid_2, param.n_black_hole_angles, dev_blackHoleBorder0);

			callKernelAsync("Smoothed shadow border 1/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32, 0,
				dev_blackHoleBorder0, dev_blackHoleBorder1, param.n_black_hole_angles);
			callKernelAsync("Smoothed shadow border 2/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32, 0,
				dev_blackHoleBorder1, dev_blackHoleBorder0, param.n_black_hole_angles);
			callKernelAsync("Smoothed shadow border 3/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32, 0,
				dev_blackHoleBorder0, dev_blackHoleBorder1, param.n_black_hole_angles);
			callKernelAsync("Smoothed shadow border 4/4", smoothBorder, numBlocks_bordersize, threadsPerBlock_32, 0,
				dev_blackHoleBorder1, dev_blackHoleBorder0, param.n_black_hole_angles);

			if (param.useAccretionDisk) {
				callKernelAsync("disk_edges", CreateDiskSummary, numBlocks_disk_edges, threadsPerBlock_32, 0, param.grid_M, param.grid_N, dev_disk_grid_2, dev_incident_grid_2, dev_disk_summary_2, dev_disk_incident_summary_2, dev_blackHoleBorder0, param.accretionDiskMaxRadius, param.n_disk_angles, param.n_disk_sample, param.max_disk_segments);
			}
		}

			

		prev_grid_nr = grid_nr;
		//cudaEventRecord(start);

		callKernelAsync("Interpolated grid", pixInterpolation,numBlocks_N1_M1_5_25, threadsPerBlock5_25, 0,
			dev_viewer, image.M, image.N, should_interpolate_grids, dev_interpolatedGrid, dev_grid, dev_grid_2, param.grid_M, param.grid_N,
			hor, ver, dev_gridGap, param.gridMaxLevel, dev_blackHoleBorder0, param.n_black_hole_angles, alpha);

		if (param.useAccretionDisk) {
			callKernelAsync("Interpolated disk grid", disk_pixInterpolation, numBlocks_N1_M1_5_25, threadsPerBlock5_25, 0,
				dev_viewer, image.M, image.N, should_interpolate_grids, dev_interpolatedDiskGrid, dev_interpolatedIncidentGrid, dev_disk_grid, dev_incident_grid,
				dev_disk_summary, dev_disk_summary_2, dev_disk_incident_summary, dev_disk_incident_summary_2, param.n_disk_angles, param.n_disk_sample, param.max_disk_segments,
				param.grid_M, param.grid_N,
				hor, ver, dev_gridGap, param.gridMaxLevel, dev_blackHoleBorder0, param.n_black_hole_angles, alpha);

			callKernelAsync("Constructed disk mask", makeDiskCheck, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
				dev_interpolatedDiskGrid, dev_diskMask, image.M, image.N);
		}
		

		callKernelAsync("Constructed black-hole shadow mask", findBlackPixels, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_interpolatedGrid, image.M, image.N, dev_blackHoleMask);

		callKernelAsync("Calculated solid angles", findArea, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_interpolatedGrid,dev_interpolatedDiskGrid, image.M, image.N, dev_solidAngles0,param.accretionDiskMaxRadius, dev_diskMask, dev_interpolatedIncidentGrid);
		
		
		callKernelAsync("Smoothed solid angles horizontally", smoothAreaH, numBlocks_N_M_5_25, threadsPerBlock5_25,0,
			dev_solidAngles1, dev_solidAngles0, dev_blackHoleMask, dev_gridGap, image.M, image.N, dev_diskMask);

		callKernelAsync("Smoothed solid angles vertically", smoothAreaV, numBlocks_N_M_5_25, threadsPerBlock5_25, 0,
			dev_solidAngles0, dev_solidAngles1, dev_blackHoleMask, dev_gridGap, image.M, image.N, dev_diskMask);
		
		
		


		if (star) {
			//TODO fix using correct camera (interpolate it)
			callKernelAsync("Cleared star cache", clearArrays, numBlocks_starsize, threadsPerBlock_32,0,
				dev_nrOfImagesPerStar, dev_starCache, q, starvis.trailnum, stars.starSize);

			callKernelAsync("Calculated gradient field for star trails", makeGradField, numBlocks_N1_M1_4_4, threadsPerBlock4_4,0,
				dev_interpolatedGrid, image.M, image.N, dev_gradient);

			callKernelAsync("Distorted star map", distortStarMap, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
				dev_starLight0, dev_interpolatedGrid, dev_diskMask, dev_blackHoleMask, dev_starPositions, dev_starTree, stars.starSize,
				dev_cameras, dev_starMagnitudes, stars.treeLevel,
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
				dev_starImage, dev_outputImage, dev_outputImage, image.M, image.N);
		}

		if (param.useAccretionDisk) {
			if (!param.useAccretionDiskTexture) {
				callKernelAsync("Calculate temperature LUT", createTemperatureTable, numBlocks_tempLUT, threadsPerBlock_32,0,
					param.accretionTemperatureLUTSize, temperatureLUT_device, (param.accretionDiskMaxRadius - 3) / (param.accretionTemperatureLUTSize - 1), param.blackholeMass, param.blackholeAccretion);

				callKernelAsync("Add accretion Disk", addAccretionDisk, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
					dev_interpolatedDiskGrid, dev_interpolatedIncidentGrid, dev_outputImage, temperatureLUT_device, (param.accretionDiskMaxRadius / param.accretionTemperatureLUTSize), param.accretionTemperatureLUTSize,
					dev_blackHoleMask, image.M, image.N, dev_cameras, dev_solidAngles0, dev_viewer, param.useLensing, dev_diskMask);
			}
			else {
				callKernelAsync("Add accretion Disk Texture", addAccretionDiskTexture, numBlocks_N_M_4_4, threadsPerBlock4_4,0,
					dev_interpolatedDiskGrid, image.M, dev_blackHoleMask, dev_outputImage, dev_accretionDiskTexture, param.accretionDiskMaxRadius,
					accretionDiskTexture.width, accretionDiskTexture.height, dev_cameras, dev_solidAngles0, dev_viewer, param.useLensing, dev_diskMask
				)
			}
		}

		

		
		cudaStreamSynchronize(stream);
		cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);

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
		
		std::cout << "frame_duration " <<
			std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - frame_start_time).count() << "ms!" <<
			std::endl << std::endl;
	}
	 
	//Save last output to file

	// Copy output vector from GPU buffer to host memory.
	std::vector<float2> grid((param.grid_M)* (param.grid_N));
	std::vector<float2> disk_grid((param.grid_M)* (param.grid_N));
	std::vector<float4> depth((image.N + 1)* (image.M + 1), { 0,0,0,1 });
	std::vector<float2> interpolated_grid((image.N + 1)* (image.M + 1));
	std::vector<float2> interpolated_disk_grid((image.N + 1)* (image.M + 1));
	std::vector<float> area((image.N)* (image.M));


	cudaMemcpyAsync(&image.result[0], dev_outputImage, image.N* image.M * 4 * sizeof(uchar), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(grid.data(), dev_grid, (param.grid_N)* (param.grid_M) * sizeof(float2), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(disk_grid.data(), dev_disk_grid_2, (param.grid_N)* (param.grid_M) * sizeof(float2), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(area.data(), dev_solidAngles0, (image.N)* (image.M) * sizeof(float), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(interpolated_grid.data(), dev_interpolatedGrid, (image.N + 1)* (image.M + 1) * sizeof(float2), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(interpolated_disk_grid.data(), dev_interpolatedDiskGrid, (image.N + 1)* (image.M + 1) * sizeof(float2), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	checkCudaErrors();

	cv::Mat img = cv::Mat(image.N, image.M, CV_8UC4, (void*)&image.result[0]);
	cv::imwrite(param.getResultFileName(grid_value, q), img, image.compressionParams);
	//Write grid results to file
	std::vector<float4> grid_image(grid.size());
	for (int i = 0; i < grid.size(); i++) {
		if (grid[i].x > -2.0) {
			grid_image[i].x = grid[i].x / PI;
			grid_image[i].y = (grid[i].y / PI2);
			grid_image[i].w = 1;
		}
		else {
			grid_image[i] = { 0,0,0,1 };
		}
	}

	std::cout << "saving as " << param.getGridResultFileName(grid_value, q, "_grid") << std::endl;

	cv::Mat gridmat = cv::Mat((param.grid_N), (param.grid_M), CV_32FC4, (void*)grid_image.data());
	cv::Mat gridUchar;
	gridmat.convertTo(gridUchar, CV_8UC4, 255.0);
	cv::imwrite(param.getGridResultFileName(grid_value, q, "_grid"), gridUchar, image.compressionParams);

	std::vector<float4> disk_grid_image(disk_grid.size());
	for (int i = 0; i < disk_grid.size(); i++) {
		if (disk_grid[i].x > -2.0 || isnan(disk_grid[i].x)) {
			if (!isnan(disk_grid[i].x)) {
				disk_grid_image[i].x = disk_grid[i].x / param.accretionDiskMaxRadius;
				disk_grid_image[i].y = (disk_grid[i].y / PI2);
				disk_grid_image[i].w = 1;
			} else {
				disk_grid_image[i].x = 0;
				disk_grid_image[i].y = 0;
				disk_grid_image[i].z = 1;
				disk_grid_image[i].w = 1;
			} 

		}
		else {

			disk_grid_image[i] = { 0,0,0,1 };
		}
	}



	cv::Mat disk_gridmat = cv::Mat((param.grid_N), (param.grid_M), CV_32FC4, (void*)disk_grid_image.data());
	cv::Mat disk_gridUchar;
	disk_gridmat.convertTo(disk_gridUchar, CV_8UC4, 255.0);
	cv::imwrite(param.getGridResultFileName(grid_value, q, "_grid_disk"), disk_gridUchar, image.compressionParams);

	
	
	float depth_max = 0;
	//for (int i = 0; i < area.size(); i++) area[i] = area[i] / 1e-2;

	


	std::vector<float4> interpolated_grid_image(interpolated_grid.size());
	for (int i = 0; i < interpolated_grid.size(); i++) {
		interpolated_grid_image[i].x = (interpolated_grid[i].x / PI);
		interpolated_grid_image[i].y = (interpolated_grid[i].y / PI2);
		interpolated_grid_image[i].w = 1;

	}

	cv::Mat depthmat = cv::Mat((image.N + 1), (image.M + 1), CV_32FC4, (void*)depth.data());
	cv::Mat depthUchar;
	depthmat.convertTo(depthUchar, CV_8U, 255.0);
	cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_depth"), depthUchar, image.compressionParams);


	std::vector<float4> interpolated_disk_grid_image(interpolated_disk_grid.size());
	for (int i = 0; i < interpolated_disk_grid.size(); i++) {
		if (!isnan(interpolated_disk_grid[i].x)) {
			interpolated_disk_grid_image[i].x = interpolated_disk_grid[i].x / param.accretionDiskMaxRadius;
			interpolated_disk_grid_image[i].y = (interpolated_disk_grid[i].y / PI2);
			interpolated_disk_grid_image[i].w = 1;
		}
		else {
			interpolated_disk_grid_image[i].x = 0;
			interpolated_disk_grid_image[i].y = 0;
			interpolated_disk_grid_image[i].w = 1;
		}

	}

	cv::Mat gridIntermat = cv::Mat((image.N + 1), (image.M + 1), CV_32FC4, (void*)interpolated_grid_image.data());
	cv::Mat gridInterUchar;
	gridIntermat.convertTo(gridInterUchar, CV_8UC4, 255.0);
	cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_interpolated_grid"), gridInterUchar, image.compressionParams);

	cv::Mat gridIntermat_disk = cv::Mat((image.N + 1), (image.M + 1), CV_32FC4, (void*)interpolated_disk_grid_image.data());
	cv::Mat gridInterUchar_disk;
	gridIntermat_disk.convertTo(gridInterUchar_disk, CV_8UC4, 255.0);
	cv::imwrite(param.getInterpolatedGridResultFileName(grid_value, q, "_interpolated_grid_disk"), gridInterUchar_disk, image.compressionParams);
	
	std::cout << "total time spend generating grids " << total_time << " total ray count " << total_rays << std::endl;
	std::cout << "total integration batches " << total_batches << " of which on the GPU " << gpu_batches << std::endl;


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

void CUDA::requestGrid(double3 cam_pos, double3 cam_speed_dir, float speed, BlackHole* bh, Parameters* param, float* dev_cam, float2* dev_grid, float2* dev_disk, float3* dev_inc){

	auto grid_time = std::chrono::high_resolution_clock::now();
	
	std::cout << "Grid requested : R = " << cam_pos.x << ", Theta =  " << cam_pos.y << ", phi = " << cam_pos.z << std::endl;

	//make camera
	Camera cam(cam_pos, cam_speed_dir, speed);


	//Save camera
	std::vector<float> camera_data = cam.getParamArray();
	copyHostToDeviceAsync(dev_cam, camera_data.data(), camera_data.size() * sizeof(float), "Requested camera");

	//generate grid
	Grid grid(&cam, bh, param);


	//Save grid
	copyHostToDeviceAsync(dev_grid, grid.grid_vector.data(), grid.grid_vector.size() * sizeof(float2), "Requested grid");
	copyHostToDeviceAsync(dev_disk, grid.disk_grid_vector.data(), grid.disk_grid_vector.size() *sizeof(float2), "Requested grid");
	copyHostToDeviceAsync(dev_inc, grid.disk_incident_vector.data(), grid.disk_incident_vector.size() *sizeof(float3), "Requested grid");


	CUDA::gridLevelCount(grid);
	std::cout << "grid_generation_duration " <<
		std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - grid_time).count() << "ms!" <<
		std::endl;
	total_time += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - grid_time).count();
	total_batches += grid.integration_batches;
	gpu_batches += grid.GPU_batches;

}

void CUDA::swap_grids() {
	float2* grid_tmp = dev_grid;
	dev_grid = dev_grid_2;
	dev_grid_2 = grid_tmp;

	float2* disk_tmp = dev_disk_grid;
	dev_disk_grid = dev_disk_grid_2;
	dev_disk_grid_2 = disk_tmp;

	float3* incident_tmp = dev_incident_grid;
	dev_incident_grid = dev_incident_grid_2;
	dev_incident_grid_2 = incident_tmp;

	float* cam_tmp = dev_cameras;
	dev_cameras = dev_cameras_2;
	dev_cameras_2 = cam_tmp;

	float2* disk_summary_tmp = dev_disk_summary;
	dev_disk_summary = dev_disk_summary_2;
	dev_disk_summary_2 = disk_summary_tmp;

	float3* disk_incident_tmp = dev_disk_incident_summary;
	dev_disk_incident_summary = dev_disk_incident_summary_2;
	dev_disk_incident_summary_2 = disk_incident_tmp;
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

/// <summary>
/// Prints the number of blocks for each level, and total rays of a grid.
/// </summary>
/// <param name="grid">The grid.</param>
void CUDA::gridLevelCount(Grid& grid) {
	int maxlevel = grid.MAXLEVEL;
	std::vector<int> check(maxlevel + 1);
	for (int p = 1; p < maxlevel + 1; p++)
		check[p] = 0;
	for (auto block : grid.blockLevels)
		check[block.second]++;
	for (int p = 1; p < maxlevel + 1; p++)
		std::cout << "lvl " << p << " blocks: " << check[p] << std::endl;
	std::cout << std::endl << "Total rays: " << grid.ray_count << std::endl;
	total_rays += grid.ray_count;
}