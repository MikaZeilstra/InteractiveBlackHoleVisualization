#include "../Header files/ShadowComputation.cuh"

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
	float old;
	old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
		__uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

	return old;
}

__global__ void findBhCenter(const int GM, const int GN, const float2* grid, float2* bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < GN && j < GM) {
		if (isnan(grid[i * GM + j].x) || isnan(grid[GM * GN + i * GM + j].x)) {
			float gridsize = PI / (1.f * GN);
			atomicMinFloat(&(bhBorder[0].x), gridsize * (float)i);
			atomicMaxFloat(&(bhBorder[0].y), gridsize * (float)i);
			atomicMinFloat(&(bhBorder[1].x), gridsize * (float)j);
			atomicMaxFloat(&(bhBorder[1].y), gridsize * (float)j);
		}
	}
}

__global__ void findBhBorders(const int GM, const int GN, const float2* grid, const int angleNum, float2* bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum * 2) {
		int ii = i / 2;
		float angle = PI2 / (1.f * angleNum) * 1.f * ii;
		float thetaChange = -sinf(angle);
		float phiChange = cosf(angle);
		float2 pt = { .5f * bhBorder[0].x + .5f * bhBorder[0].y, .5f * bhBorder[1].x + .5f * bhBorder[1].y };
		pt = { pt.x / (float)PI2 * GM, pt.y / (float)PI2 * GM };
		int2 gridpt = { int(pt.x), int(pt.y) };

		float2 gridB = { -2, -2 };
		float2 gridA = { -2, -2 };

		while (!(gridA.x > 0 && isnan(gridB.x))) {
			gridB = gridA;
			pt.x += thetaChange;
			pt.y += phiChange;
			gridpt = { int(pt.x), int(pt.y) };
			gridA = grid[(i % 2) * GM * GN + gridpt.x * GM + gridpt.y];
		}

		bhBorder[2 + i] = { (pt.x - thetaChange) * (float)PI2 / (1.f * GM), (pt.y - phiChange) * (float)PI2 / (1.f * GM) };
	}
}

__global__ void displayborders(const int angleNum, float2* bhBorder, uchar4* out, const int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum * 2) {
		int x = int(bhBorder[i + 2].x / PI2 * 1.f * M);
		int y = int(bhBorder[i + 2].y / PI2 * 1.f * M);
		unsigned char outx = 255 * (i % 2);
		unsigned char outy = 255 * (1 - i % 2);
		out[x * M + y] = { outx, outy, 0, 255 };
	}
}

__global__ void smoothBorder(const float2* bhBorder, float2* bhBorder2, const int angleNum) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < angleNum * 2) {
		if (i == 0) {
			bhBorder2[0] = bhBorder[0];
			bhBorder2[1] = bhBorder[1];
		}
		int prev = (i - 2 + 2 * angleNum) % (2 * angleNum);
		int next = (i + 2) % (2 * angleNum);
		bhBorder2[i + 2] = { 1.f / 3.f * (bhBorder[prev + 2].x + bhBorder[i + 2].x + bhBorder[next + 2].x),
							 1.f / 3.f * (bhBorder[prev + 2].y + bhBorder[i + 2].y + bhBorder[next + 2].y) };
	}
}

__global__ void findBlackPixels(const float4* thphi, const int M, const int N, unsigned char* bh) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		


		bool picheck = false;
		float t[4];
		float p[4];
		int ind = i * M1 + j;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, 0.0f);
 		if (ind == -1) bh[ijc] = 1;
		else bh[ijc] = 0;
	}
}
