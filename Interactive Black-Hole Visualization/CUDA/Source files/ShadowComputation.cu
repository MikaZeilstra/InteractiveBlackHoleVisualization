#include "../Header files/ShadowComputation.cuh"
#include "../Header files/vector_operations.cuh"

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

__global__ void findBhCenter(const int GM, const int GN, const float2* grid, const float2* grid_2, float2* bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < GN && j < GM) {
		if (isnan(grid[i * GM + j].x) || isnan(grid_2[ i * GM + j].x)) {
			atomicMinFloat(&(bhBorder[0].x), (float)i);
			atomicMaxFloat(&(bhBorder[0].y), (float)i);
			atomicMinFloat(&(bhBorder[1].x), (float)j);
			atomicMaxFloat(&(bhBorder[1].y), (float)j);
		}
	}
}

__global__ void findBhBorders(const int GM, const int GN, const float2 bh_center, const float2* grid, const int angleNum, float2* bhBorder) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum ) {
		float angle = PI2 / (1.f * angleNum) * 1.f * i;
		float2 change = { -sinf(angle), cosf(angle) };
		float2 pt = bh_center;
		int2 gridpt = { int(pt.x), int(pt.y) };

		float2 gridB = { -2, -2 };
		float2 gridA = { -2, -2 };

		while (!(gridA.x > 0 && isnan(gridB.x))) {
			gridB = gridA;
			pt = pt + change;
			gridpt = { int(pt.x), int(pt.y) };

			if (gridpt.x > GN || gridpt.x < 0 || gridpt.y > GM || gridpt.y < 0) {
				break;
			}

			gridA = grid[gridpt.x * GM + gridpt.y];


		}


		bhBorder[i] = pt - bh_center - change;
	}
}

__global__ void displayborders(const int angleNum, float2* bhBorder, uchar4* out, const int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < angleNum) {
		int x = int(bhBorder[i].x);
		int y = int(bhBorder[i].y);
		unsigned char outx = 255 * (i);
		unsigned char outy = 255 * (1 - i);
		out[x * M + y] = { outx, outy, 0, 255 };
	}
}

__global__ void smoothBorder(const float2* bhBorder, float2* bhBorder2, const int angleNum) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < angleNum) {
		int prev = (i - 1 + angleNum) % angleNum;
		int next = (i + 1) % angleNum;
	
		bhBorder2[i] = { 1.f / 3.f * (bhBorder[prev].x + bhBorder[i].x + bhBorder[next].x),
							 1.f / 3.f * (bhBorder[prev].y + bhBorder[i].y + bhBorder[next].y) };

	}
}

__global__ void findBlackPixels(const float2* thphi, const int M, const int N, unsigned char* bh) {
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
