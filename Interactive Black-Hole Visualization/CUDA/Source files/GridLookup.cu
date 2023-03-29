#include "../Header files/GridLookup.cuh"
#include "../Header files/metric.cuh"
#include "../Header files/Constants.cuh"

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCheck(volatile float* p, float factor) {
	float factor1 = PI2 * (1.f - factor);
	bool check = false;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q] < factor2) {
			p[q] += PI2;
			check = true;
		}
	}
	return check;
}

__device__ bool piCheck(volatile float3* p, float factor) {
	float factor1 = PI2 * (1.f - factor);
	bool check = false;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q].y < factor2) {
			p[q].y += PI2;
			check = true;
		}
	}
	return check;
}

__device__ bool piCheck(volatile float4* p, float factor) {
	float factor1 = PI2 * (1.f - factor);
	bool check = false;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (p[q].y < factor2) {
			p[q].y += PI2;
			check = true;
		}
	}
	return check;
}



__device__ void findBlock(const float theta, const float phi, const int g, const float4* grid,
	const int GM, const int GN, int& i, int& j, int& gap, const int level) {

	for (int s = 0; s < level + 1; s++) {
		int ngap = gap / 2;
		int k = i + ngap;
		int l = j + ngap;
		if (gap <= 1 || grid[g * GN * GM + k * GM + l].x == -2.f) return;
		else {
			float thHalf = PI2 * k / ((float)GM);
			float phHalf = PI2 * l / ((float)GM);
			if (thHalf <= theta) i = k;
			if (phHalf <= phi) j = l;
			gap = ngap;
		}
	}
}

/// <summary>
/// Checks and corrects phi values for 2-pi crossings.
/// </summary>
/// <param name="p">The phi values to check.</param>
/// <param name="factor">The factor to check if a point is close to the border.</param>
/// <returns></returns>
__device__ bool piCheckTot(float4* tp, float factor, int size) {
	float factor1 = PI2 * (1.f - factor);
	bool check = false;
	for (int q = 0; q < size; q++) {
		if (tp[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
	for (int q = 0; q < size; q++) {
		if (tp[q].y < factor2) {
			tp[q].y += PI2;
			check = true;
		}
	}
	return check;
}

__device__ bool piCheckTot(float3* tp, float factor, int size) {
	float factor1 = PI2 * (1.f - factor);
	bool check = false;
	for (int q = 0; q < size; q++) {
		if (tp[q].y > factor1) {
			check = true;
			break;
		}
	}
	if (!check) return false;
	check = false;
	float factor2 = PI2 * factor;
	for (int q = 0; q < size; q++) {
		if (tp[q].y < factor2) {
			tp[q].y += PI2;
			check = true;
		}
	}
	return check;
}

// Set values for projected pixel corners & update phi values in case of 2pi crossing.
// If the corners cross into the accretion disk we set them to a known good value
__device__ void retrievePixelCorners(const float4* thphi, float* t, float* p, int& ind, const int M, bool& picheck, float offset) {

	t[0] = thphi[ind + M1].x;
	p[0] = thphi[ind + M1].y;
	t[1] = thphi[ind].x;
	p[1] = thphi[ind].y;
	t[2] = thphi[ind + 1].x;
	p[2] = thphi[ind + 1].y;
	t[3] = thphi[ind + M1 + 1].x;
	p[3] = thphi[ind + M1 + 1].y;



#pragma unroll
	for (int q = 0; q < 4; q++) {
		if (isnan(p[q])) {
			ind = -1;
			return;
		}
		p[q] = fmodf(p[q] + offset, PI2);
	}
	// Check and correct for 2pi crossings.
	picheck = piCheck(p, PI_CHECK_FACTOR);
}


__device__ void wrapToPi(float& thetaW, float& phiW) {
	thetaW = fmodf(thetaW, PI2);
	while (thetaW < 0.f) thetaW += PI2;
	if (thetaW > PI) {
		thetaW -= 2.f * (thetaW - PI);
		phiW += PI;
	}
	while (phiW < 0.f) phiW += PI2;
	phiW = fmodf(phiW, PI2);
}

__device__ int2 hash1(int2 key, int ow) {
	return{ (key.x + ow) % ow, (key.y + ow) % ow };
}

__device__ int2 hash0(int2 key, int hw) {
	return{ (key.x + hw) % hw, (key.y + hw) % hw };
}

__device__ float3 hashLookup(int2 key, const float3* hashTable, const int2* hashPosTag, const int2* offsetTable, const int2* tableSize, const int g) {

	int ow = tableSize[g].y;
	int hw = tableSize[g].x;
	int ostart = 0;
	int hstart = 0;
	for (int q = 0; q < g; q++) {
		hstart += tableSize[q].x * tableSize[q].x;
		ostart += tableSize[q].y * tableSize[q].y;
	}

	int2 index = hash1(key, ow);
	//printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", g, ow, hw, ow*ow, hw*hw, ostart, hstart, index.x, index.y, ostart + index.x*ow + index.y, );

	int2 add = { hash0(key, hw).x + offsetTable[ostart + index.x * ow + index.y].x,
				 hash0(key, hw).y + offsetTable[ostart + index.x * ow + index.y].y };
	int2 hindex = hash0(add, hw);

	if (hashPosTag[hstart + hindex.x * hw + hindex.y].x != key.x || hashPosTag[hstart + hindex.x * hw + hindex.y].y != key.y) return{ -2.f, -2.f, -2.f };
	else return hashTable[hstart + hindex.x * hw + hindex.y];
}
