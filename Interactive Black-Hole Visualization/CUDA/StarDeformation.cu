#include "StarDeformation.cuh"

#define DIFFRACTION_THRESHOLD  65025.f

__constant__ const int tempToRGB[1173] = { 255, 56, 0, 255, 71, 0, 255, 83, 0, 255, 93, 0, 255, 101, 0, 255, 109, 0, 255, 115, 0, 255, 121, 0,
												255, 126, 0, 255, 131, 0, 255, 137, 18, 255, 142, 33, 255, 147, 44, 255, 152, 54, 255, 157, 63, 255, 161, 72,
												255, 165, 79, 255, 169, 87, 255, 173, 94, 255, 177, 101, 255, 180, 107, 255, 184, 114, 255, 187, 120,
												255, 190, 126, 255, 193, 132, 255, 196, 137, 255, 199, 143, 255, 201, 148, 255, 204, 153, 255, 206, 159,
												255, 209, 163, 255, 211, 168, 255, 213, 173, 255, 215, 177, 255, 217, 182, 255, 219, 186, 255, 221, 190,
												255, 223, 194, 255, 225, 198, 255, 227, 202, 255, 228, 206, 255, 230, 210, 255, 232, 213, 255, 233, 217,
												255, 235, 220, 255, 236, 224, 255, 238, 227, 255, 239, 230, 255, 240, 233, 255, 242, 236, 255, 243, 239,
												255, 244, 242, 255, 245, 245, 255, 246, 248, 255, 248, 251, 255, 249, 253, 254, 249, 255, 252, 247, 255,
												249, 246, 255, 247, 245, 255, 245, 243, 255, 243, 242, 255, 240, 241, 255, 239, 240, 255, 237, 239, 255,
												235, 238, 255, 233, 237, 255, 231, 236, 255, 230, 235, 255, 228, 234, 255, 227, 233, 255, 225, 232, 255,
												224, 231, 255, 222, 230, 255, 221, 230, 255, 220, 229, 255, 218, 228, 255, 217, 227, 255, 216, 227, 255,
												215, 226, 255, 214, 225, 255, 212, 225, 255, 211, 224, 255, 210, 223, 255, 209, 223, 255, 208, 222, 255,
												207, 221, 255, 207, 221, 255, 206, 220, 255, 205, 220, 255, 204, 219, 255, 203, 219, 255, 202, 218, 255,
												201, 218, 255, 201, 217, 255, 200, 217, 255, 199, 216, 255, 199, 216, 255, 198, 216, 255, 197, 215, 255,
												196, 215, 255, 196, 214, 255, 195, 214, 255, 195, 214, 255, 194, 213, 255, 193, 213, 255, 193, 212, 255,
												192, 212, 255, 192, 212, 255, 191, 211, 255, 191, 211, 255, 190, 211, 255, 190, 210, 255, 189, 210, 255,
												189, 210, 255, 188, 210, 255, 188, 209, 255, 187, 209, 255, 187, 209, 255, 186, 208, 255, 186, 208, 255,
												185, 208, 255, 185, 208, 255, 185, 207, 255, 184, 207, 255, 184, 207, 255, 183, 207, 255, 183, 206, 255,
												183, 206, 255, 182, 206, 255, 182, 206, 255, 182, 205, 255, 181, 205, 255, 181, 205, 255, 181, 205, 255,
												180, 205, 255, 180, 204, 255, 180, 204, 255, 179, 204, 255, 179, 204, 255, 179, 204, 255, 178, 203, 255,
												178, 203, 255, 178, 203, 255, 178, 203, 255, 177, 203, 255, 177, 202, 255, 177, 202, 255, 177, 202, 255,
												176, 202, 255, 176, 202, 255, 176, 202, 255, 175, 201, 255, 175, 201, 255, 175, 201, 255, 175, 201, 255,
												175, 201, 255, 174, 201, 255, 174, 201, 255, 174, 200, 255, 174, 200, 255, 173, 200, 255, 173, 200, 255,
												173, 200, 255, 173, 200, 255, 173, 200, 255, 172, 199, 255, 172, 199, 255, 172, 199, 255, 172, 199, 255,
												172, 199, 255, 171, 199, 255, 171, 199, 255, 171, 199, 255, 171, 198, 255, 171, 198, 255, 170, 198, 255,
												170, 198, 255, 170, 198, 255, 170, 198, 255, 170, 198, 255, 170, 198, 255, 169, 198, 255, 169, 197, 255,
												169, 197, 255, 169, 197, 255, 169, 197, 255, 169, 197, 255, 169, 197, 255, 168, 197, 255, 168, 197, 255,
												168, 197, 255, 168, 197, 255, 168, 196, 255, 168, 196, 255, 168, 196, 255, 167, 196, 255, 167, 196, 255,
												167, 196, 255, 167, 196, 255, 167, 196, 255, 167, 196, 255, 167, 196, 255, 166, 196, 255, 166, 195, 255,
												166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 166, 195, 255, 165, 195, 255,
												165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 195, 255, 165, 194, 255, 165, 194, 255,
												165, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255,
												164, 194, 255, 164, 194, 255, 164, 194, 255, 164, 194, 255, 163, 194, 255, 163, 194, 255, 163, 193, 255,
												163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255, 163, 193, 255,
												163, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255, 162, 193, 255,
												162, 193, 255, 162, 193, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255, 162, 192, 255,
												161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255,
												161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255, 161, 192, 255,
												160, 192, 255, 160, 192, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255,
												160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255, 160, 191, 255,
												160, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255,
												159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 191, 255, 159, 190, 255, 159, 190, 255,
												159, 190, 255, 159, 190, 255, 159, 190, 255, 159, 190, 255, 159, 190, 255, 158, 190, 255, 158, 190, 255,
												158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255,
												158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255,
												158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 158, 190, 255, 157, 190, 255, 157, 190, 255,
												157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255,
												157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255,
												157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255, 157, 189, 255,
												157, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255,
												156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 189, 255,
												156, 189, 255, 156, 189, 255, 156, 189, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255,
												156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255, 156, 188, 255,
												156, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255,
												155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255,
												155, 188, 255, 155, 188, 255, 155, 188, 255, 155, 188, 255 };



__device__ bool checkCrossProduct(float t_a, float t_b, float p_a, float p_b,
	float starTheta, float starPhi, int sgn) {
	float c1t = (float)sgn * (t_a - t_b);
	float c1p = (float)sgn * (p_a - p_b);
	float c2t = sgn ? starTheta - t_b : starTheta - t_a;
	float c2p = sgn ? starPhi - p_b : starPhi - p_a;
	return (c1t * c2p - c2t * c1p) > 0;
}

__device__ bool starInPolygon(const float* t, const float* p, float start, float starp, int sgn) {
	return (checkCrossProduct(t[0], t[1], p[0], p[1], start, starp, sgn)) &&
		(checkCrossProduct(t[1], t[2], p[1], p[2], start, starp, sgn)) &&
		(checkCrossProduct(t[2], t[3], p[2], p[3], start, starp, sgn)) &&
		(checkCrossProduct(t[3], t[0], p[3], p[0], start, starp, sgn));
}


__global__ void makeGradField(const float2* thphi, const int M, const int N, float2* grad) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N1 && j < M1) {
		float up = thphi[min(N, (i + 1)) * M1 + j].x;
		float down = thphi[max(0, (i - 1)) * M1 + j].x;
		float left = thphi[i * M1 + max(0, (j - 1))].x;
		float right = thphi[i * M1 + min(M1, j + 1)].x;
		float mid = thphi[i * M1 + j].x;
		if (mid > 0) {
			float xdir = 0;
			if (up > 0 && down > 0)
				xdir = .5f * (up - mid) - .5f * (down - mid);
			float ydir = 0;
			if (left > 0 && right > 0)
				ydir = .5f * (right - mid) - .5f * (left - mid);
			float size = sqrtf(xdir * xdir + ydir * ydir);
			grad[i * M1 + j] = float2{ -ydir, xdir };
		}
	}
}

__device__ void searchTree(const int* tree, const float* thphiPixMin, const float* thphiPixMax, const int treeLevel,
							int* searchNrs, int startNr, int& pos, int picheck) {
	float nodeStart[2] = { 0.f, 0.f + picheck * PIc };
	float nodeSize[2] = { PIc, PI2c };
	int node = 0;
	unsigned int bitMask = powf(2, treeLevel);
	int level = 0;
	int lvl = 0;
	while (bitMask != 0) {
		bitMask &= ~(1UL << (treeLevel - level));

		for (lvl = level + 1; lvl <= treeLevel; lvl++) {
			int star_n = tree[node];
			if (node != 0 && ((node + 1) & node) != 0) {
				star_n -= tree[node - 1];
			}
			int tp = lvl & 1;

			float x_overlap = max(0.f, min(thphiPixMax[0], nodeStart[0] + nodeSize[0]) - max(thphiPixMin[0], nodeStart[0]));
			float y_overlap = max(0.f, min(thphiPixMax[1], nodeStart[1] + nodeSize[1]) - max(thphiPixMin[1], nodeStart[1]));
			float overlapArea = x_overlap * y_overlap;
			bool size = overlapArea / (nodeSize[0] * nodeSize[1]) > 0.8f;
			nodeSize[tp] = nodeSize[tp] * .5f;
			if (star_n == 0) {
				node = node * 2 + 1; break;
			}

			float check = nodeStart[tp] + nodeSize[tp];
			bool lu = thphiPixMin[tp] < check;
			bool rd = thphiPixMax[tp] >= check;
			if (lvl == 1 && picheck) {
				bool tmp = lu;
				lu = rd;
				rd = tmp;
			}
			if (lvl == treeLevel || (rd && lu && size)) {
				if (rd) {
					searchNrs[startNr + pos] = node * 2 + 2;
					pos++;
				}
				if (lu) {
					searchNrs[startNr + pos] = node * 2 + 1;
					pos++;
				}
				node = node * 2 + 1;
				break;
			}
			else {
				node = node * 2 + 1;
				if (rd) bitMask |= 1UL << (treeLevel - lvl);
				if (!lu) break;
				else if (lvl == 1 && picheck) nodeStart[1] += nodeSize[1];
			}
		}
		level = treeLevel - __ffs(bitMask) + 1;
		if (level >= 0) {
			int diff = lvl - level;
			for (int i = 0; i < diff; i++) {
				int tp = (lvl - i) & 1;
				if (!(node & 1)) nodeStart[tp] -= nodeSize[tp];
				nodeSize[tp] = nodeSize[tp] * 2.f;
				node = (node - 1) / 2;
			}
			node++;
			int tp = level & 1;
			if (picheck && level == 1) nodeStart[tp] -= nodeSize[tp];
			else nodeStart[tp] += nodeSize[tp];
		}
	}
}

__device__ void addTrails(const int starsToCheck, const int starSize, const int framenumber, int* stnums, float3* trail,
	const float2* grad, int2* stCache, const int q, const int i, const int j, const int M, const int trailnum,
	const float part, const float frac, float3 hsp) {
	if (starsToCheck < starSize / 100) {
		int cache = framenumber % 2;
		int loc = atomicAdd(&(stnums[q]), 1);
		if (loc < trailnum) stCache[trailnum * cache + 2 * (trailnum * q) + loc] = { i, j };

		float traildist = M;
		float angle = PI2c;
		int2 prev;
		int num = -1;
		bool line = false;
		for (int w = 0; w < trailnum; w++) {
			int2 pr = stCache[trailnum * (1 - cache) + 2 * (trailnum * q) + w];
			if (pr.x < 0) break;
			int dx = (pr.x - i);
			int dy = (pr.y - j);
			int dxx = dx * dx;
			int dyy = dy * dy;
			if (dxx <= 1 && dyy <= 1) {
				line = false;
				break;
			}
			float2 gr;
			gr = grad[pr.x * M1 + pr.y];

			float dist = sqrtf(dxx + dyy);
			if (dist > M / 25.f) continue;
			float div = (dist * sqrtf(gr.x * gr.x + gr.y * gr.y));
			float a1 = acosf((1.f * dx * gr.x + 1.f * dy * gr.y) / div);
			float a2 = acosf((1.f * dx * -gr.x + 1.f * dy * -gr.y) / div);
			float a = min(a1, a2);
			if (a > angle || a > PIc * .25f) continue;
			else if (a < angle) {
				angle = a;
				traildist = dist;
				prev = pr;
				num = w;
				line = true;
			}
		}
		if (line) {
			int deltax = i - prev.x;
			int deltay = j - prev.y;
			int sgnDeltaX = deltax < 0 ? -1 : 1;
			int sgnDeltaY = deltay < 0 ? -1 : 1;
			float deltaerr = deltay == 0.f ? fabsf(deltax) : fabsf(deltax / (1.f * deltay));
			float error = 0.f;
			int y = prev.y;
			int x = prev.x;
			while (y != j || x != i) {
				if (error < 1.f) {
					y += sgnDeltaY;
					error += deltaerr;
				}
				if (error >= 0.5f) {
					x += sgnDeltaX;
					error -= 1.f;
				}
				float dist = distSq(x, i, y, j);
				float appMag = part - 2.5f * log10f(frac);
				float brightness = 100 * exp10f(-.4f * appMag);
				brightness *= ((traildist - sqrt(dist)) * (traildist - sqrt(dist))) / (traildist * traildist);
				float3 rgb;
				HSPtoRGB(hsp.x, hsp.y, hsp.z * brightness, rgb.x, rgb.y, rgb.z);
				rgb.x *= 255.f;
				rgb.y *= 255.f;
				rgb.z *= 255.f;
				trail[x * M + y].x = rgb.x * rgb.x;
				trail[x * M + y].y = rgb.y * rgb.y;
				trail[x * M + y].z = rgb.z * rgb.z;
				if (dist <= 1.f) break;
			}
		}
	}
}

__global__ void distortStarMap(float3* starLight, const float2* thphi, const unsigned char* bh, const float* stars, const int* tree,
	const int starSize, const float* camParam, const float* magnitude, const int treeLevel,
	const int M, const int N, const int step, float offset, int* search, int searchNr, int2* stCache,
	int* stnums, float3* trail, int trailnum, float2* grad, const int framenumber, const float2* viewthing, bool redshiftOn, bool lensingOn, const float* area) {
	
	distortStarMapWorker(starLight, thphi, bh, stars, tree,
		starSize, camParam, magnitude, treeLevel,
		M, N, step, offset, search, searchNr, stCache,
		stnums, trail, trailnum, grad, framenumber, viewthing, redshiftOn, lensingOn, area);
}

__global__ void distortStarMapSharedMem(float3* starLight, const float2* thphi, const unsigned char* bh, const float* stars, const int* tree,
	const int starSize, const float* camParam, const float* magnitude, const int treeLevel,
	const int M, const int N, const int step, float offset, int* search, int searchNr, int2* stCache,
	int* stnums, float3* trail, int trailnum, float2* grad, const int framenumber, const float2* viewthing, bool redshiftOn, bool lensingOn, const float* area) {

	int tid = threadIdx.x + blockDim.x * threadIdx.y;

	extern __shared__ float shared_memory[];

	//Copy stars into shared memory in a quick manner using coallesing for quick access
	for (int star_index_stride = 0; star_index_stride < starSize * 2; star_index_stride+= blockDim.x) {
		shared_memory[star_index_stride + tid] = stars[star_index_stride + tid];
	}

	__syncthreads();


	distortStarMapWorker(starLight, thphi, bh, shared_memory, tree,
		starSize, camParam, magnitude, treeLevel,
		M, N, step, offset, search, searchNr, stCache,
		stnums, trail, trailnum, grad, framenumber, viewthing, redshiftOn, lensingOn, area);
}

__device__ void distortStarMapWorker(float3* starLight, const float2* thphi, const unsigned char* bh, const float* stars, const int* tree,
	const int starSize, const float* camParam, const float* magnitude, const int treeLevel,
	const int M, const int N, const int step, float offset, int* search, int searchNr, int2* stCache,
	int* stnums, float3* trail, int trailnum, float2* grad, const int framenumber, const float2* viewthing, bool redshiftOn, bool lensingOn, const float* area) {

	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int filterW = step * 2 + 1;

	// Only compute if pixel is in bounds and not black hole.
	if (i < N && j < M) {
		if (bh[ijc] == 0) {

			// Set values for projected pixel corners & update phi values in case of 2pi crossing.
			volatile float t[4], p[4];
			int ind = i * M1 + j;
			bool picheck = false;
			retrievePixelCorners(thphi, const_cast<float*>(t), const_cast<float*>(p), ind, M, picheck, offset);

			// Search where in the star tree the bounding box of the projected pixel falls.
			const float thphiPixMax[2] = { max(max(t[0], t[1]), max(t[2], t[3])),
											max(max(p[0], p[1]), max(p[2], p[3])) };
			const float thphiPixMin[2] = { min(min(t[0], t[1]),  min(t[2], t[3])),
											min(min(p[0], p[1]),  min(p[2], p[3])) };
			int pos = 0;
			int startnr = searchNr * (ijc);
			searchTree(tree, thphiPixMin, thphiPixMax, treeLevel, search, startnr, pos, 0);
			if (pos == 0) return;

			// Calculate orientation and size of projected polygon (positive -> CW, negative -> CCW)
			float orient = (t[1] - t[0]) * (p[1] + p[0]) + (t[2] - t[1]) * (p[2] + p[1]) +
				(t[3] - t[2]) * (p[3] + p[2]) + (t[0] - t[3]) * (p[0] + p[3]);
			int sgn = orient < 0 ? -1 : 1;

			// Calculate redshift and lensing effect
			float redshft = 1.f;
			float frac = 1.f;
			if (lensingOn || redshiftOn) findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft, area[ijc]);
			float red = 4.f * log10f(redshft);
			float maxDistSq = (step + .5f) * (step + .5f);


			// Calculate amount of stars to check
			int starsToCheck = 0;
			for (int s = 0; s < pos; s++) {
				int node = search[startnr + s];
				int startN = 0;
				if (node != 0 && ((node + 1) & node) != 0) {
					startN = tree[node - 1];
				}
				starsToCheck += (tree[node] - startN);
			}

			// Check stars in tree leaves
			for (int s = 0; s < pos; s++) {
				int node = search[startnr + s];
				int startN = 0;
				if (node != 0 && ((node + 1) & node) != 0) {
					startN = tree[node - 1];
				}
				for (int q = startN; q < tree[node]; q++) {
					float start = stars[2 * q];
					float starp = stars[2 * q + 1];
					bool starInPoly = starInPolygon(const_cast<float*>(t), const_cast<float*>(p), start, starp, sgn);
					if (picheck && !starInPoly && starp < PI2c * .2f) {
						starp += PI2c;
						starInPoly = starInPolygon(const_cast<float*>(t), const_cast<float*>(p), start, starp, sgn);
					}
					if (starInPoly) {
						interpolate(t[0], t[1], t[2], t[3], p[0], p[1], p[2], p[3], start, starp, sgn, i, j);
						float part = magnitude[2 * q] + red;
						float temp = 46.f / redshft * ((1.f / ((0.92f * magnitude[2 * q + 1]) + 1.7f)) +
							(1.f / ((0.92f * magnitude[2 * q + 1]) + 0.62f))) - 10.f;
						int index = max(0, min((int)floorf(temp), 1170));
						float3 rgb = { tempToRGB[3 * index], tempToRGB[3 * index + 1], tempToRGB[3 * index + 2] };
						//if (magnitude[2 * q] < -1) rgb = {0,255,0};
						float3 hsp;
						RGBtoHSP(rgb.x / 255.f, rgb.y / 255.f, rgb.z / 255.f, hsp.x, hsp.y, hsp.z);

						addTrails(starsToCheck, starSize, framenumber, stnums, trail, grad, stCache, q, i, j, M, trailnum, part, frac, hsp);

						for (int u = 0; u <= 2 * step; u++) {
							for (int v = 0; v <= 2 * step; v++) {
								float dist = distSq(-step + u + .5f, start, -step + v + .5f, starp);
								if (dist > maxDistSq) continue;
								else {
									float appMag = part - 2.5f * log10f(frac * gaussian(dist, step));
									float brightness = 100 * exp10f(-.4f * appMag);
									HSPtoRGB(hsp.x, hsp.y, hsp.z * brightness, rgb.x, rgb.y, rgb.z);
									rgb.x *= 255.f;
									rgb.y *= 255.f;
									rgb.z *= 255.f;
									starLight[M * N * (filterW * u + v) + j * N + i].x += rgb.x * rgb.x;
									starLight[M * N * (filterW * u + v) + j * N + i].y += rgb.y * rgb.y;
									starLight[M * N * (filterW * u + v) + j * N + i].z += rgb.z * rgb.z;
								}
							}
						}
					}
				}
			}
		}
	}
}


__global__ void sumStarLight(float3* starLight, float3* trail, float3* out, int step, int M, int N, int filterW) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float3 brightness = { 0.f, 0.f, 0.f };
	int start = max(0, step - i);
	int stop = min(2 * step, step + N - i - 1);
	float factor = 20.f;// 00.f;
	for (int u = start; u <= stop; u++) {
		for (int v = 0; v <= 2 * step; v++) {
			int x = (i + u - step);
			int y = ((j + v - step + M) % M);
			brightness.x += factor * starLight[M * N * (filterW * u + v) + y * N + x].x;
			brightness.y += factor * starLight[M * N * (filterW * u + v) + y * N + x].y;
			brightness.z += factor * starLight[M * N * (filterW * u + v) + y * N + x].z;
		}
	}
	float factor2 = 1.f;
	brightness.x += factor2 * trail[ijc].x;
	brightness.y += factor2 * trail[ijc].y;
	brightness.z += factor2 * trail[ijc].z;
	trail[ijc] = { 0.f, 0.f, 0.f };
	out[j * N + i] = brightness;
}


__global__ void clearArrays(int* stnums, int2* stCache, const int frame, const int trailnum, const int starSize) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (i < starSize) {
		stnums[i] = 0;
		int c = frame % 2;
		for (int q = 0; q < trailnum; q++) {
			stCache[trailnum * (c)+2 * (trailnum * i) + q] = { -1, -1 };
			if (frame == 0) stCache[trailnum * (1 - c) + 2 * (trailnum * i) + q] = { -1, -1 };

		}
	}
}

__global__ void addDiffraction(float3* starLight, float3* output, const int M, const int N, const uchar3* diffraction, const int filtersize) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	int threadId = threadIdx.x + threadIdx.y * blockDim.x;

	int filterhalf = filtersize / 2;

	output[ijc] = starLight[j * N + i];

	
	// Convolve diffraction pattern with image if it reaches the threshold
	for (int diff_y =  j - filterhalf; diff_y < j + filterhalf; diff_y++) {
		//Wrap filter horizontally at M and 0 to connect 0 phi to 2 pi phi
		for (int diff_x = (i - filterhalf  +M) % M; diff_x < (i + filterhalf + M) % M; diff_x=(diff_x+1)%M) {
			//Check if filter index is in the image
			__syncwarp();
			if (diff_y >= 0 && diff_y < N) {

				//If starlight in filter meets threshold add an amount determined by diffraction to current pixel
				float3 filter_starlight = starLight[diff_y * N + diff_x];
				float max = fmaxf(fmaxf(filter_starlight.x, filter_starlight.y), filter_starlight.z);
				if (max > DIFFRACTION_THRESHOLD) {
					float div = div = 10000 * powf(max, 0.6); //TODO ask significance of these numbers
					//Get the location in the diffraction filter of our pixel relative to the pixel we are checking for starlight
					int3 filter_location = { i - diff_x + filterhalf, j - diff_y + filterhalf };

					//Get the diffraction contributed from the filtered pixel
					float3 diff = {
						filter_starlight.x / div * (float)(diffraction[filter_location.y * filtersize + filter_location.x].x),
						filter_starlight.y / div * (float)(diffraction[filter_location.y * filtersize + filter_location.x].y),
						filter_starlight.z / div * (float)(diffraction[filter_location.y * filtersize + filter_location.x].z),

					};

					//Add diffraction contribution to our pixel
					output[ijc].x += fmin(DIFFRACTION_THRESHOLD, diff.x * diff.x);
					output[ijc].y += fmin(DIFFRACTION_THRESHOLD, diff.y * diff.y);
					output[ijc].z += fmin(DIFFRACTION_THRESHOLD, diff.z * diff.z);
				}
			}
		}
	}
	
	/*
	// check whether the pixel itself (halfway in the list) has a high value (higher than threshold) If yes, mark it.
	// In a second pass do the convolution with the pattern over all the pixels that got marked.
	float max = fmaxf(fmaxf(starLight[ijc].x, starLight[ijc].y), starLight[ijc].z);
	if (max > DIFFRACTION_THRESHOLD) {
		int filterhalf = filtersize / 2;
		int startx = 0;
		int endx = filtersize;
		if (i < filterhalf) startx = filterhalf - i;
		if (i > (N - filterhalf)) endx = N - i + filterhalf;
		float div = 10000 * powf(max, 0.6);
		for (int q = startx; q < endx; q++) {
			for (int p = 0; p < filtersize; p++) {
				float3 diff = { starLight[ijc].x / div * (float)(diffraction[q * filtersize + p].x),
								starLight[ijc].y / div * (float)(diffraction[q * filtersize + p].y),
								starLight[ijc].z / div * (float)(diffraction[q * filtersize + p].z) };
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].x), fmin(65025.f, diff.x * diff.x));
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].y), fmin(65025.f, diff.y * diff.y));
				atomicAdd(&(starLight[M * (i - filterhalf + q) + (j - filterhalf + p + M) % M].z), fmin(65025.f, diff.z * diff.z));
			}
		}
	}
	*/

	
}