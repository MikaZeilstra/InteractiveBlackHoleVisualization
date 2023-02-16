#include "../Header files/ImageDeformation.cuh"

__global__ void distortEnvironmentMap(const float3* thphi, uchar4* out, const unsigned char* bh, const int2 imsize,
										const int M, const int N, float offset, float4* sumTable, const float* camParam,
										const float* solidangle, float2* viewthing, bool redshiftOn, bool lensingOn) {
	
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float4 color = { 0.f, 0.f, 0.f, 0.f };
	int ind = i * M1 + j;
	// Only compute if pixel is not black hole and i j is in image

	if (i < N && j < M) {


		if (bh[ijc] == 0 && thphi[ind].z > INFINITY_CHECK) {

			volatile float t[4], p[4];
			bool picheck = false;
			retrievePixelCorners(thphi, const_cast<float*>(t), const_cast<float*>(p), ind, M, picheck, offset);

			if (ind > 0) {

				float pixSize = PIc / float(imsize.x);
				float phMax = max(max(p[0], p[1]), max(p[2], p[3]));
				float phMin = min(min(p[0], p[1]), min(p[2], p[3]));
				int pixMax = int(phMax / pixSize);
				int pixMin = int(phMin / pixSize);

				int pixNum = pixMax - pixMin + 1;
				if (pixNum > 1 && pixMax * pixSize > phMax) pixNum -= 1;


				if (pixNum == 1) {
					float thMax = max(max(t[0], t[1]), max(t[2], t[3]));
					float thMin = min(min(t[0], t[1]), min(t[2], t[3]));
					int index = int(thMax / pixSize) * imsize.y + pixMin;
					float4 maxColor = sumTable[index];
					index = (int(thMin / pixSize) - 1) * imsize.y + pixMin;
					float4 minColor = sumTable[index];
					color.x += maxColor.x - minColor.x;
					color.y += maxColor.y - minColor.y;
					color.z += maxColor.z - minColor.z;
					color.w += maxColor.w - minColor.w;
				}
				else {
					float max1 = -100;
					float min1 = imsize.y + 1;
					float max2 = -100;
					float min2 = imsize.y + 1;
					if (pixNum < imsize.y) {
						for (int s = 0; s <= pixNum; s++) {
							float cp = (pixMin + s) * pixSize;
							for (int q = 0; q < 4; q++) {
								float ap = p[q];
								float bp = p[(q + 1) % 4];
								float at = t[q];
								float bt = t[(q + 1) % 4];

								if ((cp - ap) * (cp - bp) <= 0) {
									float ct = at + (bt - at) / (bp - ap) * (cp - ap);
									min2 = ct < min2 ? ct : min2;
									max2 = ct > max2 ? ct : max2;
								}
							}
							if (s > 0) {
								float max_ = max(max1, max2);
								float min_ = min(min1, min2);

								int index1 = int(max_ / pixSize) * imsize.y + (pixMin + s) % imsize.y;

								float4 maxColor = sumTable[index1];
								int index2 = (int(min_ / pixSize) - 1) * imsize.y + (pixMin + s) % imsize.y;
								float4 minColor;
								if (index2 > 0) minColor = sumTable[index2];
								else minColor = { 0.f, 0.f, 0.f, 0.f };

								color.x += maxColor.x - minColor.x;
								color.y += maxColor.y - minColor.y;
								color.z += maxColor.z - minColor.z;
								color.w += maxColor.w - minColor.w;
							}
							max1 = max2;
							min1 = min2;
						}
					}

					else {
						float thMax = max(max(t[0], t[1]), max(t[2], t[3]));
						float thMin = min(min(t[0], t[1]), min(t[2], t[3]));
						int thMaxPix = int(thMax / pixSize);
						int thMinPix = int(thMin / pixSize);
						//pixcount = pixNum * (thMaxPix - thMinPix);
						thMaxPix *= imsize.y;
						thMinPix *= imsize.y;
						for (int q = 0; q < pixNum; q++) {
							float4 maxColor = sumTable[thMaxPix + (pixMin + q) % imsize.y];
							float4 minColor = sumTable[thMinPix + (pixMin + q) % imsize.y];
							color.x += maxColor.x - minColor.x;
							color.y += maxColor.y - minColor.y;
							color.z += maxColor.z - minColor.z;
							color.w += maxColor.w - minColor.w;
						}

					}
				}

				

				color.x = min(255.f, powf(color.x / color.w, 1.f / 2.2f));
				color.y = min(255.f, powf(color.y / color.w, 1.f / 2.2f));
				color.z = min(255.f, powf(color.z / color.w, 1.f / 2.2f));

				float H, S, P;
				if (lensingOn || redshiftOn) {
					RGBtoHSP(color.z / 255.f, color.y / 255.f, color.x / 255.f, H, S, P);

					float redshft = 1;
					float frac = 1;
					findLensingRedshift(t, p, M, ind, camParam, viewthing, frac, redshft, solidangle[ijc]);
					if (lensingOn) P *= frac;
					if (redshiftOn) P = powf(P, redshft);
					//Redshift does not change Hue channel?
					HSPtoRGB(H, S, min(1.f, P), color.z, color.y, color.x);
					color.x *= 255.f;
					color.y *= 255.f;
					color.z *= 255.f;
				}
			}
		}
		//CHANGED
		out[ijc] = { (unsigned char)min(255, (int) color.x),   
					 (unsigned char)min(255, (int) color.y), 
					 (unsigned char)min(255, (int) color.z), 255 };
	}
}

__global__ void makePix(float3* starLight, uchar4* out, int M, int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	//extra
	//float2 h = hit[ijc];
	float disk_b = 0.f;
	float disk_g = 0.f;
	float disk_r = 0.f;
	//if (h.x > 0) {
	//	disk_b = 255.f*(h.x - 9.f) / (18.f - 9.f);
	//	disk_g = 255.f - disk_b;// h.y / PI2;
	//}// h.x > 0.f ? 255.f*(1.f - h.y / PI2) : 0.f;
	//if (h.y > PI) printf("%f, %f, %d, %d \n", h.x, h.y, i, j);
	//extra

	float3 rgb = { sqrtf(starLight[ijc].x), sqrtf(starLight[ijc].y), sqrtf(starLight[ijc].z) };
	float max = fmaxf(fmaxf(rgb.x, rgb.y), rgb.z);
	if (max > 255.f) {
		float3 hsp;
		RGBtoHSP(rgb.x / 255.f, rgb.y / 255.f, rgb.z / 255.f, hsp.x, hsp.y, hsp.z);
		hsp.z = 1.0f;
		HSPtoRGB(hsp.x, hsp.y, hsp.z, rgb.x, rgb.y, rgb.z);
		rgb.x *= 255; rgb.y *= 255; rgb.z *= 255;
	}
	out[ijc] = { (unsigned char)min(255, (int)(rgb.z + disk_b)),
				(unsigned char)min(255, (int)(rgb.y + disk_g)),
				(unsigned char)min(255, (int)(rgb.x + disk_r)), 255 };
}

__global__ void addStarsAndBackground(uchar4* stars, uchar4* background, uchar4* output, int M) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	float3 star = { (float)stars[ijc].x * stars[ijc].x, (float)stars[ijc].y * stars[ijc].y, (float)stars[ijc].z * stars[ijc].z };
	float3 bg = { (float)background[ijc].x * background[ijc].x, (float)background[ijc].y * background[ijc].y, (float)background[ijc].z * background[ijc].z };
	float p = 1.f;
	float3 out = { sqrtf(p * star.x + (2.f - p) * bg.x), sqrtf(p * star.y + (2.f - p) * bg.y), sqrtf(p * star.z + (2.f - p) * bg.z) };
	//float max = fmaxf(fmaxf(out.x, out.y), out.z);
	//if (max > 255.f) {
	//	out.y *= (255.f / max);
	//	out.z *= (255.f / max);
	//	out.x *= (255.f / max);
	//}

	//  CHANGED
	output[ijc] = { (unsigned char)min((int)out.x, 255), (unsigned char)min((int)out.y, 255), (unsigned char)min((int)out.z, 255), 255 };
}
