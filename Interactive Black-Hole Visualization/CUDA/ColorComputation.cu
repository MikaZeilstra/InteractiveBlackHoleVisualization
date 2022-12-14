#include "ColorComputation.cuh"

__global__ void findArea(const float3* thphi, const int M, const int N, float* area) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		bool picheck = false;
		float t[4];
		float p[4];
		int ind = i * M1 + j;
		retrievePixelCorners(thphi, t, p, ind, M, picheck, 0.0f);
		float th1[3] = { t[0], t[1], t[2] };
		float ph1[3] = { p[0], p[1], p[2] };
		float th2[3] = { t[0], t[2], t[3] };
		float ph2[3] = { p[0], p[2], p[3] };
		area[ijc] = calcAreax(th1, ph1) + calcAreax(th2, ph2);
	}
}

__global__ void smoothAreaH(float* areaSmooth, float* area, const unsigned char* bh, const int* gap, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		if (bh[ijc] == 1) return;

		int fs = max(max(gap[i * M1 + j], gap[i * M1 + M1 + j]), max(gap[i * M1 + j + 1], gap[i * M1 + M1 + j + 1]));
		fs = fs / 2;
		float sum = 0.f;
		int count = 0;

		for (int h = -fs; h <= fs; h++) {
			if (bh[i * M + (j + h + M) % M] == 0) {
				float ar = area[i * M + (j + h + M) % M];
				sum += ar;
				count++;
			}
		}

		areaSmooth[ijc] = sum / (1.f * count);
	}
}

__global__ void smoothAreaV(float* areaSmooth, float* area, const unsigned char* bh, const int* gap, const int M, const int N) {
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	if (i < N && j < M) {
		if (bh[ijc] == 1) return;

		int fs = max(max(gap[i * M1 + j], gap[i * M1 + M1 + j]), max(gap[i * M1 + j + 1], gap[i * M1 + M1 + j + 1]));
		fs = fs / 2;
		float sum = 0.f;
		int minv = max(0, i - fs);
		int maxv = min(N - 1, i + fs);
		int count = 0;

		for (int h = minv; h <= maxv; h++) {
			if (bh[h * M + j] == 0) {
				float ar = area[h * M + j];
				sum += ar;
				count++;
			}
		}

		areaSmooth[ijc] = sum / (1.f * count);
	}
}

__device__ float gaussian(float dist, int step) {
	float sigma = 1.f / 2.5f;// *powf(2, step - 1);
	return expf(-.5f * dist / (sigma * sigma)) * 1.f / (sigma * SQRT2PI);
}


__device__ float distSq(float t_a, float t_b, float p_a, float p_b) {
	return (t_a - t_b) * (t_a - t_b) + (p_a - p_b) * (p_a - p_b);
}


__device__ float redshift(float theta, float phi, const float* cam) {
	float xCam = sinf(theta) * cosf(phi);
	float zCam = cosf(theta);
	float yCam = sinf(theta) * sinf(phi);
	float part = (1.f - cam_speed * yCam);
	float betaPart = sqrtf(1.f - cam_speed * cam_speed) / part;

	float xFido = -sqrtf(1.f - cam_speed * cam_speed) * xCam / part;
	float zFido = -sqrtf(1.f - cam_speed * cam_speed) * zCam / part;
	float yFido = (-yCam + cam_speed) / part;
	float k = sqrtf(1.f - cam_btheta * cam_btheta);
	float phiFido = -xFido * cam_br / k + cam_bphi * yFido + cam_bphi * cam_btheta / k * zFido;

	float eF = 1.f / (cam_alpha + cam_w * cam_wbar * phiFido);
	float b = eF * cam_wbar * phiFido;

	return 1.f / (betaPart * (1.f - b * cam_w) / cam_alpha);
}


__device__ void RGBtoHSP(float  R, float  G, float  B, float& H, float& S, float& P) {
	//  Calculate the Perceived brightness.
	P = sqrtf(R * R * Pr + G * G * Pg + B * B * Pb);

	//  Calculate the Hue and Saturation.  (This part works
	//  the same way as in the HSV/B and HSL systems???.)
	if (R == G && R == B) {
		H = 0.f; S = 0.f; return;
	}
	if (R >= G && R >= B) {   //  R is largest
		if (B >= G) {
			H = 6.f / 6.f - 1.f / 6.f * (B - G) / (R - G);
			S = 1.f - G / R;
		}
		else {
			H = 0.f / 6.f + 1.f / 6.f * (G - B) / (R - B);
			S = 1.f - B / R;
		}
	}
	else if (G >= R && G >= B) {   //  G is largest
		if (R >= B) {
			H = 2.f / 6.f - 1.f / 6.f * (R - B) / (G - B);
			S = 1.f - B / G;
		}
		else {
			H = 2.f / 6.f + 1.f / 6.f * (B - R) / (G - R);
			S = 1.f - R / G;
		}
	}
	else {   //  B is largest
		if (G >= R) {
			H = 4.f / 6.f - 1.f / 6.f * (G - R) / (B - R);
			S = 1.f - R / B;
		}
		else {
			H = 4.f / 6.f + 1.f / 6.f * (R - G) / (B - G);
			S = 1.f - G / B;
		}
	}
}


__device__ void HSPtoRGB(float  H, float  S, float  P, float& R, float& G, float& B) {
	float part, minOverMax = 1.f - S;
	if (minOverMax > 0.f) {
		if (H < 1.f / 6.f) {   //  R>G>B
			H = 6.f * H;
			part = 1.f + H * (1.f / minOverMax - 1.f);
			B = P / sqrtf(Pr / minOverMax / minOverMax + Pg * part * part + Pb);
			R = B / minOverMax;
			G = B + H * (R - B);
		}
		else if (H < 2.f / 6.f) {   //  G>R>B
			H = 6.f * (-H + 2.f / 6.f);
			part = 1.f + H * (1.f / minOverMax - 1.f);
			B = P / sqrtf(Pg / minOverMax / minOverMax + Pr * part * part + Pb);
			G = B / minOverMax;
			R = B + H * (G - B);
		}
		else if (H < 3.f / 6.f) {   //  G>B>R
			H = 6.f * (H - 2.f / 6.f);
			part = 1.f + H * (1.f / minOverMax - 1.f);
			R = P / sqrtf(Pg / minOverMax / minOverMax + Pb * part * part + Pr);
			G = R / minOverMax;
			B = R + H * (G - R);
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			H = 6.f * (-H + 4.f / 6.f);
			part = 1.f + H * (1.f / minOverMax - 1.f);
			R = P / sqrtf(Pb / minOverMax / minOverMax + Pg * part * part + Pr);
			B = R / minOverMax;
			G = R + H * (B - R);
		}
		else if (H < 5.f / 6.f) {   //  B>R>G
			H = 6.f * (H - 4.f / 6.f);
			part = 1.f + H * (1.f / minOverMax - 1.f);
			G = P / sqrtf(Pb / minOverMax / minOverMax + Pr * part * part + Pg);
			B = G / minOverMax;
			R = G + H * (B - G);
		}
		else {   //  R>B>G
			H = 6.f * (-H + 6.f / 6.f);
			part = 1.f + H * (1.f / minOverMax - 1.f);
			G = P / sqrtf(Pr / minOverMax / minOverMax + Pb * part * part + Pg);
			R = G / minOverMax;
			B = G + H * (R - G);
		}
	}
	else {
		if (H < 1.f / 6.f) {   //  R>G>B
			H = 6.f * (H);
			R = sqrtf(P * P / (Pr + Pg * H * H));
			G = R * H;
			B = 0.f;
		}
		else if (H < 2.f / 6.f) {   //  G>R>B
			H = 6.f * (-H + 2.f / 6.f);
			G = sqrtf(P * P / (Pg + Pr * H * H));
			R = G * H;
			B = 0.f;
		}
		else if (H < .5f) {   //  G>B>R
			H = 6.f * (H - 2.f / 6.f);
			G = sqrtf(P * P / (Pg + Pb * H * H));
			B = G * H;
			R = 0.f;
		}
		else if (H < 4.f / 6.f) {   //  B>G>R
			H = 6.f * (-H + 4.f / 6.f);
			B = sqrtf(P * P / (Pb + Pg * H * H));
			G = B * H;
			R = 0.f;
		}
		else if (H < 5.f / 6.f) {   //  B>R>G
			H = 6.f * (H - 4.f / 6.f);
			B = sqrtf(P * P / (Pb + Pr * H * H));
			R = B * H;
			G = 0.f;
		}
		else {   //  R>B>G
			H = 6.f * (-H + 1.f);
			R = sqrtf(P * P / (Pr + Pb * H * H));
			B = R * H;
			G = 0.f;
		}
	}
}


__device__ void rgbToHsl(float  r, float  g, float  b, float& h, float& s, float& l) {
	float maxv = max(max(r, g), b);
	float minv = min(min(r, g), b);
	h = (maxv + minv) / 2.f;
	s = h;
	l = h;

	if (maxv == minv) {
		h = s = 0.f; // achromatic
	}
	else {
		float d = maxv - minv;
		s = l > 0.5f ? d / (2 - maxv - minv) : d / (maxv + minv);
		if (maxv == r) h = (g - b) / d + (g < b ? 6.f : 0.f);
		else if (maxv == g) h = (b - r) / d + 2.f;
		else h = (r - g) / d + 4.f;
		h /= 6.f;
	}
}

__device__ float hue2rgb(float p, float q, float t) {
	if (t < 0.f) t += 1.f;
	if (t > 1.f) t -= 1.f;
	if (t < 1.f / 6.f) return p + (q - p) * 6.f * t;
	if (t < 1.f / 2.f) return q;
	if (t < 2.f / 3.f) return p + (q - p) * (2.f / 3.f - t) * 6;
	return p;
}

__device__ void hslToRgb(float h, float s, float l, float& r, float& g, float& b) {
	if (s == 0) {
		r = g = b = l; // achromatic
	}
	else {
		float q = l < 0.5f ? l * (1.f + s) : l + s - l * s;
		float p = 2.f * l - q;
		r = hue2rgb(p, q, h + 1.f / 3.f);
		g = hue2rgb(p, q, h);
		b = hue2rgb(p, q, h - 1.f / 3.f);
	}
}

__device__ __host__ float calcArea(float t[4], float p[4]) {
	float x[4], y[4], z[4];
#pragma unroll
	for (int q = 0; q < 4; q++) {
		float sint = sinf(t[q]);
		x[q] = sint * cosf(p[q]);
		y[q] = sint * sinf(p[q]);
		z[q] = cosf(t[q]);
	}
	float dotpr1 = 1.f;
	dotpr1 += x[0] * x[2] + y[0] * y[2] + z[0] * z[2];
	float dotpr2 = dotpr1;
	dotpr1 += x[2] * x[1] + y[2] * y[1] + z[2] * z[1];
	dotpr1 += x[0] * x[1] + y[0] * y[1] + z[0] * z[1];
	dotpr2 += x[0] * x[3] + y[0] * y[3] + z[0] * z[3];
	dotpr2 += x[2] * x[3] + y[2] * y[3] + z[2] * z[3];
	float triprod1 = fabsf(x[0] * (y[1] * z[2] - y[2] * z[1]) -
		y[0] * (x[1] * z[2] - x[2] * z[1]) +
		z[0] * (x[1] * y[2] - x[2] * y[1]));
	float triprod2 = fabsf(x[0] * (y[2] * z[3] - y[3] * z[2]) -
		y[0] * (x[2] * z[3] - x[3] * z[2]) +
		z[0] * (x[2] * y[3] - x[3] * y[2]));
	float area = 2.f * (atanf(triprod1 / dotpr1) + atanf(triprod2 / dotpr2));
	return area;
}

__device__ __host__ float calcAreax(float t[3], float p[3]) {
	float xi[3], yi[3], zi[3];

#pragma unroll
	for (int q = 0; q < 3; q++) {
		float sint = sinf(t[q]);
		xi[q] = sint * cosf(p[q]);
		yi[q] = sint * sinf(p[q]);
		zi[q] = cosf(t[q]);
	}
	float dot01 = xi[0] * xi[1] + yi[0] * yi[1] + zi[0] * zi[1];
	float dot02 = xi[0] * xi[2] + yi[0] * yi[2] + zi[0] * zi[2];
	float dot12 = xi[2] * xi[1] + yi[2] * yi[1] + zi[2] * zi[1];
	float x[3] = { xi[0], xi[1], xi[2] };
	float y[3] = { yi[0], yi[1], yi[2] };
	float z[3] = { zi[0], zi[1], zi[2] };

	if (dot01 < dot02 && dot01 < dot12) {
		x[0] = xi[2]; x[1] = xi[0]; x[2] = xi[1];
		y[0] = yi[2]; y[1] = yi[0]; y[2] = yi[1];
		z[0] = zi[2]; z[1] = zi[0]; z[2] = zi[1];
	}
	else if (dot02 < dot12 && dot02 < dot01) {
		x[0] = xi[1]; x[1] = xi[2]; x[2] = xi[0];
		y[0] = yi[1]; y[1] = yi[2]; y[2] = yi[0];
		z[0] = zi[1]; z[1] = zi[2]; z[2] = zi[0];
	}

	float dotpr1 = 1.f;
	dotpr1 += x[0] * x[2] + y[0] * y[2] + z[0] * z[2];
	dotpr1 += x[2] * x[1] + y[2] * y[1] + z[2] * z[1];
	dotpr1 += x[0] * x[1] + y[0] * y[1] + z[0] * z[1];
	float triprod1 = fabsf(x[0] * (y[1] * z[2] - y[2] * z[1]) -
		y[0] * (x[1] * z[2] - x[2] * z[1]) +
		z[0] * (x[1] * y[2] - x[2] * y[1]));
	float area = 2.f * (atanf(triprod1 / dotpr1));
	return area;
}

__device__ void bv2rgb(float& r, float& g, float& b, float bv)    // RGB <0,1> <- BV <-0.4,+2.0> [-]
{
	float t;  r = 0.0; g = 0.0; b = 0.0; if (bv < -0.4) bv = -0.4; if (bv > 2.0) bv = 2.0;
	if ((bv >= -0.40) && (bv < 0.00)) {
		t = (bv + 0.40) / (0.00 + 0.40); r = 0.61 + (0.11 * t) + (0.1 * t * t);
	}
	else if ((bv >= 0.00) && (bv < 0.40)) {
		t = (bv - 0.00) / (0.40 - 0.00); r = 0.83 + (0.17 * t);
	}
	else if ((bv >= 0.40) && (bv < 2.10)) {
		t = (bv - 0.40) / (2.10 - 0.40); r = 1.00;
	}
	if ((bv >= -0.40) && (bv < 0.00)) {
		t = (bv + 0.40) / (0.00 + 0.40); g = 0.70 + (0.07 * t) + (0.1 * t * t);
	}
	else if ((bv >= 0.00) && (bv < 0.40)) {
		t = (bv - 0.00) / (0.40 - 0.00); g = 0.87 + (0.11 * t);
	}
	else if ((bv >= 0.40) && (bv < 1.60)) {
		t = (bv - 0.40) / (1.60 - 0.40); g = 0.98 - (0.16 * t);
	}
	else if ((bv >= 1.60) && (bv < 2.00)) {
		t = (bv - 1.60) / (2.00 - 1.60); g = 0.82 - (0.5 * t * t);
	}
	if ((bv >= -0.40) && (bv < 0.40)) {
		t = (bv + 0.40) / (0.40 + 0.40); b = 1.00;
	}
	else if ((bv >= 0.40) && (bv < 1.50)) {
		t = (bv - 0.40) / (1.50 - 0.40); b = 1.00 - (0.47 * t) + (0.1 * t * t);
	}
	else if ((bv >= 1.50) && (bv < 1.94)) {
		t = (bv - 1.50) / (1.94 - 1.50); b = 0.63 - (0.6 * t * t);
	}
}


__device__ void findLensingRedshift(volatile float* t, volatile float* p, const int M, const int ind, const float* camParam,
	const float2* viewthing, float& frac, float& redshft, float solidAngle) {
	if (solidAngle == 0.f) {	
		float th1[3] = { t[0], t[1], t[2] };
		float ph1[3] = { p[0], p[1], p[2] };
		float th2[3] = { t[0], t[2], t[3] };
		float ph2[3] = { p[0], p[2], p[3] };
		solidAngle = calcAreax(th1, ph1) + calcAreax(th2, ph2);
	}

	float ver4[4] = { viewthing[ind].x, viewthing[ind + 1].x, viewthing[ind + M1 + 1].x, viewthing[ind + M1].x };
	float hor4[4] = { viewthing[ind].y, viewthing[ind + 1].y, viewthing[ind + M1 + 1].y, viewthing[ind + M1].y };
	float pixArea = calcArea(ver4, hor4);

	frac = pixArea / solidAngle;
	float thetaCam = (ver4[0] + ver4[1] + ver4[2] + ver4[3]) * .25f;
	float phiCam = (hor4[0] + hor4[1] + hor4[2] + hor4[3]) * .25f;
	redshft = redshift(thetaCam, phiCam, camParam);
}