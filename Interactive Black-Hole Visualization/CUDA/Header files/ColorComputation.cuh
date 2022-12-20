#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "intellisense_cuda_intrinsics.cuh"
#include "Constants.cuh"
#include "GridLookup.cuh"

__global__ void findArea(const float3* thphi, const int M, const int N, float* area);
__global__ void smoothAreaH(float* areaSmooth, float* area, const unsigned char* bh, const int* gap, const int M, const int N);
__global__ void smoothAreaV(float* areaSmooth, float* area, const unsigned char* bh, const int* gap, const int M, const int N);

/// <summary>
/// Computes a semigaussian for the specified distance value.
/// </summary>
/// <param name="dist">The distance value.</param>
__device__ float gaussian(float dist, int step);

/// <summary>
/// Computes the euclidean distance between two points a and b.
/// </summary>
__device__ float distSq(float t_a, float t_b, float p_a, float p_b);

/// <summary>
/// Calculates the redshift (1+z) for the specified theta-phi on the camera sky.
/// </summary>
/// <param name="theta">The theta of the position on the camera sky.</param>
/// <param name="phi">The phi of the position on the camera sky.</param>
/// <param name="cam">The camera parameters.</param>
/// <returns></returns>
__device__ float redshift(float theta, float phi, const float* cam);


#define  Pr  .299f
#define  Pg  .587f
#define  Pb  .114f
/// <summary>
/// public domain function by Darel Rex Finley, 2006
///  This function expects the passed-in values to be on a scale
///  of 0 to 1, and uses that same scale for the return values.
///  See description/examples at alienryderflex.com/hsp.html
/// </summary>
__device__ void RGBtoHSP(float  R, float  G, float  B, float& H, float& S, float& P);

//  public domain function by Darel Rex Finley, 2006
//
//  This function expects the passed-in values to be on a scale
//  of 0 to 1, and uses that same scale for the return values.
//
//  Note that some combinations of HSP, even if in the scale
//  0-1, may return RGB values that exceed a value of 1.  For
//  example, if you pass in the HSP color 0,1,1, the result
//  will be the RGB color 2.037,0,0.
//
//  See description/examples at alienryderflex.com/hsp.html
__device__ void HSPtoRGB(float  H, float  S, float  P, float& R, float& G, float& B);

/**
* Converts an RGB color value to HSL. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
*
* @param   {number}  r       The red color value
* @param   {number}  g       The green color value
* @param   {number}  b       The blue color value
* @return  {Array}           The HSL representation
*/
__device__ void rgbToHsl(float  r, float  g, float  b, float& h, float& s, float& l);

__device__ float hue2rgb(float p, float q, float t);

/**
* Converts an HSL color value to RGB. Conversion formula
* adapted from http://en.wikipedia.org/wiki/HSL_color_space.
*
* @param   {number}  h       The hue
* @param   {number}  s       The saturation
* @param   {number}  l       The lightness
* @return  {Array}           The RGB representation
*/
__device__ void hslToRgb(float h, float s, float l, float& r, float& g, float& b);

__device__ __host__ float calcArea(float t[4], float p[4]);

__device__ __host__ float calcAreax(float t[3], float p[3]);

__device__ void bv2rgb(float& r, float& g, float& b, float bv);

__device__ void findLensingRedshift(volatile float* t, volatile float* p, const int M, const int ind, const float* camParam,
	const float2* viewthing, float& frac, float& redshft, float solidAngle);