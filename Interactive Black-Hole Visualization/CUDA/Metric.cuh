#pragma once
#include  <cuda_runtime.h>

//Contains and calculates properties of the metric
//Integrates the lightlike geodesics
//We need to explicitly instatiate several templated functions since nvcc does not see that they are used in cpp files
namespace metric {
	template <class T> static T a = 0;
	template <class T> static T asq = 0;

	template <class T> __device__ __constant__  static T a_dev = 0;
	template <class T> __device__ __constant__  static T asq_dev = 0;

	template <class T> __device__ __host__ __forceinline__ T sq(T x);
	template <class T> __device__ __host__ __forceinline__ T sq3(T x);
	template <class T> __device__ __host__ __forceinline__ T _Delta(T r, T rsq);
	template <class T> __device__ __host__ __forceinline__ T _ro(T r, T theta, T rsq, T cossq);
	template <class T> __device__ __host__ __forceinline__ T _rosq(T r, T theta, T rsq, T cossq);
	template <class T> __device__ __host__ __forceinline__ T _w(T r, T theta, T rsq, T sinsq);
	template  __device__ __host__ __forceinline__ double _w<double>(double r, double theta, double rsq, double sinsq);
	template <class T> __device__ __host__ __forceinline__ T _wbar(T r, T theta, T rsq, T sinsq, T cossq);
	template  __device__ __host__ __forceinline__ double _wbar<double>(double r, double theta, double rsq, double sinsq, double cossq);
	template <class T> __device__ __host__ __forceinline__ T _alpha(T r, T theta, T rsq, T sinsq, T cossq);
	template  __device__ __host__ __forceinline__ double _alpha<double>(double r, double theta, double rsq, double sinsq, double cossq);

	template <class T> __device__ __host__ __forceinline__ T _P(T r, T b, T rsq);
	template <class T> __device__ __host__ __forceinline__ T _R(T r, T theta, T b, T q, T rsq);
	template <class T> __device__ __host__ __forceinline__ T _BigTheta(T r, T theta, T b, T q, T sinsq, T cossq, T bsq);

	template <class T> __host__ void setAngVel(T afactor);
	template __host__ void setAngVel<double>(double afactor);

	template <class T> __host__ T calcSpeed(T r, T theta);
	template __host__ double calcSpeed < double > (double r, double theta);

	template <class T> extern __host__ bool checkCelest(T pRV, T rV, T thetaV, T bV, T qV);
	template __host__ bool checkCelest<double>(double pRV, double rV, double thetaV, double bV, double qV);


	template <class T> __device__ __host__ void derivs(volatile T* var, volatile T* varOut, T b, T q);

	__global__ void integrate_kernel(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, int size);

	__device__ __host__ void rkckIntegrate1(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV);


	template <class T>  __device__ __host__ void wrapToPi(T& thetaW, T& phiW);
	template  __device__ __host__ void wrapToPi<double>(double &thetaW, double& phiW);
	template  __device__ __host__ void wrapToPi<float>(float& thetaW, float& phiW);
}


