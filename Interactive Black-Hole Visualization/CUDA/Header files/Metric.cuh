#pragma once
#include  <cuda_runtime.h>

//Contains and calculates properties of the metric
//Integrates the lightlike geodesics
//We need to explicitly instatiate several templated functions since nvcc does not see that they are used in cpp files
namespace metric {
	template <class T> static T a = 0;
	template <class T> static T asq = 0;
	template <class T> static T accretionDiskRadius = 0;
	static bool useAccretionDisk  = false;

	template <class T> __device__ __constant__  static T a_dev = 0;
	template <class T> __device__ __constant__  static T asq_dev = 0;
	template <class T> __device__ __constant__  static T accretionDiskRadius_dev = 0;
	static bool __device__ __constant__ useAccretionDisk_dev = false;

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

	template <class T> __device__ __host__ __forceinline__ T _gtt_theta_half_pi(T r);
	template <class T> __device__ __host__ __forceinline__ T _gtphi_theta_half_pi(T r);
	template <class T> __device__ __host__ __forceinline__ T _gphiphi_theta_half_pi(T r,T rsq);
	template <class T> __device__ __host__ __forceinline__ T _Omega(T r);
	template <class T> __device__ __host__ __forceinline__ T calculate_gravitational_redshift(T r, T rsq);
	template __device__ __host__ __forceinline__ float calculate_gravitational_redshift<float>(float r, float rsq);

	template <class T> __host__ void setMetricParameters(T afactor, T accretionRadius, bool useDisk);
	template __host__ void setMetricParameters<double>(double afactor, double accretionRadius, bool useDisk);
	template __host__ void setMetricParameters<float>(float afactor, float accretionRadius, bool useDisk);

	template <class T> __device__ __host__ T calcSpeed(T r, T theta);
	template __device__ __host__ double calcSpeed < double > (double r, double theta);
	template __device__ __host__ float calcSpeed < float >(float r, float theta);

	template <class T> extern __host__ bool checkCelest(T pRV, T rV, T thetaV, T bV, T qV);
	template __host__ bool checkCelest<double>(double pRV, double rV, double thetaV, double bV, double qV);


	template <class T> __device__ __host__ void derivs(volatile T* var, volatile T* varOut, T b, T q);

	template <class T> __global__ void integrate_kernel(const T rV, const T thetaV, const T phiV, T* pRV,
		T* bV, T* qV, T* pThetaV,float3* disk_incident, int size);

	template __global__ void integrate_kernel<double>(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, float3* disk_incident, int size);
	template __global__ void integrate_kernel<float>(const float rV, const float thetaV, const float phiV, float* pRV,
		float* bV, float* qV, float* pThetaV, float3* disk_incident, int size);

	template <class T> __device__ __host__ void rkckIntegrate1(const T rV, const T thetaV, const T phiV, T* pRV,
		T* bV, T* qV, T* pThetaV, float3* disk_incident, bool savePath, float3* pathSave);
	template __device__ __host__ void rkckIntegrate1<double>(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, float3* disk_incident,  bool savePath, float3* pathSave);
	template __device__ __host__ void rkckIntegrate1<float>(const float rV, const float thetaV, const float phiV, float* pRV,
		float* bV, float* qV, float* pThetaV, float3* disk_incident,  bool savePath, float3* pathSave);


	template <class T> __device__ __host__ void stepUntilDisk(volatile T* var, volatile  T* dvdz, T& z, T& h,
		volatile T* varScal, const T b, const T q,
		volatile T* varErr, volatile T* varTemp, volatile T* aks, volatile T* varTmpInt, volatile T* varOut);

	template <class T>  __device__ __host__ bool wrapToPi(T& thetaW, T& phiW);
	template <class T>  __device__ __host__ bool wrapPhiToPi(T& phiW);

	//Variables
	template  __device__ __host__ bool wrapToPi<double>(double &thetaW, double& phiW);
	template  __device__ __host__ bool wrapToPi<float>(float& thetaW, float& phiW);

}


