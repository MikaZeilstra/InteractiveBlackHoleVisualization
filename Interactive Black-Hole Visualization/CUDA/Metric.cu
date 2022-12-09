#pragma once
#include "device_launch_parameters.h"

#include "metric.cuh"


#include "../C++/Const.h"
#include "../C++/Code.h"
#include "../C++/IntegrationDefines.h"
//#include "integration.cuh"


namespace metric {
	template <class T> __host__ void setAngVel(T afactor) {
		metric::a<T> = afactor;
		metric::asq<T> = afactor * afactor;

		cudaMemcpyToSymbol(metric::a_dev<T>, &a<T>, sizeof(T));
		cudaMemcpyToSymbol(metric::asq_dev<T>, &asq<T>, sizeof(T));
	}

	template <class T>
	__device__ __host__ __forceinline__ T sq(T x) {
		return x * x;
	}

	template <class T>
	__device__ __host__ __forceinline__ T sq3(T x) {
		return x * x * x;
	}

	template <class T> __device__ __host__ __forceinline__ T _Delta(T r, T rsq) {
		return rsq - 2. * r + BH_ASQ;
	};

	template <class T> __device__ __host__ __forceinline__ T _Sigma(T r, T theta, T rsq, T sinsq) {
		return sqrt(sq(rsq + BH_ASQ) - BH_ASQ * _Delta(r, rsq) * sinsq);
	};

	template <class T> __device__ __host__ __forceinline__ T _ro(T r, T theta, T rsq, T cossq) {
		return sqrt(rsq + BH_ASQ * cossq);
	};

	template <class T> __device__ __host__ __forceinline__ T _rosq(T r, T theta, T rsq, T cossq) {
		return rsq + BH_ASQ * cossq;
	};

	template <class T> __device__ __host__ __forceinline__ T _w(T r, T theta, T rsq, T sinsq) {
		return 2. * BH_A * r / sq(_Sigma(r, theta, rsq, sinsq));
	};

	template <class T> __device__ __host__  __forceinline__ T _wbar(T r, T theta, T rsq, T sinsq, T cossq) {
		return _Sigma(r, theta, rsq, sinsq) * sin(theta) / _ro(r, theta, rsq, cossq);
	};

	template <class T> __device__ __host__ __forceinline__  T _alpha(T r, T theta, T rsq, T sinsq, T cossq) {
		return _ro(r, theta, rsq, cossq) * sqrt(_Delta(r, rsq)) / _Sigma(r, theta, rsq, sinsq);
	};

	template <class T> __device__ __host__ __forceinline__ T _P(T r, T b, T rsq) {
		return rsq + BH_ASQ - BH_A * b;
	}

	template <class T> __device__ __host__ __forceinline__ T _R(T r, T theta, T b, T q, T rsq) {
		return sq(_P(r, b, rsq)) - _Delta(r, rsq) * (sq((b - BH_A)) + q);
	};

	template <class T> __device__ __host__ __forceinline__ T _BigTheta(T r, T theta, T b, T q, T sinsq, T cossq, T bsq) {
		return q - cossq * (bsq / sinsq - BH_ASQ);
	};

	template <class T> __host__ T calcSpeed(T r, T theta) {
		T rsq = sq(r);
		T omega = 1. / (BH_A + pow(r, 1.5));
		T sp = _wbar(r, theta, rsq, sq(sin(theta)), sq(cos(theta))) / _alpha(r, theta, rsq, sq(sin(theta)), sq(cos(theta))) * (omega - _w(r, theta, rsq, sq(sin(theta))));
		return sp;
	}

	template <class T> __host__ T findMinGoldSec(T theta, T bval, T qval, T ax, T b, T tol) {
		T gr = (sqrt(5.0) + 1.) / 2.;
		T c = b - (b - ax) / gr;
		T d = ax + (b - ax) / gr;
		while (fabs(c - d) > tol) {
			if (_R(c, theta, bval, qval, sq(c)) < _R(d, theta, bval, qval, sq(d))) {
				b = d;
			}
			else {
				ax = c;
			}
			c = b - (b - ax) / gr;
			d = ax + (b - ax) / gr;
		}
		return (ax + b) / 2;
	}

	template <class T> __host__ T checkRup(T rV, T thetaV, T bV, T qV) {
		if (BH_A == 0) return false;
		T min = findMinGoldSec(thetaV, bV, qV, rV, 4 * rV, 0.00001);
		return (_R(min, thetaV, bV, qV, sq(min)) >= 0);
	}

	template <class T> __host__ T _b0(T r0) {
		return -(sq3(r0) - 3. * sq(r0) + BH_ASQ * r0 + BH_ASQ) / (BH_A * (r0 - 1.));
	};

	template <class T> __host__ T _b0diff(T r0) {
		return (BH_ASQ + BH_A - 2. * r0 * (sq(r0) - 3. * r0 + 3.)) / (BH_A * sq(r0 - 1.));
	};

	template <class T> __host__ T _q0(T r0) {
		return -sq3(r0) * (sq3(r0) - 6. * sq(r0) + 9. * r0 - 4. * BH_ASQ) / (BH_ASQ * sq(r0 - 1.));
	};

	template <class T> __host__ T checkB_Q(T bV, T qV) {
		T _r1 = 2. * (1. + cos(2. * acos(-BH_A) / 3.));
		T _r2 = 2. * (1. + cos(2. * acos(BH_A) / 3.));
		T error = 0.0000001;
		T r0V = 2.0;
		T bcheck = 100;

		while (fabs(bV - bcheck) > error) {
			bcheck = _b0(r0V);
			T bdiffcheck = _b0diff(r0V);
			T rnew = r0V - (bcheck - bV) / bdiffcheck;
			if (rnew < 1) {
				r0V = 1.0001;
			}
			else {
				r0V = rnew;
			}
		}
		T qb = _q0(r0V);
		T _b1 = _b0(_r2);
		T _b2 = _b0(_r1);
		return ((_b1 >= bV) || (_b2 <= bV) || (qV >= qb));
	}

	template <class T> __host__ bool checkCelest(T pRV, T rV, T thetaV, T bV, T qV) {
		bool check1 = checkB_Q(bV, qV);
		bool check2 = !check1 && (pRV < 0);
		bool check3 = check1 && checkRup(rV, thetaV, bV, qV);
		return check2 || check3;
	}


	//Calculates derivatives of geodesics with the parameters given in var
	template <class T> __device__ __host__ void derivs(volatile T* var, volatile T* varOut, T b, T q) {
		T cosv = cos(thetaVar);
		T sinv = sin(thetaVar);
		T cossq = metric::sq(cosv);
		T sinsq = metric::sq(sinv);
		T bsq = metric::sq(b);
		T rsq = metric::sq(rVar);
		T delta = metric::_Delta(rVar, rsq);
		T rosq = metric::_rosq(rVar, thetaVar, rsq, cossq);
		T P = metric::_P(rVar, b, rsq);
		T prsq = metric::sq(pRVar);
		T pthetasq = metric::sq(pThetaVar);
		T R = metric::_R(rVar, thetaVar, b, q, rsq);
		T partR = (q + metric::sq(BH_A - b));
		T btheta = metric::_BigTheta(rVar, thetaVar, b, q, sinsq, cossq, bsq);
		T rosqsq = metric::sq(2 * rosq);
		T sqrosqdel = (metric::sq(rosq) * delta);
		T asqcossin = BH_ASQ * cosv * sinv;
		T rtwo = 2 * rVar - 2;

		drdz = delta / rosq * pRVar;
		dtdz = 1. / rosq * pThetaVar;
		dpdz = (2 * BH_A * P - (2 * BH_A - 2 * b) * delta + (2 * b * cossq * delta) / sinsq) / (rosq * 2 * delta);
		dprdz = (rtwo * btheta - rtwo * partR + 4 * rVar * P) / (rosq * (2 * delta)) - (prsq * rtwo) / (2 * rosq)
			+ (4 * pthetasq * rVar) / rosqsq - ((4 * rVar - 4) * (btheta * (delta)+R)) / (rosq * metric::sq(2 * delta))
			+ (4 * prsq * rVar * (delta)) / rosqsq - (rVar * (btheta * delta + R)) / sqrosqdel;
		dptdz = ((2 * cosv * sinv * (bsq / sinsq - BH_ASQ) + (2 * bsq * metric::sq3(cosv)) / metric::sq3(sinv)) * delta) /
			(rosq * 2 * delta) - (4 * asqcossin * pthetasq) / rosqsq - (4 * asqcossin * prsq * delta) /
			rosqsq + (asqcossin * (btheta * delta + R)) / sqrosqdel;
	}

#ifndef __CUDA_ARCH__
	const double BUTCHER_TABLEAU;
	const double BUTCHER_ERROR;
#else
	__device__ const double BUTCHER_TABLEAU;
	__device__ const double BUTCHER_ERROR;
#endif // !__CUDA_ARCH__


	template <class T>  __device__ __host__ void wrapToPi(T& thetaW, T& phiW) {
		thetaW = fmod(thetaW, PI2);
		if (thetaW < 0) {
			thetaW += PI2;
		}

		if (thetaW > PI) {
			thetaW -= 2 * (thetaW - PI);
			phiW += PI;
		}
		phiW = fmod(phiW, PI2);
		if (phiW < 0) {
			phiW += PI2;
		}
	}

	__device__ __host__ static void rkck(volatile double* var, volatile double* dvdz, const double h,
		volatile double* varOut, volatile double* varErr, const double b, const double q, volatile double* aks,
		volatile double* varTmpInt) {
		int i;
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + b21 * h * dvdz[i];
		metric::derivs(varTmpInt, aks, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b31 * dvdz[i] + b32 * aks[i]);
		metric::derivs(varTmpInt, (aks + 5), b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b41 * dvdz[i] + b42 * aks[i] + b43 * aks[i + 5]);
		metric::derivs(varTmpInt, (aks + 10), b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b51 * dvdz[i] + b52 * aks[i] + b53 * aks[i + 5] + b54 * aks[i + 10]);
		metric::derivs(varTmpInt, aks + 15, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b61 * dvdz[i] + b62 * aks[i] + b63 * aks[i + 5] + b64 * aks[i + 10] + b65 * aks[i + 15]);
		metric::derivs(varTmpInt, aks + 20, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varOut[i] = var[i] + h * (c1 * dvdz[i] + c3 * aks[i + 5] + c4 * aks[i + 10] + c6 * aks[i + 20]);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varErr[i] = h * (dc1 * dvdz[i] + dc3 * aks[i + 5] + dc4 * aks[i + 10] + dc5 * aks[i + 15] + dc6 * aks[i + 20]);
	}

	__device__ __host__ static void rkqs(volatile double* var, volatile  double* dvdz, double& z, double& h,
		volatile double* varScal, const double b, const double q,
		volatile double* varErr, volatile double* varTemp, volatile double* aks, volatile double* varTmpInt) {

		rkck(var, dvdz, h, varTemp, varErr, b, q, aks, varTmpInt);
		double errmax = 0.0;
		for (int i = 0; i < NUMBER_OF_EQUATIONS; i++) errmax = fmax(errmax, fabs(varErr[i] / varScal[i]));
		errmax /= MIN_ACCURACY;
		if (errmax <= 1.0) {
			z += h;
			for (int i = 0; i < NUMBER_OF_EQUATIONS; i++) var[i] = varTemp[i];
			if (errmax > ERRCON) h = SAFETY * h * pow(errmax, PGROW);
			else h = ADAPTIVE * h;
		}
		else {
			h = fmin(SAFETY * h * pow(errmax, PSHRNK), 0.1 * h);
		}

		//TODO: Why min step size 7x slower?
	}

	__device__ __host__ static constexpr int sgn(float val) {
		return (0 < val) - (val < 0);
	}

	__device__ __host__ static void odeint1(volatile double* varStart, const double b, const double q) {
		volatile double varScal[5];
		volatile double var[5];
		volatile double dvdz[5];
		volatile double varErr[5];
		volatile double varTemp[5];
		volatile double aks[25];
		volatile double varTmpInt[5];

		double z = 0.0;
		double h = INITIAL_STEP_SIZE * sgn(INTEGRATION_MAX);

		for (int i = 0; i < NUMBER_OF_EQUATIONS; i++) var[i] = varStart[i];

		for (int nstp = 0; nstp < MAXSTP; nstp++) {
			metric::derivs(var, dvdz, b, q);
			for (int i = 0; i < NUMBER_OF_EQUATIONS; i++)
				varScal[i] = fabs(var[i]) + fabs(dvdz[i] * h) + TINY;

			rkqs(var, dvdz, z, h, varScal, b, q, varErr, varTemp, aks, varTmpInt);
			if (z <= INTEGRATION_MAX) {
				for (int i = 0; i < NUMBER_OF_EQUATIONS; i++) varStart[i] = var[i];
				return;
			}
		}
	};



	__device__ __host__ void rkckIntegrate1(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV) {

		volatile double varStart[] = { rV, thetaV, phiV, *pRV, *pThetaV };

		odeint1(varStart, *bV, *qV);

		*bV = varStart[1];
		*qV = varStart[2];
		wrapToPi(*bV, *qV);

	}

	__global__ void integrate_kernel(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, int size) {
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < size) {
			rkckIntegrate1(rV, thetaV, phiV, &pRV[index], &bV[index], &qV[index], &pThetaV[index]);
		}
	};
}
