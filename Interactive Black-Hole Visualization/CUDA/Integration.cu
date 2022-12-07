#pragma once
#include "Integration.cuh"

#include <stdio.h>

__device__ __constant__ float ap;



namespace integrate_device {

	void copy_a(double a) {
		cudaMemcpyToSymbol(integrate_device::a, &a, sizeof(double));
		double asq = a * a;
		cudaMemcpyToSymbol(integrate_device::asq, &asq, sizeof(double));
	};

	__device__ const double b21 = 0.2,
		b31 = 3.0 / 40.0, b32 = 9.0 / 40.0, b41 = 0.3, b42 = -0.9, b43 = 1.2,
		b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0,
		b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0,
		b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0, c1 = 37.0 / 378.0,
		c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0,
		dc5 = -277.00 / 14336.0;
	__device__ const double dc1 = 37.0 / 378.0 - 2825.0 / 27648.0, dc3 = 250.0 / 621.0 - 18575.0 / 48384.0,
		dc4 = 125.0 / 594.0 - 13525.0 / 55296.0, dc6 = 512.0 / 1771.0 - 0.25;

	__device__ static double sq(double x) {
		return x * x;
	}

	__device__ static double sq3(double x) {
		return x * x * x;
	}

	__device__ static double _Delta(double r) {
		return sq(r) - 2. * r + asq;
	};

	__device__ static double _Sigma(double r, double theta) {
		return sqrt(sq(sq(r) + asq) - asq * _Delta(r) * sq(sin(theta)));
	};

	__device__ static double _w(double r, double theta) {
		return 2. * a * r / sq(_Sigma(r, theta));
	};

	__device__ static double _ro(double r, double theta) {
		return sqrt(sq(r) + asq * sq(cos(theta)));
	};

	__device__ static double _rosq(double r, double theta) {
		return sq(r) + asq * sq(cos(theta));
	};

	__device__ static double _wbar(double r, double theta) {
		return _Sigma(r, theta) * sin(theta) / _ro(r, theta);
	};

	__device__ static double _alpha(double r, double theta) {
		return _ro(r, theta) * sqrt(_Delta(r)) / _Sigma(r, theta);
	};

	__device__ static double _P(double r, double b) {
		return sq(r) + asq - a * b;
	}

	__device__ static double _R(double r, double theta, double b, double q) {
		return sq(_P(r, b)) - _Delta(r) * (sq((b - a)) + q);
	};

	__device__ static double _BigTheta(double r, double theta, double b, double q) {
		return q - sq(cos(theta)) * (sq(b) / sq(sin(theta)) - asq);
	};


	__device__ static void wrapToPi(double& thetaW, double& phiW) {
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

	__device__ static void wrapToPi(float& thetaW, float& phiW) {
		thetaW = fmod(thetaW, PI2);
		if (thetaW < 0) {
			thetaW += PI2;
		}

		if (thetaW > PI) {
			thetaW -= 2 * (thetaW - PI);
			phiW += PI;
		}
		while (phiW < 0) phiW += PI2;
		phiW = fmod(phiW, PI2);
		if (phiW < 0) {
			phiW += PI2;
		}
	}

	__device__ static void derivs(volatile double* var, volatile double* varOut, double b, double q) {
		double cosv = cos(thetaVar);
		double sinv = sin(thetaVar);
		double cossq = sq(cosv);
		double sinsq = sq(sinv);
		double bsq = sq(b);
		double delta = _Delta(rVar);
		double rosq = _rosq(rVar, thetaVar);
		double P = _P(rVar, b);
		double prsq = sq(pRVar);
		double pthetasq = sq(pThetaVar);
		double R = _R(rVar, thetaVar, b, q);
		double partR = (q + sq(a - b));
		double btheta = _BigTheta(rVar, thetaVar, b, q);
		double rosqsq = sq(2 * rosq);
		double sqrosqdel = (sq(rosq) * delta);
		double asqcossin = asq * cosv * sinv;
		double rtwo = 2 * rVar - 2;

		drdz = delta / rosq * pRVar;
		dtdz = 1. / rosq * pThetaVar;
		dpdz = (2 * a * P - (2 * a - 2 * b) * delta + (2 * b * cossq * delta) / sinsq) / (rosq * 2 * delta);
		dprdz = (rtwo * btheta - rtwo * partR + 4 * rVar * P) / (rosq * (2 * delta)) - (prsq * rtwo) / (2 * rosq)
			+ (4 * pthetasq * rVar) / rosqsq - ((4 * rVar - 4) * (btheta * (delta)+R)) / (rosq * sq(2 * delta))
			+ (4 * prsq * rVar * (delta)) / rosqsq - (rVar * (btheta * delta + R)) / sqrosqdel;
		dptdz = ((2 * cosv * sinv * (bsq / sinsq - asq) + (2 * bsq * sq3(cosv)) / sq3(sinv)) * delta) /
			(rosq * 2 * delta) - (4 * asqcossin * pthetasq) / rosqsq - (4 * asqcossin * prsq * delta) /
			rosqsq + (asqcossin * (btheta * delta + R)) / sqrosqdel;
	}

	__device__ static void rkck(volatile double* var, volatile double* dvdz, const double h,
		volatile double* varOut, volatile volatile double* varErr, const double b, const double q, volatile double* aks,
		volatile double* varTmpInt) {
		int i;
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + b21 * h * dvdz[i];
		derivs(varTmpInt, aks, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b31 * dvdz[i] + b32 * aks[i]);
		derivs(varTmpInt, (aks + 5), b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b41 * dvdz[i] + b42 * aks[i] + b43 * aks[i + 5]);
		derivs(varTmpInt, (aks + 10), b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b51 * dvdz[i] + b52 * aks[i] + b53 * aks[i + 5] + b54 * aks[i + 10]);
		derivs(varTmpInt, aks + 15, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varTmpInt[i] = var[i] + h * (b61 * dvdz[i] + b62 * aks[i] + b63 * aks[i + 5] + b64 * aks[i + 10] + b65 * aks[i + 15]);
		derivs(varTmpInt, aks + 20, b, q);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varOut[i] = var[i] + h * (c1 * dvdz[i] + c3 * aks[i + 5] + c4 * aks[i + 10] + c6 * aks[i + 20]);
		for (i = 0; i < NUMBER_OF_EQUATIONS; i++)
			varErr[i] = h * (dc1 * dvdz[i] + dc3 * aks[i + 5] + dc4 * aks[i + 10] + dc5 * aks[i + 15] + dc6 * aks[i + 20]);
	}

	__device__ static void rkqs(volatile double* var, volatile  double* dvdz, double& z, double& h,
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

	__device__ static constexpr int sgn(float val) {
		return (0 < val) - (val < 0);
	}

	__device__ static void odeint1(volatile double* varStart,  const double b, const double q) {
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
			derivs(var, dvdz, b, q);
			for (int i = 0; i < NUMBER_OF_EQUATIONS; i++)
				varScal[i] = fabs(var[i]) + fabs(dvdz[i] * h) + TINY;

			rkqs(var, dvdz, z, h, varScal, b, q, varErr, varTemp, aks, varTmpInt);
			if (z <= INTEGRATION_MAX) {
				for (int i = 0; i < NUMBER_OF_EQUATIONS; i++) varStart[i] = var[i];
				return;
			}
		}
	};



	__device__ static void rkckIntegrate1(const double rV, const double thetaV, const double phiV,  double *pRV,
		 double *bV,  double* qV,  double* pThetaV) {

		volatile double varStart[] = { rV, thetaV, phiV, *pRV, *pThetaV };

		odeint1(varStart, *bV,*qV);

		*bV = varStart[1];
		*qV = varStart[2];
		wrapToPi(*bV, *qV);

	}

	__global__ void integrate_kernel(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, int size){
		int index = blockDim.x * blockIdx.x + threadIdx.x;
		if (index < size) {
			rkckIntegrate1(rV, thetaV, phiV , &pRV[index], &bV[index], &qV[index], &pThetaV[index]);
		}
	};
}