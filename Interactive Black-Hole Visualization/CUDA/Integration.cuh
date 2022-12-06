#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../C++/Const.h"
#include "../C++/IntegrationDefines.h"


namespace integrate_device {
	__global__ void integrate_kernel(const double rV, const double thetaV, const double phiV, double* pRV,
		double* bV, double* qV, double* pThetaV, int size);

	//__device__ double a = 0;
};