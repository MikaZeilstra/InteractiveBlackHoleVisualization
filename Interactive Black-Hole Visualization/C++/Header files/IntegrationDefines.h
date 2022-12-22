#pragma once
#define rVar var[0]
#define thetaVar var[1]
#define phiVar var[2]
#define pRVar var[3]
#define pThetaVar var[4]


#define r_index 0
#define theta_index 1
#define phi_index 2
#define pR_index 3
#define pTheta_index 4


#define drdz varOut[0]
#define dtdz varOut[1]
#define dpdz varOut[2]
#define dprdz varOut[3]
#define dptdz varOut[4]

//#define A *a
//#define asq *asq

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
#define MAXSTP 1000
#define TINY 1.0e-30
#define ADAPTIVE 5.0

#define INITIAL_STEP_SIZE 1e-2
#define NUMBER_OF_EQUATIONS 5
#define MIN_ACCURACY 1e-5
#define MIN_STEP_SIZE -1e-5
#define INTEGRATION_MAX  -1e7

#define STEP_SAVE_INTERVAL 2

#define BUTCHER_TABLEAU b21 = 0.2, \
b31 = 3.0 / 40.0, b32 = 9.0 / 40.0,\
b41 = 0.3, b42 = -0.9, b43 = 1.2,\
b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0, \
b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0,\
c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0
#define BUTCHER_ERROR dc1 = 37.0 / 378.0 - 2825.0 / 27648.0, dc3 = 250.0 / 621.0 - 18575.0 / 48384.0,\
dc4 = 125.0 / 594.0 - 13525.0 / 55296.0, dc5 = -277.00 / 14336.0, dc6 = 512.0 / 1771.0 - 0.25

#ifndef __CUDA_ARCH__
#define BH_A metric::a<T>
#define BH_ASQ metric::asq<T>
#else
#define BH_A metric::a_dev<T>
#define BH_ASQ metric::asq_dev<T>
#endif // !__CUDA_ARCH__

#ifndef __CUDA_ARCH__
#define BH_AT metric::a<double>
#define BH_ASQT metric::asq<double>
#else
#define BH_AT metric::a_dev<double>
#define BH_ASQT metric::asq_dev<double>
#endif // !__CUDA_ARCH__
