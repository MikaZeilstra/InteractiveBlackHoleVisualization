#pragma once
//Macros for variables when integrating
#define rVar var[0]
#define thetaVar var[1]
#define phiVar var[2]
#define pRVar var[3]
#define pThetaVar var[4]

//Macros for indexes of variables when integrating
#define r_index 0
#define theta_index 1
#define phi_index 2
#define pR_index 3
#define pTheta_index 4

//Macros for derivatives of variables when integrating
#define drdz varOut[0]
#define dtdz varOut[1]
#define dpdz varOut[2]
#define dprdz varOut[3]
#define dptdz varOut[4]

//The datatype used for integration
#define INTEGRATION_PRECISION_MODE double

#define SAFETY 0.9
#define PGROW -0.2
#define PSHRNK -0.25
#define ERRCON 1.89e-4
//Maximum number of steps untill we assume the ray to be in the black hole
#define MAXSTP 1000
//Epsilon
#define TINY 1.0e-30
#define ADAPTIVE 5.0

//Starting step size
#define INITIAL_STEP_SIZE 1e-2
//Total number of equations in the system of ODEs 
#define NUMBER_OF_EQUATIONS 5
//Mininmum accuracy untill we lower the step size
#define MIN_ACCURACY 1e-5
//Minimum step size
#define MIN_STEP_SIZE -1e-7
//Total sum of steps lenghts until we consider the ray going straight again 
#define INTEGRATION_MAX  -1e7
//The minimum step size taken when accuratly determining disk intersection
#define MIN_DISK_STEP_SIZE -1e-1
//If the path is saved, after how many steps we will save the coordinates
#define STEP_SAVE_INTERVAL 2

//Butcher tablaeu used in RK integration
#define BUTCHER_TABLEAU b21 = 0.2, \
b31 = 3.0 / 40.0, b32 = 9.0 / 40.0,\
b41 = 0.3, b42 = -0.9, b43 = 1.2,\
b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0, \
b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0,\
c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0
#define BUTCHER_ERROR dc1 = 37.0 / 378.0 - 2825.0 / 27648.0, dc3 = 250.0 / 621.0 - 18575.0 / 48384.0,\
dc4 = 125.0 / 594.0 - 13525.0 / 55296.0, dc5 = -277.00 / 14336.0, dc6 = 512.0 / 1771.0 - 0.25

//The minimum radius of the accretion disk
#define MIN_STABLE_ORBIT 6


//Macros for variables which have a Device and host version
#ifndef __CUDA_ARCH__
#define BH_A metric::a<T>
#define BH_ASQ metric::asq<T>
#define BH_MAX_ACCRETION_RADIUS metric::accretionDiskRadius<T>
#define BH_USE_ACCRETION_DISK metric::useAccretionDisk
#else
#define BH_A metric::a_dev<T>
#define BH_ASQ metric::asq_dev<T>
#define BH_MAX_ACCRETION_RADIUS metric::accretionDiskRadius_dev<T>
#define BH_USE_ACCRETION_DISK metric::useAccretionDisk_dev
#endif // !__CUDA_ARCH__