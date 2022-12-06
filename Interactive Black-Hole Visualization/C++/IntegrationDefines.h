#define rVar var[0]
#define thetaVar var[1]
#define phiVar var[2]
#define pRVar var[3]
#define pThetaVar var[4]

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

#define INITIAL_STEP_SIZE 0.01
#define NUMBER_OF_EQUATIONS 5
#define MIN_ACCURACY 1e-5
//#define MIN_STEP_SIZE 0.00001
#define INTEGRATION_MAX  -10000000