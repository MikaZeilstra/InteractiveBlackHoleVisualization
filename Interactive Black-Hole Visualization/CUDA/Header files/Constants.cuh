#pragma once
#include "cuda_runtime.h"

#define R_CHANGE_THRESHOLD 10

#define TILE_W 4
#define TILE_H 4

#define ijc (i*M+j)

#define N1 (N+1)
#define M1 (M+1)
#define cam_speed cam[0]
#define cam_alpha cam[1]
#define cam_w cam[2]
#define cam_wbar cam[3]
#define cam_br cam[4]
#define cam_btheta cam[5]
#define cam_bphi cam[6]

#define SQRT2PI 2.506628274631f

#define PI2c 6.283185307179586476f
#define PIc 3.141592653589793238f

#define INFINITY 1e30
#define INFINITY_CHECK 5e29
#define PI 3.141592653589793238
#define PI1_2  (PI / 2.)
#define PI2 (PI*2.)
#define PI1_4 (PI / 4.)
#define HALF (1. / 2.)

#define SQRT3 1.73205080757
#define SQRT6 2.44948974278