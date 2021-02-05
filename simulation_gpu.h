#include "product.h"

#ifndef EXPOSURE_GPU_CUDA
#define EXPOSURE_GPU_CUDA

/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(float* exposures, InterestRateSwap payOff, float *accrual, float* spot_rates, float* drift, float* volatilities, int simN, float dt);

void calculateExposureMultiGPU(float* exposures, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int num_gpus, int simN, float dt);

#endif