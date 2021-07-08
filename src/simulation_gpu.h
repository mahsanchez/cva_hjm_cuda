#include "product.h"

#ifndef EXPOSURE_GPU_CUDA
#define EXPOSURE_GPU_CUDA

/*
   Exposure Calculation Kernel Invocation
*/
//void calculateSExposureGPU(double* exposures, InterestRateSwap payOff, double *accrual, double* spot_rates, double* drift, double* volatilities, int simN, double dt);

void calculateExposureMultiGPU(double* exposures, InterestRateSwap_D payOff, double* accrual, double* spot_rates, double* drift, double* volatilities, double scale, int num_gpus, int simN, double dt);

#endif