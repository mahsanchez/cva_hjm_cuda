#include "product.h"

#ifndef EXPOSURE_GPU_CUDA
#define EXPOSURE_GPU_CUDA

/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(double* exposures, InterestRateSwap& payOff, double* spot_rates, double* drift, double* volatilities, int simN, int size);

#endif