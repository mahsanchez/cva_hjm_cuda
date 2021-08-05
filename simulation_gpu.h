#include "product.h"

#ifndef EXPOSURE_GPU_CUDA
#define EXPOSURE_GPU_CUDA

/*
   Exposure Calculation Kernel Invocation
*/

void calculateExposureMultiGPU(double* expected_exposure, InterestRateSwap<double> payOff, double* accrual, double* spot_rates, double* drift, double* volatilities, double scale, int num_gpus, int simN, double dt);

void calculateExposureMultiGPU(float* expected_exposure, InterestRateSwap<float> payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, float scale, const int num_gpus, int scenarios, float dt);

#endif