#include "product.h"

#ifndef EXPOSURE_CPU
#define EXPOSURE_CPU

/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureCPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int exposuresCount, float dt);

#endif
