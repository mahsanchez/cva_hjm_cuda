#include "product.h"

#ifndef EXPOSURE_CPU
#define EXPOSURE_CPU

/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureCPU(float* exposures, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int simN);

#endif
