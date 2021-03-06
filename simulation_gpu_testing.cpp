#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <stdlib.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>

#include "simulation_cpu.h"
#include "simulation_gpu.h"

#define CPU_SIMULATION1
#define GPU_SIMULATION 
#define EXPECTED_EXPOSURE_DEBUG 

/*
 * Testing HJM model accelerated in GPU CUDA
 */

 // First row is the last observed forward curve (BOE data)
float spot_rates[51] = {
    0.046138361,0.045251174,0.042915805,0.04283311,0.043497719,0.044053792,0.044439518,0.044708496,0.04490347,0.045056615,0.045184474,0.045294052,0.045386152,0.045458337,0.045507803,0.045534188,
    0.045541867,0.045534237,0.045513128,0.045477583,0.04542292,0.045344477,0.04523777,0.045097856,0.044925591,0.04472353,0.044494505,0.044242804,0.043973184,0.043690404,0.043399223,0.043104398,
    0.042810688,0.042522852,0.042244909,0.041978295,0.041723875,0.041482518,0.04125509,0.041042459,0.040845492,0.040665047,0.040501255,0.040353009,0.040219084,0.040098253,0.039989288,0.039890964,
    0.039802053,0.039721437,0.03964844
};

// Volatility Calibration
/*
 * HJM Calibration (volatilities, drift)
 */
float volatilities[153] =
{
0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
0.006430655,0.006430655,0.006430655,

-0.003556543,-0.003811644,-0.004010345,-0.004155341,-0.004249328,-0.004295001,-0.004295055,-0.004252185,-0.004169088,-0.004048459,-0.003892993,-0.003705385,-0.003488331,-0.003244526,-0.002976667,-0.002687447,
-0.002379563,-0.002055710,-0.001718584,-0.001370879,-0.001015292,-0.000654518,-0.000291251,0.000071811,0.000431975,0.000786544,0.001132823,0.001468117,0.001789731,0.002094968,0.002381133,0.002645532,
0.002885468,0.003098246,0.003281171,0.003431548,0.003546680,0.003623873,0.003660430,0.003653657,0.003600859,0.003499339,0.003346403,0.003139354,0.002875498,0.002552139,0.002166581,0.001716130,
0.001198090,0.000609765,-0.000051541,

-0.004750672,-0.003908573,-0.003134891,-0.002427280,-0.001783395,-0.001200891,-0.000677421,-0.000210640,0.000201797,0.000562236,0.000873023,0.001136502,0.001355020,0.001530923,0.001666555,0.001764262,
0.001826390,0.001855285,0.001853291,0.001822755,0.001766022,0.001685437,0.001583346,0.001462095,0.001324030,0.001171495,0.001006836,0.000832399,0.000650530,0.000463573,0.000273875,0.000083782,
-0.000104363,-0.000288212,-0.000465420,-0.000633643,-0.000790533,-0.000933746,-0.001060936,-0.001169758,-0.001257866,-0.001322914,-0.001362557,-0.001374449,-0.001356245,-0.001305600,-0.001220167,-0.001097601,
-0.000935557,-0.000731689,-0.000483652
};

// Drift Callibration
float drifts[51] = {
0.0000000,0.0000362,0.0000689,0.0000992,0.0001278,0.0001552,0.0001817,0.0002075,0.0002326,0.0002569,0.0002804,0.0003029,0.0003244,0.0003446,0.0003636,0.0003812,
0.0003974,0.0004122,0.0004257,0.0004380,0.0004491,0.0004593,0.0004689,0.0004779,0.0004868,0.0004957,0.0005050,0.0005149,0.0005257,0.0005377,0.0005511,0.0005660,
0.0005826,0.0006009,0.0006211,0.0006430,0.0006667,0.0006919,0.0007184,0.0007460,0.0007744,0.0008033,0.0008323,0.0008610,0.0008892,0.0009164,0.0009425,0.0009674,
0.0009909,0.0010133,0.0010350
};


//Year Count Fractiions 30/360
std::vector<float> yearCountFractions = {
    0.04000, 0.00250, 0.25278,  0.2500642, 0.25556,  0.2513385, 0.25556,  0.25000,  0.25278,  0.25556,  0.25556,
    0.25556, 0.25556,  0.25278,  0.25278, 0.26111, 0.25278,  0.25278,  0.25278, 0.25278, 0.25278,
    0.25556, 0.25556,  0.25278,  0.25278, 0.26111, 0.25278,  0.25278,  0.25278, 0.25278, 0.25278, 
    0.25278, 0.25278 , 0.25278  , 0.25556, 0.25556, 0.25694, 0.25556, 0.25278, 0.25278, 0.25556,
    0.25556, 0.25556,  0.25278,  0.25278, 0.26111, 0.25278,  0.25278,  0.25278, 0.25278, 0.25278
};

//Year Count Fractiions 30/360
std::vector<float> accrual = {
    0.505556, 0.511111, 0.502778, 0.511111, 0.502778, 0.511111, 0.502778,  0.516667, 0.505556, 0.505556, 0.505556,
    0.505556, 0.511111, 0.502778, 0.511111, 0.502778, 0.511111, 0.502778,  0.516667, 0.505556, 0.505556,
    0.505556, 0.511111, 0.502778, 0.511111, 0.502778, 0.511111, 0.502778,  0.516667, 0.505556, 0.505556,
    0.505556, 0.511111, 0.502778, 0.511111, 0.502778, 0.511111, 0.502778,  0.516667, 0.505556, 0.505556,
    0.505556, 0.511111, 0.502778, 0.511111, 0.502778, 0.511111, 0.502778,  0.516667, 0.505556, 0.505556,
};


/*
* Interest Rate Swap Product Definition 5Y Floating, 5Y Fixing 3M reset
*/
const float expiry = 25.0; // 25.0;
const float dtau = 0.5;

std::vector<float> floating_schedule = {
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 
        19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0
};

std::vector<float> fixed_schedule = {
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0,
        10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5, 19.0,
        19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0
};


template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}



void testSimulationCPU(float* exposure_curve, int exposuresCount, InterestRateSwap& payOff, const float dt) {

    calculateExposureCPU(exposure_curve, payOff, &accrual[0], &spot_rates[0], &drifts[0], &volatilities[0], exposuresCount, dt);
}


void testSimulationMultiGPU(float* exposure_curve, const int num_gpus, int scenarios, InterestRateSwap& payOff, const float dt) {

    calculateExposureMultiGPU(exposure_curve, payOff, accrual.data(), spot_rates, drifts, volatilities, num_gpus, scenarios, dt);
}

/* TODO Today and measure throughput simulations/seconds */


int main(int argc, char** argv)
{
    int num_gpus = 0;            
    const int timepoints = 51;
    InterestRateSwap payOff(&floating_schedule[0], &floating_schedule[0], &fixed_schedule[0], 10, 0.06700, expiry, dtau);

    std::cout << "valid arguments [-cpu|-single_gpu|-multigpu] -scenarios [number] -dt [0.01 - 0.001]" << std::endl << std::endl;

    const bool cpu = get_arg(argv, argv + argc, "-cpu");
    if (!cpu) {
        const bool multi_gpu = get_arg(argv, argv + argc, "-multi_gpu");
        num_gpus = (multi_gpu) ? 4 : 1;
    }

    int scenarios = get_argval(argv, argv + argc, "-scenarios", 100000); 
    float dt = get_argval(argv, argv + argc, "-dt", 0.01);

    std::cout << "## scenarios " << scenarios << " simulations per scenarios" << payOff.expiry/dt << std::endl;

    float* expected_exposure = (float*)malloc(51 * sizeof(float));

    if (num_gpus == 0)  {
        std::cout << "## cpu measurements" << std::endl;
        testSimulationCPU(expected_exposure, scenarios, payOff, dt);
    }
    else {
        std::cout << "## Gpu measurements for num_gpus: " << num_gpus << std::endl;
        testSimulationMultiGPU(expected_exposure, num_gpus, scenarios, payOff, dt);
    }

#ifdef EXPECTED_EXPOSURE_DEBUG
    printf("Expected Exposure Profile\n");
    for (int t = 0; t < timepoints; t++) {
        printf("%1.6f ", expected_exposure[t]);
    }
    printf("\n");
#endif

    if (expected_exposure) {
        free(expected_exposure);
    }

    exit(0);
}