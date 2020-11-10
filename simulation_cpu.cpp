#include "simulation_cpu.h"

#include <iostream>
#include "mkl.h"
#include "mkl_vsl.h"

#define _TIMEPOINTS 51

struct __float2 {
    float x;
    float y;
};

/*
* Musiela SDE
*/
float __musiela_sde(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {

    float v0 = (vol0 * phi0) * sqrt_dt;
    float v1 = (vol1 * phi1) * sqrt_dt;
    float v2 = (vol2 * phi2) * sqrt_dt;

    float dfbar = drift * dt;
    dfbar += v0;
    dfbar += v1;
    dfbar += v2;

    dfbar += (dF / dtau) * dt;

    // apply Euler Maruyana
    float result = rate0 + dfbar;

    return result;
}


/*
 * Random Number generation
 */

void __initRNG2_kernel(float* rngNrmVar, const unsigned int seed, int rnd_count)
{
   VSLStreamStatePtr stream;
   vslNewStream(&stream, VSL_BRNG_MT19937, 777);
   vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, rnd_count, rngNrmVar, 0.0, 1.0);
   vslDeleteStream(&stream);
}


/*
 * Monte Carlo HJM Path Generation
*/

void __generatePaths_kernel(__float2* numeraires, int timepoints, float* spot_rates, float* drifts, float* volatilities, float* rngNrmVar, const int pathN, float dtau = 0.5, float dt = 0.01)
{
    // Simulated forward Curve
    static float simulated_rates[_TIMEPOINTS];
    static float simulated_rates0[_TIMEPOINTS];
    static float accum_rates[_TIMEPOINTS];

    // Normal variates
    float phi0;
    float phi1;
    float phi2;

    // calculated rate
    float rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const float sqrt_dt = sqrt(dt);
    int sim_blck_count = pathN / stride;

    // Initialize simulated_rates with the spot_rate values
    for (int t = 0; t < _TIMEPOINTS; t++) {
        simulated_rates[t] = spot_rates[t];
    }

    numeraires[0].x = simulated_rates[0];
    numeraires[0].y = exp(-simulated_rates[0] * dt);

    //  HJM SDE Simulation
    for (int sim_blck = 0; sim_blck < sim_blck_count; sim_blck++)
    {

        for (int sim = 1; sim <= stride; sim++)
        {
            //  initialize the random numbers phi0, phi1, phi2 for the simulation (sim) for each t,  t[i] = t[i-1] + dt
            phi0 = rngNrmVar[sim - 1];
            phi1 = rngNrmVar[sim];
            phi2 = rngNrmVar[sim + 1];

            for (int t = 0; t < _TIMEPOINTS; t++) 
            {
                float dF = 0.0;

                if (t == (_TIMEPOINTS - 1)) {
                    dF = simulated_rates[t] - simulated_rates[t-1];
                }
                else {
                    dF = simulated_rates[t+1] - simulated_rates[t];
                }

                rate = __musiela_sde(
                    drifts[t],
                    volatilities[t],
                    volatilities[_TIMEPOINTS + t],
                    volatilities[_TIMEPOINTS * 2 + t],
                    phi0,
                    phi1,
                    phi2,
                    sqrt_dt,
                    dF,
                    simulated_rates[t],
                    dtau,
                    dt
                );

                // accumulate rate for discount calculation
                accum_rates[t] += rate;

                // store the simulated rate
                simulated_rates0[t] = rate;
            }

            // update simulated rates
            for (int t = 0; t < _TIMEPOINTS; t++)
            {
                simulated_rates[t] = simulated_rates0[t];
            }

            // DEBUG
            printf("%d - %d ", sim_blck, sim);
            for (int t = 0; t < _TIMEPOINTS; t++)
            {
                printf("%f ", simulated_rates[t]);
            }
            printf("    %f %f %f \n", phi0, phi1, phi2);
        } 

        // update numeraire based on simulation block 
        numeraires[sim_blck+1].x = simulated_rates[sim_blck + 1];
        numeraires[sim_blck+1].y = exp(-accum_rates[sim_blck] * dt);
    }

    // DEBUG
    printf("Forward Rates/ Discount Factors \n");
    for (int t = 0; t < _TIMEPOINTS; t++)
    {
        printf("%f %f \n", numeraires[t].x, numeraires[t].y);
    }
}


/*
 * Exposure generation kernel
 * one to one mapping between threadIdx.x and tenor
 */
void __reduceExposure_kernel(float* exposure, __float2* numeraires, const float notional, const float K, float* accrual, int simN)
{
    static float cash_flows[_TIMEPOINTS];
    float discount_factor;
    float forward_rate;
    float sum = 0.0;

    // For each Simulation
    for (int s = 0; s < simN; s++) {
        // Calculate the Cashflows across tenors
        for (int t = 0; t < _TIMEPOINTS; t++) {
            forward_rate = numeraires[t].x;
            discount_factor = numeraires[t].y;
            cash_flows[t] = discount_factor * notional * accrual[t] * (forward_rate - K);
        }
        // Compute the IRS Mark to Market
        for (int t = 0; t <= _TIMEPOINTS; t++) {
            sum = 0.0;
            for (int i = t + 1; i < _TIMEPOINTS; i++) {
                sum += cash_flows[i];
            }
            sum = (sum > 0.0) ? sum : 0.0;
            exposure[t] = sum;
        }     
    }
}


/*
* Aggregation
*/
void cpuMatrix2DReduceAvg(float* expected_exposure, float* exposures, int simN) {

    float* d_x = 0;;
    float* d_y = 0;

    d_x = (float*) malloc(simN * sizeof(float));
    d_y = (float*) malloc(_TIMEPOINTS * sizeof(float));
   
    memset(d_x, 1.0, simN);
    memset(d_y, 0.0, _TIMEPOINTS);

    // Parameters
    const float alpha = 1.f / simN;
    float cols = (float) _TIMEPOINTS;
    float rows = (float) simN;
    const MKL_INT lda = simN;
    const MKL_INT incx = 1;
    const MKL_INT incy = 1;
    const float beta = 0.0;

    // Matrix Vector Multiplication to Reduce a Matrix by columns
    cblas_sgemv(CblasRowMajor, CblasNoTrans, _TIMEPOINTS, simN, alpha, exposures, lda, d_x, incx, beta, d_y, incy);

    // copy the results back
    for (int i = 0; i < _TIMEPOINTS; i++) {
        expected_exposure[i] = d_y[i];
    }

    // free resource
    if (d_x) {
        free(d_x);
    }

    if (d_y) {
        free(d_y);
    }
}


/*
 * Exposure Calculation Kernel Invocation
*/
void calculateExposureCPU(/*float* expected_exposure*/ float* exposures, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int simN) //dt, dtau
{
    // Allocate Expected Exposure profile
    float* expected_exposure = 0;
    expected_exposure = (float*) malloc(_TIMEPOINTS * sizeof(float));

    // Iterate across all simulations
    int pathN = 2500; // HJM Model simulation paths number
    int rnd_count = simN * pathN * 3; //

    // Allocate numeraires (forward_rate, discount_factor)
    __float2* numeraires = 0;
    numeraires = (__float2*) malloc(rnd_count * sizeof(__float2));

    // Generate the normal distributed variates
    float* rngNrmVar = 0;
    rngNrmVar = (float*) malloc(rnd_count * sizeof(float));


    // normal distributed variates generation
    __initRNG2_kernel(rngNrmVar, 1234L, rnd_count);

    for (int s = 0; s < simN; s++) {
        __generatePaths_kernel(numeraires, _TIMEPOINTS, spot_rates, drift, volatilities, &rngNrmVar[ s*pathN*3 ], pathN); //dt, dtau
        //__reduceExposure_kernel(float* exposure, float* numeraires, const float notional, const float K, float* accrual, int simN)
    }

    // Calculate the Expected Exposure Profile
    //cpuMatrix2DReduceAvg(expected_exposure, exposures, simN);


    // free resources
    if (rngNrmVar) {
        free(rngNrmVar);
    }

    if (numeraires) {
        free(numeraires);
    }

    if (expected_exposure) {
        free(expected_exposure);
    }

    /* Printing results */
    //std::cout << "Sample mean of normal distribution = " << std::endl;
}