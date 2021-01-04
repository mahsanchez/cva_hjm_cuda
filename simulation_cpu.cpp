#include "simulation_cpu.h"

#include <iostream>
#include <chrono>
#include "mkl.h"
#include "mkl_vsl.h"

#define _TIMEPOINTS 51

#undef DEBUG_HJM_SIM
#undef DEBUG_NUMERAIRE
#undef DEBUG_EXPOSURE
#define DEBUG_EXPECTED_EXPOSURE

struct __float2 {
    float x;
    float y;
};

/*
* SDE
* We simulate f(t+dt)=f(t) + dfbar   where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt (Musiela Parameterisaiton HJM SDE)
*/
float __musiela_sde(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {

    float vol_sum = vol0 * phi0;
    vol_sum += vol1 * phi1;
    vol_sum += vol2 * phi2;
    vol_sum *= sqrt(dt);

    float dfbar = drift * dt;
    dfbar += vol_sum;

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
   //vslNewStream(&stream, VSL_BRNG_MT19937, 777);
   //vslNewStream(&stream, VSL_BRNG_MRG32K3A, 777);
   vslNewStream(&stream, VSL_BRNG_SOBOL, 777);
   vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, rnd_count, rngNrmVar, 0.0, 1.0);
   vslDeleteStream(&stream);
}

/*
 Calculate dF term in Musiela Parametrization SDE
*/

inline float dFau(int t, int timepoints, float* rates) {
    float result = 0.0;

    if (t == (timepoints - 1)) {
        result = rates[t] - rates[t - 1];
    }
    else {
        result = rates[t + 1] - rates[t];
    }

    return result;
}


/*
 * Monte Carlo HJM Path Generation
*/

void __generatePaths_kernel(__float2* numeraires, int timepoints,
    float* drifts, float* volatilities, float* rngNrmVar, 
    float* spot_rates, 
    float* simulated_rates, float* simulated_rates0, float* accum_rates,
    const int pathN, int path, 
    float dtau = 0.5, float dt = 0.01)
{
    // calculated rate
    float rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const float sqrt_dt = sqrt(dt);

    // Normal variates
    float phi0 = rngNrmVar[path*3];
    float phi1 = rngNrmVar[path*3 + 1];
    float phi2 = rngNrmVar[path*3 + 2];

    // Evolve the whole curve from 0 to T 
    for (int t = 0; t < _TIMEPOINTS; t++) 
    {
        if (path == 0) {
            rate = spot_rates[t];
        }
        else {
            float dF = dFau(t, _TIMEPOINTS, simulated_rates);

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
        }
       
        // accumulate rate for discount calculation
        accum_rates[t] += rate;

        // store the simulated rate
        simulated_rates0[t] = rate;
    }

    // update numeraire based on simulation block 
    if (path % stride == 0) {
        int lindex = path / stride;
        numeraires[lindex].x = simulated_rates0[lindex];
        numeraires[lindex].y = exp(-accum_rates[lindex] * dt);
    }

#ifdef DEBUG_HJM_SIM
    printf("%d - ", path);
    for (int t = 0; t < _TIMEPOINTS; t++)
    {
        printf("%f ", simulated_rates0[t]);
    }
    printf("\n");
#endif

} 


/*
 * Exposure generation kernel
 * one to one mapping between threadIdx.x and tenor
 */
void __calculateExposure_kernel(float* exposure, __float2* numeraires, const float notional, const float K, float* accrual, int simN)
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
* Expected Exposure profile is defined as the positive average across all simulations on each pricing point 
* For simplicity each tenor point [0..50] has been considered as a pricing point
* Rather than performing an aggreation (summation) on each column of the exposure matrix (tenors, simulations)
* a BLAS library function cblas_sgemv is used instead. Basically the exposure matrix multiplied by a vector
* of 1 values perform the summation on each column for averaging purpuse. As described bellow:
* matrix (tenors, simulations) x (simulations, 1) . The final result is a reduced vector with dimention (tenors, 1)
*/
void __calculateExpectedExposure_kernel(float* expected_exposure, float* exposures, float* d_x, float* d_y, int simN) {

    // Parameters
    const MKL_INT m = simN;
    const MKL_INT n = _TIMEPOINTS;
    const MKL_INT incx = 1;
    const MKL_INT incy = 1;
    const float alpha = 1.f/simN;
    const float beta = 0.0;

    for (int t = 0; t < _TIMEPOINTS; t++) {
        d_y[t] = 0.0;
    }

    cblas_sgemv(CblasRowMajor, CblasTrans, m, n, alpha, exposures, n, d_x, incx, beta, d_y, incy);

    printf("cblas_call\n");

    for (int t = 0; t < _TIMEPOINTS; t++) {
        expected_exposure[t] = d_y[t];
    }

}


/*
 * Exposure Calculation Kernel Invocation
*/
void calculateExposureCPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int exposuresCount, float dt) //dt, dtau
{
    //

    // Auxiliary vectors
    float* d_x = 0;;
    float* d_y = 0; 
    d_x = (float*) malloc(exposuresCount * sizeof(float));
    d_y = (float*) malloc(_TIMEPOINTS * sizeof(float));

    // Allocate numeraires (forward_rate, discount_factor)
    __float2* numeraires = 0;
    numeraires = (__float2*) malloc(_TIMEPOINTS * exposuresCount * sizeof(__float2));

    float* exposures = 0;
    exposures = (float*) malloc(_TIMEPOINTS * exposuresCount * sizeof(float));

    float* _expected_exposure = 0;
    _expected_exposure = (float*) malloc(_TIMEPOINTS * sizeof(float));

    // Simulated forward Curve
    float* simulated_rates = (float*) malloc(_TIMEPOINTS * exposuresCount * sizeof(float));;
    float* simulated_rates0 = (float*) malloc(_TIMEPOINTS * exposuresCount * sizeof(float));;
    float* accum_rates = (float*) malloc(_TIMEPOINTS * exposuresCount * sizeof(float));;

    // initialize auxiliary vectors
    for (int i = 0; i < exposuresCount; i++) {
        d_x[i] = 1.0;
    }

    // reset accumulators
    for (int t = 0; t < _TIMEPOINTS * exposuresCount; t++) {
        accum_rates[t] = 0.0;
    }

    // Iterate across all simulations
    int pathN = payOff.expiry / dt; // HJM Model simulation total paths number  

    // Number of Rnd normal variates 
    int rnd_count = exposuresCount * pathN * 3; //

    // Generate the normal distributed variates
    float* rngNrmVar = 0;
    rngNrmVar = (float*) malloc(rnd_count * sizeof(float));

    // Normal distributed variates generation
    auto t_start = std::chrono::high_resolution_clock::now();

    __initRNG2_kernel(rngNrmVar, 1234L, rnd_count);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total random normal variates " << rnd_count << " generated in " << elapsed_time_ms << "(ms)" << std::endl;
    
    // Monte Carlos Simulation HJM Kernels (2500 paths)
    
    t_start = std::chrono::high_resolution_clock::now();

    for (int path = 0; path < pathN ; path++) 
    {
            for (int s = 0; s < exposuresCount; s++) 
            {
                __generatePaths_kernel(
                     &numeraires[s*_TIMEPOINTS], 
                    _TIMEPOINTS, 
                    drift, 
                    volatilities, 
                    &rngNrmVar[s*pathN*3], 
                    spot_rates,
                    &simulated_rates[s*_TIMEPOINTS], 
                    &simulated_rates0[s*_TIMEPOINTS],
                    &accum_rates[s*_TIMEPOINTS], 
                    pathN,
                    path
                ); //dt = 0.01, dtau = 0.5 
            }
            // update simulated rates (swap pointers)
            std::swap(simulated_rates, simulated_rates0);
    }

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run all " << pathN * exposuresCount << " HJM MC simulation " << elapsed_time_ms << "(ms)" << std::endl;

    // Exposure Calculation

    t_start = std::chrono::high_resolution_clock::now();

    for (int s = 0; s < exposuresCount; s++) {
        __calculateExposure_kernel(&exposures[s * _TIMEPOINTS], &numeraires[s * _TIMEPOINTS], payOff.notional, payOff.K, accrual, exposuresCount);
    }

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run all " << exposuresCount << " exposure profile calculation " << elapsed_time_ms << "(ms)" << std::endl;

    // Calculate the Expected Exposure Profile (EE[t])

    t_start = std::chrono::high_resolution_clock::now();

    __calculateExpectedExposure_kernel(_expected_exposure, exposures, d_x, d_y, exposuresCount);

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run" << exposuresCount << " expected exposure profile " << elapsed_time_ms << "(ms)" << std::endl;

#ifdef DEBUG_NUMERAIRE
    printf("Forward Rates/ Discount Factors \n");
    for (int t = 0; t < _TIMEPOINTS; t++)
    {
        printf("%f %f \n", numeraires[t].x, numeraires[t].y);
    }
#endif

#ifdef DEBUG_EXPOSURE
    printf("Exposures\n");
    for (int s = 0; s < simN; s++) {
        for (int t = 0; t < _TIMEPOINTS; t++)
        {
            printf("%f ", exposures[s * _TIMEPOINTS + t]);
        }
        printf("\n");
    }
#endif

#ifdef DEBUG_EXPECTED_EXPOSURE
    printf("Expected Exposures \n");
    for (int t = 0; t < _TIMEPOINTS; t++)
    {
        printf("%f ", _expected_exposure[t]);
    }
#endif

    // free resources
    if (d_x) {
     // free(d_x);
    }

    if (d_y) {
      //free(d_y);
    }

    if (rngNrmVar) {
       free(rngNrmVar);
    }

    if (numeraires) {
      // free(numeraires);
    }

    if (simulated_rates) {
      free( simulated_rates);
    }

    if (simulated_rates0) {
       free(simulated_rates0);
    }

    if (accum_rates) {
       free(accum_rates); 
    }

    if (exposures) {
      //free(exposures);
    }

    if (_expected_exposure) {
      //free(_expected_exposure);
    }
}