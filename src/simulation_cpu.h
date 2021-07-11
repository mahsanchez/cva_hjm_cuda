

#ifndef EXPOSURE_CPU
#define EXPOSURE_CPU

#include <iostream>
#include <chrono>
#include <curand.h>

#include "mkl.h"
#include "mkl_vsl.h"

#include "product.h"

#define _TIMEPOINTS 51

#define INTEL_MKL1
#undef DEBUG_HJM_SIM
#undef DEBUG_NUMERAIRE
#undef DEBUG_EXPOSURE
#define DEBUG_EXPECTED_EXPOSURE

template <typename real>
struct __real2 {
    real x;
    real y;
};

template <typename real>
void calculateExposureCPU(real* expected_exposure, InterestRateSwap<real> payOff, real* accrual, real* spot_rates, real* drift, real* volatilities, int exposuresCount, real dt);

/*
* SDE
* We simulate f(t+dt)=f(t) + dfbar   where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt (Musiela Parameterisaiton HJM SDE)
*/
template <typename real>
real __musiela_sde(real drift, real vol0, real vol1, real vol2, real phi0, real phi1, real phi2, real sqrt_dt, real dF, real rate0, real dtau, real dt) {

    real vol_sum = vol0 * phi0;
    vol_sum += vol1 * phi1;
    vol_sum += vol2 * phi2;
    vol_sum *= sqrt(dt);

    real dfbar = drift * dt;
    dfbar += vol_sum;

    dfbar += (dF / dtau) * dt;

    // apply Euler Maruyana
    real result = rate0 + dfbar;

    return result;
}


/*
 * Random Number generation
 */
template <typename real>
void __initRNG2_kernel(real* rngNrmVar, const unsigned long long seed, int rnd_count);

template<>
void __initRNG2_kernel<float>(float* rngNrmVar, const unsigned long long seed, int rnd_count)
{
#ifdef INTEL_MKL
    VSLStreamStatePtr stream;
    //vslNewStream(&stream, VSL_BRNG_MT19937, 777);
    //vslNewStream(&stream, VSL_BRNG_MRG32K3A, 777);
    vslNewStream(&stream, VSL_BRNG_SOBOL, 777);
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, rnd_count, rngNrmVar, 0.0, 1.0);
    vslDeleteStream(&stream);
#else
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, seed);
    curandGenerateNormal(generator, rngNrmVar, rnd_count, 0.0, 1.0);
    curandDestroyGenerator(generator);
#endif
}

template<>
void __initRNG2_kernel<double>(double* rngNrmVar, const unsigned long long seed, int rnd_count)
{
    curandGenerator_t generator;
    curandCreateGeneratorHost(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, seed);
    curandGenerateNormalDouble(generator, rngNrmVar, rnd_count, 0.0, 1.0);
    curandDestroyGenerator(generator);
}

/*
 Calculate dF term in Musiela Parametrization SDE
*/

template <typename real>
inline real dFau(int t, int timepoints, real* rates) {
    real result = 0.0;

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

template <typename real>
void __generatePaths_kernel(__real2<real>* numeraires, int timepoints,
    real* drifts, real* volatilities, real* rngNrmVar,
    real* spot_rates,
    real* simulated_rates, real* simulated_rates0, real* accum_rates,
    const int pathN, int path,
    real dtau = 0.5, real dt = 0.01)
{
    // calculated rate
    real rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const real sqrt_dt = sqrt(dt);

    // Normal variates
    real phi0 = rngNrmVar[path * 3];
    real phi1 = rngNrmVar[path * 3 + 1];
    real phi2 = rngNrmVar[path * 3 + 2];

    // Evolve the whole curve from 0 to T 
    for (int t = 0; t < _TIMEPOINTS; t++)
    {
        if (path == 0) {
            rate = spot_rates[t];
        }
        else {
            real dF = dFau(t, _TIMEPOINTS, simulated_rates);

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


// block implementation

template <typename real>
void __generatePaths_kernel_cpu(__real2<real>* numeraires, int timepoints,
    real* drifts, real* volatilities, real* rngNrmVar,
    real* spot_rates,
    real* simulated_rates, real* simulated_rates0, real* accum_rates,
    const int pathN, int path,
    real dtau = 0.5, real dt = 0.01)
{
    // calculated rate
    real rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const real sqrt_dt = sqrt(dt);

    for (int s = 0; s < stride; s++) {
        // Normal variates
        real phi0 = rngNrmVar[(path + s) * 3];
        real phi1 = rngNrmVar[(path + s) * 3 + 1];
        real phi2 = rngNrmVar[(path + s) * 3 + 2];

        // Evolve the whole curve from 0 to T 
        for (int t = 0; t < _TIMEPOINTS; t++)
        {
            if (path == 0) {
                rate = spot_rates[t];
            }
            else {
                real dF = dFau(t, _TIMEPOINTS, simulated_rates);

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

        // update simulated rates for next iteration
        for (int t = 0; t < _TIMEPOINTS; t++) {
            simulated_rates[t] = simulated_rates0[t];
        }
    }

    // update numeraire based on simulation block 
    int lindex = (path + stride) / stride;
    numeraires[lindex].x = simulated_rates0[lindex];
    numeraires[lindex].y = exp(-accum_rates[lindex] * dt);


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
template <typename real>
void __calculateExposure_kernel(real* exposure, __real2<real>* numeraires, const real notional, const real K, real* accrual, int simN, real dtau = 0.5f)
{
    static real cash_flows[_TIMEPOINTS];
    real discount_factor;
    real forward_rate;
    real sum = 0.0;
    real m = (1.0f / dtau);

    // Calculate the Cashflows across tenors
    for (int t = 0; t < _TIMEPOINTS; t++) {
        forward_rate = numeraires[t].x;
        discount_factor = numeraires[t].y;
        real libor = m * (exp(forward_rate / m) - 1.0f);
        //cash_flows[t] = discount_factor * notional * accrual[t] * (forward_rate - K);
        cash_flows[t] = discount_factor * notional * accrual[t] * (libor - K);
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

/*
* blas support libraries
*/
void blas_gemv(float* expected_exposure, float* exposures, float* d_x, float* d_y, int simN) {
    const MKL_INT m = simN;
    const MKL_INT n = _TIMEPOINTS;
    const MKL_INT incx = 1;
    const MKL_INT incy = 1;
    const float alpha = 1.f / (float)simN;
    const float beta = 0.0;

    cblas_sgemv(CblasRowMajor, CblasTrans, m, n, alpha, exposures, n, d_x, incx, beta, d_y, incy);

    printf("cblas_call cblas_sgemv\n");
}

void blas_gemv(double* expected_exposure, double* exposures, double* d_x, double* d_y, int simN) {
    const MKL_INT m = simN;
    const MKL_INT n = _TIMEPOINTS;
    const MKL_INT incx = 1;
    const MKL_INT incy = 1;
    const double alpha = 1.f / (double)simN;
    const double beta = 0.0;

    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alpha, exposures, n, d_x, incx, beta, d_y, incy);

    printf("cblas_call cblas_dgemv\n");
}

/*
* Expected Exposure profile is defined as the positive average across all simulations on each pricing point
* For simplicity each tenor point [0..50] has been considered as a pricing point
* Rather than performing an aggreation (summation) on each column of the exposure matrix (tenors, simulations)
* a BLAS library function cblas_sgemv is used instead. Basically the exposure matrix multiplied by a vector
* of 1 values perform the summation on each column for averaging purpuse. As described bellow:
* matrix (tenors, simulations) x (simulations, 1) . The final result is a reduced vector with dimention (tenors, 1)
*/
template <typename real>
void __calculateExpectedExposure_kernel(real* expected_exposure, real* exposures, real* d_x, real* d_y, int simN) {

    blas_gemv(expected_exposure, exposures, d_x, d_y, simN);

    for (int t = 0; t < _TIMEPOINTS; t++) {
        expected_exposure[t] = d_y[t];
    }
}


/*
 * Exposure Calculation Kernel Invocation
*/
template <typename real>
void calculateExposureCPU(real* _expected_exposure, InterestRateSwap<real> payOff, real* accrual, real* spot_rates, real* drift, real* volatilities, int exposuresCount, real dt) //dt, dtau
{
    real* d_x = 0;;
    real* d_y = 0;
    d_x = (real*)malloc(exposuresCount * sizeof(real));
    d_y = (real*)malloc(_TIMEPOINTS * sizeof(real));

    // Allocate numeraires (forward_rate, discount_factor)
    __real2<real>* numeraires = 0;
    numeraires = (__real2<real>*)malloc(_TIMEPOINTS * exposuresCount * sizeof(__real2<real>));

    real* exposures = 0;
    exposures = (real*)malloc(_TIMEPOINTS * exposuresCount * sizeof(real));

    // Simulated forward Curve
    real* simulated_rates = (real*)malloc(_TIMEPOINTS * exposuresCount * sizeof(real));;
    real* simulated_rates0 = (real*)malloc(_TIMEPOINTS * exposuresCount * sizeof(real));;
    real* accum_rates = (real*)malloc(_TIMEPOINTS * exposuresCount * sizeof(real));;

    // TODO -replace initialization with memset initialize auxiliary vectors
    for (int i = 0; i < exposuresCount; i++) {
        d_x[i] = 1.0;
    }

    for (int i = 0; i < _TIMEPOINTS; i++) {
        d_y[i] = 0.0;
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
    real* rngNrmVar = 0;
    rngNrmVar = (real*)malloc(rnd_count * sizeof(real));

    // Normal distributed variates generation
    auto t_start = std::chrono::high_resolution_clock::now();

    __initRNG2_kernel(rngNrmVar, 1234L, rnd_count);

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total random normal variates " << rnd_count << " generated in " << elapsed_time_ms << "(ms)" << std::endl;

    // Monte Carlos Simulation HJM Kernels (2500 paths)

    t_start = std::chrono::high_resolution_clock::now();

    int blockSize = payOff.dtau / dt;

    std::cout << "intermediate simulations " << blockSize << "number of paths" << pathN << std::endl;

    for (int path = 0; path < pathN; path += blockSize)
    {
        for (int s = 0; s < exposuresCount; s++)
        {
            __generatePaths_kernel_cpu(
                &numeraires[s * _TIMEPOINTS],
                _TIMEPOINTS,
                drift,
                volatilities,
                &rngNrmVar[s * pathN * 3],
                spot_rates,
                &simulated_rates[s * _TIMEPOINTS],
                &simulated_rates0[s * _TIMEPOINTS],
                &accum_rates[s * _TIMEPOINTS],
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
        free(simulated_rates);
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

    //if (_expected_exposure) {
      //free(_expected_exposure);
    //}
}

#endif