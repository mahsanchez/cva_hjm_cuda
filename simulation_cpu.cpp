#include "simulation_cpu.h"

#include <iostream>
#include "mkl.h"
#include "mkl_vsl.h"

#define _TIMEPOINTS 51

#define DEBUG_HJM_SIM
#define DEBUG_NUMERAIRE
#undef DEBUG_EXPOSURE
#undef DEBUG_EXPECTED_EXPOSURE

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
    float* spot_rates, float* drifts, float* volatilities, 
    float* rngNrmVar, 
    float* simulated_rates, float* simulated_rates0, float* accum_rates,
    const int pathN, float dtau = 0.5, float dt = 0.01
)
{
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

    // initialize numeraires
    numeraires[0].x = simulated_rates[0];
    numeraires[0].y = exp(-simulated_rates[0] * dt);

    //  HJM SDE Simulation
    for (int sim = 1; sim <= pathN; sim++)
        {
            //  initialize the random numbers phi0, phi1, phi2 for the simulation (sim) for each t,  t[i] = t[i-1] + dt
            phi0 = rngNrmVar[sim*3];
            phi1 = rngNrmVar[sim*3 + 1];
            phi2 = rngNrmVar[sim*3 + 2];

            for (int t = 0; t < _TIMEPOINTS; t++) 
            {
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

                // accumulate rate for discount calculation
                accum_rates[t] += rate;

                // store the simulated rate
                simulated_rates0[t] = rate;
            }

            // update numeraire based on simulation block 
            if (sim % stride == 0) {
                int lindex = sim / stride;
                numeraires[lindex].x = simulated_rates0[lindex];
                numeraires[lindex].y = exp(-accum_rates[lindex] * dt);
            }

            // update simulated rates (swap pointers)
            float* temp = simulated_rates;
            simulated_rates = simulated_rates0;
            simulated_rates0 = temp;


#ifdef DEBUG_HJM_SIM
            if (sim % stride == 0) {
                printf("%d - ", sim);
                for (int t = 0; t < _TIMEPOINTS; t++)
                {
                    printf("%f ", simulated_rates[t]);
                }
                printf("    %f %f %f \n", phi0, phi1, phi2);
            }
#endif

        }
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

    int a = 0;

}

/*
 * Initialize auxiliary vectors used during the simulation
 */
void __initVectors_kernel(float* simulated_rates, float* simulated_rates0, float* accum_rates, float *spot_rates, float dt) {

    // Initialize simulated_rates with the spot_rate values
    for (int t = 0; t < _TIMEPOINTS; t++) {
        simulated_rates[t] = spot_rates[t];
    }

    // reset internal buffers
    for (int t = 0; t < _TIMEPOINTS; t++) {
        simulated_rates0[t] = 0.0;
    }

    // reset accumulators
    for (int t = 0; t < _TIMEPOINTS; t++) {
        accum_rates[t] = 0.0;
    }
}


/*
 * Exposure Calculation Kernel Invocation
*/
void calculateExposureCPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int _simN) //dt, dtau
{
    //
    int simN = 5;

    //
    float dt = 0.01;

    // Auxiliary vectors
    float* d_x = 0;;
    float* d_y = 0;
    d_x = (float*) malloc(simN * sizeof(float));
    d_y = (float*) malloc(_TIMEPOINTS * sizeof(float));

    // Allocate numeraires (forward_rate, discount_factor)
    __float2* numeraires = 0;
    numeraires = (__float2*) malloc(_TIMEPOINTS * simN * sizeof(__float2));

    float* exposures = 0;
    exposures = (float*) malloc(_TIMEPOINTS * simN * sizeof(float));

    float* _expected_exposure = 0;
    _expected_exposure = (float*) malloc(_TIMEPOINTS * sizeof(float));

    // Simulated forward Curve
    float* simulated_rates = (float*) malloc(_TIMEPOINTS * sizeof(float));;
    float* simulated_rates0 = (float*) malloc(_TIMEPOINTS * sizeof(float));;
    float* accum_rates = (float*) malloc(_TIMEPOINTS * sizeof(float));;

    // initialize auxiliary vectors
    for (int i = 0; i < simN; i++) {
        d_x[i] = 1.0;
    }

    // Iterate across all simulations
    int pathN = 2501; // HJM Model simulation paths number
    int rnd_count = simN * pathN * 3; //

    // Generate the normal distributed variates
    float* rngNrmVar = 0;
    rngNrmVar = (float*) malloc(rnd_count * sizeof(float));

    // normal distributed variates generation
    __initRNG2_kernel(rngNrmVar, 1234L, rnd_count);

    // Monte Carlos Simulation Kernels for Exposure Generation

    for (int s = 0; s < simN; s++) {
        __initVectors_kernel(simulated_rates, simulated_rates0, accum_rates, spot_rates, dt);

        __generatePaths_kernel(&numeraires[s * _TIMEPOINTS], _TIMEPOINTS, spot_rates, drift, volatilities, &rngNrmVar[s*pathN*3], simulated_rates, simulated_rates0, accum_rates, pathN); //dt = 0.01, dtau = 0.5
        
#ifdef DEBUG_NUMERAIRE
        printf("Forward Rates/ Discount Factors \n");
        for (int t = 0; t < _TIMEPOINTS; t++)
        {
            printf("%f %f \n", numeraires[t].x, numeraires[t].y);
        }
#endif
        __calculateExposure_kernel(&exposures[s * _TIMEPOINTS], &numeraires[s * _TIMEPOINTS], payOff.notional, payOff.K, accrual, simN);
    }

    // Calculate the Expected Exposure Profile
    __calculateExpectedExposure_kernel(_expected_exposure, exposures, d_x, d_y, simN);


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
  //      free(d_x);
    }

    if (d_y) {
  //      free(d_y);
    }

    if (rngNrmVar) {
       free(rngNrmVar);
    }

    if (numeraires) {
       free(numeraires);
    }

    if (simulated_rates) {
      //free( simulated_rates);
    }

    if (simulated_rates0) {
       // free(simulated_rates0);
    }

    if (accum_rates) {
       // free(accum_rates); 
    }

    if (exposures) {
   //   free(exposures);
    }

    if (_expected_exposure) {
  //     free(_expected_exposure);
    }
}