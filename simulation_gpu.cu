
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 64

#define HJM_SDE_DEBUG
#define MC_RDM_DEBUG1
#define HJM_FORWARD_RATES
#define HJM_DISCOUNT_FACTORS
#define EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
__device__
void pricePayOff(float* exposure, float notional,  float dtau, float K, float* forward_rates, float* discount_factors, float expiry) {

    int size = (int) (expiry /dtau + 1);

    for (int i = 1; i < size - 1; i++) {
        float price = 0;
        for (int t = i; t < size; t++) {
            float sum = discount_factors[t] * notional * dtau * (forward_rates[t] - K);
            price += sum;
        }
        exposure[i] = price > 0 ? price : 0.0;
    }
}

/*
 * Path generation kernel 
 */
__global__
void generatePaths(float* exposure, 
    const float notional, const float K, float* yearCountFraction, float* spot_rates, float* drifts, float* volatilities, float dtau,
    float* d_rngNrmVar, const int pathN)
{
// Compute Parameters
    float phi0;
    float phi1;
    float phi2;
    float rate;
    float forward_rate;
    float discount_factor;
    float accum_rates = 0.0;
    __shared__ float simulated_rates[TIMEPOINTS];
    
    
 // Determine thread ID
    unsigned int tid = threadIdx.x;
    unsigned int laneid = threadIdx.x % 32;

 // Simulation Parameters
    float dt = 0.01; // 
    int delta = 100; //  
    const float sqrt_dt = sqrt(dt);

// Initialize simulated_rates
    for (int block = 0; block < TIMEPOINTS; block += BLOCKSIZE)
    {
        int t = block + tid;
        if (t < TIMEPOINTS) {
           simulated_rates[t] = spot_rates[t];
        }     
        __syncthreads(); 
    }

/**
   HJM SDE Simulation
*/
    for (int sim = 0; sim < pathN; sim++) 
    {

// one to one mapping between threadIdx.x and tenor 
// We simulate the SDE f(t+dt)=f(t) + dfbar  
// where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
        int t = threadIdx.x;

        if (t < TIMEPOINTS)
        {
            //  initialize the random numbers phi0, phi1, phi2 for the simulation (sim) for each t,  t[i] = t[i-1] + dt
            phi0 = d_rngNrmVar[blockIdx.x * pathN + sim * 3];
            phi1 = d_rngNrmVar[blockIdx.x * pathN + sim * 3 + 1];
            phi2 = d_rngNrmVar[blockIdx.x * pathN + sim * 3 + 2];

#ifdef MC_RDM_DEBUG
            if (laneid != 0) {
                printf("Thread %d Normal variates %f %f %f.\n", threadIdx.x, phi0, phi1, phi2);
            }
#endif

            float dfbar = volatilities[t] * phi0;
            dfbar += volatilities[TIMEPOINTS + t] * phi1;
            dfbar += volatilities[TIMEPOINTS * 2 + t] * phi2;
            dfbar *= sqrt_dt;
            // drift
            dfbar += drifts[t] * dt;
            // calculate dF/dtau*dt
            float dF = 0.0;

            if (t < (TIMEPOINTS - 1)) {
                dF = simulated_rates[t + 1] - simulated_rates[t];
            }
            else {
                dF = simulated_rates[t] - simulated_rates[t - 1];
            }
            dfbar += (dF / dtau) * dt;

            // apply Euler Maruyana
            rate = simulated_rates[t] + dfbar;

            //
            simulated_rates[t] = rate;
        }
         __syncthreads();

        // update numeraire based on simulation block delta
        if (t < TIMEPOINTS) 
        {
            accum_rates += rate;

            if (sim % delta == 0) {
                int tenor = sim / delta;
                if (t == tenor) {
                    forward_rate = rate;
                    discount_factor = exp(-accum_rates * dt);
#ifdef MC_RDM_DEBUG
                    printf("Thread %d DiscountFactor %f ForwardRate %f.\n", threadIdx.x, discount_factor, forward_rate);
#endif
                }       
            }
        }   
    }

    // once all simulation finished calculate each IRS cashflows
    if (threadIdx.x < TIMEPOINTS)
    {
        exposure[threadIdx.x + (blockIdx.x * blockDim.x)] = discount_factor * notional * 0.5 /*yearCountFraction[threadIdx.x]*/ * (forward_rate - K);

#ifdef MC_RDM_DEBUG
        printf("Thread %d Exposure %f DiscountFactor %f ForwardRate %f. YearCtFra %f\n", threadIdx.x, exposure[threadIdx.x] , discount_factor, forward_rate, dtau);
#endif
    }
}

// RNG init kernel
__global__ void initRNG2(curandState* const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 6000, &rngStates[tid]);
}

/*
   Exposure Calculation Kernel Invocation
   TODO - BATCHING
*/
void calculateExposureGPU(float* exposures, InterestRateSwap payOff, float* yearCountFraction, float* spot_rates, float* drift, float* volatilities, int simN) {

    int _simN = 1; // 100;
    // kernel execution configuration Determine how to divide the work between cores
    dim3 block = BLOCKSIZE; 
    dim3 grid = _simN; 

    // HJM Model number of paths
    int pathN = 5100;

    // Memory allocation 
    float* d_yearCountFraction = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
    float* d_exposures = 0;

    //
    int gpu = 0;
    cudaSetDevice(gpu);

    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMalloc((void**)&d_yearCountFraction, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));

    // spot_rates, drifts, volatilities, forward_rates, discount_factors
    checkCudaErrors(cudaMalloc((void**)&d_exposures, _simN * TIMEPOINTS * sizeof(float)));

    // random numbers buffers

#ifdef DEV_CURND_HOSTGEN  
    // initialize the rando in the device memory
    float* d_rngNrmVar = 0;
    int rnd_count = pathN * VOL_DIM;
    int rnd_count_bytes = rnd_count * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&d_rngNrmVar, rnd_count_bytes));
    float* h_rngNrmVars = (float*)malloc(rnd_count_bytes);

    curandGenerator_t randomGenerator;
    checkCudaErrors(curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_MRG32K3A));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(randomGenerator, 1234ULL));
    checkCudaErrors(curandGenerateNormal(randomGenerator, d_rngNrmVar, rnd_count, 0.0, 1.0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(h_rngNrmVars, d_rngNrmVar, rnd_count_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < rnd_count; i++) {
        printf("%d %1.4f\n", i, h_rngNrmVars[i]);
    }
    printf("\n");
    free(h_rngNrmVars);
#endif
   
    // Copy the spot_rates, drift & volatilities to device global memory
    checkCudaErrors(cudaMemcpy(d_yearCountFraction, yearCountFraction, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));

    // Initialise RNG
    //initRNG2 << < grid, block >> > (d_rngStates, m_seed);

    // Generate Paths and Compute IRS cashflows to be used for IRS Mark to Market
    generatePaths << <grid, block >> > (
        d_exposures, payOff.notional, payOff.K, yearCountFraction, d_spot_rates, d_drifts, d_volatilities, payOff.dtau, d_rngNrmVar, pathN);

    // Wait for all kernel to finish
    cudaDeviceSynchronize();

    // Use a second Kernel to Average Across TimePoints and produce the Expected Exposure Profile
#ifdef EXPOSURE_PROFILES_DEBUG
    checkCudaErrors(cudaMemcpy(exposures, d_exposures, _simN * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < _simN; i++) {
        for (int t = 0; i < TIMEPOINTS; t++) {
            printf("%1.4f ", exposures[i* TIMEPOINTS + t]);
        }
    }
    printf("\n");
#endif

    // Done with MC Simulation 
#ifdef HJM_SDE_DEBUG
    printf("MC Simulation generated");
#endif

    // Interest Rate Swap Mark to Market
    // pricePayOff(float* exposure, float notional,  float dtau, float K, float* forward_rates, float* discount_factors, float expiry) 

    // Destroy the Host API randome generator
    checkCudaErrors(curandDestroyGenerator(randomGenerator));

    // Free Reserved Memory
    if (d_rngNrmVar) {
        cudaFree(d_rngNrmVar);
    }

    if (d_yearCountFraction) {
        cudaFree(d_yearCountFraction);
    }

    if (d_spot_rates) {
        cudaFree(d_spot_rates);
    }

    if (d_drifts) {
        cudaFree(d_drifts);
    }

    if (d_volatilities) {
        cudaFree(d_volatilities);
    }
}



