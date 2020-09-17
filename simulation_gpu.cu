
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
#define MAX_BLOCK_SZ 256

#define HJM_SDE_DEBUG
#define MC_RDM_DEBUG
#define HJM_NUMERAIRE_DEBUG
#define EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
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
 * one to one mapping between threadIdx.x and tenor 
 * We simulate the SDE f(t+dt)=f(t) + dfbar  
 * where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
 */
__global__
void generatePaths(float* exposure, int _simN,
    const float notional, const float K, float* yearCountFraction, float* spot_rates, float* drifts, float* volatilities, float dtau,
    float* d_rngNrmVar, const int pathN)
{
    // Compute Parameters
    float phi0;
    float phi1;
    float phi2;
    float rate;
    __shared__ float simulated_rates[TIMEPOINTS];

    // Simulation Parameters
    float dt = 0.01; // 
    int stride = dtau/dt; // 
    const float sqrt_dt = sqrt(dt);

    //
    int sim_blck_count = pathN / stride;

    //
    int t = threadIdx.x;
    float forward_rate;
    float discount_factor;
    float accum_rates;

    // TODO loop-stride
    for (int offset = blockIdx.x * blockDim.x + threadIdx.x; offset < _simN; offset += blockDim.x * gridDim.x)
    {
        // Initialize simulated_rates
        if (threadIdx.x < TIMEPOINTS) {
            simulated_rates[threadIdx.x] = spot_rates[threadIdx.x];
        }
        __syncthreads();

        //  HJM SDE Simulation
        for (int sim_blck = 0; sim_blck < sim_blck_count; sim_blck++)
        {
            for (int sim = 0; sim < stride; sim++)
            {
                //  initialize the random numbers phi0, phi1, phi2 for the simulation (sim) for each t,  t[i] = t[i-1] + dt
                phi0 = d_rngNrmVar[blockIdx.x * pathN + sim * sim_blck * 3];
                phi1 = d_rngNrmVar[blockIdx.x * pathN + sim * sim_blck * 3 + 1];
                phi2 = d_rngNrmVar[blockIdx.x * pathN + sim * sim_blck * 3 + 2];

                if (threadIdx.x < TIMEPOINTS)
                {
#ifdef MC_RDM_DEBUG1
                    printf("Thread %d Random Normal Variates %f %f %f.\n", threadIdx.x, phi0, phi1, phi2);
#endif
                    // Musiela Parametrization SDE
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

                    // accumulate rate for discount calculation
                    accum_rates += rate;

                    // Upate the Forware Rate Curve at timepoint t for the next simulation
                    simulated_rates[t] = rate;
                }
                __syncthreads();
            }

            // update numeraire based on simulation block delta
            if ((threadIdx.x < TIMEPOINTS) && (threadIdx.x == sim_blck))
            {
                forward_rate = rate;
                discount_factor = exp(-accum_rates * dt);
#ifdef HJM_NUMERAIRE_DEBUG
                printf("Thread %d stride %d DiscountFactor %f ForwardRate %f.\n", threadIdx.x, stride, discount_factor, forward_rate);
#endif
            }
        }

        // once all simulation finished calculate each IRS cashflows
        if (threadIdx.x < TIMEPOINTS)
        {
            // calculate cash_flows
            float cash_flow = discount_factor * notional * yearCountFraction[threadIdx.x] * (forward_rate - K);
            simulated_rates[threadIdx.x] = cash_flow;

#ifdef MC_RDM_DEBUG1
            printf("Thread %d Exposure %f DiscountFactor %f ForwardRate %f. YearCtFra %f\n", threadIdx.x, simulated_rates[threadIdx.x], discount_factor, forward_rate, dtau);
#endif
        }
        __syncthreads();

        // calculate the exposure profile
        if (threadIdx.x < TIMEPOINTS)
        {
            float sum = 0.0;
            for (int t = threadIdx.x + 1; t < TIMEPOINTS; t++) {
                sum += simulated_rates[t];
            }
            exposure[offset + (blockIdx.x * TIMEPOINTS) + threadIdx.x ] = (sum > 0.0) ? sum : 0.0;
        }
    }

    
}

// RNG init kernel
__global__ void initRNG2(curandState* const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

// Parallel Reduction Kernel 
// Bandwidth: (((2^27) + 1) unsigned ints * 4 bytes/unsigned int)/(173.444 * 10^-3 s)
//  3.095 GB/s = 21.493% -> bad kernel memory bandwidth
__global__ void reduce0(float* g_odata, float* g_idata, unsigned int width, unsigned int size) {
    extern __shared__ unsigned int sdata[];

    // each thread loads one element from global to shared mem
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x * TIMEPOINTS;

    for (unsigned int t = 0; t < TIMEPOINTS; t++) {

        if (globalId < size)
        {
            sdata[threadIdx.x] = g_idata[globalId + t];
        }
        __syncthreads();

        for (unsigned int s = 1; s < width; s <<= 2) {
            if (threadIdx.x % (2 * s) == 0) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }

        // write result for this block to global mem
        if (threadIdx.x == 0) {
            g_odata[blockIdx.x * TIMEPOINTS + t] = sdata[0];
        }
    }


}

/*
*/
void gpuSumReduceAvg(float* h_expected_exposure, float* exposures, int simN) {

    // kernel execution configuration
    int blockSize = MAX_BLOCK_SZ * 2;
    dim3 block = blockSize;
    dim3 grid = (simN + blockSize - 1) / blockSize;

    // CUDA device exepected_exposure
    int exposure_size_bytes = TIMEPOINTS * sizeof(float);
    float *d_expected_exposure = 0;

    checkCudaErrors(cudaMalloc((void**)&d_expected_exposure, grid.x * exposure_size_bytes));

    // Sum data allocated for each block
    reduce0 << <grid, block, blockSize >> > (d_expected_exposure, exposures, blockSize, simN);
    cudaDeviceSynchronize();

#ifdef EXPOSURE_PROFILES_DEBUG
    float* h_exposures = (float*)malloc(simN * TIMEPOINTS * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_exposures, d_expected_exposure, simN * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < simN; i++) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", h_exposures[i * TIMEPOINTS + t]);
        }
        printf("\n");
    }
#endif

    // execute a second reduction kernel to sum all partial results
    float* d_reduced_exposures = 0;

    // reduce the partial sumation done by previous blocks
    reduce0 << <1, block, blockSize >> > (d_expected_exposure, d_reduced_exposures, blockSize, simN);
    cudaDeviceSynchronize();

    // Copy the expected exposure vector back to host
    checkCudaErrors(cudaMemcpy(h_expected_exposure, d_expected_exposure, exposure_size_bytes, cudaMemcpyDeviceToHost));

    //
    for (int t = 0; t < TIMEPOINTS; t++) {
        h_expected_exposure[t] = h_expected_exposure[t] / simN;
    }

    if (d_expected_exposure) {
        cudaFree(d_expected_exposure);
    }
}

/*
   Exposure Calculation Kernel Invocation
   TODO - BATCHING
*/
void calculateExposureGPU(float* expected_exposure, InterestRateSwap payOff, float* yearCountFraction, float* spot_rates, float* drift, float* volatilities, int simN) {

    int _simN = 256; // 100; 1024

    // HJM Model number of paths
    int pathN = 2500;

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

    // initialize the rando in the device memory
    float* d_rngNrmVar = 0;
    int rnd_count = _simN * pathN * VOL_DIM;
    int rnd_count_bytes = rnd_count * sizeof(float);

    checkCudaErrors(cudaMalloc((void**)&d_rngNrmVar, rnd_count_bytes));

    curandGenerator_t randomGenerator;
    checkCudaErrors(curandCreateGenerator(&randomGenerator, CURAND_RNG_PSEUDO_MRG32K3A));
    checkCudaErrors(curandSetPseudoRandomGeneratorSeed(randomGenerator, 1234ULL));
    checkCudaErrors(curandGenerateNormal(randomGenerator, d_rngNrmVar, rnd_count, 0.0, 1.0));
    checkCudaErrors(cudaDeviceSynchronize());

#ifdef DEV_CURND_HOSTGEN1  
    float* h_rngNrmVars = (float*)malloc(rnd_count_bytes);
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

    // Kernel Execution Configuration
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, gpu);
    dim3 block = BLOCKSIZE;
    dim3 grid = 32 * numSMs;

    // Generate Paths and Compute IRS cashflows to be used for IRS Mark to Market
    generatePaths <<< grid, block >>> (
        d_exposures, _simN, payOff.notional, payOff.K, d_yearCountFraction, d_spot_rates, d_drifts, d_volatilities, payOff.dtau, d_rngNrmVar, pathN);

    // Wait for all kernel to finish
    cudaDeviceSynchronize();

    // Use a second Kernel to Average Across TimePoints and produce the Expected Exposure Profile
#ifdef EXPOSURE_PROFILES_DEBUG1
    float* h_exposures = (float*)malloc(_simN * TIMEPOINTS * sizeof(float));
    checkCudaErrors(cudaMemcpy(h_exposures, d_exposures, _simN * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < _simN; i++) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", h_exposures[i* TIMEPOINTS + t]);
        }
        printf("\n");
    }   
#endif

    // Average across all the exposures profiles to obtain the Expected Exposure Profile

    // Done with MC Simulation 
#ifdef HJM_SDE_DEBUG
    printf("MC Simulation generated");
#endif

    // Reduce all exposures realizations and average them to obtain the Expected Exposures 
    // calculateExpectedExposure(float* expected_exposure, float* exposure, int simN)
    gpuSumReduceAvg(expected_exposure, d_exposures, pathN);

#ifdef HJM_SDE_DEBUG
    printf("Expected Exposure Calculated");
#endif

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

    if (d_exposures) {
        cudaFree(d_exposures);
    }
}



