
#include <cublas_v2.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <iostream>
#include <omp.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 64
#define MAX_BLOCK_SZ 256
#define BATCH_SZ 1000

#define HJM_SDE_DEBUG1
#define MC_RDM_DEBUG1
#define HJM_NUMERAIRE_DEBUG1
#define EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN
#define EXPOSURE_PROFILES_AGGR_DEBUG


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
void generatePaths(float* exposure, int _simN, const float notional, const float K, float* accrual, float* spot_rates, float* drifts, float* volatilities, float dtau, float* d_rngNrmVar, const int pathN)
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
    int sim_blck_count = pathN / stride;
    int t = threadIdx.x;
    float forward_rate;
    float discount_factor;
    float accum_rates;

    //for (int offset = 0; offset < _simN * TIMEPOINTS; offset += gridDim.x * TIMEPOINTS) {

        // Initialize simulated_rates with the spot_rate values
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
                phi0 = d_rngNrmVar[blockIdx.x * pathN * 3 + sim_blck * stride + sim * 3];
                phi1 = d_rngNrmVar[blockIdx.x * pathN * 3 + sim_blck * stride + sim * 3 + 1];
                phi2 = d_rngNrmVar[ blockIdx.x * pathN * 3 + sim_blck * stride + sim * 3 + 2];

                if (threadIdx.x < TIMEPOINTS)
                {
#ifdef MC_RDM_DEBUG
                    printf("BlockId %d Thread %d Random Normal Variates %f %f %f.\n", blockIdx.x, threadIdx.x, phi0, phi1, phi2);
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
                printf("Thread %d sim_blck %d DiscountFactor %f ForwardRate %f.\n", threadIdx.x, sim_blck, discount_factor, forward_rate);
#endif
            }

        }

        // once all simulation finished each thread calculate a cashflows at timepoint equals to threadIdx.x
        if (threadIdx.x < TIMEPOINTS)
        {
            // calculate cash_flows                       
            float cash_flow = discount_factor * notional * accrual[threadIdx.x] * (forward_rate - K);
            simulated_rates[threadIdx.x] = cash_flow;

#ifdef MC_RDM_DEBUG
            printf("Block %d Thread %d CashFlow %f DiscountFactor %f ForwardRate %f. YearCtFra %f\n", blockIdx.x, threadIdx.x, cash_flow, discount_factor, forward_rate, accrual[threadIdx.x]);
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
            sum = (sum > 0.0) ? sum : 0.0;
            exposure[(blockIdx.x * TIMEPOINTS) + threadIdx.x] = sum;
#ifdef MC_RDM_DEBUG
            printf("Block %d Thread %d Exposure %f \n", blockIdx.x, threadIdx.x, sum);
#endif
        }


//    }

}



/*
* Aggregation 
*/

/*
* TODO - Optimized Read Only Memory Access
*  cudaMemcpyToSymbol(vals, &s_vals, sizeof(int*));
   cudaMemcpyToSymbol(keys, &s_keys, sizeof(int*));
*/

void gpuSumReduceAvg(float* h_expected_exposure, float* exposures, int simN) {

    // CUDA device exepected_exposure
    float * d_x = 0;;
    float* d_y = 0;
 
    checkCudaErrors( cudaMalloc((void**)&d_x, simN * sizeof(float)));
    checkCudaErrors( cudaMalloc((void**)&d_y, TIMEPOINTS * sizeof(float)));
    float* identitiy_vector = (float*)malloc(simN * sizeof(float));

    for (int i = 0; i < simN; i++) {
        identitiy_vector[i] = 1.0;
    }
    checkCudaErrors( cudaMemcpy(d_x, identitiy_vector, simN * sizeof(float), cudaMemcpyHostToDevice) );
    //checkCudaErrors(cudaMemcpy(d_y, identitiy_vector, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));

    // Matrix Vector Multiplication to Reduce a Matrix by columns
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f;
    const float beta = 1.f;
    float cols = (float)TIMEPOINTS;
    float rows = (float)simN;
    
    // Applay matrix x identity vector (all 1) to do a column reduction by rows
    cublasSgemv(handle, CUBLAS_OP_N, cols, rows,  &alpha, exposures, cols, d_x, 1, &beta, d_y, 1);
    cudaDeviceSynchronize();

#ifdef DEV_CURND_HOSTGEN1 
    printf("Exposure 2D Matrix Aggregation by Cols  \n");
    printf("Matrix Cols (%d) Rows(%d) x Vector (%d) in elapsed time %f ms \n", TIMEPOINTS, simN, simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n", 2 * TIMEPOINTS * simN * 4 / elapsed_time / 1e6);
#endif

    checkCudaErrors( cudaMemcpy(h_expected_exposure, d_y, TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost) );

    // calculate average across all reduced columns
    for (int t = 0; t < TIMEPOINTS; t++) {
        h_expected_exposure[t] = h_expected_exposure[t] / simN;
    }

#ifdef EXPOSURE_PROFILES_AGGR_DEBUG
    printf("Expected Exposure Profile\n");
    for (int t = 0; t < TIMEPOINTS; t++) {
        printf("%1.4f ", h_expected_exposure[t]);
    }
    printf("\n");
#endif

    if (d_x) {
        cudaFree(d_x);
    }

    if (d_y) {
        cudaFree(d_y);
    }

    if (handle) {
       cublasDestroy(handle);
    }
}


/*
   Exposure Calculation Kernel Invocation
   TODO - BATCHING
*/
void calculateExposureGPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drift, float* volatilities, int simN) {

    int _simN = 10000; // 256; // 100; 1024
    unsigned int curve_points_size_bytes = TIMEPOINTS * sizeof(float);
    unsigned int total_curve_points_size_bytes = _simN * curve_points_size_bytes;

    // HJM Model number of paths
    int pathN = 2500;

    // Memory allocation 
    float* d_accrual = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
    float* d_exposures = 0;

    //
    int gpu = 0;
    cudaSetDevice(gpu);

    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMalloc((void**)&d_accrual, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));

    // spot_rates, drifts, volatilities, forward_rates, discount_factors
    checkCudaErrors(cudaMalloc((void**)&d_exposures, total_curve_points_size_bytes));

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
    cudaDeviceSynchronize();

#ifdef DEV_CURND_HOSTGEN1 
    printf("Generated %d normal random generators elapsed time %f ms \n", rnd_count, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n\n", rnd_count_bytes / elapsed_time / 1e6);
#endif

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
    checkCudaErrors(cudaMemcpy(d_accrual, accrual, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * curve_points_size_bytes, cudaMemcpyHostToDevice));

    // Kernel Execution Configuration
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, gpu);

    int deviceOverlap;
    cudaDeviceGetAttribute(&deviceOverlap, cudaDevAttrGpuOverlap, gpu);
    if (!deviceOverlap) {
        printf("Device will not handle overlap %d", deviceOverlap);
    }

    // streams creation
    int num_streams = _simN/BATCH_SZ;
    cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Set the CUDA execution kernel configuration
    dim3 block = BLOCKSIZE;
    dim3 grid = (BATCH_SZ * TIMEPOINTS + BLOCKSIZE - 1) / BLOCKSIZE; // 
    
    //
    // Run the simulation
    for (int s = 0; s < num_streams; s++) {
            generatePaths << < grid, block, 0, streams[s] >> > (
                d_exposures + s * TIMEPOINTS * 1000,
                1000,
                payOff.notional,
                payOff.K,
                d_accrual,
                d_spot_rates,
                d_drifts, d_volatilities,
                payOff.dtau,
                d_rngNrmVar + s * BATCH_SZ * 2500,
                pathN
            );
    }


    // Wait for all Streams to finish
    for (int s = 0; s < num_streams; s++) {
        cudaStreamSynchronize(streams[s]);
    }
  
#ifdef EXPOSURE_PROFILES_DEBUG1
    printf("Exposures Simulation batches %d run in %f ms \n", _simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n\n", TIMEPOINTS * TIMEPOINTS * _simN * 4 * 3 / elapsed_time / 1e6);
#endif

    // Use a second Kernel to Average Across TimePoints and produce the Expected Exposure Profile
#ifdef EXPOSURE_PROFILES_DEBUG1
    float* h_exposures = (float*)malloc(total_curve_points_size_bytes);
    checkCudaErrors(cudaMemcpy(h_exposures, d_exposures, total_curve_points_size_bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < _simN; i++) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", h_exposures[i* TIMEPOINTS + t]);
        }
        printf("\n");
    }   

    free(h_exposures);

    goto release_resources;
#endif

    // Average across all the exposures profiles to obtain the Expected Exposure Profile

    // Done with MC Simulation 
#ifdef HJM_SDE_DEBUG12
    printf("MC Simulation generated");
    goto release_resources;
#endif

    // Reduce all exposures realizations and average them to obtain the Expected Exposures 
    // calculateExpectedExposure(float* expected_exposure, float* exposure, int simN)
    gpuSumReduceAvg(expected_exposure, d_exposures, _simN);

#ifdef HJM_SDE_DEBUG
    printf("Expected Exposure Calculated");
#endif

release_resources:
    // Destroy the Host API randome generator
    checkCudaErrors(curandDestroyGenerator(randomGenerator));

    // destroy all created streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Free Reserved Memory
    if (d_rngNrmVar) {
        cudaFree(d_rngNrmVar);
    }

    if (d_accrual) {
        cudaFree(d_accrual);
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



