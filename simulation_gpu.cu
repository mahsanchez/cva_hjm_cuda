
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

#define FULL_MASK 0xffffffff

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 32
#define MAX_BLOCK_SZ 256
#define BATCH_SZ 1000

#define HJM_SDE_DEBUG1
#define MC_RDM_DEBUG
#define HJM_NUMERAIRE_DEBUG
#define EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN
#define EXPOSURE_PROFILES_AGGR_DEBUG
#define CONST_MEMORY1

// Constant Memory allocation
#ifdef CONST_MEMORY
    __constant__ float d_accrual[TIMEPOINTS];
    __constant__ float d_spot_rates[TIMEPOINTS];
    __constant__ float d_drifts[TIMEPOINTS];
    __constant__ float d_volatilities[VOL_DIM * TIMEPOINTS];
#endif

/*
 * Musiela Parametrization SDE
 * We simulate the SDE f(t+dt)=f(t) + dfbar  
 * where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi[i]*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
 */

__device__
float musiela_sde(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {

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

/**
* * RNG init Kernel
*/

__global__ void initRNG2(curandStateMRG32k3a* const rngStates, const unsigned int seed, int rnd_count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < rnd_count; index += blockDim.x * gridDim.x) {
        curand_init(seed, index, 0, &rngStates[index]);
    }
   
}

/*
 * Path generation kernel 
 * one to one mapping between threadIdx.x and tenor 
 */
__global__
void generatePaths(float* exposure, int timepoints, const float notional, const float K, float* accrual, float* spot_rates, float* drifts, float* volatilities, float dtau, curandStateMRG32k3a* const rngStates, const int pathN)
{
    //
    __shared__ float simulated_rates[TIMEPOINTS];
    // Compute Parameters
    __shared__ float phi0;
    __shared__ float phi1;
    __shared__ float phi2;
    float rate;
    // Simulation Parameters
    float dt = 0.01; // 
    int stride = dtau/dt; // 
    const float sqrt_dt = sqrt(dt);
    int sim_blck_count = pathN / stride;
    int t = threadIdx.x;
    float forward_rate;
    float discount_factor;
    float accum_rates;

    // Initialize simulated_rates with the spot_rate values
    if (threadIdx.x < timepoints) {
        simulated_rates[threadIdx.x] = spot_rates[threadIdx.x];
    }
    __syncthreads();

    //  HJM SDE Simulation
    for (int sim_blck = 0; sim_blck < sim_blck_count; sim_blck++)
    {
        for (int sim = 1; sim <= stride; sim++)
        {
            //  initialize the random numbers phi0, phi1, phi2 for the simulation (sim) for each t,  t[i] = t[i-1] + dt
            if (threadIdx.x == 0) {
                phi0 = curand_normal( &rngStates[blockIdx.x * 3] );
            }
            else if (threadIdx.x == 1) {
                phi1 = curand_normal( &rngStates[blockIdx.x * 3 + 1]);
            }
            else if (threadIdx.x == 2) {
                phi2 = curand_normal( &rngStates[blockIdx.x * 3 + 2]);
            }

            // synchronize threads block for next simulation step
            __syncthreads();
           
#ifdef MC_RDM_DEBUG1
            printf("BlockId %d Thread %d Random Normal Variates %f %f %f.\n", blockIdx.x, threadIdx.x, phi0, phi1, phi2);
#endif
            if (threadIdx.x < timepoints) {
                // Musiela Parametrization SDE
                float dF = 0.0;

                if (t < (timepoints - 1)) {
                    dF = simulated_rates[t + 1] - simulated_rates[t];
                }
                else {
                    dF = simulated_rates[t] - simulated_rates[t - 1];
                }

                rate = musiela_sde(
                     drifts[t], 
                     volatilities[t], 
                     volatilities[timepoints + t], 
                     volatilities[timepoints * 2 + t], 
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
                accum_rates += rate;
            }

            // Block all threads in the block waiting for rate calculation
            __syncthreads();

            // Upate the Forware Rate Curve at timepoint t for the next simulation
            if (threadIdx.x < timepoints) {
                
                simulated_rates[t] = rate;
            }

            // synchronize threads block for next simulation step
            __syncthreads();
        }

        // update numeraire based on simulation block delta
        if ((threadIdx.x < timepoints) && (threadIdx.x == sim_blck))
        {
            forward_rate = rate;
            discount_factor = exp(-accum_rates * dt);
#ifdef HJM_NUMERAIRE_DEBUG1
            printf("Thread %d sim_blck %d DiscountFactor %f ForwardRate %f.\n", threadIdx.x, sim_blck, discount_factor, forward_rate);
#endif
        }

    }

    // once all simulation finished each thread calculate a cashflows at timepoint equals to threadIdx.x
    if (threadIdx.x < timepoints)
    {
        // calculate cash_flows                       
        float cash_flow = discount_factor * notional * accrual[threadIdx.x] * (forward_rate - K);
        simulated_rates[threadIdx.x] = cash_flow;

#ifdef MC_RDM_DEBUG1
        printf("Block %d Thread %d CashFlow %f DiscountFactor %f ForwardRate %f. YearCtFra %f\n", blockIdx.x, threadIdx.x, cash_flow, discount_factor, forward_rate, accrual[threadIdx.x]);
#endif
    }
    __syncthreads();


    // calculate the exposure profile
    if (threadIdx.x < timepoints)
    {
        float sum = 0.0;
        for (int t = threadIdx.x + 1; t < TIMEPOINTS; t++) {
            sum += simulated_rates[t];
        }
        sum = (sum > 0.0) ? sum : 0.0;
        exposure[(blockIdx.x * timepoints) + threadIdx.x] = sum;
#ifdef MC_RDM_DEBUG1
        printf("Block %d Thread %d Exposure %f \n", blockIdx.x, threadIdx.x, sum);
#endif
    }

}



/*
* Aggregation 
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
*/
void calculateExposureGPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, int simN) {

    int _simN = 32000; // 1000; // 1000; // 1000; // 256; // 100; 1024
    unsigned int curve_points_size_bytes = TIMEPOINTS * sizeof(float);
    unsigned int total_curve_points_size_bytes = _simN * curve_points_size_bytes;

    // HJM Model number of paths
    int pathN = 2500;

    // Memory allocation 
#ifndef CONST_MEMORY
    float* d_accrual = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
#endif
    float* d_exposures = 0;
    curandStateMRG32k3a* rngStates = 0;

    //
    int gpu = 0;
    cudaSetDevice(gpu);

    // Copy the spot_rates, drift & volatilities to device memory
#ifndef CONST_MEMORY
    checkCudaErrors(cudaMalloc((void**)&d_accrual, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));
#endif

    // Rng buffer
    checkCudaErrors(cudaMalloc((void**)&rngStates, VOL_DIM * _simN * sizeof(curandStateMRG32k3a)));

    // Exposure profiles
    checkCudaErrors(cudaMalloc((void**)&d_exposures, _simN * TIMEPOINTS * sizeof(float)));

    const int curve_points_size_byte = TIMEPOINTS * sizeof(float);

#ifdef CONST_MEMORY
    cudaMemcpyToSymbol(d_accrual, accrual, curve_points_size_byte);
    cudaMemcpyToSymbol(d_spot_rates, spot_rates, curve_points_size_byte);
    cudaMemcpyToSymbol(d_drifts, drifts, curve_points_size_byte);
    cudaMemcpyToSymbol(d_volatilities, volatilities, VOL_DIM * curve_points_size_byte);
#endif 

    // Copy the spot_rates, drift & volatilities to device global memory Constant Memory TODO
#ifndef CONST_MEMORY
    checkCudaErrors(cudaMemcpy(d_accrual, accrual, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drifts, curve_points_size_bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * curve_points_size_bytes, cudaMemcpyHostToDevice));
#endif

    // Device Properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Kernel Execution Configuration
    int numBlocksPerSm = 0;
    int blockSize = 1024;
    int minGridSize;
    int gridSize;
    int rngCount = VOL_DIM * _simN;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, initRNG2, 0, rngCount);

    // Round up according to array size
    gridSize = (rngCount + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    initRNG2 << <gridSize, blockSize >> > (rngStates, 1234ULL, rngCount);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Rng Execution Time %fms\n", milliseconds);

    int numBlocks;        // Occupancy in terms of active blocks
    /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    numBlocksPerSm = 0;
    blockSize = 64;
    cudaGetDeviceProperties(&deviceProp, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, generatePaths, blockSize, 0);
    gridSize = deviceProp.multiProcessorCount * numBlocksPerSm;

    printf("Max Occupancy on number of blocks %d \n", numBlocksPerSm);
    printf("Max number of exposures profiles %d and simulations  %d \n", _simN, _simN * pathN);
    printf("gridSize %d and blockSize %d \n", gridSize, blockSize);

    // launch
    dim3 dimBlock(blockSize, 1, 1);
    dim3 dimGrid(gridSize, 1, 1);

    float totalMilliseconds = 0;

    for (int i = 0; i < _simN; i += gridSize) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        generatePaths << < dimGrid, dimBlock >> > (
            &d_exposures[i],
            TIMEPOINTS,
            payOff.notional,
            payOff.K,
            d_accrual,
            d_spot_rates,
            d_drifts,
            d_volatilities,
            payOff.dtau,
            rngStates,
            pathN
         );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalMilliseconds += milliseconds;
        printf("MC Simulation Execution Time %fms\n", milliseconds);
    }


    printf("MC Simulation Total Execution Time %fms\n", totalMilliseconds);
    
    // streams creation
    //int num_streams = _simN/BATCH_SZ;
    //cudaStream_t *streams = (cudaStream_t *) malloc(num_streams * sizeof(cudaStream_t));

    //for (int i = 0; i < num_streams; i++) {
    //    cudaStreamCreate(&streams[i]);
    //}

    // Set the CUDA execution kernel configuration
    //dim3 block = BLOCKSIZE;
    //dim3 grid = (BATCH_SZ * TIMEPOINTS + BLOCKSIZE - 1) / BLOCKSIZE; // 
    
    //
    // Run the simulation
    /*
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
    */

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
    //gpuSumReduceAvg(expected_exposure, d_exposures, _simN);

#ifdef HJM_SDE_DEBUG1
    printf("Expected Exposure Calculated");
#endif

release_resources:
    // Destroy the Host API randome generator
    //checkCudaErrors(curandDestroyGenerator(randomGenerator));

    // destroy all created streams
    //for (int i = 0; i < num_streams; i++) {
    //    cudaStreamCreate(&streams[i]);
    //}

    // Free Reserved Memory  d_rngNrmVar
  
    if (rngStates) {
        cudaFree(rngStates);
    }

#ifndef CONST_MEMORY
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
#endif

    if (d_exposures) {
        cudaFree(d_exposures);
    }
}



