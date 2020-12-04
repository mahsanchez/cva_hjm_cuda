
#include <cublas_v2.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <iostream>

#include "simulation_gpu.h"

#define FULL_MASK 0xffffffff
#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 32
#define WARPSIZE 32
#define MAX_BLOCK_SZ 256
#define BATCH_SZ 1000

#undef HJM_SDE_DEBUG
#define MC_RDM_DEBUG
#define HJM_NUMERAIRE_DEBUG
#define EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN
#define EXPOSURE_PROFILES_AGGR_DEBUG
#undef CONST_MEMORY
#define RNG_HOST_API
#undef RNG_DEV_API

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }


#define CURAND_CALL(x)                                 \
   {                                                   \
        if((x)!=CURAND_STATUS_SUCCESS)                 \
          printf("ERROR: CURAND call at %s:%d\n",__FILE__,__LINE__);\
                                                       \
    }  


/*
 * Musiela Parametrization SDE
 * We simulate the SDE f(t+dt)=f(t) + dfbar  
 * where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi[i]*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
 */

__device__
float __musiela_sde2(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {

    float vol_sum = vol0 * phi0;
    vol_sum += vol1 * phi1;
    vol_sum += vol2 * phi2;
    vol_sum *= sqrtf(dt);

    float dfbar = drift * dt;
    dfbar += vol_sum;

    dfbar += (dF / dtau) * dt;

    // apply Euler Maruyana
    float result = rate0 + dfbar;

    return result;
}

/*
 Calculate dF term in Musiela Parametrization SDE
*/

__device__
inline float __dFau(int t, int timepoints, float* rates) {
    float result = 0.0;

    if (t == (timepoints - 1)) {
        result = rates[t] - rates[t - 1];
    }
    else {
        result = rates[t + 1] - rates[t];
    }

    return result;
}


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

/*
 * Initialize auxiliary vectors used during the simulation
 */
__global__
void initVectors_kernel(float2* numeraires, float* simulated_rates, float* spot_rates, int exposuresCount, float dt) {
    int gindex = blockIdx.x* blockDim.x + threadIdx.x;
    int stride = TIMEPOINTS * gridDim.x;
    int N = TIMEPOINTS * exposuresCount;

    if ( (threadIdx.x < TIMEPOINTS) && (gindex < N) ) {
        float rate = spot_rates[threadIdx.x];
        float discount_factor = exp(-rate * dt);

        for (int i = gindex; i < N; i += stride) {
            // store the spot_rate
            simulated_rates[i] = rate;
            // initialize numeraire at t = 0
            if (threadIdx.x == 0) {
                numeraires[i].x = rate;
                numeraires[i].y = discount_factor;
            }
        }
    }
}


/**
* * RNG init Kernel
*/

#ifdef RNG_HOST_API
void initRNG2_kernel(float* rngNrmVar, const unsigned int seed, int rnd_count)
{
    const float mean = 0.0;  const float stddev = 1.0;
    curandGenerator_t generator;
    CURAND_CALL( curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(generator, 1234ULL) );
    CURAND_CALL( curandGenerateNormal(generator, rngNrmVar, rnd_count, mean, stddev) );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    CURAND_CALL(curandDestroyGenerator(generator));
}
#else
__global__ void initRNG2_kernel(curandStateMRG32k3a* const rngStates, const unsigned int seed, int rnd_count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < rnd_count; index += blockDim.x * gridDim.x) {
        curand_init(seed, index, 0, &rngStates[index]);
    }
}
#endif


/*
 * Monte Carlo HJM Path Generation
*/
__global__
void __generatePaths_kernel(float2* numeraires, int timepoints,
    float* drifts, float* volatilities, float* rngNrmVar,
    float* simulated_rates, float* simulated_rates0, float* accum_rates,
    const int pathN, int path, int nx, int ny, 
    float dtau = 0.5, float dt = 0.01)
{
    // calculated rate
    float rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const float sqrt_dt = sqrtf(dt);

    // Normal variates
    float phi0 = rngNrmVar[path * 3];
    float phi1 = rngNrmVar[path * 3 + 1];
    float phi2 = rngNrmVar[path * 3 + 2];

    const int t = threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int bindex = iy * nx;
    const int gindex = bindex + ix;

    // int globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    // Evolve the whole curve from 0 to T 
    if (t < TIMEPOINTS)
    {
        float dF = __dFau(t, TIMEPOINTS, &simulated_rates[bindex]);

        rate = __musiela_sde2(
            drifts[t],
            volatilities[t],
            volatilities[TIMEPOINTS + t],
            volatilities[TIMEPOINTS * 2 + t],
            phi0,
            phi1,
            phi2,
            sqrt_dt,
            dF,
            simulated_rates[gindex],
            dtau,
            dt
        );

        // accumulate rate for discount calculation
        accum_rates[gindex] += rate;

        // store the simulated rate
        simulated_rates0[gindex] = rate;
    }

    // update numeraire based on simulation block 
    if (path % stride == 0) {
        if (t < TIMEPOINTS) {
            int lindex = path / stride;
            numeraires[lindex].x = simulated_rates0[bindex + lindex];
            numeraires[lindex].y = exp(-accum_rates[bindex + lindex] * dt);
        }
    }

}



/*
 * Exposure generation kernel
 * one to one mapping between threadIdx.x and tenor
 */
__global__
void gpuReduceExposure_kernel(float* exposure, float2* numeraires, const float notional, const float K, float* accrual, int simN)
{
    __shared__ float cash_flows[TIMEPOINTS];
    float discount_factor;
    float forward_rate;
    float sum = 0.0;

    int globaltid = blockIdx.x * blockDim.x + threadIdx.x;

    for (; globaltid < simN * TIMEPOINTS; globaltid += blockDim.x * TIMEPOINTS)
    {
    // calculate and load the cash flow in shared memory
        if (threadIdx.x < TIMEPOINTS) {
            forward_rate = numeraires[globaltid].x;
            discount_factor = numeraires[globaltid].y;          
            cash_flows[threadIdx.x] = discount_factor * notional * accrual[threadIdx.x] * (forward_rate - K);
        }
        __syncthreads();

        // calculate the exposure profile
        if ( threadIdx.x <= (TIMEPOINTS - 1) )
        {
            #pragma unroll
            for (int t = threadIdx.x + 1; t < TIMEPOINTS; t++) {
                sum += cash_flows[t];
            }

            sum = (sum > 0.0) ? sum : 0.0;
        
            exposure[globaltid] = sum;

    #ifdef MC_RDM_DEBUG
            printf("Block %d Thread %d Exposure %f \n", blockIdx.x, threadIdx.x, sum);
    #endif
        }
        __syncthreads();
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

    // TODO - Move the d_x identity vector to Constant Memory
    checkCudaErrors( cudaMemcpy(d_x, identitiy_vector, simN * sizeof(float), cudaMemcpyHostToDevice) );

    // Matrix Vector Multiplication to Reduce a Matrix by columns
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f;
    const float beta = 1.f;
    float cols = (float)TIMEPOINTS;
    float rows = (float)simN;
    
    // Apply matrix x identity vector (all 1) to do a column reduction by rows
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
void calculateExposureGPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, int exposureCount) {

    //int _simN = 32000; // 1000; // 1000; // 1000; // 256; // 100; 1024
   // exposureCount = 1;

    // HJM Model simulation number of paths with timestep dt = 0.01, dtau = 0.5 for 25 years
    const int pathN = 2500;
    const float dt = 0.01;

    // Memory allocation 
    float* d_accrual = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
    float2* d_numeraire = 0;
    float* d_exposures = 0;
    float* simulated_rates = 0;
    float* simulated_rates0 = 0;
    float* accum_rates = 0;
    
    // Select the GPU Device in a multigpu setup
    int gpu = 0;
    cudaSetDevice(gpu);

    // Global memory reservation for constant input data
    CUDA_RT_CALL(cudaMalloc((void**)&d_numeraire, exposureCount * TIMEPOINTS * sizeof(float2)));  // Numeraire (discount_factor, forward_rates)
    CUDA_RT_CALL(cudaMalloc((void**)&d_exposures, exposureCount * TIMEPOINTS * sizeof(float)));   // Exposure profiles
    CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates0, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&accum_rates, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(accum_rates, 0, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMemset(simulated_rates0, 0, exposureCount * TIMEPOINTS * sizeof(float)));

    // Global memory reservation for constant input data
    CUDA_RT_CALL(cudaMalloc((void**)&d_accrual, TIMEPOINTS * sizeof(float)));  // accrual
    CUDA_RT_CALL(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));  // spot_rates
    CUDA_RT_CALL(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));  // drifts
    CUDA_RT_CALL(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));  // volatilities

    // Copy the spot_rates, drift & volatilities to device global memory Constant Memory TODO
    CUDA_RT_CALL(cudaMemcpy(d_accrual, accrual, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_drifts, drifts, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));

    // Global Memory reservation for RNG vector
#ifdef RNG_HOST_API
    float* rngNrmVar = 0;
    int rngCount = exposureCount * VOL_DIM * pathN;
    CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar, rngCount * sizeof(float)));
#else
    const int rngCount = VOL_DIM * exposureCount;
    curandStateMRG32k3a* rngStates = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&rngStates, rngCount * sizeof(curandStateMRG32k3a)));
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

    // Random Number Generation 
    auto t_start = std::chrono::high_resolution_clock::now();
#ifdef RNG_HOST_API
    initRNG2_kernel(rngNrmVar, 1234ULL, rngCount);
#else
    int blockSize = 1024;
    int gridSize = (rngCount + blockSize - 1) / blockSize;;
    initRNG2_kernel << <gridSize, blockSize >> > (rngStates, 1234ULL, rngCount);
#endif
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total random normal variates " << rngCount << " generated in " << elapsed_time_ms << "(ms)" << std::endl;

    // Obtain number of SM per GPU device
    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, gpu);
    int sM = devprop.multiProcessorCount;

    // Initialize vectors already stored in Global Memory
    // simulated_rates values are initialized to the initial spot_rate values
    int blockSize = 64;
    int gridSize = (exposureCount < sM) ? exposureCount : (exposureCount - sM + 1) / sM; // <- Grid Size exposureCount < 40 (Total number of SM RTX 2070)

    t_start = std::chrono::high_resolution_clock::now();
    initVectors_kernel<<< gridSize, blockSize >>>(d_numeraire, simulated_rates, d_spot_rates, exposureCount, dt);
    CUDA_RT_CALL(cudaDeviceSynchronize());
    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time to initialize data " << elapsed_time_ms << "(ms)" << std::endl;

#ifndef HJM_NUMERAIRE_DEBUG
    for (int s = 0; s < exposureCount; s++) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf(" %f", simulated_rates[s * TIMEPOINTS + t]);
        }
    }
    std::cout << std::endl;
#endif

    // Monte Carlo Simulation kernel execution configuration 
    // Monte Carlos Simulation HJM Kernels (2500 paths)//dt = 0.01, dtau = 0.5
    /*
    for (int path = 1; path < pathN; path++)
    {
        __generatePaths_kernel <<< exposureCount, blockSize >>> (
            numeraires,
            drift,
            volatilities,
            rngNrmVar,
            simulated_rates,
            simulated_rates0,
            accum_rates,
            pathN,
            path
        );  
        CUDA_RT_CALL(cudaDeviceSynchronize());

        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
    }
    */

    // Exposure Profile Calculation 
    // gpuReduceExposure_kernel <<<dimGrid, dimBlock>>>(d_exposures, d_numeraire, payOff.notional, payOff.K, d_accrual, _simN);
    // cudaDeviceSynchronize();
    // printf("Exposure Calculation  Time %fms\n", milliseconds);

    // Expected Exposure Profile Calculation
    // Reduce all exposures realizations and average them to obtain the Expected Exposures (2D reduction on expsure matrix)
    // gpuSumReduceAvg(expected_exposure, d_exposures, _simN);


    if (d_numeraire) {
        CUDA_RT_CALL( cudaFree(d_numeraire) );
    }
  
#ifdef RNG_HOST_API
    CUDA_RT_CALL(cudaFree(rngNrmVar));
#else
    if (rngStates) {
        CUDA_RT_CALL(cudaFree(rngStates));
    }
#endif

    if (d_accrual) {
        CUDA_RT_CALL(cudaFree(d_accrual));
    }

    if (d_spot_rates) {
        CUDA_RT_CALL(cudaFree(d_spot_rates));
    }

    if (d_drifts) {
        CUDA_RT_CALL(cudaFree(d_drifts));
    }

    if (d_volatilities) {
        CUDA_RT_CALL(cudaFree(d_volatilities));
    }

    if (d_exposures) {
        CUDA_RT_CALL(cudaFree(d_exposures));
    }

    if (simulated_rates) {
        CUDA_RT_CALL(cudaFree(simulated_rates));
    }

    if (simulated_rates0) {
        CUDA_RT_CALL(cudaFree(simulated_rates0));
    }

    if (accum_rates) {
        CUDA_RT_CALL(cudaFree(accum_rates));
    }
}



