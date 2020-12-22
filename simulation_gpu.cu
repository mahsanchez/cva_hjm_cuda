
#include <cublas_v2.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <thrust\device_vector.h>

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
#undef HJM_PATH_SIMULATION_DEBUG
#undef HJM_NUMERAIRE_DEBUG
#undef EXPOSURE_PROFILES_DEBUG
#define DEV_CURND_HOSTGEN
#undef EXPOSURE_PROFILES_AGGR_DEBUG
#define EXPECTED_EXPOSURE_DEBUG
#define CONST_MEMORY
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

#define CUBLAS_CALL(x)                                 \
   {                                                   \
        if((x)!=CUBLAS_STATUS_SUCCESS)                 \
          printf("ERROR: CUBLAS call at %s:%d\n",__FILE__,__LINE__);\
                                                       \
    } 


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
inline float __musiela_sde2(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {
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
void __generatePaths_kernel(float2* numeraires, 
    /*float* d_spot_rates,
    float* d_drifts, float* d_volatilities, */ 
    void* rngNrmVar,
    float* simulated_rates, float* simulated_rates0, float* accum_rates,
    const int pathN, int path,  
    float dtau = 0.5, float dt = 0.01)
{
    // calculated rate
    float rate;
    float sum_rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const float sqrt_dt = sqrtf(dt);

    int t = threadIdx.x;
    int gindex = blockIdx.x * TIMEPOINTS + threadIdx.x;

#ifdef RNG_HOST_API
    float phi0;
    float phi1;
    float phi2;
#else
    __shared__ float phi0;
    __shared__ float phi1;
    __shared__ float phi2;
#endif

    // Evolve the whole curve from 0 to T ( 1:1 mapping t with threadIdx.x)
    if (t < TIMEPOINTS)
    {
        if (path == 0) {
            rate = d_spot_rates[t];
        }
        else {
            // Calculate dF term in Musiela Parametrization SDE
            float dF = 0;
            if (t == (TIMEPOINTS - 1)) {
                dF = simulated_rates[gindex] - simulated_rates[gindex - 1];
            }
            else {
                dF = simulated_rates[gindex + 1] - simulated_rates[gindex];
            }

            // Normal random variates
 #ifdef RNG_HOST_API
            float *rngNrms = (float*)rngNrmVar;
            int rndIdx = blockIdx.x * pathN * VOL_DIM + path * VOL_DIM;
            phi0 = rngNrms[rndIdx];
            phi1 = rngNrms[rndIdx + 1];
            phi2 = rngNrms[rndIdx + 2];
#else
            if (threadIdx.x == 0) {
                curandStateMRG32k3a *state = (curandStateMRG32k3a*) rngNrmVar;
                curandStateMRG32k3a localState = state[blockIdx.x];
                phi0 = curand_uniform(&localState);
                phi1 = curand_uniform(&localState);
                phi2 = curand_uniform(&localState);
                state[blockIdx.x] = localState;
            }
            __syncthreads();         
#endif

            // simulate the sde
            rate = __musiela_sde2(
                d_drifts[t],
                d_volatilities[t],
                d_volatilities[TIMEPOINTS + t],
                d_volatilities[TIMEPOINTS * 2 + t],
                phi0,
                phi1,
                phi2,
                sqrt_dt,
                dF,
                simulated_rates[gindex],
                dtau,
                dt
            );
        }

#ifdef HJM_PATH_SIMULATION_DEBUG
        printf("Path %d Block %d Thread %d index %d Forward Rate %f phi0 %f phi1 %f phi2 %f \n", path, blockIdx.x, threadIdx.x, gindex, rate, phi0, phi1, phi2);
#endif
        // accumulate rate for discount calculation
        sum_rate = accum_rates[gindex];
        sum_rate += rate;
        accum_rates[gindex] = sum_rate;

        // store the simulated rate
        simulated_rates0[gindex] = rate; //

        // update numeraire based on simulation block 
        if (path % stride == 0) {
            if (t == (path / stride)) {
                numeraires[gindex].x = rate;
                numeraires[gindex].y = __expf(-sum_rate * dt);
#ifdef HJM_NUMERAIRE_DEBUG
                printf("Path %d Block %d Thread %d index %d Forward Rate %f Discount %f\n", path, blockIdx.x, threadIdx.x, gindex, rate, __expf(-sum_rate * dt));
#endif
            }
        }
    }
}



/*
 * Exposure generation kernel
 * one to one mapping between threadIdx.x and tenor
 */
__global__
void _exposure_calc_kernel(float* exposure, float2* numeraires, const float notional, const float K, /*float* d_accrual,*/ int simN, float dtau = 0.5f)
{
    __shared__ float cash_flows[TIMEPOINTS];
    float discount_factor;
    float forward_rate;
    float libor;
    float cash_flow;
    float sum = 0.0;
    float m = (1.0f / dtau);

    int globaltid = blockIdx.x * TIMEPOINTS + threadIdx.x;

    // calculate and load the cash flow in shared memory
    if (threadIdx.x < TIMEPOINTS) {
        forward_rate = numeraires[globaltid].x;
        libor = m * (__expf(forward_rate/m) - 1.0f);
        discount_factor = numeraires[globaltid].y;   
        cash_flow = discount_factor * notional * d_accrual[threadIdx.x] * (libor - K);
        cash_flows[threadIdx.x] = cash_flow;
#ifdef EXPOSURE_PROFILES_DEBUG
        printf("Block %d Thread %d Forward Rate %f libor %f Discount %f CashFlow %f \n", blockIdx.x, threadIdx.x, forward_rate, libor, discount_factor, cash_flow);
#endif
    }
    __syncthreads();

#ifdef EXPOSURE_PROFILES_DEBUG
    if (threadIdx.x == 0) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("t - indext %d CashFlow %f \n", t, cash_flows[t]);
        }
    }
#endif

    // calculate the exposure profile
    if ( threadIdx.x < TIMEPOINTS )
    {
        for (int t = threadIdx.x + 1; t < TIMEPOINTS; t++) {
            sum += cash_flows[t];
        }
        sum = (sum > 0.0) ? sum : 0.0;
        exposure[globaltid] = sum;
#ifdef EXPOSURE_PROFILES_DEBUG
        printf("Block %d Thread %d Exposure %f \n", blockIdx.x, threadIdx.x, sum);
#endif
    }
    __syncthreads();
}



/*
* Calculate Expected Exposure Profile
* 2D Aggregation using cublas sgemv
*/
void __expectedexposure_calc_kernel(float* expected_exposure, float* exposures, float *d_x, float *d_y, cublasHandle_t handle, int exposureCount) {

    const float alpha = 1.f / (float)exposureCount;
    const float beta = 1.f ;
    float cols = (float) TIMEPOINTS;
    float rows = (float) exposureCount;
    
    // Apply matrix x identity vector (all 1) to do a column reduction by rows
    CUBLAS_CALL ( cublasSgemv(handle, CUBLAS_OP_N, cols, rows,  &alpha, exposures, cols, d_x, 1, &beta, d_y, 1) );
    CUDA_RT_CALL( cudaMemcpy(expected_exposure, d_y, TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef DEV_CURND_HOSTGEN1 
    printf("Exposure 2D Matrix Aggregation by Cols  \n");
    printf("Matrix Cols (%d) Rows(%d) x Vector (%d) in elapsed time %f ms \n", TIMEPOINTS, simN, simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n", 2 * TIMEPOINTS * simN * 4 / elapsed_time / 1e6);
#endif
}


/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, int exposureCount, float dt = 0.01) {

    //exposureCount = 5000; // change exposure count here for testing 5, 10, 1000, 5000, 10000, 20000, 50000
    exposureCount = 5000;

    // HJM Model simulation number of paths with timestep dt = 0.01, expiry = 25 years
    const int pathN = payOff.expiry / dt; // 2500

    // Memory allocation 
#ifndef CONST_MEMORY
    float* d_accrual = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
#endif
    float2* d_numeraire = 0;
    float* d_exposures = 0;
    float* simulated_rates = 0;
    float* simulated_rates0 = 0;
    float* accum_rates = 0;
    float* d_x = 0;;
    float* d_y = 0;

    
    // Select the GPU Device in a multigpu setup
    int gpu = 0;
    cudaSetDevice(gpu);

    // Global memory reservation for constant input data
    CUDA_RT_CALL(cudaMalloc((void**)&d_numeraire, exposureCount * TIMEPOINTS * sizeof(float2)));  // Numeraire (discount_factor, forward_rates)
    CUDA_RT_CALL(cudaMalloc((void**)&d_exposures, exposureCount * TIMEPOINTS * sizeof(float)));   // Exposure profiles
    CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates0, exposureCount * TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&accum_rates, exposureCount * TIMEPOINTS * sizeof(float)));

    // Global memory reservation for constant input data
#ifndef CONST_MEMORY
    CUDA_RT_CALL(cudaMalloc((void**)&d_accrual, TIMEPOINTS * sizeof(float)));  // accrual
    CUDA_RT_CALL(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));  // spot_rates
    CUDA_RT_CALL(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));  // drifts
    CUDA_RT_CALL(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));  // volatilities
#endif

    // EE calculation aux vectors Global Memory (Convert to Const Memory)
    CUDA_RT_CALL(cudaMalloc((void**)&d_x, exposureCount * sizeof(float)));
    CUDA_RT_CALL(cudaMalloc((void**)&d_y, TIMEPOINTS * sizeof(float)));

    // initialize accum_rates float array
    int N = exposureCount * TIMEPOINTS;
    thrust::device_ptr<float> dev_ptr(accum_rates);
    thrust::fill(dev_ptr, dev_ptr + N, (float) 0.0f);

    // initialize simulated_rates0 float array
    thrust::device_ptr<float> dev_ptr2(simulated_rates0);
    thrust::fill(dev_ptr2, dev_ptr2 + N, (float)0.0f);

    // initialize d_x float array
    N = exposureCount;
    thrust::device_ptr<float> dev_ptr3(d_x);
    thrust::fill(dev_ptr3, dev_ptr3 + N, (float) 1.0f);

    // initialize d_x float array
    N = TIMEPOINTS;
    thrust::device_ptr<float> dev_ptr4(d_y);
    thrust::fill(dev_ptr4, dev_ptr4 + N, (float)0.0f);

    // CUBLAS handler
    cublasHandle_t handle;
    CUBLAS_CALL (cublasCreate(&handle));

#ifdef CONST_MEMORY
    CUDA_RT_CALL(cudaMemcpyToSymbol(d_accrual, accrual, TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMemcpyToSymbol(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMemcpyToSymbol(d_drifts, drifts, TIMEPOINTS * sizeof(float)));
    CUDA_RT_CALL(cudaMemcpyToSymbol(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));
#else
    CUDA_RT_CALL(cudaMemcpy(d_accrual, accrual, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_drifts, drifts, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_RT_CALL(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
#endif

    // Global Memory reservation for RNG vector
#ifdef RNG_HOST_API
    float* rngNrmVar = 0;
    int rngCount = exposureCount * VOL_DIM * pathN;
    CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar, rngCount * sizeof(float)));
#else
    const int rngCount = exposureCount;
    curandStateMRG32k3a* rngNrmVar = 0;
    CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar, rngCount * sizeof(curandStateMRG32k3a)));
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

    // kernel dimension variables
    int blockSize;
    int gridSize;

    // Random Number Generation 
    auto t_start = std::chrono::high_resolution_clock::now();
#ifdef RNG_HOST_API
    initRNG2_kernel(rngNrmVar, 1234ULL, rngCount);
#else
    blockSize = 32;
    gridSize = (rngCount + blockSize - 1) / blockSize;;
    initRNG2_kernel << <gridSize, blockSize >> > (rngNrmVar, 1234ULL, rngCount);
#endif
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total random normal variates " << rngCount << " generated in " << elapsed_time_ms << "(ms)" << std::endl;

    // Obtain number of SM per GPU device
    cudaDeviceProp devprop;
    cudaGetDeviceProperties(&devprop, gpu);
    //int sM = devprop.multiProcessorCount;

    // Monte Carlo Simulation kernel execution configuration 
    // Monte Carlos Simulation HJM Kernels (2500 paths)//dt = 0.01, dtau = 0.5
    blockSize = 64;
    gridSize = exposureCount; // (exposureCount < sM) ? exposureCount : (exposureCount - sM + 1) 

    t_start = std::chrono::high_resolution_clock::now();

    for (int path = 0; path < pathN; path++)
    {
        __generatePaths_kernel <<< gridSize, blockSize >>> (
            d_numeraire,
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

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run all " << pathN * exposureCount << " HJM MC simulation " << elapsed_time_ms << "(ms)" << std::endl;

    // Exposure Profile Calculation 

    blockSize = 64;
    gridSize = exposureCount; // 

    t_start = std::chrono::high_resolution_clock::now();

    _exposure_calc_kernel <<<gridSize, blockSize>>>(d_exposures, d_numeraire, payOff.notional, payOff.K, /*d_accrual,*/ exposureCount);
    CUDA_RT_CALL( cudaDeviceSynchronize() );

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run all " << exposureCount << " exposure profile calculation " << elapsed_time_ms << "(ms)" << std::endl;


#ifdef EXPOSURE_PROFILES_AGGR_DEBUG
    float* exposures = (float*)malloc(exposureCount * TIMEPOINTS * sizeof(float));

    CUDA_RT_CALL(cudaMemcpy(exposures, d_exposures, exposureCount * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

    // thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

    printf("Exposure Profile\n");
    for (int s = 0; s < exposureCount; s++) {
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", exposures[s * TIMEPOINTS + t]);
        }
        printf("\n");
    }
    
    free(exposures);
#endif

    // Expected Exposure Profile Calculation
    t_start = std::chrono::high_resolution_clock::now();

    float* result = (float* ) malloc( TIMEPOINTS * sizeof(float));

    __expectedexposure_calc_kernel(result, d_exposures, d_x, d_y, handle, exposureCount);

    t_end = std::chrono::high_resolution_clock::now();
    elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    std::cout << "total time taken to run" << exposureCount << " expected exposure profile " << elapsed_time_ms << "(ms)" << std::endl;

    // TODO - improve measurement GFLOPS

#ifdef EXPECTED_EXPOSURE_DEBUG
    printf("Expected Exposure Profile\n");
    for (int t = 0; t < TIMEPOINTS; t++) {
        printf("%1.4f ", result[t]);
    }
    printf("\n");
#endif

    free(result);

    // Release Resources
    if (handle) {
        CUBLAS_CALL( cublasDestroy(handle) );
    }

    if (d_x) {
        CUDA_RT_CALL(cudaFree(d_x));
    }

    if (d_y) {
        CUDA_RT_CALL( cudaFree(d_y));
    }

    if (d_numeraire) {
        CUDA_RT_CALL( cudaFree(d_numeraire) );
    }
  
    if (rngNrmVar) {
        CUDA_RT_CALL(cudaFree(rngNrmVar));
    }

#ifndef CONST_MEMORY
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
#endif

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



