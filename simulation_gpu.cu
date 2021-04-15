
#include <cublas_v2.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <omp.h>
#include <thread>
#include <thrust\device_vector.h>
#include "mkl.h"

#include <iostream>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

#include "simulation_gpu.h"

#define FULL_MASK 0xffffffff
#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 32
#define BLOCK_SIZE 64
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
#define UM_HINTS
#define TIME_COUNTERS
#define MULTI_GPU_SIMULATION1
#define OPT_SHARED_MEMORY1

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


#define TIMED_RT_CALL(x, y) \
{ \
    {auto t_start = std::chrono::high_resolution_clock::now(); \
    x; \
    auto t_end = std::chrono::high_resolution_clock::now(); \
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count(); \
    printf("%s %f (ms) \n",  y , elapsed_time_ms); }\
  \
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
 * MarketData Struct
*/
struct MarketData {
    float* accrual;
    float* spot_rates;
    float* drifts;
    float* volatilities;
};


/*
 *  CUDA utility function
 */
void cudaMemsetFloat(float *buffer, int N, float initial_value) {
    //CUDA_RT_CALL( cudaMemset(buffer, 0, N * sizeof(float)) );
    thrust::device_ptr<float> dev_ptr(buffer);
    thrust::fill(dev_ptr, dev_ptr + N, initial_value);
}

/*
 * Musiela Parametrization SDE
 * We simulate the SDE f(t+dt)=f(t) + dfbar  
 * where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi[i]*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
 */

__device__  __forceinline__  float __musiela_sde2(float drift, float vol0, float vol1, float vol2, float phi0, float phi1, float phi2, float sqrt_dt, float dF, float rate0, float dtau, float dt) {
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

void initRNG2_kernel(float* rngNrmVar, const unsigned int seed, unsigned long long offset, int rnd_count, const float mean, const float stddev)
{
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    CURAND_CALL(curandSetGeneratorOffset(generator, offset));
    CURAND_CALL(curandGenerateNormal(generator, rngNrmVar, rnd_count, mean, stddev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
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
 * Monte Carlo HJM Path Generation Constant Memory
*/
__global__
void __generatePaths_kernelOld(float2* numeraires, 
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
 * Monte Carlo HJM Path Generation Constant Memory & BlockSize multiple of TIMEPOINTS
*/

__global__
void __generatePaths_kernel(
    int numberOfPoints,
    float2* numeraires,
    float* rngNrmVar,
    float* simulated_rates,
    float* simulated_rates0, 
    float* accum_rates,
    const int pathN, int path,
    float dtau = 0.5, float dt = 0.01)
    {
        // calculated rate
        float rate;
        float sum_rate;

#ifdef RNG_HOST_API
        float phi0;
        float phi1;
        float phi2;
#else
        __shared__ float phi0;
        __shared__ float phi1;
        __shared__ float phi2;
#endif

#ifdef OPT_SHARED_MEMORY
        extern __shared__ float _ssimulated_rates[];
#endif

        // Simulation Parameters
        int stride = dtau / dt; // 
        const float sqrt_dt = sqrtf(dt);

        int t = threadIdx.x % TIMEPOINTS;
        int gindex = blockIdx.x * numberOfPoints + threadIdx.x;

#ifdef OPT_SHARED_MEMORY
        
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
            _ssimulated_rates[threadIdx.x] = simulated_rates[gindex];
        }  
        __syncthreads();   
#endif

        // Evolve the whole curve from 0 to T ( 1:1 mapping t with threadIdx.x)
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
        {
            if (path == 0) {
                rate = d_spot_rates[t];
            }
            else {
                // Calculate dF term in Musiela Parametrization SDE
                float dF = 0;

#ifdef OPT_SHARED_MEMORY
                if (t == (TIMEPOINTS - 1)) {
                    dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                }
                else {
                    dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                }
#else
                if (t == (TIMEPOINTS - 1)) {
                    dF = simulated_rates[gindex] - simulated_rates[gindex - 1];
                }
                else {
                    dF = simulated_rates[gindex + 1] - simulated_rates[gindex];
                }
#endif
                
                // Normal random variates
#ifdef RNG_HOST_API
                float* rngNrms = (float*)rngNrmVar;
                int rndIdx = blockIdx.x * pathN * VOL_DIM + path * VOL_DIM;
                phi0 = rngNrms[rndIdx];
                phi1 = rngNrms[rndIdx + 1];
                phi2 = rngNrms[rndIdx + 2];
#else
                if (threadIdx.x == 0) {
                    curandStateMRG32k3a* state = (curandStateMRG32k3a*)rngNrmVar;
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

/**
* Shared Memory & Global Access Memory optimizations & block simulation
*/

 __global__
        void __generatePaths_kernel4(
            int numberOfPoints,
            float2* numeraires,
            float* rngNrmVar,
            float* simulated_rates,
            float* simulated_rates0,
            const int pathN, int path,
            float dtau = 0.5, float dt = 0.01)
    {
        // calculated rate
        float rate;
        float sum_rate = 0;
        float phi0;
        float phi1;
        float phi2;

        extern __shared__ float _ssimulated_rates[BLOCK_SIZE];

        // Simulation Parameters
        int stride = dtau / dt; // 
        const float sqrt_dt = sqrtf(dt);

        int t = threadIdx.x % TIMEPOINTS;
        int gindex = blockIdx.x * numberOfPoints + threadIdx.x;

        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
            if ( path > 0) { 
                _ssimulated_rates[threadIdx.x] = simulated_rates[gindex];
            }
        }
        __syncthreads();
       
        //
        for (int s = 0; s < stride; s++) 
        {
            if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
            {
                if (path == 0) {
                    rate = d_spot_rates[t];
                }
                else {
                    // Calculate dF term in Musiela Parametrization SDE
                    float dF = 0;

                    if (t == (TIMEPOINTS - 1)) {
                        dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                    }
                    else {
                        dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                    }

                    // Normal random variates broadcast if access same memory location in shared memory
                    float* rngNrms = (float*)rngNrmVar;
                    int rndIdx = blockIdx.x * pathN * VOL_DIM + (path + s)* VOL_DIM;
                    phi0 = rngNrms[rndIdx];
                    phi1 = rngNrms[rndIdx + 1];
                    phi2 = rngNrms[rndIdx + 2];

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
                        _ssimulated_rates[threadIdx.x],
                        dtau,
                        dt
                    );
                }

                // accumulate rate for discount calculation
                sum_rate += rate;
                
            }

            __syncthreads();

            if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
            {
                _ssimulated_rates[threadIdx.x] = rate;
            }

            __syncthreads();

        }

        // update the rates for the next simulation block
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
        {
            simulated_rates0[gindex] = rate;
        }
        
        // update numeraire based on simulation block 
        if ( t == (path + stride) / stride ) {
            numeraires[gindex].x = rate;  // forward rate
            numeraires[gindex].y = __expf(-sum_rate * dt); // discount factor
        }
    }

/*
* Risk Factor Generation block simulation
*/

void riskFactorSim4(
    int gridSize,
    int blockSize,
    int numberOfPoints,
    float2* numeraires,
    float* rngNrmVar,
    float* simulated_rates,
    float* simulated_rates0,
    const int pathN,
    float dtau = 0.5,
    float dt = 0.01)
{
    int simBlockSize = dtau / dt;

    for (int path = 0; path < pathN; path += simBlockSize)
    {
        __generatePaths_kernel4 <<< gridSize, blockSize >>> (
            numberOfPoints,
            numeraires,
            rngNrmVar,
            simulated_rates,
            simulated_rates0,
            pathN,
            path,
            dtau,
            dt
        );

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
    }

}

/*
* Risk Factor Generation naive acceleration
*/

void riskFactorSim(
    int gridSize, 
    int blockSize, 
    int numberOfPoints,
    float2* numeraires,
    float* rngNrmVar,
    float* simulated_rates, 
    float* simulated_rates0, 
    float* accum_rates,
    const int pathN, 
    float dtau = 0.5, 
    float dt = 0.01)
{

    for (int path = 0; path < pathN; path++)
    {

#ifdef OPT_SHARED_MEMORY
        __generatePaths_kernel << < gridSize, blockSize, blockSize >> > (
            numberOfPoints,
            numeraires,
            rngNrmVar,
            simulated_rates,
            simulated_rates0,
            accum_rates,
            pathN,
            path,
            dtau,
            dt
            );
#else
        __generatePaths_kernel <<< gridSize, blockSize >>> (
            numberOfPoints,
            numeraires,
            rngNrmVar,
            simulated_rates,
            simulated_rates0,
            accum_rates,
            pathN,
            path,
            dtau,
            dt
            );
#endif
        

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
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
*  Exposure calculation
*/

void exposureCalculation(int gridSize, int blockSize, float *d_exposures, float2 *d_numeraire, float notional, float K, int scenarios) {
    _exposure_calc_kernel <<< gridSize, blockSize >>> (d_exposures, d_numeraire, notional, K, scenarios);
    CUDA_RT_CALL(cudaDeviceSynchronize());
}


/*
* Calculate Expected Exposure Profile
* 2D Aggregation using cublas sgemv
*/
void __expectedexposure_calc_kernel(float* expected_exposure, float* exposures, float *d_x, float *d_y, cublasHandle_t handle, int exposureCount) {

    const float alpha = 1.f;
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
void calculateExposureMultiGPU(float* expected_exposure, InterestRateSwap payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, const int num_gpus, int scenarios, float dt) {

    std::vector<float*> rngNrmVar(num_gpus);
    const int pathN = payOff.expiry / dt; // 25Y requires 2500 simulations
    int scenarios_gpus = scenarios / num_gpus; // total work distribution across gpus
    int rnd_count = scenarios_gpus * VOL_DIM * pathN;
    const unsigned int seed = 1234ULL;
    const float mean = 0.0;
    const float _stddev = 1.0;
    const int curveSizeBytes = TIMEPOINTS * sizeof(float); // Total memory occupancy for 51 timepoints

    // intermediate & final results memory reservation on device data
    std::vector<float2*> d_numeraire(num_gpus);
    std::vector<float*> d_exposures(num_gpus);
    std::vector<float*> simulated_rates(num_gpus);
    std::vector<float*> simulated_rates0(num_gpus);
    std::vector<float*> accum_rates(num_gpus);
    std::vector<float*> d_x(num_gpus);
    std::vector<float*> d_y(num_gpus);
    std::vector<float*> partial_exposure(num_gpus);

    // memory allocation
    for (int gpuDevice = 0; gpuDevice < num_gpus; gpuDevice++) {

        cudaSetDevice(gpuDevice);

        CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar[gpuDevice], rnd_count * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_numeraire[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(float2)));  // Numeraire (discount_factor, forward_rates)
        CUDA_RT_CALL(cudaMalloc((void**)&d_exposures[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(float)));   // Exposure profiles
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates0[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc((void**)&accum_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_x[gpuDevice], scenarios_gpus * sizeof(float)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_y[gpuDevice], TIMEPOINTS * sizeof(float)));
        partial_exposure[gpuDevice] = (float *) malloc(TIMEPOINTS * sizeof(float));

        // copy accrual, spot_rates, drifts, volatilites as marketData and copy to device constant memory
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_accrual, accrual, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_spot_rates, spot_rates, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_drifts, drifts, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_volatilities, volatilities, VOL_DIM * curveSizeBytes));

        // initialize array structures
        CUDA_RT_CALL(cudaMemset(accum_rates[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(simulated_rates0[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(float)));
        CUDA_RT_CALL(cudaMemset(d_y[gpuDevice], 0, TIMEPOINTS * sizeof(float)));
        //cudaMemsetFloat(accum_rates[gpuDevice], scenarios_gpus * TIMEPOINTS, 0.0f);
        //cudaMemsetFloat(simulated_rates0[gpuDevice], scenarios_gpus * TIMEPOINTS, 0.0f);
        //cudaMemsetFloat(d_y[gpuDevice], TIMEPOINTS, 0.0f);
        cudaMemsetFloat(d_x[gpuDevice], scenarios_gpus, 1.0f);
        
    }

    //  
    ///free(x);

    // random generation
    #pragma omp parallel  num_threads(num_gpus)
    {
        int gpuDevice = omp_get_thread_num();

        cudaSetDevice(gpuDevice);

        // create Random Numbers (change seed by adding the gpuDevice)
        unsigned long long offset = gpuDevice * rnd_count;
        TIMED_RT_CALL(
            initRNG2_kernel(rngNrmVar[gpuDevice], seed, offset, rnd_count, mean, _stddev), "normal variate generation"
        );
    }

    // risk factor evolution
    #pragma omp parallel num_threads(num_gpus)
    {
        int gpuDevice = omp_get_thread_num();

        cudaSetDevice(gpuDevice);

        // Kernel Execution Parameters
        int N = 1;
        int blockSize = N* BLOCK_SIZE;
        int numberOfPoints = N * TIMEPOINTS;
        int gridSize = scenarios_gpus / N;

        // Run Risk Factor Simulations  
        TIMED_RT_CALL(
           riskFactorSim4(
            gridSize,
            blockSize,
            numberOfPoints,
            d_numeraire[gpuDevice],
            rngNrmVar[gpuDevice],
            simulated_rates[gpuDevice],
            simulated_rates0[gpuDevice],
            pathN,
            payOff.dtau,
            dt
           ),
            "Execution Time Partial HJM MC simulation"
        );

        /*
        TIMED_RT_CALL( 
            riskFactorSim(
                gridSize,
                blockSize,
                numberOfPoints,
                d_numeraire[gpuDevice],
                rngNrmVar[gpuDevice],
                simulated_rates[gpuDevice],
                simulated_rates0[gpuDevice],
                accum_rates[gpuDevice],
                pathN,
                payOff.dtau,
                dt
            ),
            "Execution Time Partial HJM MC simulation"
        );
        */

        // Exposure Profile Calculation  TODO (d_exposures + gpuDevice * TIMEPOINTS)
        TIMED_RT_CALL(
            exposureCalculation(scenarios_gpus, BLOCK_SIZE, d_exposures[gpuDevice], d_numeraire[gpuDevice], payOff.notional, payOff.K, scenarios_gpus),
            "exposure calculation"
        );

        // Partial Expected Exposure Calculation and scattered across gpus
        cublasHandle_t handle; CUBLAS_CALL(cublasCreate(&handle));
        TIMED_RT_CALL(
             __expectedexposure_calc_kernel(partial_exposure[gpuDevice], d_exposures[gpuDevice], d_x[gpuDevice], d_y[gpuDevice], handle, scenarios_gpus),
            "partial expected exposure profile"
        );

        // free up resources
        if (handle) {
           CUBLAS_CALL(cublasDestroy(handle));
        }
    }

    // Gather the partial expected exposures and sum all
    memset(expected_exposure, 0.0f, TIMEPOINTS * sizeof(float));

    for (int gpuDevice = 1; gpuDevice < num_gpus; gpuDevice++) {
#ifdef EXPECTED_EXPOSURE_DEBUG1
        printf("Expected Exposure Profile\n");
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", partial_exposure[gpuDevice][t]);
        }
        printf("\n");
#endif
        vsAdd(TIMEPOINTS, partial_exposure[0], partial_exposure[gpuDevice], partial_exposure[0]);
    }  

    float avg = 1.0f / (float) scenarios;
    cblas_saxpy(TIMEPOINTS, avg, partial_exposure[0], 1, expected_exposure, 1);


    // free up resources

    for (int gpuDevice = 0; gpuDevice < num_gpus; gpuDevice++) {

        cudaSetDevice(gpuDevice);

        if (rngNrmVar[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(rngNrmVar[gpuDevice]));
        }

        if (d_numeraire[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(d_numeraire[gpuDevice]));
        }

        if (d_exposures[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(d_exposures[gpuDevice]));
        }

        if (simulated_rates[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(simulated_rates[gpuDevice]));
        }

        if (simulated_rates0[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(simulated_rates0[gpuDevice]));
        }

        if (accum_rates[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(accum_rates[gpuDevice]));
        }
        if (d_x[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(d_x[gpuDevice]));
        }
        if (d_y[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(d_y[gpuDevice]));
        }

        if (partial_exposure[gpuDevice]) {
            free(partial_exposure[gpuDevice]);
        }
    }
 
}
