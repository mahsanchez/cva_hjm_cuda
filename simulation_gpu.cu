
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

#include <nvtx3\nvToolsExt.h> 

#include <iostream>

using std::cout; using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

#include "simulation_gpu.h"
#include "scan_gpu.h"


#define FULL_MASK 0xffffffff
#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCKSIZE 32
#define BLOCK_SIZE 64
#define WARPSIZE 32
#define MAX_BLOCK_SZ 256
#define BATCH_SZ 1000
//#define double_ACC 1
//#define EXPECTED_EXPOSURE_DEBUG1 1

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


//#define SINGLE_PRECISION
#define DOUBLE_PRECISION
//#define SHARED_MEMORY_OPTIMIZATION
//#define CUDA_SYNCHR_OPTIMIZATION
#define MULTI_GPU_SIMULATION
#define CUDA_SYNC


//#define double double

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


#ifdef DOUBLE_PRECISION
    __constant__ double d_accrual[TIMEPOINTS];
    __constant__ double d_spot_rates[TIMEPOINTS];
    __constant__ double d_drifts[TIMEPOINTS];
    __constant__ double d_volatilities[VOL_DIM * TIMEPOINTS];
#else
    __constant__ float d_accrual[TIMEPOINTS];
    __constant__ float d_spot_rates[TIMEPOINTS];
    __constant__ float d_drifts[TIMEPOINTS];
    __constant__ float d_volatilities[VOL_DIM * TIMEPOINTS];
#endif


/*
 * MarketData Struct
*/
struct MarketData {
    double* accrual;
    double* spot_rates;
    double* drifts;
    double* volatilities;
};


/*
 *  CUDA utility function
 */

void cudaMemsetValue(double *buffer, int N, double initial_value) {
    thrust::device_ptr<double> dev_ptr(buffer);
    thrust::fill(dev_ptr, dev_ptr + N, initial_value);
}

void cudaMemsetValue(float* buffer, int N, float initial_value) {
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
    double result = rate0 + dfbar;

    return result;
}

template<typename real>
__device__  __forceinline__  double __musiela_sde2(double drift, double vol0, double vol1, double vol2, double phi0, double phi1, double phi2, double sqrt_dt, double dF, double rate0, double dtau, double dt) {
    double vol_sum = vol0 * phi0;
    vol_sum += vol1 * phi1;
    vol_sum += vol2 * phi2;
    vol_sum *= sqrtf(dt);

    double dfbar = drift * dt;
    dfbar += vol_sum;

    dfbar += (dF / dtau) * dt;

    // apply Euler Maruyana
    double result = rate0 + dfbar;

    return result;
}


/**
* * RNG init Kernel
*/

#ifdef RNG_HOST_API

void initRNG2_kernel(curandGenerator_t generator, double* rngNrmVar, const unsigned int seed, unsigned long long offset, int rnd_count, const double mean, const double stddev)
{
    //curandGenerator_t generator;
    //CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    //CURAND_CALL(curandSetGeneratorOffset(generator, offset));
    CURAND_CALL(curandGenerateNormalDouble(generator, rngNrmVar, rnd_count, mean, stddev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    //CURAND_CALL(curandDestroyGenerator(generator));
}


void initRNG2_kernel(curandGenerator_t generator, float* rngNrmVar, const unsigned int seed, unsigned long long offset, int rnd_count, const double mean, const double stddev)
{
    //curandGenerator_t generator;
    //CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    //CURAND_CALL(curandSetGeneratorOffset(generator, offset));
    CURAND_CALL(curandGenerateNormal(generator, rngNrmVar, rnd_count, mean, stddev));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    //CURAND_CALL(curandDestroyGenerator(generator));
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
 * Random initialization on device 
 */

__global__ void initRNG2_kernel_ondevice(curandStateMRG32k3a* const rngStates, const unsigned int seed, int rnd_count)
{
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (; index < rnd_count; index += blockDim.x * gridDim.x) {
        curand_init(seed, index, 0, &rngStates[index]);
    }
}

__global__ void initRNG(curandState* const rngStates, const unsigned int seed, int offset)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, tid, offset, &rngStates[tid]);
}

/*
 * Monte Carlo HJM Path Generation Constant Memory
*/
__global__
void __generatePaths_kernelOld(double2* numeraires, 
    void* rngNrmVar,
    double* simulated_rates, double* simulated_rates0, double* accum_rates,
    const int pathN, int path,  
    double dtau = 0.5, double dt = 0.01)
{
    // calculated rate
    double rate;
    double sum_rate;

    // Simulation Parameters
    int stride = dtau / dt; // 
    const double sqrt_dt = sqrtf(dt);

    int t = threadIdx.x;
    int gindex = blockIdx.x * TIMEPOINTS + threadIdx.x;

#ifdef RNG_HOST_API
    double phi0;
    double phi1;
    double phi2;
#else
    __shared__ double phi0;
    __shared__ double phi1;
    __shared__ double phi2;
#endif

    // Evolve the whole curve from 0 to T ( 1:1 mapping t with threadIdx.x)
    if (t < TIMEPOINTS)
    {
        if (path == 0) {
            rate = d_spot_rates[t];
        }
        else {
            // Calculate dF term in Musiela Parametrization SDE
            double dF = 0;
            if (t == (TIMEPOINTS - 1)) {
                dF = simulated_rates[gindex] - simulated_rates[gindex - 1];
            }
            else {
                dF = simulated_rates[gindex + 1] - simulated_rates[gindex];
            }

            // Normal random variates
 #ifdef RNG_HOST_API
            double *rngNrms = (double*)rngNrmVar;
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

template <typename real, typename real2>
__global__
void __generatePaths_kernel(
    int numberOfPoints,
    real2* numeraires,
    real* rngNrmVar,
    real* simulated_rates,
    real* simulated_rates0, 
    real* accum_rates,
    const int pathN, int path,
    real dtau = 0.5, double dt = 0.01)
    {
        // calculated rate
        real rate;
        real sum_rate;

#ifdef RNG_HOST_API
        real phi0;
        real phi1;
        real phi2;
#endif

        // Simulation Parameters
        int stride = dtau / dt; // 
        real sqrt_dt = sqrtf(dt);

        int t = threadIdx.x % TIMEPOINTS;
        int gindex = blockIdx.x * numberOfPoints + threadIdx.x;

        // Evolve the whole curve from 0 to T ( 1:1 mapping t with threadIdx.x)
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
        {
            if (path == 0) {
                rate = d_spot_rates[t];
            }
            else {
                // Calculate dF term in Musiela Parametrization SDE
                real dF = 0;

                if (t == (TIMEPOINTS - 1)) {
                    dF = simulated_rates[gindex] - simulated_rates[gindex - 1];
                }
                else {
                    dF = simulated_rates[gindex + 1] - simulated_rates[gindex];
                }
                
                // Normal random variates
#ifdef RNG_HOST_API
                real* rngNrms = (real*)rngNrmVar;
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

template <typename real, typename real2>
 __global__
        void __generatePaths_kernel4(
            int numberOfPoints,
            real2* numeraires,
            real* rngNrmVar,
            real* simulated_rates,
            real* simulated_rates0,
            real* accum_rates,
            const int pathN, 
            int path,
            real dtau = 0.5, real dt = 0.01)
    {
        // calculated rate
        real rate;
        real sum_rate = 0;
        real phi0;
        real phi1;
        real phi2;

        __shared__ real _ssimulated_rates[BLOCK_SIZE];

        // Simulation Parameters
        int stride = dtau / dt; // 
        real sqrt_dt = sqrtf(dt);

        //int t = threadIdx.x % TIMEPOINTS;
        int t = threadIdx.x;
        int gindex = blockIdx.x * numberOfPoints + threadIdx.x;

        // load the accumulated rate for a given timepoint
        sum_rate = accum_rates[gindex];

        // load the latest simulated rate from global memory
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
                    real dF = 0;

                    if (t == (TIMEPOINTS - 1)) {
                        dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                    }
                    else {
                        dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                    }

                    // Normal random variates broadcast if access same memory location in shared memory
                    real* rngNrms = (real*)rngNrmVar;
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

        // update the rates and the rate summation for the next simulation block 
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints))
        {
            simulated_rates0[gindex] = rate;
            accum_rates[gindex] = sum_rate;
        }
        
        // update numeraire based on simulation block 
        if ( t == (path + stride) / stride ) {
            numeraires[gindex].x = rate;  // forward rate
#ifdef double_ACC
      numeraires[gindex].y = expf(-sum_rate * dt);
#else
      numeraires[gindex].y = exp(-sum_rate * dt);
#endif
        }
    }


/*
* Risk Factor Generation block simulation  with Shared Memory
*/

void riskFactorSim4(
    int gridSize,
    int blockSize,
    int numberOfPoints,
    double2* numeraires,
    double* rngNrmVar,
    double* simulated_rates,
    double* simulated_rates0,
    double* accum_rates,
    const int pathN,
    double dtau = 0.5,
    double dt = 0.01)
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
            accum_rates,
            pathN,
            path,
            dtau,
            dt
        );

#ifdef CUDA_SYNC
        CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

       // update simulated rates (swap pointers)
       std::swap(simulated_rates, simulated_rates0);
    }
}

void riskFactorSim4(
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
    int simBlockSize = dtau / dt;

    for (int path = 0; path < pathN; path += simBlockSize)
    {
        __generatePaths_kernel4 <<< gridSize, blockSize >>> (
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

#ifdef CUDA_SYNC
        CUDA_RT_CALL(cudaDeviceSynchronize());
#endif
        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
    }
}

/*
 * Run the riskFactor Simulation using CUDA Streams 
*/

void riskFactorSimStream(int gridSize, int blockSize, int numberOfPoints,
    double2* numeraires,
    double* rngNrmVar,
    double* simulated_rates,
    double* simulated_rates0,
    double* accum_rates,
    const int pathN,
    int nstreams, 
    int operPerStream,
    cudaStream_t* streams, 
    double dtau = 0.5, double dt = 0.01) 
{
    int blockPerStream = gridSize / nstreams;
    int repBlock = blockPerStream / operPerStream;
    int simBlockSize = dtau / dt;

    for (int i = 0; i < blockPerStream; i += operPerStream)
    {
            for (int path = 0; path < pathN; path += simBlockSize)
            { 
                for (int b = 0; b < repBlock; b++)
                {
                    for (int s = 0; s < nstreams; s++)
                    {
                        __generatePaths_kernel4 << < repBlock, blockSize, 0, streams[s] >> > (
                            numberOfPoints,
                            numeraires + (s * blockPerStream + b * operPerStream) * numberOfPoints,
                            rngNrmVar + (s * blockPerStream + b * operPerStream) * pathN * 3,
                            simulated_rates + (s * blockPerStream + b* operPerStream) * numberOfPoints,
                            simulated_rates0 + (s * blockPerStream + b * operPerStream) * numberOfPoints,
                            accum_rates,
                            pathN,
                            path,
                            dtau,
                            dt
                        );
                    }
                }               
                // update simulated rates (swap pointers)
                std::swap(simulated_rates, simulated_rates0);
            }
    }
}


/*
* Risk Factor Generation naive acceleration
*/

void riskFactorSim(
    int gridSize, 
    int blockSize, 
    int numberOfPoints,
    double2* numeraires,
    double* rngNrmVar,
    double* simulated_rates, 
    double* simulated_rates0, 
    double* accum_rates,
    const int pathN, 
    double dtau = 0.5, 
    double dt = 0.01)
{

    for (int path = 0; path < pathN; path++)
    {
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

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
    }

}

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
        __generatePaths_kernel << < gridSize, blockSize >> > (
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

        CUDA_RT_CALL(cudaDeviceSynchronize());

        // update simulated rates (swap pointers)
        std::swap(simulated_rates, simulated_rates0);
    }

}


/*
 * Exposure generation kernel
 * one to one mapping between threadIdx.x and tenor
 */
template <typename real, typename real2>
__global__
void _exposure_calc_kernel(real* exposure, real2* numeraires, real notional, real K, int simN, real dtau = 0.5f)
{
    __shared__ real cash_flows[TIMEPOINTS];
    real discount_factor;
    real forward_rate;
    real libor;
    real cash_flow;
    real sum = 0.0;
    real m = (1.0 / dtau);

    int globaltid = blockIdx.x * TIMEPOINTS + threadIdx.x;

    // calculate and load the cash flow in shared memory
    if (threadIdx.x < TIMEPOINTS) {
        forward_rate = numeraires[globaltid].x;

#ifdef SINGLE_PRECISION
        libor = m * (expf(forward_rate / m) - 1.0);
#else
        libor = m * (exp(forward_rate / m) - 1.0);
#endif
        discount_factor = numeraires[globaltid].y;   
        cash_flow = discount_factor * notional * d_accrual[threadIdx.x] * (libor - K);
        cash_flows[threadIdx.x] = cash_flow;

#ifdef EXPOSURE_PROFILES_DEBUG
        printf("Block %d Thread %d Forward Rate %f libor %f Discount %f CashFlow %f \n", blockIdx.x, threadIdx.x, forward_rate, libor, discount_factor, cash_flow);
#endif
    }
    __syncthreads();

#ifdef EXPOSURE_PROFILES_DEBUG2
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

void exposureCalculation(int gridSize, int blockSize, double *d_exposures, double2 *d_numeraire, double notional, double K, int scenarios) {

    _exposure_calc_kernel <<< gridSize, blockSize >>> (d_exposures, d_numeraire, notional, K, scenarios);

#ifdef CUDA_SYNC
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif
}


void exposureCalculation(int gridSize, int blockSize, float* d_exposures, float2* d_numeraire, float notional, float K, int scenarios) {

    _exposure_calc_kernel << < gridSize, blockSize >> > (d_exposures, d_numeraire, notional, K, scenarios);

#ifdef CUDA_SYNC
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif
}


/*
* Calculate Expected Exposure Profile
* 2D Aggregation using cublas sgemv
*/
void __expectedexposure_calc_kernel(float* expected_exposure, float* exposures, float *d_x, float *d_y, cublasHandle_t handle, int exposureCount) {

    float alpha = 1.;
    float beta = 1. ;
    float cols = (float) TIMEPOINTS;
    float rows = (float) exposureCount;
    
    // Apply matrix x identity vector (all 1) to do a column reduction by rows

    CUBLAS_CALL(cublasSgemv(handle, CUBLAS_OP_N, cols, rows, &alpha, exposures, cols, d_x, 1, &beta, d_y, 1));

#ifdef CUDA_SYNC
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

#ifdef DEV_CURND_HOSTGEN1 
    printf("Exposure 2D Matrix Aggregation by Cols  \n");
    printf("Matrix Cols (%d) Rows(%d) x Vector (%d) in elapsed time %f ms \n", TIMEPOINTS, simN, simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n", 2 * TIMEPOINTS * simN * 4 / elapsed_time / 1e6);
#endif
}


void __expectedexposure_calc_kernel(double* expected_exposure, double* exposures, double* d_x, double* d_y, cublasHandle_t handle, int exposureCount) {

    double alpha = 1.;
    double beta = 1.;
    double cols = (double)TIMEPOINTS;
    double rows = (double)exposureCount;

    // Apply matrix x identity vector (all 1) to do a column reduction by rows
    CUBLAS_CALL(cublasDgemv(handle, CUBLAS_OP_N, cols, rows, &alpha, exposures, cols, d_x, 1, &beta, d_y, 1));

    //CUDA_RT_CALL(cudaMemcpy(expected_exposure, d_y, TIMEPOINTS * sizeof(double), cudaMemcpyDeviceToHost));

#ifdef CUDA_SYNC
    CUDA_RT_CALL(cudaDeviceSynchronize());
#endif

#ifdef DEV_CURND_HOSTGEN1 
    printf("Exposure 2D Matrix Aggregation by Cols  \n");
    printf("Matrix Cols (%d) Rows(%d) x Vector (%d) in elapsed time %f ms \n", TIMEPOINTS, simN, simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n", 2 * TIMEPOINTS * simN * 4 / elapsed_time / 1e6);
#endif
}



/*
   Exposure Calculation Kernel Invocation
*/

template <typename real>
void vAdd(int size, real *a, real* b, real* c) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

template <typename real>
void saxpy(int size, real multiplier, real *a, real *b) {
    for (int i = 0; i < size; i++) {
        b[i] = multiplier * a[i];
    }
}

template <typename real, typename real2>
void __calculateExposureMultiGPU(real* expected_exposure, InterestRateSwap<real> payOff, real* accrual, real* spot_rates, real* drifts, real* volatilities, real scale, const int num_gpus, int scenarios, real dt) {

    std::vector<real*> rngNrmVar(num_gpus);
    const int pathN = payOff.expiry / dt; // 25Y requires 2500 simulations
    int scenarios_gpus = scenarios / num_gpus; // total work distribution across gpus
    int rnd_count = scenarios_gpus * VOL_DIM * pathN;
    const unsigned int seed = 1234ULL;
    real mean = 0.0;
    real _stddev = 1.0;
    const int curveSizeBytes = TIMEPOINTS * sizeof(real); // Total memory occupancy for 51 timepoints

    std::cout << scenarios_gpus << " " << num_gpus << " pathN" << pathN << " dt " << dt << std::endl;

    // intermediate & final results memory reservation on device data
    std::vector<real2*> d_numeraire(num_gpus);
    std::vector<real*> d_exposures(num_gpus);
    std::vector<real*> simulated_rates(num_gpus);
    std::vector<real*> simulated_rates0(num_gpus);
    std::vector<real*> accum_rates(num_gpus);
    std::vector<real*> d_x(num_gpus);
    std::vector<real*> d_y(num_gpus);
    std::vector<real*> partial_exposure(num_gpus);
    std::vector<cublasHandle_t> cublas_handle(num_gpus);
    std::vector<curandGenerator_t> rngs(num_gpus);
    std::vector<curandGenerator_t> d_rngStates(num_gpus);

    nvtxRangePush("total_execution_time");

    nvtxRangePushA("data_initializing");

    auto t_start = std::chrono::high_resolution_clock::now(); 
       
    // memory allocation
    #pragma omp parallel num_threads(num_gpus)
    //for (int gpuDevice = 0; gpuDevice < num_gpus; gpuDevice++) 
    {

        int gpuDevice = omp_get_thread_num();

        cudaSetDevice(gpuDevice);

        // Reserve on device memory structures
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates0[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar[gpuDevice], rnd_count * sizeof(real)));

        CUDA_RT_CALL(cudaMalloc((void**)&d_numeraire[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(real2)));  // Numeraire (discount_factor, forward_rates)
        CUDA_RT_CALL(cudaMalloc((void**)&d_exposures[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(real)));   // Exposure profiles
        CUDA_RT_CALL(cudaMalloc((void**)&accum_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_x[gpuDevice], scenarios_gpus * sizeof(real)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_y[gpuDevice], TIMEPOINTS * sizeof(real)));

       //
        unsigned long long offset = gpuDevice * rnd_count;
        curandGenerator_t rng = rngs[gpuDevice];
        CURAND_CALL(curandCreateGenerator(&rngs[gpuDevice], CURAND_RNG_PSEUDO_DEFAULT));
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rngs[gpuDevice], seed));
        CURAND_CALL(curandSetGeneratorOffset(rngs[gpuDevice], offset));

        CUBLAS_CALL(cublasCreate(&cublas_handle[gpuDevice]));

#ifdef STREAM_ACC
        // allocate and initialize an array of stream handles
        int nstreams = 16;
        cudaStream_t* streams = (cudaStream_t*)malloc(nstreams * sizeof(cudaStream_t));
        for (int i = 0; i < nstreams; i++)
        {
            CUDA_RT_CALL(cudaStreamCreate(&(streams[i])));
        }
#endif
        // Reserve memory for ondevice random generation
        //CUDA_RT_CALL(cudaMalloc((void**)&d_rngStates[gpuDevice], scenarios_gpus * BLOCK_SIZE * sizeof(curandState)));

        partial_exposure[gpuDevice] = (real*)malloc(TIMEPOINTS * sizeof(real));

        // copy accrual, spot_rates, drifts, volatilites as marketData and copy to device constant memory
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_accrual, accrual, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_spot_rates, spot_rates, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_drifts, drifts, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_volatilities, volatilities, VOL_DIM * curveSizeBytes));

        // initialize array structures
        CUDA_RT_CALL(cudaMemset(accum_rates[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(simulated_rates0[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(real)));
        CUDA_RT_CALL(cudaMemset(d_y[gpuDevice], 0, TIMEPOINTS * sizeof(real)));
        cudaMemsetValue(d_x[gpuDevice], scenarios_gpus, 1.0f);

    }

    auto t_end = std::chrono::high_resolution_clock::now(); 
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count(); \
    printf("%f data_initializing %f (ms) \n", elapsed_time_ms);

    nvtxRangePop();
   

    // random generation
    nvtxRangePush("gpu_random_generation");

// TODO - Use different streams for each openmp threads (4)
#ifdef MULTI_GPU_SIMULATION
#pragma omp parallel num_threads(num_gpus)
    {
        int gpuDevice = omp_get_thread_num();
#else
        int gpuDevice = 0;
#endif
        cudaSetDevice(gpuDevice);

        // create Random Numbers (change seed by adding the gpuDevice)
        unsigned long long offset = gpuDevice * rnd_count;
        TIMED_RT_CALL(
            initRNG2_kernel(rngs[gpuDevice], rngNrmVar[gpuDevice], seed, offset, rnd_count, mean, _stddev), "normal variate generation"
        );
#ifdef MULTI_GPU_SIMULATION
    }
#endif
    nvtxRangePop();

    // risk factor evolution
#ifdef MULTI_GPU_SIMULATION
#pragma omp parallel num_threads(num_gpus)
    {
        int gpuDevice = omp_get_thread_num();
#endif
        cudaSetDevice(gpuDevice);

        // Kernel Execution Parameters
        int N = 1;
        int blockSize = N * BLOCK_SIZE;
        int numberOfPoints = N * TIMEPOINTS;
        int gridSize = scenarios_gpus / N;

        nvtxRangePush("Simulation");

#ifdef SHARED_MEMORY_OPTIMIZATION
        TIMED_RT_CALL(
            riskFactorSim4( 
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
#else
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
#endif
        // TRACE main
        nvtxRangePop();

        nvtxRangePush("Pricing");
        // Exposure Profile Calculation  TODO (d_exposures + gpuDevice * TIMEPOINTS)
        // Apply Scan algorithm here
        TIMED_RT_CALL(
            exposureCalculation(scenarios_gpus, BLOCK_SIZE, d_exposures[gpuDevice], d_numeraire[gpuDevice], payOff.notional, payOff.K, scenarios_gpus),
            "exposure calculation"
        );
        nvtxRangePop();

        //Replace all this block by a column reduction of the matrix
        // Partial Expected Exposure Calculation and scattered across gpus
        nvtxRangePushA("Exposure");
        
        TIMED_RT_CALL(
            __expectedexposure_calc_kernel(partial_exposure[gpuDevice], d_exposures[gpuDevice], d_x[gpuDevice], d_y[gpuDevice], cublas_handle[gpuDevice], scenarios_gpus),
            "partial expected exposure profile"
        );
        nvtxRangePop();

#ifdef MULTI_GPU_SIMULATION
    }
#endif

    nvtxRangePop();

    nvtxRangePush("Expected Exposure");

    // collect partial results and reduce them
    memset(expected_exposure, 0.0f, TIMEPOINTS * sizeof(real));
    CUDA_RT_CALL(cudaMemcpy(partial_exposure[0], d_y[0], TIMEPOINTS * sizeof(real), cudaMemcpyDeviceToHost));

#ifdef MULTI_GPU_SIMULATION
    for (int gpuDevice = 1; gpuDevice < num_gpus; gpuDevice++) {

        CUDA_RT_CALL(cudaMemcpy(partial_exposure[gpuDevice], d_y[gpuDevice], TIMEPOINTS * sizeof(real), cudaMemcpyDeviceToHost));

#ifdef EXPECTED_EXPOSURE_DEBUG
        printf("Exposure Profile\n");
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", partial_exposure[gpuDevice][t]);
        }
        printf("\n");
#endif

        vAdd(TIMEPOINTS, partial_exposure[0], partial_exposure[gpuDevice], partial_exposure[0]);
    }
#endif

    // average the result partial summation of exposure profiles
    real avg = 1.0f / (real)scenarios;
    saxpy(TIMEPOINTS, avg, partial_exposure[0], expected_exposure);

    nvtxRangePop();

    // TRACE main

    // free up resources
    #pragma omp parallel num_threads(num_gpus)
    //for (int gpuDevice = 0; gpuDevice < num_gpus; gpuDevice++) {
    {
        int gpuDevice = omp_get_thread_num();

        cudaSetDevice(gpuDevice);

#ifdef STREAM_ACC
        for (int i = 0; i < nstreams; i++)
        {
            CUDA_RT_CALL(cudaStreamDestroy(streams[i]));
        }
#endif

        CUBLAS_CALL(cublasDestroy( cublas_handle[gpuDevice] ));

        CURAND_CALL(curandDestroyGenerator(rngs[gpuDevice])); //

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

    nvtxRangePop();

}

void calculateExposureMultiGPU(double* expected_exposure, InterestRateSwap<double> payOff, double* accrual, double* spot_rates, double* drifts, double* volatilities, double scale, const int num_gpus, int scenarios, double dt) {
    __calculateExposureMultiGPU<double, double2>(expected_exposure, payOff, accrual, spot_rates, drifts, volatilities, scale, num_gpus, scenarios, dt);
}


void calculateExposureMultiGPU(float* expected_exposure, InterestRateSwap<float> payOff, float* accrual, float* spot_rates, float* drifts, float* volatilities, float scale, const int num_gpus, int scenarios, float dt) {
    __calculateExposureMultiGPU<float, float2>(expected_exposure, payOff, accrual, spot_rates, drifts, volatilities, scale, num_gpus, scenarios, dt);
}
