
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
#define EXPECTED_EXPOSURE_DEBUG1 1

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


#ifdef CONST_MEMORY
    __constant__ double d_accrual[TIMEPOINTS];
    __constant__ double d_spot_rates[TIMEPOINTS];
    __constant__ double d_drifts[TIMEPOINTS];
    __constant__ double d_volatilities[VOL_DIM * TIMEPOINTS];
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
void cudaMemsetdouble(double *buffer, int N, double initial_value) {
    //CUDA_RT_CALL( cudaMemset(buffer, 0, N * sizeof(double)) );
    thrust::device_ptr<double> dev_ptr(buffer);
    thrust::fill(dev_ptr, dev_ptr + N, initial_value);
}

/*
 * Musiela Parametrization SDE
 * We simulate the SDE f(t+dt)=f(t) + dfbar  
 * where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi[i]*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
 */

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
void initRNG2_kernel(double* rngNrmVar, const unsigned int seed, int rnd_count)
{
    const double mean = 0.0;  const double stddev = 1.0;
    curandGenerator_t generator;
    CURAND_CALL( curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) );
    CURAND_CALL( curandSetPseudoRandomGeneratorSeed(generator, 1234ULL) );
   // CURAND_CALL( curandGenerateNormal(generator, rngNrmVar, rnd_count, mean, stddev) );
    CURAND_CALL(curandGenerateNormalDouble(generator, rngNrmVar, rnd_count, mean, stddev));
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    CURAND_CALL(curandDestroyGenerator(generator));
}

void initRNG2_kernel(double* rngNrmVar, const unsigned int seed, unsigned long long offset, int rnd_count, const double mean, const double stddev)
{
    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(generator, seed));
    CURAND_CALL(curandSetGeneratorOffset(generator, offset));
    //CURAND_CALL(curandGenerateNormal(generator, rngNrmVar, rnd_count, mean, stddev));
    CURAND_CALL(curandGenerateNormalDouble(generator, rngNrmVar, rnd_count, mean, stddev));
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

__global__
void __generatePaths_kernel(
    int numberOfPoints,
    double2* numeraires,
    double* rngNrmVar,
    double* simulated_rates,
    double* simulated_rates0, 
    double* accum_rates,
    const int pathN, int path,
    double dtau = 0.5, double dt = 0.01)
    {
        // calculated rate
        double rate;
        double sum_rate;

#ifdef RNG_HOST_API
        double phi0;
        double phi1;
        double phi2;
#else
        __shared__ double phi0;
        __shared__ double phi1;
        __shared__ double phi2;
#endif

#ifdef OPT_SHARED_MEMORY
        extern __shared__ double _ssimulated_rates[];
#endif

        // Simulation Parameters
        int stride = dtau / dt; // 
        const double sqrt_dt = sqrtf(dt);

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
                double dF = 0;

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
                double* rngNrms = (double*)rngNrmVar;
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
            double2* numeraires,
            double* rngNrmVar,
            double* simulated_rates,
            double* simulated_rates0,
            const int pathN, int path,
            double dtau = 0.5, double dt = 0.01)
    {
        // calculated rate
        double rate;
        double sum_rate = 0;
        double phi0;
        double phi1;
        double phi2;

        __shared__ double _ssimulated_rates[BLOCK_SIZE];

        // Simulation Parameters
        int stride = dtau / dt; // 
        const double sqrt_dt = sqrtf(dt);

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
                    double dF = 0;

                    if (t == (TIMEPOINTS - 1)) {
                        dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                    }
                    else {
                        dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                    }

                    // Normal random variates broadcast if access same memory location in shared memory
                    double* rngNrms = (double*)rngNrmVar;
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
#ifdef double_ACC
      numeraires[gindex].y = expf(-sum_rate * dt);
#else
      numeraires[gindex].y = __expf(-sum_rate * dt);
#endif
        }
    }

/**
* Shared Memory & Global Access Memory optimizations & block simulation
*/

    __global__
        void __generatePaths_kernelShareMemRngRates(
            int numberOfPoints,
            double2* numeraires,
            double* rngNrmVar,
            double* simulated_rates,
            double* simulated_rates0,
            const int pathN, int path,
            double dtau = 0.5, double dt = 0.01)
    {
        // calculated rate
        double rate;
        double sum_rate = 0;
        double phi0;
        double phi1;
        double phi2;

        __shared__ double _ssimulated_rates[BLOCK_SIZE];
        __shared__ double3 rngNrms[BLOCK_SIZE];

        // Simulation Parameters
        int stride = dtau / dt; // 
        const double sqrt_dt = sqrtf(dt);

        int t = threadIdx.x % TIMEPOINTS;
        int gindex = blockIdx.x * numberOfPoints + threadIdx.x;

        // Store the generated Gaussian variates in shared memory
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
            int rndIdx = blockIdx.x * pathN * VOL_DIM + (path + threadIdx.x) * VOL_DIM;
            phi0 = rngNrmVar[rndIdx];
            phi1 = rngNrmVar[rndIdx + 1];
            phi2 = rngNrmVar[rndIdx + 2];
            rngNrms[threadIdx.x] = make_double3(phi0, phi1, phi2);
        }
        __syncthreads();

        // Store the initial forward rate curve in shared memory
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
            if (path > 0) {
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
                    double dF = 0;

                    if (t == (TIMEPOINTS - 1)) {
                        dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                    }
                    else {
                        dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                    }

                    // Normal random variates broadcast if access same memory location in shared memory
                    int rndIdx = s;
                    phi0 = rngNrms[rndIdx].x;
                    phi1 = rngNrms[rndIdx].y;
                    phi2 = rngNrms[rndIdx].z;

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
        if (t == (path + stride) / stride) {
            numeraires[gindex].x = rate;  // forward rate
           // numeraires[gindex].y = __expf(-sum_rate * dt); // discount factor
            numeraires[gindex].y = expf(-sum_rate * dt); // discount factor
        }
    }

/**
  * Shared Memory & Global Access Memory optimizations & block simulation
*/

__global__
            void __generatePaths_kernelSharedMemOnTFlyRandom(
                int numberOfPoints,
                double2* numeraires,
                curandState* rngStates,
                double* simulated_rates,
                double* simulated_rates0,
                const int pathN, int path,
                double dtau = 0.5, double dt = 0.01)
        {
            // calculated rate
            double rate;
            double sum_rate = 0;

            __shared__ double _ssimulated_rates[BLOCK_SIZE];
            __shared__ double3 rngNrms[BLOCK_SIZE]; // random generated gaussians

            // Simulation Parameters
            int stride = dtau / dt; // 
            const double sqrt_dt = sqrtf(dt);

            int t = threadIdx.x % TIMEPOINTS;
            int gindex = blockIdx.x * numberOfPoints + threadIdx.x;
            int tid = blockIdx.x * blockDim.x + threadIdx.x;

            // Create all the random gaussians and store them in the phi array
            curandState state = rngStates[tid];
            double phi0 = curand_normal(&state); 
            double phi1 = curand_normal(&state);
            double phi2 = curand_normal(&state);
            rngNrms[threadIdx.x] = make_double3(phi0, phi1, phi2); 
            __syncthreads();

            // Initialize the initial forward rate curve for this simulation block
            if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
                if (path > 0) {
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
                        double dF = 0;

                        if (t == (TIMEPOINTS - 1)) {
                            dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                        }
                        else {
                            dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                        }

                        // Normal random variates broadcast if access same memory location in shared memory
                        int rndIdx = s;
                        double3 phi = rngNrms[rndIdx];

                        // simulate the sde
                        rate = __musiela_sde2(
                            d_drifts[t],
                            d_volatilities[t],
                            d_volatilities[TIMEPOINTS + t],
                            d_volatilities[TIMEPOINTS * 2 + t],
                            phi.x,
                            phi.y,
                            phi.z,
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
            if (t == (path + stride) / stride) {
                numeraires[gindex].x = rate;  // forward rate
                numeraires[gindex].y = __expf(-sum_rate * dt); // discount factor
            }
}

/**
  * Shared Memory & Global Access Memory optimizations & block simulation
*/

__global__
void __generatePaths_kernelSharedMemOnTFlyRandomLoopStride(
    int numberOfPoints,
    double2* numeraires,
    curandState* rngStates,
    double* simulated_rates,
    double* simulated_rates0,
    const int pathN, int path,
    double dtau = 0.5, double dt = 0.01)
{
    // calculated rate
    double rate;
    double sum_rate = 0;

    __shared__ double _ssimulated_rates[BLOCK_SIZE];
    __shared__ double3 rngNrms[BLOCK_SIZE]; // random generated gaussians

    // Simulation Parameters
    int stride = dtau / dt; // 
    const double sqrt_dt = sqrtf(dt);

    int t = threadIdx.x % TIMEPOINTS;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int gindex = blockIdx.x * numberOfPoints + threadIdx.x; gindex < 0; gindex += gridDim.x * numberOfPoints) {
        // Create all the random gaussians and store them in the phi array
        curandState state = rngStates[tid];
        double phi0 = curand_normal(&state);
        double phi1 = curand_normal(&state);
        double phi2 = curand_normal(&state);
        rngNrms[threadIdx.x] = make_double3(phi0, phi1, phi2);
        __syncthreads();

        // Initialize the initial forward rate curve for this simulation block
        if ((threadIdx.x < numberOfPoints) && (gindex < gridDim.x * numberOfPoints)) {
            if (path > 0) {
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
                    double dF = 0;

                    if (t == (TIMEPOINTS - 1)) {
                        dF = _ssimulated_rates[threadIdx.x] - _ssimulated_rates[threadIdx.x - 1];
                    }
                    else {
                        dF = _ssimulated_rates[threadIdx.x + 1] - _ssimulated_rates[threadIdx.x];
                    }

                    // Normal random variates broadcast if access same memory location in shared memory
                    int rndIdx = s;
                    double3 phi = rngNrms[rndIdx];

                    // simulate the sde
                    rate = __musiela_sde2(
                        d_drifts[t],
                        d_volatilities[t],
                        d_volatilities[TIMEPOINTS + t],
                        d_volatilities[TIMEPOINTS * 2 + t],
                        phi.x,
                        phi.y,
                        phi.z,
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
        if (t == (path + stride) / stride) {
            numeraires[gindex].x = rate;  // forward rate
            numeraires[gindex].y = __expf(-sum_rate * dt); // discount factor
        }
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
    const int pathN,
    double dtau = 0.5,
    double dt = 0.01)
{
    int simBlockSize = dtau / dt;

    for (int path = 0; path < pathN; path += simBlockSize)
    {  
        //__generatePaths_kernel4 
        //__generatePaths_kernelShareMemRngRates
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
 * Risk Factor Generation with gausian variates generated on device 
*/

void riskFactorSimRandomsOnTFly(
    int gridSize,
    int blockSize,
    int numberOfPoints,
    double2* numeraires,
    curandState* rngNrmVar,
    double* simulated_rates,
    double* simulated_rates0,
    const int pathN,
    double dtau = 0.5,
    double dt = 0.01)
{
    int simBlockSize = dtau / dt;

    for (int path = 0; path < pathN; path += simBlockSize)
    {
        __generatePaths_kernelSharedMemOnTFlyRandom << < gridSize, blockSize >> > (
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
void _exposure_calc_kernel(double* exposure, double2* numeraires, const double notional, const double K, /*double* d_accrual,*/ int simN, double dtau = 0.5f)
{
    __shared__ double cash_flows[TIMEPOINTS];
    double discount_factor;
    double forward_rate;
    double libor;
    double cash_flow;
    double sum = 0.0;
    double m = (1.0f / dtau);

    int globaltid = blockIdx.x * TIMEPOINTS + threadIdx.x;

    // calculate and load the cash flow in shared memory
    if (threadIdx.x < TIMEPOINTS) {
        forward_rate = numeraires[globaltid].x;
        //libor = m * (__expf(forward_rate/m) - 1.0f);
#ifdef double_ACC
        libor = m * (expf(forward_rate / m) - 1.0f);
#else
        libor = m * (__expf(forward_rate / m) - 1.0f);
#endif
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

void exposureCalculation(int gridSize, int blockSize, double *d_exposures, double2 *d_numeraire, double notional, double K, int scenarios) {
    _exposure_calc_kernel <<< gridSize, blockSize >>> (d_exposures, d_numeraire, notional, K, scenarios);
    //CUDA_RT_CALL(cudaDeviceSynchronize());
}


/*
* Calculate Expected Exposure Profile
* 2D Aggregation using cublas sgemv
*/
void __expectedexposure_calc_kernel(double* expected_exposure, double* exposures, double *d_x, double *d_y, cublasHandle_t handle, int exposureCount) {

    const double alpha = 1.f;
    const double beta = 1.f ;
    double cols = (double) TIMEPOINTS;
    double rows = (double) exposureCount;
    
    // Apply matrix x identity vector (all 1) to do a column reduction by rows
    //CUBLAS_CALL ( cublasSgemv(handle, CUBLAS_OP_N, cols, rows,  &alpha, exposures, cols, d_x, 1, &beta, d_y, 1) );
    CUBLAS_CALL(cublasDgemv(handle, CUBLAS_OP_N, cols, rows, &alpha, exposures, cols, d_x, 1, &beta, d_y, 1));
    CUDA_RT_CALL( cudaMemcpy(expected_exposure, d_y, TIMEPOINTS * sizeof(double), cudaMemcpyDeviceToHost));

#ifdef DEV_CURND_HOSTGEN1 
    printf("Exposure 2D Matrix Aggregation by Cols  \n");
    printf("Matrix Cols (%d) Rows(%d) x Vector (%d) in elapsed time %f ms \n", TIMEPOINTS, simN, simN, elapsed_time);
    printf("Effective Bandwidth: %f GB/s \n", 2 * TIMEPOINTS * simN * 4 / elapsed_time / 1e6);
#endif
}


/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureMultiGPU(double* expected_exposure, InterestRateSwap_D payOff, double* accrual, double* spot_rates, double* drifts, double* volatilities, const int num_gpus, int scenarios, double dt) {

    std::vector<double*> rngNrmVar(num_gpus);
    const int pathN = payOff.expiry / dt; // 25Y requires 2500 simulations
    int scenarios_gpus = scenarios / num_gpus; // total work distribution across gpus
    int rnd_count = scenarios_gpus * VOL_DIM * pathN;
    const unsigned int seed = 1234ULL;
    const double mean = 0.0;
    const double _stddev = 1.0;
    const int curveSizeBytes = TIMEPOINTS * sizeof(double); // Total memory occupancy for 51 timepoints

    // intermediate & final results memory reservation on device data
    std::vector<double2*> d_numeraire(num_gpus);
    std::vector<double*> d_exposures(num_gpus);
    std::vector<double*> simulated_rates(num_gpus);
    std::vector<double*> simulated_rates0(num_gpus);
    std::vector<double*> accum_rates(num_gpus);
    std::vector<double*> d_x(num_gpus);
    std::vector<double*> d_y(num_gpus);  
    std::vector<double*> partial_exposure(num_gpus);
    //std::vector<curandState*> d_rngStates(num_gpus);

    // memory allocation
    for (int gpuDevice = 0; gpuDevice < num_gpus; gpuDevice++) {

        cudaSetDevice(gpuDevice);

        // Reserve on device memory structures
        CUDA_RT_CALL(cudaMalloc((void**)&rngNrmVar[gpuDevice], rnd_count * sizeof(double)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_numeraire[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(double2)));  // Numeraire (discount_factor, forward_rates)
        CUDA_RT_CALL(cudaMalloc((void**)&d_exposures[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(double)));   // Exposure profiles
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(double)));
        CUDA_RT_CALL(cudaMalloc((void**)&simulated_rates0[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(double)));
        CUDA_RT_CALL(cudaMalloc((void**)&accum_rates[gpuDevice], scenarios_gpus * TIMEPOINTS * sizeof(double)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_x[gpuDevice], scenarios_gpus * sizeof(double)));
        CUDA_RT_CALL(cudaMalloc((void**)&d_y[gpuDevice], TIMEPOINTS * sizeof(double)));

        // Reserve memory for ondevice random generation
        //CUDA_RT_CALL(cudaMalloc((void**)&d_rngStates[gpuDevice], scenarios_gpus * BLOCK_SIZE * sizeof(curandState)));

        partial_exposure[gpuDevice] = (double *) malloc(TIMEPOINTS * sizeof(double));

        // copy accrual, spot_rates, drifts, volatilites as marketData and copy to device constant memory
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_accrual, accrual, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_spot_rates, spot_rates, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_drifts, drifts, curveSizeBytes));
        CUDA_RT_CALL(cudaMemcpyToSymbol(d_volatilities, volatilities, VOL_DIM * curveSizeBytes));

        // initialize array structures
        CUDA_RT_CALL(cudaMemset(accum_rates[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(double)));
        CUDA_RT_CALL(cudaMemset(simulated_rates0[gpuDevice], 0, scenarios_gpus * TIMEPOINTS * sizeof(double)));
        CUDA_RT_CALL(cudaMemset(d_y[gpuDevice], 0, TIMEPOINTS * sizeof(double)));
        cudaMemsetdouble(d_x[gpuDevice], scenarios_gpus, 1.0f);
        
    }

    //  
    ///free(x);

    // random generation
    nvtxRangePushA("random_generation");
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
    nvtxRangePop();

    // TRACE main
    nvtxRangePushA("main");

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

        nvtxRangePushA("RiskFactor_Simulation");

        // Run Risk Factor Simulations Shared Memory Usage
        TIMED_RT_CALL(
           riskFactorSim4( //riskFactorSimShareMem
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

        // TRACE main
        nvtxRangePop();

        nvtxRangePushA("Pricing");
        // Exposure Profile Calculation  TODO (d_exposures + gpuDevice * TIMEPOINTS)
        // Apply Scan algorithm here
        TIMED_RT_CALL(
            exposureCalculation(scenarios_gpus, BLOCK_SIZE, d_exposures[gpuDevice], d_numeraire[gpuDevice], payOff.notional, payOff.K, scenarios_gpus),
            "exposure calculation"
        );
        nvtxRangePop();

        //Replace all this block by a column reduction of the matrix
        // Partial Expected Exposure Calculation and scattered across gpus
        nvtxRangePushA("Aggregation");
        cublasHandle_t handle; CUBLAS_CALL(cublasCreate(&handle));
        TIMED_RT_CALL(
             __expectedexposure_calc_kernel(partial_exposure[gpuDevice], d_exposures[gpuDevice], d_x[gpuDevice], d_y[gpuDevice], handle, scenarios_gpus),
            "partial expected exposure profile"
        );
        nvtxRangePop();

        // free up resources
        if (handle) {
           CUBLAS_CALL(cublasDestroy(handle));
        }

    }

    // Replace all this block by a simple vector sum
    // Gather the partial expected exposures and sum all
    memset(expected_exposure, 0.0f, TIMEPOINTS * sizeof(double));

    for (int gpuDevice = 1; gpuDevice < num_gpus; gpuDevice++) {
#ifdef EXPECTED_EXPOSURE_DEBUG1
        printf("Expected Exposure Profile\n");
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%1.4f ", partial_exposure[gpuDevice][t]);
        }
        printf("\n");
#endif
        vdAdd(TIMEPOINTS, partial_exposure[0], partial_exposure[gpuDevice], partial_exposure[0]);
    }  

    double avg = 1.0f / (double) scenarios;
    cblas_daxpy(TIMEPOINTS, avg, partial_exposure[0], 1, expected_exposure, 1);

    // TRACE main
    nvtxRangePop();

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

        /*
        if (d_rngStates[gpuDevice]) {
            CUDA_RT_CALL(cudaFree(d_rngStates[gpuDevice]));
        }*/

        if (partial_exposure[gpuDevice]) {
            free(partial_exposure[gpuDevice]);
        }
    }
 
}
