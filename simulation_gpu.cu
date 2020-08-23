
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3

// RNG init kernel
__global__ void initRNG2(curandState* const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}


/*
 * Heath-Jarrow-Morton SDE
 * evolve the whole forward rate curve using the SDE on HJM model
 * We simulate f(t+dt)=f(t) + dfbar  where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt  and phi ~ N(0,1)
 * calculate diffusion as SUM(Vol_i*phi*SQRT(dt))
 * calculate the drift as m(t)*dt
 * calculate dfbar as dF/dtau*dt
*/
__device__
void sde_evolve(double* drifts, double* volatilities, double* fwd_rates, double* fwd_rates0, double dtau, double dt, int size, curandState& state)
{
    for (int t = 0; t < size; t++) {
        // difussion
        double dfbar = volatilities[t] * curand_normal_double(&state);
        dfbar += volatilities[size + t] * curand_normal_double(&state);
        dfbar += volatilities[size * 2 + t] * curand_normal_double(&state);
        dfbar *= sqrt(dt);

        // dift
        dfbar += drifts[t] * dt;

        // calculate dF/dtau*dt
        double dF = 0.0;
        if (t < (size - 1)) {
            dF += fwd_rates0[t + 1] - fwd_rates0[t];
        }
        else {
            dF += fwd_rates0[t] - fwd_rates0[t - 1];
        }
        dfbar += (dF / dtau) * dt;

        // apply Euler Maruyana
        fwd_rates[t] = fwd_rates0[t] + dfbar;
    }
}

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
__device__
void pricePayOff(double* exposure, InterestRateSwap& payOff, double* forward_rates, double* discount_factors, double expiry) {

    int size = payOff.expiry / payOff.dtau + 1;

    for (int i = 1; i < size - 1; i++) {
        double price = 0;
        for (int t = i; t < size; t++) {
            double sum = discount_factors[t] * payOff.notional * payOff.dtau * (forward_rates[t] - payOff.K);
            price += sum;
        }
        exposure[i] = fmax(price, 0.0);
    }
}

/*
 * Path generation kernel & InterestRateSwap Exposure Calculation
 */
__global__ 
void generatePaths(double* exposures, double* spot_rates, double* drift, double* volatilities, double* forward_rates, double* discount_factors,
    InterestRateSwap& payOff, curandState* rngStates, const int pathN, const int size
)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    //unsigned int step = gridDim.x * blockDim.x;

    // Compute Parameters
    __shared__ double fwd_rates[TILE_DIM];
    __shared__ double fwd_rates0[TILE_DIM];
    __shared__ double accum_rates[TILE_DIM];

    // TODO copy drift && volatilities to shared memory (thread cooperation)
    // __shared__ double sh_drifts[TIMEPOINTS];
    // __shared__ double sh_volatilities[TIMEPOINTS*VOL_DIM];
    // __syncthreads();

    // Simulation Parameters
    double dt = payOff.expiry / pathN;
    int delta = (int)(1.0 / dt);

    // Initialise the RNG
    curandState localState = rngStates[tid];

    // local memory thread base position
    int base = threadIdx.x * size;

    // evolve the hjm model
    for (int sim = 1; sim < pathN; sim++) {

        // TODO - halve this inner loop twice to save share memory space
        // evolve the hjm sde
        sde_evolve(drift, volatilities, &fwd_rates[base], &fwd_rates0[base], payOff.dtau, dt, size, localState);

        //accumulate the sum of rates between sim-1 and sim
        for (int i = base; i < base + size; i++) {
            accum_rates[i] = accum_rates[i] + fwd_rates[i];
        }

        //register sim whose index constribute to numeraire tenors (potentially coalescing memory access across all threads)
        if (sim % delta == 0) {
            int index = base + sim / delta;
            forward_rates[index] = fwd_rates[index];
            discount_factors[index] = accum_rates[index];
        }

        //swap fwd_rates vectors
        for (int i = 0; i < TILE_DIM; i++) {
            fwd_rates0[i] = fwd_rates[i];
        }
    }

    // compute discount factors
    forward_rates[base] = spot_rates[0];
    discount_factors[base] = spot_rates[0];

    // (potentially coalescing memory access across all threads)
    for (int i = base; i < base + size; i++) {
        discount_factors[i] = exp(-discount_factors[i] * dt);
    }

    // Wait till all threads in the block has finished to exploit coalescing memory access on forward_rates and discount_factors global memory arrays
    __syncthreads();

    // perform mark to market and compute exposure (potentially coalescing memory access across all threads)
    pricePayOff(&exposures[tid * size], payOff, &forward_rates[base], &discount_factors[base], payOff.expiry);
}


/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(double* exposures, InterestRateSwap& payOff, double* spot_rates, double* drift, double* volatilities, int simN, int size) {

    // kernel execution configuration Determine how to divide the work between cores
    int blockSize = 32;
    dim3 block = blockSize;
    dim3 grid = (simN + blockSize - 1) / blockSize;

    // seed
    double m_seed = 1234;

    // Memory allocation 
    curandState* d_rngStates = 0;
    double *d_spot_rates = 0;
    double* d_drifts = 0;
    double* d_volatilities = 0;
    double* d_forward_rates = 0;
    double* d_discount_factors = 0;
    double* d_exposures = 0;

    // spot_rates, drifts, volatilities, forward_rates, discount_factors, exposures
    checkCudaErrors( cudaMalloc((void**)&d_rngStates, grid.x * block.x * sizeof(curandState)) );
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_forward_rates, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_discount_factors, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_exposures, grid.x * block.x * TIMEPOINTS * sizeof(double)));

    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));

    // HJM Model number of paths
    int pathN = 2000;
    
    // Initialise RNG
    initRNG2<<< grid, block >>>(d_rngStates, m_seed);

    // Generate Paths and Compute IRS Mark to Market
    generatePaths <<<grid, block >>> (exposures, d_spot_rates, d_drifts, d_volatilities, d_forward_rates, d_discount_factors, payOff, d_rngStates, pathN, TIMEPOINTS);

    // Copy partial results back
    checkCudaErrors( cudaMemcpy(&exposures[0], d_exposures, grid.x * sizeof(double), cudaMemcpyDeviceToHost) );

    // Free Reserved Memory
    checkCudaErrors(cudaFree(d_rngStates));
    checkCudaErrors(cudaFree(d_spot_rates));
    checkCudaErrors(cudaFree(d_drifts));
    checkCudaErrors(cudaFree(d_volatilities));
    checkCudaErrors(cudaFree(d_forward_rates));
    checkCudaErrors(cudaFree(d_discount_factors));
    checkCudaErrors(cudaFree(d_exposures));
}

