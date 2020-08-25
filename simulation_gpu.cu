
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCK_SIZE 32

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
void sde_evolve(double* drifts, double* volatilities, double* fwd_rates, double* fwd_rates0, double dtau, double dt, int size, curandState& state, int base)
{
    double phi0 = curand_normal_double(&state);
    double phi1 = curand_normal_double(&state);
    double phi2 = curand_normal_double(&state);

    for (int t = 0; t < size; t++) {
        // difussion
        double dfbar = volatilities[t] * phi0;
        dfbar += volatilities[size + t] * phi1;
        dfbar += volatilities[size*2 + t] * phi2;
        dfbar *= sqrt(dt);

        // dift
        dfbar += drifts[t] * dt;

        // calculate dF/dtau*dt
        double dF = 0.0;
        if (t < (size - 1)) {
            dF += fwd_rates0[base + t + 1] - fwd_rates0[base + t];
        }
        else {
            dF += fwd_rates0[base + t] - fwd_rates0[base + t - 1];
        }
        dfbar += (dF / dtau) * dt;

        // apply Euler Maruyana
        fwd_rates[base + t] = fwd_rates0[base + t] + dfbar;

#ifdef DEBUG_HJM_SDE
        //printf("%f ", fwd_rates[base + t]);
#endif
    }

#ifdef DEBUG_HJM_SDE
    printf(" %f %f %f\n", phi0, phi1, phi2);
#endif
}

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
__device__
void pricePayOff(double* exposure, InterestRateSwap payOff, double* forward_rates, double* discount_factors, double expiry) {

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
    double *fwd_rates, double *fwd_rates0, InterestRateSwap payOff, curandState* rngStates, const int pathN, const int size
)
{
    // Compute Parameters
    __shared__ double accum_rates[TILE_DIM * BLOCK_SIZE];
    
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_tid = threadIdx.x * size;

    // Simulation Parameters
    double dt = 0.01; // payOff.expiry / pathN;
    int delta = 100/2; //  (int)(1.0 / dt);

    // Initialise the RNG
    curandState localState = rngStates[tid];

    // local memory thread base position
    int base = tid * size;

    // initialize fwd_rates0 with the spot_rates
    for (int t = 0; t < size; t++) {
        fwd_rates0[base + t] = spot_rates[t];
    }

    // evolve the hjm model
    for (int sim = 1; sim < pathN; sim++) {

        printf("simulation (%d-%d) %d\n", blockIdx.x, threadIdx.x, sim);

        // evolve the hjm sde
        sde_evolve(drift, volatilities, fwd_rates, fwd_rates0, payOff.dtau, dt, size, localState, base);

        //accumulate the sum of rates between sim-1 and sim
        for (int i = 0; i < size; i++) {
            accum_rates[local_tid + i] = accum_rates[local_tid + i] + fwd_rates[base + i];
        }

        //register sim whose index constribute to numeraire tenors (potentially coalescing memory access across all threads)
        if (sim % delta == 0) {
            int index = sim / delta;
            forward_rates[base + index] = fwd_rates[base + index];
            discount_factors[base + index] = accum_rates[local_tid + index];
        }

        //swap fwd_rates vectors
        for (int i = 0; i < size; i++) {
            fwd_rates0[base + i] = fwd_rates[base + i];
        }
    }

    // compute discount factors
    forward_rates[base] = spot_rates[0];
    discount_factors[base] = spot_rates[0];

    // (potentially coalescing memory access across all threads)
    for (int i = 0; i <  size; i++) {
        discount_factors[base + i] = exp(-discount_factors[base + i] * dt);
    }

#ifndef DEBUG_HJM_SDE
    for (int i = 0; i < size; i++) {
        printf(" %f ", discount_factors[base + i]);
    }  
    printf("\n");
    for (int i = 0; i < size; i++) {
        printf(" %f ", forward_rates[base + i]);
    }
    printf("\n");
#endif
  

    // perform mark to market and compute exposure (potentially coalescing memory access across all threads)
    pricePayOff(&exposures[tid * size], payOff, &forward_rates[base], &discount_factors[base], payOff.expiry); 
}


/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(double* exposures, InterestRateSwap payOff, double* spot_rates, double* drift, double* volatilities, int simN, int size) {

    // kernel execution configuration Determine how to divide the work between cores
    int blockSize = 32;
    dim3 block = blockSize; // blockSize;
    dim3 grid = (simN + blockSize - 1) / blockSize;

    // seed
    double m_seed = 1234;

    // Memory allocation 
    InterestRateSwap d_payOff(0, 0, 0, payOff.notional, payOff.K, payOff.expiry, payOff.dtau);
    curandState* d_rngStates = 0;
    double* d_fwd_rates; 
    double* d_fwd_rates0;
    double* d_spot_rates = 0;
    double* d_drifts = 0;
    double* d_volatilities = 0;
    double* d_forward_rates = 0;
    double* d_discount_factors = 0;
    double* d_exposures = 0; 

    // Allocate memory for Interest Rate Swap Schedule
    checkCudaErrors(cudaMalloc((void**)&d_payOff.pricing_points, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_payOff.floating_schedule, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_payOff.fixed_schedule, TIMEPOINTS * sizeof(double)));
    //checkCudaErrors(cudaMemcpy(d_payOff.pricing_points, payOff.pricing_points, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_payOff.floating_schedule, payOff.floating_schedule, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    //checkCudaErrors(cudaMemcpy(d_payOff.fixed_schedule, payOff.fixed_schedule, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    
    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(double)));

    // spot_rates, drifts, volatilities, forward_rates, discount_factors, exposures
    checkCudaErrors(cudaMalloc((void**)&d_rngStates, grid.x * block.x * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_fwd_rates, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_fwd_rates0, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_forward_rates, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_discount_factors, grid.x * block.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_exposures, grid.x * block.x * TIMEPOINTS * sizeof(double)));

    // Copy the spot_rates, drift & volatilities to device global memory
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));

   
    // HJM Model number of paths
    int pathN = 2500;
    
    // Initialise RNG
    initRNG2<<< grid, block >>>(d_rngStates, m_seed);

    // Generate Paths and Compute IRS Mark to Market
    generatePaths <<<grid, block >>> (
        exposures, d_spot_rates, d_drifts, d_volatilities, d_forward_rates, d_discount_factors,
        d_fwd_rates, d_fwd_rates0, d_payOff, d_rngStates, pathN, TIMEPOINTS
    );
    
    // Wait for all kernel to finish
    cudaDeviceSynchronize();

    // Copy partial results back
    //checkCudaErrors( cudaMemcpy(&exposures[0], d_exposures, grid.x * sizeof(double), cudaMemcpyDeviceToHost) );

    // Free Reserved Memory
    if (d_rngStates) {
        cudaFree(d_rngStates);
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

    if (d_forward_rates) {
        cudaFree(d_forward_rates);
    }

    if (d_discount_factors) {
        cudaFree(d_discount_factors);
    }

    if (d_exposures) {
        cudaFree(d_exposures);
    }

    if (d_fwd_rates) {
        cudaFree(d_fwd_rates);
    }

    if (d_fwd_rates0) {
        cudaFree(d_fwd_rates0);
    }
}



