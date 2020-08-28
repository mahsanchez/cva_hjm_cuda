
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCK_SIZE 32

#define HJM_SDE_DEBUG



// RNG init kernel
__global__ void initRNG2(curandState* const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}


/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
void pricePayOff(double* exposure, InterestRateSwap payOff, double* forward_rates, double* discount_factors, double expiry) {

    int size = (int) payOff.expiry / payOff.dtau + 1;

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
 * Path generation kernel 
 */
__global__
void generatePaths(double* forward_rates, double *discount_factors, double* spot_rates, double* drifts, double* volatilities, double dtau, curandState* rngStates, const int pathN)
{
// Compute Parameters
    double phi0;
    double phi1;
    double phi2;
    double rate;
    double accum_rates[2];
    __shared__ double rates0[TIMEPOINTS];
    
    
 // Determine thread ID
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int laneid = threadIdx.x % 32;

 // Simulation Parameters
    double dt = 0.01; // 
    int delta = 100; //  
    const double sqrt_dt = sqrt(dt);

// Initialise the RNG
    curandState localState = rngStates[bid];

// Initialize rates0
    for (int block = 0; block < TIMEPOINTS; block += BLOCK_SIZE)
    {
        int t = block + tid;
        if (t < TIMEPOINTS) {
           rates0[t] = spot_rates[t];
        }     
        __syncthreads(); 
    }

// Simulation
    for (int sim = 0; sim < pathN; sim++) 
    {
        if (laneid == 0) {
            phi0 = curand_normal_double(&localState);
            phi1 = curand_normal_double(&localState);
            phi2 = curand_normal_double(&localState);
        }
// Broadcast random values across the whole warp
        __shfl_sync(0xffffffff, phi0, 0);
        __shfl_sync(0xffffffff, phi1, 0);
        __shfl_sync(0xffffffff, phi2, 0);

#ifdef HJM_SDE_DEBUG
        if (laneid != 0) {
            printf("Thread %d Normal variates %f %f %f.\n", threadIdx.x, phi0, phi1, phi2);
        }
#endif

// initialize rate0

        for (int block = 0; block < TIMEPOINTS; block += BLOCK_SIZE) 
        {
            int t = block + tid;

            if (t < TIMEPOINTS)
            {
                // We simulate the SDE f(t+dt)=f(t) + dfbar  
                // where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt and phi ~ N(0,1)
                double dfbar = volatilities[t] * phi0;
                dfbar += volatilities[TIMEPOINTS + t] * phi1;
                dfbar += volatilities[TIMEPOINTS * 2 + t] * phi2;
                dfbar *= sqrt_dt;
                // drift
                dfbar += drifts[t] * dt;
                // calculate dF/dtau*dt
                double dF = 0.0;

                if (t < (TIMEPOINTS - 1)) {
                    dF = rates0[t + 1] - rates0[t];
                }
                else {
                    dF = rates0[t] - rates0[t - 1];
                }
                dfbar += (dF / dtau) * dt;

                // apply Euler Maruyana
                rate = rates0[t] + dfbar;

                //
                rates0[t] = rate;
            }
             __syncthreads();

             // update numeraire based on simulation
             if (t < TIMEPOINTS) 
             {
                int i = (t < BLOCK_SIZE) ?  0 : 1;
                accum_rates[i] = accum_rates[i] + rate;

                if (sim % delta == 0) {
                    int index = sim / delta;
                    if (t == index) {
                       forward_rates[t] = rate;
                       double r = (t < BLOCK_SIZE) ? accum_rates[0] : accum_rates[1];
                       discount_factors[t] = exp(-r * dt);
                    }
                    
                }
            }
             __syncthreads();
            
        }

    }

    // Initialize numeraire at timepoint 0
    if (tid == 0) {
        double r = rates0[0];
        forward_rates[0] = r;
        discount_factors[0] = exp(-r * dt);
    }
}


/*
   Exposure Calculation Kernel Invocation
*/
void calculateExposureGPU(double* exposures, InterestRateSwap payOff, double* spot_rates, double* drift, double* volatilities, int simN) {

    // kernel execution configuration Determine how to divide the work between cores
    int blockSize = 32;
    dim3 block = blockSize; // blockSize;
    dim3 grid = (simN + blockSize - 1) / blockSize;

    // seed
    double m_seed = 1234;

    // Memory allocation 
    curandState* d_rngStates = 0;
    double* d_spot_rates = 0;
    double* d_drifts = 0;
    double* d_volatilities = 0;
    double* d_forward_rates = 0;
    double* d_discount_factors = 0;

    // CPU Host variables
    double* forward_rates = 0;
    double* discount_factors = 0;

    
    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(double)));

    // spot_rates, drifts, volatilities, forward_rates, discount_factors
    checkCudaErrors(cudaMalloc((void**)&d_rngStates, grid.x * block.x * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_forward_rates, grid.x * TIMEPOINTS * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&d_discount_factors, grid.x * TIMEPOINTS * sizeof(double)));

    // Copy the spot_rates, drift & volatilities to device global memory
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(double), cudaMemcpyHostToDevice));
   
    // HJM Model number of paths
    int pathN = 2500;
    
    // Initialise RNG
    initRNG2<<< grid, block >>>(d_rngStates, m_seed);

    // Generate Paths and Compute IRS Mark to Market
    generatePaths <<<grid, block >>> (d_forward_rates, d_discount_factors, d_spot_rates, d_drifts, d_volatilities, payOff.dtau, d_rngStates, pathN);
    
    // Wait for all kernel to finish
    cudaDeviceSynchronize();

    // Copy partial results back
    checkCudaErrors( cudaMemcpy(forward_rates, d_forward_rates, TIMEPOINTS * sizeof(double), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaMemcpy(discount_factors, d_discount_factors, TIMEPOINTS * sizeof(double), cudaMemcpyDeviceToHost));

    // Price Interest Rate Swap
    pricePayOff(exposures, payOff, forward_rates, discount_factors, payOff.expiry);

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
}



