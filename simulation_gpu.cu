
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include "simulation_gpu.h"

#define TILE_DIM 51
#define TIMEPOINTS 51
#define VOL_DIM 3
#define BLOCK_SIZE 32

#define HJM_SDE_DEBUG
#define MC_RDM_DEBUG1
#define HJM_FORWARD_RATES
#define HJM_DISCOUNT_FACTORS
#define EXPOSURE_PROFILES_DEBUG

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
__device__
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


// RNG init kernel
__global__ void initRNG2(curandState* const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 6000, &rngStates[tid]);
}



/*
 * Path generation kernel 
 */
__global__
void generatePaths(float* exposure, float* forward_rates, float *discount_factors, float* spot_rates, float* drifts, float* volatilities, float dtau, curandState* rngStates, const int pathN)
{
// Compute Parameters
    float phi0;
    float phi1;
    float phi2;
    float rate;
    float accum_rates[2];
    __shared__ float rates0[TIMEPOINTS];
    
    
 // Determine thread ID
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int laneid = threadIdx.x % 32;

 // Simulation Parameters
    float dt = 0.01; // 
    int delta = 100; //  
    const float sqrt_dt = sqrt(dt);

// Initialise the RNG
    curandState  localState = rngStates[bid];

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
            phi0 = curand_normal(&localState);
            phi1 = curand_normal(&localState);
            phi2 = curand_normal(&localState);
        }
// Broadcast random values across the whole warp
        phi0 = __shfl_sync(0xffffffff, phi0, 0);
        phi1 = __shfl_sync(0xffffffff, phi1, 0);
        phi2 = __shfl_sync(0xffffffff, phi2, 0);

#ifdef MC_RDM_DEBUG
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
                float dfbar = volatilities[t] * phi0;
                dfbar += volatilities[TIMEPOINTS + t] * phi1;
                dfbar += volatilities[TIMEPOINTS * 2 + t] * phi2;
                dfbar *= sqrt_dt;
                // drift
                dfbar += drifts[t] * dt;
                // calculate dF/dtau*dt
                float dF = 0.0;

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
                accum_rates[i] += rate;

                if (sim % delta == 0) {
                    int index = sim / delta;
                    if (t == index) {
                       forward_rates[blockIdx.x * TIMEPOINTS + t] = rate;
                       float r = (t < BLOCK_SIZE) ? accum_rates[0] : accum_rates[1];
                       discount_factors[blockIdx.x * TIMEPOINTS + t] = exp(-r * dt);
                    }       
                }
            }   
        }
    }

}


/*
   Exposure Calculation Kernel Invocation
   TODO - BATCHING
*/
void calculateExposureGPU(float* exposures, InterestRateSwap payOff, float* spot_rates, float* drift, float* volatilities, int simN) {

    int _simN = 100;
    // kernel execution configuration Determine how to divide the work between cores
    int blockSize = 32;
    dim3 block = blockSize; // blockSize;
    dim3 grid = _simN; // 64 _simN;

    // seed
    float m_seed = 1234;

    // Memory allocation 
    curandState* d_rngStates = 0;
    float* d_spot_rates = 0;
    float* d_drifts = 0;
    float* d_volatilities = 0;
    float* d_forward_rates = 0;
    float* d_discount_factors = 0;
    float* d_exposures = 0;

    // CPU Host variables
    float* forward_rates = 0;
    float* discount_factors = 0;

    //
    forward_rates = (float*)malloc(grid.x * TIMEPOINTS * sizeof(float));
    discount_factors = (float*)malloc(grid.x * TIMEPOINTS * sizeof(float));

    // Copy the spot_rates, drift & volatilities to device memory
    checkCudaErrors(cudaMalloc((void**)&d_spot_rates, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_drifts, TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_volatilities, VOL_DIM * TIMEPOINTS * sizeof(float)));

    // spot_rates, drifts, volatilities, forward_rates, discount_factors
    checkCudaErrors(cudaMalloc((void**)&d_rngStates, grid.x * block.x * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void**)&d_forward_rates, grid.x * TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_discount_factors, grid.x * TIMEPOINTS * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&d_exposures, grid.x * TIMEPOINTS * sizeof(float)));

    // Copy the spot_rates, drift & volatilities to device global memory
    checkCudaErrors(cudaMemcpy(d_spot_rates, spot_rates, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_drifts, drift, TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_volatilities, volatilities, VOL_DIM * TIMEPOINTS * sizeof(float), cudaMemcpyHostToDevice));

    // HJM Model number of paths
    int pathN = 3800;

    // Initialise RNG
    initRNG2 << < grid, block >> > (d_rngStates, m_seed);

    // Generate Paths and Compute IRS Mark to Market
    generatePaths << <grid, block >> > (d_exposures, d_forward_rates, d_discount_factors, d_spot_rates, d_drifts, d_volatilities, payOff.dtau, d_rngStates, pathN);

    // Wait for all kernel to finish
    cudaDeviceSynchronize();

    // Done with MC Simulation 
#ifdef HJM_SDE_DEBUG
    printf("MC Simulation generated");
#endif

    // Copy partial results back
    checkCudaErrors(cudaMemcpy(forward_rates, d_forward_rates, grid.x * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(discount_factors, d_discount_factors, grid.x * TIMEPOINTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Interest Rate Swap Mark to Market
    // pricePayOff(float* exposure, float notional,  float dtau, float K, float* forward_rates, float* discount_factors, float expiry) 

#ifdef HJM_DISCOUNT_FACTORS
    for (int i = 0; i < _simN; i++) {
        printf("Discount Factors: %d\n", i);
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%d : %f ", t, discount_factors[i * TIMEPOINTS + t]);
        }
        printf("\n");
    }
#endif
#ifdef HJM_FORWARD_RATES
    for (int i = 0; i < _simN; i++) {
        printf("Forward Rate Curve: %d\n", i);
        for (int t = 0; t < TIMEPOINTS; t++) {
            printf("%d : %f ", t, forward_rates[i * TIMEPOINTS + t]);
        }
        printf("\n");
    }
#endif

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

    if (d_exposures) {
        cudaFree(d_discount_factors);
    }

    if (d_forward_rates) {
        cudaFree(d_forward_rates);
    }

    if (d_discount_factors) {
        cudaFree(d_discount_factors);
    }

    if (forward_rates) {
        free(forward_rates);
    }

    if (discount_factors) {
        free(discount_factors);
    }
}



