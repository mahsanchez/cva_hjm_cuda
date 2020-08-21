#ifdef EXPOSURE_GPU_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed)
{
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}

// Reduction Kernel
template <typename Real>
__device__ Real reduce_sum(Real in)
{
    SharedMemory<Real> sdata;

    // Perform first level of reduction:
    // - Write to shared memory
    unsigned int ltid = threadIdx.x;

    sdata[ltid] = in;
    __syncthreads();

    // Do reduction in shared mem
    for (unsigned int s = blockDim.x / 2 ; s > 0 ; s >>= 1)
    {
        if (ltid < s)
        {
            sdata[ltid] += sdata[ltid + s];
        }

        __syncthreads();
    }

    return sdata[0];
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
void sde_evolve(float *drift,
           float *volatility,
           float *fwd_rates,
           float *fwd_rates0,
           double dtau,
           double dt,
           int size,
           curandState &state)
{
    for (int t = 0; t < size; t++) {
        // difussion
        double dfbar = volatilities[t] * curand_normal_double(&state);
        dfbar += volatilities[size + t] * curand_normal_double(&state);
        dfbar += volatilities[size*2 + t] * curand_normal_double(&state);
        dfbar *= std::sqrt(dt);

        // dift
        dfbar += drifts[t] * dt;

        // calculate dF/dtau*dt
        double dF = 0.0;
        if (t < (size - 1)) {
            dF += fwd_rates0[t+1] - fwd_rates0[t];
        }
        else {
            dF += fwd_rates0[t] - fwd_rates0[t-1];
        }
        dfbar += (dF/dtau) * dt;

        // apply Euler Maruyana
        fwd_rates[t] = fwd_rates0[t] + dfbar;
    }
}

/*
 * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
*/
__device__
void pricePayOff(double *exposure, PayOff &payOff, double *forward_rates, double *discount_factors, double expiry) {

    int size = payOff.expiry/payOff.dtau + 1;

    for (int i = 1; i < size-1; i++) {
        double price = 0;
        for (int t = i; t < size; t++) {
            double sum = discount_factors[t] * payOff.notional * payOff.dtau * ( forward_rates[t] - payOff.K );
            price += sum;
        }
        exposure[i] = max(price, 0.0);
    }
}

/*
 * Path generation kernel [Exposure]
 */
__global__ void generatePaths(double *exposures, double *drift, double *volatilities, curandState *rngStates, PayOff &payOff, double dt, double dtau, const int size)
{
        // Determine thread ID
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int step = gridDim.x * blockDim.x;

        // Compute Parameters
        __shared__ double fwd_rates[];
        __shared__ double fwd_rates0[];
        __shared__ double accum_rates[];
        __shared__ double forward_rates[];
        __shared__ double discount_factors[];

        // Simulation Parameters
        int delta = (int) (1.0 / dt);
        int simN = (int) delta * expiry/dtau;

        // Initialise the RNG
        curandState localState = rngStates[tid];

        // local memory thread base position
        int base = threadIdx.x * size;

        // evolve the hjm model
        for (int sim = 1; sim < simN; sim++) {

            // evolve the hjm sde
            sde_evolve(drift, volatilities, &fwd_rates[base], fwd_rates0[base], dtau, dt, size, localState);

            //accumulate the sum of rates between sim-1 and sim
            for (int i = base; i < size; i++) {
                accum_rates[i] = accum_rates[i] + fwd_rates[i];
            }

            //register sim whose index constribute to numeraire tenors
            if (sim % delta == 0) {
               int index = base + sim/delta;
               forward_rates[index] = fwd_rates[index];
               discount_factors[index] = accum_rates[index];
            }

            //swap fwd_rates vectors
            swap(fwd_rates, fwd_rates0);
        }

        // compute discount factors
        forward_rates[base] = spot_rates[0];
        discount_factors[base] = spot_rates[0];
        for (int i = base; i < size; i++) {
             discount_factors[i] = exp(-discount_factors[i] * dt);
        }

        // perform mark to market and compute exposure
        pricePayOff(exposure[tid*size], payOff, &forward_rates[base], &discount_factors[base], expiry);
}

// kernel invodation

void calculareExposureGPU(double *exposures, PauOff &payOff, double *spot_rates, double *drift, double *volatilities, int simN) {

    // calculate execution configuration
    int blockSize = 256;
    int numBlocks = (simN + blockSize - 1) / blockSize;
    int tile_width = blockSize * size;

    // seed
    double m_seed = 1234;

    // Allocate memory for exposures
    double *d_exposures = 0;
    cudaResult = cudaMalloc((void **)&d_exposures, grid.x * block.x * sizeof(double));

    // Allocate memory for RNG states
    curandState *d_rngStates = 0;
    cudaResult = cudaMalloc((void **)&d_rngStates, grid.x * block.x * sizeof(curandState));

    // Initialise RNG
    initRNG<<<grid, block>>>(d_rngStates, m_seed);

    //Copy the spot_rates, drift & volatilities to constant memory
    d_spot_rates, d_drift, d_volatilities

    // Generate Paths and Compute IRS Mark to Market
    generatePaths<<<grid, block>>>(exposures, d_spot_rates, d_drift, d_volatilities, rngStates,  payOff, dt, dtau, size);

    // Copy partial results back
    vector<Real> values(grid.x);
    cudaResult = cudaMemcpy(&exposures[0], d_exposures, grid.x * sizeof(Real), cudaMemcpyDeviceToHost);
}

#endif