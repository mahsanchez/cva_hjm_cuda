#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iterator>
#include <iomanip>
#include <random>
#include "product.h"

#include "mkl.h"
#include "mkl_vsl.h"

#ifdef EXPOSURE_GPU_CUDA
#include "simulation_gpu.h"
#endif

using namespace std;

/*
 * Gaussian Random Number generators using Intel MKL
 */
class VSLRNGRandomGenerator {
public:
    VSLRNGRandomGenerator() {
        vslNewStream( &stream, VSL_BRNG_MT19937, 777 );
    }

    ~VSLRNGRandomGenerator() {
        vslDeleteStream( &stream );
    }

    void operator() (double *phi_random, int count) {
        vdRngGaussian( VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, count, phi_random, 0.0, 1.0 );
    }
private:
    VSLStreamStatePtr stream;
};


/*
 * (Musiela Parameterisation HJM) We simulate f(t+dt)=f(t) + dfbar  where SDE dfbar =  m(t)*dt+SUM(Vol_i*phi*SQRT(dt))+dF/dtau*dt  and phi ~ N(0,1)
 */
class HJMStochasticProcess {
public:
    HJMStochasticProcess(std::vector<double> &spot_rates_, std::vector<double> &drifts_, std::vector<std::vector<double>>& volatilities_, int dimension_, double dt_, double dtau_, int size_) :
            spot_rates(spot_rates_), drifts(drifts_), volatilities(volatilities_), dimension(dimension_), dt(dt_), dtau(dtau_), size(size_) {
    }

    inline void evolve(std::vector<double> &fwd_rates, std::vector<double> &fwd_rates0, double *gaussian_rand) {

        // evolve the sde
        for (int t = 0; t < size; t++) {
            // calculate diffusion term SUM(Vol_i*phi*SQRT(dt))
            double dfbar =  volatilities[0][t] * gaussian_rand[0];
            dfbar +=  volatilities[1][t] * gaussian_rand[1];
            dfbar +=  volatilities[2][t] * gaussian_rand[2];
            dfbar *= std::sqrt(dt);

            // calculate the drift m(t)*dt
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
private:
    std::vector<double> &drifts;
    std::vector<double> &spot_rates;
    std::vector<std::vector<double>>& volatilities;
    int size;
    int dimension;
    double dt;
    double dtau;
};

/*
 * HeathJarrowMortonModel StochasticProcess
 */
class HeathJarrowMortonModel {
public:

    HeathJarrowMortonModel(std::vector<double> &spot_rates_, std::vector<double> &drifts_, std::vector<std::vector<double>>& volatilities_, int dimension_, double dt_, double dtau_, double expiry_) :
            spot_rates(spot_rates_), drifts(drifts_), volatilities(volatilities_),  dimension(dimension_), dt(dt_), dtau(dtau_), expiry(expiry_),
            stochasticProcess(spot_rates_, drifts_, volatilities_, dimension_, dt_, dtau_, expiry_/dtau_ + 1)
    {
        size = expiry/dtau + 1;
        forward_rates = std::vector<double>(size);
        discount_factors = std::vector<double>(size);
    }

    // LMM SDE
    void simulate(double *gaussian_rand) {
        int delta = (int) (1.0 / dt);
        int simN = (int) delta * expiry/dtau;

        //
        std::vector<double> fwd_rates(size);
        std::vector<double> fwd_rates0(size);
        std::vector<double> accum_rates(size);

        // initialize the simulation with the spot_rates curve
        std::copy(spot_rates.begin(), spot_rates.end(), fwd_rates0.begin());

        // evolve the hjm model
        for (int sim = 1; sim < simN; sim++) {

            // evolve the hjm sde
            stochasticProcess.evolve(fwd_rates, fwd_rates0, &gaussian_rand[dimension*sim]);

            //accumulate the sum of rates between sim-1 and sim
            std::transform (fwd_rates.begin(), fwd_rates.end(), accum_rates.begin(), accum_rates.begin(), std::plus<double>());

            //register sim whose index constribute to numeraire tenors
            if (sim % delta == 0) {
               int index = sim/delta;
               forward_rates[index] = fwd_rates[index];
               discount_factors[index] = accum_rates[index];
            }

#ifndef DEBUG_HJM_DE
            std::cout << sim << " ";
            for (int i = 0; i < fwd_rates.size(); i++) {
                std::cout << fwd_rates[i] << " ";
            }
            std::cout << "   " << gaussian_rand[dimension*sim] << " " << gaussian_rand[dimension*sim+1] << " " << gaussian_rand[dimension*sim+2];
            std::cout << std::endl;
#endif
            //swap fwd_rates vectors
            std::copy(fwd_rates.begin(), fwd_rates.end(), fwd_rates0.begin());
        }

        // compute discount factors
        forward_rates[0] = spot_rates[0];
        discount_factors[0] = spot_rates[0];
        std::transform(discount_factors.begin(), discount_factors.end(), discount_factors.begin(), [&](double x) {
            return std::exp(-x * dt);
        });

#ifndef DEBUG_HJM_NUMERAIRE
        std::cout << "simulated forward_rates";
        for (int i = 0; i < size; i++) {
            std::cout << i << " " << fwd_rates[i];
            std::cout << std::endl;
        }
        std::cout << "discount_factors";
        for (int i = 0; i < size; i++) {
            std::cout << i << " " << discount_factors[i];
            std::cout << std::endl;
        }
#endif
    }

    // return forward_rates & discount_factors
    void numeraire(std::vector<double> &forward_rates, std::vector<double> &discount_factors) {
        std::copy(this->forward_rates.begin(), this->forward_rates.end(), forward_rates.begin());
        std::copy(this->discount_factors.begin(), this->discount_factors.end(), discount_factors.begin());
    }

    //return number of randoms to use per simulation
    int randomCount() {
        int count = expiry/dtau;
        return (1.0/dt) * count;
    }

    inline int getSize() {
        return size;
    }

private:
    HJMStochasticProcess stochasticProcess;
    std::vector<double> forward_rates;
    std::vector<double> discount_factors;
    std::vector<double> &drifts;
    std::vector<double> &spot_rates;
    std::vector<std::vector<double>>& volatilities;
    int dimension;
    double dt;
    double dtau;
    int size;
    double expiry;
};


/*
* Monte Carlo Simulation Engine
* Run simulation to generate the stochastics Forward Rates Risk Factors Grid using LMM Model
* Simulates forward rates using Euler-Maruyama time-stepping procedure.
* MC capture simulation statistics (MC Error, stddeviation, avg)
 */
template<typename InterestRateModel, typename PayOff>
class MonteCarloSimulation {
public:
    MonteCarloSimulation(PayOff &payOff_, InterestRateModel &model_, double *phi_random_, int simN_) :
            payOff(payOff_), phi_random(phi_random_), model(model_), simN(simN_)
     {
        forward_rates = std::vector<double>(model.getSize());
        discount_factors = std::vector<double>(model.getSize());
    }

    /**
     * Monte Carlo Method calculation method engine
     */
    void calculate(std::vector<std::vector<double>> &exposures, double &duration) {

#ifdef EXPOSURE_CPU
        // calculate exposure profiles using CPU
        auto start = std::chrono::high_resolution_clock::now();
        // Run simulation to generate Forward Rates Risk Factors Grid using HJM Model and Musiela SDE ( CURAND )
        for (int sim = 1; sim < simN; sim++) {
            // Evolve the Forward Rates Risk Factor Simulation Path using Interest Rate Model
            generatePaths(&phi_random[sim * model.randomCount()]);

            // Interest Rate Swap Mark to Market pricing the IRS across all pricing points
            pricePayOff(exposures[sim]);
        }

       // Display EE[t] curve realization for simulation sim
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        duration = elapsed.count();
#else
        // calculate exposure profiles using CUDA
        calculateExposureGPU(exposures, payOff, spot_rates, drift, volatilities, simN);
#endif
    }

    /*
     * Trivial implementation to Evolve the Forward Rates Risk Factor Simulation using HJM MonteCarlo
     */
    void generatePaths(double *phi_random) {
        model.simulate(phi_random);
    }

    /*
     * Mark to Market Exposure profile calculation for Interest Rate Swap (marktomarket) pricePayOff()
     */
    void pricePayOff(std::vector<double> &exposure) {

        // Simulated Discount Factors and Forward Rates Curves
        model.numeraire(forward_rates, discount_factors);

        // IRS Pricer
        auto irs_pricer = [&]() {
            double price = 0;
            for (int i = 0; i < forward_rates.size(); i++) {
                double sum = discount_factors[i] * payOff.notional * payOff.dtau * ( forward_rates[i] - payOff.K );
                price += sum;
            }
            return price;
        };

        // Mark to Market
        for (int i = 1; i < model.getSize(); i++) {
            exposure[i] = std::max(irs_pricer(), 0.0);
        }
    }

protected:
    double *phi_random;
    PayOff &payOff;
    InterestRateModel &model;
    std::vector<double> forward_rates;
    std::vector<double> discount_factors;
    int simN;
};


