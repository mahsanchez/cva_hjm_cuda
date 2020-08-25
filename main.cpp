#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/p_square_quantile.hpp>

#include "simulation.h"

using namespace boost::accumulators;

// ATM Market Cap Volatility
std::vector<double> tenor = {
        0.0, 0.50, 1.00, 1.50, 2.0, 2.50, 3.0, 3.50, 4.0, 4.50, 5.0, 5.50, 6.0, 6.50, 7.0, 7.50, 8.0, 8.5, 9.0, 9.5, 10.0, 10.50,
        11, 11.50, 12.0, 12.50, 13.0, 13.50, 14.0, 14.50, 15.0, 15.50, 16.0, 16.50, 17.0, 17.50, 18.0, 18.50, 19.0, 19.50,
        20.0, 20.50, 21.0, 21.50, 22.0, 22.50, 23.0, 23.50, 24.0, 24.50, 25.0

};

// Zero Coupon Bonds
std::vector<double> zcb = {
        0.9947527, 0.9892651, 0.9834984, 0.9774658, 0.9712884, 0.9648035, 0.9580084, 0.9509789, 0.9440868, 0.9369436, 0.9295484, 0.9219838,
        0.9145031, 0.9068886, 0.8990590, 0.8911017, 0.8833709, 0.8754579, 0.8673616, 0.8581725
};


// Year Fractions
std::vector<double> yearFractions_ = {
        0.25000, 0.25278, 0.25556, 0.25556, 0.25000, 0.25278, 0.25556, 0.25556, 0.25000, 0.25278, 0.25556, 0.25556, 0.25278, 0.25278, 0.26111,
        0.25278, 0.25278, 0.25278, 0.25278, 0.25278
};

std::vector<double> yearFractions0_ = {
        0.0, 0.248804, 0.501322, 0.502435, 0.859816, 1.08345, 1.30411, 2.35441, 1.95083, 2.19285, 2.43783, 3.4725, 3.08374,  3.31253, 3.63714,
        3.97711, 4.16463, 4.41673, 4.66739, 4.81921
};

// First row is the last observed forward curve (BOE data)
std::vector<double> spot_rates = {
        0.046138361,0.045251174,0.042915805,0.04283311,0.043497719,0.044053792,0.044439518,0.044708496,0.04490347,0.045056615,0.045184474,0.045294052,0.045386152,0.045458337,0.045507803,0.045534188,
        0.045541867,0.045534237,0.045513128,0.045477583,0.04542292,0.045344477,0.04523777,0.045097856,0.044925591,0.04472353,0.044494505,0.044242804,0.043973184,0.043690404,0.043399223,0.043104398,
        0.042810688,0.042522852,0.042244909,0.041978295,0.041723875,0.041482518,0.04125509,0.041042459,0.040845492,0.040665047,0.040501255,0.040353009,0.040219084,0.040098253,0.039989288,0.039890964,
        0.039802053,0.039721437,0.03964844
};

// Volatility Calibration
/*
 * HJM Calibration (volatilities, drift)
 */
std::vector<std::vector<double>> volatilities = //(3, std::vector<double>(51, 0.0));
{
        {0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
         0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
         0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,0.006430655,
         0.006430655,0.006430655,0.006430655
         },
        {-0.003556543,-0.003811644,-0.004010345,-0.004155341,-0.004249328,-0.004295001,-0.004295055,-0.004252185,-0.004169088,-0.004048459,-0.003892993,-0.003705385,-0.003488331,-0.003244526,-0.002976667,
         -0.002687447,-0.002379563,-0.00205571,-0.001718584,-0.001370879,-0.001015292,-0.000654518,-0.000291251,7.18113E-05,0.000431975,0.000786544,0.001132823,0.001468117,0.001789731,0.002094968,0.002381133,
         0.002645532,0.002885468,0.003098246,0.003281171,0.003431548,0.00354668,0.003623873,0.00366043,0.003653657,0.003600859,0.003499339,0.003346403,0.003139354,0.002875498,0.002552139,0.002166581,0.00171613,
         0.00119809,0.000609765,-5.15406E-05
         },
        {-0.004750672,-0.003908573,-0.003134891,-0.00242728,-0.001783395,-0.001200891,-0.000677421,-0.00021064,0.000201797,0.000562236,0.000873023,0.001136502,0.00135502,0.001530923,0.001666555,0.001764262,
         0.00182639,0.001855285,0.001853291,0.001822755,0.001766022,0.001685437,0.001583346,0.001462095,0.00132403,0.001171495,0.001006836,0.000832399,0.00065053,0.000463573,0.000273875,8.37816E-05,-0.000104363,
         -0.000288212,-0.00046542,-0.000633643,-0.000790533,-0.000933746,-0.001060936,-0.001169758,-0.001257866,-0.001322914,-0.001362557,-0.001374449,-0.001356245,-0.0013056,-0.001220167,-0.001097601,-0.000935557,
         -0.000731689,-0.000483652
        }
};

// Drift Callibration
std::vector<double> drifts = {
        0.000000,0.000036,0.000069,0.000099,0.000128,0.000155,0.000182,0.000208,0.000233,0.000257,0.000280,0.000303,0.000324,0.000345,0.000364,0.000381,0.000397,0.000412,0.000426,
        0.000438,0.000449,0.000459,0.000469,0.000478,0.000487,0.000496,0.000505,0.000515,0.000526,0.000538,0.000551,0.000566,0.000583,0.000601,0.000621,0.000643,0.000667,0.000692,0.000718,0.000746,0.000774,
        0.000803,0.000832,0.000861,0.000889,0.000916,0.000943,0.000967,0.000991,0.001013,0.001035
};


// CDS Spreads IRELAND
std::vector<double> spreads = {
        208.5, 187.5, 214, 235, 255, 272, 225, 233, 218, 215, 203, 199, 202, 196.71, 225.92, 219.69, 229.74, 232.12, 223.02, 224.45, 212,
        211.51, 206.25, 203.37, 212.94, 211.02, 210.06, 206.23, 211.49, 209.09, 204.3, 204.77, 199.98, 200.94, 202.38, 205.72, 204.76,
        210.02, 209.54, 209.54, 212.41, 213.35, 208.57, 208.56, 220.05, 226.26, 227.2, 222.89, 228.63, 231.5, 247.75, 255.37, 251.07,
        256.33, 252.01, 254.88, 246.98, 238.12, 241.95, 244.33, 252.45, 250.53, 246.71, 256.26, 255.78, 257.19, 247.63, 237.12, 234.73,
        226.36, 218, 219.9, 224.68, 221.81, 220.38, 211.77, 203.17, 206.04, 220, 218, 225, 217.5, 215, 220.5, 250.25, 260.5, 269.5, 258, 258.5,
        263, 274.5, 265.5, 268.5, 273.5, 275, 271.5, 263.75, 275, 287, 281, 271, 280.25, 284.5, 272, 275.5, 264.25, 274.25, 269.5, 264.5, 256.5,
        258, 265, 260, 262.5, 268, 272.25, 271, 274, 278.5, 278.5, 284.5, 290, 274, 264.5, 262.5, 247.75, 250.5, 248, 244.75, 246.5, 237.5, 240.5,
        236, 245.5, 237.75, 234.25, 235, 224, 215.5, 217, 217, 220.5, 208.5, 202.47, 203.43, 206.77, 205.33, 200.05, 202.91, 205.05, 222.51, 218.9,
        218.43, 221.31, 217.24, 218.67, 216.52, 216.5, 217.94, 208.37, 205.01, 200.95, 203.1, 203.81, 206.2, 204.28, 200.93, 202.36, 200.44, 197.8,
        199.23, 209.74, 211.18, 214.05, 215, 228.62, 233.63, 222.86, 218.8, 214.49, 217.36, 216.4, 213.52, 215.43, 219.49, 214.22, 218.05, 211.1,
        205.13, 207.75, 201.78, 199.39, 200.58, 199.14, 192.45, 188.38, 191.24, 192.9, 192.18, 190.26, 187.88, 186.44, 183.09, 181.18, 194.75,
        194.75, 200.75, 203, 204, 207.5, 207.25, 209, 205.75, 207.5, 203.5, 202.25, 199.5, 202, 201, 198.25, 191.5, 187.75, 188.75, 190, 193, 193.75,
        200, 200, 204, 194.5, 192.25, 189.5, 188.5, 186.5, 187.5, 193.75, 196.5, 205, 204.25, 208, 211.75, 217, 213.25, 212.5, 213.75, 211.25, 214,
        220.5, 212.5, 228, 225.75, 226.5, 233, 228.25, 225.5, 229, 229.5, 220.25, 220, 220, 223, 221, 216.5, 211.25, 199.75, 192.5, 193.94, 192.74,
        193.7, 195.6, 194.4, 195.36, 193.92, 185.79, 179.81, 162.13, 165.23, 168.81, 172.63, 168.81, 164.51, 159.01, 159.25, 154.95,
        152.08, 153.75, 153.74, 157.32, 160.66, 157.79, 152.54, 152.54, 156.84, 159.22, 162.08, 161.13, 166.38, 168.75, 73.75, 174.93, 182.8, 185.9,
        213.58, 212.15, 207.84, 214.76, 211.17, 206.62, 199.46, 196.37, 191.59, 183.95, 185.15, 172.97, 169.38, 160.08, 162.47, 159.6, 157.21, 147.91,
        145.29, 146.01, 141.95, 143.38, 136, 124.55, 119.07, 116.69, 122.17, 122.41, 122.41, 136, 136.5, 138.75, 138.5, 136.5, 135, 131.5, 130.75, 131.5,
        133.25, 133.75, 134, 136.5, 133.75, 131, 121.5, 120.5, 115.5, 114.25, 110.25, 110, 110, 110.5, 111, 111.5, 110, 112, 112, 113.25, 115.5, 120, 115.75,
        112.75, 114, 112.25, 114.5, 117, 118.25, 127, 139.5, 131, 131.75, 133, 130, 127, 132.25, 128.5, 132.25, 134.5, 136.5, 135, 138.25, 138, 134.5, 138,
        138.5, 135, 130.75, 132.5, 129.75, 125.5, 124.5, 125, 120, 119, 121.05, 122.49, 122.97, 124.88, 119.61, 119.61, 126.79, 126.78, 127.26, 127.73, 127.26,
        122.47, 118.17, 118.41, 118.65, 120.08, 123.42, 119.84, 118.64, 122.71, 124.86, 121.27, 120.79
};

struct cva_stats {
    double average;
    double median;
    double max;
    double quantile_75;
    double quantile_95;
    double quartiles;
};


/**
 * Calculate General Statistis (Mean, Percentile 95%)
 * @param stats_vector
 * @param exposures
 * @param timepoints_size
 * @param simN
 */

void report_statistics(std::vector<cva_stats>& stats_vector, std::vector<std::vector<double>>& exposures, int timepoints_size, int simN) {
    std::vector<Real> _distribution(simN);
    GeneralStatistics statistics;

    for (int t = 0; t < timepoints_size; t++) {
        for (int s = 0; s < simN; s++) {
            _distribution[s] = exposures[s][t];
        }
        statistics.addSequence(_distribution.begin(), _distribution.end());

        stats_vector[t].average = statistics.mean();
        stats_vector[t].quantile_75 = statistics.percentile(0.75);
        stats_vector[t].quantile_95 = statistics.percentile(0.95);
        stats_vector[t].max = statistics.max();

        statistics.reset();
    }
}

/*
 * Main Entry Point
 */

int main() {
    const double expiry = 25.0;
    const double dtau = 0.5;

    /*
     * Interest Rate Swap Product Definition 5Y Floating, 5Y Fixing 3M reset
     */
    std::vector<double> floating_schedule = {
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0
    };
    std::vector<double> fixed_schedule = {
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0
    };
    InterestRateSwap payOff(floating_schedule, floating_schedule,  fixed_schedule, 10, 0.025, expiry, dtau);

    /**
     * Monte Carlo Simulation Simulation & Exposure Profiles Generation
     */
    int simN = 3000; // Total Number of Simulations

    // Initialize the gaussians variates
    //  Increase the simulations numbers and analyze convergence and std deviation
    double duration = 0.0;
    int dimension = 3;
    double dt = expiry/simN;
    int size = expiry/dtau + 1;

    // Random Number Generation
    int random_count = (1.0/dt) * size * simN;
    double *phi_random = (double *) malloc(random_count * sizeof(double));
    VSLRNGRandomGenerator vsl_gaussian;
    vsl_gaussian(&phi_random[0], random_count);

    // HJM - Interest Rate  Model
    HeathJarrowMortonModel heathJarrowMortonModel(spot_rates, drifts, volatilities, dimension, dt, dtau, expiry);

    // Generate one Exposure Profile by Simulation Step
    std::vector<std::vector<double>> exposures(simN, std::vector<double>(size, 0.0));

    // Monte Carlo Simulation Engine generate the Exposure IRS Grid
    MonteCarloSimulation<HeathJarrowMortonModel, InterestRateSwap> mc_engine(payOff, heathJarrowMortonModel, phi_random, simN);
    mc_engine.calculate(exposures, duration);

    // free resources
    free(phi_random);

#ifdef DEBUG_EXPOSURE_CVA
    std::cout << "Exposures Profile" << std::endl;
    for (int i = 0; i < exposures.size(); i++) {
        for (int j = 0; j < exposures[j].size(); j++) {
            std::cout << exposures[i][j] << " ";
        }
        std::cout << std::endl;
    }
#endif

    /*
     * Counter Party Credit Risk Analytics & Expected Exposure Profile
     */
    double recovery = 0.04;      // Recovery Rates
    size = tenor.size() - 1;

    // Expected Exposure Profile
    std::vector<cva_stats> stats_vector;
    stats_vector.resize(size);

    // Use reduction to generate the expectation on the distribution across timepoints the expected exposure profile for the IRS
    // Calculate Statistics max, median, quartiles, 97.5th percentile on exposures
    // Calculate Potential Future Exposure (PFE) at the 97.5th percentile and media of the Positive EE
    report_statistics(stats_vector, exposures, size, simN);

    // Calculate Expected Exposure (EE)  EPE(t) = 𝔼 [max(V , 0)|Ft]
    std::vector<double> expected_exposure(tenor.size(), 0.0);
    std::transform(stats_vector.begin(), stats_vector.end(), expected_exposure.begin(), [](cva_stats s) {
        return s.average;
    });

    // Report Expected Exposure Profile Curve
#ifdef DEBUG_EXPOSURE_CVA
    std::cout << "Tenors" << std::endl;
    for (int j = 0; j < tenor.size(); j++) {
        std::cout << tenor[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "Expected Exposure EE[t]" << std::endl;
    for (int j = 0; j < expected_exposure.size(); j++) {
        std::cout << expected_exposure[j] << " ";
    }
    std::cout << std::endl;
    std::cout << "Potential Future Exposure (0.95) PFE[t]" << std::endl;
    for (int j = 0; j < stats_vector.size(); j++) {
        std::cout << stats_vector[j].quantile_95 << " ";
    }
    std::cout << std::endl;
#endif

    /*
     * Counter Party Credit Risk Analytics & Credit Value Adjustment
     */

    InterpolatedFICurve zcbCurve(tenor, zcb);
    InterpolatedFICurve spotRatesCurve(tenor, spot_rates);
    InterpolatedSpreadsCurve spreadsCurve(tenor, spreads, 0.0001);

    // Survival Probability Bootstrapping
    SurvivalProbCurve survivalProbCurve(tenor, spreadsCurve, zcbCurve, recovery);

#ifdef DEBUG_CDS_CVA
    std::cout << "CDS - Survival Probabilities" << std::endl;
    for (int i = 0; i < tenor.size(); i++) {
        std::cout << tenor[i] << " " << survivalProbCurve(tenor[i]) << std::endl;
    }
    std::cout << std::endl;
#endif

    // Calculate The Unilateral CVA - Credit Value Adjustment Metrics Calculation.
    // For two conterparties A - B. Counterparty A want to know how much is loss in a contract due to B defaulting

    /*
    expected_exposure = {0.0, 0.253899346, 0.262649146,0.264752078 , 0.269196504, 0.26545791, 0.258220705,	0.251405001, 0.244940384, 0.232968488, 0.21709824,
                         0.201386603, 0.184246938, 0.164233229, 0.142736561, 0.121057271, 0.098013021, 0.07231877, 0.042947986, 0.0 };
    */

    InterpolatedFICurve expectedExposureCurve(tenor, expected_exposure);

    /*
     * CVA Calculation
     * CVA =  E [ (1 - R) [ DF[t] * EE[t] * dPD[t] ] ]
     */
    std::vector<double> defaultProbabilities;
    std::transform(tenor.begin(), tenor.end(), std::back_inserter(defaultProbabilities), [&](double t) {
        return 1.0 - survivalProbCurve(t);
    });

    double cva = 0.0;
    for (int i = 1; i < tenor.size(); i++) {
        cva += (1 - recovery) * zcbCurve(tenor[i]) * expectedExposureCurve(tenor[i]) * (defaultProbabilities[i] - defaultProbabilities[i-1]);
    }

#ifdef DEBUG_EXPOSURE_CVA
    std::cout << "Credit value Adjustment " << std::endl;
    std::cout << std::setprecision(6)<< std::fixed << cva << " " << simN << " " << duration << std::endl;
#endif

    exit(0);
}