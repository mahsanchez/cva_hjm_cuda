#ifndef INTEREST_RATE_SWAP
#define INTEREST_RATE_SWAP

struct InterestRateSwap {
    InterestRateSwap(float* pricing_points_, float* floating_schedule_, float* fixed_schedule_, float notional_, float K_, float expiry_, float dtau_) :
        pricing_points(pricing_points_), floating_schedule(floating_schedule_), fixed_schedule(fixed_schedule_), notional(notional_), K(K_), expiry(expiry_), dtau(dtau_)
    {}

    float* pricing_points;
    float* floating_schedule;
    float* fixed_schedule;
    float notional;
    float K;
    float dtau;
    float expiry;
};

struct InterestRateSwap_D {
    InterestRateSwap_D(double* pricing_points_, double* realing_schedule_, double* fixed_schedule_, double notional_, double K_, double expiry_, double dtau_) :
        pricing_points(pricing_points_), realing_schedule(realing_schedule_), fixed_schedule(fixed_schedule_), notional(notional_), K(K_), expiry(expiry_), dtau(dtau_)
    {}

    double* pricing_points;
    double* realing_schedule;
    double* fixed_schedule;
    double notional;
    double K;
    double dtau;
    double expiry;
};

#endif