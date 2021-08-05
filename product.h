#ifndef INTEREST_RATE_SWAP
#define INTEREST_RATE_SWAP

template <typename real>
struct InterestRateSwap {
    InterestRateSwap(real* pricing_points_, real* realing_schedule_, real* fixed_schedule_, real notional_, real K_, real expiry_, real dtau_) :
        pricing_points(pricing_points_), realing_schedule(realing_schedule_), fixed_schedule(fixed_schedule_), notional(notional_), K(K_), expiry(expiry_), dtau(dtau_)
    {}

    real* pricing_points;
    real* realing_schedule;
    real* fixed_schedule;
    real notional;
    real K;
    real dtau;
    real expiry;
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