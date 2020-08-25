

struct InterestRateSwap {
    InterestRateSwap(double* pricing_points_, double* floating_schedule_, double* fixed_schedule_, double notional_, double K_, double expiry_, double dtau_) :
        pricing_points(pricing_points_), floating_schedule(floating_schedule_), fixed_schedule(fixed_schedule_), notional(notional_), K(K_), expiry(expiry_), dtau(dtau_)
    {}

    double* pricing_points;
    double* floating_schedule;
    double* fixed_schedule;
    double notional;
    double K;
    double dtau;
    double expiry;
};