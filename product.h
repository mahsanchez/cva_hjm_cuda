

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