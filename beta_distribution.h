#pragma once
#include <random>


class BetaDistribution {
 public:
  explicit BetaDistribution(double a = 2.0, double b = 2.0)
      : a_gamma_(a), b_gamma_(b) { }

  template <typename URNG>
  double operator()(URNG& engine) {
      return generate(engine);
  }

 private:
  std::gamma_distribution<double> a_gamma_, b_gamma_;

  template <typename URNG>
  double generate(URNG& engine) {
      double x = a_gamma_(engine);
      return x / (x + b_gamma_(engine));
  }
};

