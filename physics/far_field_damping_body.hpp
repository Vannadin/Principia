#pragma once

#include "physics/far_field_damping.hpp"

#include "numerics/elementary_functions.hpp"

namespace principia {
namespace physics {
namespace _far_field_damping {
namespace internal {

using namespace principia::numerics::_elementary_functions;

inline FarFieldDamping::FarFieldDamping(Length const& outer_threshold)
    : outer_threshold_(outer_threshold),
      inner_threshold_(outer_threshold / 3),
      outer_threshold²_(Pow<2>(outer_threshold)),
      one_over_shell_width_(1 / (outer_threshold_ - inner_threshold_)) {}

inline Length const& FarFieldDamping::outer_threshold() const {
  return outer_threshold_;
}

inline Length const& FarFieldDamping::inner_threshold() const {
  return inner_threshold_;
}

inline Square<Length> const& FarFieldDamping::outer_threshold²() const {
  return outer_threshold²_;
}

inline void FarFieldDamping::ComputeDampedRadialQuantities(
    Length const& r_norm,
    double& σ,
    double& σʹr) const {
  if (r_norm <= inner_threshold_) {
    // Below the inner threshold, σ = 1.
    σ = 1;
    σʹr = 0;
  } else if (r_norm >= outer_threshold_) {
    // Above the outer threshold, σ = 0.
    σ = 0;
    σʹr = 0;
  } else {
    // On the shell, σ is the C² quintic sigmoid 1 − 10x³ + 15x⁴ − 6x⁵ of
    // x = (r − inner_threshold) / (outer_threshold − inner_threshold).
    double const x = (r_norm - inner_threshold_) * one_over_shell_width_;
    double const x² = x * x;
    double const one_minus_x = 1 - x;
    σ = 1 - x² * x * (10 + x * (-15 + 6 * x));
    σʹr = -30 * x² * one_minus_x * one_minus_x *
          (r_norm * one_over_shell_width_);
  }
}

inline void FarFieldDamping::ComputeDampedRadialQuantities(
    Length const& r_norm,
    double& σ,
    double& σʹr,
    double& σʺr²) const {
  if (r_norm <= inner_threshold_ || r_norm >= outer_threshold_) {
    ComputeDampedRadialQuantities(r_norm, σ, σʹr);
    σʺr² = 0;
  } else {
    double const x = (r_norm - inner_threshold_) * one_over_shell_width_;
    double const x² = x * x;
    double const one_minus_x = 1 - x;
    double const r_over_shell_width = r_norm * one_over_shell_width_;
    σ = 1 - x² * x * (10 + x * (-15 + 6 * x));
    σʹr = -30 * x² * one_minus_x * one_minus_x * r_over_shell_width;
    σʺr² = -60 * x * one_minus_x * (1 - 2 * x) * Pow<2>(r_over_shell_width);
  }
}

}  // namespace internal
}  // namespace _far_field_damping
}  // namespace physics
}  // namespace principia
