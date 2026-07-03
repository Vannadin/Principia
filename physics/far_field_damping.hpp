#pragma once

#include "base/algebra.hpp"
#include "quantities/quantities.hpp"

namespace principia {
namespace physics {
namespace _far_field_damping {
namespace internal {

using namespace principia::base::_algebra;
using namespace principia::quantities::_quantities;

// Specification of the damping of the far field of a massive body, acting as a
// radial multiplier on its point-mass potential:
//   V_damped = σ(‖r‖) V(r).
// σ is 1 below the inner threshold, 0 above the outer threshold, and a C²
// quintic sigmoid in between.  The damped potential is a smooth conservative
// truncation of the far field: bodies beyond the outer threshold exert exactly
// no force, without the potential discontinuity that a hard cutoff would
// introduce (which would inject energy into the flows at every crossing).
// This is the far-field analogue of `HarmonicDamping`, which damps the
// individual harmonics of the geopotential.
class FarFieldDamping final {
 public:
  // No damping, σ = 1 everywhere.
  FarFieldDamping() = default;
  explicit FarFieldDamping(Length const& outer_threshold);

  // Above this threshold, the contribution of the body to the potential is 0,
  // i.e., σ = 0.
  Length const& outer_threshold() const;
  // Below this threshold, the contribution of the body to the potential is
  // undamped, σ = 1.
  // This class depends on the invariant: outer_threshold = 3 * inner_threshold.
  Length const& inner_threshold() const;

  // The square of the outer threshold, to make it possible to elide bodies
  // without computing a square root.
  Square<Length> const& outer_threshold²() const;

  // Sets σ and σ′ r according to σ as defined by `*this`.  The acceleration
  // deriving from the damped point-mass potential −σ μ / r is
  //   (σ − σ′ r) μ Δq / r³,
  // where Δq is the vector from the field point to the body.
  void ComputeDampedRadialQuantities(Length const& r_norm,
                                     double& σ,
                                     double& σʹr) const;

  // Same as above, but also computes σ″ r², which is needed for the jerk.
  void ComputeDampedRadialQuantities(Length const& r_norm,
                                     double& σ,
                                     double& σʹr,
                                     double& σʺr²) const;

 private:
  Length outer_threshold_ = Infinity<Length>;
  Length inner_threshold_ = Infinity<Length>;
  Square<Length> outer_threshold²_ = Infinity<Square<Length>>;
  // The inverse of the width of the sigmoid shell,
  // 1 / (outer_threshold - inner_threshold).
  Inverse<Length> one_over_shell_width_;
};

}  // namespace internal

using internal::FarFieldDamping;

}  // namespace _far_field_damping
}  // namespace physics
}  // namespace principia

#include "physics/far_field_damping_body.hpp"
