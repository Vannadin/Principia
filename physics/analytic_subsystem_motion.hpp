#pragma once

#include <optional>
#include <vector>

#include "geometry/instant.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/kepler_orbit.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/quantities.hpp"

namespace principia {
namespace physics {
namespace _analytic_subsystem_motion {
namespace internal {

using namespace principia::geometry::_instant;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_kepler_orbit;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;

// A closed-form model of the motion of the bodies of one stellar subsystem
// beyond an `epoch` at which their true states were last known.  The bodies
// form a tree given by `Member::parent`; each Jacobi vector — the barycentre
// of a satellite subtree relative to the barycentre of the bodies interior to
// it — is propagated independently, on the two-body orbit osculating at the
// epoch when the pair is bound and close enough for the elements to be
// well-conditioned, and linearly otherwise; the barycentre of the whole
// subsystem moves uniformly.  The model reproduces the given states at the
// epoch with matching velocities, so a trajectory continued by it is C¹ at
// the seam.
template<typename Frame>
class AnalyticSubsystemMotion final {
  static_assert(!Frame::may_rotate);

 public:
  struct Member final {
    GravitationalParameter gravitational_parameter;
    // At the epoch; the positions of all the members must share one origin.
    DegreesOfFreedom<Frame> degrees_of_freedom;
    // The index of the member to which this one is gravitationally
    // subordinate; nullopt for the single root.  A body orbiting a close pair
    // must designate the pair's primary: the model then propagates it about
    // the barycentre of the pair, not about either component.
    std::optional<int> parent;
  };

  // If `previous` is given it must model the same tree, and a pair whose
  // orbital energy is within `hysteresis_band_` of zero keeps the tier it had
  // in `previous`: a borderline pair flipping tiers between rebuilds would
  // move the far end of the extrapolation discontinuously.
  AnalyticSubsystemMotion(std::vector<Member> const& members,
                          Instant const& epoch,
                          AnalyticSubsystemMotion const* previous = nullptr);

  Instant const& epoch() const;

  // The state of the given member at `t`, in the same frame and relative to
  // the same origin as the states given at construction.  `t` may lie on
  // either side of the epoch; the model remains an osculation *at* the epoch.
  DegreesOfFreedom<Frame> EvaluateDegreesOfFreedom(int member,
                                                   Instant const& t) const;

  // Whether the Jacobi vector positioning `member`'s subtree is propagated on
  // a two-body orbit; `member` must not be the root.
  bool JacobiVectorIsKeplerian(int member) const;

 private:
  // The Jacobi vector of the subtree of one member relative to the cluster
  // interior to it.
  struct JacobiVector final {
    // Kept for the linear tier and for the hysteresis energy test.
    RelativeDegreesOfFreedom<Frame> state_at_epoch;
    // Nullopt in the linear tier.
    std::optional<KeplerOrbit<Frame>> kepler;
    // μ(satellite subtree) / μ(inner cluster ∪ satellite subtree): the factor
    // by which this vector displaces the barycentre of the cluster including
    // the satellite from that of the inner cluster.
    double weight;
  };

  // Whether the pair with the given relative state and total gravitational
  // parameter gets the two-body tier.
  static bool PairIsKeplerian(
      RelativeDegreesOfFreedom<Frame> const& state,
      GravitationalParameter const& pair_gravitational_parameter,
      std::optional<bool> previous_tier_is_keplerian);

  RelativeDegreesOfFreedom<Frame> EvaluateJacobiVector(JacobiVector const& v,
                                                       Instant const& t) const;

  // A pair whose energy is within this fraction of its potential scale
  // μ / (2 r) — equivalently, whose separation is within this fraction of its
  // |semimajor axis| — is borderline: its tier is kept across rebuilds.
  static constexpr double hysteresis_band_ = 0.05;
  // An eccentricity this close to 1 falls back to the linear tier:
  // `KeplerOrbit` is singular at the parabola (`StateVectors` is not
  // implemented at e = 1, and the elliptic elements lose all precision next
  // to it).
  static constexpr double near_parabolic_band_ = 1e-6;

  Instant const epoch_;
  std::vector<std::optional<int>> parents_;
  // The path from the root (inclusive) to each member (inclusive).
  std::vector<std::vector<int>> paths_;
  // The children of each member, ordered by increasing separation from it at
  // the epoch; the inner clusters of the Jacobi vectors accumulate in this
  // order.
  std::vector<std::vector<int>> ordered_children_;
  // Indexed by the satellite member; nullopt for the root.
  std::vector<std::optional<JacobiVector>> jacobi_;
  DegreesOfFreedom<Frame> barycentre_at_epoch_;
};

}  // namespace internal

using internal::AnalyticSubsystemMotion;

}  // namespace _analytic_subsystem_motion
}  // namespace physics
}  // namespace principia

#include "physics/analytic_subsystem_motion_body.hpp"
