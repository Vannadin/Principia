#pragma once

#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/trajectory.hpp"

namespace principia {
namespace physics {
namespace _translated_trajectory {
namespace internal {

using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_trajectory;

// A view of a `Trajectory` whose positions are translated by a displacement
// affine in time, `displacement + velocity_offset * (time - epoch)`, and
// whose velocities are translated by `velocity_offset`.  This is the
// conversion between the representations of two subsystems whose local
// origins move relative to each other.  The viewed trajectory must outlive
// this object.
template<typename Frame>
class TranslatedTrajectory : public Trajectory<Frame> {
 public:
  // A translation by a constant `displacement`; the velocities are
  // unaffected.
  TranslatedTrajectory(Trajectory<Frame> const& trajectory,
                       Displacement<Frame> const& displacement);

  // A translation by `displacement_at_epoch + velocity_offset * (t - epoch)`;
  // the velocities are translated by `velocity_offset`.
  TranslatedTrajectory(Trajectory<Frame> const& trajectory,
                       Displacement<Frame> const& displacement_at_epoch,
                       Velocity<Frame> const& velocity_offset,
                       Instant const& epoch);

  // The `Trajectory` API.
  Instant t_min() const override;
  Instant t_max() const override;
  Position<Frame> EvaluatePosition(Instant const& time) const override;
  Velocity<Frame> EvaluateVelocity(Instant const& time) const override;
  DegreesOfFreedom<Frame> EvaluateDegreesOfFreedom(
      Instant const& time) const override;

 private:
  Displacement<Frame> displacement_at(Instant const& time) const;

  Trajectory<Frame> const& trajectory_;
  Displacement<Frame> const displacement_;
  Velocity<Frame> const velocity_offset_;
  Instant const epoch_;
};

}  // namespace internal

using internal::TranslatedTrajectory;

}  // namespace _translated_trajectory
}  // namespace physics
}  // namespace principia

#include "physics/translated_trajectory_body.hpp"
