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

// A view of a `Trajectory` whose positions are uniformly translated by a
// constant `displacement`.  The velocities are unaffected.  The viewed
// trajectory must outlive this object.
template<typename Frame>
class TranslatedTrajectory : public Trajectory<Frame> {
 public:
  TranslatedTrajectory(Trajectory<Frame> const& trajectory,
                       Displacement<Frame> const& displacement);

  // The `Trajectory` API.
  Instant t_min() const override;
  Instant t_max() const override;
  Position<Frame> EvaluatePosition(Instant const& time) const override;
  Velocity<Frame> EvaluateVelocity(Instant const& time) const override;
  DegreesOfFreedom<Frame> EvaluateDegreesOfFreedom(
      Instant const& time) const override;

 private:
  Trajectory<Frame> const& trajectory_;
  Displacement<Frame> const displacement_;
};

}  // namespace internal

using internal::TranslatedTrajectory;

}  // namespace _translated_trajectory
}  // namespace physics
}  // namespace principia

#include "physics/translated_trajectory_body.hpp"
