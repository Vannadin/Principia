#pragma once

#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "physics/analytic_subsystem_motion.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/trajectory.hpp"

namespace principia {
namespace physics {
namespace _extrapolated_trajectory {
namespace internal {

using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::physics::_analytic_subsystem_motion;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_trajectory;

// A view of a `Trajectory` extended past a `horizon` by an analytic model of
// the body's future motion: at or before the horizon it evaluates the viewed
// trajectory, beyond it the given `member` of the `model`.  The horizon is
// captured once — it does not follow later prolongations of the viewed
// trajectory — but `t_min` stays live.  The viewed objects must outlive this
// object.
template<typename Frame>
class ExtrapolatedTrajectory : public Trajectory<Frame> {
 public:
  // `horizon` must not precede the model's epoch, and the trajectory must be
  // evaluable at the horizon.
  ExtrapolatedTrajectory(Trajectory<Frame> const& trajectory,
                         Instant const& horizon,
                         AnalyticSubsystemMotion<Frame> const& model,
                         int member);

  // The `Trajectory` API.  `t_max` is the infinite future: the model is
  // defined at all times.
  Instant t_min() const override;
  Instant t_max() const override;
  Position<Frame> EvaluatePosition(Instant const& time) const override;
  Velocity<Frame> EvaluateVelocity(Instant const& time) const override;
  DegreesOfFreedom<Frame> EvaluateDegreesOfFreedom(
      Instant const& time) const override;

 private:
  Trajectory<Frame> const& trajectory_;
  Instant const horizon_;
  AnalyticSubsystemMotion<Frame> const& model_;
  int const member_;
};

}  // namespace internal

using internal::ExtrapolatedTrajectory;

}  // namespace _extrapolated_trajectory
}  // namespace physics
}  // namespace principia

#include "physics/extrapolated_trajectory_body.hpp"
