#pragma once

#include "physics/translated_trajectory.hpp"

namespace principia {
namespace physics {
namespace _translated_trajectory {
namespace internal {

template<typename Frame>
TranslatedTrajectory<Frame>::TranslatedTrajectory(
    Trajectory<Frame> const& trajectory,
    Displacement<Frame> const& displacement)
    : trajectory_(trajectory),
      displacement_(displacement) {}

template<typename Frame>
Instant TranslatedTrajectory<Frame>::t_min() const {
  return trajectory_.t_min();
}

template<typename Frame>
Instant TranslatedTrajectory<Frame>::t_max() const {
  return trajectory_.t_max();
}

template<typename Frame>
Position<Frame> TranslatedTrajectory<Frame>::EvaluatePosition(
    Instant const& time) const {
  return trajectory_.EvaluatePosition(time) + displacement_;
}

template<typename Frame>
Velocity<Frame> TranslatedTrajectory<Frame>::EvaluateVelocity(
    Instant const& time) const {
  return trajectory_.EvaluateVelocity(time);
}

template<typename Frame>
DegreesOfFreedom<Frame> TranslatedTrajectory<Frame>::EvaluateDegreesOfFreedom(
    Instant const& time) const {
  DegreesOfFreedom<Frame> const degrees_of_freedom =
      trajectory_.EvaluateDegreesOfFreedom(time);
  return {degrees_of_freedom.position() + displacement_,
          degrees_of_freedom.velocity()};
}

}  // namespace internal
}  // namespace _translated_trajectory
}  // namespace physics
}  // namespace principia
