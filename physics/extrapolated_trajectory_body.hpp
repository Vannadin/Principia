#pragma once

#include "physics/extrapolated_trajectory.hpp"

#include "absl/log/check.h"

namespace principia {
namespace physics {
namespace _extrapolated_trajectory {
namespace internal {

template<typename Frame>
ExtrapolatedTrajectory<Frame>::ExtrapolatedTrajectory(
    Trajectory<Frame> const& trajectory,
    Instant const& horizon,
    AnalyticSubsystemMotion<Frame> const& model,
    int const member)
    : trajectory_(trajectory),
      horizon_(horizon),
      model_(model),
      member_(member) {
  CHECK_LE(model_.epoch(), horizon_);
  CHECK_LE(horizon_, trajectory_.t_max());
}

template<typename Frame>
Instant ExtrapolatedTrajectory<Frame>::t_min() const {
  return trajectory_.t_min();
}

template<typename Frame>
Instant ExtrapolatedTrajectory<Frame>::t_max() const {
  return InfiniteFuture;
}

template<typename Frame>
Position<Frame> ExtrapolatedTrajectory<Frame>::EvaluatePosition(
    Instant const& time) const {
  if (time <= horizon_) {
    return trajectory_.EvaluatePosition(time);
  }
  return model_.EvaluateDegreesOfFreedom(member_, time).position();
}

template<typename Frame>
Velocity<Frame> ExtrapolatedTrajectory<Frame>::EvaluateVelocity(
    Instant const& time) const {
  if (time <= horizon_) {
    return trajectory_.EvaluateVelocity(time);
  }
  return model_.EvaluateDegreesOfFreedom(member_, time).velocity();
}

template<typename Frame>
DegreesOfFreedom<Frame>
ExtrapolatedTrajectory<Frame>::EvaluateDegreesOfFreedom(
    Instant const& time) const {
  if (time <= horizon_) {
    return trajectory_.EvaluateDegreesOfFreedom(time);
  }
  return model_.EvaluateDegreesOfFreedom(member_, time);
}

}  // namespace internal
}  // namespace _extrapolated_trajectory
}  // namespace physics
}  // namespace principia
