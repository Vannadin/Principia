#pragma once

#include "physics/jacobi_coordinates.hpp"

#include <cmath>
#include <vector>

#include "absl/log/log.h"
#include "geometry/instant.hpp"

namespace principia {
namespace physics {
namespace _jacobi_coordinates {
namespace internal {

using namespace principia::geometry::_instant;

template<typename Frame>
JacobiCoordinates<Frame>::JacobiCoordinates(MassiveBody const& primary)
    : primary_gravitational_parameter_(primary.gravitational_parameter()) {
  static DegreesOfFreedom<PrimocentricFrame> const motionless_origin = {
      PrimocentricFrame::origin, PrimocentricFrame::unmoving};
  primocentric_dof_.emplace_back(motionless_origin);
  system_barycentre_.Add(primocentric_dof_.back(),
                         primary.gravitational_parameter());
}

template<typename Frame>
void JacobiCoordinates<Frame>::Add(
    MassiveBody const& body,
    RelativeDegreesOfFreedom<Frame> const& dof_relative_to_system) {
  primocentric_dof_.emplace_back(system_barycentre_.Get() +
                                 id_fp_(dof_relative_to_system));
  system_barycentre_.Add(primocentric_dof_.back(),
                         body.gravitational_parameter());
}

template<typename Frame>
void JacobiCoordinates<Frame>::Add(
    MassiveBody const& body,
    KeplerianElements<Frame> const& osculating_elements_relative_to_system) {
  Instant const epoch;
  auto const& elements = osculating_elements_relative_to_system;
  // A mean motion or period is interpreted against the gravitational
  // parameter of the whole inner system; if it was computed for a two-body
  // orbit around the primary alone — as KSP does — the semimajor axis that
  // we derive is inflated by the cube root of the ratio of these parameters.
  // Negligible for planets around a star, grievous for stars around stars.
  if (!elements.semimajor_axis.has_value() &&
      (elements.mean_motion.has_value() || elements.period.has_value())) {
    double const inflation =
        std::cbrt((system_barycentre_.weight() +
                   body.gravitational_parameter()) /
                  primary_gravitational_parameter_);
    LOG_IF(WARNING, inflation > 1.01)
        << "Deriving the semimajor axis of " << body.name()
        << " from its mean motion or period yields " << inflation
        << " times the two-body value; specify the semimajor axis instead "
        << "if the period was computed against the primary alone";
  }
  Add(body,
      KeplerOrbit<Frame>(/*primary=*/System(),
                         /*secondary=*/body,
                         osculating_elements_relative_to_system,
                         epoch).StateVectors(epoch));
}

template<typename Frame>
MassiveBody JacobiCoordinates<Frame>::System() const {
  // A point mass.
  return MassiveBody(MassiveBody::Parameters(system_barycentre_.weight()));
}

template<typename Frame>
std::vector<RelativeDegreesOfFreedom<Frame>>
JacobiCoordinates<Frame>::BarycentricDegreesOfFreedom() const {
  DegreesOfFreedom<PrimocentricFrame> const system_barycentre =
      system_barycentre_.Get();
  std::vector<RelativeDegreesOfFreedom<Frame>> result;
  result.reserve(primocentric_dof_.size());
  for (auto const& dof : primocentric_dof_) {
    result.emplace_back(id_pf_(dof - system_barycentre));
  }
  return result;
}

template<typename Frame>
Identity<typename JacobiCoordinates<Frame>::PrimocentricFrame, Frame> const
    JacobiCoordinates<Frame>::id_pf_;
template<typename Frame>
Identity<Frame, typename JacobiCoordinates<Frame>::PrimocentricFrame> const
    JacobiCoordinates<Frame>::id_fp_;

}  // namespace internal
}  // namespace _jacobi_coordinates
}  // namespace physics
}  // namespace principia
