#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/rotation.hpp"
#include "geometry/space.hpp"
#include "ksp_plugin/frames.hpp"
#include "physics/analytic_subsystem_motion.hpp"
#include "physics/body_centred_non_rotating_reference_frame.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/extrapolated_trajectory.hpp"
#include "physics/massive_body.hpp"
#include "physics/reference_frame.hpp"
#include "physics/similar_motion.hpp"
#include "quantities/named_quantities.hpp"
#include "serialization/physics.pb.h"

namespace principia {
namespace ksp_plugin {
namespace _extrapolating_plotting_frame {
namespace internal {

using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_space;
using namespace principia::ksp_plugin::_frames;
using namespace principia::physics::_analytic_subsystem_motion;
using namespace principia::physics::_body_centred_non_rotating_reference_frame;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_extrapolated_trajectory;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_reference_frame;
using namespace principia::physics::_similar_motion;
using namespace principia::quantities::_named_quantities;

// A decorator around the renderer's plotting frame, owned by the planetarium
// of one plot and never serialized.  With a twin — a body-centred
// non-rotating frame of the same centre whose trajectory is continued
// analytically past `horizon` — the kinematic conversions beyond the horizon
// evaluate the twin and `render_t_max` becomes unbounded, which is what lets
// a plot outlive the ephemeris; `t_max` keeps reporting the real frame's
// bound, which feeds the guards and marker clamps that must stay at the real
// horizon.  Without a twin every operation delegates to the real frame.
class ExtrapolatingPlottingFrame final : public PlottingFrame {
 public:
  explicit ExtrapolatingPlottingFrame(not_null<PlottingFrame const*> real);

  // The `real` frame must be centred on `centre`, and `member` must designate
  // `centre` in the `model`; `horizon` must not precede the model's epoch.
  ExtrapolatingPlottingFrame(
      not_null<Ephemeris<Barycentric> const*> ephemeris,
      not_null<PlottingFrame const*> real,
      not_null<MassiveBody const*> centre,
      std::shared_ptr<AnalyticSubsystemMotion<Barycentric> const> model,
      int member,
      Instant const& horizon);

  Instant t_min() const override;
  Instant t_max() const override;
  Instant render_t_max() const override;
  int subsystem() const override;
  std::optional<Ephemeris<Barycentric>::Anchor> anchor() const override;

  SimilarMotion<Barycentric, Navigation> ToThisFrameAtTimeSimilarly(
      Instant const& t) const override;
  SimilarMotion<Navigation, Barycentric> FromThisFrameAtTimeSimilarly(
      Instant const& t) const override;

  Vector<Acceleration, Navigation> GeometricAcceleration(
      Instant const& t,
      DegreesOfFreedom<Navigation> const& degrees_of_freedom) const override;
  Vector<Acceleration, Navigation> RotationFreeGeometricAccelerationAtRest(
      Instant const& t,
      Position<Navigation> const& position) const override;
  SpecificEnergy GeometricPotential(
      Instant const& t,
      Position<Navigation> const& position) const override;
  Rotation<Frenet<Navigation>, Navigation> FrenetFrame(
      Instant const& t,
      DegreesOfFreedom<Navigation> const& degrees_of_freedom) const override;

  void WriteToMessage(
      not_null<serialization::ReferenceFrame*> message) const override;

 private:
  not_null<PlottingFrame const*> const real_;
  Instant const horizon_;
  std::shared_ptr<AnalyticSubsystemMotion<Barycentric> const> const model_;
  std::unique_ptr<ExtrapolatedTrajectory<Barycentric> const> const view_;
  std::unique_ptr<
      BodyCentredNonRotatingReferenceFrame<Barycentric, Navigation> const>
      const twin_;
};

// Rewires `members` so that exactly one root remains: of the members whose
// parent lies outside the subsystem, the most massive becomes the root and
// the others become its satellites.  The components of a NearStars binary
// are parented to a common distant primary, not to each other, so restricting
// the KSP tree to the subsystem leaves them all rootless.
void AdoptTheRootless(
    std::vector<AnalyticSubsystemMotion<Barycentric>::Member>& members);

}  // namespace internal

using internal::AdoptTheRootless;
using internal::ExtrapolatingPlottingFrame;

}  // namespace _extrapolating_plotting_frame
}  // namespace ksp_plugin
}  // namespace principia
