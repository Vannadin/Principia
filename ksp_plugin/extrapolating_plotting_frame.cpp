#include "ksp_plugin/extrapolating_plotting_frame.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "base/not_null.hpp"

namespace principia {
namespace ksp_plugin {
namespace _extrapolating_plotting_frame {
namespace internal {

using namespace principia::base::_not_null;

ExtrapolatingPlottingFrame::ExtrapolatingPlottingFrame(
    not_null<PlottingFrame const*> const real)
    : real_(real),
      horizon_(InfiniteFuture) {}

ExtrapolatingPlottingFrame::ExtrapolatingPlottingFrame(
    not_null<Ephemeris<Barycentric> const*> const ephemeris,
    not_null<PlottingFrame const*> const real,
    not_null<MassiveBody const*> const centre,
    std::shared_ptr<AnalyticSubsystemMotion<Barycentric> const> model,
    int const member,
    Instant const& horizon)
    : real_(real),
      horizon_(horizon),
      model_(std::move(model)),
      view_(std::make_unique<ExtrapolatedTrajectory<Barycentric>>(
          *ephemeris->trajectory(centre),
          horizon,
          *model_,
          member)),
      twin_(std::make_unique<
            BodyCentredNonRotatingReferenceFrame<Barycentric, Navigation>>(
          ephemeris,
          centre,
          view_.get())) {}

Instant ExtrapolatingPlottingFrame::t_min() const {
  return real_->t_min();
}

Instant ExtrapolatingPlottingFrame::t_max() const {
  return real_->t_max();
}

Instant ExtrapolatingPlottingFrame::render_t_max() const {
  return twin_ == nullptr ? real_->t_max() : InfiniteFuture;
}

int ExtrapolatingPlottingFrame::subsystem() const {
  return real_->subsystem();
}

std::optional<Ephemeris<Barycentric>::Anchor>
ExtrapolatingPlottingFrame::anchor() const {
  return real_->anchor();
}

SimilarMotion<Barycentric, Navigation>
ExtrapolatingPlottingFrame::ToThisFrameAtTimeSimilarly(
    Instant const& t) const {
  if (twin_ != nullptr && t > horizon_) {
    return twin_->ToThisFrameAtTimeSimilarly(t);
  }
  return real_->ToThisFrameAtTimeSimilarly(t);
}

SimilarMotion<Navigation, Barycentric>
ExtrapolatingPlottingFrame::FromThisFrameAtTimeSimilarly(
    Instant const& t) const {
  if (twin_ != nullptr && t > horizon_) {
    return twin_->FromThisFrameAtTimeSimilarly(t);
  }
  return real_->FromThisFrameAtTimeSimilarly(t);
}

Vector<Acceleration, Navigation>
ExtrapolatingPlottingFrame::GeometricAcceleration(
    Instant const& t,
    DegreesOfFreedom<Navigation> const& degrees_of_freedom) const {
  return real_->GeometricAcceleration(t, degrees_of_freedom);
}

Vector<Acceleration, Navigation>
ExtrapolatingPlottingFrame::RotationFreeGeometricAccelerationAtRest(
    Instant const& t,
    Position<Navigation> const& position) const {
  return real_->RotationFreeGeometricAccelerationAtRest(t, position);
}

SpecificEnergy ExtrapolatingPlottingFrame::GeometricPotential(
    Instant const& t,
    Position<Navigation> const& position) const {
  return real_->GeometricPotential(t, position);
}

Rotation<Frenet<Navigation>, Navigation>
ExtrapolatingPlottingFrame::FrenetFrame(
    Instant const& t,
    DegreesOfFreedom<Navigation> const& degrees_of_freedom) const {
  return real_->FrenetFrame(t, degrees_of_freedom);
}

void ExtrapolatingPlottingFrame::WriteToMessage(
    not_null<serialization::ReferenceFrame*> const message) const {
  real_->WriteToMessage(message);
}

void AdoptTheRootless(
    std::vector<AnalyticSubsystemMotion<Barycentric>::Member>& members) {
  int root = -1;
  for (int m = 0; m < static_cast<int>(members.size()); ++m) {
    if (!members[m].parent.has_value() &&
        (root == -1 ||
         members[m].gravitational_parameter >
             members[root].gravitational_parameter)) {
      root = m;
    }
  }
  CHECK_NE(root, -1) << "no rootless member";
  for (int m = 0; m < static_cast<int>(members.size()); ++m) {
    if (m != root && !members[m].parent.has_value()) {
      members[m].parent = root;
    }
  }
}

}  // namespace internal
}  // namespace _extrapolating_plotting_frame
}  // namespace ksp_plugin
}  // namespace principia
