#include "ksp_plugin/renderer.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "base/ranges.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/permutation.hpp"
#include "physics/body_centred_body_direction_reference_frame.hpp"

namespace principia {
namespace ksp_plugin {
namespace _renderer {
namespace internal {

using namespace principia::base::_ranges;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_permutation;
using namespace principia::physics::_body_centred_body_direction_reference_frame;  // NOLINT

Renderer::Renderer(not_null<Celestial const*> const sun,
                   not_null<std::unique_ptr<PlottingFrame>> plotting_frame,
                   Ephemeris<Barycentric> const* const ephemeris)
    : sun_(sun),
      plotting_frame_(std::move(plotting_frame)),
      ephemeris_(ephemeris) {}

void Renderer::SetPlottingFrame(
    not_null<std::unique_ptr<PlottingFrame>> plotting_frame) {
  plotting_frame_ = std::move(plotting_frame);
}

not_null<PlottingFrame const*> Renderer::GetPlottingFrame() const {
  return target_ ? target_->target_frame.get()
                 : plotting_frame_.get();
}

void Renderer::SetTargetVessel(
    not_null<Vessel*> const vessel,
    not_null<Celestial const*> const celestial,
    not_null<Ephemeris<Barycentric> const*> const ephemeris) {
  if (!target_ ||
      target_->vessel != vessel ||
      target_->celestial != celestial) {
    target_.emplace(vessel, celestial, ephemeris);
  }
}

void Renderer::ClearTargetVessel() {
  target_ = std::nullopt;
}

void Renderer::ClearTargetVesselIf(not_null<Vessel*> const vessel) {
  if (target_ && target_->vessel == vessel) {
    target_ = std::nullopt;
  }
}

bool Renderer::HasTargetVessel() const {
  return static_cast<bool>(target_);
}

Vessel& Renderer::GetTargetVessel() {
  CHECK(target_);
  return *target_->vessel;
}

Vessel const& Renderer::GetTargetVessel() const {
  CHECK(target_);
  return *target_->vessel;
}

DiscreteTrajectory<World>
Renderer::RenderBarycentricTrajectoryInWorld(
    Instant const& time,
    DiscreteTrajectory<Barycentric>::iterator const& begin,
    DiscreteTrajectory<Barycentric>::iterator const& end,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    std::optional<WorldRegistration> const& registration) const {
  auto const trajectory_in_plotting_frame =
      RenderBarycentricTrajectoryInPlotting(begin, end, placement);
  auto trajectory_in_world =
      RenderPlottingTrajectoryInWorld(time,
                                      trajectory_in_plotting_frame.begin(),
                                      trajectory_in_plotting_frame.end(),
                                      sun_world_position,
                                      planetarium_rotation,
                                      registration);
  return trajectory_in_world;
}

DiscreteTrajectory<Navigation>
Renderer::RenderBarycentricTrajectoryInPlotting(
    DiscreteTrajectory<Barycentric>::iterator const& begin,
    DiscreteTrajectory<Barycentric>::iterator const& end,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement) const {
  Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
      GetPlottingFrame()->placement();
  // A void flight-plan coast reaches beyond the plotting frame's domain, and
  // a trimmed past can recede behind a stale plan's start; the frame cannot
  // be evaluated on either side.
  Instant const frame_t_min = GetPlottingFrame()->t_min();
  Instant const frame_t_max = GetPlottingFrame()->t_max();
  DiscreteTrajectory<Navigation> trajectory;
  for (auto it = begin; it != end; ++it) {
    auto const& [time, degrees_of_freedom] = *it;
    if (time < frame_t_min) {
      continue;
    }
    if (time > frame_t_max) {
      break;
    }
    if (target_) {
      auto const prediction = target_->vessel->prediction();
      if (time < prediction->t_min()) {
        continue;
      } else if (time > prediction->t_max()) {
        break;
      }
    }
    // The conversion is evaluated at each point's own time, into the plotting
    // frame's own placement: against a target-vessel frame — whose origin is
    // the target's anchored prediction — the anchors difference on the sector
    // lattice, so a void rendezvous is plotted at its true relative geometry.
    auto const [conversion_displacement, conversion_velocity] =
        PlacementConversion(placement, frame_placement, time);
    DegreesOfFreedom<Barycentric> const plotting_degrees_of_freedom = {
        degrees_of_freedom.position() + conversion_displacement,
        degrees_of_freedom.velocity() + conversion_velocity};
    trajectory.Append(time,
                      BarycentricToPlotting(time)(plotting_degrees_of_freedom))
        .IgnoreError();
  }
  return trajectory;
}

DiscreteTrajectory<World>
Renderer::RenderPlottingTrajectoryInWorld(
    Instant const& time,
    DiscreteTrajectory<Navigation>::iterator const& begin,
    DiscreteTrajectory<Navigation>::iterator const& end,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    std::optional<WorldRegistration> const& registration) const {
  return RenderPlottingContainerInWorld<DiscreteTrajectory>(
      time,
      begin, end,
      sun_world_position,
      planetarium_rotation,
      [](DiscreteTrajectory<World>& trajectory,
         Instant const& t,
         DegreesOfFreedom<World> const& world_degrees_of_freedom) {
        trajectory.Append(t, world_degrees_of_freedom).IgnoreError();
      },
      registration);
}

DistinguishedPoints<World> Renderer::RenderDistinguishedPointsInWorld(
    Instant const& time,
    DistinguishedPoints<Barycentric>::const_iterator const begin,
    DistinguishedPoints<Barycentric>::const_iterator const end,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    std::optional<WorldRegistration> const& registration) const {
  Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
      GetPlottingFrame()->placement();
  if (registration.has_value()) {
    // The velocities are unaffected by the anchoring: their error only enters
    // through the angular motion of the frame.
    AnchoredWorldMapping const anchored_mapping(
        *this, time, *registration, planetarium_rotation, placement);
    DistinguishedPoints<World> world_points;
    for (auto const& [t, degrees_of_freedom] : Range(begin, end)) {
      auto const [conversion_displacement, conversion_velocity] =
          PlacementConversion(placement, frame_placement, t);
      DegreesOfFreedom<Barycentric> const converted_degrees_of_freedom = {
          degrees_of_freedom.position() + conversion_displacement,
          degrees_of_freedom.velocity() + conversion_velocity};
      auto const plotting_degrees_of_freedom =
          BarycentricToPlotting(t)(converted_degrees_of_freedom);
      world_points.emplace(
          t,
          DegreesOfFreedom<World>(
              anchored_mapping(t, degrees_of_freedom.position()),
              Permutation<Navigation, World>(
                  Permutation<Navigation, World>::CoordinatePermutation::YXZ)(
                  PlottingToWorld(t, planetarium_rotation).scale() *
                  plotting_degrees_of_freedom.velocity())));
    }
    return world_points;
  }
  DistinguishedPoints<Navigation> plotting_points;
  for (auto const& [t, degrees_of_freedom] : Range(begin, end)) {
    // The conversion is evaluated at each point's own time, into the plotting
    // frame's own placement; see `RenderBarycentricTrajectoryInPlotting`.
    auto const [conversion_displacement, conversion_velocity] =
        PlacementConversion(placement, frame_placement, t);
    DegreesOfFreedom<Barycentric> const converted_degrees_of_freedom = {
        degrees_of_freedom.position() + conversion_displacement,
        degrees_of_freedom.velocity() + conversion_velocity};
    auto const plotting_degrees_of_freedom =
        BarycentricToPlotting(t)(converted_degrees_of_freedom);
    plotting_points.emplace(t, plotting_degrees_of_freedom);
  }
  return RenderPlottingContainerInWorld<DistinguishedPoints>(
      time,
      plotting_points.begin(),
      plotting_points.end(),
      sun_world_position,
      planetarium_rotation,
      [](DistinguishedPoints<World>& world_points,
         Instant const& t,
         DegreesOfFreedom<World> const& world_degrees_of_freedom) {
        world_points.emplace(t, world_degrees_of_freedom);
      },
      /*registration=*/std::nullopt);
}

std::vector<Renderer::Node>
Renderer::RenderNodes(
    Instant const& time,
    Trajectory<Barycentric> const& trajectory,
    DistinguishedPoints<Navigation>::const_iterator const& begin,
    DistinguishedPoints<Navigation>::const_iterator const& end,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    std::optional<WorldRegistration> const& registration) const {
  std::vector<Node> nodes;
  // The nodes were found on `Navigation` positions, which are absolutes at
  // the distance to the plotting frame's origin; registered, they are placed
  // from `trajectory`, whose representation is local, at their own times —
  // which keep the rounding of the search.
  std::optional<AnchoredWorldMapping> anchored_mapping;
  std::optional<Similarity<Navigation, World>>
      from_plotting_frame_to_world_at_current_time;
  if (registration.has_value()) {
    anchored_mapping.emplace(
        *this, time, *registration, planetarium_rotation, placement);
  } else {
    from_plotting_frame_to_world_at_current_time =
        PlottingToWorld(time, sun_world_position, planetarium_rotation);
  }
  for (auto const& [t, degrees_of_freedom] : Range(begin, end)) {
    DegreesOfFreedom<Navigation> const& navigation_degrees_of_freedom =
        degrees_of_freedom;
    ConformalMap<double, Navigation, World> const
        from_plotting_frame_to_world_at_t =
            PlottingToWorld(t, planetarium_rotation);
    nodes.push_back(
        {.time = t,
         .position =
             anchored_mapping.has_value()
                 ? (*anchored_mapping)(t, trajectory.EvaluatePosition(t))
                 : (*from_plotting_frame_to_world_at_current_time)(
                       navigation_degrees_of_freedom.position()),
         .apparent_inclination =
             AngleBetween(Bivector<double, Navigation>({0, 0, 1}),
                          Wedge(navigation_degrees_of_freedom.position() -
                                    Navigation::origin,
                                navigation_degrees_of_freedom.velocity())),
         .out_of_plane_velocity =
             from_plotting_frame_to_world_at_t.scale() *
             navigation_degrees_of_freedom.velocity().coordinates().z});
  }
  return nodes;
}

SimilarMotion<Barycentric, Navigation> Renderer::BarycentricToPlotting(
    Instant const& time) const {
  return GetPlottingFrame()->ToThisFrameAtTimeSimilarly(time);
}

RigidTransformation<Barycentric, World> Renderer::BarycentricToWorld(
    Instant const& time,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement) const {
  Position<Barycentric> sun_position = sun_->current_position(time);
  auto const [conversion_displacement, conversion_velocity] =
      PlacementConversion({sun_->subsystem(), std::nullopt}, placement, time);
  sun_position += conversion_displacement;
  return RigidTransformation<Barycentric, World>(
      sun_position,
      sun_world_position,
      BarycentricToWorld(planetarium_rotation));
}

OrthogonalMap<Barycentric, World> Renderer::BarycentricToWorld(
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return OrthogonalMap<WorldSun, World>::Identity() *
         BarycentricToWorldSun(planetarium_rotation);
}

OrthogonalMap<Barycentric, WorldSun> Renderer::BarycentricToWorldSun(
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return sun_looking_glass.Inverse().Forget<OrthogonalMap>() *
         planetarium_rotation.Forget<OrthogonalMap>();
}

OrthogonalMap<Frenet<Navigation>, World> Renderer::FrenetToWorld(
    Instant const& time,
    NavigationManœuvre const& manœuvre,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  // Same "abilities some consider unnatural" as above.  A frame that does not
  // cover the manœuvre — a stale plan can start before a trimmed past, a void
  // plan can burn past the render ceiling — maps it through the nearest state
  // it has, on either side.
  Instant const initial_time =
      std::min(std::max(manœuvre.initial_time(), GetPlottingFrame()->t_min()),
               GetPlottingFrame()->render_t_max());
  return PlottingToWorld(time, planetarium_rotation).orthogonal_map¹₁() *
         BarycentricToPlotting(initial_time)
             .conformal_map().orthogonal_map¹₁() *
         manœuvre.FrenetFrame();
}

OrthogonalMap<Frenet<Navigation>, World> Renderer::FrenetToWorld(
    Vessel const& vessel,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  auto const& back = vessel.psychohistory()->back();
  auto const [conversion_displacement, conversion_velocity] =
      PlacementConversion(vessel.placement(),
                          GetPlottingFrame()->placement(),
                          back.time);
  DegreesOfFreedom<Barycentric> const barycentric_degrees_of_freedom = {
      back.degrees_of_freedom.position() + conversion_displacement,
      back.degrees_of_freedom.velocity() + conversion_velocity};
  DegreesOfFreedom<Navigation> const plotting_frame_degrees_of_freedom =
      BarycentricToPlotting(back.time)(barycentric_degrees_of_freedom);
  Rotation<Frenet<Navigation>, Navigation> const
      frenet_frame_to_plotting_frame =
          GetPlottingFrame()->FrenetFrame(
              back.time,
              plotting_frame_degrees_of_freedom);

  return PlottingToWorld(back.time, planetarium_rotation).orthogonal_map¹₁() *
         frenet_frame_to_plotting_frame.Forget<OrthogonalMap>();
}

OrthogonalMap<Frenet<Navigation>, World> Renderer::FrenetToWorld(
    Vessel const& vessel,
    NavigationFrame const& navigation_frame,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  auto const& back = vessel.psychohistory()->back();
  auto const [conversion_displacement, conversion_velocity] =
      PlacementConversion(vessel.placement(),
                          navigation_frame.placement(),
                          back.time);
  DegreesOfFreedom<Barycentric> const degrees_of_freedom = {
      back.degrees_of_freedom.position() + conversion_displacement,
      back.degrees_of_freedom.velocity() + conversion_velocity};
  auto const to_navigation = navigation_frame.ToThisFrameAtTime(back.time);
  auto const from_navigation = to_navigation.orthogonal_map().Inverse();
  auto const frenet_frame =
      navigation_frame.FrenetFrame(
          back.time,
          to_navigation(degrees_of_freedom)).Forget<OrthogonalMap>();
  return BarycentricToWorld(planetarium_rotation) * from_navigation *
         frenet_frame;
}

ConformalMap<double, Navigation, Barycentric> Renderer::PlottingToBarycentric(
    Instant const& time) const {
  return GetPlottingFrame()->FromThisFrameAtTimeSimilarly(time).conformal_map();
}

Similarity<Navigation, World> Renderer::PlottingToWorld(
    Instant const& time,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return BarycentricToWorld(time,
                            sun_world_position,
                            planetarium_rotation,
                            GetPlottingFrame()->placement())
             .Forget<Similarity>() *
         GetPlottingFrame()->FromThisFrameAtTimeSimilarly(time).similarity();
}

Similarity<Navigation, World> Renderer::RegisteredPlottingToWorld(
    Instant const& time,
    WorldRegistration const& registration,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  // Into the plotting frame's placement, as the rendered points are; the
  // conversion is on the sector lattice, so a vessel anchored in the void
  // keeps its true geometry relative to the frame.
  Position<Barycentric> position = registration.position;
  if (Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
          GetPlottingFrame()->placement();
      registration.placement != frame_placement) {
    position +=
        PlacementConversion(registration.placement, frame_placement, time)
            .first;
  }
  return Similarity<Navigation, World>(
      BarycentricToPlotting(time).similarity()(position),
      registration.world,
      PlottingToWorld(time, planetarium_rotation));
}

Renderer::AnchoredWorldMapping::AnchoredWorldMapping(
    Renderer const& renderer,
    Instant const& time,
    WorldRegistration const& registration,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement)
    : plotting_frame_(renderer.GetPlottingFrame()),
      t_ref_(registration.time),
      world_(registration.world),
      plotting_to_world_(
          renderer.PlottingToWorld(time, planetarium_rotation)),
      q_ref_(registration.position) {
  // The conversion from `placement` to the plotting frame's, folded at the
  // registration's time and extrapolated affinely by `operator()`: the anchor
  // terms are exactly affine, and the subsystem barycentres are inertial to
  // the precision of a rendering.  Folding once keeps the void-scale rounding
  // constant, so it displaces the plotted set without deforming it.
  if (Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
          plotting_frame_->placement();
      placement != frame_placement) {
    auto const conversion =
        renderer.PlacementConversion(placement, frame_placement, t_ref_);
    offset_ = conversion.first;
    velocity_offset_ = conversion.second;
  }
  // The registration, brought into `placement`; see
  // `RegisteredPlottingToWorld`.
  if (registration.placement != placement) {
    q_ref_ +=
        renderer
            .PlacementConversion(registration.placement, placement, t_ref_)
            .first;
  }
  SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t_ref =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t_ref_);
  auto const& similarity_ref = to_plotting_frame_at_t_ref.similarity();
  frame_origin_ref_ = similarity_ref.Inverse()(Navigation::origin);
  reference_ = (q_ref_ + offset_) - frame_origin_ref_;
  reference_in_navigation_ = similarity_ref.linear_map()(reference_);
}

Position<World> Renderer::AnchoredWorldMapping::operator()(
    Instant const& t,
    Position<Barycentric> const& position) const {
  SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t);
  auto const& similarity = to_plotting_frame_at_t.similarity();
  // A local difference, plus the affine part of the conversion at `t`.
  Displacement<Barycentric> const from_reference =
      (position - q_ref_) + velocity_offset_ * (t - t_ref_);
  Displacement<Barycentric> const frame_origin_motion =
      similarity.Inverse()(Navigation::origin) - frame_origin_ref_;
  // Exactly zero for an inertial frame, whose linear map does not change; for
  // a rotating or pulsating one this rounds at the ULP of the reference
  // distance, dwarfed by the arc or the dilation at that distance.
  Displacement<Navigation> const sweep =
      similarity.linear_map()(reference_) - reference_in_navigation_;
  Displacement<Navigation> const reference_relative =
      sweep +
      similarity.linear_map()(from_reference - frame_origin_motion);
  return world_ + plotting_to_world_(reference_relative);
}

ConformalMap<double, Navigation, World> Renderer::PlottingToWorld(
    Instant const& time,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return BarycentricToWorld(planetarium_rotation).Forget<ConformalMap>() *
         PlottingToBarycentric(time);
}

RigidTransformation<World, Barycentric> Renderer::WorldToBarycentric(
    Instant const& time,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement) const {
  return BarycentricToWorld(
             time, sun_world_position, planetarium_rotation, placement)
      .Inverse();
}

OrthogonalMap<World, Barycentric> Renderer::WorldToBarycentric(
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return BarycentricToWorld(planetarium_rotation).Inverse();
}

Similarity<World, Navigation> Renderer::WorldToPlotting(
    Instant const& time,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation) const {
  return BarycentricToPlotting(time).similarity() *
         WorldToBarycentric(time,
                            sun_world_position,
                            planetarium_rotation,
                            GetPlottingFrame()->placement())
             .Forget<Similarity>();
}

Rotation<CameraCompensatedReference, World> Renderer::CameraReferenceRotation(
    Instant const& time,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    Rotation<CameraCompensatedReference, CameraReference> const&
        camera_compensation) const {
  Permutation<Barycentric, CelestialSphere> const celestial_mirror(
      OddPermutation::XZY);
  Permutation<CameraReference, Navigation> const camera_mirror(
      OddPermutation::XZY);
  return (BarycentricToWorld(planetarium_rotation) *
          GetPlottingFrame()->FromThisFrameAtTimeSimilarly(time)
              .conformal_map().orthogonal_map¹₁() *
          camera_mirror.Forget<OrthogonalMap>()).AsRotation() *
         camera_compensation;
}

void Renderer::WriteToMessage(
    not_null<serialization::Renderer*> message) const {
  plotting_frame_->WriteToMessage(message->mutable_plotting_frame());
  // No serialization of the `target_`.
}

not_null<std::unique_ptr<Renderer>> Renderer::ReadFromMessage(
    serialization::Renderer const& message,
    not_null<Celestial const*> sun,
    not_null<Ephemeris<Barycentric> const*> const ephemeris) {
  return make_not_null_unique<Renderer>(
      sun,
      PlottingFrame::ReadFromMessage(message.plotting_frame(), ephemeris),
      ephemeris);
}

Renderer::Target::Target(
    not_null<Vessel*> const vessel,
    not_null<Celestial const*> const celestial,
    not_null<Ephemeris<Barycentric> const*> const ephemeris)
    : vessel(vessel),
      celestial(celestial),
      target_frame(
          make_not_null_unique<
              BodyCentredBodyDirectionReferenceFrame<Barycentric, Navigation>>(
              ephemeris,
              [this]() -> auto& { return *this->vessel->prediction(); },
              celestial->body(),
              [this]() { return this->vessel->placement().subsystem; },
              [this]() { return this->vessel->placement().anchor; })) {}

std::pair<Displacement<Barycentric>, Velocity<Barycentric>>
Renderer::PlacementConversion(
    Ephemeris<Barycentric>::SubsystemPlacement const& from,
    Ephemeris<Barycentric>::SubsystemPlacement const& to,
    Instant const& t) const {
  if (ephemeris_ == nullptr) {
    // Tests construct renderers without an ephemeris; the anchors do not need
    // one.
    return Ephemeris<Barycentric>::Anchor::Conversion(from.anchor,
                                                      to.anchor,
                                                      t);
  }
  return ephemeris_->placement_conversion(from, to, t);
}

template<template<typename Frame> typename Container>
Container<World> Renderer::RenderPlottingContainerInWorld(
    Instant const& time,
    Container<Navigation>::const_iterator const& begin,
    Container<Navigation>::const_iterator const& end,
    Position<World> const& sun_world_position,
    Rotation<Barycentric, AliceSun> const& planetarium_rotation,
    std::function<void(Container<World>&,
                       Instant const&,
                       DegreesOfFreedom<World> const&)> const& append,
    std::optional<WorldRegistration> const& registration) const {
  Container<World> result;

  //   Dinanzi a me non fuor cose create
  //   se non etterne, e io etterno duro.
  //   Lasciate ogne speranza, voi ch’intrate.
  //
  // This function does unnatural things.
  // - It identifies positions in the plotting frame with those of world using
  // the rigid transformation at the current time, instead of transforming each
  // position according to the transformation at its time.  This hides the fact
  // that we are considering an observer fixed in the plotting frame.
  // - Instead of applying the full rigid motion and consistently transforming
  // the velocities, or even just applying the orthogonal map, it simply
  // identifies the axes of `World` with those of the plotting frame. This is
  // because we are interested in the magnitude of the velocity (the speed) in
  // the plotting frame, as well as the coordinates (in frames with a physically
  // significant plane, the z coordinate becomes the out-of-plane velocity).
  // We apply the scaling at the time of the velocity, instead of the scaling at
  // `time` or no scaling, because we want speeds in current metres per second,
  // not in constant metres (at `time`) per second, nor in constant lunar
  // distances (masquerading as metres) per second.
  // The resulting `DegreesOfFreedom` should be seen as no more than a
  // convenient hack to send a plottable position together with a velocity in
  // the coordinates we want.  In fact, it needs an articial permutation to
  // avoid a violation of handedness.
  // TODO(phl): This will no longer be needed once we have support for
  // projections; instead of these convenient lies we can simply say that the
  // camera is fixed in the plotting frame and project there; additional data
  // can be gathered from the velocities in the plotting frame as needed and
  // sent directly to be shown in markers.
  Similarity<Navigation, World> const
      from_plotting_frame_to_world_at_current_time =
          registration.has_value()
              ? RegisteredPlottingToWorld(
                    time, *registration, planetarium_rotation)
              : PlottingToWorld(
                    time, sun_world_position, planetarium_rotation);
  for (auto const& [t, degrees_of_freedom] : Range(begin, end)) {
    DegreesOfFreedom<Navigation> const& navigation_degrees_of_freedom =
        degrees_of_freedom;
    ConformalMap<double, Navigation, World> const
        from_plotting_frame_to_world_at_t =
            PlottingToWorld(t, planetarium_rotation);
    DegreesOfFreedom<World> const world_degrees_of_freedom = {
        from_plotting_frame_to_world_at_current_time(
            navigation_degrees_of_freedom.position()),
        Permutation<Navigation, World>(
            Permutation<Navigation, World>::CoordinatePermutation::YXZ)(
            from_plotting_frame_to_world_at_t.scale() *
            navigation_degrees_of_freedom.velocity())};
    append(result, t, world_degrees_of_freedom);
  }
  return result;
}

}  // namespace internal
}  // namespace _renderer
}  // namespace ksp_plugin
}  // namespace principia
