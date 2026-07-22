#include "ksp_plugin/interface.hpp"

#include <optional>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "geometry/grassmann.hpp"
#include "journal/method.hpp"
#include "journal/profiles.hpp"  // 🧙 For generated profiles.
#include "ksp_plugin/frames.hpp"
#include "quantities/constants.hpp"
#include "quantities/si.hpp"

namespace principia {
namespace interface {

using namespace principia::geometry::_grassmann;
using namespace principia::journal::_method;
using namespace principia::ksp_plugin::_frames;
using namespace principia::quantities::_constants;
using namespace principia::quantities::_si;

XYZ __cdecl principia__VesselBinormal(Plugin const* const plugin,
                                      char const* const vessel_guid) {
  journal::Method<journal::VesselBinormal> m({plugin, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToXYZ(plugin->VesselBinormal(vessel_guid)));
}

// Clears the on-rails burn on the pile up containing the given vessel.
// `plugin` must not be null.  No transfer of ownership.
void __cdecl principia__VesselClearOnRailsBurn(Plugin const* const plugin,
                                               char const* const vessel_guid) {
  journal::Method<journal::VesselClearOnRailsBurn> m({plugin, vessel_guid});
  CHECK(plugin != nullptr);
  plugin->ClearVesselOnRailsBurn(vessel_guid);
  return m.Return();
}

// Calls `plugin->VesselFromParent` with the arguments given.
// `plugin` must not be null.  No transfer of ownership.
QP __cdecl principia__VesselFromParent(Plugin const* const plugin,
                                       int const parent_index,
                                       char const* const vessel_guid) {
  journal::Method<journal::VesselFromParent> m(
      {plugin, parent_index, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToQP(plugin->VesselFromParent(parent_index, vessel_guid)));
}

OrbitAnalysis* __cdecl principia__VesselGetAnalysis(
    Plugin* const plugin,
    char const* const vessel_guid,
    int const* const revolutions_per_cycle,
    int const* const days_per_cycle,
    int const ground_track_revolution) {
  journal::Method<journal::VesselGetAnalysis> m({plugin,
                                                 vessel_guid,
                                                 revolutions_per_cycle,
                                                 days_per_cycle,
                                                 ground_track_revolution});
  CHECK(plugin != nullptr);
  Vessel& vessel = *plugin->GetVessel(vessel_guid);
  vessel.RefreshOrbitAnalysis();
  not_null<OrbitAnalysis*> const analysis =
      NewOrbitAnalysis(vessel.orbit_analysis(),
                       *plugin,
                       revolutions_per_cycle,
                       days_per_cycle,
                       ground_track_revolution);
  analysis->progress_of_next_analysis = vessel.progress_of_orbit_analysis();
  return m.Return(analysis);
}

// Returns the placement under which the vessel's trajectory is represented:
// the subsystem relative to whose local origin its coordinates are expressed,
// and, if the vessel coasts anchored in the inter-subsystem void, the sector
// cell and intra-cell offset of its anchor (zero when unanchored).  The cell
// indices are integers exactly representable as doubles.
// Returns the void-navigation readouts of the vessel: whether it coasts
// force-free outside every star's damped far field, the nearest star, and the
// vessel's state relative to the target and reference celestials (indices < 0
// when not selected), all at `Barycentric` precision.
void __cdecl principia__VesselGetNavigationState(
    Plugin const* const plugin,
    char const* const vessel_guid,
    int const target_celestial_index,
    int const reference_celestial_index,
    bool* const in_void,
    int* const nearest_star_index,
    double* const nearest_star_distance,
    XYZ* const position_wrt_target,
    XYZ* const velocity_wrt_target,
    XYZ* const velocity_wrt_reference) {
  journal::Method<journal::VesselGetNavigationState> m(
      {plugin,
       vessel_guid,
       target_celestial_index,
       reference_celestial_index},
      {in_void,
       nearest_star_index,
       nearest_star_distance,
       position_wrt_target,
       velocity_wrt_target,
       velocity_wrt_reference});
  CHECK(plugin != nullptr);
  Plugin::NavigationState const state = plugin->VesselNavigationState(
      vessel_guid,
      target_celestial_index < 0
          ? std::nullopt
          : std::make_optional(target_celestial_index),
      reference_celestial_index < 0
          ? std::nullopt
          : std::make_optional(reference_celestial_index));
  *in_void = state.in_void;
  *nearest_star_index = state.nearest_star_index;
  *nearest_star_distance = state.nearest_star_distance / Metre;
  *position_wrt_target = ToXYZ(state.position_wrt_target.coordinates() / Metre);
  *velocity_wrt_target =
      ToXYZ(state.velocity_wrt_target.coordinates() / (Metre / Second));
  *velocity_wrt_reference =
      ToXYZ(state.velocity_wrt_reference.coordinates() / (Metre / Second));
  return m.Return();
}

void __cdecl principia__VesselGetPlacement(Plugin const* const plugin,
                                           char const* const vessel_guid,
                                           int* const subsystem,
                                           bool* const has_anchor,
                                           XYZ* const anchor_cell,
                                           XYZ* const anchor_local) {
  journal::Method<journal::VesselGetPlacement> m(
      {plugin, vessel_guid},
      {subsystem, has_anchor, anchor_cell, anchor_local});
  CHECK(plugin != nullptr);
  Vessel const& vessel = *plugin->GetVessel(vessel_guid);
  *subsystem = vessel.subsystem();
  auto const& anchor = vessel.anchor();
  *has_anchor = anchor.has_value();
  if (anchor.has_value()) {
    auto const& cell = anchor->offset.cell;
    *anchor_cell = XYZ{static_cast<double>(cell.x),
                       static_cast<double>(cell.y),
                       static_cast<double>(cell.z)};
    *anchor_local = ToXYZ(anchor->offset.local.coordinates() / Metre);
  } else {
    *anchor_cell = XYZ{0, 0, 0};
    *anchor_local = XYZ{0, 0, 0};
  }
  return m.Return();
}

// Makes the vessel inherit its parent vessel's placement if it is fresh;
// a no-op otherwise.  Must be called before the vessel's parts are inserted.
void __cdecl principia__VesselInheritPlacement(
    Plugin const* const plugin,
    char const* const vessel_guid,
    char const* const parent_vessel_guid) {
  journal::Method<journal::VesselInheritPlacement> m(
      {plugin, vessel_guid, parent_vessel_guid});
  CHECK(plugin != nullptr);
  plugin->InheritVesselPlacement(vessel_guid, parent_vessel_guid);
  return m.Return();
}

// Returns the vessel's present `World` degrees of freedom through the
// placement conversions, so that the caller can place an on-rails vessel in
// the scene without routing through the void-scale absolutes that quantize
// the stock orbit-driven placement at metres per ULP.
QP __cdecl principia__VesselGetWorldDegreesOfFreedom(
    Plugin const* const plugin,
    char const* const vessel_guid,
    Origin const origin) {
  journal::Method<journal::VesselGetWorldDegreesOfFreedom> m(
      {plugin, vessel_guid, origin});
  CHECK(plugin != nullptr);
  return m.Return(ToQP(plugin->VesselWorldDegreesOfFreedom(
      vessel_guid,
      origin.reference_part_id,
      plugin->BarycentricToWorld(
          origin.reference_part_is_unmoving,
          origin.reference_part_id,
          origin.reference_part_is_at_origin
              ? std::nullopt
              : std::make_optional(FromXYZ<Position<World>>(
                    origin.main_body_centre_in_world))),
      plugin->CurrentTime())));
}

AdaptiveStepParameters __cdecl
principia__VesselGetPredictionAdaptiveStepParameters(
    Plugin const* const plugin,
    char const* const vessel_guid) {
  journal::Method<journal::VesselGetPredictionAdaptiveStepParameters> m(
      {plugin, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToAdaptiveStepParameters(
      plugin->GetVessel(vessel_guid)->prediction_adaptive_step_parameters()));
}

XYZ __cdecl principia__VesselNormal(Plugin const* const plugin,
                                    char const* const vessel_guid) {
  journal::Method<journal::VesselNormal> m({plugin, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToXYZ(plugin->VesselNormal(vessel_guid)));
}

void __cdecl principia__VesselRequestAnalysis(Plugin* const plugin,
                                              char const* const vessel_guid,
                                              double const mission_duration) {
  journal::Method<journal::VesselRequestAnalysis> m(
      {plugin, vessel_guid, mission_duration});
  CHECK(plugin != nullptr);
  Vessel& vessel = *plugin->GetVessel(vessel_guid);
  plugin->ClearOrbitAnalysersOfVesselsOtherThan(vessel);
  vessel.RequestOrbitAnalysis(mission_duration * Second);
  return m.Return();
}

// Sets the on-rails burn on the pile up containing the given vessel, to be
// applied by the next catch-up; see `Plugin::SetVesselOnRailsBurn`.
// `plugin` must not be null.  No transfer of ownership.
void __cdecl principia__VesselSetOnRailsBurn(
    Plugin const* const plugin,
    char const* const vessel_guid,
    double const thrust_in_kilonewtons,
    double const specific_impulse_in_seconds_g0,
    double const initial_mass_in_tonnes,
    XYZ const direction,
    double const max_duration) {
  journal::Method<journal::VesselSetOnRailsBurn> m(
      {plugin,
       vessel_guid,
       thrust_in_kilonewtons,
       specific_impulse_in_seconds_g0,
       initial_mass_in_tonnes,
       direction,
       max_duration});
  CHECK(plugin != nullptr);
  plugin->SetVesselOnRailsBurn(
      vessel_guid,
      thrust_in_kilonewtons * Kilo(Newton),
      specific_impulse_in_seconds_g0 * Second * StandardGravity,
      initial_mass_in_tonnes * Tonne,
      Vector<double, World>(FromXYZ(direction)),
      max_duration * Second);
  return m.Return();
}

void __cdecl principia__VesselSetPredictionAdaptiveStepParameters(
    Plugin const* const plugin,
    char const* const vessel_guid,
    AdaptiveStepParameters const adaptive_step_parameters) {
  journal::Method<journal::VesselSetPredictionAdaptiveStepParameters> m(
      {plugin, vessel_guid, adaptive_step_parameters});
  CHECK(plugin != nullptr);
  plugin->SetPredictionAdaptiveStepParameters(
      vessel_guid, FromAdaptiveStepParameters(adaptive_step_parameters));
  return m.Return();
}

XYZ __cdecl principia__VesselTangent(Plugin const* const plugin,
                                     char const* const vessel_guid) {
  journal::Method<journal::VesselTangent> m({plugin, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToXYZ(plugin->VesselTangent(vessel_guid)));
}

XYZ __cdecl principia__VesselVelocity(Plugin const* const plugin,
                                      char const* const vessel_guid) {
  journal::Method<journal::VesselVelocity> m({plugin, vessel_guid});
  CHECK(plugin != nullptr);
  return m.Return(ToXYZ(plugin->VesselVelocity(vessel_guid)));
}

}  // namespace interface
}  // namespace principia
