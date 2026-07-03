#include "ksp_plugin/interface.hpp"

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
