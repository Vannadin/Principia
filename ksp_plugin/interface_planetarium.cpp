#include "ksp_plugin/interface.hpp"

#include <algorithm>
#include <limits>

#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "geometry/affine_map.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/perspective.hpp"
#include "geometry/r3_element.hpp"
#include "geometry/rotation.hpp"
#include "geometry/space_transformations.hpp"
#include "journal/method.hpp"
#include "journal/profiles.hpp"  // 🧙 For generated profiles.
#include "ksp_plugin/celestial.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin/planetarium.hpp"
#include "ksp_plugin/renderer.hpp"
#include "ksp_plugin/vessel.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"

namespace principia {
namespace interface {

using namespace principia::geometry::_affine_map;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_perspective;
using namespace principia::geometry::_r3_element;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_space_transformations;
using namespace principia::journal::_method;
using namespace principia::ksp_plugin::_celestial;
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_planetarium;
using namespace principia::ksp_plugin::_renderer;
using namespace principia::ksp_plugin::_vessel;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;

namespace {

// Where the anchor of a registered plot is written, or null if there is no
// registration.  The consumer of these plots picks the convention it decodes
// the anchor with from the reference it holds, not from what we did here, so a
// nonzero anchor must always be the registered one: without a registration we
// ask for no anchor at all and plot absolutely, as we did before anchoring
// existed.  The alternative — anchoring on the camera's absolute position while
// the consumer reassembles a displacement — translates the whole plot by the
// reference, which is a void-scale quantity in the tracking station.
R3Element<double>* AnchorFor(
    std::optional<Planetarium::Registration> const& registration,
    R3Element<double>& anchor_coordinates) {
  return registration.has_value() ? &anchor_coordinates : nullptr;
}

// The same for a celestial, whose trajectory is a continuous one and whose
// coordinates carry no anchor.
std::optional<Planetarium::Registration> CelestialRegistration(
    Plugin const& plugin,
    Celestial const& celestial) {
  auto const& trajectory = celestial.trajectory();
  Instant const t = plugin.CurrentTime();
  if (t < trajectory.t_min() || t > trajectory.t_max()) {
    return std::nullopt;
  }
  return Planetarium::Registration{
      .time = t,
      .position = celestial.current_position(t),
      .placement = {celestial.subsystem(), std::nullopt}};
}

// The registration point of a vessel's plot: where the vessel is now, which
// is where the scene draws it.  Absent if the vessel is not where we are
// looking: a trajectory may be empty, or lag the current time while it catches
// up, and plotting unregistered is better than evaluating out of the domain,
// which is fatal.
std::optional<Planetarium::Registration> VesselRegistration(
    Plugin const& plugin,
    Vessel const& vessel) {
  auto const& trajectory = vessel.trajectory();
  Instant const t = plugin.CurrentTime();
  if (trajectory.empty() || t < trajectory.t_min() || t > trajectory.t_max()) {
    return std::nullopt;
  }
  return Planetarium::Registration{.time = t,
                                   .position = trajectory.EvaluatePosition(t),
                                   .placement = vessel.placement()};
}

}  // namespace

Planetarium* __cdecl principia__PlanetariumCreate(
    Plugin const* const plugin,
    XYZ const sun_world_position,
    XYZ const xyz_opengl_camera_x_in_world,
    XYZ const xyz_opengl_camera_y_in_world,
    XYZ const xyz_opengl_camera_z_in_world,
    XYZ const xyz_camera_position_in_world,
    double const focal,
    double const field_of_view,
    double const inverse_scale_factor,
    double const angular_resolution,
    XYZ const scaled_space_origin) {
  journal::Method<journal::PlanetariumCreate> m({plugin,
                                                 sun_world_position,
                                                 xyz_opengl_camera_x_in_world,
                                                 xyz_opengl_camera_y_in_world,
                                                 xyz_opengl_camera_z_in_world,
                                                 xyz_camera_position_in_world,
                                                 focal,
                                                 field_of_view,
                                                 inverse_scale_factor,
                                                 angular_resolution,
                                                 scaled_space_origin});
  Renderer const& renderer = ABSL_DIE_IF_NULL(plugin)->renderer();

  Multivector<double, World, 1> const opengl_camera_x_in_world(
      FromXYZ(xyz_opengl_camera_x_in_world));
  Multivector<double, World, 1> const opengl_camera_y_in_world(
      FromXYZ(xyz_opengl_camera_y_in_world));
  Multivector<double, World, 2> const opengl_camera_z_in_world(
      FromXYZ(xyz_opengl_camera_z_in_world));
  // Note the minus sign for z below because our convention with respect to the
  // orientation of z is opposite that of OpenGL.
  Rotation<Camera, World> const camera_to_world_rotation(
      opengl_camera_x_in_world,
      opengl_camera_y_in_world,
      -opengl_camera_z_in_world);
  Position<World> const camera_position_in_world =
      FromXYZ<Position<World>>(xyz_camera_position_in_world);

  RigidTransformation<Camera, World> const camera_to_world_affine_map(
      Camera::origin,
      camera_position_in_world,
      camera_to_world_rotation.Forget<OrthogonalMap>());
  Similarity<World, Navigation> const
      world_to_plotting_affine_map =
          renderer.WorldToPlotting(plugin->CurrentTime(),
                                   FromXYZ<Position<World>>(sun_world_position),
                                   plugin->PlanetariumRotation());

  Planetarium::Parameters const parameters(
      /*sphere_radius_multiplier=*/1.0,
      angular_resolution * Radian,
      field_of_view * Radian);
  Perspective<Navigation, Camera> const perspective(
      world_to_plotting_affine_map *
          camera_to_world_affine_map.Forget<Similarity>(),
      focal * Metre);

  auto const plotting_to_scaled_space =
      Planetarium::MakePlottingToScaledSpaceConversion(
          world_to_plotting_affine_map,
          FromXYZ<Position<World>>(scaled_space_origin),
          inverse_scale_factor * (1 / Metre));
  auto const plotting_to_scaled_space_displacement =
      Planetarium::MakePlottingToScaledSpaceDisplacementConversion(
          world_to_plotting_affine_map,
          inverse_scale_factor * (1 / Metre));
  return m.Return(
      plugin->NewPlanetarium(
          parameters,
          perspective,
          plotting_to_scaled_space,
          plotting_to_scaled_space_displacement).release());
}

void __cdecl principia__PlanetariumDelete(
    Planetarium const** const planetarium) {
  journal::Method<journal::PlanetariumDelete> m({planetarium}, {planetarium});
  CHECK(planetarium != nullptr);
  TakeOwnership(planetarium);
  return m.Return();
}

// Computes the arrival ghost of the flight plan of the vessel with the given
// GUID: the vertex at which the celestial with the given index sits when the
// plan ends, using the analytic model of its subsystem where that is beyond
// the ephemeris.  The vertex and its anchor follow the conventions of an
// anchored plot.
void __cdecl principia__PlanetariumPlotArrivalGhost(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    char const* const vessel_guid,
    int const celestial_index,
    bool* const plotted,
    XYZ* const ghost,
    XYZ* const anchor) {
  journal::Method<journal::PlanetariumPlotArrivalGhost> m(
      {planetarium, plugin, vessel_guid, celestial_index},
      {plotted, ghost, anchor});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *plotted = false;
  R3Element<double> ghost_coordinates;
  R3Element<double> anchor_coordinates;

  Vessel const& vessel = *plugin->GetVessel(vessel_guid);
  CHECK(vessel.has_flight_plan()) << vessel_guid;
  auto const registration = VesselRegistration(*plugin, vessel);
  if (registration.has_value()) {
    auto const t = vessel.flight_plan().actual_final_time();
    auto const degrees_of_freedom =
        plugin->CelestialFutureDegreesOfFreedom(celestial_index, t);
    ScaledSpacePoint const point = planetarium->PlotPoint(
        t,
        degrees_of_freedom.position(),
        {plugin->GetCelestial(celestial_index).subsystem(), std::nullopt},
        anchor_coordinates,
        *registration);
    ghost_coordinates = R3Element<double>(point.x, point.y, point.z);
    *plotted = true;
  }
  *ghost = ToXYZ(ghost_coordinates);
  *anchor = ToXYZ(anchor_coordinates);
  return m.Return();
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendering of the segment with the given index in the flight plan of the
// vessel with the given GUID.
void __cdecl principia__PlanetariumPlotFlightPlanSegment(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    char const* const vessel_guid,
    int const index,
    double const* const t_max,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    int* const vertex_count,
    XYZ* const anchor,
    int* const seam_vertex_count) {
  journal::Method<journal::PlanetariumPlotFlightPlanSegment> m(
      {planetarium, plugin, vessel_guid, index, t_max, vertices, vertices_size},
      {vertex_count, anchor, seam_vertex_count});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  *seam_vertex_count = 0;
  R3Element<double> anchor_coordinates;

  Vessel const& vessel = *plugin->GetVessel(vessel_guid);
  CHECK(vessel.has_flight_plan()) << vessel_guid;
  auto const registration = VesselRegistration(*plugin, vessel);
  auto const segment = vessel.flight_plan().GetSegment(index);
  // TODO(egg): this is ugly; we should centralize rendering.
  // If this is a burn and we cannot render the beginning of the burn, we
  // render none of it, otherwise we try to render the Frenet trihedron at the
  // start and we fail.
  if (index % 2 == 0 ||
      segment->empty() ||
      segment->front().time >= plugin->renderer().GetPlottingFrame()->t_min()) {
    planetarium->PlotMethod4(
        *segment,
        segment->begin(),
        segment->end(),
        t_max == nullptr ? InfiniteFuture : FromGameTime(*plugin, *t_max),
        /*reverse=*/false,
        [vertices, vertex_count](ScaledSpacePoint const& vertex) {
          vertices[(*vertex_count)++] = vertex;
        },
        vertices_size,
        vessel.flight_plan().placement(),
        AnchorFor(registration, anchor_coordinates),
        registration,
        seam_vertex_count);
  }
  *anchor = ToXYZ(anchor_coordinates);
  return m.Return();
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendered prediction of the vessel with the given GUID.
void __cdecl principia__PlanetariumPlotPrediction(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    char const* const vessel_guid,
    double const* const t_max,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    int* const vertex_count,
    XYZ* const anchor,
    int* const seam_vertex_count) {
  journal::Method<journal::PlanetariumPlotPrediction> m(
      {planetarium, plugin, vessel_guid, t_max, vertices, vertices_size},
      {vertex_count, anchor, seam_vertex_count});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  *seam_vertex_count = 0;
  R3Element<double> anchor_coordinates;

  auto const vessel = plugin->GetVessel(vessel_guid);
  auto const registration = VesselRegistration(*plugin, *vessel);
  auto const prediction = vessel->prediction();
  planetarium->PlotMethod4(
      *prediction,
      prediction->begin(),
      prediction->end(),
      t_max == nullptr ? InfiniteFuture : FromGameTime(*plugin, *t_max),
      /*reverse=*/false,
      [vertices, vertex_count](ScaledSpacePoint const& vertex) {
        vertices[(*vertex_count)++] = vertex;
      },
      vertices_size,
      vessel->placement(),
      AnchorFor(registration, anchor_coordinates),
      registration,
      seam_vertex_count);
  *anchor = ToXYZ(anchor_coordinates);
  return m.Return();
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendered past trajectory of the vessel with the given GUID; the
// trajectory goes back `max_history_length` seconds before the present time (or
// to the earliest time available if the relevant `t_min` is more recent).
void __cdecl principia__PlanetariumPlotPsychohistory(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    char const* const vessel_guid,
    double const max_history_length,
    double const* const t_max,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    int* const vertex_count,
    XYZ* const anchor,
    int* const seam_vertex_count) {
  journal::Method<journal::PlanetariumPlotPsychohistory> m(
      {planetarium,
       plugin,
       vessel_guid,
       max_history_length,
       t_max,
       vertices,
       vertices_size},
      {vertex_count, anchor, seam_vertex_count});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  *seam_vertex_count = 0;
  R3Element<double> anchor_coordinates;

  // Do not plot the psychohistory when there is a target vessel as it is
  // misleading.
  if (plugin->renderer().HasTargetVessel()) {
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  } else {
    auto const vessel = plugin->GetVessel(vessel_guid);
    auto const registration = VesselRegistration(*plugin, *vessel);
    auto const& trajectory = vessel->trajectory();
    auto const& psychohistory = vessel->psychohistory();

    Instant const desired_first_time =
        plugin->CurrentTime() - max_history_length * Second;

    // Since we would want to plot starting from `desired_first_time`, ask the
    // reanimator to reconstruct the past.  That may take a while, during which
    // time the history will be shorter than desired.
    vessel->RequestReanimation(desired_first_time);

    planetarium->PlotMethod4(
        trajectory,
        trajectory.lower_bound(desired_first_time),
        psychohistory->end(),
        t_max == nullptr ? InfiniteFuture : FromGameTime(*plugin, *t_max),
        /*reverse=*/true,
        [vertices, vertex_count](ScaledSpacePoint const& vertex) {
          vertices[(*vertex_count)++] = vertex;
        },
        vertices_size,
        vessel->placement(),
        AnchorFor(registration, anchor_coordinates),
        registration,
        seam_vertex_count);
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  }
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendered past trajectory of the celestial with the given index; the
// trajectory goes back `max_history_length` seconds before the present time (or
// to the earliest time available if the relevant `t_min` is more recent).
void __cdecl principia__PlanetariumPlotCelestialPastTrajectory(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    int const celestial_index,
    double const max_history_length,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    double* const minimal_distance_from_camera,
    int* const vertex_count,
    XYZ* const anchor) {
  journal::Method<journal::PlanetariumPlotCelestialPastTrajectory> m(
      {planetarium,
       plugin,
       celestial_index,
       max_history_length,
       vertices,
       vertices_size},
      {minimal_distance_from_camera, vertex_count, anchor});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  R3Element<double> anchor_coordinates;

  // Do not plot the past when there is a target vessel as it is misleading.
  if (plugin->renderer().HasTargetVessel()) {
    *minimal_distance_from_camera = std::numeric_limits<double>::infinity();
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  } else {
    auto const& celestial = plugin->GetCelestial(celestial_index);
    auto const& celestial_trajectory = celestial.trajectory();
    auto const registration = CelestialRegistration(*plugin, celestial);
    Instant const desired_first_time =
        plugin->CurrentTime() - max_history_length * Second;

    // Since we would want to plot starting from `desired_first_time`, ask the
    // reanimator to reconstruct the past.  That may take a while, during which
    // time the history will be shorter than desired.
    plugin->RequestReanimation(desired_first_time);

    Instant const first_time =
        std::max(desired_first_time, celestial_trajectory.t_min());
    Length minimal_distance;
    planetarium->PlotMethod4(
        celestial_trajectory,
        first_time,
        /*last_time=*/plugin->CurrentTime(),
        /*reverse=*/true,
        [vertices, vertex_count](ScaledSpacePoint const& vertex) {
          vertices[(*vertex_count)++] = vertex;
        },
        vertices_size,
        &minimal_distance,
        {celestial.subsystem(), std::nullopt},
        AnchorFor(registration, anchor_coordinates),
        registration);
    *minimal_distance_from_camera = minimal_distance / Metre;
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  }
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendered future trajectory of the celestial with the given index; the
// trajectory goes as far as the furthest of the final time of the prediction or
// that of the flight plan.
void __cdecl principia__PlanetariumPlotCelestialFutureTrajectory(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    int const celestial_index,
    char const* const vessel_guid,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    double* const minimal_distance_from_camera,
    int* const vertex_count,
    XYZ* const anchor) {
  journal::Method<journal::PlanetariumPlotCelestialFutureTrajectory> m(
      {planetarium,
       plugin,
       celestial_index,
       vessel_guid,
       vertices,
       vertices_size},
      {minimal_distance_from_camera, vertex_count, anchor});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  R3Element<double> anchor_coordinates;

  // Do not plot the past when there is a target vessel as it is misleading.
  // TODO(egg): This is the future, not the past!
  if (plugin->renderer().HasTargetVessel()) {
    *minimal_distance_from_camera = std::numeric_limits<double>::infinity();
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  } else {
    auto const& vessel = *plugin->GetVessel(vessel_guid);
    auto const& celestial = plugin->GetCelestial(celestial_index);
    auto const& celestial_trajectory = celestial.trajectory();
    Instant const prediction_final_time = vessel.prediction()->t_max();
    // A void flight plan may end beyond the ephemeris, where the celestial has
    // no trajectory to plot.
    Instant const final_time = std::min(
        vessel.has_flight_plan()
            ? std::max(GetFlightPlan(*plugin, vessel_guid).actual_final_time(),
                       prediction_final_time)
            : prediction_final_time,
        celestial_trajectory.t_max());
    auto const registration = CelestialRegistration(*plugin, celestial);
    // No need to request reanimation here because the current time of the
    // plugin is necessarily covered.
    Length minimal_distance;
    planetarium->PlotMethod4(
        celestial_trajectory,
        /*first_time=*/plugin->CurrentTime(),
        /*last_time=*/final_time,
        /*reverse=*/false,
        [vertices, vertex_count](ScaledSpacePoint const& vertex) {
          vertices[(*vertex_count)++] = vertex;
        },
        vertices_size,
        &minimal_distance,
        {celestial.subsystem(), std::nullopt},
        AnchorFor(registration, anchor_coordinates),
        registration);
    *minimal_distance_from_camera = minimal_distance / Metre;
    *anchor = ToXYZ(anchor_coordinates);
    return m.Return();
  }
}

// Fills the array of size `vertices_size` at `vertices` with vertices for the
// rendered prediction of the vessel with the given GUID.
void __cdecl principia__PlanetariumPlotEquipotential(
    Planetarium const* const planetarium,
    Plugin const* const plugin,
    int const index,
    ScaledSpacePoint* const vertices,
    int const vertices_size,
    int* const vertex_count,
    XYZ* const anchor) {
  journal::Method<journal::PlanetariumPlotEquipotential> m(
      {planetarium, plugin, index, vertices, vertices_size},
      {vertex_count, anchor});
  CHECK(plugin != nullptr);
  CHECK(planetarium != nullptr);
  *vertex_count = 0;
  R3Element<double> anchor_coordinates;

  auto const& equipotentials =
      *ABSL_DIE_IF_NULL(plugin->geometric_potential_plotter().equipotentials());
  CHECK_GE(index, 0);
  CHECK_LT(index, equipotentials.lines.size());
  DiscreteTrajectory<Navigation> const& equipotential =
      equipotentials.lines[index];

  planetarium->PlotMethod4(
      equipotential,
      equipotential.front().time,
      equipotential.back().time,
      /*reverse=*/false,
      [vertices, vertex_count](ScaledSpacePoint const& vertex) {
        vertices[(*vertex_count)++] = vertex;
      },
      vertices_size,
      /*minimal_distance=*/nullptr,
      Ephemeris<Barycentric>::SubsystemPlacement::Stock(),
      &anchor_coordinates);
  *anchor = ToXYZ(anchor_coordinates);
  return m.Return();
}

}  // namespace interface
}  // namespace principia
