#pragma once

#include <optional>
#include <vector>

#include "base/algebra.hpp"
#include "base/not_null.hpp"
#include "geometry/instant.hpp"
#include "geometry/perspective.hpp"
#include "geometry/r3_element.hpp"
#include "geometry/rp2_point.hpp"
#include "geometry/space.hpp"
#include "geometry/space_transformations.hpp"
#include "geometry/sphere.hpp"
#include "ksp_plugin/frames.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "physics/trajectory.hpp"
#include "quantities/quantities.hpp"

namespace principia {
namespace ksp_plugin {
namespace _planetarium {
namespace internal {

using namespace principia::base::_algebra;
using namespace principia::base::_not_null;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_perspective;
using namespace principia::geometry::_r3_element;
using namespace principia::geometry::_rp2_point;
using namespace principia::geometry::_space;
using namespace principia::geometry::_space_transformations;
using namespace principia::geometry::_sphere;
using namespace principia::ksp_plugin::_frames;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_trajectory;
using namespace principia::quantities::_quantities;

// Corresponds to a UnityEngine.Vector3 representing a position in KSP’s
// ScaledSpace.
extern "C" struct ScaledSpacePoint {
  static inline ScaledSpacePoint FromCoordinates(
      R3Element<double> const& coordinates);

  float x;
  float y;
  float z;
};

static_assert(std::is_pod<ScaledSpacePoint>::value,
              "NavigationFrameParameters is used for interfacing");

// A planetarium is an ephemeris together with a perspective.  In this setting
// it is possible to draw trajectories in the projective plane.
class Planetarium {
 public:
  class Parameters final {
   public:
    // `sphere_radius_multiplier` defines the "dark area" around a celestial
    // where we don't draw trajectories.  `angular_resolution` defines the limit
    // beyond which spheres don't participate in hiding.  `field_of_view`
    // is the half-angle of a cone outside of which not plotting takes place.
    explicit Parameters(double sphere_radius_multiplier,
                        Angle const& angular_resolution,
                        Angle const& field_of_view);

   private:
    double const sphere_radius_multiplier_;
    Angle const angular_resolution_;
    double const sin²_angular_resolution_;
    double const tan_angular_resolution_;
    double const tan_field_of_view_;
    friend class Planetarium;
  };

  using PlottingToScaledSpaceConversion =
      std::function<R3Element<double>(Instant const&,
                                      Position<Navigation> const&)>;

  // The linear part of the above, for displacements.  An anchored plot is
  // emitted through this map so that no absolute position at the magnitude of
  // an interstellar distance is ever formed per vertex.
  using PlottingToScaledSpaceDisplacementConversion =
      std::function<R3Element<double>(Displacement<Navigation> const&)>;

  // The point against which an anchored plot is registered: a position both
  // the plotter and its consumer can name, at a time at which both know it.
  // It need not lie on the plotted trajectory — it is the plotted vessel or
  // body at the present, which is where the scene draws that object — and
  // the plot is reported as displacements from it, so that neither side ever
  // expresses it in scaled space across the distance to the plotting frame's
  // origin.  `position` is in the same placement as the plotted trajectory.
  struct Registration final {
    Instant time;
    Position<Barycentric> position;
  };

  // The conversion used in production.  It never forms the world position of a
  // plotted point: at interstellar distances that position rounds to the ULP of
  // a star's distance, independently for each point, which is a jitter of the
  // shape of the plot and not a displacement of it.
  static PlottingToScaledSpaceConversion MakePlottingToScaledSpaceConversion(
      Similarity<World, Navigation> const& world_to_plotting,
      Position<World> const& scaled_space_origin,
      Inverse<Length> const& inverse_scale_factor);

  // The linear part of the conversion above.
  static PlottingToScaledSpaceDisplacementConversion
  MakePlottingToScaledSpaceDisplacementConversion(
      Similarity<World, Navigation> const& world_to_plotting,
      Inverse<Length> const& inverse_scale_factor);

  // TODO(phl): All this Navigation is weird.  Should it be named Plotting?
  // In particular Navigation vs. NavigationFrame is a mess.
  // `plotting_to_scaled_space_displacement` must be the linear part of
  // `plotting_to_scaled_space`; it may only be omitted if no anchored plot is
  // ever requested.
  Planetarium(Parameters const& parameters,
              Perspective<Navigation, Camera> perspective,
              not_null<Ephemeris<Barycentric> const*> ephemeris,
              not_null<PlottingFrame const*> plotting_frame,
              PlottingToScaledSpaceConversion plotting_to_scaled_space,
              PlottingToScaledSpaceDisplacementConversion
                  plotting_to_scaled_space_displacement = nullptr);

  // NOTE: unlike `PlotMethod4`, methods 0–3 feed the trajectory to the
  // plotting frame without any subsystem conversion: they must only be given
  // trajectories represented in the plotting frame's subsystem.  They have no
  // production callers; if one is rewired into the interface, convert as
  // `PlotMethod4` does.
  // A no-op method that just returns all the points in the trajectory defined
  // by `begin` and `end`.
  RP2Lines<Length, Camera> PlotMethod0(
      DiscreteTrajectory<Barycentric> const& trajectory,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end,
      Instant const& now,
      bool reverse) const;

  // A method that coalesces segments until they are larger than the angular
  // resolution.
  RP2Lines<Length, Camera> PlotMethod1(
      DiscreteTrajectory<Barycentric> const& trajectory,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end,
      Instant const& now,
      bool reverse) const;

  // A method that plots the cubic Hermite spline interpolating the trajectory,
  // using an adaptive step size to keep the error between the straight segments
  // and the actual spline below and close to the angular resolution.
  RP2Lines<Length, Camera> PlotMethod2(
      Trajectory<Barycentric> const& trajectory,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end,
      Instant const& now,
      bool reverse) const;

  // The same method, operating on the `Trajectory` interface.
  RP2Lines<Length, Camera> PlotMethod2(
      Trajectory<Barycentric> const& trajectory,
      Instant const& first_time,
      Instant const& last_time,
      Instant const& now,
      bool reverse,
      Length* minimal_distance = nullptr) const;

  // A method similar to PlotMethod2, but which produces a three-dimensional
  // trajectory in scaled space instead of projecting and hiding.  It uses the
  // apparent angle of the sagitta as the metric to analyse curvature.
  void PlotMethod3(
      Trajectory<Barycentric> const& trajectory,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end,
      Instant const& t_max,
      bool reverse,
      std::function<void(ScaledSpacePoint const&)> const& add_point,
      int max_points) const;

  // The same method, operating on the `Trajectory` interface for any frame that
  // can be converted to `Navigation`.
  template<typename Frame>
  void PlotMethod3(
      Trajectory<Frame> const& trajectory,
      Instant const& first_time,
      Instant const& last_time,
      bool reverse,
      std::function<void(ScaledSpacePoint const&)> const& add_point,
      int max_points,
      Length* minimal_distance = nullptr) const;

  // A method similar to PlotMethod4, but which uses the RMS of the apparent
  // distance between the trajectory and line segments.  `placement` gives the
  // subsystem and anchor relative to which the positions of `trajectory` are
  // represented.  If `anchor_out` is not null and the ephemeris has multiple
  // subsystems, the vertices passed to `add_point` are expressed relative to
  // an anchor at the camera position, whose scaled-space coordinates are
  // returned in `anchor_out`; the vertices then quantize at the float ULP of
  // their distance from the camera instead of that of their distance to the
  // scaled-space origin, and the caller must translate the drawn mesh by the
  // anchor.  With a single subsystem the anchor is zero and the vertices
  // reproduce the absolute rendering bit for bit.
  void PlotMethod4(
      Trajectory<Barycentric> const& trajectory,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end,
      Instant const& t_max,
      bool reverse,
      std::function<void(ScaledSpacePoint const&)> const& add_point,
      int max_points,
      Ephemeris<Barycentric>::SubsystemPlacement const& placement =
          Ephemeris<Barycentric>::SubsystemPlacement::Stock(),
      R3Element<double>* anchor_out = nullptr,
      std::optional<Registration> const& registration = std::nullopt) const;

  // The same method, operating on the `Trajectory` interface for any frame that
  // can be converted to `Navigation`.  `placement` and `registration` are only
  // meaningful when `Frame` is `Barycentric`.
  template<typename Frame>
  void PlotMethod4(
      Trajectory<Frame> const& trajectory,
      Instant const& first_time,
      Instant const& last_time,
      bool reverse,
      std::function<void(ScaledSpacePoint const&)> const& add_point,
      int max_points,
      Length* minimal_distance = nullptr,
      Ephemeris<Barycentric>::SubsystemPlacement const& placement =
          Ephemeris<Barycentric>::SubsystemPlacement::Stock(),
      R3Element<double>* anchor_out = nullptr,
      std::optional<Registration> const& registration = std::nullopt) const;

 private:
  // The displacement-native variant of `PlotMethod4`, used for an anchored
  // plot.  Everything at the magnitude of the distance to the plotting
  // frame's origin is folded into per-plot constants, whose rounding — a few
  // ULPs of that distance — displaces the plot as a whole without deforming
  // it.  For an inertial plotting frame no such magnitude then survives per
  // vertex, so the plot does not quantize at the double ULP of that distance;
  // for a rotating one the sweep of the reference still rounds per vertex at
  // that ULP, which is not a regression (the legacy path rounds identically)
  // and is dwarfed by the arc the frame's rotation sweeps at that distance.
  void PlotMethod4Anchored(
      Trajectory<Barycentric> const& trajectory,
      Instant const& first_time,
      Instant const& last_time,
      bool reverse,
      std::function<void(ScaledSpacePoint const&)> const& add_point,
      int max_points,
      Length* minimal_distance,
      Ephemeris<Barycentric>::SubsystemPlacement const& placement,
      R3Element<double>& anchor_out,
      Registration const& registration) const;

  // Computes the coordinates of the spheres that represent the `ephemeris_`
  // bodies.  These coordinates are in the `plotting_frame_` at time `now`.
  std::vector<Sphere<Navigation>> ComputePlottableSpheres(
      Instant const& now) const;

  // Computes the segments of the trajectory defined by `begin` and `end` that
  // are not hidden by the `plottable_spheres`.
  Segments<Navigation> ComputePlottableSegments(
      const std::vector<Sphere<Navigation>>& plottable_spheres,
      DiscreteTrajectory<Barycentric>::iterator begin,
      DiscreteTrajectory<Barycentric>::iterator end) const;

  // Computes the proper motion (in the astronomical sense) of the given point
  // and velocity seen from a frame centered at the origin of `Camera`.  This
  // is a uniform rotation on a great circle.
  AngularVelocity<Navigation> ProperMotion(
      DegreesOfFreedom<Navigation> const& degrees_of_freedom) const;

  Parameters const parameters_;
  Perspective<Navigation, Camera> const perspective_;
  not_null<Ephemeris<Barycentric> const*> const ephemeris_;
  not_null<PlottingFrame const*> const plotting_frame_;
  PlottingToScaledSpaceConversion plotting_to_scaled_space_;
  PlottingToScaledSpaceDisplacementConversion
      plotting_to_scaled_space_displacement_;
};

inline ScaledSpacePoint ScaledSpacePoint::FromCoordinates(
    R3Element<double> const& coordinates) {
  return ScaledSpacePoint{.x = static_cast<float>(coordinates.x),
                          .y = static_cast<float>(coordinates.y),
                          .z = static_cast<float>(coordinates.z)};
}

}  // namespace internal

using internal::Planetarium;
using internal::ScaledSpacePoint;

}  // namespace _planetarium
}  // namespace ksp_plugin
}  // namespace principia

#include "ksp_plugin/planetarium_body.hpp"
