#include "ksp_plugin/planetarium.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "base/algebra.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/sign.hpp"
#include "numerics/elementary_functions.hpp"
#include "numerics/hermite3.hpp"
#include "numerics/quadrature.hpp"
#include "physics/massive_body.hpp"
#include "physics/similar_motion.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/si.hpp"

namespace principia {
namespace ksp_plugin {
namespace _planetarium {
namespace internal {

using namespace principia::base::_algebra;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_sign;
using namespace principia::numerics::_elementary_functions;
using namespace principia::numerics::_hermite3;
using namespace principia::numerics::_quadrature;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_similar_motion;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_si;

namespace {
constexpr int max_plot_method_2_steps = 10'000;
}  // namespace

Planetarium::Parameters::Parameters(double const sphere_radius_multiplier,
                                    Angle const& angular_resolution,
                                    Angle const& field_of_view)
    : sphere_radius_multiplier_(sphere_radius_multiplier),
      angular_resolution_(angular_resolution),
      sin²_angular_resolution_(Pow<2>(Sin(angular_resolution))),
      tan_angular_resolution_(Tan(angular_resolution)),
      tan_field_of_view_(Tan(field_of_view)) {}

Planetarium::Planetarium(
    Parameters const& parameters,
    Perspective<Navigation, Camera> perspective,
    not_null<Ephemeris<Barycentric> const*> const ephemeris,
    not_null<PlottingFrame const*> const plotting_frame,
    PlottingToScaledSpaceConversion plotting_to_scaled_space,
    PlottingToScaledSpaceDisplacementConversion
        plotting_to_scaled_space_displacement)
    : parameters_(parameters),
      perspective_(std::move(perspective)),
      ephemeris_(ephemeris),
      plotting_frame_(plotting_frame),
      plotting_to_scaled_space_(std::move(plotting_to_scaled_space)),
      plotting_to_scaled_space_displacement_(
          std::move(plotting_to_scaled_space_displacement)) {}

Planetarium::Planetarium(
    Parameters const& parameters,
    Perspective<Navigation, Camera> perspective,
    not_null<Ephemeris<Barycentric> const*> const ephemeris,
    std::unique_ptr<PlottingFrame const> plotting_frame,
    PlottingToScaledSpaceConversion plotting_to_scaled_space,
    PlottingToScaledSpaceDisplacementConversion
        plotting_to_scaled_space_displacement)
    : parameters_(parameters),
      perspective_(std::move(perspective)),
      ephemeris_(ephemeris),
      owned_plotting_frame_(std::move(plotting_frame)),
      plotting_frame_(owned_plotting_frame_.get()),
      plotting_to_scaled_space_(std::move(plotting_to_scaled_space)),
      plotting_to_scaled_space_displacement_(
          std::move(plotting_to_scaled_space_displacement)) {}

Planetarium::PlottingToScaledSpaceConversion
Planetarium::MakePlottingToScaledSpaceConversion(
    Similarity<World, Navigation> const& world_to_plotting,
    Position<World> const& scaled_space_origin,
    Inverse<Length> const& inverse_scale_factor) {
  auto const plotting_to_world = world_to_plotting.Inverse();
  return [linear_map = plotting_to_world.linear_map(),
          origin_offset =
              plotting_to_world(Navigation::origin) - scaled_space_origin,
          inverse_scale_factor](Instant const&,
                                Position<Navigation> const& plotted_point) {
    return ((origin_offset +
             linear_map(plotted_point - Navigation::origin)) *
            inverse_scale_factor).coordinates();
  };
}

Planetarium::PlottingToScaledSpaceDisplacementConversion
Planetarium::MakePlottingToScaledSpaceDisplacementConversion(
    Similarity<World, Navigation> const& world_to_plotting,
    Inverse<Length> const& inverse_scale_factor) {
  return [linear_map = world_to_plotting.Inverse().linear_map(),
          inverse_scale_factor](Displacement<Navigation> const& displacement) {
    return (linear_map(displacement) * inverse_scale_factor).coordinates();
  };
}

RP2Lines<Length, Camera> Planetarium::PlotMethod0(
    DiscreteTrajectory<Barycentric> const& trajectory,
    DiscreteTrajectory<Barycentric>::iterator const /*begin*/,
    DiscreteTrajectory<Barycentric>::iterator const /*end*/,
    Instant const& now,
    bool const /*reverse*/) const {
  auto const plottable_begin = trajectory.lower_bound(plotting_frame_->t_min());
  auto const plottable_end = trajectory.lower_bound(plotting_frame_->t_max());
  auto const plottable_spheres = ComputePlottableSpheres(now);
  auto const plottable_segments = ComputePlottableSegments(plottable_spheres,
                                                           plottable_begin,
                                                           plottable_end);

  auto const field_of_view_radius² =
      perspective_.focal() * perspective_.focal() *
      parameters_.tan_field_of_view_ * parameters_.tan_field_of_view_;
  std::optional<Position<Navigation>> previous_position;
  RP2Lines<Length, Camera> rp2_lines;
  for (auto const& plottable_segment : plottable_segments) {
    // Apply the projection to the current plottable segment.
    auto const rp2_first = perspective_(plottable_segment.first);
    auto const rp2_second = perspective_(plottable_segment.second);

    // If the segment is entirely outside the field of view, ignore it.
    Length const x1 = rp2_first.x();
    Length const y1 = rp2_first.y();
    Length const x2 = rp2_second.x();
    Length const y2 = rp2_second.y();
    if (x1 * x1 + y1 * y1 > field_of_view_radius² &&
        x2 * x2 + y2 * y2 > field_of_view_radius²) {
      continue;
    }

    // Create a new ℝP² line when two segments are not consecutive.  Don't
    // compare ℝP² points for equality, that's expensive.
    bool const are_consecutive =
        previous_position == plottable_segment.first;
    previous_position = plottable_segment.second;

    if (are_consecutive) {
      rp2_lines.back().push_back(rp2_second);
    } else {
      RP2Line<Length, Camera> const rp2_line = {rp2_first, rp2_second};
      rp2_lines.push_back(rp2_line);
    }
  }
  return rp2_lines;
}

RP2Lines<Length, Camera> Planetarium::PlotMethod1(
    DiscreteTrajectory<Barycentric> const& trajectory,
    DiscreteTrajectory<Barycentric>::iterator const begin,
    DiscreteTrajectory<Barycentric>::iterator const end,
    Instant const& now,
    bool const reverse) const {
  Length const focal_plane_tolerance =
      perspective_.focal() * parameters_.tan_angular_resolution_;
  auto const focal_plane_tolerance² =
      focal_plane_tolerance * focal_plane_tolerance;

  auto const rp2_lines = PlotMethod0(trajectory, begin, end, now, reverse);

  RP2Lines<Length, Camera> new_rp2_lines;
  for (auto const& rp2_line : rp2_lines) {
    RP2Line<Length, Camera> new_rp2_line;
    std::optional<RP2Point<Length, Camera>> start_rp2_point;
    for (int i = 0; i < rp2_line.size(); ++i) {
      RP2Point<Length, Camera> const& rp2_point = rp2_line[i];
      if (i == 0) {
        new_rp2_line.push_back(rp2_point);
        start_rp2_point = rp2_point;
      } else if (Pow<2>(rp2_point.x() - start_rp2_point->x()) +
                 Pow<2>(rp2_point.y() - start_rp2_point->y()) >
                     focal_plane_tolerance²) {
        // TODO(phl): This creates a segment if the tolerance is exceeded.  It
        // should probably create a segment that stays just below the tolerance.
        new_rp2_line.push_back(rp2_point);
        start_rp2_point = rp2_point;
      } else if (i == rp2_line.size() - 1) {
        new_rp2_line.push_back(rp2_point);
      }
    }
    new_rp2_lines.push_back(std::move(new_rp2_line));
  }
  return new_rp2_lines;
}

RP2Lines<Length, Camera> Planetarium::PlotMethod2(
    Trajectory<Barycentric> const& trajectory,
    DiscreteTrajectory<Barycentric>::iterator const begin,
    DiscreteTrajectory<Barycentric>::iterator const end,
    Instant const& now,
    bool const reverse) const {
  if (begin == end) {
    return {};
  }
  auto last = end;
  --last;
  auto const begin_time = std::max(begin->time, plotting_frame_->t_min());
  auto const last_time = std::min(last->time, plotting_frame_->t_max());
  return PlotMethod2(trajectory, begin_time, last_time, now, reverse);
}

RP2Lines<Length, Camera> Planetarium::PlotMethod2(
    Trajectory<Barycentric> const& trajectory,
    Instant const& first_time,
    Instant const& last_time,
    Instant const& now,
    bool const reverse,
    Length* const minimal_distance) const {
  RP2Lines<Length, Camera> lines;
  auto const plottable_spheres = ComputePlottableSpheres(now);
  double const tan²_angular_resolution =
      Pow<2>(parameters_.tan_angular_resolution_);
  auto const final_time = reverse ? first_time : last_time;
  auto previous_time = reverse ? last_time : first_time;

  if (minimal_distance != nullptr) {
    *minimal_distance = Infinity<Length>;
  }

  Sign const direction = reverse ? Sign::Negative() : Sign::Positive();
  if (direction * (final_time - previous_time) <= Time{}) {
    return lines;
  }
  SimilarMotion<Barycentric, Navigation> to_plotting_frame_at_t =
      plotting_frame_->ToThisFrameAtTimeSimilarly(previous_time);
  DegreesOfFreedom<Navigation> const initial_degrees_of_freedom =
      to_plotting_frame_at_t(
          trajectory.EvaluateDegreesOfFreedom(previous_time));
  Position<Navigation> previous_position =
      initial_degrees_of_freedom.position();
  Velocity<Navigation> previous_velocity =
      initial_degrees_of_freedom.velocity();
  Time Δt = final_time - previous_time;

  Instant t;
  double estimated_tan²_error;
  std::optional<DegreesOfFreedom<Barycentric>>
      degrees_of_freedom_in_barycentric;
  Position<Navigation> position;
  Square<Length> minimal_squared_distance = Infinity<Square<Length>>;

  std::optional<Position<Navigation>> last_endpoint;

  int steps_accepted = 0;

  goto estimate_tan²_error;

  while (steps_accepted < max_plot_method_2_steps &&
         direction * (previous_time - final_time) < Time{}) {
    do {
      // One square root because we have squared errors, another one because the
      // errors are quadratic in time (in other words, two square roots because
      // the squared errors are quartic in time).
      // A safety factor prevents catastrophic retries.
      Δt *= 0.9 * Sqrt(Sqrt(tan²_angular_resolution / estimated_tan²_error));
    estimate_tan²_error:
      t = previous_time + Δt;
      if (direction * (t - final_time) > Time{}) {
        t = final_time;
        Δt = t - previous_time;
      }
      Position<Navigation> const extrapolated_position =
          previous_position + previous_velocity * Δt;
      to_plotting_frame_at_t = plotting_frame_->ToThisFrameAtTimeSimilarly(t);
      degrees_of_freedom_in_barycentric =
          trajectory.EvaluateDegreesOfFreedom(t);
      position = to_plotting_frame_at_t.similarity()(
                     degrees_of_freedom_in_barycentric->position());

      // The quadratic term of the error between the linear interpolation and
      // the actual function is maximized halfway through the segment, so it is
      // 1/2 (Δt/2)² f″(t-Δt) = (1/2 Δt² f″(t-Δt)) / 4; the squared error is
      // thus (1/2 Δt² f″(t-Δt))² / 16.
      estimated_tan²_error =
          perspective_.Tan²AngularDistance(extrapolated_position, position) /
          16;
    } while (estimated_tan²_error > tan²_angular_resolution);
    ++steps_accepted;

    // TODO(egg): also limit to field of view.
    auto const segment_behind_focal_plane =
        perspective_.SegmentBehindFocalPlane(
            Segment<Navigation>(previous_position, position));

    previous_time = t;
    previous_position = position;
    previous_velocity =
        to_plotting_frame_at_t(*degrees_of_freedom_in_barycentric).velocity();

    if (!segment_behind_focal_plane) {
      continue;
    }

    if (minimal_distance != nullptr) {
      minimal_squared_distance =
          std::min(minimal_squared_distance,
                   perspective_.SquaredDistanceFromCamera(position));
    }

    auto const visible_segments = perspective_.VisibleSegments(
                                      *segment_behind_focal_plane,
                                      plottable_spheres);
    for (auto const& segment : visible_segments) {
      if (last_endpoint != segment.first) {
        lines.emplace_back();
        lines.back().push_back(perspective_(segment.first));
      }
      lines.back().push_back(perspective_(segment.second));
      last_endpoint = segment.second;
    }
  }
  if (minimal_distance != nullptr) {
    *minimal_distance = Sqrt(minimal_squared_distance);
  }
  return lines;
}

void Planetarium::PlotMethod3(
    Trajectory<Barycentric> const& trajectory,
    DiscreteTrajectory<Barycentric>::iterator begin,
    DiscreteTrajectory<Barycentric>::iterator end,
    Instant const& t_max,
    bool const reverse,
    std::function<void(ScaledSpacePoint const&)> const& add_point,
    int max_points) const {
  if (begin == end) {
    return;
  }
  auto last = std::prev(end);
  auto const begin_time = std::max(begin->time, plotting_frame_->t_min());
  auto const last_time =
      std::min({last->time, plotting_frame_->render_t_max(), t_max});
  PlotMethod3(
      trajectory, begin_time, last_time, reverse, add_point, max_points);
}

void Planetarium::PlotMethod4(
    Trajectory<Barycentric> const& trajectory,
    DiscreteTrajectory<Barycentric>::iterator begin,
    DiscreteTrajectory<Barycentric>::iterator end,
    Instant const& t_max,
    bool const reverse,
    std::function<void(ScaledSpacePoint const&)> const& add_point,
    int max_points,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    R3Element<double>* const anchor_out,
    std::optional<Registration> const& registration,
    int* const seam_vertex_count) const {
  if (anchor_out != nullptr) {
    *anchor_out = R3Element<double>{};
  }
  if (seam_vertex_count != nullptr) {
    *seam_vertex_count = 0;
  }
  if (begin == end) {
    return;
  }
  auto last = std::prev(end);
  auto const begin_time = std::max(begin->time, plotting_frame_->t_min());
  auto const last_time =
      std::min({last->time, plotting_frame_->render_t_max(), t_max});
  // The time-based overload seams the plot at the plotting frame's own
  // horizon when a seam is requested.
  PlotMethod4(trajectory, begin_time, last_time, reverse, add_point,
              max_points, /*minimal_distance=*/nullptr, placement, anchor_out,
              registration, seam_vertex_count);
}

ScaledSpacePoint Planetarium::PlotPoint(
    Instant const& t,
    Position<Barycentric> const& position,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    R3Element<double>& anchor_out,
    Registration const& registration) const {
  // This mirrors `PlotMethod4Anchored` for a single point; see there for the
  // error analysis of each term.  A frame that does not cover `t` maps the
  // point through the nearest state it has, on either side; the point itself
  // stays at its own time.
  CHECK(plotting_to_scaled_space_displacement_ != nullptr);
  Instant const t_frame = std::min(std::max(t, plotting_frame_->t_min()),
                                   plotting_frame_->render_t_max());
  Instant const t_ref = registration.time;

  Displacement<Barycentric> offset;
  Velocity<Barycentric> velocity_offset;
  if (Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
          plotting_frame_->placement();
      placement != frame_placement) {
    auto const conversion =
        ephemeris_->placement_conversion(placement, frame_placement, t_ref);
    offset = conversion.first;
    velocity_offset = conversion.second;
  }
  Position<Barycentric> q_ref = registration.position;
  if (registration.placement != placement) {
    q_ref += ephemeris_
                 ->placement_conversion(
                     registration.placement, placement, t_ref)
                 .first;
  }
  SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t_ref =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t_ref);
  auto const& similarity_ref = to_plotting_frame_at_t_ref.similarity();
  Position<Barycentric> const frame_origin_ref =
      similarity_ref.Inverse()(Navigation::origin);
  Displacement<Barycentric> const reference = (q_ref + offset) -
                                              frame_origin_ref;
  Displacement<Navigation> const reference_in_navigation =
      similarity_ref.linear_map()(reference);
  Position<Navigation> const nav_ref = Navigation::origin +
                                       reference_in_navigation;
  Displacement<Navigation> const camera_to_reference =
      nav_ref - perspective_.camera();
  anchor_out = plotting_to_scaled_space_displacement_(-camera_to_reference);

  SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t_frame);
  auto const& similarity = to_plotting_frame_at_t.similarity();
  Displacement<Barycentric> const from_reference_in_barycentric =
      (position - q_ref) + velocity_offset * (t - t_ref);
  Displacement<Barycentric> const frame_origin_motion =
      similarity.Inverse()(Navigation::origin) - frame_origin_ref;
  Displacement<Navigation> const sweep =
      similarity.linear_map()(reference) - reference_in_navigation;
  Displacement<Navigation> const reference_relative =
      sweep + similarity.linear_map()(from_reference_in_barycentric -
                                      frame_origin_motion);
  return ScaledSpacePoint::FromCoordinates(
      plotting_to_scaled_space_displacement_(camera_to_reference +
                                             reference_relative));
}

void Planetarium::PlotMethod4Anchored(
    Trajectory<Barycentric> const& trajectory,
    Instant const& first_time,
    Instant const& last_time,
    bool const reverse,
    std::function<void(ScaledSpacePoint const&)> const& add_point,
    int const max_points,
    Length* const minimal_distance,
    Ephemeris<Barycentric>::SubsystemPlacement const& placement,
    R3Element<double>& anchor_out,
    Registration const& registration) const {
  // Anchored plotting emits through the linear part of the conversion, which
  // must therefore be given at construction.
  CHECK(plotting_to_scaled_space_displacement_ != nullptr);
  auto const final_time = reverse ? first_time : last_time;
  auto previous_time = reverse ? last_time : first_time;

  if (minimal_distance != nullptr) {
    *minimal_distance = Infinity<Length>;
  }

  Sign const direction = reverse ? Sign::Negative() : Sign::Positive();
  if (direction * (final_time - previous_time) <= Time{}) {
    return;
  }

  // The reference is given by the caller rather than taken from the plot: a
  // flight-plan segment starts at a manœuvre, not at the present, and the
  // consumer can only name the plotted object where it is now.
  Instant const t_ref = registration.time;

  // The affine-in-time conversion from the trajectory's placement to the
  // plotting frame's, exact because it is folded at `t_ref` (see
  // `TranslatedTrajectory`).  It is applied algebraically below rather than
  // through a translated trajectory, whose evaluated positions would round at
  // the magnitude of the offset for each vertex.
  Displacement<Barycentric> offset;
  Velocity<Barycentric> velocity_offset;
  if (Ephemeris<Barycentric>::SubsystemPlacement const frame_placement =
          plotting_frame_->placement();
      placement != frame_placement) {
    auto const conversion =
        ephemeris_->placement_conversion(placement, frame_placement, t_ref);
    offset = conversion.first;
    velocity_offset = conversion.second;
  }

  // Everything at the magnitude of the distance to the plotting frame's
  // origin is computed once here; only displacements appear per vertex.  The
  // registration is brought into the trajectory's placement on the sector
  // lattice, where two anchors difference exactly.
  Position<Barycentric> q_ref = registration.position;
  if (registration.placement != placement) {
    q_ref += ephemeris_
                 ->placement_conversion(
                     registration.placement, placement, t_ref)
                 .first;
  }
  SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t_ref =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t_ref);
  auto const& similarity_ref = to_plotting_frame_at_t_ref.similarity();
  Position<Barycentric> const frame_origin_ref =
      similarity_ref.Inverse()(Navigation::origin);
  Displacement<Barycentric> const reference = (q_ref + offset) -
                                              frame_origin_ref;
  Displacement<Navigation> const reference_in_navigation =
      similarity_ref.linear_map()(reference);
  Position<Navigation> const nav_ref = Navigation::origin +
                                       reference_in_navigation;
  Displacement<Navigation> const camera_to_reference =
      nav_ref - perspective_.camera();

  // The anchor is the displacement from the reference to the camera, not a
  // position: the adapter draws the mesh at the scene's own mapping of the
  // reference minus this, so the camera term cancels against the one carried
  // by every vertex and nothing has to express either point in scaled space
  // across the distance to the plotting frame's origin.
  anchor_out = plotting_to_scaled_space_displacement_(-camera_to_reference);

  // Evaluates the trajectory point at `t` as a displacement from the camera,
  // together with its velocity in the plotting frame.
  auto const evaluate = [this, &offset, &q_ref, &reference,
                         &reference_in_navigation, &frame_origin_ref, t_ref,
                         &trajectory, &velocity_offset](Instant const& t) {
    auto const degrees_of_freedom = trajectory.EvaluateDegreesOfFreedom(t);
    SimilarMotion<Barycentric, Navigation> const to_plotting_frame_at_t =
        plotting_frame_->ToThisFrameAtTimeSimilarly(t);
    auto const& similarity = to_plotting_frame_at_t.similarity();
    Displacement<Barycentric> const from_reference_in_barycentric =
        (degrees_of_freedom.position() - q_ref) +
        velocity_offset * (t - t_ref);
    Displacement<Barycentric> const frame_origin_motion =
        similarity.Inverse()(Navigation::origin) - frame_origin_ref;
    // Exactly zero for an inertial frame, whose linear map does not change;
    // for a rotating one this is the sweep of the reference, which rounds at
    // the ULP of the reference distance per vertex — as the legacy path does
    // — and is dwarfed by the arc the rotation sweeps at that distance.
    Displacement<Navigation> const sweep =
        similarity.linear_map()(reference) - reference_in_navigation;
    Displacement<Navigation> const reference_relative =
        sweep + similarity.linear_map()(from_reference_in_barycentric -
                                        frame_origin_motion);
    // The position of this degrees of freedom rounds at the magnitude of the
    // offset, but it only enters the velocity through the angular motion of
    // the frame, where its error is negligible.
    Velocity<Navigation> const velocity =
        to_plotting_frame_at_t(
            DegreesOfFreedom<Barycentric>(
                degrees_of_freedom.position() + offset +
                    velocity_offset * (t - t_ref),
                degrees_of_freedom.velocity() + velocity_offset)).velocity();
    return std::make_pair(reference_relative, velocity);
  };

  // The proper motion of a point seen from the camera, as in `ProperMotion`
  // but on camera-relative displacements.
  auto const proper_motion = [](Displacement<Navigation> const& r,
                                Velocity<Navigation> const& velocity) {
    return Wedge(r, velocity) / r.Norm²() * Radian;
  };

  auto const [initial_reference_relative, initial_velocity] =
      evaluate(previous_time);
  Displacement<Navigation> previous_camera_relative =
      camera_to_reference + initial_reference_relative;
  Vector<AngularFrequency, Navigation> previous_projected_velocity =
      proper_motion(previous_camera_relative, initial_velocity) *
      Normalize(previous_camera_relative);

  Time Δt = final_time - previous_time;

  add_point(ScaledSpacePoint::FromCoordinates(
      plotting_to_scaled_space_displacement_(previous_camera_relative)));
  int points_added = 1;

  Instant t;
  Angle rms_apparent_distance;
  Displacement<Navigation> reference_relative;
  Displacement<Navigation> camera_relative;
  Vector<AngularFrequency, Navigation> projected_velocity;
  Square<Length> minimal_squared_distance = Infinity<Square<Length>>;

  goto estimate_tan²_error;

  while (points_added < max_points &&
         direction * (previous_time - final_time) < Time{}) {
    do {
      // See `PlotMethod4` for the error analysis; this is the same stepping,
      // on camera-relative displacements.
      Δt *= 0.9 * Sqrt(parameters_.angular_resolution_ / rms_apparent_distance);
    estimate_tan²_error:
      t = previous_time + Δt;
      if (t == previous_time) {
        LOG(ERROR) << "At time " << t
                   << ", step size is effectively zero.  Singularity or "
                      "stiff system suspected.";
        return;
      }
      if (direction * (t - final_time) > Time{}) {
        t = final_time;
        Δt = t - previous_time;
      }

      auto const evaluated = evaluate(t);
      reference_relative = evaluated.first;
      camera_relative = camera_to_reference + reference_relative;
      Velocity<Navigation> const& velocity = evaluated.second;

      Velocity<Navigation> const linear_velocity =
          (camera_relative - previous_camera_relative) / Δt;

      projected_velocity = proper_motion(camera_relative, velocity) *
                           Normalize(camera_relative);
      auto const previous_projected_linear_velocity =
          proper_motion(previous_camera_relative, linear_velocity) *
          Normalize(previous_camera_relative);
      auto const projected_linear_velocity =
          proper_motion(camera_relative, linear_velocity) *
          Normalize(camera_relative);

      Hermite3<Vector<Angle, Navigation>, Instant> const error_approximation(
          {previous_time, t},
          {Vector<Angle, Navigation>{}, Vector<Angle, Navigation>{}},
          {previous_projected_velocity - previous_projected_linear_velocity,
           projected_velocity - projected_linear_velocity});

      rms_apparent_distance =
          Sqrt(GaussLegendre<4>(
                   [&error_approximation](Instant const& time) {
                     return error_approximation(time).Norm²();
                   },
                   previous_time,
                   t) /
               Δt);
    } while (rms_apparent_distance > parameters_.angular_resolution_);

    previous_time = t;
    previous_camera_relative = camera_relative;
    previous_projected_velocity = projected_velocity;

    add_point(ScaledSpacePoint::FromCoordinates(
        plotting_to_scaled_space_displacement_(camera_relative)));
    ++points_added;

    if (minimal_distance != nullptr) {
      minimal_squared_distance =
          std::min(minimal_squared_distance, camera_relative.Norm²());
    }
  }
  if (minimal_distance != nullptr) {
    *minimal_distance = Sqrt(minimal_squared_distance);
  }
}

std::vector<Sphere<Navigation>> Planetarium::ComputePlottableSpheres(
    Instant const& now) const {
  SimilarMotion<Barycentric, Navigation> const similar_motion_at_now =
      plotting_frame_->ToThisFrameAtTimeSimilarly(now);
  std::vector<Sphere<Navigation>> plottable_spheres;

  int const plotting_subsystem = plotting_frame_->subsystem();
  auto const& bodies = ephemeris_->bodies();
  for (not_null<MassiveBody const*> const body : bodies) {
    auto const trajectory = ephemeris_->trajectory(body);
    Length const mean_radius = body->mean_radius();
    Position<Barycentric> centre_in_barycentric =
        trajectory->EvaluatePosition(now);
    if (int const body_subsystem = ephemeris_->subsystem_of_body(body);
        body_subsystem != plotting_subsystem) {
      centre_in_barycentric +=
          ephemeris_->subsystem_conversion(body_subsystem,
                                           plotting_subsystem,
                                           now);
    }
    Sphere<Navigation> plottable_sphere(
        similar_motion_at_now.similarity()(centre_in_barycentric),
        parameters_.sphere_radius_multiplier_ * mean_radius);
    // If the sphere is seen under an angle that is very small it doesn't
    // participate in hiding.
    if (perspective_.SphereSin²HalfAngle(plottable_sphere) >
        parameters_.sin²_angular_resolution_) {
      plottable_spheres.emplace_back(std::move(plottable_sphere));
    }
  }
  return plottable_spheres;
}

Segments<Navigation> Planetarium::ComputePlottableSegments(
    const std::vector<Sphere<Navigation>>& plottable_spheres,
    DiscreteTrajectory<Barycentric>::iterator const begin,
    DiscreteTrajectory<Barycentric>::iterator const end) const {
  Segments<Navigation> all_segments;
  if (begin == end) {
    return all_segments;
  }
  auto it1 = begin;
  Instant t1 = it1->time;
  SimilarMotion<Barycentric, Navigation> similar_motion_at_t1 =
      plotting_frame_->ToThisFrameAtTimeSimilarly(t1);
  Position<Navigation> p1 =
      similar_motion_at_t1(it1->degrees_of_freedom).position();

  auto it2 = it1;
  while (++it2 != end) {
    // Processing one segment of the trajectory.
    Instant const t2 = it2->time;

    // Transform the degrees of freedom to the plotting frame.
    SimilarMotion<Barycentric, Navigation> const similar_motion_at_t2 =
        plotting_frame_->ToThisFrameAtTimeSimilarly(t2);
    Position<Navigation> const p2 =
        similar_motion_at_t2(it2->degrees_of_freedom).position();

    // Find the part of the segment that is behind the focal plane.  We don't
    // care about things that are in front of the focal plane.
    const Segment<Navigation> segment = {p1, p2};
    auto const segment_behind_focal_plane =
        perspective_.SegmentBehindFocalPlane(segment);
    if (segment_behind_focal_plane) {
      // Find the part(s) of the segment that are not hidden by spheres.  These
      // are the ones we want to plot.
      auto segments = perspective_.VisibleSegments(*segment_behind_focal_plane,
                                                   plottable_spheres);
      std::move(segments.begin(),
                segments.end(),
                std::back_inserter(all_segments));
    }

    it1 = it2;
    t1 = t2;
    similar_motion_at_t1 = similar_motion_at_t2;
    p1 = p2;
  }

  return all_segments;
}

}  // namespace internal
}  // namespace _planetarium
}  // namespace ksp_plugin
}  // namespace principia
