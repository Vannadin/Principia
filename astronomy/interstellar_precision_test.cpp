#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "astronomy/frames.hpp"
#include "base/not_null.hpp"
#include "geometry/barycentre_calculator.hpp"
#include "geometry/frame.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "gtest/gtest.h"
#include "integrators/embedded_explicit_runge_kutta_nyström_integrator.hpp"
#include "integrators/methods.hpp"
#include "integrators/symmetric_linear_multistep_integrator.hpp"
#include "integrators/symplectic_runge_kutta_nyström_integrator.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/apsides.hpp"
#include "physics/body_centred_body_direction_reference_frame.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "physics/oblate_body.hpp"
#include "physics/rigid_motion.hpp"
#include "physics/rotating_body.hpp"
#include "physics/sector.hpp"
#include "physics/solar_system.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "serialization/physics.pb.h"
#include "testing_utilities/almost_equals.hpp"
#include "testing_utilities/matchers.hpp"  // 🧙 For EXPECT_OK.
#include "testing_utilities/numerics.hpp"

namespace principia {
namespace astronomy {

using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::Lt;
using namespace principia::astronomy::_frames;
using namespace principia::base::_not_null;
using namespace principia::geometry::_barycentre_calculator;
using namespace principia::geometry::_frame;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_embedded_explicit_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::integrators::_symplectic_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_apsides;
using namespace principia::physics::_body_centred_body_direction_reference_frame;  // NOLINT
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_oblate_body;
using namespace principia::physics::_rigid_motion;
using namespace principia::physics::_rotating_body;
using namespace principia::physics::_sector;
using namespace principia::physics::_solar_system;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
using namespace principia::testing_utilities::_matchers;
using namespace principia::testing_utilities::_numerics;

// Two star-planet systems with bit-identical initial conditions (up to a
// translation), the second one 4 × 10¹⁶ m away from the first.  In exact
// arithmetic the two systems evolve identically, so any difference between
// their star-planet separations measures the round-off error caused by the
// distance of the second system to the origin.  The distance and the orbit
// radius are chosen so that the initial positions of the remote system are
// exactly representable.
class InterstellarPrecisionTest : public ::testing::Test {
 protected:
  static constexpr GravitationalParameter μ_star =
      TerrestrialGravitationalParameter;
  static constexpr GravitationalParameter μ_planet = 1e-6 * μ_star;
  static constexpr Length orbit_radius = 1.0e9 * Metre;

  static Time Period() {
    return 2 * π * Sqrt(Pow<3>(orbit_radius) / (μ_star + μ_planet));
  }

  // The (exactly representable) displacement from the local system to the
  // remote one.
  static Displacement<ICRS> ToRemoteSystem() {
    return Displacement<ICRS>({4.0e16 * Metre, 0 * Metre, 0 * Metre});
  }

  // Constructs an ephemeris containing the two systems.  If `subsystems` is
  // empty all the positions are represented in a single frame; otherwise each
  // system is represented relative to its own local origin.  If
  // `far_field_damping_floor` is strictly positive, the far field is damped
  // (see `FarFieldDamping`), both as seen by massless bodies and between the
  // massive bodies; if `far_field_damping_epsilon` is strictly positive as
  // well, the massive-massive cutoff is relative instead.
  static not_null<std::unique_ptr<Ephemeris<ICRS>>> MakeEphemeris(
      std::vector<int> const& subsystems,
      Acceleration const& far_field_damping_floor = {},
      Velocity<ICRS> const& remote_system_velocity = {},
      double const far_field_damping_epsilon = 0) {
    Instant const t0;
    Displacement<ICRS> const to_remote_system = ToRemoteSystem();

    Speed const v = Sqrt((μ_star + μ_planet) / orbit_radius);

    // A circular orbit at rest in the barycentric frame of each system.
    Displacement<ICRS> const star_to_planet(
        {0 * Metre, orbit_radius, 0 * Metre});
    Velocity<ICRS> const star_velocity({-v * (μ_planet / (μ_star + μ_planet)),
                                        0 * Metre / Second,
                                        0 * Metre / Second});
    Velocity<ICRS> const planet_velocity({v * (μ_star / (μ_star + μ_planet)),
                                          0 * Metre / Second,
                                          0 * Metre / Second});

    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star A", μ_star)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("planet A", μ_planet)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star B", μ_star)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("planet B", μ_planet)));

    std::vector<DegreesOfFreedom<ICRS>> const initial_state{
        {ICRS::origin, star_velocity},
        {ICRS::origin + star_to_planet, planet_velocity},
        {ICRS::origin + to_remote_system,
         star_velocity + remote_system_velocity},
        {ICRS::origin + to_remote_system + star_to_planet,
         planet_velocity + remote_system_velocity}};

    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            /*step=*/Period() / 1000),
        subsystems,
        far_field_damping_floor,
        far_field_damping_epsilon);
  }

  // Evolves a massless probe in a circular orbit around each star for 10 years
  // and returns the largest difference between the probe-star separations of
  // the two systems.  This exercises the massless integration path.
  static Length ComputeMaxProbeSeparationError(bool const use_subsystems) {
    auto const ephemeris = MakeEphemeris(
        use_subsystems ? std::vector<int>{0, 0, 1, 1} : std::vector<int>{});
    int const remote_subsystem = use_subsystems ? 1 : 0;

    Instant const t0;
    Instant const t_final = t0 + 10 * JulianYear;
    Displacement<ICRS> const to_remote_system = ToRemoteSystem();

    Length const probe_orbit_radius = 2 * orbit_radius;
    Speed const v_probe = Sqrt(μ_star / probe_orbit_radius);
    Displacement<ICRS> const star_to_probe(
        {0 * Metre, probe_orbit_radius, 0 * Metre});
    Speed const v_star = -Sqrt((μ_star + μ_planet) / orbit_radius) *
                         (μ_planet / (μ_star + μ_planet));
    Velocity<ICRS> const probe_velocity(
        {v_probe + v_star, 0 * Metre / Second, 0 * Metre / Second});

    // In the subsystem representation the remote probe has the same initial
    // state as the local one; in the single-origin representation it is
    // displaced by (the exactly representable) `to_remote_system`.
    DiscreteTrajectory<ICRS> probe_a;
    DiscreteTrajectory<ICRS> probe_b;
    EXPECT_OK(probe_a.Append(
        t0,
        DegreesOfFreedom<ICRS>(ICRS::origin + star_to_probe, probe_velocity)));
    EXPECT_OK(probe_b.Append(
        t0,
        DegreesOfFreedom<ICRS>(
            use_subsystems ? ICRS::origin + star_to_probe
                           : ICRS::origin + to_remote_system + star_to_probe,
            probe_velocity)));

    Ephemeris<ICRS>::AdaptiveStepParameters const adaptive_parameters(
        EmbeddedExplicitRungeKuttaNyströmIntegrator<
            DormandالمكاوىPrince1986RKN434FM,
            Ephemeris<ICRS>::NewtonianMotionEquation>(),
        /*max_steps=*/std::numeric_limits<std::int64_t>::max(),
        /*length_integration_tolerance=*/1 * Metre,
        /*speed_integration_tolerance=*/1e-3 * Metre / Second);
    EXPECT_OK(ephemeris->FlowWithAdaptiveStep(
        &probe_a,
        Ephemeris<ICRS>::NoIntrinsicAcceleration,
        t_final,
        adaptive_parameters,
        Ephemeris<ICRS>::unlimited_max_ephemeris_steps,
        Ephemeris<ICRS>::SubsystemPlacement::Stock()));
    EXPECT_OK(ephemeris->FlowWithAdaptiveStep(
        &probe_b,
        Ephemeris<ICRS>::NoIntrinsicAcceleration,
        t_final,
        adaptive_parameters,
        Ephemeris<ICRS>::unlimited_max_ephemeris_steps,
        {remote_subsystem, std::nullopt}));

    auto const& star_a = *ephemeris->trajectory(ephemeris->bodies()[0]);
    auto const& star_b = *ephemeris->trajectory(ephemeris->bodies()[2]);

    Length max_separation_error;
    for (Instant t = t0; t <= t_final; t += 1 * Day) {
      Length const separation_a =
          (probe_a.EvaluatePosition(t) - star_a.EvaluatePosition(t)).Norm();
      Length const separation_b =
          (probe_b.EvaluatePosition(t) - star_b.EvaluatePosition(t)).Norm();
      max_separation_error =
          std::max(max_separation_error, Abs(separation_b - separation_a));
    }
    return max_separation_error;
  }

  // Evolves the two systems for 100 years and returns the largest difference
  // between their star-planet separations.
  static Length ComputeMaxSeparationError(std::vector<int> const& subsystems) {
    auto const ephemeris = MakeEphemeris(subsystems);
    EXPECT_OK(ephemeris->Prolong(Instant() + 100 * JulianYear));

    auto const& star_a = *ephemeris->trajectory(ephemeris->bodies()[0]);
    auto const& planet_a = *ephemeris->trajectory(ephemeris->bodies()[1]);
    auto const& star_b = *ephemeris->trajectory(ephemeris->bodies()[2]);
    auto const& planet_b = *ephemeris->trajectory(ephemeris->bodies()[3]);

    Length max_separation_error;
    for (Instant t = ephemeris->t_min(); t <= ephemeris->t_max();
         t += 1 * Day) {
      Length const separation_a =
          (planet_a.EvaluatePosition(t) - star_a.EvaluatePosition(t)).Norm();
      Length const separation_b =
          (planet_b.EvaluatePosition(t) - star_b.EvaluatePosition(t)).Norm();
      max_separation_error =
          std::max(max_separation_error, Abs(separation_b - separation_a));
    }
    return max_separation_error;
  }
};

#if !defined(_DEBUG)

TEST_F(InterstellarPrecisionTest, RemoteSystemSeparation) {
  Length const error_without_subsystems =
      ComputeMaxSeparationError(/*subsystems=*/{});
  Length const error_with_subsystems =
      ComputeMaxSeparationError(/*subsystems=*/{0, 0, 1, 1});
  LOG(INFO) << "Separation error without subsystems: "
            << error_without_subsystems;
  LOG(INFO) << "Separation error with subsystems: " << error_with_subsystems;
  EXPECT_THAT(error_without_subsystems, Gt(1 * Metre));
  EXPECT_THAT(error_with_subsystems, Lt(1 * Milli(Metre)));
}

// Checks that the Jacobian and jerk computations, which involve
// cross-subsystem pairs, are unaffected by the representation.
TEST_F(InterstellarPrecisionTest, JacobianAndJerk) {
  auto const control = MakeEphemeris(/*subsystems=*/{});
  auto const with_subsystems = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  Instant const t = Instant() + 10 * Period();
  EXPECT_OK(control->Prolong(t));
  EXPECT_OK(with_subsystems->Prolong(t));

  Displacement<ICRS> const δq({0 * Metre, 1 * Metre, 0 * Metre});
  auto const jacobian_control =
      control->ComputeJacobianOnMassiveBody(control->bodies()[0], t);
  auto const jacobian_with_subsystems =
      with_subsystems->ComputeJacobianOnMassiveBody(
          with_subsystems->bodies()[0], t);
  EXPECT_THAT(RelativeError((jacobian_control * δq).Norm(),
                            (jacobian_with_subsystems * δq).Norm()),
              Lt(1e-9));

  auto const jerks_control = control->ComputeGravitationalJerkOnMassiveBodies(
      control->bodies(), control->EvaluateAllDegreesOfFreedom(t), t);
  auto const jerks_with_subsystems =
      with_subsystems->ComputeGravitationalJerkOnMassiveBodies(
          with_subsystems->bodies(),
          with_subsystems->EvaluateAllDegreesOfFreedom(t),
          t);
  EXPECT_THAT(
      RelativeError(jerks_control[0].Norm(), jerks_with_subsystems[0].Norm()),
      Lt(1e-9));
  // The two systems are congruent, so the remote star must undergo the same
  // jerk as the local one.
  EXPECT_THAT(RelativeError(jerks_with_subsystems[0].Norm(),
                            jerks_with_subsystems[2].Norm()),
              Lt(1e-9));
}

TEST_F(InterstellarPrecisionTest, ClusterSubsystems) {
  Length const threshold = 1e14 * Metre;
  auto const far = ToRemoteSystem();
  auto const near = Displacement<ICRS>({1e13 * Metre, 0 * Metre, 0 * Metre});

  // A single system, however far from the origin, is not partitioned.
  EXPECT_THAT(
      ClusterSubsystems<ICRS>({ICRS::origin, ICRS::origin + near}, threshold),
      IsEmpty());
  EXPECT_THAT(ClusterSubsystems<ICRS>(
                  {ICRS::origin + far, ICRS::origin + far + near}, threshold),
              IsEmpty());

  // Two systems, interleaved: the subsystems are numbered by order of first
  // appearance.
  EXPECT_THAT(ClusterSubsystems<ICRS>({ICRS::origin,
                                       ICRS::origin + far,
                                       ICRS::origin + near,
                                       ICRS::origin + far + near},
                                      threshold),
              ElementsAre(0, 1, 0, 1));

  // Single-linkage chaining: a body bridging two otherwise-distant groups
  // merges them.
  EXPECT_THAT(ClusterSubsystems<ICRS>({ICRS::origin,
                                       ICRS::origin + near,
                                       ICRS::origin + 2 * near,
                                       ICRS::origin + far},
                                      threshold),
              ElementsAre(0, 0, 0, 1));

  // The subsystem of each body is exposed by the ephemeris; note that the
  // subsystems are given per input body but exposed per (reordered) body.
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_EQ(0, ephemeris->subsystem_of_body(ephemeris->bodies()[0]));
  EXPECT_EQ(0, ephemeris->subsystem_of_body(ephemeris->bodies()[1]));
  EXPECT_EQ(1, ephemeris->subsystem_of_body(ephemeris->bodies()[2]));
  EXPECT_EQ(1, ephemeris->subsystem_of_body(ephemeris->bodies()[3]));
}

// Checks the per-subsystem barycentric data recorded at construction: total
// gravitational parameters, initial barycentres relative to the local origins,
// and the linear extrapolation of the barycentres.
TEST_F(InterstellarPrecisionTest, SubsystemBarycentre) {
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  Instant const t0;

  EXPECT_EQ(μ_star + μ_planet, ephemeris->subsystem_gravitational_parameter(0));
  EXPECT_EQ(μ_star + μ_planet, ephemeris->subsystem_gravitational_parameter(1));

  // The two systems have bit-identical initial conditions relative to their
  // local origins, so their barycentric data are bit-identical too.
  EXPECT_EQ(ephemeris->subsystem_barycentre(0, t0),
            ephemeris->subsystem_barycentre(1, t0));
  EXPECT_EQ(ephemeris->subsystem_barycentre_velocity(0),
            ephemeris->subsystem_barycentre_velocity(1));

  // The star is at the local origin, so the barycentre is at the mass-weighted
  // fraction of the star-planet separation.
  Displacement<ICRS> const star_to_planet({0 * Metre, orbit_radius, 0 * Metre});
  EXPECT_THAT(ephemeris->subsystem_barycentre(0, t0) - ICRS::origin,
              AlmostEquals(star_to_planet * (μ_planet / (μ_star + μ_planet)),
                           0, 4));

  // The orbit is at rest in the barycentric frame of its system, so the
  // velocity of the barycentre vanishes up to rounding.
  EXPECT_THAT(ephemeris->subsystem_barycentre_velocity(0).Norm(),
              Lt(1e-15 * Sqrt((μ_star + μ_planet) / orbit_radius)));

  // The local origin moves with the barycentre, so the barycentre of the
  // integrated system stays at its (constant) initial local position except
  // for the pull of the remote system; over Δt the deviation is ½ a Δt² with
  // a = μ / d².  (About 12 mm here, dwarfing the integration error.)
  Instant const t_final = t0 + 10 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  auto const& star_a = *ephemeris->trajectory(ephemeris->bodies()[0]);
  auto const& planet_a = *ephemeris->trajectory(ephemeris->bodies()[1]);
  Position<ICRS> const integrated_barycentre =
      Barycentre({star_a.EvaluatePosition(t_final),
                  planet_a.EvaluatePosition(t_final)},
                 {μ_star, μ_planet});
  Length const predicted_deviation = 0.5 * (μ_star + μ_planet) *
                                     Pow<2>(t_final - t0) /
                                     Pow<2>(ToRemoteSystem().Norm());
  EXPECT_THAT(
      (integrated_barycentre - ephemeris->subsystem_barycentre(0, t_final))
          .Norm(),
      AllOf(Gt(0.99 * predicted_deviation), Lt(1.01 * predicted_deviation)));
}

// A remote system given a bulk velocity of the order of the fastest NearStars
// stars: the local origin moves with the barycentre of its subsystem, so the
// local coordinates stay bounded and the local evolution is that of the same
// system at rest; the cross-subsystem physics — offsets, jerk — sees the
// true, time-dependent geometry.
TEST_F(InterstellarPrecisionTest, MovingRemoteSubsystem) {
  Velocity<ICRS> const bulk_velocity({300 * Kilo(Metre) / Second,
                                      0 * Metre / Second,
                                      0 * Metre / Second});
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1},
                                       /*far_field_damping_floor=*/{},
                                       bulk_velocity);
  Instant const t0;
  Instant const t_final = t0 + 100 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));

  auto const& star_a = *ephemeris->trajectory(ephemeris->bodies()[0]);
  auto const& planet_a = *ephemeris->trajectory(ephemeris->bodies()[1]);
  auto const& star_b = *ephemeris->trajectory(ephemeris->bodies()[2]);
  auto const& planet_b = *ephemeris->trajectory(ephemeris->bodies()[3]);

  // Over a century the remote system moves by ~9.5e14 m, but its local
  // coordinates remain bounded: the origin moves with it.
  EXPECT_THAT((star_b.EvaluatePosition(t_final) - ICRS::origin).Norm(),
              Lt(1e10 * Metre));

  // The bulk velocity is removed from the stored velocities, so the local
  // evolution is insensitive to it: the star-planet separations of the two
  // systems agree as tightly as they do for systems at rest.
  Length max_separation_error;
  for (Instant t = t0; t <= t_final; t += 30 * Day) {
    Length const separation_a =
        (planet_a.EvaluatePosition(t) - star_a.EvaluatePosition(t)).Norm();
    Length const separation_b =
        (planet_b.EvaluatePosition(t) - star_b.EvaluatePosition(t)).Norm();
    max_separation_error =
        std::max(max_separation_error, Abs(separation_b - separation_a));
  }
  EXPECT_THAT(max_separation_error, Lt(1 * Milli(Metre)));

  // The inter-subsystem offset is affine in time and tracks the bulk motion.
  EXPECT_EQ(ToRemoteSystem(),
            ephemeris->subsystem_conversion(/*s1=*/1, /*s2=*/0, t0));
  EXPECT_THAT(ephemeris->subsystem_conversion(/*s1=*/1, /*s2=*/0, t_final),
              AlmostEquals(ToRemoteSystem() + bulk_velocity * (t_final - t0),
                           0, 4));
  EXPECT_THAT(ephemeris->subsystem_velocity_conversion(/*s1=*/1, /*s2=*/0),
              AlmostEquals(bulk_velocity, 0, 8));

  // The cross-subsystem jerk sees the true relative velocity.  For a pair of
  // isolated stars receding at the bulk velocity the jerk is dominated by its
  // velocity-dependent term, so this fails outright if the stored velocities
  // are not converted between subsystems.  (The four-body systems above make
  // a poor control: in a single frame the remote system's own geometry is
  // quantized at ~9 m, i.e. ~1e-8 of the orbit radius.)
  auto const make_pair_ephemeris = [&](std::vector<int> const& subsystems) {
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star A", μ_star)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star B", μ_star)));
    std::vector<DegreesOfFreedom<ICRS>> const initial_state{
        {ICRS::origin, ICRS::unmoving},
        {ICRS::origin + ToRemoteSystem(), bulk_velocity}};
    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            /*step=*/Period() / 1000),
        subsystems);
  };
  auto const pair_control = make_pair_ephemeris(/*subsystems=*/{});
  auto const pair_with_subsystems = make_pair_ephemeris(/*subsystems=*/{0, 1});
  Instant const t_jerk = t0 + Period() / 3;
  EXPECT_OK(pair_control->Prolong(t_jerk));
  EXPECT_OK(pair_with_subsystems->Prolong(t_jerk));
  auto const jerks_control =
      pair_control->ComputeGravitationalJerkOnMassiveBodies(
          pair_control->bodies(),
          pair_control->EvaluateAllDegreesOfFreedom(t_jerk),
          t_jerk);
  auto const jerks_with_subsystems =
      pair_with_subsystems->ComputeGravitationalJerkOnMassiveBodies(
          pair_with_subsystems->bodies(),
          pair_with_subsystems->EvaluateAllDegreesOfFreedom(t_jerk),
          t_jerk);
  for (int b = 0; b < 2; ++b) {
    EXPECT_THAT(RelativeError(jerks_control[b].Norm(),
                              jerks_with_subsystems[b].Norm()),
                Lt(1e-9));
  }
}

// Checks that the apsides of a cross-subsystem pair are those of the true
// relative trajectory.  The two systems are congruent and synchronized, so the
// separation between planet A and star B oscillates exactly between
// `to_remote_system.Norm() ∓ orbit_radius`.
TEST_F(InterstellarPrecisionTest, Apsides) {
  Displacement<ICRS> const to_remote_system = ToRemoteSystem();
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_OK(ephemeris->Prolong(Instant() + 2 * Period()));

  DistinguishedPoints<ICRS> apoapsides1;
  DistinguishedPoints<ICRS> periapsides1;
  DistinguishedPoints<ICRS> apoapsides2;
  DistinguishedPoints<ICRS> periapsides2;
  ephemeris->ComputeApsides(/*body1=*/ephemeris->bodies()[1],
                            /*body2=*/ephemeris->bodies()[2],
                            apoapsides1, periapsides1,
                            apoapsides2, periapsides2);

  EXPECT_THAT(apoapsides1.size(), Gt(0));
  EXPECT_THAT(periapsides1.size(), Gt(0));

  auto const separation = [&ephemeris](Instant const& t,
                                       DegreesOfFreedom<ICRS> const& dof1,
                                       DegreesOfFreedom<ICRS> const& dof2) {
    auto const offset =
        ephemeris->inter_subsystem_offset(/*s1=*/0, /*s2=*/1, t);
    return offset.Collapse(dof1.position() - dof2.position()).Norm();
  };
  for (auto const& [t, degrees_of_freedom] : periapsides1) {
    EXPECT_THAT(AbsoluteError(to_remote_system.Norm() - orbit_radius,
                              separation(t,
                                         degrees_of_freedom,
                                         periapsides2.at(t))),
                Lt(100 * Metre));
  }
  for (auto const& [t, degrees_of_freedom] : apoapsides1) {
    EXPECT_THAT(AbsoluteError(to_remote_system.Norm() + orbit_radius,
                              separation(t,
                                         degrees_of_freedom,
                                         apoapsides2.at(t))),
                Lt(100 * Metre));
  }
}

// Same as `RemoteSystemSeparation`, but for the massless integration path.
TEST_F(InterstellarPrecisionTest, MasslessProbe) {
  Length const error_without_subsystems =
      ComputeMaxProbeSeparationError(/*use_subsystems=*/false);
  Length const error_with_subsystems =
      ComputeMaxProbeSeparationError(/*use_subsystems=*/true);
  LOG(INFO) << "Probe separation error without subsystems: "
            << error_without_subsystems;
  LOG(INFO) << "Probe separation error with subsystems: "
            << error_with_subsystems;
  EXPECT_THAT(error_without_subsystems, Gt(1 * Metre));
  EXPECT_THAT(error_with_subsystems, Lt(1 * Milli(Metre)));
}

// With the far field damped, the damping is inert near a star (it only
// removes the negligible pull of the remote system) and a probe in the void
// between the systems is exactly force-free: it coasts inertially.
TEST_F(InterstellarPrecisionTest, FarFieldDampedCoast) {
  Instant const t0;
  Acceleration const far_field_damping_floor =
      1e-12 * Metre / Pow<2>(Second);
  auto const control = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  auto const damped = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1},
                                    far_field_damping_floor);
  EXPECT_OK(control->Prolong(t0 + Period() / 10));
  EXPECT_OK(damped->Prolong(t0 + Period() / 10));

  // Near star A the damping only removes the pull of the remote system,
  // μ_star / (4 × 10¹⁶ m)² ≈ 2.5 × 10⁻¹⁹ m/s².
  Position<ICRS> const near_star_a =
      ICRS::origin +
      Displacement<ICRS>({0 * Metre, 2 * orbit_radius, 0 * Metre});
  EXPECT_THAT(
      (damped->ComputeGravitationalAccelerationOnMasslessBody(
           near_star_a, t0, /*subsystem=*/0) -
       control->ComputeGravitationalAccelerationOnMasslessBody(
           near_star_a, t0, /*subsystem=*/0)).Norm(),
      Lt(1e-18 * Metre / Pow<2>(Second)));

  // In the void between the systems the damped field vanishes exactly.
  Position<ICRS> const mid_void = ICRS::origin + 0.5 * ToRemoteSystem();
  EXPECT_NE(control->ComputeGravitationalAccelerationOnMasslessBody(
                mid_void, t0, /*subsystem=*/0),
            (Vector<Acceleration, ICRS>{}));
  EXPECT_EQ(damped->ComputeGravitationalAccelerationOnMasslessBody(
                mid_void, t0, /*subsystem=*/0),
            (Vector<Acceleration, ICRS>{}));

  // A probe coasting through the void for ten years keeps its velocity
  // bit-for-bit and moves in a straight line.
  Velocity<ICRS> const v0({3 * Kilo(Metre) / Second,
                           1 * Kilo(Metre) / Second,
                           0 * Metre / Second});
  DiscreteTrajectory<ICRS> probe;
  EXPECT_OK(probe.Append(t0, DegreesOfFreedom<ICRS>(mid_void, v0)));
  EXPECT_OK(damped->FlowWithAdaptiveStep(
      &probe,
      Ephemeris<ICRS>::NoIntrinsicAcceleration,
      t0 + 10 * JulianYear,
      Ephemeris<ICRS>::AdaptiveStepParameters(
          EmbeddedExplicitRungeKuttaNyströmIntegrator<
              DormandالمكاوىPrince1986RKN434FM,
              Ephemeris<ICRS>::NewtonianMotionEquation>(),
          /*max_steps=*/std::numeric_limits<std::int64_t>::max(),
          /*length_integration_tolerance=*/1 * Metre,
          /*speed_integration_tolerance=*/1e-3 * Metre / Second),
      Ephemeris<ICRS>::unlimited_max_ephemeris_steps,
      Ephemeris<ICRS>::SubsystemPlacement::Stock()));
  auto const& [final_time, final_degrees_of_freedom] = probe.back();
  EXPECT_EQ(final_time, t0 + 10 * JulianYear);
  EXPECT_EQ(final_degrees_of_freedom.velocity(), v0);
  EXPECT_THAT(((final_degrees_of_freedom.position() - mid_void) -
               v0 * (final_time - t0)).Norm(),
              Lt(1 * Kilo(Metre)));
}

// An anchored representation is a pure relabelling: a probe coasting in the
// void, represented relative to an anchor moving with it, has exactly-zero
// acceleration (the damping cutoff sees the true distances through the
// anchor), so its anchored coordinates do not move at all.  Without the
// anchor term the same coordinates would sit deep in star A's field and feel
// spurious gravity.
TEST_F(InterstellarPrecisionTest, AnchoredVoidCoast) {
  Instant const t0;
  Acceleration const far_field_damping_floor =
      1e-12 * Metre / Pow<2>(Second);
  auto const damped = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1},
                                    far_field_damping_floor);
  EXPECT_OK(damped->Prolong(t0 + Period() / 10));

  Position<ICRS> const mid_void = ICRS::origin + 0.5 * ToRemoteSystem();
  Position<ICRS> const near_star_a =
      ICRS::origin +
      Displacement<ICRS>({0 * Metre, 2 * orbit_radius, 0 * Metre});
  Velocity<ICRS> const v0({3 * Kilo(Metre) / Second,
                           1 * Kilo(Metre) / Second,
                           0 * Metre / Second});

  // The void query: false near a star, true in the void, false everywhere
  // when the far field is not damped.
  EXPECT_FALSE(damped->FarFieldIsZero(near_star_a, /*subsystem=*/0, t0));
  EXPECT_TRUE(damped->FarFieldIsZero(mid_void, /*subsystem=*/0, t0));
  EXPECT_TRUE(damped->FarFieldIsZero(mid_void, /*subsystem=*/1, t0));
  auto const undamped = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_OK(undamped->Prolong(t0 + Period() / 10));
  EXPECT_FALSE(undamped->FarFieldIsZero(mid_void, /*subsystem=*/0, t0));

  // The player-facing report has a floor of its own: at the damping's floor it
  // agrees with the cutoff, and it does not move with it — a report at the
  // actual pull of the stars in the middle of the void still sees them.
  EXPECT_TRUE(damped->FarFieldIsBelow(
      far_field_damping_floor, mid_void, /*subsystem=*/0, t0));
  EXPECT_FALSE(damped->FarFieldIsBelow(
      far_field_damping_floor, near_star_a, /*subsystem=*/0, t0));
  EXPECT_FALSE(damped->FarFieldIsBelow(
      1e-19 * Metre / Pow<2>(Second), mid_void, /*subsystem=*/0, t0));

  // The anchor is adopted at the probe, moving with it; the anchored
  // coordinates start small and force-free coasting keeps them exactly there.
  Displacement<ICRS> const anchored_q₀(
      {1.0e6 * Metre, 0 * Metre, 0 * Metre});
  Ephemeris<ICRS>::Anchor const anchor{
      .offset = SectorDisplacement<ICRS>::Split(
          (mid_void - ICRS::origin) - anchored_q₀),
      .velocity = v0,
      .epoch = t0};

  serialization::Ephemeris::Anchor message;
  anchor.WriteToMessage(&message);
  EXPECT_EQ(anchor, Ephemeris<ICRS>::Anchor::ReadFromMessage(message));

  // A pre-sector message has no `sector_offset`; reading one adopts the
  // nearest cell, the residual going into the local part of the collapsed
  // legacy offset.  (A legacy offset had already rounded at the void
  // magnitude when it was written; that rounding is inherited, not repaired.)
  message.clear_sector_offset();
  auto const migrated = Ephemeris<ICRS>::Anchor::ReadFromMessage(message);
  EXPECT_EQ(migrated.offset,
            SectorDisplacement<ICRS>::Split(anchor.offset.Collapse()));
  EXPECT_EQ(migrated.velocity, anchor.velocity);
  EXPECT_EQ(migrated.epoch, anchor.epoch);

  // The single-sided conversions — exactly one anchor present, the common
  // case of an anchored vessel against an unanchored one — collapse the one
  // anchor, with mirrored signs.
  {
    Instant const t1 = t0 + 1 * JulianYear;
    auto const [from_offset, from_velocity] =
        Ephemeris<ICRS>::Anchor::Conversion(anchor, std::nullopt, t1);
    EXPECT_EQ(anchor.OffsetAt(t1), from_offset);
    EXPECT_EQ(anchor.velocity, from_velocity);
    auto const [to_offset, to_velocity] =
        Ephemeris<ICRS>::Anchor::Conversion(std::nullopt, anchor, t1);
    EXPECT_EQ(-anchor.OffsetAt(t1), to_offset);
    EXPECT_EQ(-anchor.velocity, to_velocity);
  }

  DiscreteTrajectory<ICRS> probe;
  EXPECT_OK(probe.Append(
      t0,
      DegreesOfFreedom<ICRS>(ICRS::origin + anchored_q₀, ICRS::unmoving)));
  EXPECT_OK(damped->FlowWithAdaptiveStep(
      &probe,
      Ephemeris<ICRS>::NoIntrinsicAcceleration,
      t0 + 10 * JulianYear,
      Ephemeris<ICRS>::AdaptiveStepParameters(
          EmbeddedExplicitRungeKuttaNyströmIntegrator<
              DormandالمكاوىPrince1986RKN434FM,
              Ephemeris<ICRS>::NewtonianMotionEquation>(),
          /*max_steps=*/std::numeric_limits<std::int64_t>::max(),
          /*length_integration_tolerance=*/1 * Metre,
          /*speed_integration_tolerance=*/1e-3 * Metre / Second),
      Ephemeris<ICRS>::unlimited_max_ephemeris_steps,
      {/*subsystem=*/0, anchor}));
  auto const& [final_time, final_degrees_of_freedom] = probe.back();
  EXPECT_EQ(final_time, t0 + 10 * JulianYear);
  EXPECT_EQ(final_degrees_of_freedom.velocity(), ICRS::unmoving);
  EXPECT_EQ(final_degrees_of_freedom.position(), ICRS::origin + anchored_q₀);

  // The failure mode the kernel term prevents: at the anchored coordinates,
  // interpreted without the anchor, the field is star A's — far from zero.
  EXPECT_THAT(damped->ComputeGravitationalAccelerationOnMasslessBody(
                  ICRS::origin + anchored_q₀, t0, /*subsystem=*/0).Norm(),
              Gt(1 * Metre / Pow<2>(Second)));
}

// Two vessels homed to DIFFERENT subsystems meet in the void — the canonical
// inter-star rendezvous.  `placement_conversion` composes the subsystem and
// anchor terms on the sector lattice, where the void-scale cells cancel
// exactly, and collapses once at the magnitude of the small result: the slow
// approach is resolved to sub-mm.  Composing `subsystem_conversion` with a
// separately collapsed `Anchor::Conversion` — the pre-fix path, and the
// failure this test is first against — materializes the ~4e16 m inter-star
// distance in each term, so the approach is snapped to its ~8 m ULP grid.
TEST_F(InterstellarPrecisionTest, CrossSubsystemPlacementConversion) {
  Instant const t0;
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});

  // A station homed to subsystem 0 and a visitor homed to subsystem 1, five
  // metres apart in the void midway between the systems, the visitor closing
  // at half a metre per second.  Each anchor offset is relative to the local
  // origin of its own subsystem, so the two offsets differ by the full
  // inter-star distance.
  Velocity<ICRS> const cruise({3e4 * Metre / Second,
                               0 * Metre / Second,
                               0 * Metre / Second});
  Velocity<ICRS> const approach({-0.5 * Metre / Second,
                                 0 * Metre / Second,
                                 0 * Metre / Second});
  Ephemeris<ICRS>::Anchor const station_anchor{
      .offset = SectorDisplacement<ICRS>::Split(
          Displacement<ICRS>({2e16 * Metre, 0 * Metre, 0 * Metre})),
      .velocity = cruise,
      .epoch = t0};
  Ephemeris<ICRS>::Anchor const visitor_anchor{
      .offset = SectorDisplacement<ICRS>::Split(
          Displacement<ICRS>({(2e16 + 5 - 4e16) * Metre,
                              0 * Metre,
                              0 * Metre})),
      .velocity = cruise + approach,
      .epoch = t0};
  Ephemeris<ICRS>::SubsystemPlacement const station_placement(
      /*subsystem=*/0, station_anchor);
  Ephemeris<ICRS>::SubsystemPlacement const visitor_placement(
      /*subsystem=*/1, visitor_anchor);

  auto const placement_at = [&](Instant const& t) {
    return ephemeris->placement_conversion(
        visitor_placement, station_placement, t).first;
  };
  auto const pre_fix_at = [&](Instant const& t) {
    return ephemeris->subsystem_conversion(/*s1=*/1, /*s2=*/0, t) +
           Ephemeris<ICRS>::Anchor::Conversion(
               visitor_anchor, station_anchor, t).first;
  };

  // Co-located placements difference to a small displacement, not to a
  // void-scale artifact.
  Displacement<ICRS> const placement₀ = placement_at(t0);
  EXPECT_THAT(placement₀.Norm(), Lt(1 * Kilo(Metre)));

  Displacement<ICRS> const pre_fix₀ = pre_fix_at(t0);
  Length max_placement_error;
  Length max_pre_fix_error;
  for (int minute = 30; minute <= 360; minute += 30) {
    Instant const t = t0 + minute * Minute;
    Displacement<ICRS> const true_change = approach * (t - t0);
    max_placement_error =
        std::max(max_placement_error,
                 (placement_at(t) - placement₀ - true_change).Norm());
    max_pre_fix_error =
        std::max(max_pre_fix_error,
                 (pre_fix_at(t) - pre_fix₀ - true_change).Norm());
  }
  // The composed lattice: cells cancel, the collapse happens at the magnitude
  // of the approach.
  EXPECT_THAT(max_placement_error, Lt(1 * Milli(Metre)));
  // The pre-fix composition rounds each term at the ULP of the inter-star
  // distance.
  EXPECT_THAT(max_pre_fix_error, Gt(1 * Metre));
}

// With the far field damped, cross-system pairs of massive bodies are damped
// too: the backbone of each system evolves as if it were isolated.  This
// removes a physically meaningless secular drift of each system as a whole
// toward the remote one (immeasurable in-system: it cancels in any relative
// quantity) and leaves the in-system dynamics unaffected.
TEST_F(InterstellarPrecisionTest, FarFieldDampedBackbone) {
  Instant const t0;
  Acceleration const far_field_damping_floor =
      1e-12 * Metre / Pow<2>(Second);
  auto const control = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  auto const damped = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1},
                                    far_field_damping_floor);
  Instant const t_final = t0 + 100 * JulianYear;
  EXPECT_OK(control->Prolong(t_final));
  EXPECT_OK(damped->Prolong(t_final));

  auto const star_a = damped->bodies()[0];
  auto const planet_a = damped->bodies()[1];
  auto const control_star_a = control->bodies()[0];
  auto const control_planet_a = control->bodies()[1];

  // At the initial time the damping removes exactly the pull of the remote
  // system on star A, (μ_star + μ_planet) / (4 × 10¹⁶ m)² ≈ 2.5 × 10⁻¹⁹ m/s².
  Vector<Acceleration, ICRS> const removed_pull =
      control->ComputeGravitationalAccelerationOnMassiveBody(control_star_a,
                                                             t0) -
      damped->ComputeGravitationalAccelerationOnMassiveBody(star_a, t0);
  EXPECT_THAT(removed_pull.Norm(), Gt(2e-19 * Metre / Pow<2>(Second)));
  EXPECT_THAT(removed_pull.Norm(), Lt(3e-19 * Metre / Pow<2>(Second)));

  // Over a century the undamped system A as a whole falls toward the remote
  // system by ½ a t² ≈ 1.2 m; the damped one does not.
  Length const secular_drift =
      ((control->trajectory(control_star_a)->EvaluatePosition(t_final) -
        control->trajectory(control_star_a)->EvaluatePosition(t0)) -
       (damped->trajectory(star_a)->EvaluatePosition(t_final) -
        damped->trajectory(star_a)->EvaluatePosition(t0))).Norm();
  LOG(INFO) << "Secular drift of the undamped system over a century: "
            << secular_drift;
  EXPECT_THAT(secular_drift, Gt(1.0 * Metre));
  EXPECT_THAT(secular_drift, Lt(1.5 * Metre));

  // The in-system relative dynamics is unaffected: over the century the
  // star A–planet A separation of the damped ephemeris tracks the control
  // well below the fitting tolerance.
  Length max_separation_difference;
  for (int i = 0; i <= 1000; ++i) {
    Instant const t = t0 + i * (t_final - t0) / 1000;
    Length const control_separation =
        (control->trajectory(control_planet_a)->EvaluatePosition(t) -
         control->trajectory(control_star_a)->EvaluatePosition(t)).Norm();
    Length const damped_separation =
        (damped->trajectory(planet_a)->EvaluatePosition(t) -
         damped->trajectory(star_a)->EvaluatePosition(t)).Norm();
    max_separation_difference =
        std::max(max_separation_difference,
                 Abs(damped_separation - control_separation));
  }
  LOG(INFO) << "Max in-system separation difference over a century: "
            << max_separation_difference;
  EXPECT_THAT(max_separation_difference, Lt(1 * Milli(Metre)));
}

// The partition generalizes beyond two subsystems: with three stars in three
// subsystems, each star's local field is felt only within its own system and
// the far field of the other two is damped to exactly zero, while the
// inter-subsystem geometry (the pairwise distances) is reproduced exactly.
TEST_F(InterstellarPrecisionTest, ThreeSubsystems) {
  Instant const t0;
  Acceleration const far_field_damping_floor =
      1e-12 * Metre / Pow<2>(Second);
  // Three stars on the x axis, 4 × 10¹⁶ m apart, each its own subsystem.
  Length const spacing = 4.0e16 * Metre;
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  std::vector<DegreesOfFreedom<ICRS>> initial_state;
  for (int i = 0; i < 3; ++i) {
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star " + std::to_string(i), μ_star)));
    initial_state.emplace_back(
        ICRS::origin +
            Displacement<ICRS>({i * spacing, 0 * Metre, 0 * Metre}),
        ICRS::unmoving);
  }
  auto const ephemeris = make_not_null_unique<Ephemeris<ICRS>>(
      std::move(bodies),
      initial_state,
      t0,
      Ephemeris<ICRS>::AccuracyParameters(
          /*fitting_tolerance=*/0.1 * Milli(Metre),
          /*geopotential_tolerance=*/0x1p-24),
      Ephemeris<ICRS>::FixedStepParameters(
          SymmetricLinearMultistepIntegrator<
              QuinlanTremaine1990Order12,
              Ephemeris<ICRS>::NewtonianMotionEquation>(),
          /*step=*/1 * JulianYear),
      /*subsystems=*/std::vector<int>{0, 1, 2},
      far_field_damping_floor);
  EXPECT_OK(ephemeris->Prolong(t0 + 1 * JulianYear));

  // Each star sits in a distinct subsystem.
  EXPECT_EQ(0, ephemeris->subsystem_of_body(ephemeris->bodies()[0]));
  EXPECT_EQ(1, ephemeris->subsystem_of_body(ephemeris->bodies()[1]));
  EXPECT_EQ(2, ephemeris->subsystem_of_body(ephemeris->bodies()[2]));

  // A probe just inside each star's inner threshold (expressed relative to that
  // star's own subsystem origin) feels that star's unattenuated pull and
  // nothing from the other two.
  Length const near = 3.0e12 * Metre;  // < inner threshold (~6.6 × 10¹² m).
  for (int s = 0; s < 3; ++s) {
    Position<ICRS> const probe =
        ICRS::origin + Displacement<ICRS>({0 * Metre, near, 0 * Metre});
    Vector<Acceleration, ICRS> const a =
        ephemeris->ComputeGravitationalAccelerationOnMasslessBody(
            probe, t0, /*subsystem=*/s);
    // The pull points toward the local star (−y here) and matches the bare
    // two-body value; the other two stars contribute exactly nothing.
    Vector<Acceleration, ICRS> const expected(
        {0 * Metre / Pow<2>(Second),
         -μ_star / Pow<2>(near),
         0 * Metre / Pow<2>(Second)});
    EXPECT_THAT(RelativeError(expected, a), Lt(1e-6));
  }

  // The inter-subsystem geometry is exact: star 2 is 2·spacing from star 0.
  Displacement<ICRS> const star0 =
      ephemeris->trajectory(ephemeris->bodies()[0])->EvaluatePosition(t0) -
      ICRS::origin;
  Displacement<ICRS> const conversion_2_to_0 =
      ephemeris->subsystem_conversion(/*s1=*/2, /*s2=*/0, t0);
  Displacement<ICRS> const star2_in_0 =
      (ephemeris->trajectory(ephemeris->bodies()[2])->EvaluatePosition(t0) -
       ICRS::origin) +
      conversion_2_to_0;
  EXPECT_THAT(AbsoluteError(2 * spacing, (star2_in_0 - star0).Norm()),
              Lt(1 * Metre));
}

// Checks that a reference frame whose primary and secondary belong to
// different subsystems reconstitutes their true relative geometry: the X axis
// of a frame defined by star A and star B points from star A to star B.
TEST_F(InterstellarPrecisionTest, CrossSubsystemReferenceFrame) {
  using Navigation = Frame<serialization::Frame::TestTag,
                           Arbitrary,
                           Handedness::Right,
                           serialization::Frame::TEST>;
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  Instant const t = Instant() + Period();
  EXPECT_OK(ephemeris->Prolong(t));

  auto const star_a = ephemeris->bodies()[0];
  auto const star_b = ephemeris->bodies()[2];
  BodyCentredBodyDirectionReferenceFrame<ICRS, Navigation> const frame(
      ephemeris.get(), star_a, star_b);
  EXPECT_EQ(0, frame.subsystem());

  // The degrees of freedom of star B, represented relative to the local
  // origin of the subsystem of star A.
  Displacement<ICRS> const conversion =
      ephemeris->subsystem_conversion(/*s1=*/1, /*s2=*/0, t);
  DegreesOfFreedom<ICRS> const star_b_degrees_of_freedom = {
      ephemeris->trajectory(star_b)->EvaluatePosition(t) + conversion,
      ephemeris->trajectory(star_b)->EvaluateVelocity(t)};
  Length const separation =
      (star_b_degrees_of_freedom.position() -
       ephemeris->trajectory(star_a)->EvaluatePosition(t)).Norm();

  RigidMotion<ICRS, Navigation> const to_this_frame =
      frame.ToThisFrameAtTime(t);
  Displacement<Navigation> const star_b_in_frame =
      to_this_frame(star_b_degrees_of_freedom).position() - Navigation::origin;
  EXPECT_THAT(AbsoluteError(separation, star_b_in_frame.coordinates().x),
              Lt(100 * Metre));
  EXPECT_THAT(Abs(star_b_in_frame.coordinates().y), Lt(100 * Metre));
  EXPECT_THAT(Abs(star_b_in_frame.coordinates().z), Lt(100 * Metre));
}

TEST_F(InterstellarPrecisionTest, Serialization) {
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_OK(ephemeris->Prolong(Instant() + Period()));

  serialization::Ephemeris message;
  ephemeris->WriteToMessage(&message);
  EXPECT_EQ(4, message.body_subsystem_size());
  EXPECT_EQ(2, message.subsystem_origin_offset_size());
  EXPECT_EQ(2, message.subsystem_barycentre_size());
  EXPECT_TRUE(message.has_subsystem_barycentre_time());

  auto const ephemeris_read = Ephemeris<ICRS>::ReadFromMessage(
      /*desired_t_min=*/InfiniteFuture, message);
  // After deserialization, the client must prolong as needed.
  EXPECT_OK(ephemeris_read->Prolong(ephemeris->t_max()));

  EXPECT_EQ(ephemeris->t_min(), ephemeris_read->t_min());
  for (Instant t = ephemeris->t_min(); t <= ephemeris->t_max();
       t += (ephemeris->t_max() - ephemeris->t_min()) / 100) {
    EXPECT_OK(ephemeris_read->Prolong(t));
    for (int b = 0; b < 4; ++b) {
      EXPECT_EQ(ephemeris->trajectory(ephemeris->bodies()[b])
                    ->EvaluateDegreesOfFreedom(t),
                ephemeris_read->trajectory(ephemeris_read->bodies()[b])
                    ->EvaluateDegreesOfFreedom(t));
    }
  }

  // The barycentric data survive the round-trip bit-exactly, including when
  // extrapolated far beyond the integrated range.
  Instant const t_extrapolated = ephemeris->t_min() + 100 * JulianYear;
  for (int s = 0; s < 2; ++s) {
    EXPECT_EQ(ephemeris->subsystem_gravitational_parameter(s),
              ephemeris_read->subsystem_gravitational_parameter(s));
    EXPECT_EQ(ephemeris->subsystem_barycentre(s, t_extrapolated),
              ephemeris_read->subsystem_barycentre(s, t_extrapolated));
    EXPECT_EQ(ephemeris->subsystem_barycentre_velocity(s),
              ephemeris_read->subsystem_barycentre_velocity(s));
  }

  serialization::Ephemeris second_message;
  ephemeris_read->WriteToMessage(&second_message);
  EXPECT_THAT(message, EqualsProto(second_message));
}

// With a relative cutoff, a pair is damped to zero where its interaction
// falls below ε times the characteristic central acceleration of the body it
// acts on.  A feather-weight perturber that the absolute floor keeps is cut,
// and nothing else about the victim's acceleration changes.
TEST_F(InterstellarPrecisionTest, RelativeFarFieldCutoff) {
  Instant const t0;
  Acceleration const floor = 1e-12 * Metre / Pow<2>(Second);
  double const ε = 1e-8;

  // A star, a tightly-bound victim, and a perturber whose pull on the victim,
  // ~1e-11 m/s², sits between the absolute floor and ε times the victim's
  // central acceleration of ~4e-2 m/s².
  auto const make_three_body_ephemeris =
      [&floor, &t0](double const epsilon, bool const with_perturber) {
        std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
        bodies.push_back(make_not_null_unique<MassiveBody>(
            MassiveBody::Parameters("star", μ_star)));
        bodies.push_back(make_not_null_unique<MassiveBody>(
            MassiveBody::Parameters("victim",
                                    1 * Pow<3>(Metre) / Pow<2>(Second))));
        std::vector<DegreesOfFreedom<ICRS>> initial_state{
            {ICRS::origin, ICRS::unmoving},
            {ICRS::origin +
                 Displacement<ICRS>({1e8 * Metre, 0 * Metre, 0 * Metre}),
             Velocity<ICRS>({0 * Metre / Second,
                             Sqrt(μ_star / (1e8 * Metre)),
                             0 * Metre / Second})}};
        if (with_perturber) {
          bodies.push_back(make_not_null_unique<MassiveBody>(
              MassiveBody::Parameters("perturber", 1e-6 * μ_star)));
          initial_state.push_back(
              {ICRS::origin +
                   Displacement<ICRS>({6.4e9 * Metre, 0 * Metre, 0 * Metre}),
               Velocity<ICRS>({0 * Metre / Second,
                               Sqrt(μ_star / (6.4e9 * Metre)),
                               0 * Metre / Second})});
        }
        return make_not_null_unique<Ephemeris<ICRS>>(
            std::move(bodies),
            initial_state,
            t0,
            Ephemeris<ICRS>::AccuracyParameters(
                /*fitting_tolerance=*/0.1 * Milli(Metre),
                /*geopotential_tolerance=*/0x1p-24),
            Ephemeris<ICRS>::FixedStepParameters(
                SymmetricLinearMultistepIntegrator<
                    QuinlanTremaine1990Order12,
                    Ephemeris<ICRS>::NewtonianMotionEquation>(),
                /*step=*/10 * Second),
            /*subsystems=*/std::vector<int>{},
            floor,
            epsilon);
      };

  auto const relative = make_three_body_ephemeris(ε, /*with_perturber=*/true);
  auto const absolute = make_three_body_ephemeris(0, /*with_perturber=*/true);
  auto const two_body = make_three_body_ephemeris(ε, /*with_perturber=*/false);
  EXPECT_OK(relative->Prolong(t0 + 1000 * Second));
  EXPECT_OK(absolute->Prolong(t0 + 1000 * Second));
  EXPECT_OK(two_body->Prolong(t0 + 1000 * Second));
  auto const& victim = relative->bodies()[1];

  // The relative cutoff removes the perturber and only the perturber: the
  // acceleration matches an ephemeris where it does not exist, to within the
  // Chebyshev fitting noise of the positions, which is two decades below the
  // perturber's ~1e-11 m/s² pull.
  EXPECT_THAT((two_body->ComputeGravitationalAccelerationOnMassiveBody(
                   two_body->bodies()[1], t0) -
               relative->ComputeGravitationalAccelerationOnMassiveBody(
                   victim, t0)).Norm(),
              Lt(1e-13 * Metre / Pow<2>(Second)));

  // The absolute floor keeps the perturber, whose pull is above it.
  Vector<Acceleration, ICRS> const perturber_term =
      absolute->ComputeGravitationalAccelerationOnMassiveBody(
          absolute->bodies()[1], t0) -
      relative->ComputeGravitationalAccelerationOnMassiveBody(victim, t0);
  EXPECT_THAT(perturber_term.Norm(),
              AllOf(Gt(9.5e-12 * Metre / Pow<2>(Second)),
                    Lt(1.05e-11 * Metre / Pow<2>(Second))));
}

// The characteristic acceleration is taken at the apoapsis, and the sigmoid
// shell of a relative pair threshold behaves like the absolute one: a victim
// on an eccentric orbit meets a perturber sitting on its shell, whose damped
// pull carries the factor σ − σ′r ≈ 2.3 at mid-shell.  Holding the
// contribution in a band around that value pins three things at once: the
// apoapsis (a periapsis-based threshold cuts this pair entirely), the max()
// symmetrization (the min side would also cut it), and the σ path of the
// relative table.
TEST_F(InterstellarPrecisionTest, RelativeFarFieldShellAndApoapsis) {
  Instant const t0;
  Acceleration const floor = 1e-12 * Metre / Pow<2>(Second);
  double const ε = 1e-8;

  // The victim starts at the apoapsis of an e = 0.36 orbit (tangential
  // velocity at 0.8 of circular), so its apoapsis distance is exactly 1e8 m
  // and its characteristic acceleration is exactly μ_star / (1e8 m)².  The
  // perturber sits 7.06e8 m away, at x ≈ 0.56 of the pair's sigmoid shell,
  // which spans [3.33e8 m, 1e9 m].
  auto const make = [&floor, &t0](double const epsilon,
                                  bool const with_perturber) {
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star", μ_star)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("victim",
                                1 * Pow<3>(Metre) / Pow<2>(Second))));
    std::vector<DegreesOfFreedom<ICRS>> initial_state{
        {ICRS::origin, ICRS::unmoving},
        {ICRS::origin +
             Displacement<ICRS>({1e8 * Metre, 0 * Metre, 0 * Metre}),
         Velocity<ICRS>({0 * Metre / Second,
                         0.8 * Sqrt(μ_star / (1e8 * Metre)),
                         0 * Metre / Second})}};
    if (with_perturber) {
      bodies.push_back(make_not_null_unique<MassiveBody>(
          MassiveBody::Parameters("perturber", 1e-6 * μ_star)));
      initial_state.push_back(
          {ICRS::origin +
               Displacement<ICRS>({8.06e8 * Metre, 0 * Metre, 0 * Metre}),
           Velocity<ICRS>({0 * Metre / Second,
                           Sqrt(μ_star / (8.06e8 * Metre)),
                           0 * Metre / Second})});
    }
    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            /*step=*/10 * Second),
        /*subsystems=*/std::vector<int>{},
        floor,
        epsilon);
  };

  auto const relative = make(ε, /*with_perturber=*/true);
  auto const two_body = make(ε, /*with_perturber=*/false);
  EXPECT_OK(relative->Prolong(t0 + 1000 * Second));
  EXPECT_OK(two_body->Prolong(t0 + 1000 * Second));

  // The raw pull is 8.0e-10 m/s²; on the shell it contributes
  // (σ − σ′r) ≈ 2.3 times that.  Cutting it entirely (periapsis or min())
  // gives zero and keeping it undamped gives 1.0 — both outside the band.
  Acceleration const raw_pull = 1e-6 * μ_star / Pow<2>(7.06e8 * Metre);
  EXPECT_THAT((relative->ComputeGravitationalAccelerationOnMassiveBody(
                   relative->bodies()[1], t0) -
               two_body->ComputeGravitationalAccelerationOnMassiveBody(
                   two_body->bodies()[1], t0)).Norm(),
              AllOf(Gt(1.5 * raw_pull), Lt(3.0 * raw_pull)));
}

// A body unbound from its dominant attractor has no apoapsis to characterize
// it and is exempted from the relative cutoff: it keeps feeling everything,
// however far, where the absolute floor would have cut it loose.
TEST_F(InterstellarPrecisionTest, RelativeFarFieldUnboundExemption) {
  Instant const t0;
  Acceleration const floor = 1e-12 * Metre / Pow<2>(Second);

  auto const make_comet_ephemeris = [&floor, &t0](double const epsilon) {
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star", μ_star)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("comet",
                                1 * Pow<3>(Metre) / Pow<2>(Second))));
    // Radially outbound at more than twice the escape velocity.
    std::vector<DegreesOfFreedom<ICRS>> const initial_state{
        {ICRS::origin, ICRS::unmoving},
        {ICRS::origin +
             Displacement<ICRS>({1e15 * Metre, 0 * Metre, 0 * Metre}),
         Velocity<ICRS>(
             {2 * Metre / Second, 0 * Metre / Second, 0 * Metre / Second})}};
    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            /*step=*/1 * JulianYear),
        /*subsystems=*/std::vector<int>{},
        floor,
        epsilon);
  };

  // The absolute floor cuts the star loose from the comet at this distance.
  auto const absolute = make_comet_ephemeris(0);
  EXPECT_OK(absolute->Prolong(t0 + 20 * JulianYear));
  EXPECT_EQ(absolute->ComputeGravitationalAccelerationOnMassiveBody(
                absolute->bodies()[1], t0),
            (Vector<Acceleration, ICRS>{}));

  // The relative cutoff exempts the unbound comet.
  auto const relative = make_comet_ephemeris(1e-8);
  EXPECT_OK(relative->Prolong(t0 + 20 * JulianYear));
  EXPECT_THAT(relative->ComputeGravitationalAccelerationOnMassiveBody(
                  relative->bodies()[1], t0).Norm(),
              AllOf(Gt(3e-16 * Metre / Pow<2>(Second)),
                    Lt(5e-16 * Metre / Pow<2>(Second))));
}

// The relative cutoff survives serialization: ε and the characteristic
// accelerations are written rather than recomputed, so that the physics does
// not drift across a save-load cycle.  The star is oblate and listed second,
// so that the internal body order differs from the input order and the
// characteristic accelerations — all decades apart — pin the permutation on
// the wire.
TEST_F(InterstellarPrecisionTest, RelativeFarFieldSerialization) {
  Instant const t0;
  Acceleration const floor = 1e-12 * Metre / Pow<2>(Second);

  // A planet at 1e8 m, an oblate star, and an outer body whose ~1e-11 m/s²
  // pull on the planet the absolute floor keeps and the relative cutoff
  // discards, so that ε being honoured after a reload is visible in the
  // trajectories.
  auto const make = [&floor, &t0](double const epsilon) {
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("planet",
                                1 * Pow<3>(Metre) / Pow<2>(Second))));
    bodies.push_back(make_not_null_unique<OblateBody<ICRS>>(
        μ_star,
        RotatingBody<ICRS>::Parameters(/*mean_radius=*/1e7 * Metre,
                                       /*reference_angle=*/1 * Radian,
                                       /*reference_instant=*/t0,
                                       /*angular_frequency=*/
                                       1e-3 * Radian / Second,
                                       /*right_ascension_of_pole=*/0 * Radian,
                                       /*declination_of_pole=*/π / 2 * Radian),
        OblateBody<ICRS>::Parameters(/*j2=*/1e-6,
                                     /*reference_radius=*/1e7 * Metre)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("outer", 1e-6 * μ_star)));
    std::vector<DegreesOfFreedom<ICRS>> const initial_state{
        {ICRS::origin +
             Displacement<ICRS>({1e8 * Metre, 0 * Metre, 0 * Metre}),
         Velocity<ICRS>({0 * Metre / Second,
                         Sqrt(μ_star / (1e8 * Metre)),
                         0 * Metre / Second})},
        {ICRS::origin, ICRS::unmoving},
        {ICRS::origin +
             Displacement<ICRS>({6.4e9 * Metre, 0 * Metre, 0 * Metre}),
         Velocity<ICRS>({0 * Metre / Second,
                         Sqrt(μ_star / (6.4e9 * Metre)),
                         0 * Metre / Second})}};
    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            /*step=*/10 * Second),
        /*subsystems=*/std::vector<int>{},
        floor,
        epsilon);
  };

  Instant const t_final = t0 + 200'000 * Second;
  auto const ephemeris = make(/*epsilon=*/1e-8);
  EXPECT_OK(ephemeris->Prolong(t_final));

  serialization::Ephemeris message;
  ephemeris->WriteToMessage(&message);
  EXPECT_TRUE(message.has_far_field_damping_epsilon());
  EXPECT_EQ(3, message.characteristic_acceleration_size());

  // The characteristic accelerations are in input order — planet held by the
  // star at ~4e-2, star held by the outer body at ~1e-11, outer held by the
  // star at ~1e-5 — and a permutation of any two of them lands decades away.
  auto const characteristic_acceleration = [&message](int const i) {
    return Acceleration::ReadFromMessage(
        message.characteristic_acceleration(i));
  };
  Acceleration const unit = 1 * Metre / Pow<2>(Second);
  EXPECT_THAT(characteristic_acceleration(0),
              AllOf(Gt(1e-2 * unit), Lt(1e-1 * unit)));
  EXPECT_THAT(characteristic_acceleration(1),
              AllOf(Gt(1e-12 * unit), Lt(1e-10 * unit)));
  EXPECT_THAT(characteristic_acceleration(2),
              AllOf(Gt(1e-6 * unit), Lt(1e-4 * unit)));

  auto const ephemeris_read = Ephemeris<ICRS>::ReadFromMessage(
      /*desired_t_min=*/InfiniteFuture, message);
  EXPECT_OK(ephemeris_read->Prolong(ephemeris->t_max()));
  for (Instant t = ephemeris->t_min(); t <= ephemeris->t_max();
       t += (ephemeris->t_max() - ephemeris->t_min()) / 100) {
    EXPECT_OK(ephemeris_read->Prolong(t));
    for (int b = 0; b < 3; ++b) {
      EXPECT_EQ(ephemeris->trajectory(ephemeris->bodies()[b])
                    ->EvaluateDegreesOfFreedom(t),
                ephemeris_read->trajectory(ephemeris_read->bodies()[b])
                    ->EvaluateDegreesOfFreedom(t));
    }
  }

  serialization::Ephemeris second_message;
  ephemeris_read->WriteToMessage(&second_message);
  EXPECT_THAT(message, EqualsProto(second_message));

  // And ε is not decorative: the same roster with the absolute criterion
  // keeps the outer body's pull on the planet and drifts away measurably.
  auto const absolute = make(/*epsilon=*/0);
  EXPECT_OK(absolute->Prolong(t_final));
  EXPECT_THAT((ephemeris->trajectory(ephemeris->bodies()[0])
                   ->EvaluatePosition(t_final) -
               absolute->trajectory(absolute->bodies()[0])
                   ->EvaluatePosition(t_final)).Norm(),
              Gt(1 * Milli(Metre)));
}

// A/B measurement harness for damping the far field within a single subsystem:
// the real solar system is integrated with and without damping from the same
// initial state, and the divergence is compared against the integrator's own
// step error, measured the same way by halving the step.  A floor whose
// divergence sits below that yardstick discards only what the integrator does
// not know.  This is a measurement, not a regression test; run it manually
// with --gtest_also_run_disabled_tests.
TEST_F(InterstellarPrecisionTest, DISABLED_FarFieldDampingSweep) {
  SolarSystem<ICRS> const solar_system(
      SOLUTION_DIR / "astronomy" / "sol_gravity_model.proto.txt",
      SOLUTION_DIR / "astronomy" /
          "sol_initial_state_jd_2433282_500000000.proto.txt");
  Instant const t0 = solar_system.epoch();
  Instant const t_final = t0 + 10 * JulianYear;

  auto const make_ephemeris =
      [&solar_system](Time const& step,
                      Acceleration const& far_field_damping_floor,
                      double const far_field_damping_epsilon) {
        std::vector<DegreesOfFreedom<ICRS>> initial_state;
        for (std::string const& name : solar_system.names()) {
          initial_state.push_back(solar_system.degrees_of_freedom(name));
        }
        return make_not_null_unique<Ephemeris<ICRS>>(
            solar_system.MakeAllMassiveBodies(),
            initial_state,
            solar_system.epoch(),
            Ephemeris<ICRS>::AccuracyParameters(
                /*fitting_tolerance=*/1 * Milli(Metre),
                /*geopotential_tolerance=*/0x1p-24),
            Ephemeris<ICRS>::FixedStepParameters(
                SymplecticRungeKuttaNyströmIntegrator<
                    BlanesMoan2002SRKN14A,
                    Ephemeris<ICRS>::NewtonianMotionEquation>(),
                step),
            /*subsystems=*/std::vector<int>{},
            far_field_damping_floor,
            far_field_damping_epsilon);
      };

  // The largest position divergence of each body between two ephemerides.
  auto const divergences = [t0, t_final](Ephemeris<ICRS> const& a,
                                         Ephemeris<ICRS> const& b) {
    std::vector<Length> max(a.bodies().size());
    for (Instant t = t0; t <= t_final; t += 30 * Day) {
      for (int i = 0; i < a.bodies().size(); ++i) {
        max[i] = std::max(
            max[i],
            (a.trajectory(a.bodies()[i])->EvaluatePosition(t) -
             b.trajectory(b.bodies()[i])->EvaluatePosition(t)).Norm());
      }
    }
    return max;
  };

  Time const step = 35 * Minute;
  auto const control = make_ephemeris(step, Acceleration{}, 0);
  EXPECT_OK(control->Prolong(t_final));

  // The yardstick is per body: comparing global maxima would let the step
  // error of a fast inner moon launder the damping's harm to a slow outer
  // body, whose own step error is far smaller.
  auto const halved = make_ephemeris(step / 2, Acceleration{}, 0);
  EXPECT_OK(halved->Prolong(t_final));
  std::vector<Length> const yardstick = divergences(*control, *halved);
  for (int i = 0; i < yardstick.size(); ++i) {
    LOG(ERROR) << "Yardstick over 10 a: " << yardstick[i] << " at "
               << control->bodies()[i]->name();
  }

  for (Acceleration const floor : {1e-16 * Metre / Pow<2>(Second),
                                   1e-15 * Metre / Pow<2>(Second),
                                   1e-14 * Metre / Pow<2>(Second),
                                   1e-13 * Metre / Pow<2>(Second),
                                   1e-12 * Metre / Pow<2>(Second)}) {
    auto const damped = make_ephemeris(step, floor, 0);
    EXPECT_OK(damped->Prolong(t_final));
    std::vector<Length> const divergence = divergences(*control, *damped);
    double worst_ratio = 0;
    int worst = 0;
    for (int i = 0; i < divergence.size(); ++i) {
      if (yardstick[i] > Length{} &&
          divergence[i] / yardstick[i] > worst_ratio) {
        worst_ratio = divergence[i] / yardstick[i];
        worst = i;
      }
      LOG(ERROR) << "Floor " << floor << ": "
                 << control->bodies()[i]->name() << " diverges "
                 << divergence[i] << " = "
                 << (yardstick[i] > Length{} ? divergence[i] / yardstick[i]
                                             : 0.0) << " yardsticks";
    }
    LOG(ERROR) << "Floor " << floor << ": worst "
               << control->bodies()[worst]->name() << " at " << worst_ratio
               << " yardsticks";
  }

  // The same measurement for the relative criterion; the floor only carries
  // the massless damping and the pair cutoff is ε.
  for (double const ε : {1e-6, 1e-8, 1e-10, 1e-12}) {
    auto const damped =
        make_ephemeris(step, 1e-12 * Metre / Pow<2>(Second), ε);
    EXPECT_OK(damped->Prolong(t_final));
    std::vector<Length> const divergence = divergences(*control, *damped);
    double worst_ratio = 0;
    int worst = 0;
    for (int i = 0; i < divergence.size(); ++i) {
      if (yardstick[i] > Length{} &&
          divergence[i] / yardstick[i] > worst_ratio) {
        worst_ratio = divergence[i] / yardstick[i];
        worst = i;
      }
      LOG(ERROR) << "Epsilon " << ε << ": "
                 << control->bodies()[i]->name() << " diverges "
                 << divergence[i] << " = "
                 << (yardstick[i] > Length{} ? divergence[i] / yardstick[i]
                                             : 0.0) << " yardsticks";
    }
    LOG(ERROR) << "Epsilon " << ε << ": worst "
               << control->bodies()[worst]->name() << " at " << worst_ratio
               << " yardsticks";
  }
}

#endif

}  // namespace astronomy
}  // namespace principia
