#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "astronomy/frames.hpp"
#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "gtest/gtest.h"
#include "integrators/embedded_explicit_runge_kutta_nyström_integrator.hpp"
#include "integrators/methods.hpp"
#include "integrators/symmetric_linear_multistep_integrator.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/apsides.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "serialization/physics.pb.h"
#include "testing_utilities/matchers.hpp"  // 🧙 For EXPECT_OK.
#include "testing_utilities/numerics.hpp"

namespace principia {
namespace astronomy {

using ::testing::Gt;
using ::testing::Lt;
using namespace principia::astronomy::_frames;
using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_embedded_explicit_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_apsides;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
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

  // Constructs an ephemeris containing the two systems.  If `subsystems` is
  // empty all the positions are represented in a single frame; otherwise each
  // system is represented relative to its own local origin.
  static not_null<std::unique_ptr<Ephemeris<ICRS>>> MakeEphemeris(
      std::vector<int> const& subsystems) {
    Instant const t0;
    Displacement<ICRS> const to_remote_system(
        {4.0e16 * Metre, 0 * Metre, 0 * Metre});

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
        {ICRS::origin + to_remote_system, star_velocity},
        {ICRS::origin + to_remote_system + star_to_planet, planet_velocity}};

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
    Displacement<ICRS> const to_remote_system(
        {4.0e16 * Metre, 0 * Metre, 0 * Metre});

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
        /*subsystem=*/0));
    EXPECT_OK(ephemeris->FlowWithAdaptiveStep(
        &probe_b,
        Ephemeris<ICRS>::NoIntrinsicAcceleration,
        t_final,
        adaptive_parameters,
        Ephemeris<ICRS>::unlimited_max_ephemeris_steps,
        /*subsystem=*/remote_subsystem));

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
      control->bodies(), control->EvaluateAllDegreesOfFreedom(t));
  auto const jerks_with_subsystems =
      with_subsystems->ComputeGravitationalJerkOnMassiveBodies(
          with_subsystems->bodies(),
          with_subsystems->EvaluateAllDegreesOfFreedom(t));
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
  auto const far = Displacement<ICRS>({4e16 * Metre, 0 * Metre, 0 * Metre});
  auto const near = Displacement<ICRS>({1e13 * Metre, 0 * Metre, 0 * Metre});

  // A single system, however far from the origin, is not partitioned.
  EXPECT_THAT(
      ClusterSubsystems<ICRS>({ICRS::origin, ICRS::origin + near}, threshold),
      ::testing::IsEmpty());
  EXPECT_THAT(ClusterSubsystems<ICRS>(
                  {ICRS::origin + far, ICRS::origin + far + near}, threshold),
              ::testing::IsEmpty());

  // Two systems, interleaved: the subsystems are numbered by order of first
  // appearance.
  EXPECT_THAT(ClusterSubsystems<ICRS>({ICRS::origin,
                                       ICRS::origin + far,
                                       ICRS::origin + near,
                                       ICRS::origin + far + near},
                                      threshold),
              ::testing::ElementsAre(0, 1, 0, 1));

  // Single-linkage chaining: a body bridging two otherwise-distant groups
  // merges them.
  EXPECT_THAT(ClusterSubsystems<ICRS>({ICRS::origin,
                                       ICRS::origin + near,
                                       ICRS::origin + 2 * near,
                                       ICRS::origin + far},
                                      threshold),
              ::testing::ElementsAre(0, 0, 0, 1));

  // The subsystem of each body is exposed by the ephemeris; note that the
  // subsystems are given per input body but exposed per (reordered) body.
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_EQ(0, ephemeris->subsystem_of_body(ephemeris->bodies()[0]));
  EXPECT_EQ(0, ephemeris->subsystem_of_body(ephemeris->bodies()[1]));
  EXPECT_EQ(1, ephemeris->subsystem_of_body(ephemeris->bodies()[2]));
  EXPECT_EQ(1, ephemeris->subsystem_of_body(ephemeris->bodies()[3]));
}

// Checks that the apsides of a cross-subsystem pair are those of the true
// relative trajectory.  The two systems are congruent and synchronized, so the
// separation between planet A and star B oscillates exactly between
// `to_remote_system.Norm() ∓ orbit_radius`.
TEST_F(InterstellarPrecisionTest, Apsides) {
  Displacement<ICRS> const to_remote_system(
      {4.0e16 * Metre, 0 * Metre, 0 * Metre});
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

  auto const& offset = ephemeris->inter_subsystem_offset(/*s1=*/0, /*s2=*/1);
  auto const separation = [&offset](DegreesOfFreedom<ICRS> const& dof1,
                                    DegreesOfFreedom<ICRS> const& dof2) {
    return (offset.value +
            (offset.error + (dof1.position() - dof2.position()))).Norm();
  };
  for (auto const& [t, degrees_of_freedom] : periapsides1) {
    EXPECT_THAT(AbsoluteError(to_remote_system.Norm() - orbit_radius,
                              separation(degrees_of_freedom,
                                         periapsides2.at(t))),
                Lt(100 * Metre));
  }
  for (auto const& [t, degrees_of_freedom] : apoapsides1) {
    EXPECT_THAT(AbsoluteError(to_remote_system.Norm() + orbit_radius,
                              separation(degrees_of_freedom,
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

TEST_F(InterstellarPrecisionTest, Serialization) {
  auto const ephemeris = MakeEphemeris(/*subsystems=*/{0, 0, 1, 1});
  EXPECT_OK(ephemeris->Prolong(Instant() + Period()));

  serialization::Ephemeris message;
  ephemeris->WriteToMessage(&message);
  EXPECT_EQ(4, message.body_subsystem_size());
  EXPECT_EQ(2, message.subsystem_origin_offset_size());

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

  serialization::Ephemeris second_message;
  ephemeris_read->WriteToMessage(&second_message);
  EXPECT_THAT(message, EqualsProto(second_message));
}

#endif

}  // namespace astronomy
}  // namespace principia
