#include <algorithm>
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
#include "integrators/methods.hpp"
#include "integrators/symmetric_linear_multistep_integrator.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
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
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
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

#endif

}  // namespace astronomy
}  // namespace principia
