#include "ksp_plugin/orbit_analyser.hpp"

#include <memory>
#include <string>
#include <vector>

#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "astronomy/epoch.hpp"
#include "astronomy/frames.hpp"
#include "astronomy/orbit_recurrence.hpp"
#include "astronomy/standard_product_3.hpp"
#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integrators/methods.hpp"
#include "integrators/symmetric_linear_multistep_integrator.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin/integrators.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/body_surface_reference_frame.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "physics/rotating_body.hpp"
#include "physics/sector.hpp"
#include "physics/solar_system.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/approximate_quantity.hpp"
#include "testing_utilities/is_near.hpp"
#include "testing_utilities/matchers.hpp"  // 🧙 For EXPECT_OK.

namespace principia {
namespace ksp_plugin {

using ::testing::AllOf;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::IsNull;
using ::testing::Lt;
using ::testing::Optional;
using ::testing::Property;
using namespace principia::astronomy::_epoch;
using namespace principia::astronomy::_frames;
using namespace principia::astronomy::_orbit_recurrence;
using namespace principia::astronomy::_standard_product_3;
using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_integrators;
using namespace principia::ksp_plugin::_orbit_analyser;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_body_surface_reference_frame;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_rotating_body;
using namespace principia::physics::_sector;
using namespace principia::physics::_solar_system;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_approximate_quantity;
using namespace principia::testing_utilities::_is_near;

class OrbitAnalyserTest : public testing::Test {
 protected:
  OrbitAnalyserTest()
      : earth_1957_(RemoveAllButEarth(SolarSystem<Barycentric>(
            SOLUTION_DIR / "astronomy" / "sol_gravity_model.proto.txt",
            SOLUTION_DIR / "astronomy" /
                "sol_initial_state_jd_2436116_311504629.proto.txt",
            /*ignore_frame=*/true))),
        // As there is only one body left in this ephemeris, integration is
        // exact up to roundoff error regardless of the step size.  Use a long
        // step so that we do not waste time computing the ephemeris.
        ephemeris_(earth_1957_.MakeEphemeris(
            /*accuracy_parameters=*/{/*fitting_tolerance=*/1 * Milli(Metre),
                                     /*geopotential_tolerance=*/0x1p-24},
            Ephemeris<Barycentric>::FixedStepParameters(
                SymmetricLinearMultistepIntegrator<
                    QuinlanTremaine1990Order12,
                    Ephemeris<Barycentric>::NewtonianMotionEquation>(),
                /*step=*/1 * JulianYear))),
        earth_(*earth_1957_.rotating_body(*ephemeris_, "Earth")),
        itrs_(ephemeris_.get(), &earth_),
        topex_poséidon_(SOLUTION_DIR / "astronomy" / "standard_product_3" /
                            "grgtop03.b97344.e97348.D_S.sp3",
                        StandardProduct3::Dialect::GRGS) {}

  SolarSystem<Barycentric> earth_1957_;
  not_null<std::unique_ptr<Ephemeris<Barycentric>>> ephemeris_;
  RotatingBody<Barycentric> const& earth_;
  BodySurfaceReferenceFrame<Barycentric, ITRS> itrs_;
  StandardProduct3 topex_poséidon_;

 private:
  static SolarSystem<Barycentric> RemoveAllButEarth(
      SolarSystem<Barycentric> solar_system) {
    std::vector<std::string> const names = solar_system.names();
    for (auto const& name : names) {
      if (name != "Earth") {
        solar_system.RemoveMassiveBody(name);
      }
    }
    return solar_system;
  }
};

TEST_F(OrbitAnalyserTest, TinyAnalysis) {
  OrbitAnalyser analyser(ephemeris_.get(), DefaultHistoryParameters());
  EXPECT_THAT(analyser.analysis(), IsNull());
  EXPECT_THAT(analyser.progress_of_next_analysis(), Eq(0));
  auto const& arc =
      *topex_poséidon_.orbit(
          {StandardProduct3::SatelliteGroup::General, 1}).front();
  EXPECT_OK(ephemeris_->Prolong(arc.begin()->time));
  analyser.RequestAnalysis(
      {.first_time = arc.begin()->time,
       .first_degrees_of_freedom = itrs_.FromThisFrameAtTime(arc.begin()->time)(
           arc.begin()->degrees_of_freedom),
       .mission_duration = Abs(J2000 - arc.begin()->time) * 0x1p-45});
  do {
    absl::SleepFor(absl::Milliseconds(10));
    analyser.RefreshAnalysis();
  } while (analyser.analysis() == nullptr);
}


TEST_F(OrbitAnalyserTest, TOPEXPoséidon) {
  OrbitAnalyser analyser(ephemeris_.get(), DefaultHistoryParameters());
  EXPECT_THAT(analyser.analysis(), IsNull());
  EXPECT_THAT(analyser.progress_of_next_analysis(), Eq(0));
  auto const& arc =
      *topex_poséidon_.orbit(
          {StandardProduct3::SatelliteGroup::General, 1}).front();
  EXPECT_OK(ephemeris_->Prolong(arc.begin()->time));
  analyser.RequestAnalysis(
      {.first_time = arc.begin()->time,
       .first_degrees_of_freedom = itrs_.FromThisFrameAtTime(arc.begin()->time)(
           arc.begin()->degrees_of_freedom),
       .mission_duration = 3 * Hour});
  while (analyser.progress_of_next_analysis() != 1) {
    absl::SleepFor(absl::Milliseconds(10));
  }
  // Since `progress_of_next_analysis` only tracks the integration, not the
  // analysis, we have no guarantee that an analysis is available immediately.
  do {
    absl::SleepFor(absl::Milliseconds(10));
    analyser.RefreshAnalysis();
  } while (analyser.analysis() == nullptr);
  EXPECT_THAT(analyser.analysis()
                  ->elements()
                  ->mean_semimajor_axis_interval()
                  .midpoint(),
              IsNear(7714_(1) * Kilo(Metre)));
  EXPECT_THAT(analyser.analysis()->recurrence(),
              Optional(AllOf(Property(&OrbitRecurrence::νₒ, 13),
                             Property(&OrbitRecurrence::Dᴛₒ, -3),
                             Property(&OrbitRecurrence::Cᴛₒ, 10))));
  EXPECT_THAT(analyser.analysis()
                      ->equatorial_crossings()
                      ->longitudes_reduced_to_pass(1)
                      .measure() *
                  TerrestrialEquatorialRadius / Radian,
              IsNear(93_(1) * Metre));
  // [13; -1; 3] is the subcycle of [13; -3; 10].
  analyser.analysis()->SetRecurrence({13, -1, 3});
  EXPECT_THAT(analyser.analysis()
                      ->equatorial_crossings()
                      ->longitudes_reduced_to_pass(1)
                      .measure() *
                  TerrestrialEquatorialRadius / Radian,
              IsNear(8211_(1) * Metre));
  // Back to the auto-detected recurrence.
  analyser.analysis()->ResetRecurrence();
  EXPECT_THAT(analyser.analysis()->recurrence(),
              Optional(AllOf(Property(&OrbitRecurrence::νₒ, 13),
                             Property(&OrbitRecurrence::Dᴛₒ, -3),
                             Property(&OrbitRecurrence::Cᴛₒ, 10))));
}

// A two-subsystem interstellar ephemeris with a damped far field: the stock
// Sun ("home star", subsystem 0) and TRAPPIST-1 ("remote star", subsystem 1,
// 40 light-years out, coasting on its stock placeholder orbit), from the
// NearStars test configuration.
not_null<std::unique_ptr<Ephemeris<Barycentric>>> MakeInterstellarEphemeris(
    Instant const& t0,
    GravitationalParameter const& μ_home,
    GravitationalParameter const& μ_star,
    Displacement<Barycentric> const& to_star) {
  Velocity<Barycentric> const star_velocity(
      {0 * Metre / Second,
       Sqrt(μ_home / to_star.Norm()),
       0 * Metre / Second});
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  bodies.push_back(make_not_null_unique<RotatingBody<Barycentric>>(
      MassiveBody::Parameters("home star", μ_home),
      RotatingBody<Barycentric>::Parameters(
          /*mean_radius=*/2.616e8 * Metre,
          /*reference_angle=*/0 * Degree,
          /*reference_instant=*/t0,
          /*angular_frequency=*/1e-6 * Radian / Second,
          /*right_ascension_of_pole=*/0 * Degree,
          /*declination_of_pole=*/90 * Degree)));
  bodies.push_back(make_not_null_unique<RotatingBody<Barycentric>>(
      MassiveBody::Parameters("remote star", μ_star),
      RotatingBody<Barycentric>::Parameters(
          /*mean_radius=*/8.4e7 * Metre,
          /*reference_angle=*/0 * Degree,
          /*reference_instant=*/t0,
          /*angular_frequency=*/2.2e-5 * Radian / Second,
          /*right_ascension_of_pole=*/0 * Degree,
          /*declination_of_pole=*/90 * Degree)));
  std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
      {Barycentric::origin, Velocity<Barycentric>{}},
      {Barycentric::origin + to_star, star_velocity}};
  std::vector<int> const subsystems{0, 1};
  return make_not_null_unique<Ephemeris<Barycentric>>(
      std::move(bodies),
      initial_state,
      t0,
      Ephemeris<Barycentric>::AccuracyParameters(
          /*fitting_tolerance=*/1 * Milli(Metre),
          /*geopotential_tolerance=*/0x1p-24),
      Ephemeris<Barycentric>::FixedStepParameters(
          SymmetricLinearMultistepIntegrator<
              QuinlanTremaine1990Order12,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          /*step=*/1 * Day),
      subsystems,
      /*far_field_damping_floor=*/1e-12 * Metre / Pow<2>(Second));
}

// Reproduces the in-game analysis of a vessel in a low circular orbit around a
// star 40 light-years out — the interstellar regime: two subsystems, damped
// far field, the vessel represented relative to the remote star's local
// origin.  The gravitational parameters, separation, orbit radius,
// eccentricity and inclination replicate the session where the in-game
// analyser emitted negative, wildly varying anomalistic and nodal periods on
// every refresh; a near-circular orbit's periods must instead be positive and
// close to the Keplerian period.
TEST_F(OrbitAnalyserTest, InterstellarStarOrbit) {
  Instant const t0;
  GravitationalParameter const μ_home =
      1.1723328e18 * si::Unit<GravitationalParameter>;
  GravitationalParameter const μ_star =
      1.1917577113616e19 * si::Unit<GravitationalParameter>;
  auto const ephemeris = MakeInterstellarEphemeris(
      t0,
      μ_home,
      μ_star,
      Displacement<Barycentric>({3.848e17 * Metre, 0 * Metre, 0 * Metre}));

  // The vessel, in the remote star's subsystem representation (the star sits
  // at its own local origin): the in-game orbit — a 1.5829e9 m circular orbit,
  // e ≈ 5.4e-5, i ≈ 0.034°.
  Length const orbit_radius = 1.5829e9 * Metre;
  Speed const v_circular = Sqrt(μ_star / orbit_radius);
  Angle const inclination = 0.034 * Degree;
  DegreesOfFreedom<Barycentric> const first_degrees_of_freedom{
      Barycentric::origin +
          Displacement<Barycentric>({orbit_radius, 0 * Metre, 0 * Metre}),
      Velocity<Barycentric>({0 * Metre / Second,
                             v_circular * (1 + 2.7e-5) * Cos(inclination),
                             v_circular * (1 + 2.7e-5) * Sin(inclination)})};
  Time const keplerian_period =
      2 * π * Sqrt(Pow<3>(orbit_radius) / μ_star);

  EXPECT_OK(ephemeris->Prolong(t0 + 1 * Second));
  OrbitAnalyser analyser(ephemeris.get(), DefaultHistoryParameters());
  analyser.RequestAnalysis(
      {.first_time = t0,
       .first_degrees_of_freedom = first_degrees_of_freedom,
       .subsystem = 1,
       .mission_duration = 2 * keplerian_period});
  for (int i = 0; i < 30'000 && analyser.analysis() == nullptr; ++i) {
    absl::SleepFor(absl::Milliseconds(10));
    analyser.RefreshAnalysis();
  }
  ASSERT_THAT(analyser.analysis(), ::testing::NotNull())
      << "the analysis never completed";

  auto const& elements = analyser.analysis()->elements();
  ASSERT_TRUE(elements.has_value());
  EXPECT_THAT(elements->mean_semimajor_axis_interval().midpoint(),
              IsNear(1.5829_(1) * Giga(Metre)));
  // The in-game failure mode: negative, refresh-to-refresh-flapping periods.
  EXPECT_THAT(elements->anomalistic_period(),
              AllOf(Gt(0.99 * keplerian_period), Lt(1.01 * keplerian_period)));
  EXPECT_THAT(elements->nodal_period(),
              AllOf(Gt(0.99 * keplerian_period), Lt(1.01 * keplerian_period)));
}

}  // namespace ksp_plugin
}  // namespace principia
