#include "ksp_plugin/plugin.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "astronomy/frames.hpp"
#include "astronomy/time_scales.hpp"
#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/permutation.hpp"
#include "geometry/rotation.hpp"
#include "geometry/space.hpp"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "integrators/embedded_explicit_runge_kutta_nyström_integrator.hpp"
#include "integrators/methods.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin/identification.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "physics/solar_system.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/approximate_quantity.hpp"
#include "testing_utilities/is_near.hpp"
#include "testing_utilities/numerics_matchers.hpp"
#include "testing_utilities/solar_system_factory.hpp"

namespace principia {
namespace ksp_plugin {

using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::Le;
using ::testing::Lt;
using ::testing::SizeIs;
using namespace principia::astronomy::_frames;
using namespace principia::astronomy::_time_scales;
using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_permutation;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_space;
using namespace principia::integrators::_embedded_explicit_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_methods;
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_identification;
using namespace principia::ksp_plugin::_plugin;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_solar_system;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_approximate_quantity;
using namespace principia::testing_utilities::_is_near;
using namespace principia::testing_utilities::_numerics_matchers;
using namespace principia::testing_utilities::_solar_system_factory;

namespace {

constexpr PartId part_id = 789;
GUID const vessel_guid = "123-456";
constexpr char part_name[] = "Picard's desk";
constexpr char vessel_name[] = "NCC-1701-D";

}  // namespace

class PluginIntegrationTestWithoutPlugin : public testing::Test {
 protected:
  PluginIntegrationTestWithoutPlugin()
      : solar_system_(
            SolarSystemFactory::AtСпутник1Launch(
                SolarSystemFactory::Accuracy::MinorAndMajorBodies)),
        initial_time_("JD2451545.0625"),
        planetarium_rotation_(1 * Radian) {
    satellite_initial_displacement_ =
        Displacement<AliceSun>({3111.0 * Kilo(Metre),
                                4400.0 * Kilo(Metre),
                                3810.0 * Kilo(Metre)});
    auto const tangent =
        satellite_initial_displacement_ * Bivector<double, AliceSun>({1, 2, 3});
    Vector<double, AliceSun> const unit_tangent = Normalize(tangent);
    EXPECT_THAT(
        InnerProduct(unit_tangent,
                     satellite_initial_displacement_ /
                         satellite_initial_displacement_.Norm()),
        Eq(0));
    // This yields a circular orbit.
    satellite_initial_velocity_ =
        Sqrt(solar_system_->gravitational_parameter(
                 SolarSystemFactory::name(SolarSystemFactory::Earth)) /
                 satellite_initial_displacement_.Norm()) * unit_tangent;
  }

  not_null<std::unique_ptr<SolarSystem<ICRS>>> solar_system_;
  std::string initial_time_;
  Angle planetarium_rotation_;

  // These initial conditions will yield a low circular orbit around Earth.
  Displacement<AliceSun> satellite_initial_displacement_;
  Velocity<AliceSun> satellite_initial_velocity_;
};

class PluginIntegrationTest : public PluginIntegrationTestWithoutPlugin {
 protected:
  PluginIntegrationTest()
      : plugin_(std::make_unique<Plugin>(initial_time_,
                                         initial_time_,
                                         planetarium_rotation_)) {}

  void InsertAllSolarSystemBodies() {
    for (int index = SolarSystemFactory::Sun;
         index <= SolarSystemFactory::LastBody;
         ++index) {
      std::optional<Index> const parent_index =
          index == SolarSystemFactory::Sun
              ? std::nullopt
              : std::make_optional(SolarSystemFactory::parent(index));
      plugin_->InsertCelestialAbsoluteCartesian(
          index,
          parent_index,
          solar_system_->gravity_model_message(
              SolarSystemFactory::name(index)),
          solar_system_->cartesian_initial_state_message(
              SolarSystemFactory::name(index)));
    }
  }

  std::unique_ptr<Plugin> plugin_;
};

TEST_F(PluginIntegrationTest, AdvanceTimeWithCelestialsOnly) {
  InsertAllSolarSystemBodies();
  plugin_->EndInitialization();
#if defined(_DEBUG)
  Time const δt = 2 * Second;
#else
  Time const δt = 0.02 * Second;
#endif
  Angle const planetarium_rotation = 42 * Radian;
  // We step for long enough that we will find a new segment.
  Instant const initial_time = ParseTT(initial_time_);
  Instant t = initial_time;
  for (t += δt; t < initial_time + 10 * 45 * Minute; t += δt) {
    plugin_->AdvanceTime(t, planetarium_rotation);
  }
  EXPECT_THAT(plugin_->CelestialFromParent(SolarSystemFactory::Earth)
                  .displacement()
                  .Norm(),
              RelativeErrorFrom(1 * AstronomicalUnit, Lt(0.01)));
  serialization::Plugin plugin_message;
  plugin_->WriteToMessage(&plugin_message);
  plugin_ = nullptr;
  plugin_ = Plugin::ReadFromMessage(plugin_message);
  // Having saved and loaded, we compute a new segment again, this probably
  // exercises apocalypse-type bugs.
  for (; t < initial_time + 20 * 45 * Minute; t += δt) {
    plugin_->AdvanceTime(t, planetarium_rotation);
  }
  EXPECT_THAT(plugin_->CelestialFromParent(SolarSystemFactory::Earth)
                  .displacement()
                  .Norm(),
              RelativeErrorFrom(1 * AstronomicalUnit, Lt(0.01)));
}

TEST_F(PluginIntegrationTest, BodyCentredNonrotatingNavigationIntegration) {
  InsertAllSolarSystemBodies();
  plugin_->EndInitialization();

  bool inserted;
  plugin_->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              SolarSystemFactory::Earth,
                              /*loaded=*/false,
                              inserted);
  plugin_->InsertUnloadedPart(
      part_id,
      part_name,
      vessel_guid,
      RelativeDegreesOfFreedom<AliceSun>(satellite_initial_displacement_,
                                         satellite_initial_velocity_));
  plugin_->PrepareToReportCollisions();
  plugin_->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  plugin_->renderer().SetPlottingFrame(
      plugin_->NewBodyCentredNonRotatingNavigationFrame(
          SolarSystemFactory::Earth));
  // We'll check that our orbit is rendered as circular (actually, we only check
  // that it is rendered within a thin spherical shell around the Earth).
  Length perigee = std::numeric_limits<double>::infinity() * Metre;
  Length apogee = -std::numeric_limits<double>::infinity() * Metre;
  Permutation<AliceSun, World> const alice_sun_to_world =
      Permutation<AliceSun, World>(OddPermutation::XZY);
  Time const δt_long = 10 * Minute;
#if defined(_DEBUG)
  Time const δt_short = 1 * Minute;
#else
  Time const δt_short = 0.02 * Second;
#endif
  Instant const initial_time = ParseTT(initial_time_);
  Instant t = initial_time + δt_short;
  // Exercise #267 by having small time steps at the beginning of the trajectory
  // that are not synchronized with those of the Earth.
  for (; t < initial_time + δt_long; t += δt_short) {
    plugin_->AdvanceTime(
        t,
        1 * Radian / Pow<2>(Minute) * Pow<2>(t - initial_time));
    plugin_->InsertOrKeepVessel(vessel_guid,
                                vessel_name,
                                SolarSystemFactory::Earth,
                                /*loaded=*/false,
                                inserted);
    VesselSet collided_vessels;
    plugin_->CatchUpLaggingVessels(collided_vessels);
  }
  for (; t < initial_time + 12 * Hour; t += δt_long) {
    plugin_->AdvanceTime(
        t,
        1 * Radian / Pow<2>(Minute) * Pow<2>(t - initial_time));
    VesselSet collided_vessels;
    plugin_->CatchUpLaggingVessels(collided_vessels);
    plugin_->InsertOrKeepVessel(vessel_guid,
                                vessel_name,
                                SolarSystemFactory::Earth,
                                /*loaded=*/false,
                                inserted);
    // We give the sun an arbitrary nonzero velocity in `World`.
    Position<World> const sun_world_position =
        World::origin + Velocity<World>(
            { 0.1 * AstronomicalUnit / Hour,
             -1.0 * AstronomicalUnit / Hour,
              0.0 * AstronomicalUnit / Hour}) * (t - initial_time);
    auto const& vessel = *plugin_->GetVessel(vessel_guid);
    auto const& trajectory = vessel.trajectory();
    auto const psychohistory = vessel.psychohistory();
    auto const rendered_trajectory =
        plugin_->renderer().RenderBarycentricTrajectoryInWorld(
            plugin_->CurrentTime(),
            trajectory.begin(),
            psychohistory->end(),
            sun_world_position,
            plugin_->PlanetariumRotation());
    EXPECT_THAT(rendered_trajectory, SizeIs(AllOf(Ge(6), Le(4262))));
    Position<World> const earth_world_position =
        sun_world_position + alice_sun_to_world(plugin_->CelestialFromParent(
                                 SolarSystemFactory::Earth).displacement());
    for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
      Length const distance =
          (degrees_of_freedom.position() - earth_world_position).Norm();
      perigee = std::min(perigee, distance);
      apogee = std::max(apogee, distance);
    }
    EXPECT_THAT(Abs(apogee - perigee), Lt(3 * Metre));
  }
}

TEST_F(PluginIntegrationTest, BarycentricRotatingNavigationIntegration) {
  InsertAllSolarSystemBodies();
  plugin_->EndInitialization();

  bool inserted;
  plugin_->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              SolarSystemFactory::Earth,
                              /*loaded=*/false,
                              inserted);
  // A vessel at the Lagrange point L₅.
  RelativeDegreesOfFreedom<AliceSun> const from_the_earth_to_the_moon =
      plugin_->CelestialFromParent(SolarSystemFactory::Moon);
  Displacement<AliceSun> const from_the_earth_to_l5 =
      from_the_earth_to_the_moon.displacement() / 2 -
          Normalize(from_the_earth_to_the_moon.velocity()) *
              from_the_earth_to_the_moon.displacement().Norm() * Sqrt(3) / 2;
  Velocity<AliceSun> const initial_velocity =
      Rotation<AliceSun, AliceSun>(
          π / 3 * Radian,
          Wedge(from_the_earth_to_the_moon.velocity(),
                from_the_earth_to_the_moon.displacement()))(
              from_the_earth_to_the_moon.velocity());
  plugin_->InsertUnloadedPart(part_id,
                              part_name,
                              vessel_guid,
                              {from_the_earth_to_l5, initial_velocity});
  plugin_->PrepareToReportCollisions();
  plugin_->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  plugin_->renderer().SetPlottingFrame(
      plugin_->NewBarycentricRotatingNavigationFrame(SolarSystemFactory::Earth,
                                                     SolarSystemFactory::Moon));
  Permutation<AliceSun, World> const alice_sun_to_world =
      Permutation<AliceSun, World>(OddPermutation::XZY);
  Time const δt_long = 1 * Hour;
#if defined(_DEBUG)
  Time const duration = 12 * Hour;
  Time const δt_short = 20 * Second;
#else
  Time const duration = 20 * Day;
  Time const δt_short = 0.02 * Second;
#endif
  Instant const initial_time = ParseTT(initial_time_);
  Instant t = initial_time + δt_short;
  // Exercise #267 by having small time steps at the beginning of the trajectory
  // that are not synchronized with those of the Earth and Moon.
  for (; t < initial_time + δt_long; t += δt_short) {
    plugin_->AdvanceTime(
        t,
        1 * Radian / Pow<2>(Minute) * Pow<2>(t - initial_time));
    VesselSet collided_vessels;
    plugin_->CatchUpLaggingVessels(collided_vessels);
    plugin_->InsertOrKeepVessel(vessel_guid,
                                vessel_name,
                                SolarSystemFactory::Earth,
                                /*loaded=*/false,
                                inserted);
  }
  for (; t < initial_time + duration; t += δt_long) {
    plugin_->AdvanceTime(
        t,
        1 * Radian / Pow<2>(Minute) * Pow<2>(t - initial_time));
    VesselSet collided_vessels;
    plugin_->CatchUpLaggingVessels(collided_vessels);
    plugin_->InsertOrKeepVessel(vessel_guid,
                                vessel_name,
                                SolarSystemFactory::Earth,
                                /*loaded=*/false,
                                inserted);
  }
  plugin_->AdvanceTime(t,
                       1 * Radian / Pow<2>(Minute) * Pow<2>(t - initial_time));
  VesselSet collided_vessels;
  plugin_->CatchUpLaggingVessels(collided_vessels);
  plugin_->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              SolarSystemFactory::Earth,
                              /*loaded=*/false,
                              inserted);
  // We give the sun an arbitrary nonzero velocity in `World`.
  Position<World> const sun_world_position =
      World::origin + Velocity<World>(
          { 0.1 * AstronomicalUnit / Hour,
           -1.0 * AstronomicalUnit / Hour,
            0.0 * AstronomicalUnit / Hour}) * (t - initial_time);
  auto const& vessel = *plugin_->GetVessel(vessel_guid);
  auto const& trajectory = vessel.trajectory();
  auto const psychohistory = vessel.psychohistory();
  auto const rendered_trajectory =
      plugin_->renderer().RenderBarycentricTrajectoryInWorld(
          plugin_->CurrentTime(),
          trajectory.begin(),
          psychohistory->end(),
          sun_world_position,
          plugin_->PlanetariumRotation());
  EXPECT_THAT(rendered_trajectory, SizeIs(AnyOf(3, 89)));
  Position<World> const earth_world_position =
      sun_world_position +
      alice_sun_to_world(plugin_->CelestialFromParent(SolarSystemFactory::Earth)
                             .displacement());
  Position<World> const moon_world_position =
      earth_world_position +
      alice_sun_to_world(plugin_->CelestialFromParent(SolarSystemFactory::Moon)
                             .displacement());
  Length const earth_moon = (moon_world_position - earth_world_position).Norm();
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    Position<World> const position = degrees_of_freedom.position();
    Length const satellite_earth = (position - earth_world_position).Norm();
    Length const satellite_moon = (position - moon_world_position).Norm();
    EXPECT_THAT(satellite_earth, RelativeErrorFrom(earth_moon, Lt(0.0907)));
    EXPECT_THAT(satellite_moon, RelativeErrorFrom(earth_moon, Lt(0.131)));
    EXPECT_THAT(satellite_earth, RelativeErrorFrom(satellite_moon, Lt(0.148)));
  }
  // Check that there are no spikes in the rendered trajectory, i.e., that three
  // consecutive points form a sufficiently flat triangle.  This tests issue
  // #256.
  auto it0 = rendered_trajectory.begin();
  CHECK(it0 != rendered_trajectory.end());
  auto it1 = it0;
  ++it1;
  CHECK(it1 != rendered_trajectory.end());
  auto it2 = it1;
  ++it2;
  while (it2 != rendered_trajectory.end()) {
    EXPECT_THAT((it0->degrees_of_freedom.position() -
                 it2->degrees_of_freedom.position())
                    .Norm(),
                Gt(((it0->degrees_of_freedom.position() -
                     it1->degrees_of_freedom.position())
                        .Norm() +
                    (it1->degrees_of_freedom.position() -
                     it2->degrees_of_freedom.position())
                        .Norm()) /
                   1.5))
        << it0->time;
    ++it0;
    ++it1;
    ++it2;
  }
}

// Checks that we correctly predict a full circular orbit around a massive body
// with unit gravitational parameter at unit distance.  Since predictions are
// only computed on `AdvanceTime()`, we advance time by a small amount.
TEST_F(PluginIntegrationTestWithoutPlugin, Prediction) {
  Index const celestial = 0;
  Plugin plugin("JD2451545.0", "JD2451545.0", 0 * Radian);
  serialization::GravityModel::Body gravity_model;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      R"(name                    : "Celestial"
         gravitational_parameter : "1 m^3/s^2"
         reference_instant       : "JD2451545.0"
         mean_radius             : "1 m"
         axis_right_ascension    : "0 deg"
         axis_declination        : "90 deg"
         reference_angle         : "1 rad"
         angular_frequency       : "1 rad/s")",
      &gravity_model));
  serialization::InitialState::Keplerian::Body initial_state;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      R"(name : "Celestial")",
      &initial_state));
  plugin.InsertCelestialJacobiKeplerian(
      celestial,
      /*parent_index=*/std::nullopt,
      gravity_model,
      initial_state);
  plugin.EndInitialization();

  bool inserted;
  plugin.InsertOrKeepVessel(vessel_guid,
                            vessel_name,
                            celestial,
                            /*loaded=*/false,
                            inserted);
  plugin.InsertUnloadedPart(
      part_id,
      part_name,
      vessel_guid,
      {Displacement<AliceSun>({1 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>(
           {0 * Metre / Second, 1 * Metre / Second, 0 * Metre / Second})});
  plugin.PrepareToReportCollisions();
  plugin.FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  plugin.renderer().SetPlottingFrame(
      plugin.NewBodyCentredNonRotatingNavigationFrame(celestial));
  Ephemeris<Barycentric>::AdaptiveStepParameters const adaptive_step_parameters(
      EmbeddedExplicitRungeKuttaNyströmIntegrator<
          DormandالمكاوىPrince1986RKN434FM,
          Ephemeris<Barycentric>::NewtonianMotionEquation>(),
      /*max_steps=*/14,
      /*length_integration_tolerance=*/1 * Milli(Metre),
      /*speed_integration_tolerance=*/1 * Milli(Metre) / Second);
  plugin.SetPredictionAdaptiveStepParameters(vessel_guid,
                                             adaptive_step_parameters);
  plugin.AdvanceTime(Instant() + 1e-10 * Second, 0 * Radian);

  // Polling for the integration to happen.
  do {
    plugin.UpdatePrediction({vessel_guid});
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  } while (plugin.GetVessel(vessel_guid)->prediction()->size() != 15);

  auto const prediction = plugin.GetVessel(vessel_guid)->prediction();
  auto const rendered_prediction =
      plugin.renderer().RenderBarycentricTrajectoryInWorld(
          plugin.CurrentTime(),
          prediction->begin(),
          prediction->end(),
          World::origin,
          plugin.PlanetariumRotation());
  EXPECT_EQ(15, rendered_prediction.size());
  int index = 0;
  for (auto it = rendered_prediction.begin();
       it != rendered_prediction.end();
       ++it, ++index) {
    auto const& position = it->degrees_of_freedom.position();
    EXPECT_THAT((position - World::origin).Norm(),
                AbsoluteErrorFrom(1 * Metre, Lt(0.5 * Milli(Metre))));
    if (index >= 5) {
      EXPECT_THAT((position - World::origin).Norm(),
                  AbsoluteErrorFrom(1 * Metre, Gt(0.1 * Milli(Metre))));
    }
  }
  EXPECT_THAT(
      rendered_prediction.back().degrees_of_freedom.position(),
      AbsoluteErrorFrom(Displacement<World>({1 * Metre, 0 * Metre, 0 * Metre}) +
                            World::origin,
                        IsNear(29_(1) * Milli(Metre))));
}

// An end-to-end test of the partitioning of a multi-star system into
// subsystems: a vessel coasts from one star system to another one
// 4 × 10¹⁶ m away, and its representation gets rebased on the way.  The
// parent-relative degrees of freedom and the rendered trajectory must remain
// consistent throughout, and the state must survive a save/load cycle.
TEST_F(PluginIntegrationTestWithoutPlugin, InterstellarRebase) {
  Index const star_a = 0;
  Index const star_b = 1;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name                    : "star A"
           gravitational_parameter : "1.3e20 m^3/s^2"
           reference_instant       : "JD2451545.0"
           mean_radius             : "1e6 m"
           axis_right_ascension    : "0 deg"
           axis_declination        : "90 deg"
           reference_angle         : "0 rad"
           angular_frequency       : "1 rad/s")",
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name : "star A"
           x    : "0 m"
           y    : "0 m"
           z    : "0 m"
           vx   : "0 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star_a,
                                            /*parent_index=*/std::nullopt,
                                            gravity_model,
                                            initial_state);
  }
  {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name                    : "star B"
           gravitational_parameter : "1.3e20 m^3/s^2"
           reference_instant       : "JD2451545.0"
           mean_radius             : "1e6 m"
           axis_right_ascension    : "0 deg"
           axis_declination        : "90 deg"
           reference_angle         : "0 rad"
           angular_frequency       : "1 rad/s")",
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name : "star B"
           x    : "4e16 m"
           y    : "0 m"
           z    : "0 m"
           vx   : "0 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star_b,
                                            /*parent_index=*/star_a,
                                            gravity_model,
                                            initial_state);
  }
  plugin->EndInitialization();

  // The system was partitioned into two subsystems.
  EXPECT_NE(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetCelestial(star_b).subsystem());

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid,
                            vessel_name,
                            star_a,
                            /*loaded=*/false,
                            inserted);
  Vector<double, AliceSun> const to_star_b =
      Normalize(plugin->CelestialFromParent(star_b).displacement());
  Speed const v = 1e12 * Metre / Second;
  plugin->InsertUnloadedPart(
      part_id,
      part_name,
      vessel_guid,
      {(1e9 * Metre) * to_star_b, v * to_star_b});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  auto const& vessel = *plugin->GetVessel(vessel_guid);
  int const initial_subsystem = vessel.subsystem();
  EXPECT_EQ(plugin->GetCelestial(star_a).subsystem(), initial_subsystem);

  // Coast past the midpoint of the void; the rebase happens there.
  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 24'000 * Second;
  int rebases = 0;
  int previous_subsystem = initial_subsystem;
  std::optional<Displacement<AliceSun>> previous_displacement;
  for (Instant t = Instant() + δt; t <= t_final; t += δt) {
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              star_a,
                              /*loaded=*/false,
                              inserted);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);

    auto const from_parent = plugin->VesselFromParent(star_a, vessel_guid);
    if (previous_displacement.has_value()) {
      // The motion seen through `VesselFromParent` is continuous across the
      // rebase.
      EXPECT_THAT((from_parent.displacement() - *previous_displacement).Norm(),
                  RelativeErrorFrom(v * δt, Lt(1e-6)));
    }
    previous_displacement = from_parent.displacement();
    if (vessel.subsystem() != previous_subsystem) {
      previous_subsystem = vessel.subsystem();
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(), vessel.subsystem());

  // The distance from star B to its parent star A is unaffected by their
  // distinct representations.
  EXPECT_THAT(plugin->CelestialFromParent(star_b).displacement().Norm(),
              AbsoluteErrorFrom(4e16 * Metre, Lt(1 * Kilo(Metre))));

  // The rendered trajectory is consistent whether it is plotted in a frame
  // centred on the origin star or one centred on the destination star, and it
  // puts the vessel at its true distance from star A (the sun, anchored at
  // `World::origin`).
  Length const expected_distance = 1e9 * Metre + v * (t_final - Instant());
  auto const& trajectory = vessel.trajectory();
  auto const render_distance = [&](Index const centre) {
    plugin->renderer().SetPlottingFrame(
        plugin->NewBodyCentredNonRotatingNavigationFrame(centre));
    auto const rendered = plugin->renderer().RenderBarycentricTrajectoryInWorld(
        plugin->CurrentTime(),
        trajectory.begin(),
        trajectory.end(),
        World::origin,
        plugin->PlanetariumRotation(),
        vessel.subsystem());
    return (rendered.back().degrees_of_freedom.position() - World::origin)
        .Norm();
  };
  Length const rendered_around_a = render_distance(star_a);
  Length const rendered_around_b = render_distance(star_b);
  EXPECT_THAT(rendered_around_a,
              AbsoluteErrorFrom(expected_distance, Lt(1e6 * Metre)));
  EXPECT_THAT(rendered_around_b,
              AbsoluteErrorFrom(rendered_around_a, Lt(100 * Metre)));

  // Save and reload; the vessel keeps its representation and its state.  Only
  // one `Plugin` may exist at a time, so destroy it before reading.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  int const subsystem_before_save = vessel.subsystem();
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  auto const& vessel2 = *plugin2->GetVessel(vessel_guid);
  EXPECT_EQ(subsystem_before_save, vessel2.subsystem());
  EXPECT_THAT(
      plugin2->VesselFromParent(star_a, vessel_guid).displacement().Norm(),
      AbsoluteErrorFrom(previous_displacement->Norm(), Lt(100 * Metre)));

  // Keep flying after the reload.
  for (Instant t = t_final + δt; t <= t_final + 2 * δt; t += δt) {
    plugin2->AdvanceTime(t, 1 * Radian);
    plugin2->InsertOrKeepVessel(vessel_guid,
                                vessel_name,
                                star_a,
                                /*loaded=*/false,
                                inserted);
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
    auto const from_parent = plugin2->VesselFromParent(star_a, vessel_guid);
    EXPECT_THAT((from_parent.displacement() - *previous_displacement).Norm(),
                RelativeErrorFrom(v * δt, Lt(1e-6)));
    previous_displacement = from_parent.displacement();
  }
}

}  // namespace ksp_plugin
}  // namespace principia
