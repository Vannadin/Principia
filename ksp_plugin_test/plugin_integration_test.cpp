#include "ksp_plugin/plugin.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
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
#include "testing_utilities/almost_equals.hpp"
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
using namespace principia::ksp_plugin::_vessel;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_solar_system;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
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

// A vessel burning on rails: the burn accelerates it along the commanded
// direction with Циолковский's Δv and depletes the mass of its pile up, the
// prediction anticipates the burn, and a frame without a burn coasts.
TEST_F(PluginIntegrationTestWithoutPlugin, OnRailsBurn) {
  Index const star = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name                    : "star"
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
        R"(name : "star"
           x    : "0 m"
           y    : "0 m"
           z    : "0 m"
           vx   : "0 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star,
                                             /*parent_index=*/std::nullopt,
                                             gravity_model,
                                             initial_state);
  }
  plugin->EndInitialization();

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid,
                             vessel_name,
                             star,
                             /*loaded=*/false,
                             inserted);
  // Far enough from the star that gravity is negligible against the burn.
  plugin->InsertUnloadedPart(
      part_id,
      part_name,
      vessel_guid,
      {Displacement<AliceSun>({1e12 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  auto& vessel = *plugin->GetVessel(vessel_guid);

  Force const thrust = 1 * Newton;
  SpecificImpulse const specific_impulse = 1e4 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;
  Time const δt = 100 * Second;

  // Three frames of burning under warp; the game owns the mass bookkeeping,
  // handing the current mass to each catch-up.
  Mass const m0 = 1 * Kilogram;
  Mass m = m0;
  Velocity<AliceSun> const v0 =
      plugin->VesselFromParent(star, vessel_guid).velocity();
  Instant t;
  for (int i = 0; i < 3; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid,
                               vessel_name,
                               star,
                               /*loaded=*/false,
                               inserted);
    plugin->SetVesselOnRailsBurn(vessel_guid,
                                 thrust,
                                 specific_impulse,
                                 /*initial_mass=*/m,
                                 Vector<double, World>({0, 1, 0}),
                                 /*max_duration=*/1 * Hour);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    m -= δt * mass_flow;
  }
  Speed const Δv = specific_impulse * std::log(m0 / m);
  Velocity<AliceSun> const Δv_vector =
      plugin->VesselFromParent(star, vessel_guid).velocity() - v0;
  EXPECT_THAT(Δv_vector.Norm(), RelativeErrorFrom(Δv, Lt(1e-3)));
  // `World` and `AliceSun` differ by the XZY permutation (and the planetarium
  // rotation, which cancels between the burn conversion and `FromParent`):
  // the y direction commanded in `World` must come out along z.
  EXPECT_THAT(Δv_vector.coordinates().z, RelativeErrorFrom(Δv, Lt(1e-3)));

  // The prediction anticipates the burn continuing until its propellant runs
  // out.
  do {
    plugin->UpdatePrediction({vessel_guid});
    using namespace std::chrono_literals;
    std::this_thread::sleep_for(100ms);
  } while (vessel.prediction()->empty() ||
           vessel.prediction()->back().time < t + 1800 * Second);
  Speed const predicted_gain =
      (vessel.prediction()->back().degrees_of_freedom.velocity() -
       vessel.psychohistory()->back().degrees_of_freedom.velocity())
          .Norm();
  EXPECT_THAT(predicted_gain, Gt(1000 * Metre / Second));

  // A frame without a burn coasts.
  Velocity<AliceSun> const v2 =
      plugin->VesselFromParent(star, vessel_guid).velocity();
  t += δt;
  plugin->AdvanceTime(t, 1 * Radian);
  plugin->InsertOrKeepVessel(vessel_guid,
                             vessel_name,
                             star,
                             /*loaded=*/false,
                             inserted);
  VesselSet collided_vessels;
  plugin->CatchUpLaggingVessels(collided_vessels);
  EXPECT_THAT(
      (plugin->VesselFromParent(star, vessel_guid).velocity() - v2).Norm(),
      Lt(0.05 * Metre / Second));

  // The prediction no longer anticipates the burn either.  Force a synchronous
  // recomputation (as `ActivatePlayer` does in-game) so we read the prediction
  // freshly recomputed from the burn-less state, not the stale one the burn
  // left behind.  Measured over a fixed 1800 s horizon (the prediction itself
  // extends far further, so gravity alone accumulates over its whole length),
  // the gain collapses from the burn's > 1 km/s to the negligible gravity at
  // this distance.
  Vessel::MakeSynchronous();
  plugin->UpdatePrediction({vessel_guid});
  Vessel::MakeAsynchronous();
  auto const& from_state = vessel.psychohistory()->back();
  Instant const horizon = from_state.time + 1800 * Second;
  Speed const coasting_gain =
      (vessel.prediction()->EvaluateVelocity(horizon) -
       from_state.degrees_of_freedom.velocity())
          .Norm();
  EXPECT_THAT(coasting_gain, Lt(1 * Metre / Second));
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

  // Coast across the void; the stars have equal masses, so the dominance
  // boundary is the midpoint, and the hysteresis margin puts the rebase a
  // bit past it, where star B dominates by `rebase_dominance_margin`.
  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 33'600 * Second;
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

// The mass-based rebase boundary, end-to-end, on an unequal pair: star A is
// sixteen times heavier than star B, so the dominance boundary sits at 4/5 of
// the way — far past the geometric midpoint — and the hysteresis margin puts
// the actual rebase past x/(D−x) = 4√3, at x ≈ 3.50 × 10¹⁶ m.  A third star
// system C, grazed en route (closest approach 6 × 10¹⁵ m, where its dominance
// is still below A's), must not capture the representation: the transit
// rebases exactly once, directly from A to B.
TEST_F(PluginIntegrationTestWithoutPlugin, InterstellarRebaseMassBoundary) {
  Index const star_a = 0;
  Index const star_b = 1;
  Index const star_c = 2;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  auto const insert_star = [&plugin](Index const index,
                                     std::optional<Index> const parent_index,
                                     std::string const& name,
                                     std::string const& gravitational_parameter,
                                     std::string const& x,
                                     std::string const& y) {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name                    : ")", name, R"("
                        gravitational_parameter : ")",
                     gravitational_parameter, R"("
                        reference_instant       : "JD2451545.0"
                        mean_radius             : "1e6 m"
                        axis_right_ascension    : "0 deg"
                        axis_declination        : "90 deg"
                        reference_angle         : "0 rad"
                        angular_frequency       : "1 rad/s")"),
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name : ")", name, R"("
                        x    : ")", x, R"("
                        y    : ")", y, R"("
                        z    : "0 m"
                        vx   : "0 m/s"
                        vy   : "0 m/s"
                        vz   : "0 m/s")"),
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(
        index, parent_index, gravity_model, initial_state);
  };
  insert_star(star_a, std::nullopt, "star A", "2.08e21 m^3/s^2", "0 m", "0 m");
  insert_star(star_b, star_a, "star B", "1.3e20 m^3/s^2", "4e16 m", "0 m");
  insert_star(star_c, star_a, "star C", "1.3e20 m^3/s^2", "2e16 m", "6e15 m");
  plugin->EndInitialization();

  // Three distinct subsystems.
  int const subsystem_a = plugin->GetCelestial(star_a).subsystem();
  int const subsystem_b = plugin->GetCelestial(star_b).subsystem();
  int const subsystem_c = plugin->GetCelestial(star_c).subsystem();
  EXPECT_NE(subsystem_a, subsystem_b);
  EXPECT_NE(subsystem_a, subsystem_c);
  EXPECT_NE(subsystem_b, subsystem_c);

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
  EXPECT_EQ(subsystem_a, vessel.subsystem());

  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 37'200 * Second;
  int rebases = 0;
  int previous_subsystem = vessel.subsystem();
  for (Instant t = Instant() + δt; t <= t_final; t += δt) {
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              star_a,
                              /*loaded=*/false,
                              inserted);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    // The graze never captures the representation.
    EXPECT_NE(subsystem_c, vessel.subsystem());
    if (t == Instant() + 20'400 * Second) {
      // Closest approach to star C: still A's.
      EXPECT_EQ(subsystem_a, vessel.subsystem());
    }
    if (t == Instant() + 26'400 * Second) {
      // Past the geometric midpoint, but A, heavier, still dominates.
      EXPECT_EQ(subsystem_a, vessel.subsystem());
    }
    if (t == Instant() + 33'600 * Second) {
      // Past the mass-weighted balance at 3.2e16 m, but short of the
      // hysteresis margin: still A's.
      EXPECT_EQ(subsystem_a, vessel.subsystem());
    }
    if (vessel.subsystem() != previous_subsystem) {
      previous_subsystem = vessel.subsystem();
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(subsystem_b, vessel.subsystem());
}

// A vessel crosses the void into a subsystem that moves at the speed of the
// fastest nearby stars.  The motion seen through `VesselFromParent` must be
// continuous through the rebase — which now involves a trajectory whose
// points are hours old, translated at their own times, and a velocity
// re-expressed relative to the destination's moving origin — and the coast
// velocity, which is frame-invariant, must be unchanged throughout.
TEST_F(PluginIntegrationTestWithoutPlugin,
       InterstellarRebaseIntoMovingSubsystem) {
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
           vx   : "-3e5 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star_b,
                                            /*parent_index=*/star_a,
                                            gravity_model,
                                            initial_state);
  }
  plugin->EndInitialization();
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
  Velocity<AliceSun> const v0 =
      plugin->VesselFromParent(star_a, vessel_guid).velocity();

  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 33'600 * Second;
  int rebases = 0;
  int previous_subsystem = vessel.subsystem();
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
      EXPECT_THAT((from_parent.displacement() - *previous_displacement).Norm(),
                  RelativeErrorFrom(v * δt, Lt(1e-6)));
    }
    previous_displacement = from_parent.displacement();
    // The coast velocity relative to star A is unaffected by the
    // representation switch into the moving subsystem.
    EXPECT_THAT((from_parent.velocity() - v0).Norm() / v0.Norm(), Lt(1e-9));
    if (vessel.subsystem() != previous_subsystem) {
      previous_subsystem = vessel.subsystem();
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(), vessel.subsystem());
}

// WS1 × WS3: a vessel runs an on-rails burn (WS3) while it crosses the void and
// its representation is rebased into another subsystem (WS1).  The burn state
// must survive the subsystem switch and the accumulated Δv must be correct —
// velocity is frame-invariant across subsystems, so the burn, commanded in a
// fixed direction perpendicular to the (dominant) coast velocity, integrates
// cleanly straight through the rebase.
TEST_F(PluginIntegrationTestWithoutPlugin, InterstellarRebaseDuringOnRailsBurn) {
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

  Force const thrust = 0.1 * Newton;
  SpecificImpulse const specific_impulse = 1e4 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;
  Mass const m0 = 1 * Kilogram;
  Mass m = m0;

  // The velocity before any burning; the coast component (along `to_star_b`)
  // dominates and, in the force-free void, is unchanged by the crossing.
  Velocity<AliceSun> const v0 =
      plugin->VesselFromParent(star_a, vessel_guid).velocity();

  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 33'600 * Second;
  int rebases = 0;
  int previous_subsystem = initial_subsystem;
  for (Instant t = Instant() + δt; t <= t_final; t += δt) {
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid,
                              vessel_name,
                              star_a,
                              /*loaded=*/false,
                              inserted);
    // A burn perpendicular to the coast (commanded in `World`; the constant
    // planetarium rotation makes it a fixed inertial direction), re-armed each
    // catch-up per the one-shot contract.
    plugin->SetVesselOnRailsBurn(vessel_guid,
                                 thrust,
                                 specific_impulse,
                                 /*initial_mass=*/m,
                                 Vector<double, World>({0, 1, 0}),
                                 /*max_duration=*/1 * Hour);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    m -= δt * mass_flow;
    if (vessel.subsystem() != previous_subsystem) {
      previous_subsystem = vessel.subsystem();
      ++rebases;
    }
  }

  // The vessel rebased exactly once and ended in star B's subsystem, despite
  // burning throughout.
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(), vessel.subsystem());

  // The burn survived the rebase: the accumulated Δv is Циолковский's, and it
  // lies transverse to the coast (in the z direction, since a `World` y command
  // comes out along `AliceSun` z), separable from the dominant coast velocity.
  Speed const Δv = specific_impulse * std::log(m0 / m);
  Velocity<AliceSun> const Δv_vector =
      plugin->VesselFromParent(star_a, vessel_guid).velocity() - v0;
  EXPECT_THAT(Δv_vector.Norm(), RelativeErrorFrom(Δv, Lt(1e-3)));
  EXPECT_THAT(Δv_vector.coordinates().z, RelativeErrorFrom(Δv, Lt(1e-3)));
}

// WS3 × serialization: the trajectory produced by an on-rails burn survives a
// save/load cycle (the velocity gained is in the serialized history), while the
// burn itself does not (it is a transient re-armed each frame by the game).  A
// reloaded vessel therefore coasts until the burn is re-armed, and can then
// continue burning.
TEST_F(PluginIntegrationTestWithoutPlugin, OnRailsBurnSurvivesSaveLoad) {
  Index const star = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name                    : "star"
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
        R"(name : "star"
           x    : "0 m"
           y    : "0 m"
           z    : "0 m"
           vx   : "0 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star,
                                             /*parent_index=*/std::nullopt,
                                             gravity_model,
                                             initial_state);
  }
  plugin->EndInitialization();

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid,
                             vessel_name,
                             star,
                             /*loaded=*/false,
                             inserted);
  // Far enough from the star that gravity is negligible against the burn.
  plugin->InsertUnloadedPart(
      part_id,
      part_name,
      vessel_guid,
      {Displacement<AliceSun>({1e12 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  Force const thrust = 1 * Newton;
  SpecificImpulse const specific_impulse = 1e4 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;
  Time const δt = 100 * Second;
  Mass const m0 = 1 * Kilogram;
  Mass m = m0;
  Instant t;

  // Two frames of burning.
  for (int i = 0; i < 2; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                               /*loaded=*/false, inserted);
    plugin->SetVesselOnRailsBurn(vessel_guid, thrust, specific_impulse,
                                 /*initial_mass=*/m,
                                 Vector<double, World>({0, 1, 0}),
                                 /*max_duration=*/1 * Hour);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    m -= δt * mass_flow;
  }
  Velocity<AliceSun> const v_burned =
      plugin->VesselFromParent(star, vessel_guid).velocity();

  // Save and reload.  Only one `Plugin` may exist at a time, so destroy first.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);

  // The velocity gained by the burn is preserved: it lives in the serialized
  // trajectory, not in the (transient) burn.
  EXPECT_THAT(
      (plugin2->VesselFromParent(star, vessel_guid).velocity() - v_burned)
          .Norm(),
      Lt(1e-6 * Metre / Second));

  // The burn did NOT survive: a frame without re-arming it is a pure coast.
  Velocity<AliceSun> const v_reloaded =
      plugin2->VesselFromParent(star, vessel_guid).velocity();
  t += δt;
  plugin2->AdvanceTime(t, 1 * Radian);
  plugin2->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                              /*loaded=*/false, inserted);
  {
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
  }
  EXPECT_THAT(
      (plugin2->VesselFromParent(star, vessel_guid).velocity() - v_reloaded)
          .Norm(),
      Lt(0.05 * Metre / Second));

  // Re-arming the burn after the reload resumes thrust.
  Velocity<AliceSun> const v_before_resume =
      plugin2->VesselFromParent(star, vessel_guid).velocity();
  t += δt;
  plugin2->AdvanceTime(t, 1 * Radian);
  plugin2->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                              /*loaded=*/false, inserted);
  plugin2->SetVesselOnRailsBurn(vessel_guid, thrust, specific_impulse,
                                /*initial_mass=*/m,
                                Vector<double, World>({0, 1, 0}),
                                /*max_duration=*/1 * Hour);
  {
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
  }
  Speed const Δv_resumed = specific_impulse * std::log(m / (m - δt * mass_flow));
  EXPECT_THAT(
      (plugin2->VesselFromParent(star, vessel_guid).velocity() -
       v_before_resume).Norm(),
      RelativeErrorFrom(Δv_resumed, Lt(1e-3)));
}

// Reanimation reconstructs a segment by re-integrating with
// `NoIntrinsicAccelerations` (`vessel.cpp` ReanimateOneCheckpoint), which for a
// burn would produce a gravitational coast rather than the powered arc.  That
// only happens for a *collapsible* segment; a segment carrying an on-rails burn
// is now kept non-collapsible (see `Vessel::IsCollapsible` and
// `Vessel::AdvanceTime`), so it is checkpointed and reconstructed exactly.
// This test drives a short burn (fully serialized, so the reload does not
// actually forget it) and confirms the burn survives a save/load/reanimation
// round-trip; the collapsibility guarantee that makes it robust to an actual
// drop is covered by `VesselTest.IsCollapsible`.
TEST_F(PluginIntegrationTestWithoutPlugin, OnRailsBurnHistorySurvivesReanimation) {
  Index const star = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        R"(name                    : "star"
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
        R"(name : "star"
           x    : "0 m"
           y    : "0 m"
           z    : "0 m"
           vx   : "0 m/s"
           vy   : "0 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star,
                                             /*parent_index=*/std::nullopt,
                                             gravity_model,
                                             initial_state);
  }
  plugin->EndInitialization();

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                             /*loaded=*/false, inserted);
  // Far from the star (gravity negligible) and at rest, so the reconstruction
  // is essentially a stationary point and any velocity is the burn's.
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({1e12 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  Force const thrust = 1 * Newton;
  SpecificImpulse const specific_impulse = 1e4 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;
  Time const δt = 100 * Second;
  Mass m = 1 * Kilogram;
  Instant const t_start;
  Instant t = t_start;
  Instant const t_sample = t_start + 3 * δt;

  // Burn for several frames.
  for (int i = 0; i < 6; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                               /*loaded=*/false, inserted);
    plugin->SetVesselOnRailsBurn(vessel_guid, thrust, specific_impulse,
                                 /*initial_mass=*/m,
                                 Vector<double, World>({0, 1, 0}),
                                 /*max_duration=*/1 * Hour);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    m -= δt * mass_flow;
  }

  // The actual (burn-affected) state at the sample instant.
  DegreesOfFreedom<Barycentric> const actual =
      plugin->GetVessel(vessel_guid)->trajectory().EvaluateDegreesOfFreedom(
          t_sample);

  // Save and reload; then reanimate the forgotten past.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  auto const vessel2 = plugin2->GetVessel(vessel_guid);
  vessel2->AwaitReanimation(t_start, /*quiet=*/true);

  // Whether the reload actually forgot the burn history determines the answer.
  bool const reanimated = vessel2->trajectory().t_min() <= t_sample;
  LOG(ERROR) << "reanimated t_min <= t_sample: " << reanimated
             << "; vessel2 t_min = " << vessel2->trajectory().t_min();
  ASSERT_TRUE(reanimated);
  DegreesOfFreedom<Barycentric> const reconstructed =
      vessel2->trajectory().EvaluateDegreesOfFreedom(t_sample);
  Length const position_divergence =
      (reconstructed.position() - actual.position()).Norm();
  Speed const velocity_divergence =
      (reconstructed.velocity() - actual.velocity()).Norm();
  LOG(ERROR) << "Reanimation-vs-actual at the sample instant — position: "
             << position_divergence << ", velocity: " << velocity_divergence
             << " (actual speed " << actual.velocity().Norm() << ").";
  // The burn actually happened (non-trivial speed) and the reloaded history
  // reproduces it exactly: the burn survives serialization untouched.
  EXPECT_THAT(actual.velocity().Norm(), Gt(100 * Metre / Second));
  EXPECT_THAT(position_divergence, Lt(1 * Milli(Metre)));
  EXPECT_THAT(velocity_divergence, Lt(1e-6 * Metre / Second));
}

// WS4 minimal-fork load-bearing assumption (R7 §9.1): a vessel that goes
// unmanaged is destroyed, and on re-adoption a fresh Vessel is constructed with
// its subsystem taken from its parent celestial.  So a warped vessel re-adopted
// around a destination star in another subsystem lands in THAT subsystem — with
// its trajectory anchored at the destination's local origin, not offset by the
// interstellar distance.  This confirms no C++ re-seed fork is needed for the
// minimal-fork warp (the assignment is automatic).
TEST_F(PluginIntegrationTestWithoutPlugin, WarpReadoptionLandsInDestinationSubsystem) {
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
  EXPECT_NE(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetCelestial(star_b).subsystem());

  bool inserted;
  Length const orbit = 1e9 * Metre;

  // A vessel adopted around star A (the origin system).
  GUID const guid_a = "vessel-A";
  plugin->InsertOrKeepVessel(guid_a, "A", star_a, /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      101, "part-A", guid_a,
      {Displacement<AliceSun>({orbit, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});

  // A vessel adopted around star B — this is the re-adoption a warp produces at
  // the destination (a fresh Vessel whose parent is the destination star).
  GUID const guid_b = "vessel-B";
  plugin->InsertOrKeepVessel(guid_b, "B", star_b, /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      102, "part-B", guid_b,
      {Displacement<AliceSun>({orbit, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});

  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  // Each vessel is tagged with its parent star's subsystem — automatically,
  // from the constructor (no distance-triggered rebase, no re-seed fork).
  EXPECT_EQ(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetVessel(guid_a)->subsystem());
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            plugin->GetVessel(guid_b)->subsystem());
  EXPECT_NE(plugin->GetVessel(guid_a)->subsystem(),
            plugin->GetVessel(guid_b)->subsystem());

  // The destination vessel is anchored at star B's local origin: its trajectory
  // sits ~1e9 m from that origin, NOT ~4e16 m (which is what a wrong subsystem
  // assignment relative to star A would produce — the precision catastrophe
  // WS1 exists to prevent).
  Length const raw_offset_from_own_origin =
      (plugin->GetVessel(guid_b)->trajectory().back().degrees_of_freedom.position() -
       Barycentric::origin).Norm();
  EXPECT_THAT(raw_offset_from_own_origin, Lt(2e9 * Metre));
  // And VesselFromParent still reports the true near-star-B orbit.
  EXPECT_THAT(plugin->VesselFromParent(star_b, guid_b).displacement().Norm(),
              AbsoluteErrorFrom(orbit, Lt(1 * Kilo(Metre))));
}

// Docking two vessels whose representations live in different subsystems.  The
// rebase is only evaluated for unloaded vessels and its dominance hysteresis
// makes the subsystem of a deep-void vessel depend on its approach history, so
// a void rendezvous legitimately puts parts tagged with distinct subsystems
// into one collision subset.  The pile-up constructor requires a single
// representation; the plugin must reconcile the vessels (to the subsystem
// carrying the largest part mass) before collecting the pile-ups instead of
// dying on the pile-up's consistency check.
TEST_F(PluginIntegrationTestWithoutPlugin, CrossSubsystemDockingReconciles) {
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
  EXPECT_NE(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetCelestial(star_b).subsystem());

  bool inserted;

  // A rendezvous in the middle of the void, where μ/d² dominance is
  // ambiguous: the (heavier) station arrived from star A, the (lighter)
  // visitor from star B, so they hold different subsystems while sitting
  // 10 m apart physically.
  Length const midpoint = 2e16 * Metre;
  GUID const guid_a = "station";
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      101, "part-A", guid_a,
      {Displacement<AliceSun>({midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  GUID const guid_b = "visitor";
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_b,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      102, "part-B", guid_b,
      {Displacement<AliceSun>({10 * Metre - midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->GetVessel(guid_a)->part(101)->set_mass(3 * Kilogram);
  plugin->GetVessel(guid_b)->part(102)->set_mass(1 * Kilogram);
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  EXPECT_NE(plugin->GetVessel(guid_a)->subsystem(),
            plugin->GetVessel(guid_b)->subsystem());
  Displacement<AliceSun> const visitor_from_star_b =
      plugin->VesselFromParent(star_b, guid_b).displacement();

  // Next frame: the vessels dock.
  Instant const t = Instant() + 100 * Second;
  plugin->AdvanceTime(t, 1 * Radian);
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_b,
                             /*loaded=*/false, inserted);
  plugin->GetVessel(guid_a)->KeepPart(101);
  plugin->GetVessel(guid_b)->KeepPart(102);
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(101, 102);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  // Both vessels were reconciled to the heavier side (the station's
  // subsystem) and share a single pile-up in that subsystem.
  int const station_subsystem = plugin->GetCelestial(star_a).subsystem();
  EXPECT_EQ(station_subsystem, plugin->GetVessel(guid_a)->subsystem());
  EXPECT_EQ(station_subsystem, plugin->GetVessel(guid_b)->subsystem());
  auto* const pile_up_a =
      plugin->GetVessel(guid_a)->part(101)->containing_pile_up();
  auto* const pile_up_b =
      plugin->GetVessel(guid_b)->part(102)->containing_pile_up();
  ASSERT_NE(nullptr, pile_up_a);
  EXPECT_EQ(pile_up_a, pile_up_b);
  EXPECT_EQ(station_subsystem, pile_up_a->subsystem());

  // The rebase changed only the representation of the visitor, not its
  // physical state: it is still where it was relative to star B (both are at
  // rest, so the elapsed frame moves nothing), up to the rounding of the
  // interstellar translation (~ULP of 4e16 m ≈ 10 m).
  EXPECT_THAT((plugin->VesselFromParent(star_b, guid_b).displacement() -
               visitor_from_star_b).Norm(),
              Lt(100 * Metre));
}

// At 1000 light years (9.46e18 m) the per-subsystem representation keeps every
// distance-dependent error confined to the one-time cross-subsystem
// translation rounding, ~ULP(9.46e18 m) ≈ 2 km: physics local to the
// destination star is exact to the metre, and a void rendezvous reconciles to
// within a few ULPs.  This certifies that the interstellar machinery has no
// distance wall of its own well beyond the NearStars roster (40 ly).
TEST_F(PluginIntegrationTestWithoutPlugin, ThousandLightYearScale) {
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
           x    : "9.46e18 m"
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
  // Clustering still partitions two stars 1000 ly apart into two subsystems.
  EXPECT_NE(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetCelestial(star_b).subsystem());

  bool inserted;
  Length const orbit = 1e9 * Metre;
  Length const midpoint = 4.73e18 * Metre;

  // A vessel adopted around the destination star — the state a warp arrival
  // produces.  Its precision must not depend on the 1000 ly to its origin.
  GUID const guid_orbiter = "orbiter";
  plugin->InsertOrKeepVessel(guid_orbiter, "orbiter", star_b,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      201, "part-orbiter", guid_orbiter,
      {Displacement<AliceSun>({orbit, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});

  // A mid-void rendezvous, 10 m apart, arrived from opposite stars.
  GUID const guid_a = "station";
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      202, "part-A", guid_a,
      {Displacement<AliceSun>({midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  GUID const guid_b = "visitor";
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_b,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      203, "part-B", guid_b,
      {Displacement<AliceSun>({10 * Metre - midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->GetVessel(guid_a)->part(202)->set_mass(3 * Kilogram);
  plugin->GetVessel(guid_b)->part(203)->set_mass(1 * Kilogram);
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  // Destination locality: the orbiter's raw coordinates are anchored at
  // star B's origin (~1e9 m, not ~1e19 m), and its reported orbit is exact to
  // well below the metre — no distance dependence.
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            plugin->GetVessel(guid_orbiter)->subsystem());
  EXPECT_THAT((plugin->GetVessel(guid_orbiter)
                   ->trajectory().back().degrees_of_freedom.position() -
               Barycentric::origin).Norm(),
              Lt(2e9 * Metre));
  EXPECT_THAT(
      plugin->VesselFromParent(star_b, guid_orbiter).displacement().Norm(),
      AbsoluteErrorFrom(orbit, Lt(1 * Metre)));

  EXPECT_NE(plugin->GetVessel(guid_a)->subsystem(),
            plugin->GetVessel(guid_b)->subsystem());
  Displacement<AliceSun> const visitor_from_star_b =
      plugin->VesselFromParent(star_b, guid_b).displacement();

  // Next frame: the void vessels dock.
  Instant const t = Instant() + 100 * Second;
  plugin->AdvanceTime(t, 1 * Radian);
  plugin->InsertOrKeepVessel(guid_orbiter, "orbiter", star_b,
                             /*loaded=*/false, inserted);
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_b,
                             /*loaded=*/false, inserted);
  plugin->GetVessel(guid_orbiter)->KeepPart(201);
  plugin->GetVessel(guid_a)->KeepPart(202);
  plugin->GetVessel(guid_b)->KeepPart(203);
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(202, 203);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  // The reconciliation translated the visitor across 9.46e18 m of
  // representation change; the entire distance-dependent cost is a few ULPs
  // of that translation (ULP ≈ 2 km), physically a fixed sub-10-km offset.
  int const station_subsystem = plugin->GetCelestial(star_a).subsystem();
  EXPECT_EQ(station_subsystem, plugin->GetVessel(guid_a)->subsystem());
  EXPECT_EQ(station_subsystem, plugin->GetVessel(guid_b)->subsystem());
  EXPECT_EQ(plugin->GetVessel(guid_a)->part(202)->containing_pile_up(),
            plugin->GetVessel(guid_b)->part(203)->containing_pile_up());
  EXPECT_THAT((plugin->VesselFromParent(star_b, guid_b).displacement() -
               visitor_from_star_b).Norm(),
              Lt(10 * Kilo(Metre)));
}

// An unloaded vessel coasting where the damped far field is exactly zero
// adopts an anchor — a private origin moving with it — so that its stored
// coordinates stay near zero instead of growing to ~10¹⁶ m.  The anchor
// survives a save, and loading the vessel drops it without moving the vessel
// physically.
TEST_F(PluginIntegrationTestWithoutPlugin, VoidCoastAdoptsAnchor) {
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

  bool inserted;
  // A vessel orbiting star A: near a star, no anchor.
  GUID const guid_near = "near";
  plugin->InsertOrKeepVessel(guid_near, "near", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      301, "part-near", guid_near,
      {Displacement<AliceSun>({1e9 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  // A vessel adrift in the middle of the void.
  GUID const guid_void = "drifter";
  plugin->InsertOrKeepVessel(guid_void, "drifter", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      302, "part-drifter", guid_void,
      {Displacement<AliceSun>({2e16 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>({0 * Metre / Second,
                           3 * Kilo(Metre) / Second,
                           0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  EXPECT_FALSE(plugin->GetVessel(guid_near)->anchor().has_value());
  not_null<Vessel*> const drifter = plugin->GetVessel(guid_void);
  ASSERT_TRUE(drifter->anchor().has_value());
  // The anchored coordinates and velocity are near zero (the coordinates were
  // ~2e16 m); the anchor records where the vessel really is and how it moves.
  EXPECT_THAT((drifter->trajectory().back().degrees_of_freedom.position() -
               Barycentric::origin).Norm(),
              Lt(1 * Kilo(Metre)));
  EXPECT_THAT(drifter->trajectory().back().degrees_of_freedom.velocity()
                  .Norm(),
              Lt(1e-6 * Metre / Second));
  EXPECT_THAT(drifter->anchor()->offset.Norm(),
              AbsoluteErrorFrom(2e16 * Metre, Lt(1e12 * Metre)));
  EXPECT_THAT(drifter->anchor()->velocity.Norm(),
              AbsoluteErrorFrom(3 * Kilo(Metre) / Second,
                                Lt(1 * Metre / Second)));
  auto const saved_anchor = *drifter->anchor();

  // The representation survives a save.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  not_null<Vessel*> const drifter2 = plugin2->GetVessel(guid_void);
  ASSERT_TRUE(drifter2->anchor().has_value());
  EXPECT_EQ(saved_anchor, *drifter2->anchor());

  // Loading the vessel drops the anchor and restores subsystem-relative
  // coordinates without moving the vessel physically.
  auto const& [t_head, anchored_degrees_of_freedom] =
      drifter2->trajectory().back();
  Position<Barycentric> const expected_position =
      anchored_degrees_of_freedom.position() +
      drifter2->anchor()->OffsetAt(t_head);
  plugin2->InsertOrKeepVessel(guid_void, "drifter", star_a,
                              /*loaded=*/true, inserted);
  EXPECT_FALSE(drifter2->anchor().has_value());
  EXPECT_EQ(expected_position,
            drifter2->trajectory().back().degrees_of_freedom.position());
}

// Inserts two equal stars 4×10¹⁶ m apart (star A at the origin, star B beyond
// it).  The void midpoint at 2×10¹⁶ m is beyond the far-field-damping threshold
// (~1.2 ly) of both, so a vessel there anchors.  Shared by the WS6-4 tests.
void InsertTwoStarVoid(Plugin& plugin) {
  Index const star_a = 0;
  Index const star_b = 1;
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
    plugin.InsertCelestialAbsoluteCartesian(star_a,
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
    plugin.InsertCelestialAbsoluteCartesian(star_b,
                                            /*parent_index=*/star_a,
                                            gravity_model,
                                            initial_state);
  }
  plugin.EndInitialization();
}

// WS6-4: an anchored vessel's prediction must be a force-free coast, not a
// plunge.  The prognostication is seeded from the near-origin anchored
// coordinates; without the anchor the integrator reads them as
// subsystem-relative (on top of the home star at the origin) and the
// prediction plunges into it.
TEST_F(PluginIntegrationTestWithoutPlugin, AnchoredVoidPredictionCoasts) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  bool inserted;
  GUID const guid_void = "drifter";
  plugin->InsertOrKeepVessel(guid_void, "drifter", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      302, "part-drifter", guid_void,
      {Displacement<AliceSun>({2e16 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>({0 * Metre / Second,
                           3 * Kilo(Metre) / Second,
                           0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  not_null<Vessel*> const drifter = plugin->GetVessel(guid_void);
  ASSERT_TRUE(drifter->anchor().has_value());

  // Predict synchronously so the read is deterministic.
  Vessel::MakeSynchronous();
  plugin->UpdatePrediction({guid_void});
  Vessel::MakeAsynchronous();

  auto const& from_state = drifter->psychohistory()->back();
  Instant const horizon = from_state.time + 1800 * Second;
  Speed const coasting_gain =
      (drifter->prediction()->EvaluateVelocity(horizon) -
       from_state.degrees_of_freedom.velocity())
          .Norm();
  // Force-free void: the predicted velocity is unchanged over the horizon.  A
  // plunge (the pre-fix behaviour) would show a huge gain toward the origin.
  EXPECT_THAT(coasting_gain, Lt(1 * Milli(Metre) / Second));
}

}  // namespace ksp_plugin
}  // namespace principia
