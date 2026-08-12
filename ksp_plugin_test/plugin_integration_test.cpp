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

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "astronomy/frames.hpp"
#include "astronomy/time_scales.hpp"
#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/permutation.hpp"
#include "geometry/perspective.hpp"
#include "geometry/r3_element.hpp"
#include "geometry/rotation.hpp"
#include "geometry/signature.hpp"
#include "geometry/space.hpp"
#include "geometry/space_transformations.hpp"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "integrators/embedded_explicit_generalized_runge_kutta_nyström_integrator.hpp"  // NOLINT
#include "integrators/embedded_explicit_runge_kutta_nyström_integrator.hpp"
#include "integrators/methods.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin/identification.hpp"
#include "ksp_plugin/part.hpp"
#include "ksp_plugin/planetarium.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "physics/rigid_motion.hpp"
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
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_permutation;
using namespace principia::geometry::_perspective;
using namespace principia::geometry::_r3_element;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_signature;
using namespace principia::geometry::_space;
using namespace principia::geometry::_space_transformations;
using namespace principia::integrators::_embedded_explicit_generalized_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_embedded_explicit_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_methods;
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_identification;
using namespace principia::ksp_plugin::_part;
using namespace principia::ksp_plugin::_pile_up;
using namespace principia::ksp_plugin::_planetarium;
using namespace principia::ksp_plugin::_plugin;
using namespace principia::ksp_plugin::_vessel;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_rigid_motion;
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

  // A direction that is not a number neither collapses under normalization nor
  // compares equal to zero.  This is before the first catch-up, so clearing the
  // burn here does not disturb the prediction tested at the end.
  plugin->SetVesselOnRailsBurn(vessel_guid,
                               thrust,
                               specific_impulse,
                               /*initial_mass=*/1 * Kilogram,
                               Vector<double, World>({0, 1, 0}),
                               /*max_duration=*/1 * Hour);
  plugin->SetVesselOnRailsBurn(
      vessel_guid,
      thrust,
      specific_impulse,
      /*initial_mass=*/1 * Kilogram,
      Vector<double, World>({0,
                             std::numeric_limits<double>::quiet_NaN(),
                             0}),
      /*max_duration=*/1 * Hour);
  EXPECT_FALSE(vessel.part(part_id)->containing_pile_up()->
                   on_rails_burn().has_value());

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

  // The prediction coasts: what it answers is where the orbit goes if the
  // engine is cut now, not where the burn takes the vessel.  Anticipating the
  // burn to propellant exhaustion gained 4.6 km/s here; the coast turns 275 m/s
  // of velocity over however far the prediction happens to reach.
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
  EXPECT_THAT(predicted_gain, Lt(500 * Metre / Second));

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
  int const initial_subsystem = vessel.placement().subsystem;
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
    if (vessel.placement().subsystem != previous_subsystem) {
      previous_subsystem = vessel.placement().subsystem;
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            vessel.placement().subsystem);

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
        vessel.placement());
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
  int const subsystem_before_save = vessel.placement().subsystem;
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  auto const& vessel2 = *plugin2->GetVessel(vessel_guid);
  EXPECT_EQ(subsystem_before_save, vessel2.placement().subsystem);
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

// The void-navigation readouts: the force-free boundary, the nearest star,
// and the target- and reference-relative states across distinct subsystem
// representations.
TEST_F(PluginIntegrationTestWithoutPlugin, VoidNavigationState) {
  Index const star_a = 0;
  Index const star_b = 1;
  auto const plugin =
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
           vy   : "5e3 m/s"
           vz   : "0 m/s")",
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(star_b,
                                            /*parent_index=*/star_a,
                                            gravity_model,
                                            initial_state);
  }
  plugin->EndInitialization();

  GUID const near_a_guid = "near-a";
  GUID const void_guid = "in-void";
  bool inserted;
  Vector<double, AliceSun> const to_star_b =
      Normalize(plugin->CelestialFromParent(star_b).displacement());
  Speed const v = 1e3 * Metre / Second;
  Speed const v_star_b = 5e3 * Metre / Second;
  plugin->InsertOrKeepVessel(near_a_guid,
                            "near A",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  plugin->InsertUnloadedPart(111,
                             "near part",
                             near_a_guid,
                             {(1e10 * Metre) * to_star_b, v * to_star_b});
  plugin->InsertOrKeepVessel(void_guid,
                            "in the void",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  plugin->InsertUnloadedPart(222,
                             "void part",
                             void_guid,
                             {(2.1e16 * Metre) * to_star_b, v * to_star_b});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  Time const δt = 1200 * Second;
  plugin->AdvanceTime(Instant() + δt, 1 * Radian);
  plugin->InsertOrKeepVessel(near_a_guid,
                            "near A",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  plugin->InsertOrKeepVessel(void_guid,
                            "in the void",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  VesselSet collided_vessels;
  plugin->CatchUpLaggingVessels(collided_vessels);

  // Near star A the far field is undamped and star A is the nearest star.
  auto const near_state = plugin->VesselNavigationState(
      near_a_guid,
      /*target_index=*/star_b,
      /*reference_index=*/star_a);
  EXPECT_FALSE(near_state.in_void);
  EXPECT_EQ(star_a, near_state.nearest_star_index);
  EXPECT_THAT(near_state.nearest_star_distance,
              RelativeErrorFrom(1e10 * Metre, Lt(1e-2)));

  // Between the stars, beyond both outer thresholds √(μ/floor) ≈ 1.14e16 m,
  // the vessel coasts force-free and star B is the nearer star.  The target-
  // and reference-relative states cross the subsystem representations: the
  // vessel is still represented in star A's subsystem while star B, in its
  // own moving subsystem, exercises the position and velocity conversions.
  EXPECT_EQ(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetVessel(void_guid)->placement().subsystem);
  EXPECT_NE(plugin->GetCelestial(star_b).subsystem(),
            plugin->GetVessel(void_guid)->placement().subsystem);
  auto const void_state = plugin->VesselNavigationState(
      void_guid,
      /*target_index=*/star_b,
      /*reference_index=*/star_a);
  EXPECT_TRUE(void_state.in_void);
  EXPECT_EQ(star_b, void_state.nearest_star_index);
  EXPECT_THAT(void_state.nearest_star_distance,
              RelativeErrorFrom(1.9e16 * Metre, Lt(1e-3)));
  // The components are checked in `Barycentric`, where the vessel cruises
  // along +x and star B moves along +y; magnitude-only checks would let a
  // misdirected subsystem conversion through.
  EXPECT_THAT(void_state.position_wrt_target.coordinates().x,
              RelativeErrorFrom(-1.9e16 * Metre, Lt(1e-3)));
  EXPECT_THAT(void_state.position_wrt_target.coordinates().y,
              AbsoluteErrorFrom(-v_star_b * δt, Lt(1 * Kilo(Metre))));
  EXPECT_THAT(void_state.velocity_wrt_target.coordinates().x,
              RelativeErrorFrom(v, Lt(1e-6)));
  EXPECT_THAT(void_state.velocity_wrt_target.coordinates().y,
              RelativeErrorFrom(-v_star_b, Lt(1e-6)));
  EXPECT_THAT(void_state.velocity_wrt_reference.coordinates().x,
              RelativeErrorFrom(v, Lt(1e-6)));
  EXPECT_THAT(void_state.velocity_wrt_reference.coordinates().y,
              AbsoluteErrorFrom(0 * Metre / Second,
                                Lt(1e-3 * Metre / Second)));
}

// A subsystem's primary is its heaviest body, not its lowest-index one: a
// token-mass barycentre node inserted below its star must lend neither its
// name to the subsystem nor its distance to the nearest-star readout.
TEST_F(PluginIntegrationTestWithoutPlugin, SubsystemPrimary) {
  Index const star_a = 0;
  Index const node = 1;
  Index const star_b = 2;
  auto const plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  auto const insert_body = [&plugin](Index const index,
                                     std::optional<Index> const parent_index,
                                     std::string const& name,
                                     std::string const& gravitational_parameter,
                                     std::string const& x) {
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
                        y    : "0 m"
                        z    : "0 m"
                        vx   : "0 m/s"
                        vy   : "0 m/s"
                        vz   : "0 m/s")"),
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(
        index, parent_index, gravity_model, initial_state);
  };
  insert_body(star_a, std::nullopt, "star A", "1.3e20 m^3/s^2", "0 m");
  // The node sits between its star and the vessel below, so that a primary
  // picked by index would win both readouts.
  insert_body(node, star_a, "node", "1 m^3/s^2", "3.999e16 m");
  insert_body(star_b, node, "star B", "1.3e20 m^3/s^2", "4e16 m");
  plugin->EndInitialization();

  // The node clusters with star B while preceding it in index.
  int const subsystem_b = plugin->GetCelestial(star_b).subsystem();
  EXPECT_EQ(subsystem_b, plugin->GetCelestial(node).subsystem());
  EXPECT_NE(subsystem_b, plugin->GetCelestial(star_a).subsystem());
  EXPECT_EQ(star_a,
            plugin->SubsystemPrimary(plugin->GetCelestial(star_a).subsystem()));
  EXPECT_EQ(star_b, plugin->SubsystemPrimary(subsystem_b));

  // A vessel coasting between the stars: the nearest star is star B itself,
  // not the node in front of it.
  GUID const vessel_guid = "in-void";
  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid,
                            "in the void",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  Vector<double, AliceSun> const to_star_b =
      Normalize(plugin->CelestialFromParent(node).displacement());
  Speed const v = 1e3 * Metre / Second;
  plugin->InsertUnloadedPart(111,
                             "void part",
                             vessel_guid,
                             {(2.1e16 * Metre) * to_star_b, v * to_star_b});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  Time const δt = 1200 * Second;
  plugin->AdvanceTime(Instant() + δt, 1 * Radian);
  plugin->InsertOrKeepVessel(vessel_guid,
                            "in the void",
                            star_a,
                            /*loaded=*/false,
                            inserted);
  VesselSet collided_vessels;
  plugin->CatchUpLaggingVessels(collided_vessels);

  auto const state =
      plugin->VesselNavigationState(vessel_guid,
                                    /*target_index=*/std::nullopt,
                                    /*reference_index=*/std::nullopt);
  EXPECT_EQ(star_b, state.nearest_star_index);
  // Tight enough to reject the node's distance, 5.3e-4 nearer.
  EXPECT_THAT(state.nearest_star_distance,
              RelativeErrorFrom(1.9e16 * Metre, Lt(1e-4)));
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
  EXPECT_EQ(subsystem_a, vessel.placement().subsystem);

  Time const δt = 1200 * Second;
  Instant const t_final = Instant() + 37'200 * Second;
  int rebases = 0;
  int previous_subsystem = vessel.placement().subsystem;
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
    EXPECT_NE(subsystem_c, vessel.placement().subsystem);
    if (t == Instant() + 20'400 * Second) {
      // Closest approach to star C: still A's.
      EXPECT_EQ(subsystem_a, vessel.placement().subsystem);
    }
    if (t == Instant() + 26'400 * Second) {
      // Past the geometric midpoint, but A, heavier, still dominates.
      EXPECT_EQ(subsystem_a, vessel.placement().subsystem);
    }
    if (t == Instant() + 33'600 * Second) {
      // Past the mass-weighted balance at 3.2e16 m, but short of the
      // hysteresis margin: still A's.
      EXPECT_EQ(subsystem_a, vessel.placement().subsystem);
    }
    if (vessel.placement().subsystem != previous_subsystem) {
      previous_subsystem = vessel.placement().subsystem;
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(subsystem_b, vessel.placement().subsystem);
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
  int previous_subsystem = vessel.placement().subsystem;
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
    if (vessel.placement().subsystem != previous_subsystem) {
      previous_subsystem = vessel.placement().subsystem;
      ++rebases;
    }
  }
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            vessel.placement().subsystem);
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
  int const initial_subsystem = vessel.placement().subsystem;

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
    if (vessel.placement().subsystem != previous_subsystem) {
      previous_subsystem = vessel.placement().subsystem;
      ++rebases;
    }
  }

  // The vessel rebased exactly once and ended in star B's subsystem, despite
  // burning throughout.
  EXPECT_EQ(1, rebases);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            vessel.placement().subsystem);

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
            plugin->GetVessel(guid_a)->placement().subsystem);
  EXPECT_EQ(plugin->GetCelestial(star_b).subsystem(),
            plugin->GetVessel(guid_b)->placement().subsystem);
  EXPECT_NE(plugin->GetVessel(guid_a)->placement().subsystem,
            plugin->GetVessel(guid_b)->placement().subsystem);

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
  EXPECT_NE(plugin->GetVessel(guid_a)->placement().subsystem,
            plugin->GetVessel(guid_b)->placement().subsystem);
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
  EXPECT_EQ(station_subsystem,
            plugin->GetVessel(guid_a)->placement().subsystem);
  EXPECT_EQ(station_subsystem,
            plugin->GetVessel(guid_b)->placement().subsystem);
  auto* const pile_up_a =
      plugin->GetVessel(guid_a)->part(101)->containing_pile_up();
  auto* const pile_up_b =
      plugin->GetVessel(guid_b)->part(102)->containing_pile_up();
  ASSERT_NE(nullptr, pile_up_a);
  EXPECT_EQ(pile_up_a, pile_up_b);
  EXPECT_EQ(station_subsystem, pile_up_a->placement().subsystem);

  // The rebase changed only the representation of the visitor, not its
  // physical state: it is still where it was relative to star B (both are at
  // rest, so the elapsed frame moves nothing), up to the rounding of the
  // interstellar translation (~ULP of 4e16 m ≈ 10 m).
  EXPECT_THAT((plugin->VesselFromParent(star_b, guid_b).displacement() -
               visitor_from_star_b).Norm(),
              Lt(100 * Metre));

  // Both vessels adopted their FIRST anchors in the catch-up above (they
  // reached the void unanchored), so the shared pile-up was asked to move
  // twice at the same epoch.  The second move must translate it from the
  // placement the first one left it in — the vessels' 10 m separation — not
  // by the second vessel's own ~2e16 m delta: compounded translations would
  // teleport the docked pair by the void distance at the next advance.  Run
  // that next advance and require the pair to still be where it was.
  Instant const t2 = t + 100 * Second;
  plugin->AdvanceTime(t2, 1 * Radian);
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
            plugin->GetVessel(guid_orbiter)->placement().subsystem);
  EXPECT_THAT((plugin->GetVessel(guid_orbiter)
                   ->trajectory().back().degrees_of_freedom.position() -
               Barycentric::origin).Norm(),
              Lt(2e9 * Metre));
  EXPECT_THAT(
      plugin->VesselFromParent(star_b, guid_orbiter).displacement().Norm(),
      AbsoluteErrorFrom(orbit, Lt(1 * Metre)));

  EXPECT_NE(plugin->GetVessel(guid_a)->placement().subsystem,
            plugin->GetVessel(guid_b)->placement().subsystem);
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
  EXPECT_EQ(station_subsystem,
            plugin->GetVessel(guid_a)->placement().subsystem);
  EXPECT_EQ(station_subsystem,
            plugin->GetVessel(guid_b)->placement().subsystem);
  EXPECT_EQ(plugin->GetVessel(guid_a)->part(202)->containing_pile_up(),
            plugin->GetVessel(guid_b)->part(203)->containing_pile_up());
  EXPECT_THAT((plugin->VesselFromParent(star_b, guid_b).displacement() -
               visitor_from_star_b).Norm(),
              Lt(10 * Kilo(Metre)));
}

// An unloaded vessel coasting where the damped far field is exactly zero
// adopts an anchor — a private origin moving with it — so that its stored
// coordinates stay near zero instead of growing to ~10¹⁶ m.  The anchor
// survives a save, and loading the vessel keeps it — the loaded paths are
// placement-aware — without moving the vessel physically.
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

  EXPECT_FALSE(plugin->GetVessel(guid_near)->placement().anchor.has_value());
  not_null<Vessel*> const drifter = plugin->GetVessel(guid_void);
  ASSERT_TRUE(drifter->placement().anchor.has_value());
  // The anchored coordinates and velocity are near zero (the coordinates were
  // ~2e16 m); the anchor records where the vessel really is and how it moves.
  EXPECT_THAT((drifter->trajectory().back().degrees_of_freedom.position() -
               Barycentric::origin).Norm(),
              Lt(1 * Kilo(Metre)));
  EXPECT_THAT(drifter->trajectory().back().degrees_of_freedom.velocity()
                  .Norm(),
              Lt(1e-6 * Metre / Second));
  EXPECT_THAT(drifter->placement().anchor->offset.Collapse().Norm(),
              AbsoluteErrorFrom(2e16 * Metre, Lt(1e12 * Metre)));
  EXPECT_THAT(drifter->placement().anchor->velocity.Norm(),
              AbsoluteErrorFrom(3 * Kilo(Metre) / Second,
                                Lt(1 * Metre / Second)));
  auto const saved_anchor = *drifter->placement().anchor;

  // The representation survives a save.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  not_null<Vessel*> const drifter2 = plugin2->GetVessel(guid_void);
  ASSERT_TRUE(drifter2->placement().anchor.has_value());
  EXPECT_EQ(saved_anchor, *drifter2->placement().anchor);

  // Loading the vessel keeps the anchor and the small anchored coordinates:
  // the loaded paths convert between placements instead of demanding the
  // subsystem-relative absolute.
  Position<Barycentric> const expected_position =
      drifter2->trajectory().back().degrees_of_freedom.position();
  plugin2->InsertOrKeepVessel(guid_void, "drifter", star_a,
                              /*loaded=*/true, inserted);
  ASSERT_TRUE(drifter2->placement().anchor.has_value());
  EXPECT_EQ(saved_anchor, *drifter2->placement().anchor);
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

// Two unanchored vessels that dock into a single pile-up in the deep void
// share ONE re-anchor verdict: the pile-up adopts one anchor and the plugin
// replays it onto every member, so both vessels and the pile-up hold the same
// anchor at the same epoch.  A single verdict per group means the members move
// in lock-step, so a following advance preserves their relative geometry — the
// per-vessel path used to move the shared pile-up once per member and fling the
// pair by the void distance.
TEST_F(PluginIntegrationTestWithoutPlugin, SharedPileUpReAnchorsOnce) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  bool inserted;

  // A pair sitting 10 m apart deep in star A's void (2 × 10¹⁶ m out), both
  // reaching it unanchored — well beyond the re-anchor bound.
  Length const midpoint = 2e16 * Metre;
  GUID const guid_a = "station";
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      101, "part-A", guid_a,
      {Displacement<AliceSun>({midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  GUID const guid_b = "visitor";
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      102, "part-B", guid_b,
      {Displacement<AliceSun>({midpoint + 10 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->GetVessel(guid_a)->part(101)->set_mass(3 * Kilogram);
  plugin->GetVessel(guid_b)->part(102)->set_mass(1 * Kilogram);
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  // Next frame: the vessels dock into one pile-up, then catch up once.
  Instant const t = Instant() + 100 * Second;
  plugin->AdvanceTime(t, 1 * Radian);
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_a,
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

  // A single shared verdict: both vessels and the pile-up hold the same anchor
  // at the same epoch.
  auto* const pile_up =
      plugin->GetVessel(guid_a)->part(101)->containing_pile_up();
  ASSERT_NE(nullptr, pile_up);
  EXPECT_EQ(pile_up,
            plugin->GetVessel(guid_b)->part(102)->containing_pile_up());
  ASSERT_TRUE(pile_up->placement().anchor.has_value());
  ASSERT_TRUE(plugin->GetVessel(guid_a)->placement().anchor.has_value());
  ASSERT_TRUE(plugin->GetVessel(guid_b)->placement().anchor.has_value());
  EXPECT_EQ(*pile_up->placement().anchor,
            *plugin->GetVessel(guid_a)->placement().anchor);
  EXPECT_EQ(*pile_up->placement().anchor,
            *plugin->GetVessel(guid_b)->placement().anchor);
  EXPECT_EQ(pile_up->placement().anchor->epoch,
            plugin->GetVessel(guid_a)->placement().anchor->epoch);
  EXPECT_EQ(pile_up->placement().anchor->epoch,
            plugin->GetVessel(guid_b)->placement().anchor->epoch);

  Displacement<AliceSun> const visitor_from_star_a =
      plugin->VesselFromParent(star_a, guid_b).displacement();

  // A follow-up CatchUpVessel for one member re-evaluates against the
  // already-anchored pile-up, returns no change, and leaves the shared anchor
  // of every member intact.
  {
    auto const future = plugin->CatchUpVessel(guid_a);
    VesselSet collided_vessels;
    plugin->WaitForVesselToCatchUp(*future, collided_vessels);
  }
  EXPECT_EQ(*plugin->GetVessel(guid_a)->placement().anchor,
            *plugin->GetVessel(guid_b)->placement().anchor);

  // The next advance preserves the pair's relative geometry: one verdict moved
  // both members together, so nothing compounds.
  Instant const t2 = t + 100 * Second;
  plugin->AdvanceTime(t2, 1 * Radian);
  plugin->InsertOrKeepVessel(guid_a, "station", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_a,
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
  EXPECT_THAT((plugin->VesselFromParent(star_a, guid_b).displacement() -
               visitor_from_star_a).Norm(),
              Lt(100 * Metre));
}

// The loaded path in the deep void: a loaded vessel keeps its anchor, the
// inbound and outbound World conversions are placement-aware, and the
// World-side layout survives at void scale — within a vessel, across a part
// transfer between two anchored vessels, and between the two vessels.  A
// vessel spawning loaded in the void adopts an anchor at catch-up.  Without
// the placement-aware conversions the layout quantizes at the ~4 m ULP of the
// ~2×10¹⁶ m subsystem-relative absolute, and a velocity sign error would show
// up as twice the 3 km/s cruise, so the bounds below have teeth.
TEST_F(PluginIntegrationTestWithoutPlugin, LoadedVoidVesselKeepsAnchor) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  // An unloaded drifter deep in the void adopts an anchor, and so does a
  // tender a kilometre away.  The 3 km/s cruise velocity ends up in the
  // anchors, giving the velocity terms of the placement conversions teeth.
  bool inserted;
  GUID const guid = "drifter";
  GUID const tender_guid = "tender";
  Velocity<AliceSun> const cruise_velocity(
      {0 * Metre / Second,
       3 * Kilo(Metre) / Second,
       0 * Metre / Second});
  plugin->InsertOrKeepVessel(guid, "drifter", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      401, "scaffold", guid,
      {Displacement<AliceSun>({2e16 * Metre, 0 * Metre, 0 * Metre}),
       cruise_velocity});
  plugin->InsertOrKeepVessel(tender_guid, "tender", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      404, "tender scaffold", tender_guid,
      {Displacement<AliceSun>({2e16 * Metre, 1 * Kilo(Metre), 0 * Metre}),
       cruise_velocity});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  not_null<Vessel*> const drifter = plugin->GetVessel(guid);
  not_null<Vessel*> const tender = plugin->GetVessel(tender_guid);
  ASSERT_TRUE(drifter->placement().anchor.has_value());
  ASSERT_TRUE(tender->placement().anchor.has_value());

  // A tick goes by (the loaded insertion evaluates the main-body frame just
  // before the current time, which must not precede the ephemeris).
  plugin->AdvanceTime(plugin->CurrentTime() + 60 * Second, 1 * Radian);

  // The vessel loads with two fresh parts half a metre apart, replacing the
  // scaffold, the way a scene reconstruction would.  The World-side inputs
  // put the vessel near the World origin and the main body a void away.
  plugin->InsertOrKeepVessel(guid, "drifter", star_a,
                             /*loaded=*/true, inserted);
  plugin->InsertOrKeepVessel(tender_guid, "tender", star_a,
                             /*loaded=*/true, inserted);
  ASSERT_TRUE(drifter->placement().anchor.has_value());
  ASSERT_TRUE(tender->placement().anchor.has_value());
  Mass const mass = 1000 * Kilogram;
  // The main body's World coordinates must be consistent with the scene in
  // which the scaffold sits at the World origin — the way the game hands them
  // to the plugin — else the inserted parts would map a long way from the
  // vessel's true position.
  DegreesOfFreedom<World> const main_body_degrees_of_freedom =
      plugin->CelestialWorldDegreesOfFreedom(
          star_a, 401,
          plugin->BarycentricToWorld(/*reference_part_is_unmoving=*/true, 401,
                                     /*main_body_centre=*/std::nullopt),
          plugin->CurrentTime());
  // Independent teeth for the celestial path and for the velocity terms: in
  // the scene where the drifter is unmoving, the main body sits a void away
  // and recedes at the cruise speed.
  EXPECT_THAT((main_body_degrees_of_freedom.position() - World::origin).Norm(),
              AbsoluteErrorFrom(2e16 * Metre, Lt(1e12 * Metre)));
  EXPECT_THAT(main_body_degrees_of_freedom.velocity().Norm(),
              AbsoluteErrorFrom(3 * Kilo(Metre) / Second,
                                Lt(1 * Metre / Second)));
  Displacement<World> const layout_offset(
      {0.5 * Metre, 0.25 * Metre, 0 * Metre});
  auto const part_motion = [](Position<World> const& position) {
    return RigidMotion<EccentricPart, World>(
        RigidTransformation<EccentricPart, World>(
            EccentricPart::origin,
            position,
            OrthogonalMap<EccentricPart, World>::Identity()),
        World::nonrotating,
        World::unmoving);
  };
  plugin->InsertOrKeepLoadedPart(
      402, "pod", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin), 20 * Milli(Second));
  plugin->InsertOrKeepLoadedPart(
      403, "outrigger", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + layout_offset), 20 * Milli(Second));
  // The outrigger is then transferred to the tender — a part moving between
  // two anchored loaded vessels, exercising the placement conversion at the
  // transfer site.  The transfer changes the representation, not the physics.
  plugin->InsertOrKeepLoadedPart(
      403, "outrigger", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      tender_guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + layout_offset), 20 * Milli(Second));
  // A stray vessel spawns loaded, anchorless, in the void: the catch-up must
  // adopt an anchor for it even though it is loaded.
  GUID const stray_guid = "stray";
  plugin->InsertOrKeepVessel(stray_guid, "stray", star_a,
                             /*loaded=*/true, inserted);
  EXPECT_FALSE(plugin->GetVessel(stray_guid)->placement().anchor.has_value());
  plugin->InsertOrKeepLoadedPart(
      405, "stray pod", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      stray_guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin +
                  Displacement<World>({200 * Metre, 0 * Metre, 0 * Metre})),
      20 * Milli(Second));
  // A vessel born of the drifter — an EVA kerbal — inherits the drifter's
  // placement BEFORE its part is inserted: the insertion then converts
  // through the anchored (small) coordinates and the spawn is exact, where
  // the anchorless stray above representably rounds through the 2e16 m
  // absolute (measured ~0.4 m).
  GUID const newborn_guid = "newborn";
  plugin->InsertOrKeepVessel(newborn_guid, "newborn", star_a,
                             /*loaded=*/true, inserted);
  plugin->InheritVesselPlacement(newborn_guid, guid);
  plugin->InsertOrKeepLoadedPart(
      406, "newborn pod", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      newborn_guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin +
                  Displacement<World>({300 * Metre, 0 * Metre, 0 * Metre})),
      20 * Milli(Second));
  // A packed newborn — the unloaded split path — inherits too, and
  // `InsertUnloadedPart` must then express the celestial-relative state in
  // the vessel's anchored placement; without that conversion the part would
  // land a whole anchor offset (~2e16 m) away.
  RelativeDegreesOfFreedom<AliceSun> const drifter_from_star =
      plugin->VesselFromParent(star_a, guid);
  RelativeDegreesOfFreedom<AliceSun> const packed_newborn_from_star(
      drifter_from_star.displacement() +
          Displacement<AliceSun>({500 * Metre, 0 * Metre, 0 * Metre}),
      drifter_from_star.velocity());
  GUID const packed_newborn_guid = "packedborn";
  plugin->InsertOrKeepVessel(packed_newborn_guid, "packedborn", star_a,
                             /*loaded=*/false, inserted);
  plugin->InheritVesselPlacement(packed_newborn_guid, guid);
  plugin->InsertUnloadedPart(407, "packed pod", packed_newborn_guid,
                             packed_newborn_from_star);
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  ASSERT_TRUE(drifter->placement().anchor.has_value());
  ASSERT_TRUE(tender->placement().anchor.has_value());
  EXPECT_TRUE(plugin->GetVessel(stray_guid)->placement().anchor.has_value());

  // The round trip back to `World`: the reference part sits at the World
  // origin, unmoving despite the 3 km/s anchored cruise, and the half-metre
  // geometry survives to sub-mm, void scale notwithstanding — including for
  // the outrigger, which is now another anchored vessel's part and goes
  // through the anchor-to-anchor placement conversion.
  auto const barycentric_to_world = plugin->BarycentricToWorld(
      /*reference_part_is_unmoving=*/true, 402, /*main_body_centre=*/
      std::nullopt);
  DegreesOfFreedom<World> const pod_degrees_of_freedom =
      plugin->GetPartActualMotion(402, 402, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  DegreesOfFreedom<World> const outrigger_degrees_of_freedom =
      plugin->GetPartActualMotion(403, 402, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  EXPECT_THAT((pod_degrees_of_freedom.position() - World::origin).Norm(),
              Lt(1 * Milli(Metre)));
  EXPECT_THAT(pod_degrees_of_freedom.velocity().Norm(),
              Lt(1e-3 * Metre / Second));
  EXPECT_THAT(((outrigger_degrees_of_freedom.position() -
                pod_degrees_of_freedom.position()) -
               layout_offset).Norm(),
              Lt(1 * Milli(Metre)));
  EXPECT_THAT((outrigger_degrees_of_freedom.velocity() -
               pod_degrees_of_freedom.velocity()).Norm(),
              Lt(1e-3 * Metre / Second));

  // The vessel-level World mapping — used by the adapter to place on-rails
  // vessels in the scene — agrees with the part pipeline: the stray,
  // anchored on its own anchor, maps to where its part maps, at rest
  // relative to it, despite the 3 km/s anchored cruise carried by the
  // anchors on both sides of the conversion.
  DegreesOfFreedom<World> const stray_part_degrees_of_freedom =
      plugin->GetPartActualMotion(405, 402, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  DegreesOfFreedom<World> const stray_degrees_of_freedom =
      plugin->VesselWorldDegreesOfFreedom(stray_guid, 402,
                                          barycentric_to_world,
                                          plugin->CurrentTime());
  EXPECT_THAT((stray_degrees_of_freedom.position() -
               stray_part_degrees_of_freedom.position()).Norm(),
              Lt(1 * Milli(Metre)));
  EXPECT_THAT((stray_degrees_of_freedom.velocity() -
               stray_part_degrees_of_freedom.velocity()).Norm(),
              Lt(1e-2 * Metre / Second));

  // The inherited newborn carries the drifter's anchor bit for bit, and its
  // spawn is exact where the anchorless stray rounds through the void
  // absolute (the fail-first for this bound was measured empirically: the
  // stray misses an intended-position assertion at this tolerance by
  // ~0.4 m).
  not_null<Vessel*> const newborn = plugin->GetVessel(newborn_guid);
  ASSERT_TRUE(newborn->placement().anchor.has_value());
  EXPECT_EQ(*drifter->placement().anchor, *newborn->placement().anchor);
  DegreesOfFreedom<World> const newborn_degrees_of_freedom =
      plugin->GetPartActualMotion(406, 402, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  EXPECT_THAT(((newborn_degrees_of_freedom.position() -
                pod_degrees_of_freedom.position()) -
               Displacement<World>({300 * Metre, 0 * Metre, 0 * Metre}))
                  .Norm(),
              Lt(1 * Milli(Metre)));
  EXPECT_THAT((newborn_degrees_of_freedom.velocity() -
               pod_degrees_of_freedom.velocity()).Norm(),
              Lt(1e-3 * Metre / Second));

  // The packed newborn inherited the anchor too — without this tooth an
  // inheritance that silently failed would still pass the position bound
  // below (unanchored, same subsystem: the old path also lands nearby).
  not_null<Vessel*> const packed_newborn =
      plugin->GetVessel(packed_newborn_guid);
  ASSERT_TRUE(packed_newborn->placement().anchor.has_value());
  EXPECT_EQ(*drifter->placement().anchor, *packed_newborn->placement().anchor);
  // Its unloaded insertion survived the anchored placement: it sits ~500 m
  // from the drifter, not an anchor offset away.  The tolerance absorbs the
  // celestial-relative round trip, which collapses the 2e16 m absolute into
  // doubles on both legs.
  EXPECT_THAT((plugin->VesselFromParent(star_a, packed_newborn_guid)
                   .displacement() -
               drifter_from_star.displacement() -
               Displacement<AliceSun>({500 * Metre, 0 * Metre, 0 * Metre}))
                  .Norm(),
              Lt(100 * Metre));
}

// A staging separation in the deep void: the new vessel receiving the
// separated parts inherits the anchor, and the collision-merged pile-up
// re-expresses under one shared anchor instead of dropping to absolutes.
// Without either, the separating pieces round through a ~2×10¹⁶ m absolute
// whose ~4 m ULP scatters them into each other — the in-game fling.
TEST_F(PluginIntegrationTestWithoutPlugin, VoidStagingKeepsAnchors) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  // An anchored two-part vessel deep in the void, cruising at 3 km/s.
  bool inserted;
  GUID const guid = "stack";
  plugin->InsertOrKeepVessel(guid, "stack", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      500, "stack scaffold", guid,
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
  not_null<Vessel*> const stack = plugin->GetVessel(guid);
  ASSERT_TRUE(stack->placement().anchor.has_value());
  plugin->AdvanceTime(plugin->CurrentTime() + 60 * Second, 1 * Radian);

  // The vessel loads with two parts 2 m apart.
  plugin->InsertOrKeepVessel(guid, "stack", star_a, /*loaded=*/true, inserted);
  Mass const mass = 1000 * Kilogram;
  DegreesOfFreedom<World> const main_body_degrees_of_freedom =
      plugin->CelestialWorldDegreesOfFreedom(
          star_a, 500,
          plugin->BarycentricToWorld(/*reference_part_is_unmoving=*/true, 500,
                                     /*main_body_centre=*/std::nullopt),
          plugin->CurrentTime());
  Displacement<World> const stage_offset({2 * Metre, 0 * Metre, 0 * Metre});
  auto const part_motion = [](Position<World> const& position) {
    return RigidMotion<EccentricPart, World>(
        RigidTransformation<EccentricPart, World>(
            EccentricPart::origin,
            position,
            OrthogonalMap<EccentricPart, World>::Identity()),
        World::nonrotating,
        World::unmoving);
  };
  plugin->InsertOrKeepLoadedPart(
      501, "upper stage", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin), 20 * Milli(Second));
  plugin->InsertOrKeepLoadedPart(
      502, "booster", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + stage_offset), 20 * Milli(Second));
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(501, 502);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  ASSERT_TRUE(stack->placement().anchor.has_value());
  auto const stack_anchor = *stack->placement().anchor;
  // Captured for the bit-identity assertion at the end: the fix keeps the
  // anchor VERBATIM through the whole staging choreography, whereas a
  // drop/re-adopt cycle would mint a new anchor with a new epoch.
  auto const original_anchor = stack_anchor;
  auto const part_q = [](not_null<Vessel*> const vessel, PartId const id) {
    return vessel->part(id)->rigid_motion()(
        {RigidPart::origin, RigidPart::unmoving}).position();
  };
  // Diagnostic bisection point (a): the loaded stack's parts are 2 m apart in
  // the anchored representation.
  EXPECT_THAT(((part_q(stack, 502) - part_q(stack, 501)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(a) after load-phase catch-up";

  // Staging happens on the next frame — the game advances the clock before
  // reporting the scene, and the loaded bookkeeping stamps part states one
  // tick in the past.  The booster becomes a fresh loaded vessel, still
  // touching the stack (the separation contact is reported as a collision,
  // merging the pile-ups for this tick).
  plugin->AdvanceTime(plugin->CurrentTime() + 20 * Milli(Second), 1 * Radian);
  GUID const booster_guid = "booster";
  plugin->InsertOrKeepVessel(guid, "stack", star_a, /*loaded=*/true, inserted);
  plugin->InsertOrKeepVessel(booster_guid, "booster", star_a,
                             /*loaded=*/true, inserted);
  plugin->InsertOrKeepLoadedPart(
      501, "upper stage", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin), 20 * Milli(Second));
  plugin->InsertOrKeepLoadedPart(
      502, "booster", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      booster_guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + stage_offset), 20 * Milli(Second));
  not_null<Vessel*> const booster = plugin->GetVessel(booster_guid);
  // The fresh vessel inherited the anchor at the part transfer.
  ASSERT_TRUE(booster->placement().anchor.has_value());
  EXPECT_EQ(stack_anchor, *booster->placement().anchor);
  // Diagnostic bisection point (b): the transfer preserved the geometry.
  EXPECT_THAT(((part_q(booster, 502) - part_q(stack, 501)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(b) after the transfer, before the merge";
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(501, 502);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  // Diagnostic bisection point (c): the merge preserved the geometry.
  EXPECT_THAT(((part_q(booster, 502) - part_q(stack, 501)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(c) after the merge, before catch-up";
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  // Diagnostic bisection point (d): catch-up preserved the geometry.
  EXPECT_THAT(((part_q(booster, 502) - part_q(stack, 501)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(d) after catch-up";

  // The in-game fling was a drop/adopt cycle over SEVERAL frames of
  // persistent separation contact; play three more such frames.
  for (int frame = 0; frame < 3; ++frame) {
    plugin->AdvanceTime(plugin->CurrentTime() + 20 * Milli(Second),
                        1 * Radian);
    plugin->InsertOrKeepVessel(guid, "stack", star_a,
                               /*loaded=*/true, inserted);
    plugin->InsertOrKeepVessel(booster_guid, "booster", star_a,
                               /*loaded=*/true, inserted);
    plugin->InsertOrKeepLoadedPart(
        501, "upper stage", mass, EccentricPart::origin,
        MakeWaterSphereInertiaTensor(mass),
        /*is_solid_rocket_motor=*/false,
        guid, star_a, main_body_degrees_of_freedom,
        part_motion(World::origin), 20 * Milli(Second));
    plugin->InsertOrKeepLoadedPart(
        502, "booster", mass, EccentricPart::origin,
        MakeWaterSphereInertiaTensor(mass),
        /*is_solid_rocket_motor=*/false,
        booster_guid, star_a, main_body_degrees_of_freedom,
        part_motion(World::origin + stage_offset), 20 * Milli(Second));
    plugin->PrepareToReportCollisions();
    plugin->ReportPartCollision(501, 502);
    plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    ASSERT_TRUE(stack->placement().anchor.has_value()) << "frame " << frame;
    ASSERT_TRUE(booster->placement().anchor.has_value()) << "frame " << frame;
    EXPECT_EQ(*stack->placement().anchor, *booster->placement().anchor)
        << "frame " << frame;
  }

  // The collision-merged subset kept a single shared anchor rather than
  // dropping to absolutes — indeed the anchor is BIT-IDENTICAL to the one
  // adopted before staging: no drop/re-adopt cycle happened at all.
  ASSERT_TRUE(stack->placement().anchor.has_value());
  ASSERT_TRUE(booster->placement().anchor.has_value());
  EXPECT_EQ(*stack->placement().anchor, *booster->placement().anchor);
  EXPECT_EQ(original_anchor, *stack->placement().anchor);
  EXPECT_EQ(original_anchor, *booster->placement().anchor);

  // The separation geometry survives to sub-mm: the pieces are 2 m apart,
  // not scattered by the ~4 m ULP of the void absolute.
  auto const barycentric_to_world = plugin->BarycentricToWorld(
      /*reference_part_is_unmoving=*/true, 501, /*main_body_centre=*/
      std::nullopt);
  DegreesOfFreedom<World> const upper_degrees_of_freedom =
      plugin->GetPartActualMotion(501, 501, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  DegreesOfFreedom<World> const booster_degrees_of_freedom =
      plugin->GetPartActualMotion(502, 501, barycentric_to_world)(
          {EccentricPart::origin, EccentricPart::unmoving});
  EXPECT_THAT(((booster_degrees_of_freedom.position() -
                upper_degrees_of_freedom.position()) -
               stage_offset).Norm(),
              Lt(1 * Milli(Metre)));
  EXPECT_THAT((booster_degrees_of_freedom.velocity() -
               upper_degrees_of_freedom.velocity()).Norm(),
              Lt(1e-3 * Metre / Second));
}

// A staging separation after the parent vessel dominance-rebased away from
// its stock parent celestial's subsystem — the in-game norm for a vessel deep
// in the void, whose stock parent stays the departure star while dominance
// hands it to the destination.  The fresh vessel created to receive the
// separated part is born with the STOCK parent's subsystem, so inheritance
// must carry the splitting vessel's whole placement — subsystem and anchor —
// or the transfer rounds the part through a ~3×10¹⁶ m absolute (~4 m ULP)
// and the pieces scatter — the observed in-game staging failure.
TEST_F(PluginIntegrationTestWithoutPlugin,
       VoidStagingInheritsPlacementAcrossRebase) {
  Index const star_a = 0;
  Index const star_b = 1;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  // An unloaded vessel 4/5 of the way to star B: star B dominates the
  // equal-mass pair 16:1, far beyond the hysteresis margin, so the first
  // catch-up rebases the vessel to star B's subsystem and adopts an anchor —
  // while its stock parent celestial remains star A.
  bool inserted;
  GUID const guid = "deepstack";
  plugin->InsertOrKeepVessel(guid, "deepstack", star_a,
                             /*loaded=*/false, inserted);
  Displacement<AliceSun> const b_from_a =
      plugin->CelestialFromParent(star_b).displacement();
  plugin->InsertUnloadedPart(
      600, "deep scaffold", guid,
      {0.8 * b_from_a,
       Velocity<AliceSun>({0 * Metre / Second,
                           3 * Kilo(Metre) / Second,
                           0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  not_null<Vessel*> const stack = plugin->GetVessel(guid);
  ASSERT_TRUE(stack->placement().anchor.has_value());
  ASSERT_EQ(plugin->GetCelestial(star_b).subsystem(),
            stack->placement().subsystem);
  ASSERT_NE(plugin->GetCelestial(star_a).subsystem(),
            stack->placement().subsystem);
  plugin->AdvanceTime(plugin->CurrentTime() + 60 * Second, 1 * Radian);

  // The vessel loads with two parts 2 m apart; the stock main body handed to
  // the loaded path is still star A.
  plugin->InsertOrKeepVessel(guid, "deepstack", star_a,
                             /*loaded=*/true, inserted);
  Mass const mass = 1000 * Kilogram;
  DegreesOfFreedom<World> const main_body_degrees_of_freedom =
      plugin->CelestialWorldDegreesOfFreedom(
          star_a, 600,
          plugin->BarycentricToWorld(/*reference_part_is_unmoving=*/true, 600,
                                     /*main_body_centre=*/std::nullopt),
          plugin->CurrentTime());
  Displacement<World> const stage_offset({2 * Metre, 0 * Metre, 0 * Metre});
  auto const part_motion = [](Position<World> const& position) {
    return RigidMotion<EccentricPart, World>(
        RigidTransformation<EccentricPart, World>(
            EccentricPart::origin,
            position,
            OrthogonalMap<EccentricPart, World>::Identity()),
        World::nonrotating,
        World::unmoving);
  };
  plugin->InsertOrKeepLoadedPart(
      601, "deep upper stage", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin), 20 * Milli(Second));
  plugin->InsertOrKeepLoadedPart(
      602, "deep booster", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + stage_offset), 20 * Milli(Second));
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(601, 602);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  ASSERT_TRUE(stack->placement().anchor.has_value());
  auto const stack_anchor = *stack->placement().anchor;
  auto const part_q = [](not_null<Vessel*> const vessel, PartId const id) {
    return vessel->part(id)->rigid_motion()(
        {RigidPart::origin, RigidPart::unmoving}).position();
  };
  EXPECT_THAT(((part_q(stack, 602) - part_q(stack, 601)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(a) after load-phase catch-up";

  // Staging on the next frame: the booster becomes a fresh loaded vessel,
  // created — as the adapter does — with the STOCK parent celestial, star A.
  plugin->AdvanceTime(plugin->CurrentTime() + 20 * Milli(Second), 1 * Radian);
  GUID const booster_guid = "deepbooster";
  plugin->InsertOrKeepVessel(guid, "deepstack", star_a,
                             /*loaded=*/true, inserted);
  plugin->InsertOrKeepVessel(booster_guid, "deepbooster", star_a,
                             /*loaded=*/true, inserted);
  plugin->InsertOrKeepLoadedPart(
      601, "deep upper stage", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin), 20 * Milli(Second));
  plugin->InsertOrKeepLoadedPart(
      602, "deep booster", mass, EccentricPart::origin,
      MakeWaterSphereInertiaTensor(mass),
      /*is_solid_rocket_motor=*/false,
      booster_guid, star_a, main_body_degrees_of_freedom,
      part_motion(World::origin + stage_offset), 20 * Milli(Second));
  not_null<Vessel*> const booster = plugin->GetVessel(booster_guid);
  // The whole placement was inherited at the transfer: subsystem AND anchor.
  EXPECT_EQ(stack->placement().subsystem, booster->placement().subsystem);
  ASSERT_TRUE(booster->placement().anchor.has_value());
  EXPECT_EQ(stack_anchor, *booster->placement().anchor);
  EXPECT_THAT(((part_q(booster, 602) - part_q(stack, 601)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(b) after the transfer, before the merge";
  plugin->PrepareToReportCollisions();
  plugin->ReportPartCollision(601, 602);
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  EXPECT_THAT(((part_q(booster, 602) - part_q(stack, 601)).Norm()),
              AbsoluteErrorFrom(2 * Metre, Lt(1 * Milli(Metre))))
      << "(c) after the merge and catch-up";
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
  ASSERT_TRUE(drifter->placement().anchor.has_value());

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

// R1/WS6-5/6: a flight plan created from an anchored void vessel must coast
// weightlessly, not plunge.  Its coast segment is flowed from the near-origin
// anchored coordinates by `FlightPlan::CoastSegment`; the pre-fix code carried
// `subsystem_` but no anchor (the F1 defect's live sixth recurrence), so the
// integrator read the coordinates as subsystem-relative and the plan plunged
// into the home star.  `SubsystemPlacement` now carries the anchor into every
// flow, so the coast is force-free.
TEST_F(PluginIntegrationTestWithoutPlugin, AnchoredVoidFlightPlanCoasts) {
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
  ASSERT_TRUE(drifter->placement().anchor.has_value());

  drifter->CreateFlightPlan(
      drifter->trajectory().back().time + 1 * Hour,
      1 * Kilogram,
      Ephemeris<Barycentric>::AdaptiveStepParameters(
          EmbeddedExplicitRungeKuttaNyströmIntegrator<
              DormandالمكاوىPrince1986RKN434FM,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          /*max_steps=*/1000,
          /*length_integration_tolerance=*/1 * Milli(Metre),
          /*speed_integration_tolerance=*/1 * Milli(Metre) / Second),
      Ephemeris<Barycentric>::GeneralizedAdaptiveStepParameters(
          EmbeddedExplicitGeneralizedRungeKuttaNyströmIntegrator<
              Fine1987RKNG34,
              Ephemeris<Barycentric>::GeneralizedNewtonianMotionEquation>(),
          /*max_steps=*/1000,
          /*length_integration_tolerance=*/1 * Milli(Metre),
          /*speed_integration_tolerance=*/1 * Milli(Metre) / Second));
  ASSERT_TRUE(drifter->has_flight_plan());
  auto const& flight_plan = drifter->flight_plan();
  // No manœuvres: a single coast segment to the desired final time.
  ASSERT_EQ(1, flight_plan.number_of_segments());
  auto const coast = flight_plan.GetSegment(0);
  Speed const coasting_gain =
      (coast->back().degrees_of_freedom.velocity() -
       coast->front().degrees_of_freedom.velocity())
          .Norm();
  // A plunge would show a huge gain toward the origin.
  EXPECT_THAT(coasting_gain, Lt(1 * Milli(Metre) / Second));
}

// WS6-6 (capstone): two vessels a few metres apart deep in the inter-stellar
// void each adopt their own anchor.  The anchor is what makes a mm-scale
// rendezvous representable at all: a vessel's *stored* coordinates stay near
// its local origin (≈ 0), so the ULP of its own dynamics is sub-micron — mm
// motions are resolvable — whereas an unanchored void vessel would carry ~2e16 m
// coordinates whose ULP is ~4 m, quantizing away any metres-scale manœuvre.
// The anchor also records where each vessel truly is, so the render / map-view /
// ClosestApproaches placement recovers a *local* rendezvous (the two vessels
// metres apart), not a ~2e16 m-scale artifact.  (With sector anchors the
// offsets of the two anchors difference exactly; the residual visible below is
// the quantization of the *absolute* insertion positions at the plugin
// boundary, which no internal representation can repair.  See
// `CoMovingVoidVesselsConversionIsExact` for the relative-geometry claim.)
TEST_F(PluginIntegrationTestWithoutPlugin, CoAnchoredVoidVesselsKeepLocalPrecision) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  Length const midpoint = 2e16 * Metre;

  bool inserted;
  GUID const guid_a = "station";
  plugin->InsertOrKeepVessel(guid_a, "station", star_a, /*loaded=*/false,
                             inserted);
  plugin->InsertUnloadedPart(
      401, "part-a", guid_a,
      {Displacement<AliceSun>({midpoint, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  GUID const guid_b = "visitor";
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_a, /*loaded=*/false,
                             inserted);
  plugin->InsertUnloadedPart(
      402, "part-b", guid_b,
      {Displacement<AliceSun>({midpoint + 5 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  not_null<Vessel*> const station = plugin->GetVessel(guid_a);
  not_null<Vessel*> const visitor = plugin->GetVessel(guid_b);
  ASSERT_TRUE(station->placement().anchor.has_value());
  ASSERT_TRUE(visitor->placement().anchor.has_value());

  // Each vessel's stored coordinates stay near its own local origin.  At this
  // magnitude the ULP is sub-micron, so the vessel's own dynamics resolve mm
  // motions — the property the anchor exists to provide.  Unanchored, these
  // would be ~2e16 m (ULP ~4 m).
  auto const& [t_a, dof_a] = station->trajectory().back();
  auto const& [t_b, dof_b] = visitor->trajectory().back();
  EXPECT_THAT((dof_a.position() - Barycentric::origin).Norm(), Lt(1e9 * Metre));
  EXPECT_THAT((dof_b.position() - Barycentric::origin).Norm(), Lt(1e9 * Metre));

  // The anchor records the true void position: adding it back places each
  // vessel at the ~2e16 m midpoint (render placement is correct, not collapsed
  // onto the home star at the origin).
  Position<Barycentric> const true_a =
      dof_a.position() + station->placement().anchor->OffsetAt(t_a);
  Position<Barycentric> const true_b =
      dof_b.position() + visitor->placement().anchor->OffsetAt(t_b);
  EXPECT_THAT((true_a - Barycentric::origin).Norm(),
              AllOf(Gt(1e16 * Metre), Lt(3e16 * Metre)));
  EXPECT_THAT((true_b - Barycentric::origin).Norm(),
              AllOf(Gt(1e16 * Metre), Lt(3e16 * Metre)));

  // The two anchored vessels are recovered as a *local* rendezvous — metres
  // apart, not ~2e16 m — so a closest-approach plot sees their true relative
  // geometry.  The exact separation carries the quantization of the absolute
  // insertion positions (a few metres); the point is it is not a void-scale
  // artifact.
  EXPECT_THAT((true_b - true_a).Norm(), Lt(1 * Kilo(Metre)));
}

// Sector anchors: the offsets of two anchors difference exactly in their cell
// part, so the relative geometry of two anchored void vessels — read through
// `Anchor::Conversion` — tracks a slow rendezvous approach to sub-mm.
// Differencing two *collapsed* offsets instead (the pre-sector
// representation, and the failure this test is first against) rounds each
// collapse at the ~4 m ULP of the void distance: the smooth approach is
// snapped to the void grid, metres away from the true relative geometry.
TEST_F(PluginIntegrationTestWithoutPlugin,
       ApproachingVoidVesselsConversionIsExact) {
  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  Length const midpoint = 2e16 * Metre;
  // An interstellar cruise along the void axis; the visitor closes on the
  // station at half a metre per second.
  Velocity<AliceSun> const cruise({3 * Kilo(Metre) / Second,
                                   0 * Metre / Second,
                                   0 * Metre / Second});
  Velocity<AliceSun> const approach({-0.5 * Metre / Second,
                                     0 * Metre / Second,
                                     0 * Metre / Second});

  bool inserted;
  GUID const guid_a = "station";
  plugin->InsertOrKeepVessel(guid_a, "station", star_a, /*loaded=*/false,
                             inserted);
  plugin->InsertUnloadedPart(
      411, "part-a", guid_a,
      {Displacement<AliceSun>({midpoint, 0 * Metre, 0 * Metre}), cruise});
  GUID const guid_b = "visitor";
  plugin->InsertOrKeepVessel(guid_b, "visitor", star_a, /*loaded=*/false,
                             inserted);
  plugin->InsertUnloadedPart(
      412, "part-b", guid_b,
      {Displacement<AliceSun>({midpoint + 5 * Metre, 0 * Metre, 0 * Metre}),
       cruise + approach});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  not_null<Vessel*> const station = plugin->GetVessel(guid_a);
  not_null<Vessel*> const visitor = plugin->GetVessel(guid_b);
  ASSERT_TRUE(station->placement().anchor.has_value());
  ASSERT_TRUE(visitor->placement().anchor.has_value());
  Velocity<Barycentric> const relative_velocity =
      visitor->placement().anchor->velocity -
      station->placement().anchor->velocity;

  Instant const t0 = station->trajectory().back().time;
  auto const conversion_at = [&](Instant const& t) {
    return Ephemeris<Barycentric>::Anchor::Conversion(
        visitor->placement().anchor, station->placement().anchor, t).first;
  };
  auto const paired_collapse_at = [&](Instant const& t) {
    return visitor->placement().anchor->OffsetAt(t) -
           station->placement().anchor->OffsetAt(t);
  };

  Displacement<Barycentric> const conversion₀ = conversion_at(t0);
  Displacement<Barycentric> const paired₀ = paired_collapse_at(t0);
  Length max_conversion_error;
  Length max_paired_error;
  for (int minute = 30; minute <= 360; minute += 30) {
    Instant const t = t0 + minute * Minute;
    // The true change in the anchors' relative placement is the linear
    // approach.
    Displacement<Barycentric> const true_change =
        relative_velocity * (t - t0);
    max_conversion_error =
        std::max(max_conversion_error,
                 (conversion_at(t) - conversion₀ - true_change).Norm());
    max_paired_error =
        std::max(max_paired_error,
                 (paired_collapse_at(t) - paired₀ - true_change).Norm());
  }
  LOG(INFO) << "conversion error: " << max_conversion_error
            << ", paired-collapse error: " << max_paired_error;

  // The conversion differences the cells exactly and the small terms at
  // their own magnitude: the approach is resolved to sub-mm.
  EXPECT_THAT(max_conversion_error, Lt(1 * Milli(Metre)));
  // The pre-sector path snaps the approach to the ~4 m void grid.
  EXPECT_THAT(max_paired_error, Gt(1 * Metre));
}

// The affine-term re-anchor: a long anchored coast grows the anchor's
// v (t - epoch) term; when it crosses the re-anchor bound (lowered here
// through the test seam so that a minute of coast crosses it) the vessel
// adopts a fresh anchor, folding the sector parts.  The fold must be
// continuous: the represented position q + A(t) is preserved across the
// event, and because the anchored velocity of a force-free coast is zero and
// the anchors difference exactly on the lattice, the fold perturbation is
// directly measurable — at sub-mm — as (q2 - q1) + Conversion(A2, A1, t2).
TEST_F(PluginIntegrationTestWithoutPlugin, LongCoastReAnchorsContinuously) {
  PileUp::re_anchor_bound_for_testing_ = 1e5 * Metre;
  absl::Cleanup restore_bound = [] {
    PileUp::re_anchor_bound_for_testing_ = 1e12 * Metre;
  };

  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  bool inserted;
  GUID const guid_void = "drifter";
  plugin->InsertOrKeepVessel(guid_void, "drifter", star_a,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      421, "part-drifter", guid_void,
      {Displacement<AliceSun>({2e16 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>({3 * Kilo(Metre) / Second,
                           0 * Metre / Second,
                           0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  not_null<Vessel*> const drifter = plugin->GetVessel(guid_void);
  ASSERT_TRUE(drifter->placement().anchor.has_value());
  auto const anchor_1 = *drifter->placement().anchor;
  Instant const t1 = drifter->trajectory().back().time;
  DegreesOfFreedom<Barycentric> const dof1 =
      drifter->trajectory().back().degrees_of_freedom;

  // A minute of coast at 3 km/s: the affine term reaches 1.8e5 m, crossing
  // the lowered bound.
  Instant t = t1;
  for (int i = 0; i < 6; ++i) {
    t += 10 * Second;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(guid_void, "drifter", star_a,
                               /*loaded=*/false, inserted);
    plugin->PrepareToReportCollisions();
    plugin->FreeVesselsAndPartsAndCollectPileUps(10 * Second);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  ASSERT_TRUE(drifter->placement().anchor.has_value());
  auto const anchor_2 = *drifter->placement().anchor;
  Instant const t2 = drifter->trajectory().back().time;
  DegreesOfFreedom<Barycentric> const dof2 =
      drifter->trajectory().back().degrees_of_freedom;

  // The trigger fired: the anchor was refreshed at a later epoch, and the
  // fresh affine term is small again.
  EXPECT_GT(anchor_2.epoch, anchor_1.epoch);
  EXPECT_THAT((anchor_2.velocity * (t2 - anchor_2.epoch)).Norm(),
              Lt(1e5 * Metre));

  // Continuity of the represented position across the re-anchor(s); see the
  // derivation in the test comment.  The fold error scales with the folded
  // magnitudes — at the lowered bound they are ~1e5 m, so a correct fold is
  // exact to well below a micron here; the micron threshold catches formula
  // and sign errors, while the sub-mm bound of the production-scale fold
  // (magnitudes ~1e12 m) is established by the rounding analysis in
  // `AdoptAnchor`, not by this test.
  Displacement<Barycentric> const continuity_error =
      (dof2.position() - dof1.position()) +
      Ephemeris<Barycentric>::Anchor::Conversion(anchor_2, anchor_1, t2).first;
  EXPECT_THAT(continuity_error.Norm(), Lt(1 * Micro(Metre)));
}

// R2 drop-path golden fixture.  The ">20k-point history can't be cheaply
// verified" narrative (sessions 8, 11, 18) was a test-seam gap, not a real
// barrier: `Vessel::max_points_to_serialize_for_testing_` is the only obstacle.
// Lowered here, a SHORT anchored burn is actually forgotten on save and
// reconstructed by reanimation — the exact path WS6-4's Checkpoint anchor field
// serves.  Fail-first character: without that anchor the checkpoint's
// near-origin anchored coordinates are re-integrated as subsystem-relative (on
// top of the home star), so the reconstruction plunges and diverges from the
// actual burn arc by astronomical distances.  With it, the reconstruction
// matches the actual state to sub-millimetre.  This is the shared harness that
// would have caught s8's mistaken accept, s11 #2, and WS6-4 fail-first.
TEST_F(PluginIntegrationTestWithoutPlugin, AnchoredBurnSurvivesDropAndReanimation) {
  // Force the collapsible-history drop with a handful of points instead of
  // ~20'000, then restore the production value for the rest of the suite.
  Vessel::max_points_to_serialize_for_testing_ = 2;
  absl::Cleanup restore_max_points = [] {
    Vessel::max_points_to_serialize_for_testing_ = 20'000;
  };

  Index const star_a = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  InsertTwoStarVoid(*plugin);

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                             /*loaded=*/false, inserted);
  // Deep in the void between the two stars, so the vessel anchors.
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({2e16 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>()});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  ASSERT_TRUE(plugin->GetVessel(vessel_guid)->placement().anchor.has_value());

  Force const thrust = 1 * Newton;
  SpecificImpulse const specific_impulse = 1e4 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;
  Time const δt = 100 * Second;
  Mass m = 1 * Kilogram;
  Instant const t_start;
  Instant t = t_start;
  // In the first burn cycle, so it is dropped and must be reanimated.
  Instant const t_sample = t_start + 2 * δt;

  // Several burn/coast cycles.  Each burn is a non-collapsible arc; clearing it
  // and coasting transitions the vessel back to collapsible, and that
  // transition (detected in `FreeVesselsAndPartsAndCollectPileUps`) writes a
  // checkpoint.  A single checkpoint is fully restored on load, so it is the
  // *older* checkpoints that reanimation must walk back and reconstruct — hence
  // several cycles.
  for (int cycle = 0; cycle < 5; ++cycle) {
    for (int i = 0; i < 3; ++i) {
      t += δt;
      plugin->AdvanceTime(t, 1 * Radian);
      plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                                 /*loaded=*/false, inserted);
      plugin->SetVesselOnRailsBurn(vessel_guid, thrust, specific_impulse,
                                   /*initial_mass=*/m,
                                   Vector<double, World>({0, 1, 0}),
                                   /*max_duration=*/1 * Hour);
      VesselSet collided_vessels;
      plugin->CatchUpLaggingVessels(collided_vessels);
      m -= δt * mass_flow;
    }
    plugin->ClearVesselOnRailsBurn(vessel_guid);
    for (int i = 0; i < 2; ++i) {
      t += δt;
      plugin->AdvanceTime(t, 1 * Radian);
      plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                                 /*loaded=*/false, inserted);
      plugin->PrepareToReportCollisions();
      plugin->FreeVesselsAndPartsAndCollectPileUps(δt);
      VesselSet collided_vessels;
      plugin->CatchUpLaggingVessels(collided_vessels);
    }
  }

  // The actual (burn-affected) state at the sample instant.
  DegreesOfFreedom<Barycentric> const actual =
      plugin->GetVessel(vessel_guid)->trajectory().EvaluateDegreesOfFreedom(
          t_sample);
  EXPECT_THAT(actual.velocity().Norm(), Gt(100 * Metre / Second));

  // Save and reload; the earlier burn cycles must now actually be dropped.
  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  auto const plugin2 = Plugin::ReadFromMessage(message);
  auto const vessel2 = plugin2->GetVessel(vessel_guid);
  ASSERT_GT(vessel2->trajectory().t_min(), t_sample)
      << "the drop did not engage — the sample was still serialized, so the "
         "reconstruction path is not exercised";

  // Reanimate the forgotten past and confirm the sample was reconstructed
  // through the anchor rather than plunging into the home star.
  vessel2->AwaitReanimation(t_start, /*quiet=*/true);
  ASSERT_LE(vessel2->trajectory().t_min(), t_sample);
  ASSERT_TRUE(vessel2->placement().anchor.has_value());
  DegreesOfFreedom<Barycentric> const reconstructed =
      vessel2->trajectory().EvaluateDegreesOfFreedom(t_sample);
  Length const position_divergence =
      (reconstructed.position() - actual.position()).Norm();
  Speed const velocity_divergence =
      (reconstructed.velocity() - actual.velocity()).Norm();
  LOG(ERROR) << "Reanimation-vs-actual after a forced drop — position: "
             << position_divergence << ", velocity: " << velocity_divergence;
  // Reanimation re-integrates the dropped collapsible coasts, so the match is
  // metre-scale, not bit-exact (cf. `VesselTest.Reanimator`, which only checks
  // continuity).  The anchored trajectory is expressed in near-origin local
  // coordinates (~20 km from the barycentric origin, the anchor carrying the
  // ~2e16 m void offset separately); WITHOUT the checkpoint anchor the
  // reconstruction re-integrates those near-origin coordinates as
  // subsystem-relative, so the home star's field applies at ~20 km and the
  // reconstruction diverges catastrophically.  These finite bounds are the
  // discriminator.
  EXPECT_THAT(position_divergence, Lt(10 * Metre));
  EXPECT_THAT(velocity_divergence, Lt(0.1 * Metre / Second));
}

// Rendering a trajectory local to a star 40 light-years out must not quantize
// its shape: the render transformation used to anchor its origin at the sun,
// 3.848e17 m away from the plotted geometry, so every vertex was rounded at
// the ~64 m ULP of that magnitude and the km-scale shape danced by metres as
// the rounding phase churned — the in-game map-view jitter.  The rotation
// part of the transformation preserves distances, so the World-space
// separation of adjacent rendered vertices must reproduce their barycentric
// separation far below the ULP of the interstellar distance.
TEST_F(PluginIntegrationTestWithoutPlugin, InterstellarRenderShapeIsExact) {
  Index const star_a = 0;
  Index const star_b = 1;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  auto const insert_star = [&plugin](Index const index,
                                     std::optional<Index> const parent_index,
                                     std::string const& name,
                                     std::string const& x) {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name                    : ")", name, R"("
                        gravitational_parameter : "1.19e19 m^3/s^2"
                        reference_instant       : "JD2451545.0"
                        mean_radius             : "8.4e7 m"
                        axis_right_ascension    : "0 deg"
                        axis_declination        : "90 deg"
                        reference_angle         : "0 rad"
                        angular_frequency       : "1 rad/s")"),
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name : ")", name, R"("
                        x    : ")", x, R"("
                        y    : "0 m"
                        z    : "0 m"
                        vx   : "0 m/s"
                        vy   : "0 m/s"
                        vz   : "0 m/s")"),
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(
        index, parent_index, gravity_model, initial_state);
  };
  insert_star(star_a, std::nullopt, "star A", "0 m");
  insert_star(star_b, star_a, "star B", "3.848e17 m");
  plugin->EndInitialization();
  EXPECT_NE(plugin->GetCelestial(star_a).subsystem(),
            plugin->GetCelestial(star_b).subsystem());

  // A vessel in a low orbit around the remote star.
  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_b,
                             /*loaded=*/false, inserted);
  Length const orbit_radius = 1.5829e9 * Metre;
  Speed const v_circular =
      Sqrt(1.19e19 * Pow<3>(Metre) / Pow<2>(Second) / orbit_radius);
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({orbit_radius, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>(
           {0 * Metre / Second, v_circular, 0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  // A third of an orbit of coast, so the downsampled trajectory retains a
  // curved arc of many points.
  Time const δt = 1200 * Second;
  Instant t;
  for (int i = 0; i < 30; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_b,
                               /*loaded=*/false, inserted);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  auto const& vessel = *plugin->GetVessel(vessel_guid);
  auto const& trajectory = vessel.trajectory();
  plugin->renderer().SetPlottingFrame(
      plugin->NewBodyCentredNonRotatingNavigationFrame(star_b));
  // In game the World origin floats with the active vessel, so the sun's
  // World position is interstellar-huge and the rendered points land near the
  // origin, where they are finely representable.  Reproduce that: a first
  // render anchored at the origin locates the geometry, a second render
  // shifts the sun so the geometry lands at the origin.  (Without the shift
  // the OUTPUT coordinates would sit at 4e17 m and quantize at their own ULP
  // no matter how the transformation computes.)
  auto const locate = plugin->renderer().RenderBarycentricTrajectoryInWorld(
      plugin->CurrentTime(),
      trajectory.begin(),
      trajectory.end(),
      World::origin,
      plugin->PlanetariumRotation(),
      vessel.placement());
  ASSERT_GT(locate.size(), 3);
  Position<World> const sun_world_position =
      World::origin -
      (locate.front().degrees_of_freedom.position() - World::origin);
  auto const rendered = plugin->renderer().RenderBarycentricTrajectoryInWorld(
      plugin->CurrentTime(),
      trajectory.begin(),
      trajectory.end(),
      sun_world_position,
      plugin->PlanetariumRotation(),
      vessel.placement());
  ASSERT_GT(rendered.size(), 3);

  // The rendered shape reproduces the barycentric shape: adjacent-vertex
  // separations agree to sub-millimetre (the rotation is an isometry), far
  // below the ~64 m ULP quantization of the pre-fix sun-anchored transform.
  auto it_rendered = rendered.begin();
  auto it_barycentric = trajectory.begin();
  std::optional<Position<World>> previous_rendered;
  std::optional<Position<Barycentric>> previous_barycentric;
  Length max_shape_error;
  for (; it_rendered != rendered.end(); ++it_rendered, ++it_barycentric) {
    Position<World> const rendered_position =
        it_rendered->degrees_of_freedom.position();
    Position<Barycentric> const barycentric_position =
        it_barycentric->degrees_of_freedom.position();
    if (previous_rendered.has_value()) {
      Length const rendered_separation =
          (rendered_position - *previous_rendered).Norm();
      Length const barycentric_separation =
          (barycentric_position - *previous_barycentric).Norm();
      max_shape_error =
          std::max(max_shape_error,
                   Abs(rendered_separation - barycentric_separation));
    }
    previous_rendered = rendered_position;
    previous_barycentric = barycentric_position;
  }
  EXPECT_THAT(max_shape_error, Lt(1 * Milli(Metre)));
}

// The float32 vertex buffer quantizes each plotted vertex at the ULP of its
// distance to the scaled-space origin — which the game moves around (stock
// and KSPCF recentre it at the camera focus, not at the plotted geometry): at
// a star focus the absolute vertices of a 1.58e9 m orbit quantized at ~190 m
// real and the drawn line churned frame to frame; with the origin left at
// interstellar magnitude they quantized at ~2.5e10 m and the line shattered.
// Anchored plotting must instead bound every vertex's error by the ULP of its
// distance to the camera — angularly sub-pixel from the actual viewpoint —
// wherever the origin is.
TEST_F(PluginIntegrationTestWithoutPlugin,
       PlotMethod4AnchorsVerticesAtTheCamera) {
  Index const star_a = 0;
  Index const star_b = 1;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  auto const insert_star = [&plugin](Index const index,
                                     std::optional<Index> const parent_index,
                                     std::string const& name,
                                     std::string const& x) {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name                    : ")", name, R"("
                        gravitational_parameter : "1.19e19 m^3/s^2"
                        reference_instant       : "JD2451545.0"
                        mean_radius             : "8.4e7 m"
                        axis_right_ascension    : "0 deg"
                        axis_declination        : "90 deg"
                        reference_angle         : "0 rad"
                        angular_frequency       : "1 rad/s")"),
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name : ")", name, R"("
                        x    : ")", x, R"("
                        y    : "0 m"
                        z    : "0 m"
                        vx   : "0 m/s"
                        vy   : "0 m/s"
                        vz   : "0 m/s")"),
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(
        index, parent_index, gravity_model, initial_state);
  };
  insert_star(star_a, std::nullopt, "star A", "0 m");
  insert_star(star_b, star_a, "star B", "3.848e17 m");
  plugin->EndInitialization();

  // A vessel in a low orbit around the remote star.
  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_b,
                             /*loaded=*/false, inserted);
  Length const orbit_radius = 1.5829e9 * Metre;
  Speed const v_circular =
      Sqrt(1.19e19 * Pow<3>(Metre) / Pow<2>(Second) / orbit_radius);
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({orbit_radius, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>(
           {0 * Metre / Second, v_circular, 0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));

  // A third of an orbit of coast, so the downsampled trajectory retains a
  // curved arc of many points.
  Time const δt = 1200 * Second;
  Instant t;
  for (int i = 0; i < 30; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_b,
                               /*loaded=*/false, inserted);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  auto const& vessel = *plugin->GetVessel(vessel_guid);
  auto const& trajectory = vessel.trajectory();
  plugin->renderer().SetPlottingFrame(
      plugin->NewBodyCentredNonRotatingNavigationFrame(star_b));

  // The scaled-space origins whose choice must not affect the vertex error:
  // recentred at the star focus, recentred at the vessel, and left at the sun
  // of a system 3.848e17 m away (no recentring).
  Position<Navigation> const origins[] = {
      Navigation::origin,
      Navigation::origin +
          Displacement<Navigation>({orbit_radius, 0 * Metre, 0 * Metre}),
      Navigation::origin -
          Displacement<Navigation>(
              {3.848e17 * Metre, 0 * Metre, 0 * Metre})};
  for (Position<Navigation> const& scaled_space_origin : origins) {
    std::vector<R3Element<double>> reference;
    auto const planetarium = plugin->NewPlanetarium(
        Planetarium::Parameters(/*sphere_radius_multiplier=*/1.0,
                                /*angular_resolution=*/0.001 * Radian,
                                /*field_of_view=*/1 * Radian),
        Perspective<Navigation, Camera>(
            RigidTransformation<Navigation, Camera>(
                Navigation::origin,
                Camera::origin,
                Signature<Navigation, Camera>::CentralInversion()
                    .Forget<OrthogonalMap>()).Forget<Similarity>(),
            /*focal=*/1 * Metre),
        [&reference, scaled_space_origin](
            Instant const&, Position<Navigation> const& plotted_point) {
          constexpr auto inverse_scale_factor = 1 / (6000 * Metre);
          auto const coordinates =
              ((plotted_point - scaled_space_origin) * inverse_scale_factor)
                  .coordinates();
          reference.push_back(coordinates);
          return coordinates;
        },
        [&reference](Displacement<Navigation> const& displacement) {
          constexpr auto inverse_scale_factor = 1 / (6000 * Metre);
          auto const coordinates =
              (displacement * inverse_scale_factor).coordinates();
          reference.push_back(coordinates);
          return coordinates;
        });
    std::vector<ScaledSpacePoint> vertices;
    R3Element<double> anchor;
    planetarium->PlotMethod4(
        trajectory,
        trajectory.begin(),
        trajectory.end(),
        InfiniteFuture,
        /*reverse=*/false,
        [&vertices](ScaledSpacePoint const& vertex) {
          vertices.push_back(vertex);
        },
        /*max_points=*/10'000,
        vessel.placement(),
        &anchor);
    ASSERT_GT(vertices.size(), 3);
    // The anchor is the camera position, exactly; its conversion is recorded
    // as the first `reference` entry, ahead of the camera-relative vertex
    // displacements recorded by the linear conversion.
    ASSERT_EQ(reference.size(), vertices.size() + 1);
    EXPECT_EQ(anchor.x, reference.front().x);
    EXPECT_EQ(anchor.y, reference.front().y);
    EXPECT_EQ(anchor.z, reference.front().z);
    reference.erase(reference.begin());

    // The guarantees, asserted as the adapter consumes them.
    // SHAPE: the float vertex buffer alone reproduces the geometry relative
    // to the anchor at the float ULP of each vertex's distance from the
    // camera, irrespective of where the scaled-space origin is; 6000 is the
    // scale factor.
    for (int i = 0; i < vertices.size(); ++i) {
      R3Element<double> const vertex(vertices[i].x,
                                     vertices[i].y,
                                     vertices[i].z);
      double const shape_error_in_metres =
          (vertex - reference[i]).Norm() * 6000;
      double const distance_from_camera_in_metres =
          reference[i].Norm() * 6000;
      EXPECT_LE(shape_error_in_metres,
                1e-3 + 1.5e-7 * distance_from_camera_in_metres)
          << "origin at " << scaled_space_origin << ", vertex " << i;
    }
    // PLACEMENT: the adapter translates the mesh by the float-cast anchor,
    // one rounding common to every vertex — it rigidly shifts the mesh by at
    // most the ULP of the anchor's own magnitude (small once the game
    // recentres the origin at the camera focus, which keeps the camera near
    // the origin) and cannot affect the shape asserted above.
    R3Element<double> const float_anchor(static_cast<float>(anchor.x),
                                         static_cast<float>(anchor.y),
                                         static_cast<float>(anchor.z));
    EXPECT_LE((float_anchor - anchor).Norm() * 6000,
              1e-3 + 1.5e-7 * anchor.Norm() * 6000)
        << "origin at " << scaled_space_origin;
  }
}

// With a single subsystem the anchor must be exactly zero and the vertices
// bit-identical to the absolute rendering: a stock installation renders
// unchanged.
TEST_F(PluginIntegrationTestWithoutPlugin, PlotMethod4StockVerticesUnchanged) {
  Index const star = 0;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  serialization::GravityModel::Body gravity_model;
  CHECK(google::protobuf::TextFormat::ParseFromString(
      R"(name                    : "star"
         gravitational_parameter : "1.19e19 m^3/s^2"
         reference_instant       : "JD2451545.0"
         mean_radius             : "8.4e7 m"
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
  plugin->InsertCelestialAbsoluteCartesian(
      star, std::nullopt, gravity_model, initial_state);
  plugin->EndInitialization();

  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                             /*loaded=*/false, inserted);
  Length const orbit_radius = 1.5829e9 * Metre;
  Speed const v_circular =
      Sqrt(1.19e19 * Pow<3>(Metre) / Pow<2>(Second) / orbit_radius);
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({orbit_radius, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>(
           {0 * Metre / Second, v_circular, 0 * Metre / Second})});
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  Time const δt = 1200 * Second;
  Instant t;
  for (int i = 0; i < 30; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star,
                               /*loaded=*/false, inserted);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }

  auto const& vessel = *plugin->GetVessel(vessel_guid);
  auto const& trajectory = vessel.trajectory();
  plugin->renderer().SetPlottingFrame(
      plugin->NewBodyCentredNonRotatingNavigationFrame(star));
  std::vector<R3Element<double>> reference;
  auto const planetarium = plugin->NewPlanetarium(
      Planetarium::Parameters(/*sphere_radius_multiplier=*/1.0,
                              /*angular_resolution=*/0.001 * Radian,
                              /*field_of_view=*/1 * Radian),
      Perspective<Navigation, Camera>(
          RigidTransformation<Navigation, Camera>(
              Navigation::origin,
              Camera::origin,
              Signature<Navigation, Camera>::CentralInversion()
                  .Forget<OrthogonalMap>()).Forget<Similarity>(),
          /*focal=*/1 * Metre),
      [&reference](Instant const&,
                   Position<Navigation> const& plotted_point) {
        constexpr auto inverse_scale_factor = 1 / (6000 * Metre);
        auto const coordinates =
            ((plotted_point - Navigation::origin) * inverse_scale_factor)
                .coordinates();
        reference.push_back(coordinates);
        return coordinates;
      },
      // A stock installation never plots anchored, so the linear conversion
      // is not needed: the legacy path must not touch it.
      /*plotting_to_scaled_space_displacement=*/nullptr);
  std::vector<ScaledSpacePoint> vertices;
  R3Element<double> anchor(1, 1, 1);
  planetarium->PlotMethod4(
      trajectory,
      trajectory.begin(),
      trajectory.end(),
      InfiniteFuture,
      /*reverse=*/false,
      [&vertices](ScaledSpacePoint const& vertex) {
        vertices.push_back(vertex);
      },
      /*max_points=*/10'000,
      vessel.placement(),
      &anchor);
  ASSERT_GT(vertices.size(), 3);
  ASSERT_EQ(reference.size(), vertices.size());
  EXPECT_EQ(anchor.x, 0);
  EXPECT_EQ(anchor.y, 0);
  EXPECT_EQ(anchor.z, 0);
  for (int i = 0; i < vertices.size(); ++i) {
    EXPECT_EQ(vertices[i].x, static_cast<float>(reference[i].x)) << i;
    EXPECT_EQ(vertices[i].y, static_cast<float>(reference[i].y)) << i;
    EXPECT_EQ(vertices[i].z, static_cast<float>(reference[i].z)) << i;
  }
}

// The golden mission: one vessel, one continuous timeline, traversing every
// shipped interstellar regime end-to-end.  From adoption on, a workstream is
// done only when its own gates are green AND this mission is green; new
// workstreams extend the mission rather than adding disjoint one-off e2es
// where possible.  The stages, and the regime each one certifies:
//   1. launch burn out of a planetary orbit, anticipated by the prediction
//      (on-rails burn);
//   2. cruise burn under warp to interstellar speed (on-rails burn);
//   3. coast into the void: the vessel anchors where the damped far field
//      vanishes and KEEPS its origin subsystem — dominance is deliberately
//      suspended while anchored — with per-step continuity throughout;
//   4. save/load mid-void while anchored: subsystem, anchor, flight plan and
//      the whole burn history round-trip through serialization;
//   5. a flight plan created on the anchored vessel coasts force-free, and —
//      once the void exit rebases the vessel — survives the rebase
//      (`FlightPlan::Rebase`) and renders consistently around either star;
//   6. void exit: the anchor is dropped and the vessel rebases into star B's
//      subsystem, exactly once, in the same step;
//   7. retro-burn (re-armed after the reload), then a rendezvous and docking
//      near star B: one pile-up in the destination subsystem.
// Deliberate descopes vs the adopted mission spec, certified elsewhere:
//   - WS1 absolute precision vs an independently integrated control is not
//     asserted here; the coast asserts per-step continuity to 1e-12 relative;
//   - the rendezvous is an insertion 10 m away, not a closing approach, and
//     mm-scale co-anchored coherence is structurally unreachable through
//     absolute-coordinate insertion (void ULP ~metres) — see
//     `CoAnchoredVoidVesselsKeepLocalPrecision` and
//     `ApproachingVoidVesselsConversionIsExact` for the mm claims;
//   - reanimation after an actual history drop is certified by
//     `AnchoredBurnSurvivesDropAndReanimation`; here the reload is asserted
//     drop-free and the burn history compared through it.
// A warp leg is deliberately absent: warp is adapter-side by design (the
// vessel is released during the cruise), so a headless mission cannot
// represent it; its acceptance lives in the WS4 handoff.
TEST_F(PluginIntegrationTestWithoutPlugin, GoldenMission) {
  Index const star_a = 0;
  Index const planet = 1;
  Index const star_b = 2;
  auto plugin =
      std::make_unique<Plugin>("JD2451545.0", "JD2451545.0", 1 * Radian);
  auto const insert_body = [&plugin](Index const index,
                                     std::optional<Index> const parent_index,
                                     std::string const& name,
                                     std::string const& gravitational_parameter,
                                     std::string const& mean_radius,
                                     std::string const& x,
                                     std::string const& vy) {
    serialization::GravityModel::Body gravity_model;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name                    : ")", name, R"("
                        gravitational_parameter : ")",
                     gravitational_parameter, R"("
                        reference_instant       : "JD2451545.0"
                        mean_radius             : ")", mean_radius, R"("
                        axis_right_ascension    : "0 deg"
                        axis_declination        : "90 deg"
                        reference_angle         : "0 rad"
                        angular_frequency       : "1 rad/s")"),
        &gravity_model));
    serialization::InitialState::Cartesian::Body initial_state;
    CHECK(google::protobuf::TextFormat::ParseFromString(
        absl::StrCat(R"(name : ")", name, R"("
                        x    : ")", x, R"("
                        y    : "0 m"
                        z    : "0 m"
                        vx   : "0 m/s"
                        vy   : ")", vy, R"("
                        vz   : "0 m/s")"),
        &initial_state));
    plugin->InsertCelestialAbsoluteCartesian(
        index, parent_index, gravity_model, initial_state);
  };
  insert_body(star_a, std::nullopt, "star A", "1.3e20 m^3/s^2", "1e6 m",
              "0 m", "0 m/s");
  // The planet's vy is its circular-orbit speed √(μ_star / 1e12 m).
  insert_body(planet, star_a, "planet", "4e14 m^3/s^2", "6.4e6 m",
              "1e12 m", "11401.754250991378 m/s");
  insert_body(star_b, star_a, "star B", "1.3e20 m^3/s^2", "1e6 m",
              "4e16 m", "0 m/s");
  plugin->EndInitialization();

  // The planet clusters with its star; the destination is its own subsystem.
  int const subsystem_a = plugin->GetCelestial(star_a).subsystem();
  int const subsystem_b = plugin->GetCelestial(star_b).subsystem();
  EXPECT_EQ(subsystem_a, plugin->GetCelestial(planet).subsystem());
  EXPECT_NE(subsystem_a, subsystem_b);

  // The AliceSun direction of star B, and the World direction that commands a
  // burn along it.  World and AliceSun differ by the XZY permutation — the
  // planetarium rotation cancels between the burn conversion and
  // `FromParent` (cf. `OnRailsBurn`) — so commanding an AliceSun direction
  // means swapping its y and z coordinates.  Note that AliceSun x̂ is NOT the
  // direction of star B: the rotation does not cancel against the celestial
  // insertion, so the star sits rotated in AliceSun coordinates.
  Vector<double, AliceSun> const to_star_b =
      Normalize(plugin->CelestialFromParent(star_b).displacement());
  Vector<double, World> const to_star_b_in_world({to_star_b.coordinates().x,
                                                  to_star_b.coordinates().z,
                                                  to_star_b.coordinates().y});

  // A low circular orbit around the planet; vy = √(μ_planet / 7e6 m).
  bool inserted;
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, planet,
                             /*loaded=*/false, inserted);
  plugin->InsertUnloadedPart(
      part_id, part_name, vessel_guid,
      {Displacement<AliceSun>({7e6 * Metre, 0 * Metre, 0 * Metre}),
       Velocity<AliceSun>({0 * Metre / Second,
                           7559.289460184544 * Metre / Second,
                           0 * Metre / Second})});
  plugin->GetVessel(vessel_guid)->part(part_id)->set_mass(3 * Kilogram);
  plugin->PrepareToReportCollisions();
  plugin->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  auto const& vessel = *plugin->GetVessel(vessel_guid);
  EXPECT_EQ(subsystem_a, vessel.placement().subsystem);
  EXPECT_FALSE(vessel.placement().anchor.has_value());

  // ——— Stage 1: launch burn ———
  SpecificImpulse const specific_impulse = 1e12 * Metre / Second;
  GravitationalParameter const μ_planet =
      4e14 * Pow<3>(Metre) / Pow<2>(Second);
  Mass m = 1000 * Kilogram;
  Instant t;

  Force const launch_thrust = 3e7 * Newton;
  Variation<Mass> const launch_mass_flow = launch_thrust / specific_impulse;
  Time const launch_δt = 100 * Second;
  Mass const m_before_launch = m;
  Velocity<AliceSun> const v_leo =
      plugin->VesselFromParent(planet, vessel_guid).velocity();
  t += launch_δt;
  plugin->AdvanceTime(t, 1 * Radian);
  plugin->InsertOrKeepVessel(vessel_guid, vessel_name, planet,
                             /*loaded=*/false, inserted);
  plugin->SetVesselOnRailsBurn(vessel_guid, launch_thrust, specific_impulse,
                               /*initial_mass=*/m,
                               to_star_b_in_world,
                               /*max_duration=*/1 * Hour);
  {
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
  }
  m -= launch_δt * launch_mass_flow;

  // Циолковский's Δv along the commanded direction — star B's; the tolerance
  // leaves room for the planet's gravity over the burn.
  Speed const launch_Δv = specific_impulse * std::log(m_before_launch / m);
  auto const post_launch = plugin->VesselFromParent(planet, vessel_guid);
  Velocity<AliceSun> const launch_Δv_vector = post_launch.velocity() - v_leo;
  EXPECT_THAT(launch_Δv_vector.Norm(), RelativeErrorFrom(launch_Δv, Lt(2e-3)));
  EXPECT_THAT(InnerProduct(launch_Δv_vector, to_star_b),
              RelativeErrorFrom(launch_Δv, Lt(2e-3)));
  // The orbit was raised to an escape: positive specific orbital energy with
  // respect to the planet.
  EXPECT_THAT(0.5 * post_launch.velocity().Norm²() -
                  μ_planet / post_launch.displacement().Norm(),
              Gt(SpecificEnergy{}));

  // The prediction coasts even with a burn armed: at F/m ≈ 3e4 m/s²
  // anticipating it would gain ~5e7 m/s over the next half hour, whereas what
  // gravity imparts out here is metres per second.
  {
    Vessel::MakeSynchronous();
    absl::Cleanup const restore_asynchronous = [] {
      Vessel::MakeAsynchronous();
    };
    plugin->UpdatePrediction({vessel_guid});
    auto const& from_state = vessel.psychohistory()->back();
    Instant const horizon = from_state.time + 1800 * Second;
    Speed const predicted_gain =
        (vessel.prediction()->EvaluateVelocity(horizon) -
         from_state.degrees_of_freedom.velocity()).Norm();
    EXPECT_THAT(predicted_gain, Lt(10 * Metre / Second));
  }

  // ——— Stage 2: cruise burn under warp to interstellar speed ———
  Force const cruise_thrust = 2e11 * Newton;
  Variation<Mass> const cruise_mass_flow = cruise_thrust / specific_impulse;
  Time const δt = 1200 * Second;
  Mass const m_before_cruise = m;
  Velocity<AliceSun> const v_before_cruise =
      plugin->VesselFromParent(star_a, vessel_guid).velocity();
  for (int i = 0; i < 3; ++i) {
    t += δt;
    plugin->AdvanceTime(t, 1 * Radian);
    plugin->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                               /*loaded=*/false, inserted);
    plugin->SetVesselOnRailsBurn(vessel_guid, cruise_thrust, specific_impulse,
                                 /*initial_mass=*/m,
                                 to_star_b_in_world,
                                 /*max_duration=*/1 * Hour);
    VesselSet collided_vessels;
    plugin->CatchUpLaggingVessels(collided_vessels);
    m -= δt * cruise_mass_flow;
  }
  Speed const cruise_Δv = specific_impulse * std::log(m_before_cruise / m);
  EXPECT_THAT((plugin->VesselFromParent(star_a, vessel_guid).velocity() -
               v_before_cruise).Norm(),
              RelativeErrorFrom(cruise_Δv, Lt(1e-3)));
  plugin->ClearVesselOnRailsBurn(vessel_guid);

  auto const coast_frame = [&](Plugin& p) {
    t += δt;
    p.AdvanceTime(t, 1 * Radian);
    p.InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                         /*loaded=*/false, inserted);
    p.PrepareToReportCollisions();
    p.FreeVesselsAndPartsAndCollectPileUps(δt);
    VesselSet collided_vessels;
    p.CatchUpLaggingVessels(collided_vessels);
  };

  // Named bounds shared below.  A star's far-field outer threshold is
  // √(μ/floor) = √(1.3e20 / 1e-12) ≈ 1.14e16 m (`far_field_damping_floor`,
  // plugin.cpp); beyond it from EVERY body the field is exactly zero and an
  // unloaded vessel anchors.  1.4e16 m adds margin for the discrete steps.
  Length const void_threshold = 1.4e16 * Metre;
  // Absolute void coordinates quantize at the plugin boundary at the ULP of
  // their ~1e16–4e16 m magnitude — metres; 100 m bounds every assertion that
  // crosses that boundary.
  Length const void_ulp_tolerance = 100 * Metre;

  // Per-step checks shared by the two coast loops below: continuity of the
  // motion seen through `VesselFromParent`, and the count of subsystem
  // changes.  The continuity budget: reads quantize at the void ULP (~4 m)
  // and each re-anchor folds sub-mm residuals, against a step of
  // v·δt ≈ 1.5e15 m — 1e-12 relative leaves ~10³ m of headroom while still
  // catching any metre-scale-per-kilometre representation glitch.
  int rebases = 0;
  int previous_subsystem = vessel.placement().subsystem;
  std::optional<Displacement<AliceSun>> previous_displacement;
  std::optional<Speed> cruise_speed;
  auto const step_checks = [&](Plugin const& p, Vessel const& v) {
    auto const from_parent = p.VesselFromParent(star_a, vessel_guid);
    if (!cruise_speed.has_value()) {
      cruise_speed = from_parent.velocity().Norm();
    }
    if (previous_displacement.has_value()) {
      // Continuous at every step: through anchor adoption, re-anchoring, the
      // reload, and the void-exit drop + rebase alike.
      EXPECT_THAT((from_parent.displacement() - *previous_displacement).Norm(),
                  RelativeErrorFrom(*cruise_speed * δt, Lt(1e-12)));
    }
    previous_displacement = from_parent.displacement();
    if (v.placement().subsystem != previous_subsystem) {
      previous_subsystem = v.placement().subsystem;
      ++rebases;
    }
  };

  // ——— Stage 3: coast into the void; the anchor is adopted and the origin
  // subsystem is KEPT (dominance is suspended while anchored) ———
  Length const mid_void = 1.8e16 * Metre;
  for (int step = 0;
       step < 20 &&
       !(vessel.placement().anchor.has_value() &&
         plugin->VesselFromParent(star_a, vessel_guid).displacement().Norm() >
             mid_void);
       ++step) {
    coast_frame(*plugin);
    step_checks(*plugin, vessel);
    if (previous_displacement->Norm() > void_threshold &&
        plugin->VesselFromParent(star_b, vessel_guid).displacement().Norm() >
            void_threshold) {
      // Beyond the far-field threshold of both stars the coast is force-free:
      // the vessel must be anchored.
      EXPECT_TRUE(vessel.placement().anchor.has_value());
    }
  }
  ASSERT_TRUE(vessel.placement().anchor.has_value());
  EXPECT_EQ(subsystem_a, vessel.placement().subsystem);
  EXPECT_EQ(0, rebases);

  // ——— Stage 5 (creation): a flight plan on the anchored void vessel.  Its
  // 3-hour coast crosses the far-field shell toward star B, and the vessel's
  // later rebase carries it through `FlightPlan::Rebase`. ———
  Instant const flight_plan_epoch = vessel.psychohistory()->back().time;
  auto const state_at_plan = plugin->VesselFromParent(star_a, vessel_guid);
  auto const state_at_plan_from_b =
      plugin->VesselFromParent(star_b, vessel_guid);
  plugin->GetVessel(vessel_guid)->CreateFlightPlan(
      flight_plan_epoch + 3 * Hour,
      m,
      Ephemeris<Barycentric>::AdaptiveStepParameters(
          EmbeddedExplicitRungeKuttaNyströmIntegrator<
              DormandالمكاوىPrince1986RKN434FM,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          /*max_steps=*/1000,
          /*length_integration_tolerance=*/1 * Milli(Metre),
          /*speed_integration_tolerance=*/1 * Milli(Metre) / Second),
      Ephemeris<Barycentric>::GeneralizedAdaptiveStepParameters(
          EmbeddedExplicitGeneralizedRungeKuttaNyströmIntegrator<
              Fine1987RKNG34,
              Ephemeris<Barycentric>::GeneralizedNewtonianMotionEquation>(),
          /*max_steps=*/1000,
          /*length_integration_tolerance=*/1 * Milli(Metre),
          /*speed_integration_tolerance=*/1 * Milli(Metre) / Second));
  ASSERT_TRUE(vessel.has_flight_plan());
  {
    auto const& flight_plan = plugin->GetVessel(vessel_guid)->flight_plan();
    ASSERT_EQ(1, flight_plan.number_of_segments());
    auto const coast = flight_plan.GetSegment(0);
    // Force-free void: the plan coasts; the damped shell of star B near its
    // end contributes ≲ 1e-9 m/s.
    EXPECT_THAT((coast->back().degrees_of_freedom.velocity() -
                 coast->front().degrees_of_freedom.velocity()).Norm(),
                Lt(1 * Milli(Metre) / Second));
  }

  // ——— Stage 4: save/load mid-void, anchored ———
  // Sample state inside the launch burn, for the history round-trip check.
  Instant const t_sample = Instant() + 50 * Second;
  DegreesOfFreedom<Barycentric> const actual_sample =
      vessel.trajectory().EvaluateDegreesOfFreedom(t_sample);

  auto const anchor_before_save = *vessel.placement().anchor;
  auto const state_before_save = plugin->VesselFromParent(star_a, vessel_guid);

  serialization::Plugin message;
  plugin->WriteToMessage(&message);
  plugin = nullptr;
  // Everything from here on operates on `plugin2`; a stage spliced in below
  // must keep using it, and keep the mass bookkeeping `m` in sync through any
  // burn it adds.
  auto const plugin2 = Plugin::ReadFromMessage(message);
  auto const& vessel2 = *plugin2->GetVessel(vessel_guid);
  EXPECT_EQ(subsystem_a, vessel2.placement().subsystem);
  ASSERT_TRUE(vessel2.placement().anchor.has_value());
  EXPECT_EQ(anchor_before_save, *vessel2.placement().anchor);
  EXPECT_TRUE(vessel2.has_flight_plan());
  EXPECT_THAT(
      (plugin2->VesselFromParent(star_a, vessel_guid).displacement() -
       state_before_save.displacement()).Norm(),
      Lt(void_ulp_tolerance));

  // The burn history round-trips through serialization.  The reload is
  // asserted drop-free — the drop-and-reanimate path is certified by
  // `AnchoredBurnSurvivesDropAndReanimation` — and the tolerance covers the
  // lossy point compression (~the 10 m downsampling tolerance) plus the ULP
  // of the anchored representation, whose pre-void points sit ~2e16 m from
  // the local origin (ULP ≈ 4 m).
  ASSERT_LE(vessel2.trajectory().t_min(), t_sample)
      << "the reload dropped history; this mission asserts the drop-free "
         "round-trip";
  plugin2->GetVessel(vessel_guid)->AwaitReanimation(Instant(), /*quiet=*/true);
  DegreesOfFreedom<Barycentric> const reloaded_sample =
      vessel2.trajectory().EvaluateDegreesOfFreedom(t_sample);
  EXPECT_THAT((reloaded_sample.position() - actual_sample.position()).Norm(),
              Lt(20 * Metre));
  EXPECT_THAT((reloaded_sample.velocity() - actual_sample.velocity()).Norm(),
              Lt(0.1 * Metre / Second));

  // ——— Stage 6: void exit — the vessel rebases into star B's subsystem,
  // exactly once; the anchor survives the retag (the conversion is folded
  // into it), keeping the representation uniform across the boundary ———
  for (int step = 0;
       step < 15 && vessel2.placement().subsystem != subsystem_b; ++step) {
    coast_frame(*plugin2);
    step_checks(*plugin2, vessel2);
  }
  ASSERT_EQ(subsystem_b, vessel2.placement().subsystem);
  EXPECT_EQ(1, rebases);
  // Uniform representation: entering star B's field KEEPS the anchor — an
  // unanchored representation at these subsystem distances would quantize
  // the parts at the domain ULP (the owner-visible part gaps at TRAPPIST-1).
  EXPECT_TRUE(vessel2.placement().anchor.has_value());

  // ——— Stage 5 (checks): the flight plan survived the reload AND the rebase,
  // still coasts, and renders consistently ———
  ASSERT_TRUE(vessel2.has_flight_plan());
  // A reloaded flight plan is held lazily in serialized form, so the rebase
  // above skipped it; deserializing it now exercises the lazy-rebase path
  // that re-expresses it in the vessel's current subsystem.
  plugin2->GetVessel(vessel_guid)->ReadFlightPlanFromMessage();
  auto const& flight_plan = plugin2->GetVessel(vessel_guid)->flight_plan();
  EXPECT_EQ(subsystem_b, flight_plan.placement().subsystem);
  ASSERT_EQ(1, flight_plan.number_of_segments());
  auto const coast = flight_plan.GetSegment(0);
  EXPECT_THAT((coast->back().degrees_of_freedom.velocity() -
               coast->front().degrees_of_freedom.velocity()).Norm(),
              Lt(1 * Milli(Metre) / Second));

  // The plan's endpoint is where a 3-hour cruise from its creation state
  // puts it.  Rendered in `World` — whose origin pins the sun, star A — the
  // endpoint's distance is invariant under the choice of plotting-frame
  // centre, so the two-frame agreement below is a sanity bound on the
  // cross-subsystem placement conversion, NOT a map-view check; the genuine
  // star-B-frame check is the plotting-frame render that follows.
  Length const expected_distance_from_a =
      (state_at_plan.displacement() +
       state_at_plan.velocity() * (3 * Hour)).Norm();
  Length const expected_distance_from_b =
      (state_at_plan_from_b.displacement() +
       state_at_plan_from_b.velocity() * (3 * Hour)).Norm();
  auto const& all_segments = flight_plan.GetAllSegments();
  auto const render_distance = [&](Index const centre) {
    plugin2->renderer().SetPlottingFrame(
        plugin2->NewBodyCentredNonRotatingNavigationFrame(centre));
    auto const rendered =
        plugin2->renderer().RenderBarycentricTrajectoryInWorld(
            plugin2->CurrentTime(),
            all_segments.begin(),
            all_segments.end(),
            World::origin,
            plugin2->PlanetariumRotation(),
            flight_plan.placement());
    return (rendered.back().degrees_of_freedom.position() - World::origin)
        .Norm();
  };
  Length const rendered_around_a = render_distance(star_a);
  Length const rendered_around_b = render_distance(star_b);
  EXPECT_THAT(rendered_around_a,
              AbsoluteErrorFrom(expected_distance_from_a, Lt(1e6 * Metre)));
  EXPECT_THAT(rendered_around_b,
              AbsoluteErrorFrom(rendered_around_a, Lt(void_ulp_tolerance)));
  {
    plugin2->renderer().SetPlottingFrame(
        plugin2->NewBodyCentredNonRotatingNavigationFrame(star_b));
    auto const rendered =
        plugin2->renderer().RenderBarycentricTrajectoryInPlotting(
            all_segments.begin(),
            all_segments.end(),
            flight_plan.placement());
    EXPECT_THAT((rendered.back().degrees_of_freedom.position() -
                 Navigation::origin).Norm(),
                AbsoluteErrorFrom(expected_distance_from_b, Lt(1e6 * Metre)));
  }

  // The map-view vertex path over the same anchored void flight plan: the
  // plotted float vertices are anchored at the camera, so they carry the ULP
  // of their distance from it — angularly sub-pixel — rather than that of
  // their distance to the scaled-space origin (the lattice-anchored-rendering
  // render check of this mission).
  {
    std::vector<R3Element<double>> reference;
    auto const planetarium = plugin2->NewPlanetarium(
        Planetarium::Parameters(/*sphere_radius_multiplier=*/1.0,
                                /*angular_resolution=*/0.001 * Radian,
                                /*field_of_view=*/1 * Radian),
        Perspective<Navigation, Camera>(
            RigidTransformation<Navigation, Camera>(
                Navigation::origin,
                Camera::origin,
                Signature<Navigation, Camera>::CentralInversion()
                    .Forget<OrthogonalMap>()).Forget<Similarity>(),
            /*focal=*/1 * Metre),
        [&reference](Instant const&,
                     Position<Navigation> const& plotted_point) {
          constexpr auto inverse_scale_factor = 1 / (6000 * Metre);
          auto const coordinates =
              ((plotted_point - Navigation::origin) * inverse_scale_factor)
                  .coordinates();
          reference.push_back(coordinates);
          return coordinates;
        },
        [&reference](Displacement<Navigation> const& displacement) {
          constexpr auto inverse_scale_factor = 1 / (6000 * Metre);
          auto const coordinates =
              (displacement * inverse_scale_factor).coordinates();
          reference.push_back(coordinates);
          return coordinates;
        });
    std::vector<ScaledSpacePoint> vertices;
    R3Element<double> anchor;
    planetarium->PlotMethod4(
        all_segments,
        all_segments.begin(),
        all_segments.end(),
        InfiniteFuture,
        /*reverse=*/false,
        [&vertices](ScaledSpacePoint const& vertex) {
          vertices.push_back(vertex);
        },
        /*max_points=*/10'000,
        flight_plan.placement(),
        &anchor);
    // Seen from a star-B-centred camera the void-distant plan subtends a tiny
    // angle, so the adaptive sampling emits only the two endpoints — which
    // suffice here: the far endpoint carries the whole void span.
    ASSERT_GE(vertices.size(), 2);
    // The anchor is the displacement from the plan's own starting point — the
    // vessel's position at the present, which is where the scene draws its
    // icon — to the camera, and it is recorded as the first `reference`
    // entry, ahead of the camera-relative vertex displacements.  Adding it
    // back to the first vertex therefore lands on the starting point.
    ASSERT_EQ(reference.size(), vertices.size() + 1);
    EXPECT_EQ(anchor.x, reference.front().x);
    EXPECT_EQ(anchor.y, reference.front().y);
    EXPECT_EQ(anchor.z, reference.front().z);
    EXPECT_LE((R3Element<double>(vertices.front().x,
                                 vertices.front().y,
                                 vertices.front().z) +
               anchor).Norm() * 6000,
              1e-3 + 1.5e-7 * anchor.Norm() * 6000);
    reference.erase(reference.begin());
    // SHAPE: the float vertex buffer reproduces the plan's geometry relative
    // to the anchor at the float ULP of each vertex's distance from the
    // camera; the adapter's float-cast of the anchor is a single common-mode
    // translation which cannot affect it.
    for (int i = 0; i < vertices.size(); ++i) {
      R3Element<double> const vertex(vertices[i].x,
                                     vertices[i].y,
                                     vertices[i].z);
      double const shape_error_in_metres =
          (vertex - reference[i]).Norm() * 6000;
      double const distance_from_camera_in_metres =
          reference[i].Norm() * 6000;
      EXPECT_LE(shape_error_in_metres,
                1e-3 + 1.5e-7 * distance_from_camera_in_metres)
          << "vertex " << i;
    }
  }

  // ——— Stage 7: retro-burn, re-armed after the reload ———
  Force const retro_thrust = 2e12 * Newton;
  Variation<Mass> const retro_mass_flow = retro_thrust / specific_impulse;
  Velocity<AliceSun> const v_before_retro =
      plugin2->VesselFromParent(star_a, vessel_guid).velocity();
  Speed const cruise_velocity_along = InnerProduct(v_before_retro, to_star_b);
  Mass const m_before_retro = m;
  // The burn duration that exactly exhausts the cruise velocity:
  // Δv = Isp ln(m₀/m₁), inverted for the propellant time at this mass flow.
  Time const retro_duration =
      (m / retro_mass_flow) *
      (1 - std::exp(-cruise_velocity_along / specific_impulse));
  t += δt;
  plugin2->AdvanceTime(t, 1 * Radian);
  plugin2->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                              /*loaded=*/false, inserted);
  plugin2->SetVesselOnRailsBurn(vessel_guid, retro_thrust, specific_impulse,
                                /*initial_mass=*/m,
                                -to_star_b_in_world,
                                /*max_duration=*/retro_duration);
  {
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
  }
  m -= retro_duration * retro_mass_flow;
  Speed const retro_Δv = specific_impulse * std::log(m_before_retro / m);
  auto const post_retro = plugin2->VesselFromParent(star_a, vessel_guid);
  EXPECT_THAT((post_retro.velocity() - v_before_retro).Norm(),
              RelativeErrorFrom(retro_Δv, Lt(1e-3)));
  // The interstellar cruise is essentially cancelled: what remains is the
  // burn's 1e-3 modelling tolerance plus ~2e4 m/s of orbital residue.
  EXPECT_THAT(post_retro.velocity().Norm(), Lt(0.01 * cruise_velocity_along));
  plugin2->ClearVesselOnRailsBurn(vessel_guid);
  coast_frame(*plugin2);

  // ——— Stage 7 (continued): rendezvous and docking near star B ———
  GUID const station_guid = "456-789";
  PartId const station_part_id = 790;
  auto const traveller_from_star_b =
      plugin2->VesselFromParent(star_b, vessel_guid);
  plugin2->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                              /*loaded=*/false, inserted);
  plugin2->InsertOrKeepVessel(station_guid, "station", star_b,
                              /*loaded=*/false, inserted);
  plugin2->InsertUnloadedPart(
      station_part_id, "station part", station_guid,
      {traveller_from_star_b.displacement() +
           Displacement<AliceSun>({10 * Metre, 0 * Metre, 0 * Metre}),
       traveller_from_star_b.velocity()});
  plugin2->GetVessel(station_guid)->part(station_part_id)->set_mass(
      1 * Kilogram);
  plugin2->PrepareToReportCollisions();
  plugin2->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
  }
  not_null<Vessel*> const station = plugin2->GetVessel(station_guid);
  EXPECT_EQ(subsystem_b, station->placement().subsystem);
  // Uniform representation: the station's coordinates in star B's subsystem
  // exceed the re-anchor bound, so it anchors inside the star's field too.
  EXPECT_TRUE(station->placement().anchor.has_value());
  Displacement<AliceSun> const separation_before_docking =
      plugin2->VesselFromParent(star_b, station_guid).displacement() -
      plugin2->VesselFromParent(star_b, vessel_guid).displacement();
  EXPECT_THAT(separation_before_docking.Norm(), Lt(void_ulp_tolerance));

  // Dock.
  t += δt;
  plugin2->AdvanceTime(t, 1 * Radian);
  plugin2->InsertOrKeepVessel(vessel_guid, vessel_name, star_a,
                              /*loaded=*/false, inserted);
  plugin2->InsertOrKeepVessel(station_guid, "station", star_b,
                              /*loaded=*/false, inserted);
  plugin2->GetVessel(vessel_guid)->KeepPart(part_id);
  plugin2->GetVessel(station_guid)->KeepPart(station_part_id);
  plugin2->PrepareToReportCollisions();
  plugin2->ReportPartCollision(part_id, station_part_id);
  plugin2->FreeVesselsAndPartsAndCollectPileUps(20 * Milli(Second));
  {
    VesselSet collided_vessels;
    plugin2->CatchUpLaggingVessels(collided_vessels);
  }

  // One pile-up in the destination subsystem, the local geometry preserved
  // through the reconciliation: mission complete.
  auto* const pile_up_traveller =
      plugin2->GetVessel(vessel_guid)->part(part_id)->containing_pile_up();
  auto* const pile_up_station =
      plugin2->GetVessel(station_guid)->part(station_part_id)
          ->containing_pile_up();
  ASSERT_NE(nullptr, pile_up_traveller);
  EXPECT_EQ(pile_up_traveller, pile_up_station);
  EXPECT_EQ(subsystem_b, pile_up_traveller->placement().subsystem);
  EXPECT_EQ(subsystem_b, vessel2.placement().subsystem);
  EXPECT_EQ(subsystem_b, station->placement().subsystem);
  EXPECT_EQ(1, rebases);
  Displacement<AliceSun> const separation_after_docking =
      plugin2->VesselFromParent(star_b, station_guid).displacement() -
      plugin2->VesselFromParent(star_b, vessel_guid).displacement();
  EXPECT_THAT((separation_after_docking - separation_before_docking).Norm(),
              Lt(void_ulp_tolerance));
}

}  // namespace ksp_plugin
}  // namespace principia
