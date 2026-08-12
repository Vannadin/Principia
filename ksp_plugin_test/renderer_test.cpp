#include "ksp_plugin/renderer.hpp"

#include <memory>

#include "base/not_null.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/rotation.hpp"
#include "geometry/space.hpp"
#include "geometry/space_transformations.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin_test/mock_celestial.hpp"
#include "ksp_plugin_test/mock_vessel.hpp"
#include "physics/apsides.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "physics/mock_continuous_trajectory.hpp"
#include "physics/mock_ephemeris.hpp"
#include "physics/mock_rigid_reference_frame.hpp"
#include "physics/rigid_motion.hpp"
#include "physics/sector.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/almost_equals.hpp"
#include "testing_utilities/componentwise.hpp"
#include "testing_utilities/discrete_trajectory_factories.hpp"

namespace principia {
namespace ksp_plugin {

using ::testing::Gt;
using ::testing::Lt;
using ::testing::Ref;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::_;
using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_space;
using namespace principia::geometry::_space_transformations;
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_renderer;
using namespace principia::ksp_plugin_test::_mock_celestial;
using namespace principia::ksp_plugin_test::_mock_vessel;
using namespace principia::physics::_apsides;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_mock_continuous_trajectory;
using namespace principia::physics::_mock_ephemeris;
using namespace principia::physics::_mock_rigid_reference_frame;
using namespace principia::physics::_rigid_motion;
using namespace principia::physics::_sector;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
using namespace principia::testing_utilities::_componentwise;
using namespace principia::testing_utilities::_discrete_trajectory_factories;

class RendererTest : public ::testing::Test {
 protected:
  RendererTest()
      : renderer_(&celestial_,
                  std::make_unique<
                      MockRigidReferenceFrame<Barycentric, Navigation>>()),
        reference_frame_(renderer_.GetPlottingFrame()) {}

  Instant const t0_;
  MockCelestial const celestial_;
  Renderer renderer_;
  not_null<MockRigidReferenceFrame<Barycentric, Navigation> const*> const
      reference_frame_;
};

TEST_F(RendererTest, TargetVessel) {
  MockEphemeris<Barycentric> ephemeris;
  MockContinuousTrajectory<Barycentric> celestial_trajectory;
  EXPECT_CALL(ephemeris, trajectory(_))
      .WillRepeatedly(Return(&celestial_trajectory));

  MockVessel vessel;
  DiscreteTrajectory<Barycentric> vessel_trajectory;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(/*v=*/Barycentric::unmoving,
                                  /*Δt=*/1 * Second,
                                  /*t1=*/t0_,
                                  /*t2=*/t0_ + 1 * Second),
      /*to=*/vessel_trajectory);

  EXPECT_CALL(vessel, prediction())
      .WillRepeatedly(Return(vessel_trajectory.segments().begin()));

  renderer_.SetTargetVessel(&vessel, &celestial_, &ephemeris);
  EXPECT_TRUE(renderer_.HasTargetVessel());
  EXPECT_THAT(renderer_.GetTargetVessel(), Ref(vessel));
  MockVessel other_vessel;
  renderer_.ClearTargetVesselIf(&other_vessel);
  EXPECT_THAT(renderer_.GetTargetVessel(), Ref(vessel));
  renderer_.ClearTargetVesselIf(&vessel);
  EXPECT_FALSE(renderer_.HasTargetVessel());
  renderer_.SetTargetVessel(&vessel, &celestial_, &ephemeris);
  renderer_.ClearTargetVessel();
  EXPECT_FALSE(renderer_.HasTargetVessel());
}

TEST_F(RendererTest, RenderBarycentricTrajectoryInPlottingWithoutTargetVessel) {
  auto const vx = 6 * Metre / Second;
  auto const vy = 5 * Metre / Second;
  auto const vz = 4 * Metre / Second;
  Velocity<Barycentric> const v({vx, vy, vz});
  DiscreteTrajectory<Barycentric> trajectory_to_render;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(v,
                                  /*Δt=*/1 * Second,
                                  /*t1=*/t0_,
                                  /*t2=*/t0_ + 10 * Second),
      /*to=*/trajectory_to_render);

  RigidMotion<Barycentric, Navigation> const rigid_motion(
      RigidTransformation<Barycentric, Navigation>::Identity(),
      Barycentric::nonrotating,
      Barycentric::unmoving);
  for (Instant t = t0_; t < t0_ + 10 * Second; t += 1 * Second) {
    EXPECT_CALL(*reference_frame_, ToThisFrameAtTime(t))
        .WillOnce(Return(rigid_motion));
  }

  auto const rendered_trajectory =
      renderer_.RenderBarycentricTrajectoryInPlotting(
          trajectory_to_render.begin(),
          trajectory_to_render.end());

  EXPECT_EQ(10, rendered_trajectory.size());
  int index = 0;
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    EXPECT_EQ(t0_ + index * Second, time);
    EXPECT_THAT(degrees_of_freedom,
                Componentwise(
                    AlmostEquals(Navigation::origin + Displacement<Navigation>(
                                                          {vx * (time - t0_),
                                                           vy * (time - t0_),
                                                           vz * (time - t0_)}),
                                 0),
                    AlmostEquals(Velocity<Navigation>({vx, vy, vz}), 0)));
    ++index;
  }
}

// WS6-5: a vessel coasting in the void is anchored — its trajectory is near the
// origin and the real offset lives on the anchor.  Rendering must add the
// anchor, evaluated at each point's own time, or the vessel plots on top of its
// home star instead of out in the void.
TEST_F(RendererTest, RenderBarycentricTrajectoryInPlottingWithAnchor) {
  auto const vx = 6 * Metre / Second;
  auto const vy = 5 * Metre / Second;
  auto const vz = 4 * Metre / Second;
  Velocity<Barycentric> const v({vx, vy, vz});
  DiscreteTrajectory<Barycentric> trajectory_to_render;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(v,
                                  /*Δt=*/1 * Second,
                                  /*t1=*/t0_,
                                  /*t2=*/t0_ + 10 * Second),
      /*to=*/trajectory_to_render);

  RigidMotion<Barycentric, Navigation> const rigid_motion(
      RigidTransformation<Barycentric, Navigation>::Identity(),
      Barycentric::nonrotating,
      Barycentric::unmoving);
  for (Instant t = t0_; t < t0_ + 10 * Second; t += 1 * Second) {
    EXPECT_CALL(*reference_frame_, ToThisFrameAtTime(t))
        .WillOnce(Return(rigid_motion));
  }

  auto const ax = 1 * Metre / Second;
  auto const ay = 2 * Metre / Second;
  auto const az = 3 * Metre / Second;
  Ephemeris<Barycentric>::Anchor const anchor{
      .offset = SectorDisplacement<Barycentric>::Split(
          Displacement<Barycentric>({100 * Metre, 200 * Metre,
                                     300 * Metre})),
      .velocity = Velocity<Barycentric>({ax, ay, az}),
      .epoch = t0_};

  auto const rendered_trajectory =
      renderer_.RenderBarycentricTrajectoryInPlotting(
          trajectory_to_render.begin(),
          trajectory_to_render.end(),
          {/*subsystem=*/0, anchor});

  EXPECT_EQ(10, rendered_trajectory.size());
  int index = 0;
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    EXPECT_EQ(t0_ + index * Second, time);
    EXPECT_THAT(
        degrees_of_freedom,
        Componentwise(
            AlmostEquals(
                Navigation::origin +
                    Displacement<Navigation>(
                        {(vx + ax) * (time - t0_) + 100 * Metre,
                         (vy + ay) * (time - t0_) + 200 * Metre,
                         (vz + az) * (time - t0_) + 300 * Metre}),
                0),
            AlmostEquals(Velocity<Navigation>({vx + ax, vy + ay, vz + az}), 0)));
    ++index;
  }
}

TEST_F(RendererTest, RenderBarycentricTrajectoryInPlottingWithTargetVessel) {
  MockEphemeris<Barycentric> ephemeris;
  MockContinuousTrajectory<Barycentric> celestial_trajectory;
  EXPECT_CALL(ephemeris, trajectory(_))
      .WillRepeatedly(Return(&celestial_trajectory));

  DiscreteTrajectory<Barycentric> trajectory_to_render;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(
          /*v=*/Velocity<Barycentric>(
              {6 * Metre / Second, 5 * Metre / Second, 4 * Metre / Second}),
          /*Δt=*/1 * Second,
          /*t1=*/t0_,
          /*t2=*/t0_ + 10 * Second),
      /*to=*/trajectory_to_render);

  // The prediction is shorter than the `trajectory_to_render`.
  MockVessel vessel;
  DiscreteTrajectory<Barycentric> vessel_trajectory;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(
          DegreesOfFreedom<Barycentric>(
              Barycentric::origin,
              Velocity<Barycentric>({1 * Metre / Second,
                                     2 * Metre / Second,
                                     3 * Metre / Second})),
          /*Δt=*/1 * Second,
          /*t0=*/t0_,
          /*t1=*/t0_ + 3 * Second,
          /*t2=*/t0_ + 8 * Second),
      /*to=*/vessel_trajectory);
  EXPECT_CALL(vessel, prediction())
      .WillRepeatedly(Return(vessel_trajectory.segments().begin()));
  Ephemeris<Barycentric>::SubsystemPlacement const stock_placement =
      Ephemeris<Barycentric>::SubsystemPlacement::Stock();
  EXPECT_CALL(vessel, placement()).WillRepeatedly(ReturnRef(stock_placement));

  for (Instant t = t0_ + 3 * Second; t < t0_ + 8 * Second; t += 1 * Second) {
    EXPECT_CALL(celestial_trajectory, EvaluateDegreesOfFreedom(t))
        .WillOnce(Return(DegreesOfFreedom<Barycentric>(
            Barycentric::origin + Displacement<Barycentric>(
                                      {300 * Metre, 200 * Metre, 100 * Metre}),
            Barycentric::unmoving)));
  }

  renderer_.SetTargetVessel(&vessel, &celestial_, &ephemeris);
  auto const rendered_trajectory =
      renderer_.RenderBarycentricTrajectoryInPlotting(
          trajectory_to_render.begin(),
          trajectory_to_render.end());

  EXPECT_EQ(5, rendered_trajectory.size());
  int index = 3;
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    EXPECT_EQ(t0_ + index * Second, time);
    // The degrees of freedom are computed using a real dynamic frame, not a
    // mock.  No point in re-doing the computation here, we just check that the
    // numbers are reasonable.
    EXPECT_LT((degrees_of_freedom.position() - Navigation::origin).Norm(),
              42 * Metre);
    EXPECT_LT(degrees_of_freedom.velocity().Norm(), 6 * Metre / Second);
    ++index;
  }
}

// A void rendezvous rendered in a target-vessel frame: the target frame's
// representation carries the target's own anchor, and the active vessel is
// converted into that placement — the anchors difference on the sector
// lattice, so the plot shows the true local geometry.  Pre-fix (the failure
// this test is first against) the active anchor was added absolutely while
// the target frame stayed anchored-local, so the rendered positions were
// void-scale (~2e16 m) artifacts.
TEST_F(RendererTest, RenderBarycentricTrajectoryInPlottingWithAnchoredTarget) {
  MockEphemeris<Barycentric> ephemeris;
  MockContinuousTrajectory<Barycentric> celestial_trajectory;
  EXPECT_CALL(ephemeris, trajectory(_))
      .WillRepeatedly(Return(&celestial_trajectory));

  // The active vessel's trajectory, in its own anchored coordinates.
  DiscreteTrajectory<Barycentric> trajectory_to_render;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(
          /*v=*/Velocity<Barycentric>(
              {6 * Metre / Second, 5 * Metre / Second, 4 * Metre / Second}),
          /*Δt=*/1 * Second,
          /*t1=*/t0_,
          /*t2=*/t0_ + 10 * Second),
      /*to=*/trajectory_to_render);

  // The target vessel's prediction, in its own anchored coordinates.
  MockVessel vessel;
  DiscreteTrajectory<Barycentric> vessel_trajectory;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(
          DegreesOfFreedom<Barycentric>(
              Barycentric::origin,
              Velocity<Barycentric>({1 * Metre / Second,
                                     2 * Metre / Second,
                                     3 * Metre / Second})),
          /*Δt=*/1 * Second,
          /*t0=*/t0_,
          /*t1=*/t0_ + 3 * Second,
          /*t2=*/t0_ + 8 * Second),
      /*to=*/vessel_trajectory);
  EXPECT_CALL(vessel, prediction())
      .WillRepeatedly(Return(vessel_trajectory.segments().begin()));

  // Both vessels are ~2e16 m into the void, eight metres apart.
  std::optional<Ephemeris<Barycentric>::Anchor> const target_anchor(
      Ephemeris<Barycentric>::Anchor{
          .offset = SectorDisplacement<Barycentric>::Split(
              Displacement<Barycentric>(
                  {2e16 * Metre, 0 * Metre, 0 * Metre})),
          .velocity = Velocity<Barycentric>(),
          .epoch = t0_});
  Ephemeris<Barycentric>::Anchor const active_anchor{
      .offset = SectorDisplacement<Barycentric>::Split(
          Displacement<Barycentric>(
              {2e16 * Metre, 8 * Metre, 0 * Metre})),
      .velocity = Velocity<Barycentric>(),
      .epoch = t0_};
  Ephemeris<Barycentric>::SubsystemPlacement const target_placement{
      0, target_anchor};
  EXPECT_CALL(vessel, placement()).WillRepeatedly(ReturnRef(target_placement));

  Position<Barycentric> const celestial_position =
      Barycentric::origin +
      Displacement<Barycentric>({300 * Metre, 200 * Metre, 100 * Metre});
  for (Instant t = t0_ + 3 * Second; t < t0_ + 8 * Second; t += 1 * Second) {
    EXPECT_CALL(celestial_trajectory, EvaluateDegreesOfFreedom(t))
        .WillRepeatedly(Return(DegreesOfFreedom<Barycentric>(
            celestial_position, Barycentric::unmoving)));
  }

  renderer_.SetTargetVessel(&vessel, &celestial_, &ephemeris);
  auto const rendered_trajectory =
      renderer_.RenderBarycentricTrajectoryInPlotting(
          trajectory_to_render.begin(),
          trajectory_to_render.end(),
          {/*subsystem=*/0, active_anchor});

  // The rendered positions are the true local rendezvous geometry — tens of
  // metres — not the ~2e16 m void distance.
  EXPECT_EQ(5, rendered_trajectory.size());
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    EXPECT_LT((degrees_of_freedom.position() - Navigation::origin).Norm(),
              150 * Metre);
  }

  // The frame's direction axis is computed at the target's TRUE position —
  // deep in the void, the celestial almost exactly in the −x direction — so
  // the transform maps the true target-to-celestial direction to the frame's
  // +x axis.  (Pre-fix the axis was computed from the raw near-origin
  // coordinates, pointing at the celestial from the home star instead.)  The
  // inputs of the transform are in the frame's anchored representation.
  Instant const t_check = t0_ + 5 * Second;
  auto const to_plotting = renderer_.BarycentricToPlotting(t_check);
  DegreesOfFreedom<Barycentric> const target_dof =
      vessel_trajectory.EvaluateDegreesOfFreedom(t_check);
  Position<Barycentric> const celestial_in_frame_representation =
      celestial_position - target_anchor->OffsetAt(t_check);
  Displacement<Navigation> const image =
      to_plotting({celestial_in_frame_representation,
                   Barycentric::unmoving}).position() -
      to_plotting(target_dof).position();
  EXPECT_GT(image.coordinates().x / image.Norm(), 0.999);
}

TEST_F(RendererTest, RenderPlottingTrajectoryInWorldWithoutTargetVessel) {
  DiscreteTrajectory<Navigation> trajectory_to_render;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(
          /*v=*/Velocity<Navigation>(
              {6 * Metre / Second, 5 * Metre / Second, 4 * Metre / Second}),
          /*Δt=*/1 * Second,
          /*t1=*/t0_,
          /*t2=*/t0_ + 10 * Second),
      /*to=*/trajectory_to_render);

  Instant const rendering_time = t0_ + 5 * Second;
  Position<World> const sun_world_position =
      World::origin +
      Displacement<World>({300 * Metre, 200 * Metre, 100 * Metre});
  Rotation<Barycentric, AliceSun> const planetarium_rotation(
      1 * Radian,
      Bivector<double, Barycentric>({1.0, 1.1, 1.2}),
      DefinesFrame<AliceSun>{});
  RigidMotion<Navigation, Barycentric> const rigid_motion(
      RigidTransformation<Navigation, Barycentric>::Identity(),
      Navigation::nonrotating,
      Navigation::unmoving);
  EXPECT_CALL(*reference_frame_, FromThisFrameAtTime(_))
      .WillRepeatedly(Return(rigid_motion));

  auto const rendered_trajectory =
      renderer_.RenderPlottingTrajectoryInWorld(rendering_time,
                                                trajectory_to_render.begin(),
                                                trajectory_to_render.end(),
                                                sun_world_position,
                                                planetarium_rotation);

  EXPECT_EQ(10, rendered_trajectory.size());
  int index = 0;
  for (auto const& [time, degrees_of_freedom] : rendered_trajectory) {
    EXPECT_EQ(t0_ + index * Second, time);
    // The degrees of freedom are computed using real geometrical transforms.
    // No point in re-doing the computation here, we just check that the numbers
    // are reasonable.
    EXPECT_LT((degrees_of_freedom.position() - World::origin).Norm(),
              452 * Metre);
    EXPECT_LT(degrees_of_freedom.velocity().Norm(), 9 * Metre / Second);
    ++index;
  }
}

// A marker must land where the scene draws the vessel it belongs to.  Anchored
// on the Sun, whose world position is itself a void-scale absolute at
// interstellar distance, the rendering pairs it with the Sun's own position and
// the local geometry of the plot — the frame's 700 m below — falls off the
// bottom of that sum, afresh every frame as the two void-scale terms drift.
// Registered on the vessel, both terms of the pair are local and the geometry
// is exact.
TEST_F(RendererTest, RenderDistinguishedPointsInWorldRegisteredOnTheVessel) {
  // A subsystem 2⁶² m — some 490 light-years — from the Sun.  Positions there
  // are represented relative to the subsystem, so they are small; the Sun's is
  // the void-scale one, where the ULP is 1024 m.
  Displacement<Barycentric> const to_the_sun(
      {-0x1p62 * Metre, 0 * Metre, 0 * Metre});
  // The plotting frame is centred 700 m from the subsystem's origin: a local
  // quantity below that ULP, which is what such a sum cannot carry.
  Displacement<Barycentric> const to_the_frame(
      {700 * Metre, 0 * Metre, 0 * Metre});
  // The vessel is at the subsystem's origin, and the scene, whose own origin
  // follows it, draws it at the origin of `World`.
  Displacement<Barycentric> const from_the_vessel(
      {300 * Metre, 0 * Metre, 0 * Metre});

  Instant const rendering_time = t0_ + 5 * Second;
  auto const planetarium_rotation =
      Rotation<Barycentric, AliceSun>::Identity();
  RigidMotion<Navigation, Barycentric> const from_plotting(
      RigidTransformation<Navigation, Barycentric>(
          Navigation::origin,
          Barycentric::origin + to_the_frame,
          OrthogonalMap<Navigation, Barycentric>::Identity()),
      Navigation::nonrotating,
      Navigation::unmoving);
  EXPECT_CALL(*reference_frame_, FromThisFrameAtTime(_))
      .WillRepeatedly(Return(from_plotting));
  EXPECT_CALL(*reference_frame_, ToThisFrameAtTime(_))
      .WillRepeatedly(Return(from_plotting.Inverse()));
  EXPECT_CALL(celestial_, current_position(_))
      .WillRepeatedly(Return(Barycentric::origin + to_the_sun));

  DistinguishedPoints<Barycentric> points;
  points.emplace(rendering_time,
                 DegreesOfFreedom<Barycentric>(
                     Barycentric::origin + from_the_vessel,
                     Barycentric::unmoving));

  auto const barycentric_to_world =
      renderer_.BarycentricToWorld(planetarium_rotation);
  Position<World> const sun_world_position =
      World::origin + barycentric_to_world(to_the_sun);
  Position<World> const expected =
      World::origin + barycentric_to_world(from_the_vessel);

  auto const registered = renderer_.RenderDistinguishedPointsInWorld(
      rendering_time,
      points.begin(), points.end(),
      sun_world_position,
      planetarium_rotation,
      Ephemeris<Barycentric>::SubsystemPlacement::Stock(),
      Renderer::WorldRegistration{
          .navigation = from_plotting.rigid_transformation().Inverse()(
              Barycentric::origin),
          .world = World::origin});
  ASSERT_EQ(1, registered.size());
  // Exactly, not approximately: the subtraction of the two operands is exact
  // (they are within a factor of two), and the same linear map is then applied
  // to the same bits the expectation uses.
  EXPECT_THAT(registered.begin()->second.position(), AlmostEquals(expected, 0));

  auto const unregistered = renderer_.RenderDistinguishedPointsInWorld(
      rendering_time,
      points.begin(), points.end(),
      sun_world_position,
      planetarium_rotation);
  ASSERT_EQ(1, unregistered.size());
  // The void-scale sum quantizes at 1024 m, so the frame's own 700 m cannot
  // survive it.  The residual measures 1724 m, but only ~324 m of that is the
  // quantization itself — the rest comes from the rotation's own rounding of a
  // 2⁶² m displacement, which a change of compiler or of contraction could
  // move.  The bound is therefore set below every plausible outcome and still
  // far above the registered path, which is exact.
  EXPECT_THAT((unregistered.begin()->second.position() - expected).Norm(),
              Gt(300 * Metre));
}

TEST_F(RendererTest, Serialization) {
  serialization::Renderer message;
  EXPECT_CALL(*reference_frame_, WriteToMessage(_));
  renderer_.WriteToMessage(&message);
  EXPECT_TRUE(message.has_plotting_frame());
}

}  // namespace ksp_plugin
}  // namespace principia
