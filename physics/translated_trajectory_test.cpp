#include "physics/translated_trajectory.hpp"

#include "geometry/frame.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "gtest/gtest.h"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "quantities/si.hpp"
#include "serialization/geometry.pb.h"
#include "testing_utilities/almost_equals.hpp"
#include "testing_utilities/componentwise.hpp"
#include "testing_utilities/discrete_trajectory_factories.hpp"

namespace principia {
namespace physics {

using namespace principia::geometry::_frame;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_translated_trajectory;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
using namespace principia::testing_utilities::_componentwise;
using namespace principia::testing_utilities::_discrete_trajectory_factories;

class TranslatedTrajectoryTest : public ::testing::Test {
 protected:
  using World = Frame<serialization::Frame::TestTag,
                      Inertial,
                      Handedness::Right,
                      serialization::Frame::TEST>;

  Instant const t0_;
};

TEST_F(TranslatedTrajectoryTest, Translation) {
  DegreesOfFreedom<World> const degrees_of_freedom(
      World::origin + Displacement<World>({1 * Metre, 2 * Metre, 3 * Metre}),
      Velocity<World>({4 * Metre / Second,
                       5 * Metre / Second,
                       6 * Metre / Second}));
  DiscreteTrajectory<World> trajectory;
  AppendTrajectoryTimeline(
      NewLinearTrajectoryTimeline(degrees_of_freedom,
                                  /*Δt=*/1 * Second,
                                  /*t1=*/t0_,
                                  /*t2=*/t0_ + 10 * Second),
      trajectory);

  Displacement<World> const displacement(
      {100 * Metre, -200 * Metre, 300 * Metre});
  TranslatedTrajectory<World> const translated_trajectory(trajectory,
                                                          displacement);

  EXPECT_EQ(trajectory.t_min(), translated_trajectory.t_min());
  EXPECT_EQ(trajectory.t_max(), translated_trajectory.t_max());
  Instant const t = t0_ + 5.5 * Second;
  EXPECT_THAT(translated_trajectory.EvaluatePosition(t),
              AlmostEquals(trajectory.EvaluatePosition(t) + displacement, 0));
  EXPECT_EQ(trajectory.EvaluateVelocity(t),
            translated_trajectory.EvaluateVelocity(t));
  EXPECT_THAT(
      translated_trajectory.EvaluateDegreesOfFreedom(t),
      Componentwise(
          AlmostEquals(trajectory.EvaluatePosition(t) + displacement, 0),
          AlmostEquals(trajectory.EvaluateVelocity(t), 0)));
}

}  // namespace physics
}  // namespace principia
