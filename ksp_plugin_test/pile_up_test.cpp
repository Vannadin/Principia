#include "ksp_plugin/pile_up.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "astronomy/epoch.hpp"
#include "base/not_null.hpp"
#include "geometry/frame.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/quaternion.hpp"
#include "geometry/space.hpp"
#include "geometry/space_transformations.hpp"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "integrators/embedded_explicit_runge_kutta_nyström_integrator.hpp"
#include "integrators/methods.hpp"
#include "integrators/symplectic_runge_kutta_nyström_integrator.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin/identification.hpp"
#include "ksp_plugin/integrators.hpp"
#include "ksp_plugin/part.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory_segment_iterator.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "physics/mock_ephemeris.hpp"
#include "physics/rigid_motion.hpp"
#include "physics/sector.hpp"
#include "physics/tensors.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/almost_equals.hpp"
#include "testing_utilities/componentwise.hpp"
#include "testing_utilities/matchers.hpp"
#include "testing_utilities/numerics_matchers.hpp"

namespace principia {
namespace ksp_plugin {

using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::IsEmpty;
using ::testing::Lt;
using ::testing::MockFunction;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::Truly;
using ::testing::_;
using namespace principia::astronomy::_epoch;
using namespace principia::base::_not_null;
using namespace principia::geometry::_frame;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_quaternion;
using namespace principia::geometry::_space;
using namespace principia::geometry::_space_transformations;
using namespace principia::integrators::_embedded_explicit_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symplectic_runge_kutta_nyström_integrator;  // NOLINT
using namespace principia::ksp_plugin::_frames;
using namespace principia::ksp_plugin::_identification;
using namespace principia::ksp_plugin::_integrators;
using namespace principia::ksp_plugin::_part;
using namespace principia::ksp_plugin::_pile_up;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory_segment_iterator;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_mock_ephemeris;
using namespace principia::physics::_rigid_motion;
using namespace principia::physics::_sector;
using namespace principia::physics::_tensors;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
using namespace principia::testing_utilities::_componentwise;
using namespace principia::testing_utilities::_matchers;
using namespace principia::testing_utilities::_numerics_matchers;

// A helper class to expose the internal state of a pile-up for testing.
class TestablePileUp : public PileUp {
 public:
  using PileUp::PileUp;
  using PileUp::DeformPileUpIfNeeded;
  using PileUp::AdvanceTime;
  using PileUp::NudgeParts;

  Mass const& mass() const {
    return mass_;
  }

  Vector<Force, Barycentric> const& intrinsic_force() const {
    return intrinsic_force_;
  }

  DiscreteTrajectorySegmentIterator<Barycentric> psychohistory() const {
    return psychohistory_;
  }

  auto const& trajectory() const {
    return trajectory_;
  }

  void AppendToTrajectory(
      Instant const& t,
      DegreesOfFreedom<Barycentric> const& degrees_of_freedom) {
    CHECK_OK(trajectory_.Append(t, degrees_of_freedom));
  }

  PartTo<RigidMotion<RigidPart, NonRotatingPileUp>> const&
  actual_part_rigid_motion() const {
    return actual_part_rigid_motion_;
  }

  PartTo<RigidMotion<RigidPart, Apparent>> const&
  apparent_part_rigid_motion() const {
    return apparent_part_rigid_motion_;
  }
};

class PileUpTest : public testing::Test {
 protected:
  using CorrectedPileUp = Frame<struct CorrectedPileUpTag, NonRotating>;
  using Vessel = Frame<struct VesselTag>;

  PileUpTest()
      : inertia_tensor1_(MakeWaterSphereInertiaTensor(mass1_)),
        inertia_tensor2_(MakeWaterSphereInertiaTensor(mass2_)),
        p1_(part_id1_,
            "p1",
            mass1_,
            EccentricPart::origin,
            inertia_tensor1_,
            RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
                p1_dof_),
            /*deletion_callback=*/nullptr),
        p2_(part_id2_,
            "p2",
            mass2_,
            EccentricPart::origin,
            inertia_tensor2_,
            RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
                p2_dof_),
            /*deletion_callback=*/nullptr) {}

  void CheckPreDeformPileUpInvariants(TestablePileUp& pile_up) {
    EXPECT_EQ(3 * Kilogram, pile_up.mass());

    EXPECT_THAT(
        pile_up.psychohistory()->back().degrees_of_freedom,
        Componentwise(AlmostEquals(Barycentric::origin +
                                       Displacement<Barycentric>(
                                           {13.0 / 3.0 * Metre,
                                            4.0 * Metre,
                                            11.0 / 3.0 * Metre}), 0),
                      AlmostEquals(Velocity<Barycentric>(
                                       {130.0 / 3.0 * Metre / Second,
                                        40.0 * Metre / Second,
                                        110.0 / 3.0 * Metre / Second}), 4)));

    EXPECT_THAT(
        pile_up.actual_part_rigid_motion().at(&p1_)({RigidPart::origin,
                                                     RigidPart::unmoving}),
        Componentwise(AlmostEquals(NonRotatingPileUp::origin +
                                       Displacement<NonRotatingPileUp>(
                                           {-10.0 / 3.0 * Metre,
                                            -2.0 * Metre,
                                            -2.0 / 3.0 * Metre}), 1),
                      AlmostEquals(Velocity<NonRotatingPileUp>(
                                       {-100.0 / 3.0 * Metre / Second,
                                        -20.0 * Metre / Second,
                                        -20.0 / 3.0 * Metre / Second}), 7)));
    EXPECT_THAT(
        pile_up.actual_part_rigid_motion().at(&p2_)({RigidPart::origin,
                                                     RigidPart::unmoving}),
        Componentwise(AlmostEquals(NonRotatingPileUp::origin +
                                       Displacement<NonRotatingPileUp>(
                                           {5.0 / 3.0 * Metre,
                                            1.0 * Metre,
                                            1.0 / 3.0 * Metre}), 3),
                      AlmostEquals(Velocity<NonRotatingPileUp>(
                                       {50.0 / 3.0 * Metre / Second,
                                        10.0 * Metre / Second,
                                        10.0 / 3.0 * Metre / Second}), 17)));

    // Centre of mass of `p1_` and `p2_` in `Apparent`, in SI units:
    //   {1 / 9, -1 / 3, -2 / 9} {10 / 9, -10 / 3, -20 / 9}
    DegreesOfFreedom<Apparent> const p1_dof(
        Apparent::origin +
            Displacement<Apparent>(
                {-11.0 / 3.0 * Metre, -1.0 * Metre, 2.0 / 3.0 * Metre}),
        Velocity<Apparent>({-110.0 / 3.0 * Metre / Second,
                            -10.0 * Metre / Second,
                            20.0 / 3.0 * Metre / Second}));
    DegreesOfFreedom<Apparent> const p2_dof(
        Apparent::origin + Displacement<Apparent>(
                               {2.0 * Metre, 0.0 * Metre, -2.0 / 3.0 * Metre}),
        Velocity<Apparent>({20.0 * Metre / Second,
                            0.0 * Metre / Second,
                            -20.0 / 3.0 * Metre / Second}));
    pile_up.SetPartApparentRigidMotion(
        &p1_, RigidMotion<RigidPart, Apparent>::MakeNonRotatingMotion(p1_dof));
    pile_up.SetPartApparentRigidMotion(
        &p2_, RigidMotion<RigidPart, Apparent>::MakeNonRotatingMotion(p2_dof));

    EXPECT_THAT(
        pile_up.apparent_part_rigid_motion().at(&p1_)(
            {RigidPart::origin, RigidPart::unmoving}),
        Componentwise(
            AlmostEquals(
                Apparent::origin +
                    Displacement<Apparent>(
                        {-11.0 / 3.0 * Metre, -1.0 * Metre, 2.0 / 3.0 * Metre}),
                0),
            AlmostEquals(Velocity<Apparent>({-110.0 / 3.0 * Metre / Second,
                                             -10.0 * Metre / Second,
                                             20.0 / 3.0 * Metre / Second}),
                         4)));
    EXPECT_THAT(
        pile_up.apparent_part_rigid_motion().at(&p2_)(
            {RigidPart::origin, RigidPart::unmoving}),
        Componentwise(
            AlmostEquals(
                Apparent::origin +
                    Displacement<Apparent>(
                        {2.0 * Metre, 0.0 * Metre, -2.0 / 3.0 * Metre}),
                0),
            AlmostEquals(Velocity<Apparent>({20.0 * Metre / Second,
                                             0.0 * Metre / Second,
                                             -20.0 / 3.0 * Metre / Second}),
                         4)));
  }

  void CheckPreAdvanceTimeInvariants(TestablePileUp& pile_up) {
    EXPECT_THAT(pile_up.actual_part_rigid_motion().at(&p1_)(
                    {RigidPart::origin, RigidPart::unmoving}),
                Componentwise(AlmostEquals(NonRotatingPileUp::origin +
                                               Displacement<NonRotatingPileUp>(
                                                   {-34.0 / 9.0 * Metre,
                                                    -2.0 / 3.0 * Metre,
                                                    8.0 / 9.0 * Metre}),
                                           1122),
                              AlmostEquals(Velocity<NonRotatingPileUp>(
                                               {-340.0 / 9.0 * Metre / Second,
                                                -20.0 / 3.0 * Metre / Second,
                                                80.0 / 9.0 * Metre / Second}),
                                           1372)));
    EXPECT_THAT(pile_up.actual_part_rigid_motion().at(&p2_)(
                    {RigidPart::origin, RigidPart::unmoving}),
                Componentwise(AlmostEquals(NonRotatingPileUp::origin +
                                               Displacement<NonRotatingPileUp>(
                                                   {17.0 / 9.0 * Metre,
                                                    1.0 / 3.0 * Metre,
                                                    -4.0 / 9.0 * Metre}),
                                           1122),
                              AlmostEquals(Velocity<NonRotatingPileUp>(
                                               {170.0 / 9.0 * Metre / Second,
                                                10.0 / 3.0 * Metre / Second,
                                                -40.0 / 9.0 * Metre / Second}),
                                           1373)));
    EXPECT_THAT(pile_up.apparent_part_rigid_motion(), IsEmpty());
  }

  MockFunction<void()> deletion_callback_;

  PartId const part_id1_ = 111;
  PartId const part_id2_ = 222;
  Mass const mass1_ = 1 * Kilogram;
  Mass const mass2_ = 2 * Kilogram;
  InertiaTensor<RigidPart> inertia_tensor1_;
  InertiaTensor<RigidPart> inertia_tensor2_;

  // Centre of mass of `p1_` and `p2_` in `Barycentric`, in SI units:
  //   {13 / 3, 4, 11 / 3} {130 / 3, 40, 110 / 3}
  DegreesOfFreedom<Barycentric> const p1_dof_ = DegreesOfFreedom<Barycentric>(
      Barycentric::origin +
          Displacement<Barycentric>({1 * Metre, 2 * Metre, 3 * Metre}),
      Velocity<Barycentric>(
          {10 * Metre / Second, 20 * Metre / Second, 30 * Metre / Second}));
  DegreesOfFreedom<Barycentric> const p2_dof_ = DegreesOfFreedom<Barycentric>(
      Barycentric::origin +
          Displacement<Barycentric>({6 * Metre, 5 * Metre, 4 * Metre}),
      Velocity<Barycentric>(
          {60 * Metre / Second, 50 * Metre / Second, 40 * Metre / Second}));

  Part p1_;
  Part p2_;
};

TEST_F(PileUpTest, MidStepIntrinsicForce) {
  // An empty ephemeris; the parameters don't matter, since there are no bodies
  // to integrate.
  // NOTE(egg): ... except we have to put a body because `Ephemeris` doesn't
  // want to be empty.  We put a tiny one very far.
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  bodies.emplace_back(make_not_null_unique<MassiveBody>(1 * Kilogram));
  std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
      DegreesOfFreedom<Barycentric>{
          Barycentric::origin +
              Displacement<Barycentric>(
                  {std::pow(2, 100) * Metre, 0 * Metre, 0 * Metre}),
          Barycentric::unmoving}};
  Ephemeris<Barycentric> ephemeris{
      std::move(bodies),
      initial_state,
      /*initial_time=*/J2000,
      /*accuracy_parameters=*/{/*fitting_tolerance=*/1 * Metre,
                               /*geopotential_tolerance=*/0x1p-24},
      Ephemeris<Barycentric>::FixedStepParameters{
          SymplecticRungeKuttaNyströmIntegrator<
              BlanesMoan2002SRKN6B,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          1 * Second}};

  Time const fixed_step = 10 * Second;
  Ephemeris<Barycentric>::FixedStepParameters const fixed_parameters{
      SymplecticRungeKuttaNyströmIntegrator<
          BlanesMoan2002SRKN6B,
          Ephemeris<Barycentric>::NewtonianMotionEquation>(),
      fixed_step};
  Ephemeris<Barycentric>::AdaptiveStepParameters const adaptive_parameters{
      EmbeddedExplicitRungeKuttaNyströmIntegrator<
          DormandالمكاوىPrince1986RKN434FM,
          Ephemeris<Barycentric>::NewtonianMotionEquation>(),
      /*max_steps=*/std::numeric_limits<std::int64_t>::max(),
      /*length_integration_tolerance*/ 1 * Micro(Metre),
      /*speed_integration_tolerance=*/1 * Micro(Metre) / Second};

  EXPECT_CALL(deletion_callback_, Call()).Times(1);
  TestablePileUp pile_up({&p1_}, J2000,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         deletion_callback_.AsStdFunction());
  Velocity<Barycentric> const old_velocity =
      p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving}).velocity();

  EXPECT_OK(pile_up.AdvanceTime(J2000 + 1.5 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(
      p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving}).velocity(),
      AlmostEquals(old_velocity, 2));

  Vector<Acceleration, Barycentric> const a{{1729 * Metre / Pow<2>(Second),
                                             -168 * Metre / Pow<2>(Second),
                                             504 * Metre / Pow<2>(Second)}};
  p1_.apply_intrinsic_force(p1_.mass() * a);
  pile_up.RecomputeFromParts();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 2 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(
      p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving}).velocity(),
      AlmostEquals(old_velocity + 0.5 * fixed_step * a, 1));
}

TEST_F(PileUpTest, OnRailsBurn) {
  // The same quasi-empty ephemeris as in `MidStepIntrinsicForce`.
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  bodies.emplace_back(make_not_null_unique<MassiveBody>(1 * Kilogram));
  std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
      DegreesOfFreedom<Barycentric>{
          Barycentric::origin +
              Displacement<Barycentric>(
                  {std::pow(2, 100) * Metre, 0 * Metre, 0 * Metre}),
          Barycentric::unmoving}};
  Ephemeris<Barycentric> ephemeris{
      std::move(bodies),
      initial_state,
      /*initial_time=*/J2000,
      /*accuracy_parameters=*/{/*fitting_tolerance=*/1 * Metre,
                               /*geopotential_tolerance=*/0x1p-24},
      Ephemeris<Barycentric>::FixedStepParameters{
          SymplecticRungeKuttaNyströmIntegrator<
              BlanesMoan2002SRKN6B,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          1 * Second}};

  Time const fixed_step = 10 * Second;

  EXPECT_CALL(deletion_callback_, Call()).Times(1);
  TestablePileUp pile_up({&p1_}, J2000,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         deletion_callback_.AsStdFunction());
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 1.5 * fixed_step));
  pile_up.NudgeParts();

  auto const velocity = [this]() {
    return p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving})
        .velocity();
  };
  Vector<double, Barycentric> const direction({1, 0, 0});
  Force const thrust = 2 * Newton;
  SpecificImpulse const specific_impulse = 100 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;

  // A burn whose propellant outlasts the step: it thrusts over the entire
  // [1.5 fixed_step, 2 fixed_step] interval, gaining Циолковский's Δv.  The
  // game owns the mass bookkeeping, so `mass_` is untouched.
  Velocity<Barycentric> const v0 = velocity();
  Mass const m0 = mass1_;
  pile_up.set_on_rails_burn({thrust,
                             specific_impulse,
                             /*initial_mass=*/m0,
                             direction,
                             /*max_duration=*/1 * Hour});
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 2 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(pile_up.mass(), AlmostEquals(mass1_, 0));
  EXPECT_FALSE(pile_up.on_rails_burn().has_value());
  Mass const m1 = m0 - 0.5 * fixed_step * mass_flow;
  Speed const Δv1 = specific_impulse * std::log(m0 / m1);
  EXPECT_THAT((velocity() - v0).Norm(),
              AbsoluteErrorFrom(Δv1, Lt(1e-6 * Metre / Second)));
  EXPECT_THAT((velocity() - v0).coordinates().x,
              AbsoluteErrorFrom(Δv1, Lt(1e-6 * Metre / Second)));

  // A burn that exhausts its propellant midway through the step, coasting
  // beyond; the game hands us the mass left by the previous burn.
  Velocity<Barycentric> const v1 = velocity();
  Time const burn_duration = 2 * Second;
  pile_up.set_on_rails_burn({thrust,
                             specific_impulse,
                             /*initial_mass=*/m1,
                             direction,
                             burn_duration});
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 3 * fixed_step));
  pile_up.NudgeParts();
  Mass const m2 = m1 - burn_duration * mass_flow;
  Speed const Δv2 = specific_impulse * std::log(m1 / m2);
  EXPECT_THAT((velocity() - v1).Norm(),
              AbsoluteErrorFrom(Δv2, Lt(1e-6 * Metre / Second)));

  // With no burn set, the next step is a pure coast.
  Velocity<Barycentric> const v2 = velocity();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 4 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(velocity(), AlmostEquals(v2, 0, 8));
}

TEST_F(PileUpTest, OnRailsBurnPrecedenceAndClear) {
  // The same quasi-empty ephemeris as in `MidStepIntrinsicForce`.
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  bodies.emplace_back(make_not_null_unique<MassiveBody>(1 * Kilogram));
  std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
      DegreesOfFreedom<Barycentric>{
          Barycentric::origin +
              Displacement<Barycentric>(
                  {std::pow(2, 100) * Metre, 0 * Metre, 0 * Metre}),
          Barycentric::unmoving}};
  Ephemeris<Barycentric> ephemeris{
      std::move(bodies),
      initial_state,
      /*initial_time=*/J2000,
      /*accuracy_parameters=*/{/*fitting_tolerance=*/1 * Metre,
                               /*geopotential_tolerance=*/0x1p-24},
      Ephemeris<Barycentric>::FixedStepParameters{
          SymplecticRungeKuttaNyströmIntegrator<
              BlanesMoan2002SRKN6B,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          1 * Second}};

  Time const fixed_step = 10 * Second;

  EXPECT_CALL(deletion_callback_, Call()).Times(1);
  TestablePileUp pile_up({&p1_}, J2000,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         deletion_callback_.AsStdFunction());
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 1.5 * fixed_step));
  pile_up.NudgeParts();

  auto const velocity = [this]() {
    return p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving})
        .velocity();
  };
  OnRailsBurn const burn{/*thrust=*/2 * Newton,
                         /*specific_impulse=*/100 * Metre / Second,
                         /*initial_mass=*/mass1_,
                         /*direction=*/Vector<double, Barycentric>({1, 0, 0}),
                         /*max_duration=*/1 * Hour};

  // An intrinsic force wins over a simultaneous burn, and consumes it.
  Vector<Acceleration, Barycentric> const a{{3 * Metre / Pow<2>(Second),
                                             -4 * Metre / Pow<2>(Second),
                                             12 * Metre / Pow<2>(Second)}};
  p1_.apply_intrinsic_force(p1_.mass() * a);
  pile_up.RecomputeFromParts();
  pile_up.set_on_rails_burn(burn);
  Velocity<Barycentric> const v0 = velocity();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 2 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT((velocity() - (v0 + 0.5 * fixed_step * a)).Norm(),
              Lt(1e-10 * Metre / Second));
  EXPECT_FALSE(pile_up.on_rails_burn().has_value());
  EXPECT_FALSE(pile_up.on_rails_burn_for_prediction().has_value());

  // With the force gone and no new burn, the consumed burn must not refire.
  p1_.clear_intrinsic_force();
  pile_up.RecomputeFromParts();
  Velocity<Barycentric> const v1 = velocity();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 3 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(velocity(), AlmostEquals(v1, 0, 8));

  // A cleared burn does not fire.
  pile_up.set_on_rails_burn(burn);
  pile_up.clear_on_rails_burn();
  Velocity<Barycentric> const v2 = velocity();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 4 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(velocity(), AlmostEquals(v2, 0, 8));
  EXPECT_FALSE(pile_up.on_rails_burn_for_prediction().has_value());

  // A catch-up with nothing to do still consumes the burn.
  pile_up.set_on_rails_burn(burn);
  EXPECT_OK(pile_up.DeformAndAdvanceTime(J2000 + 4 * fixed_step));
  EXPECT_FALSE(pile_up.on_rails_burn().has_value());
}

// An on-rails burn (WS3) in the force-free void between two star systems whose
// far field is damped (WS2/WS2b).  The probe sits off the inter-star axis, so
// an *undamped* field would push it in −y; the damping makes the void exactly
// force-free, and the burn (along +x) must therefore produce a pure +x Δv with
// no gravitational contamination.  Exercises the massless burn integration
// through the multi-subsystem, far-field-damped ephemeris kernels.
TEST_F(PileUpTest, OnRailsBurnInDampedVoid) {
  // Two equal stars, each its own subsystem, 10⁸ m apart on the x axis.
  Mass const star_mass = 1e24 * Kilogram;
  Length const separation = 1e8 * Metre;
  Acceleration const far_field_damping_floor = 1 * Metre / Pow<2>(Second);
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
  bodies.emplace_back(make_not_null_unique<MassiveBody>(star_mass));
  bodies.emplace_back(make_not_null_unique<MassiveBody>(star_mass));
  std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
      DegreesOfFreedom<Barycentric>{Barycentric::origin, Barycentric::unmoving},
      DegreesOfFreedom<Barycentric>{
          Barycentric::origin +
              Displacement<Barycentric>({separation, 0 * Metre, 0 * Metre}),
          Barycentric::unmoving}};
  Ephemeris<Barycentric> ephemeris{
      std::move(bodies),
      initial_state,
      /*initial_time=*/J2000,
      /*accuracy_parameters=*/{/*fitting_tolerance=*/1 * Metre,
                               /*geopotential_tolerance=*/0x1p-24},
      Ephemeris<Barycentric>::FixedStepParameters{
          SymplecticRungeKuttaNyströmIntegrator<
              BlanesMoan2002SRKN6B,
              Ephemeris<Barycentric>::NewtonianMotionEquation>(),
          1 * Second},
      /*subsystems=*/std::vector<int>{0, 1},
      far_field_damping_floor};

  // The probe sits well inside the void (√2 · 5×10⁷ m ≈ 7×10⁷ m from each
  // star, far beyond the ~8×10⁶ m cutoff radius) and off the x axis, so an
  // undamped field would pull it in −y.
  Part probe(part_id1_,
             "probe",
             mass1_,
             EccentricPart::origin,
             inertia_tensor1_,
             RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
                 DegreesOfFreedom<Barycentric>(
                     Barycentric::origin +
                         Displacement<Barycentric>(
                             {5e7 * Metre, 5e7 * Metre, 0 * Metre}),
                     Barycentric::unmoving)),
             /*deletion_callback=*/nullptr);

  Time const fixed_step = 10 * Second;
  EXPECT_CALL(deletion_callback_, Call()).Times(1);
  TestablePileUp pile_up({&probe}, J2000,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         deletion_callback_.AsStdFunction());
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 1.5 * fixed_step));
  pile_up.NudgeParts();

  auto const velocity = [&probe]() {
    return probe.rigid_motion()({RigidPart::origin, RigidPart::unmoving})
        .velocity();
  };

  // The void is exactly force-free: with no burn the probe coasts, keeping its
  // (zero) velocity.  An undamped field would have added a −y drift here.
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 2.5 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(velocity().Norm(), Lt(1e-9 * Metre / Second));

  Vector<double, Barycentric> const direction({1, 0, 0});
  Force const thrust = 2 * Newton;
  SpecificImpulse const specific_impulse = 100 * Metre / Second;
  Variation<Mass> const mass_flow = thrust / specific_impulse;

  // A burn along +x whose propellant outlasts the step: the Δv is Циолковский's
  // and lies purely along +x — the damped void contributes no acceleration.
  Velocity<Barycentric> const v0 = velocity();
  Mass const m0 = mass1_;
  pile_up.set_on_rails_burn({thrust,
                             specific_impulse,
                             /*initial_mass=*/m0,
                             direction,
                             /*max_duration=*/1 * Hour});
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 3.5 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_FALSE(pile_up.on_rails_burn().has_value());
  Mass const m1 = m0 - fixed_step * mass_flow;
  Speed const Δv = specific_impulse * std::log(m0 / m1);
  Velocity<Barycentric> const Δv_vector = velocity() - v0;
  EXPECT_THAT(Δv_vector.coordinates().x,
              AbsoluteErrorFrom(Δv, Lt(1e-6 * Metre / Second)));
  // No gravitational contamination transverse to the thrust.
  EXPECT_THAT(Δv_vector.coordinates().y,
              AbsoluteErrorFrom(0 * Metre / Second, Lt(1e-9 * Metre / Second)));
  EXPECT_THAT(Δv_vector.coordinates().z,
              AbsoluteErrorFrom(0 * Metre / Second, Lt(1e-9 * Metre / Second)));

  // A burn that exhausts its propellant midway coasts (force-free) afterwards.
  Velocity<Barycentric> const v1 = velocity();
  Time const burn_duration = 4 * Second;
  pile_up.set_on_rails_burn({thrust,
                             specific_impulse,
                             /*initial_mass=*/m1,
                             direction,
                             burn_duration});
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 4.5 * fixed_step));
  pile_up.NudgeParts();
  Mass const m2 = m1 - burn_duration * mass_flow;
  Speed const Δv2 = specific_impulse * std::log(m1 / m2);
  EXPECT_THAT((velocity() - v1).Norm(),
              AbsoluteErrorFrom(Δv2, Lt(1e-6 * Metre / Second)));

  // With no burn set, the next step is a pure coast in the void.
  Velocity<Barycentric> const v2 = velocity();
  EXPECT_OK(pile_up.AdvanceTime(J2000 + 5.5 * fixed_step));
  pile_up.NudgeParts();
  EXPECT_THAT(velocity(), AlmostEquals(v2, 0, 8));
}

TEST_F(PileUpTest, Serialization) {
  MockEphemeris<Barycentric> ephemeris;
  p1_.apply_intrinsic_force(
      Vector<Force, Barycentric>({1 * Newton, 2 * Newton, 3 * Newton}));
  p2_.apply_intrinsic_force(
      Vector<Force, Barycentric>({11 * Newton, 21 * Newton, 31 * Newton}));
  EXPECT_CALL(deletion_callback_, Call()).Times(2);
  TestablePileUp const pile_up({&p1_, &p2_},
                               J2000,
                               DefaultPsychohistoryParameters(),
                               DefaultHistoryParameters(),
                               &ephemeris,
                               deletion_callback_.AsStdFunction());

  serialization::PileUp message;
  pile_up.WriteToMessage(&message);

  EXPECT_EQ(2, message.part_id_size());
  EXPECT_EQ(part_id1_, message.part_id(0));
  EXPECT_EQ(part_id2_, message.part_id(1));
  EXPECT_EQ(2, message.history().segment_size());
  EXPECT_EQ(1, message.history().segment(0).zfp().timeline_size());
  EXPECT_EQ(1, message.history().segment(1).zfp().timeline_size());
  EXPECT_EQ(2, message.actual_part_rigid_motion().size());
  EXPECT_TRUE(message.apparent_part_rigid_motion().empty());

  auto const part_id_to_part = [this](PartId const part_id) {
    if (part_id == part_id1_) {
      return &p1_;
    }
    if (part_id == part_id2_) {
      return &p2_;
    }
    LOG(FATAL) << "Unexpected part id " << part_id;
    std::abort();
  };
  auto const p = PileUp::ReadFromMessage(message,
                                         part_id_to_part,
                                         &ephemeris,
                                         deletion_callback_.AsStdFunction());

  serialization::PileUp second_message;
  p->WriteToMessage(&second_message);
  EXPECT_THAT(message, EqualsProto(second_message));
}

TEST_F(PileUpTest, SerializationCompatibility) {
  MockEphemeris<Barycentric> ephemeris;
  p1_.apply_intrinsic_force(
      Vector<Force, Barycentric>({1 * Newton, 2 * Newton, 3 * Newton}));
  p2_.apply_intrinsic_force(
      Vector<Force, Barycentric>({11 * Newton, 21 * Newton, 31 * Newton}));
  EXPECT_CALL(deletion_callback_, Call()).Times(2);
  TestablePileUp const pile_up({&p1_, &p2_},
                               J2000,
                               DefaultPsychohistoryParameters(),
                               DefaultHistoryParameters(),
                               &ephemeris,
                               deletion_callback_.AsStdFunction());

  serialization::PileUp message;
  pile_up.WriteToMessage(&message);

  // Clear the children to simulate pre-Cesàro serialization.
  message.mutable_history()->clear_children();
  EXPECT_EQ(1, message.history().segment(0).zfp().timeline_size());
  EXPECT_EQ(1, message.history().segment(1).zfp().timeline_size());

  auto const part_id_to_part = [this](PartId const part_id) {
    if (part_id == part_id1_) {
      return &p1_;
    }
    if (part_id == part_id2_) {
      return &p2_;
    }
    LOG(FATAL) << "Unexpected part id " << part_id;
    std::abort();
  };
  auto const p = PileUp::ReadFromMessage(message,
                                         part_id_to_part,
                                         &ephemeris,
                                         deletion_callback_.AsStdFunction());

  EXPECT_CALL(ephemeris, FlowWithAdaptiveStep(_, _, _, _, _, _))
      .WillOnce(DoAll(
          AppendToDiscreteTrajectory(DegreesOfFreedom<Barycentric>(
              Barycentric::origin +
                  Displacement<Barycentric>({1.0 * Metre,
                                             14.0 * Metre,
                                             31.0 / 3.0 * Metre}),
              Velocity<Barycentric>({10.0 * Metre / Second,
                                     140.0 * Metre / Second,
                                     310.0 / 3.0 * Metre / Second}))),
          Return(absl::OkStatus())));
  EXPECT_OK(p->DeformAndAdvanceTime(J2000 + 1 * Second));
}

// The mass-based rebase boundary and its hysteresis, on an unequal pair of
// subsystems: A (the pile-up's, heavy) and B (light), separated by 4 × 10¹⁶ m.
// The dominance boundary μ_A/d_A² = μ_B/d_B² is not the geometric midpoint —
// it sits at d_B/d_A = √(μ_B/μ_A) — and switching representations further
// requires the destination to dominate by the hysteresis margin, in either
// direction.
TEST_F(PileUpTest, MassBasedRebaseHysteresis) {
  // This test pins the dominance hysteresis in isolation: the coordinates
  // used here exceed the production re-anchor bound, so raise it out of the
  // way lest the uniform-representation invariant adopt anchors mid-test.
  Length const saved_bound = PileUp::re_anchor_bound_for_testing_;
  PileUp::re_anchor_bound_for_testing_ = 1e20 * Metre;
  absl::Cleanup restore_bound = [saved_bound] {
    PileUp::re_anchor_bound_for_testing_ = saved_bound;
  };
  Length const separation = 4e16 * Metre;
  GravitationalParameter const μ_b =
      1.3e20 * Pow<3>(Metre) / Pow<2>(Second);
  GravitationalParameter const μ_a = 16 * μ_b;
  Displacement<Barycentric> const b_from_a(
      {separation, 0 * Metre, 0 * Metre});

  MockEphemeris<Barycentric> ephemeris;
  EXPECT_CALL(ephemeris, number_of_subsystems())
      .WillRepeatedly(Return(2));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(0))
      .WillRepeatedly(ReturnRef(μ_a));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(1))
      .WillRepeatedly(ReturnRef(μ_b));
  // The barycentre of each subsystem is at rest at its local origin.
  EXPECT_CALL(ephemeris, subsystem_barycentre(_, _))
      .WillRepeatedly(Return(Barycentric::origin));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 0, _))
      .WillRepeatedly(Return(Displacement<Barycentric>{}));
  EXPECT_CALL(ephemeris, subsystem_conversion(1, 1, _))
      .WillRepeatedly(Return(Displacement<Barycentric>{}));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 1, _))
      .WillRepeatedly(Return(-b_from_a));
  EXPECT_CALL(ephemeris, subsystem_conversion(1, 0, _))
      .WillRepeatedly(Return(b_from_a));

  Velocity<Barycentric> const v(
      {1 * Metre / Second, 0 * Metre / Second, 0 * Metre / Second});
  auto const at_x = [&](Length const& x) {
    return DegreesOfFreedom<Barycentric>(
        Barycentric::origin +
            Displacement<Barycentric>({x, 0 * Metre, 0 * Metre}),
        v);
  };

  Instant const t0 = J2000;
  TestablePileUp pile_up({&p1_}, t0,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         /*deletion_callback=*/nullptr);

  // Past the geometric midpoint — nearer to B than to A — but A, sixteen
  // times heavier, still dominates: no rebase.
  pile_up.AppendToTrajectory(t0 + 1 * Second, at_x(2.5e16 * Metre));
  EXPECT_TRUE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(0, pile_up.subsystem());

  // B dominates, but by less than the hysteresis margin: still no rebase.
  pile_up.AppendToTrajectory(t0 + 2 * Second, at_x(3.45e16 * Metre));
  EXPECT_TRUE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(0, pile_up.subsystem());

  // B dominates beyond the margin: the pile-up is rebased, and its trajectory
  // is now represented relative to B's local origin.
  pile_up.AppendToTrajectory(t0 + 3 * Second, at_x(3.6e16 * Metre));
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(1, pile_up.subsystem());
  EXPECT_EQ(at_x(3.6e16 * Metre).position() - b_from_a,
            pile_up.trajectory().back().degrees_of_freedom.position());

  // Back on the A side of the dominance boundary — A is the (raw) dominant
  // subsystem again — but not beyond the margin: the hysteresis keeps the
  // pile-up on B, so weaving around the boundary does not churn.
  pile_up.AppendToTrajectory(t0 + 4 * Second,
                             at_x(3.0e16 * Metre - separation));
  EXPECT_TRUE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(1, pile_up.subsystem());

  // Deep into A's system: the margin is exceeded and the pile-up returns.
  pile_up.AppendToTrajectory(t0 + 5 * Second,
                             at_x(1.0e16 * Metre - separation));
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(0, pile_up.subsystem());
}

// A rebase into a subsystem whose local origin moves relative to the pile-up's:
// each point of the trajectory must be translated by the offset at the POINT's
// own time and its velocity by the relative velocity of the origins.
// Translating the whole history by the offset at a single instant would
// misplace a point aged Δt by v_rel · Δt — a day of history against a
// 300 km/s pair is off by 2.6 × 10¹⁰ m.
TEST_F(PileUpTest, RebaseTranslatesEachPointAtItsOwnTime) {
  // This test pins the unanchored translation path in isolation; keep the
  // uniform-representation invariant from anchoring the pile-up first.
  Length const saved_bound = PileUp::re_anchor_bound_for_testing_;
  PileUp::re_anchor_bound_for_testing_ = 1e20 * Metre;
  absl::Cleanup restore_bound = [saved_bound] {
    PileUp::re_anchor_bound_for_testing_ = saved_bound;
  };
  Length const separation = 4e16 * Metre;
  GravitationalParameter const μ_b =
      1.3e20 * Pow<3>(Metre) / Pow<2>(Second);
  GravitationalParameter const μ_a = 16 * μ_b;
  Displacement<Barycentric> const b_from_a_at_t0(
      {separation, 0 * Metre, 0 * Metre});
  Velocity<Barycentric> const v_rel({300 * Kilo(Metre) / Second,
                                     0 * Metre / Second,
                                     0 * Metre / Second});
  Instant const t0 = J2000;
  auto const b_from_a = [&](Instant const& t) {
    return b_from_a_at_t0 + v_rel * (t - t0);
  };

  MockEphemeris<Barycentric> ephemeris;
  EXPECT_CALL(ephemeris, number_of_subsystems())
      .WillRepeatedly(Return(2));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(0))
      .WillRepeatedly(ReturnRef(μ_a));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(1))
      .WillRepeatedly(ReturnRef(μ_b));
  EXPECT_CALL(ephemeris, subsystem_barycentre(_, _))
      .WillRepeatedly(Return(Barycentric::origin));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 0, _))
      .WillRepeatedly(Return(Displacement<Barycentric>{}));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 1, _))
      .WillRepeatedly(Invoke([&](int, int, Instant const& t) {
        return -b_from_a(t);
      }));
  EXPECT_CALL(ephemeris, subsystem_velocity_conversion(0, 1))
      .WillRepeatedly(Return(-v_rel));

  Velocity<Barycentric> const v(
      {1 * Metre / Second, 0 * Metre / Second, 0 * Metre / Second});
  auto const at_x = [&](Length const& x) {
    return DegreesOfFreedom<Barycentric>(
        Barycentric::origin +
            Displacement<Barycentric>({x, 0 * Metre, 0 * Metre}),
        v);
  };

  // The pile-up's front point is its construction centre of mass; place a lone
  // part there so the front is a day-old point where B dominates beyond the
  // margin, then append a fresh present state.
  Part p_far(part_id1_,
             "p-far",
             mass1_,
             EccentricPart::origin,
             inertia_tensor1_,
             RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
                 at_x(3.5e16 * Metre)),
             /*deletion_callback=*/nullptr);
  Instant const t1 = t0 + 1 * Day;
  TestablePileUp pile_up({&p_far}, t0,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         /*deletion_callback=*/nullptr);
  Position<Barycentric> const front_before =
      pile_up.trajectory().front().degrees_of_freedom.position();
  pile_up.AppendToTrajectory(t1, at_x(3.8e16 * Metre));
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(1, pile_up.subsystem());

  // The points are translated at their own times, not at the time of the
  // rebase, and the velocities are translated by the origins' relative
  // velocity.  The present state (`back`) was appended exactly, so it survives
  // bit for bit; the front is the construction centre of mass, whose
  // translation rounds at the ULP of the void-scale affine (metres at
  // 4e16 m) — far below the 2.6e10 m own-time signal this test discriminates.
  auto const& front = pile_up.trajectory().front();
  auto const& back = pile_up.trajectory().back();
  EXPECT_THAT((front.degrees_of_freedom.position() -
               (front_before - b_from_a(t0))).Norm(),
              Lt(1 * Kilo(Metre)));
  EXPECT_EQ(at_x(3.8e16 * Metre).position() - b_from_a(t1),
            back.degrees_of_freedom.position());
  EXPECT_EQ(v - v_rel, back.degrees_of_freedom.velocity());
}

// The uniform-representation invariant and the anchor-folding rebase.  The
// policy is regime-blind — none of this depends on whether the far field
// vanishes: coordinates beyond the re-anchor bound adopt an anchor in a
// star's domain exactly as in the void (unanchored subsystem coordinates of
// ~1e15 m quantize part geometry at 0.125 m — owner-visible gaps), and a
// dominance retag of an anchored pile-up folds the subsystem conversion into
// the anchor, leaving every represented point bit for bit intact where the
// old drop-and-translate would have rounded each point at the ULP of the
// inter-subsystem distance.
TEST_F(PileUpTest, AnchoredRebaseFoldsConversionIntoAnchor) {
  Length const saved_bound = PileUp::re_anchor_bound_for_testing_;
  absl::Cleanup restore_bound = [saved_bound] {
    PileUp::re_anchor_bound_for_testing_ = saved_bound;
  };

  Length const separation = 4e16 * Metre;
  GravitationalParameter const μ_b =
      1.3e20 * Pow<3>(Metre) / Pow<2>(Second);
  GravitationalParameter const μ_a = 16 * μ_b;
  Displacement<Barycentric> const b_from_a_at_t0(
      {separation, 0 * Metre, 0 * Metre});
  Velocity<Barycentric> const v_rel({300 * Kilo(Metre) / Second,
                                     0 * Metre / Second,
                                     0 * Metre / Second});
  Instant const t0 = J2000;
  auto const b_from_a = [&](Instant const& t) {
    return b_from_a_at_t0 + v_rel * (t - t0);
  };

  MockEphemeris<Barycentric> ephemeris;
  EXPECT_CALL(ephemeris, number_of_subsystems())
      .WillRepeatedly(Return(2));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(0))
      .WillRepeatedly(ReturnRef(μ_a));
  EXPECT_CALL(ephemeris, subsystem_gravitational_parameter(1))
      .WillRepeatedly(ReturnRef(μ_b));
  EXPECT_CALL(ephemeris, subsystem_barycentre(_, _))
      .WillRepeatedly(Return(Barycentric::origin));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 0, _))
      .WillRepeatedly(Return(Displacement<Barycentric>{}));
  EXPECT_CALL(ephemeris, subsystem_conversion(1, 1, _))
      .WillRepeatedly(Return(Displacement<Barycentric>{}));
  EXPECT_CALL(ephemeris, subsystem_conversion(0, 1, _))
      .WillRepeatedly(Invoke([&](int, int, Instant const& t) {
        return -b_from_a(t);
      }));
  EXPECT_CALL(ephemeris, subsystem_conversion(1, 0, _))
      .WillRepeatedly(Invoke([&](int, int, Instant const& t) {
        return b_from_a(t);
      }));
  EXPECT_CALL(ephemeris, subsystem_velocity_conversion(0, 1))
      .WillRepeatedly(Return(-v_rel));

  Velocity<Barycentric> const v(
      {1 * Metre / Second, 0 * Metre / Second, 0 * Metre / Second});
  auto const at_x = [&](Length const& x) {
    return DegreesOfFreedom<Barycentric>(
        Barycentric::origin +
            Displacement<Barycentric>({x, 0 * Metre, 0 * Metre}),
        v);
  };

  TestablePileUp pile_up({&p1_}, t0 - 1 * Second,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         /*deletion_callback=*/nullptr);

  // Act 1: coordinates beyond the bound adopt an anchor — B raw-dominates
  // here but within the hysteresis margin, so this is pure representation,
  // no retag — and the true position survives the collapse bit for bit.
  pile_up.AppendToTrajectory(t0, at_x(3.45e16 * Metre));
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(0, pile_up.subsystem());
  ASSERT_TRUE(pile_up.anchor().has_value());
  auto const anchor0 = *pile_up.anchor();
  EXPECT_EQ(at_x(3.45e16 * Metre).position(),
            pile_up.trajectory().back().degrees_of_freedom.position() +
                anchor0.OffsetAt(t0));
  EXPECT_EQ(Velocity<Barycentric>(),
            pile_up.trajectory().back().degrees_of_freedom.velocity() +
                anchor0.velocity - v);

  // Act 2: a point where B dominates beyond the margin.  Raise the bound so
  // that the retag is observed in isolation from re-anchoring.
  PileUp::re_anchor_bound_for_testing_ = 1e20 * Metre;
  Instant const t1 = t0 + 1 * Day;
  DegreesOfFreedom<Barycentric> const represented_at_t1(
      at_x(3.8e16 * Metre).position() - anchor0.OffsetAt(t1),
      v - anchor0.velocity);
  pile_up.AppendToTrajectory(t1, represented_at_t1);
  auto const front_before =
      pile_up.trajectory().front().degrees_of_freedom;
  auto const back_before = pile_up.trajectory().back().degrees_of_freedom;
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  EXPECT_EQ(1, pile_up.subsystem());
  ASSERT_TRUE(pile_up.anchor().has_value());
  auto const anchor1 = *pile_up.anchor();

  // The money assertion: the represented points are bit-for-bit intact.
  EXPECT_EQ(front_before.position(),
            pile_up.trajectory().front().degrees_of_freedom.position());
  EXPECT_EQ(front_before.velocity(),
            pile_up.trajectory().front().degrees_of_freedom.velocity());
  EXPECT_EQ(back_before.position(),
            pile_up.trajectory().back().degrees_of_freedom.position());
  EXPECT_EQ(back_before.velocity(),
            pile_up.trajectory().back().degrees_of_freedom.velocity());

  // The conversion was folded into the anchor: at two distinct epochs — which
  // pins both the intercept and the slope of the affine fold — the new
  // anchor differs from the old one by the (moving) conversion, up to the
  // rounding of the fold at the ULP of the separation.
  EXPECT_EQ(anchor0.velocity - v_rel, anchor1.velocity);
  EXPECT_THAT((anchor1.OffsetAt(t0) -
               (anchor0.OffsetAt(t0) - b_from_a(t0))).Norm(),
              Lt(32 * Metre));
  EXPECT_THAT((anchor1.OffsetAt(t1) -
               (anchor0.OffsetAt(t1) - b_from_a(t1))).Norm(),
              Lt(32 * Metre));

  // Act 3: back at the production bound, in-domain coordinate growth renews
  // the anchor; the true position moves by at most the documented sub-mm
  // fold residual.
  PileUp::re_anchor_bound_for_testing_ = saved_bound;
  Instant const t2 = t1 + 1 * Second;
  DegreesOfFreedom<Barycentric> const grown(
      pile_up.trajectory().back().degrees_of_freedom.position() +
          Displacement<Barycentric>({2e12 * Metre, 0 * Metre, 0 * Metre}),
      pile_up.trajectory().back().degrees_of_freedom.velocity());
  Position<Barycentric> const true_before =
      grown.position() + anchor1.OffsetAt(t2);
  pile_up.AppendToTrajectory(t2, grown);
  EXPECT_FALSE(pile_up.RebaseIfNeeded().empty());
  ASSERT_TRUE(pile_up.anchor().has_value());
  EXPECT_NE(anchor1, *pile_up.anchor());
  Position<Barycentric> const true_after =
      pile_up.trajectory().back().degrees_of_freedom.position() +
      pile_up.anchor()->OffsetAt(t2);
  EXPECT_THAT((true_after - true_before).Norm(), Lt(1 * Milli(Metre)));
}

// The construction reconcile that replaces the plugin's pre-collect merge
// unification: parts docking from distinct subsystems are voted onto the
// heaviest subsystem (mass beats index, ties break toward the smallest index)
// and the divergent parts are re-expressed there through `placement_conversion`
// while their geometry is preserved; an all-agree pile-up takes the
// byte-identical early-out.
TEST_F(PileUpTest, PileUpConstructionReconcilesDivergentPlacements) {
  // p1_ (1 kg, subsystem 0, unanchored) docks with the heavier p2_ (2 kg,
  // subsystem 1, anchored): the vote adopts subsystem 1 — the higher index —
  // and p2_'s anchor, and only the divergent p1_ is converted.
  MockEphemeris<Barycentric> ephemeris;
  Ephemeris<Barycentric>::Anchor const anchor{
      .offset = SectorDisplacement<Barycentric>::Split(
          Displacement<Barycentric>({1e15 * Metre, 0 * Metre, 0 * Metre})),
      .velocity = Velocity<Barycentric>(),
      .epoch = J2000};
  p2_.set_placement({1, anchor});

  // The known conversion the vote applies to p1_, and the rigid motion it must
  // produce (built exactly as the constructor does).
  Displacement<Barycentric> const conversion_displacement(
      {7 * Metre, -5 * Metre, 3 * Metre});
  Velocity<Barycentric> const conversion_velocity(
      {2 * Metre / Second, 4 * Metre / Second, -6 * Metre / Second});
  auto const from_p1 =
      [](Ephemeris<Barycentric>::SubsystemPlacement const& placement) {
        return placement.subsystem == 0 && !placement.anchor.has_value();
      };
  auto const to_target =
      [&anchor](Ephemeris<Barycentric>::SubsystemPlacement const& placement) {
        return placement.subsystem == 1 && placement.anchor == anchor;
      };
  EXPECT_CALL(ephemeris,
              placement_conversion(Truly(from_p1), Truly(to_target), J2000))
      .WillOnce(Return(std::make_pair(conversion_displacement,
                                      conversion_velocity)));
  RigidMotion<Barycentric, Barycentric> const conversion_motion(
      RigidTransformation<Barycentric, Barycentric>(
          Barycentric::origin,
          Barycentric::origin + conversion_displacement,
          OrthogonalMap<Barycentric, Barycentric>::Identity()),
      Barycentric::nonrotating,
      -conversion_velocity);
  auto const p1_rigid_motion_before = p1_.rigid_motion();
  auto const p2_rigid_motion_before = p2_.rigid_motion();
  auto const p1_expected =
      (conversion_motion * p1_rigid_motion_before)(
          {RigidPart::origin, RigidPart::unmoving});

  TestablePileUp pile_up({&p1_, &p2_}, J2000,
                         DefaultPsychohistoryParameters(),
                         DefaultHistoryParameters(),
                         &ephemeris,
                         /*deletion_callback=*/nullptr);

  // The heavier subsystem wins the vote and its part's anchor is adopted.
  EXPECT_EQ(1, pile_up.subsystem());
  ASSERT_TRUE(pile_up.anchor().has_value());
  EXPECT_EQ(anchor, *pile_up.anchor());

  // p1_ is retagged and its rigid motion converted; p2_ is left untouched.
  EXPECT_EQ(1, p1_.placement().subsystem);
  ASSERT_TRUE(p1_.placement().anchor.has_value());
  EXPECT_EQ(anchor, *p1_.placement().anchor);
  auto const p1_actual =
      p1_.rigid_motion()({RigidPart::origin, RigidPart::unmoving});
  EXPECT_EQ(p1_expected.position(), p1_actual.position());
  EXPECT_EQ(p1_expected.velocity(), p1_actual.velocity());
  auto const p2_actual =
      p2_.rigid_motion()({RigidPart::origin, RigidPart::unmoving});
  EXPECT_EQ(p2_rigid_motion_before(
                {RigidPart::origin, RigidPart::unmoving}).position(),
            p2_actual.position());
  EXPECT_EQ(p2_rigid_motion_before(
                {RigidPart::origin, RigidPart::unmoving}).velocity(),
            p2_actual.velocity());

  // The conversion is a rigid motion, so the parts' relative geometry is
  // preserved: their separation in the pile-up frame equals the separation of
  // their converted Barycentric positions.
  EXPECT_EQ(2u, pile_up.actual_part_rigid_motion().size());
  Length const separation_barycentric =
      (p1_expected.position() - p2_actual.position()).Norm();
  Length const separation_pile_up =
      (pile_up.actual_part_rigid_motion().at(&p1_)(
           {RigidPart::origin, RigidPart::unmoving}).position() -
       pile_up.actual_part_rigid_motion().at(&p2_)(
           {RigidPart::origin, RigidPart::unmoving}).position()).Norm();
  EXPECT_THAT(separation_pile_up, AlmostEquals(separation_barycentric, 0, 8));

  // A tie in the mass vote breaks toward the smallest subsystem index.
  MockEphemeris<Barycentric> tie_ephemeris;
  auto const anchorless =
      [](int const subsystem) {
        return [subsystem](
            Ephemeris<Barycentric>::SubsystemPlacement const& placement) {
          return placement.subsystem == subsystem &&
                 !placement.anchor.has_value();
        };
      };
  EXPECT_CALL(tie_ephemeris,
              placement_conversion(
                  Truly(anchorless(1)), Truly(anchorless(0)), _))
      .WillOnce(Return(std::make_pair(Displacement<Barycentric>{},
                                      Velocity<Barycentric>{})));
  Part pa(121, "pa", mass1_, EccentricPart::origin, inertia_tensor1_,
          RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
              p1_dof_),
          /*deletion_callback=*/nullptr);
  Part pb(122, "pb", mass1_, EccentricPart::origin, inertia_tensor1_,
          RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
              p2_dof_),
          /*deletion_callback=*/nullptr);
  pb.set_placement({1, std::nullopt});
  TestablePileUp tie_pile_up({&pa, &pb}, J2000,
                             DefaultPsychohistoryParameters(),
                             DefaultHistoryParameters(),
                             &tie_ephemeris,
                             /*deletion_callback=*/nullptr);
  EXPECT_EQ(0, tie_pile_up.subsystem());

  // The all-agree path is a byte-identical early-out: no conversion is
  // requested and the parts' rigid motions are untouched.
  MockEphemeris<Barycentric> agree_ephemeris;
  EXPECT_CALL(agree_ephemeris, placement_conversion(_, _, _)).Times(0);
  Part pc(123, "pc", mass1_, EccentricPart::origin, inertia_tensor1_,
          RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
              p1_dof_),
          /*deletion_callback=*/nullptr);
  Part pd(124, "pd", mass2_, EccentricPart::origin, inertia_tensor2_,
          RigidMotion<EccentricPart, Barycentric>::MakeNonRotatingMotion(
              p2_dof_),
          /*deletion_callback=*/nullptr);
  auto const pc_before =
      pc.rigid_motion()({RigidPart::origin, RigidPart::unmoving});
  auto const pd_before =
      pd.rigid_motion()({RigidPart::origin, RigidPart::unmoving});
  TestablePileUp agree_pile_up({&pc, &pd}, J2000,
                               DefaultPsychohistoryParameters(),
                               DefaultHistoryParameters(),
                               &agree_ephemeris,
                               /*deletion_callback=*/nullptr);
  EXPECT_EQ(0, agree_pile_up.subsystem());
  EXPECT_FALSE(agree_pile_up.anchor().has_value());
  EXPECT_EQ(pc_before.position(),
            pc.rigid_motion()(
                {RigidPart::origin, RigidPart::unmoving}).position());
  EXPECT_EQ(pc_before.velocity(),
            pc.rigid_motion()(
                {RigidPart::origin, RigidPart::unmoving}).velocity());
  EXPECT_EQ(pd_before.position(),
            pd.rigid_motion()(
                {RigidPart::origin, RigidPart::unmoving}).position());
  EXPECT_EQ(pd_before.velocity(),
            pd.rigid_motion()(
                {RigidPart::origin, RigidPart::unmoving}).velocity());
}

}  // namespace ksp_plugin
}  // namespace principia
