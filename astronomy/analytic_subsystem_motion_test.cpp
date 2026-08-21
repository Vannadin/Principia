#include "physics/analytic_subsystem_motion.hpp"

#include <memory>
#include <optional>
#include <string>
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
#include "physics/extrapolated_trajectory.hpp"
#include "physics/kepler_orbit.hpp"
#include "physics/massive_body.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/matchers.hpp"  // 🧙 For EXPECT_OK.

namespace principia {
namespace astronomy {

using namespace principia::astronomy::_frames;
using namespace principia::base::_not_null;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_analytic_subsystem_motion;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_extrapolated_trajectory;
using namespace principia::physics::_kepler_orbit;
using namespace principia::physics::_massive_body;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;

// A/B tests of the analytic extrapolation of celestial motion against real
// integrations of the same systems: the model must track each tier of the
// hierarchy — lone stars, binary pairs, bodies orbiting a pair — within the
// budgets of the void-line rendering design, over spans of centuries.
class AnalyticSubsystemMotionTest : public ::testing::Test {
 protected:
  using Model = AnalyticSubsystemMotion<ICRS>;

  not_null<std::unique_ptr<Ephemeris<ICRS>>> MakeEphemeris(
      std::vector<GravitationalParameter> const& gravitational_parameters,
      std::vector<DegreesOfFreedom<ICRS>> const& initial_state,
      Time const& step) {
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    for (int i = 0; i < static_cast<int>(gravitational_parameters.size());
         ++i) {
      bodies.push_back(make_not_null_unique<MassiveBody>(
          MassiveBody::Parameters("body " + std::to_string(i),
                                  gravitational_parameters[i])));
    }
    return make_not_null_unique<Ephemeris<ICRS>>(
        std::move(bodies),
        initial_state,
        t0_,
        Ephemeris<ICRS>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<ICRS>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<ICRS>::NewtonianMotionEquation>(),
            step));
  }

  std::vector<Model::Member> MembersAt(
      Ephemeris<ICRS> const& ephemeris,
      Instant const& t,
      std::vector<std::optional<int>> const& parents) {
    std::vector<Model::Member> members;
    auto const& bodies = ephemeris.bodies();
    for (int i = 0; i < static_cast<int>(bodies.size()); ++i) {
      members.push_back(
          Model::Member{bodies[i]->gravitational_parameter(),
                        ephemeris.trajectory(bodies[i])
                            ->EvaluateDegreesOfFreedom(t),
                        parents[i]});
    }
    return members;
  }

  // Two clusters separated by `separation` along x, with relative velocity
  // `relative_speed` along y, the total momentum zero, and the barycentre at
  // the origin.
  std::pair<DegreesOfFreedom<ICRS>, DegreesOfFreedom<ICRS>> BarycentricPair(
      GravitationalParameter const& μ1,
      GravitationalParameter const& μ2,
      Length const& separation,
      Speed const& relative_speed) {
    Displacement<ICRS> const r(
        {separation, 0 * Metre, 0 * Metre});
    Velocity<ICRS> const v(
        {0 * Metre / Second, relative_speed, 0 * Metre / Second});
    double const f1 = μ2 / (μ1 + μ2);
    double const f2 = μ1 / (μ1 + μ2);
    return {DegreesOfFreedom<ICRS>(ICRS::origin - f1 * r, -f1 * v),
            DegreesOfFreedom<ICRS>(ICRS::origin + f2 * r, f2 * v)};
  }

  Instant const t0_;
};

TEST_F(AnalyticSubsystemMotionTest, SingleMemberIsAffine) {
  DegreesOfFreedom<ICRS> const initial(
      ICRS::origin + Displacement<ICRS>({100 * AstronomicalUnit,
                                         -40 * AstronomicalUnit,
                                         7 * AstronomicalUnit}),
      Velocity<ICRS>({30 * Kilo(Metre) / Second,
                      -5 * Kilo(Metre) / Second,
                      1 * Kilo(Metre) / Second}));
  std::vector<Model::Member> const members = {
      {SolarGravitationalParameter, initial, std::nullopt}};
  Model const model(members, t0_);
  Instant const t = t0_ + 1000 * JulianYear;
  DegreesOfFreedom<ICRS> const actual = model.EvaluateDegreesOfFreedom(0, t);
  EXPECT_LT((actual.position() -
             (initial.position() + initial.velocity() * (t - t0_))).Norm(),
            1 * Metre);
  EXPECT_LT((actual.velocity() - initial.velocity()).Norm(),
            1e-9 * Metre / Second);
}

TEST_F(AnalyticSubsystemMotionTest, UnboundPairIsLinear) {
  GravitationalParameter const μ = SolarGravitationalParameter;
  Length const r = 100 * AstronomicalUnit;
  Speed const v = 3 * Sqrt(2 * (μ + μ) / r);
  auto const [dof1, dof2] = BarycentricPair(μ, μ, r, v);
  std::vector<Model::Member> const members = {{μ, dof1, std::nullopt},
                                              {μ, dof2, 0}};
  Model const model(members, t0_);
  EXPECT_FALSE(model.JacobiVectorIsKeplerian(1));
  Instant const t = t0_ + 1000 * JulianYear;
  for (int i = 0; i < 2; ++i) {
    DegreesOfFreedom<ICRS> const actual =
        model.EvaluateDegreesOfFreedom(i, t);
    DegreesOfFreedom<ICRS> const expected(
        members[i].degrees_of_freedom.position() +
            members[i].degrees_of_freedom.velocity() * (t - t0_),
        members[i].degrees_of_freedom.velocity());
    EXPECT_LT((actual.position() - expected.position()).Norm(), 5 * Metre);
    EXPECT_LT((actual.velocity() - expected.velocity()).Norm(),
              1e-6 * Metre / Second);
  }
}

// The defect named by the review of the void-line rendering design: a model
// anchored on the star's centre freezes the planetary-reflex velocity into
// the drift, and moves off by whole astronomical units over centuries; a
// model anchored on the subsystem barycentre tracks the star.
TEST_F(AnalyticSubsystemMotionTest, StarWithPlanetReflex) {
  GravitationalParameter const μ_star = SolarGravitationalParameter;
  GravitationalParameter const μ_planet = 1e-3 * SolarGravitationalParameter;
  Length const a = 5 * AstronomicalUnit;
  Speed const v = Sqrt((μ_star + μ_planet) / a);
  Time const period = 2 * π * Sqrt(Pow<3>(a) / (μ_star + μ_planet));
  auto const [star_dof, planet_dof] =
      BarycentricPair(μ_star, μ_planet, a, v);
  auto const ephemeris = MakeEphemeris({μ_star, μ_planet},
                                       {star_dof, planet_dof},
                                       period / 300);
  Instant const t_capture = t0_ + 10 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_capture));
  auto const members = MembersAt(*ephemeris, t_capture, {std::nullopt, 0});
  Model const model(members, t_capture);
  std::vector<Model::Member> const star_only = {members[0]};
  Model const star_anchored(star_only, t_capture);
  Instant const t_final = t_capture + 500 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  Position<ICRS> const truth =
      ephemeris->trajectory(ephemeris->bodies()[0])
          ->EvaluatePosition(t_final);
  Length const model_error =
      (model.EvaluateDegreesOfFreedom(0, t_final).position() - truth).Norm();
  Length const star_anchored_error =
      (star_anchored.EvaluateDegreesOfFreedom(0, t_final).position() - truth)
          .Norm();
  LOG(INFO) << "barycentre-anchored: " << model_error / AstronomicalUnit
            << " au; star-anchored: "
            << star_anchored_error / AstronomicalUnit << " au";
  EXPECT_LT(model_error, 0.02 * AstronomicalUnit);
  EXPECT_GT(star_anchored_error, 0.4 * AstronomicalUnit);
  EXPECT_GT(star_anchored_error, 20 * model_error);
}

TEST_F(AnalyticSubsystemMotionTest, BinaryPair) {
  GravitationalParameter const μ1 = 1.1 * SolarGravitationalParameter;
  GravitationalParameter const μ2 = 0.9 * SolarGravitationalParameter;
  Length const a = 23.5 * AstronomicalUnit;
  double const e = 0.52;
  Length const r_apoapsis = a * (1 + e);
  Speed const v_apoapsis = Sqrt((μ1 + μ2) * (2 / r_apoapsis - 1 / a));
  Time const period = 2 * π * Sqrt(Pow<3>(a) / (μ1 + μ2));
  auto const [dof1, dof2] = BarycentricPair(μ1, μ2, r_apoapsis, v_apoapsis);
  auto const ephemeris = MakeEphemeris({μ1, μ2},
                                       {dof1, dof2},
                                       period / 1000);
  Instant const t_capture = t0_ + 5 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_capture));
  auto const members = MembersAt(*ephemeris, t_capture, {std::nullopt, 0});
  Model const model(members, t_capture);
  EXPECT_TRUE(model.JacobiVectorIsKeplerian(1));
  Instant const t_final = t_capture + 500 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  for (int i = 0; i < 2; ++i) {
    Length const error =
        (model.EvaluateDegreesOfFreedom(i, t_final).position() -
         ephemeris->trajectory(ephemeris->bodies()[i])
             ->EvaluatePosition(t_final)).Norm();
    LOG(INFO) << "component " << i << ": " << error / AstronomicalUnit
              << " au";
    EXPECT_LT(error, 1e-3 * a);
  }
}

// The epoch of the fit is the phase at which the elements are worst
// conditioned.
TEST_F(AnalyticSubsystemMotionTest, EccentricPairAnchoredAtPeriapsis) {
  GravitationalParameter const μ1 = 1.1 * SolarGravitationalParameter;
  GravitationalParameter const μ2 = 0.9 * SolarGravitationalParameter;
  Length const a = 20 * AstronomicalUnit;
  double const e = 0.92;
  Length const r_periapsis = a * (1 - e);
  Speed const v_periapsis = Sqrt((μ1 + μ2) * (2 / r_periapsis - 1 / a));
  Time const period = 2 * π * Sqrt(Pow<3>(a) / (μ1 + μ2));
  auto const [dof1, dof2] =
      BarycentricPair(μ1, μ2, r_periapsis, v_periapsis);
  auto const ephemeris = MakeEphemeris({μ1, μ2},
                                       {dof1, dof2},
                                       period / 20'000);
  Instant const t_final = t0_ + 500 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  auto const members = MembersAt(*ephemeris, t0_, {std::nullopt, 0});
  Model const model(members, t0_);
  EXPECT_TRUE(model.JacobiVectorIsKeplerian(1));
  for (int i = 0; i < 2; ++i) {
    Length const error =
        (model.EvaluateDegreesOfFreedom(i, t_final).position() -
         ephemeris->trajectory(ephemeris->bodies()[i])
             ->EvaluatePosition(t_final)).Norm();
    LOG(INFO) << "component " << i << ": " << error / AstronomicalUnit
              << " au";
    EXPECT_LT(error, 1e-3 * a);
  }
}

// A body around a pair is propagated about the barycentre of the pair —
// never about either component, the other defect named by the review.
TEST_F(AnalyticSubsystemMotionTest, CircumbinaryBody) {
  GravitationalParameter const μ_a = SolarGravitationalParameter;
  GravitationalParameter const μ_b = 0.8 * SolarGravitationalParameter;
  GravitationalParameter const μ_c = 0.5 * SolarGravitationalParameter;
  Length const a_inner = 2 * AstronomicalUnit;
  Length const r_outer = 200 * AstronomicalUnit;
  Speed const v_inner = Sqrt((μ_a + μ_b) / a_inner);
  Speed const v_outer = Sqrt((μ_a + μ_b + μ_c) / r_outer);
  Time const period_inner = 2 * π * Sqrt(Pow<3>(a_inner) / (μ_a + μ_b));
  auto [dof_a, dof_b] = BarycentricPair(μ_a, μ_b, a_inner, v_inner);
  Displacement<ICRS> const r_outer_vector(
      {0 * Metre, r_outer, 0 * Metre});
  Velocity<ICRS> const v_outer_vector(
      {-v_outer, 0 * Metre / Second, 0 * Metre / Second});
  double const f_inner = μ_c / (μ_a + μ_b + μ_c);
  double const f_outer = (μ_a + μ_b) / (μ_a + μ_b + μ_c);
  dof_a = {dof_a.position() - f_inner * r_outer_vector,
           dof_a.velocity() - f_inner * v_outer_vector};
  dof_b = {dof_b.position() - f_inner * r_outer_vector,
           dof_b.velocity() - f_inner * v_outer_vector};
  DegreesOfFreedom<ICRS> const dof_c(
      ICRS::origin + f_outer * r_outer_vector,
      f_outer * v_outer_vector);
  auto const ephemeris = MakeEphemeris({μ_a, μ_b, μ_c},
                                       {dof_a, dof_b, dof_c},
                                       period_inner / 300);
  Instant const t_capture = t0_ + 5 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_capture));
  auto const members =
      MembersAt(*ephemeris, t_capture, {std::nullopt, 0, 0});
  Model const model(members, t_capture);
  EXPECT_TRUE(model.JacobiVectorIsKeplerian(1));
  EXPECT_TRUE(model.JacobiVectorIsKeplerian(2));
  Instant const t_final = t_capture + 500 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  Length const error =
      (model.EvaluateDegreesOfFreedom(2, t_final).position() -
       ephemeris->trajectory(ephemeris->bodies()[2])
           ->EvaluatePosition(t_final)).Norm();
  LOG(INFO) << "circumbinary: " << error / AstronomicalUnit << " au";
  EXPECT_LT(error, 1e-4 * r_outer);
}

// Past a few dozen orbits beyond the epoch the phase of a planet is not a
// prediction, so the model promises the shape of the orbit, never the
// position on it: only the elements are asserted.
TEST_F(AnalyticSubsystemMotionTest, PlanetElementsHold) {
  GravitationalParameter const μ_star = SolarGravitationalParameter;
  GravitationalParameter const μ_inner = 3e-6 * SolarGravitationalParameter;
  GravitationalParameter const μ_outer = 1e-3 * SolarGravitationalParameter;
  Length const a_inner = 2 * AstronomicalUnit;
  Length const a_outer = 10 * AstronomicalUnit;
  Speed const v_inner = Sqrt((μ_star + μ_inner) / a_inner);
  Speed const v_outer = Sqrt((μ_star + μ_inner + μ_outer) / a_outer);
  Time const period_inner = 2 * π * Sqrt(Pow<3>(a_inner) / (μ_star + μ_inner));
  auto [dof_star, dof_inner] =
      BarycentricPair(μ_star, μ_inner, a_inner, v_inner);
  Displacement<ICRS> const r_outer_vector(
      {0 * Metre, a_outer, 0 * Metre});
  Velocity<ICRS> const v_outer_vector(
      {-v_outer, 0 * Metre / Second, 0 * Metre / Second});
  double const f_inner = μ_outer / (μ_star + μ_inner + μ_outer);
  double const f_outer = (μ_star + μ_inner) / (μ_star + μ_inner + μ_outer);
  dof_star = {dof_star.position() - f_inner * r_outer_vector,
              dof_star.velocity() - f_inner * v_outer_vector};
  dof_inner = {dof_inner.position() - f_inner * r_outer_vector,
               dof_inner.velocity() - f_inner * v_outer_vector};
  DegreesOfFreedom<ICRS> const dof_outer(
      ICRS::origin + f_outer * r_outer_vector,
      f_outer * v_outer_vector);
  auto const ephemeris = MakeEphemeris({μ_star, μ_inner, μ_outer},
                                       {dof_star, dof_inner, dof_outer},
                                       period_inner / 300);
  Instant const t_capture = t0_ + 5 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_capture));
  auto const members =
      MembersAt(*ephemeris, t_capture, {std::nullopt, 0, 0});
  Model const model(members, t_capture);
  Instant const t_final = t_capture + 500 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_final));
  RelativeDegreesOfFreedom<ICRS> const model_relative =
      model.EvaluateDegreesOfFreedom(1, t_final) -
      model.EvaluateDegreesOfFreedom(0, t_final);
  RelativeDegreesOfFreedom<ICRS> const true_relative =
      ephemeris->trajectory(ephemeris->bodies()[1])
          ->EvaluateDegreesOfFreedom(t_final) -
      ephemeris->trajectory(ephemeris->bodies()[0])
          ->EvaluateDegreesOfFreedom(t_final);
  MassiveBody const star{MassiveBody::Parameters{μ_star}};
  MassiveBody const planet{MassiveBody::Parameters{μ_inner}};
  auto const& model_elements =
      KeplerOrbit<ICRS>(star, planet, model_relative, t_final)
          .elements_at_epoch();
  auto const& true_elements =
      KeplerOrbit<ICRS>(star, planet, true_relative, t_final)
          .elements_at_epoch();
  EXPECT_LT(Abs(*model_elements.semimajor_axis -
                *true_elements.semimajor_axis) /
                *true_elements.semimajor_axis,
            1e-3);
  EXPECT_LT(std::abs(*model_elements.eccentricity -
                     *true_elements.eccentricity),
            0.02);
  EXPECT_LT(AngleBetween(Wedge(model_relative.displacement(),
                               model_relative.velocity()),
                         Wedge(true_relative.displacement(),
                               true_relative.velocity())),
            1 * Degree);
}

// One ULP inside the parabola the elliptic elements are garbage and
// `KeplerOrbit` is singular, so a near-parabolic pair falls back to the
// linear tier even though it is bound.
TEST_F(AnalyticSubsystemMotionTest, NearParabolicPairFallsBackToLinear) {
  GravitationalParameter const μ = SolarGravitationalParameter;
  Length const r = 50 * AstronomicalUnit;
  Speed const v = Sqrt(2 * (μ + μ) / r) * (1 - 1e-9);
  auto const [dof1, dof2] = BarycentricPair(μ, μ, r, v);
  std::vector<Model::Member> const members = {{μ, dof1, std::nullopt},
                                              {μ, dof2, 0}};
  Model const model(members, t0_);
  EXPECT_FALSE(model.JacobiVectorIsKeplerian(1));
  Instant const t = t0_ + 100 * JulianYear;
  DegreesOfFreedom<ICRS> const actual = model.EvaluateDegreesOfFreedom(1, t);
  DegreesOfFreedom<ICRS> const expected(
      dof2.position() + dof2.velocity() * (t - t0_),
      dof2.velocity());
  EXPECT_LT((actual.position() - expected.position()).Norm(), 5 * Metre);
}

// A pair whose separation is a few percent of its semimajor axis is
// borderline — the sign of its energy is noise — so its tier follows the
// previous model instead of flipping, in both directions.
TEST_F(AnalyticSubsystemMotionTest, BorderlineTierIsSticky) {
  GravitationalParameter const μ = SolarGravitationalParameter;
  Length const r = 1 * AstronomicalUnit;
  Length const a = 40 * AstronomicalUnit;
  Speed const v_bound = Sqrt((μ + μ) * (2 / r - 1 / a));
  Speed const v_unbound = Sqrt(2 * (μ + μ) / r) * (1 + 1e-3);
  auto const [bound1, bound2] = BarycentricPair(μ, μ, r, v_bound);
  auto const [unbound1, unbound2] = BarycentricPair(μ, μ, r, v_unbound);
  std::vector<Model::Member> const bound = {{μ, bound1, std::nullopt},
                                            {μ, bound2, 0}};
  std::vector<Model::Member> const unbound = {{μ, unbound1, std::nullopt},
                                              {μ, unbound2, 0}};
  Model const keplerian(bound, t0_);
  EXPECT_TRUE(keplerian.JacobiVectorIsKeplerian(1));
  Model const sticky_keplerian(unbound, t0_, &keplerian);
  EXPECT_TRUE(sticky_keplerian.JacobiVectorIsKeplerian(1));
  Model const linear(unbound, t0_);
  EXPECT_FALSE(linear.JacobiVectorIsKeplerian(1));
  Model const sticky_linear(bound, t0_, &linear);
  EXPECT_FALSE(sticky_linear.JacobiVectorIsKeplerian(1));
}

// Advancing the capture time by one prolongation step must not move the far
// end of the extrapolation.
TEST_F(AnalyticSubsystemMotionTest, SeamStability) {
  GravitationalParameter const μ1 = 1.1 * SolarGravitationalParameter;
  GravitationalParameter const μ2 = 0.9 * SolarGravitationalParameter;
  Length const a = 23.5 * AstronomicalUnit;
  double const e = 0.52;
  Length const r_apoapsis = a * (1 + e);
  Speed const v_apoapsis = Sqrt((μ1 + μ2) * (2 / r_apoapsis - 1 / a));
  Time const period = 2 * π * Sqrt(Pow<3>(a) / (μ1 + μ2));
  auto const [dof1, dof2] = BarycentricPair(μ1, μ2, r_apoapsis, v_apoapsis);
  auto const ephemeris = MakeEphemeris({μ1, μ2},
                                       {dof1, dof2},
                                       period / 1000);
  Instant const t_before = t0_ + 5 * JulianYear;
  Instant const t_after = t_before + JulianYear / 2;
  EXPECT_OK(ephemeris->Prolong(t_after));
  Model const before(MembersAt(*ephemeris, t_before, {std::nullopt, 0}),
                     t_before);
  Model const after(MembersAt(*ephemeris, t_after, {std::nullopt, 0}),
                    t_after,
                    &before);
  Instant const t_final = t_before + 500 * JulianYear;
  for (int i = 0; i < 2; ++i) {
    Length const seam_shift =
        (before.EvaluateDegreesOfFreedom(i, t_final).position() -
         after.EvaluateDegreesOfFreedom(i, t_final).position()).Norm();
    LOG(INFO) << "component " << i << ": " << seam_shift / AstronomicalUnit
              << " au";
    EXPECT_LT(seam_shift, 1e-3 * a);
  }
}

TEST_F(AnalyticSubsystemMotionTest, ExtrapolatedTrajectoryView) {
  GravitationalParameter const μ_star = SolarGravitationalParameter;
  GravitationalParameter const μ_planet = 1e-3 * SolarGravitationalParameter;
  Length const a = 5 * AstronomicalUnit;
  Speed const v = Sqrt((μ_star + μ_planet) / a);
  Time const period = 2 * π * Sqrt(Pow<3>(a) / (μ_star + μ_planet));
  auto const [star_dof, planet_dof] =
      BarycentricPair(μ_star, μ_planet, a, v);
  auto const ephemeris = MakeEphemeris({μ_star, μ_planet},
                                       {star_dof, planet_dof},
                                       period / 300);
  Instant const t_horizon = t0_ + 10 * JulianYear;
  EXPECT_OK(ephemeris->Prolong(t_horizon));
  Model const model(MembersAt(*ephemeris, t_horizon, {std::nullopt, 0}),
                    t_horizon);
  auto const& star_trajectory =
      *ephemeris->trajectory(ephemeris->bodies()[0]);
  ExtrapolatedTrajectory<ICRS> const view(
      star_trajectory, t_horizon, model, /*member=*/0);
  EXPECT_EQ(view.t_min(), star_trajectory.t_min());
  EXPECT_EQ(view.t_max(), InfiniteFuture);
  Instant const t_below = t_horizon - JulianYear;
  EXPECT_EQ(view.EvaluateDegreesOfFreedom(t_below),
            star_trajectory.EvaluateDegreesOfFreedom(t_below));
  EXPECT_EQ(view.EvaluatePosition(t_below),
            star_trajectory.EvaluatePosition(t_below));
  Instant const t_beyond = t_horizon + 100 * JulianYear;
  EXPECT_EQ(view.EvaluateDegreesOfFreedom(t_beyond),
            model.EvaluateDegreesOfFreedom(0, t_beyond));
  EXPECT_EQ(view.EvaluateVelocity(t_beyond),
            model.EvaluateDegreesOfFreedom(0, t_beyond).velocity());
  // The horizon was captured at construction: even where the trajectory has
  // since been prolonged, the view keeps evaluating the model.
  EXPECT_OK(ephemeris->Prolong(t_horizon + 2 * JulianYear));
  Instant const t_prolonged = t_horizon + JulianYear;
  EXPECT_EQ(view.EvaluateDegreesOfFreedom(t_prolonged),
            model.EvaluateDegreesOfFreedom(0, t_prolonged));
}

}  // namespace astronomy
}  // namespace principia
