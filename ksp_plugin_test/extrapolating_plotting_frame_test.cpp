#include "ksp_plugin/extrapolating_plotting_frame.hpp"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "base/not_null.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "gtest/gtest.h"
#include "integrators/methods.hpp"
#include "integrators/symmetric_linear_multistep_integrator.hpp"
#include "ksp_plugin/frames.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/analytic_subsystem_motion.hpp"
#include "physics/body_centred_non_rotating_reference_frame.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/ephemeris.hpp"
#include "physics/massive_body.hpp"
#include "quantities/astronomy.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/numbers.hpp"  // 🧙 For π.
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/matchers.hpp"  // 🧙 For EXPECT_OK.

namespace principia {
namespace ksp_plugin {

using namespace principia::base::_not_null;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_methods;
using namespace principia::integrators::_symmetric_linear_multistep_integrator;
using namespace principia::ksp_plugin::_extrapolating_plotting_frame;
using namespace principia::ksp_plugin::_frames;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_analytic_subsystem_motion;
using namespace principia::physics::_body_centred_non_rotating_reference_frame;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_ephemeris;
using namespace principia::physics::_massive_body;
using namespace principia::quantities::_astronomy;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;

class ExtrapolatingPlottingFrameTest : public ::testing::Test {
 protected:
  using Model = AnalyticSubsystemMotion<Barycentric>;

  ExtrapolatingPlottingFrameTest()
      : μ_star_(TerrestrialGravitationalParameter),
        μ_planet_(1e-3 * TerrestrialGravitationalParameter),
        orbit_radius_(1e9 * Metre),
        period_(2 * π * Sqrt(Pow<3>(orbit_radius_) / (μ_star_ + μ_planet_))) {
    Speed const v = Sqrt((μ_star_ + μ_planet_) / orbit_radius_);
    Displacement<Barycentric> const r(
        {orbit_radius_, 0 * Metre, 0 * Metre});
    Velocity<Barycentric> const v_relative(
        {0 * Metre / Second, v, 0 * Metre / Second});
    double const f_star = μ_planet_ / (μ_star_ + μ_planet_);
    double const f_planet = μ_star_ / (μ_star_ + μ_planet_);
    std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies;
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("star", μ_star_)));
    bodies.push_back(make_not_null_unique<MassiveBody>(
        MassiveBody::Parameters("planet", μ_planet_)));
    std::vector<DegreesOfFreedom<Barycentric>> const initial_state{
        {Barycentric::origin - f_star * r, -f_star * v_relative},
        {Barycentric::origin + f_planet * r, f_planet * v_relative}};
    ephemeris_ = std::make_unique<Ephemeris<Barycentric>>(
        std::move(bodies),
        initial_state,
        t0_,
        Ephemeris<Barycentric>::AccuracyParameters(
            /*fitting_tolerance=*/0.1 * Milli(Metre),
            /*geopotential_tolerance=*/0x1p-24),
        Ephemeris<Barycentric>::FixedStepParameters(
            SymmetricLinearMultistepIntegrator<
                QuinlanTremaine1990Order12,
                Ephemeris<Barycentric>::NewtonianMotionEquation>(),
            /*step=*/period_ / 300));
    EXPECT_OK(ephemeris_->Prolong(t0_ + 2 * period_));
    star_ = ephemeris_->bodies()[0];
    real_frame_ = std::make_unique<
        BodyCentredNonRotatingReferenceFrame<Barycentric, Navigation>>(
        ephemeris_.get(), star_);
  }

  std::shared_ptr<Model const> MakeModel(Instant const& t) {
    std::vector<Model::Member> members;
    for (int i = 0; i < 2; ++i) {
      auto const body = ephemeris_->bodies()[i];
      members.push_back(
          Model::Member{body->gravitational_parameter(),
                        ephemeris_->trajectory(body)
                            ->EvaluateDegreesOfFreedom(t),
                        i == 0 ? std::nullopt : std::optional<int>(0)});
    }
    return std::make_shared<Model const>(members, t);
  }

  GravitationalParameter const μ_star_;
  GravitationalParameter const μ_planet_;
  Length const orbit_radius_;
  Time const period_;
  Instant const t0_;
  std::unique_ptr<Ephemeris<Barycentric>> ephemeris_;
  MassiveBody const* star_ = nullptr;
  std::unique_ptr<
      BodyCentredNonRotatingReferenceFrame<Barycentric, Navigation>>
      real_frame_;
};

TEST_F(ExtrapolatingPlottingFrameTest, DelegatesWithoutTwin) {
  ExtrapolatingPlottingFrame const frame(real_frame_.get());
  EXPECT_EQ(real_frame_->t_min(), frame.t_min());
  EXPECT_EQ(real_frame_->t_max(), frame.t_max());
  EXPECT_EQ(real_frame_->t_max(), frame.render_t_max());
  EXPECT_EQ(real_frame_->subsystem(), frame.subsystem());
  Instant const t = t0_ + period_ / 2;
  DegreesOfFreedom<Barycentric> const dof(
      Barycentric::origin +
          Displacement<Barycentric>({1 * Metre, 2 * Metre, 3 * Metre}),
      Velocity<Barycentric>({4 * Metre / Second,
                             5 * Metre / Second,
                             6 * Metre / Second}));
  EXPECT_EQ(real_frame_->ToThisFrameAtTimeSimilarly(t)(dof),
            frame.ToThisFrameAtTimeSimilarly(t)(dof));
}

TEST_F(ExtrapolatingPlottingFrameTest, TwinExtendsPastTheHorizon) {
  Instant const horizon = ephemeris_->t_max();
  auto const model = MakeModel(horizon);
  ExtrapolatingPlottingFrame const frame(ephemeris_.get(),
                                         real_frame_.get(),
                                         check_not_null(star_),
                                         model,
                                         /*member=*/0,
                                         horizon);
  // `t_max` never lies; the render ceiling is what extends.
  EXPECT_EQ(real_frame_->t_max(), frame.t_max());
  EXPECT_EQ(InfiniteFuture, frame.render_t_max());
  // Below the horizon the decorator is the real frame.
  Instant const t_below = t0_ + period_ / 3;
  DegreesOfFreedom<Barycentric> const dof(
      Barycentric::origin +
          Displacement<Barycentric>({1 * Metre, 2 * Metre, 3 * Metre}),
      Velocity<Barycentric>({4 * Metre / Second,
                             5 * Metre / Second,
                             6 * Metre / Second}));
  EXPECT_EQ(real_frame_->ToThisFrameAtTimeSimilarly(t_below)(dof),
            frame.ToThisFrameAtTimeSimilarly(t_below)(dof));
  // Beyond it the twin serves, and keeps the centre at the origin.
  Instant const t_beyond = horizon + 100 * period_;
  DegreesOfFreedom<Navigation> const centre_in_frame =
      frame.ToThisFrameAtTimeSimilarly(t_beyond)(
          model->EvaluateDegreesOfFreedom(0, t_beyond));
  EXPECT_EQ(Navigation::origin, centre_in_frame.position());
  EXPECT_EQ(Navigation::unmoving, centre_in_frame.velocity());
  // And the round trip inverts.
  EXPECT_EQ(Navigation::origin,
            frame.ToThisFrameAtTimeSimilarly(t_beyond)(
                frame.FromThisFrameAtTimeSimilarly(t_beyond)(
                    DegreesOfFreedom<Navigation>(
                        Navigation::origin,
                        Navigation::unmoving))).position());
}

TEST_F(ExtrapolatingPlottingFrameTest, AdoptTheRootless) {
  DegreesOfFreedom<Barycentric> const dof(Barycentric::origin,
                                          Barycentric::unmoving);
  std::vector<Model::Member> members = {
      {1 * Pow<3>(Metre) / Pow<2>(Second), dof, std::nullopt},
      {2 * Pow<3>(Metre) / Pow<2>(Second), dof, std::nullopt},
      {0.5 * Pow<3>(Metre) / Pow<2>(Second), dof, std::nullopt}};
  AdoptTheRootless(members);
  EXPECT_FALSE(members[1].parent.has_value());
  EXPECT_EQ(1, *members[0].parent);
  EXPECT_EQ(1, *members[2].parent);

  // A member already parented in the subsystem is left alone.
  std::vector<Model::Member> rooted = {
      {1 * Pow<3>(Metre) / Pow<2>(Second), dof, std::nullopt},
      {2 * Pow<3>(Metre) / Pow<2>(Second), dof, 0}};
  AdoptTheRootless(rooted);
  EXPECT_FALSE(rooted[0].parent.has_value());
  EXPECT_EQ(0, *rooted[1].parent);
}

}  // namespace ksp_plugin
}  // namespace principia
