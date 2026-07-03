#include "physics/far_field_damping.hpp"

#include "base/algebra.hpp"
#include "gtest/gtest.h"
#include "numerics/elementary_functions.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "testing_utilities/almost_equals.hpp"
#include "testing_utilities/numerics_matchers.hpp"

namespace principia {
namespace physics {

using ::testing::Lt;
using namespace principia::base::_algebra;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_far_field_damping;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;
using namespace principia::testing_utilities::_almost_equals;
using namespace principia::testing_utilities::_numerics_matchers;

class FarFieldDampingTest : public ::testing::Test {
 protected:
  FarFieldDampingTest() : damping_(outer_threshold_) {}

  // Evaluates σ at `r`.
  double σ(Length const& r) const {
    double σ;
    double σʹr;
    damping_.ComputeDampedRadialQuantities(r, σ, σʹr);
    return σ;
  }

  // Evaluates σ′ at `r`.
  Inverse<Length> σʹ(Length const& r) const {
    double σ;
    double σʹr;
    damping_.ComputeDampedRadialQuantities(r, σ, σʹr);
    return σʹr / r;
  }

  Length const outer_threshold_ = 3e15 * Metre;
  FarFieldDamping damping_;
};

TEST_F(FarFieldDampingTest, Thresholds) {
  EXPECT_EQ(damping_.outer_threshold(), outer_threshold_);
  EXPECT_EQ(damping_.inner_threshold(), outer_threshold_ / 3);
  EXPECT_EQ(damping_.outer_threshold²(), Pow<2>(outer_threshold_));
}

TEST_F(FarFieldDampingTest, Sigmoid) {
  Length const inner = damping_.inner_threshold();
  Length const outer = damping_.outer_threshold();
  Length const shell_width = outer - inner;

  // Undamped inside the inner threshold, exactly zero above the outer one.
  EXPECT_EQ(σ(inner / 2), 1);
  EXPECT_EQ(σ(inner), 1);
  EXPECT_EQ(σ(outer), 0);
  EXPECT_EQ(σ(2 * outer), 0);

  // At the midpoint of the shell the quintic sigmoid is 1/2.
  EXPECT_THAT(σ(inner + shell_width / 2), AlmostEquals(0.5, 0, 4));

  // σ decreases monotonically on the shell.
  double previous_σ = 1;
  for (int i = 1; i < 100; ++i) {
    double const current_σ = σ(inner + i * shell_width / 100);
    EXPECT_LT(current_σ, previous_σ);
    previous_σ = current_σ;
  }
}

TEST_F(FarFieldDampingTest, Continuity) {
  Length const inner = damping_.inner_threshold();
  Length const outer = damping_.outer_threshold();
  Length const ε = 1e-8 * (outer - inner);

  // σ, σ′ and σ″ are continuous at both thresholds: just inside the shell
  // they are close to their values just outside of it.
  for (Length const& r : {inner + ε, outer - ε}) {
    double σ;
    double σʹr;
    double σʺr²;
    damping_.ComputeDampedRadialQuantities(r, σ, σʹr, σʺr²);
    double const σ_at_threshold = r < 2 * inner ? 1 : 0;
    EXPECT_THAT(σ, AbsoluteErrorFrom(σ_at_threshold, Lt(1e-15)));
    EXPECT_THAT(σʹr, AbsoluteErrorFrom(0.0, Lt(1e-13)));
    EXPECT_THAT(σʺr², AbsoluteErrorFrom(0.0, Lt(1e-5)));
  }
}

TEST_F(FarFieldDampingTest, Derivatives) {
  Length const inner = damping_.inner_threshold();
  Length const shell_width = damping_.outer_threshold() - inner;
  Length const h = 1e-5 * shell_width;

  for (int i = 1; i < 10; ++i) {
    Length const r = inner + i * shell_width / 10;
    double σ_at_r;
    double σʹr;
    double σʺr²;
    damping_.ComputeDampedRadialQuantities(r, σ_at_r, σʹr, σʺr²);

    // σ′ against a central difference of σ.
    Inverse<Length> const numerical_σʹ = (σ(r + h) - σ(r - h)) / (2 * h);
    EXPECT_THAT(σʹr / r, RelativeErrorFrom(numerical_σʹ, Lt(1e-8)));

    // σ″ against a central difference of σ′; at the midpoint of the shell σ″
    // vanishes, so a relative comparison is meaningless there.
    if (i != 5) {
      Inverse<Square<Length>> const numerical_σʺ =
          (σʹ(r + h) - σʹ(r - h)) / (2 * h);
      EXPECT_THAT(σʺr² / Pow<2>(r), RelativeErrorFrom(numerical_σʺ, Lt(1e-6)));
    }
  }
}

TEST_F(FarFieldDampingTest, AccelerationIsGradientOfDampedPotential) {
  GravitationalParameter const μ = 1.327e20 * Pow<3>(Metre) / Pow<2>(Second);
  Length const inner = damping_.inner_threshold();
  Length const shell_width = damping_.outer_threshold() - inner;
  Length const h = 1e-5 * shell_width;

  // The damped potential is V(r) = −σ(r) μ / r; the radial acceleration
  // toward the body deriving from it is (σ − σ′ r) μ / r², which is what the
  // kernels of `Ephemeris` compute.
  auto const V = [&](Length const& r) { return -σ(r) * μ / r; };
  for (int i = 1; i < 10; ++i) {
    Length const r = inner + i * shell_width / 10;
    double σ_at_r;
    double σʹr;
    damping_.ComputeDampedRadialQuantities(r, σ_at_r, σʹr);
    Acceleration const damped_acceleration = (σ_at_r - σʹr) * μ / Pow<2>(r);
    Acceleration const numerical_acceleration =
        (V(r + h) - V(r - h)) / (2 * h);
    EXPECT_THAT(damped_acceleration,
                RelativeErrorFrom(numerical_acceleration, Lt(1e-8)));
  }
}

}  // namespace physics
}  // namespace principia
