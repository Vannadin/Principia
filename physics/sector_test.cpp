#include "physics/sector.hpp"

#include "geometry/frame.hpp"
#include "geometry/space.hpp"
#include "gtest/gtest.h"
#include "numerics/double_precision.hpp"
#include "numerics/elementary_functions.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"
#include "serialization/geometry.pb.h"

namespace principia {
namespace physics {

using namespace principia::geometry::_frame;
using namespace principia::geometry::_space;
using namespace principia::numerics::_double_precision;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_sector;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;

class SectorTest : public ::testing::Test {
 protected:
  using World = Frame<serialization::Frame::TestTag,
                      Inertial,
                      Handedness::Right,
                      serialization::Frame::TEST>;

  static Displacement<World> MakeDisplacement(double const x_metres,
                                              double const y_metres,
                                              double const z_metres) {
    return Displacement<World>(
        {x_metres * Metre, y_metres * Metre, z_metres * Metre});
  }
};

// `Split` decomposes exactly: collapsing the result recovers the input bit
// for bit, at every magnitude from zero to interstellar.
TEST_F(SectorTest, SplitCollapseRoundTrip) {
  for (auto const& displacement :
       {MakeDisplacement(0, 0, 0),
        MakeDisplacement(5, -3, 2),
        MakeDisplacement(1e9, -1e9, 0.125),
        MakeDisplacement(1.1e12, -0.9e12, 0.55e12),
        MakeDisplacement(2e16, -2e16, 1e13),
        MakeDisplacement(-9.46e18, 9.46e18, 2e16)}) {
    auto const sector = SectorDisplacement<World>::Split(displacement);
    EXPECT_EQ(sector.Collapse(), displacement);
  }
}

// After `Split` the remainder is at most half a sector per component.
TEST_F(SectorTest, SplitBounds) {
  auto const sector =
      SectorDisplacement<World>::Split(MakeDisplacement(2e16, -1.7e16, 3e12));
  EXPECT_LE(Abs(sector.local.coordinates().x), sector_side / 2);
  EXPECT_LE(Abs(sector.local.coordinates().y), sector_side / 2);
  EXPECT_LE(Abs(sector.local.coordinates().z), sector_side / 2);
}

// The difference of two sector displacements is exact: the cells subtract as
// integers and the locals are small.  Two points 5 m apart, ~2e16 m from the
// origin — where the ULP of a collapsed displacement is ~4 m — difference to
// exactly 5 m.
TEST_F(SectorTest, ExactDifference) {
  auto const here = SectorDisplacement<World>::Split(
      MakeDisplacement(2e16, -2e16, 1e13));
  auto there = here;
  there += MakeDisplacement(5, 0, 0);
  auto const relative = there - here;
  EXPECT_EQ(SectorIndex{}, relative.cell);
  EXPECT_EQ(MakeDisplacement(5, 0, 0), relative.local);

  // Same, across a cell boundary.
  auto beyond = here;
  beyond.cell.x += 1;
  beyond.local -= Displacement<World>({sector_side, 0 * Metre, 0 * Metre});
  beyond += MakeDisplacement(5, 0, 0);
  auto folded = beyond - here;
  folded.Recenter();
  EXPECT_EQ(SectorIndex{}, folded.cell);
  EXPECT_EQ(MakeDisplacement(5, 0, 0), folded.local);
}

// `Recenter` moves whole cells out of the local part without changing the
// value.
TEST_F(SectorTest, RecenterExact) {
  SectorDisplacement<World> sector{
      .cell = {3, -2, 0},
      .local = MakeDisplacement(2.3e12, -1.8e12, 0.4e12)};
  auto const collapsed = sector.Collapse();
  sector.Recenter();
  EXPECT_EQ(collapsed, sector.Collapse());
  EXPECT_LE(Abs(sector.local.coordinates().x), sector_side / 2);
  EXPECT_LE(Abs(sector.local.coordinates().y), sector_side / 2);
  EXPECT_LE(Abs(sector.local.coordinates().z), sector_side / 2);
}

// The double-precision form is exact and round-trips losslessly through
// `Split` — the property serialization relies on.
TEST_F(SectorTest, DoublePrecisionRoundTrip) {
  SectorDisplacement<World> const sector{
      .cell = {18190, -18190, 9},
      .local = MakeDisplacement(5.0009765625e11, -3.25, 0.125)};
  DoublePrecision<Displacement<World>> const dp = sector.ToDoublePrecision();
  auto const reread = SectorDisplacement<World>::Split(dp);
  EXPECT_EQ(sector.cell, reread.cell);
  EXPECT_EQ(sector.local, reread.local);
}

// Half-cell ties are canonical: a component at exactly ±sector_side/2
// decomposes as −sector_side/2 of the appropriate cell, in `Split` and
// `Recenter` alike, so value-equal decompositions compare equal — in
// particular the double-precision round trip stays representation-lossless
// at the boundary.
TEST_F(SectorTest, CanonicalHalfCellTies) {
  Length const half = sector_side / 2;
  for (double const cells : {-2.5, -1.5, -0.5, 0.5, 1.5, 2.5}) {
    Displacement<World> const displacement(
        {cells * sector_side, 0 * Metre, 0 * Metre});
    auto const sector = SectorDisplacement<World>::Split(displacement);
    EXPECT_EQ(-half, sector.local.coordinates().x) << cells;
    EXPECT_EQ(displacement, sector.Collapse()) << cells;
    auto const reread =
        SectorDisplacement<World>::Split(sector.ToDoublePrecision());
    EXPECT_EQ(sector.cell, reread.cell) << cells;
    EXPECT_EQ(sector.local, reread.local) << cells;
  }

  // `Recenter` lands ties on the same canonical side as `Split`.
  SectorDisplacement<World> sector{
      .cell = {0, 0, 0},
      .local = Displacement<World>({-1.5 * sector_side, 0 * Metre, 0 * Metre})};
  auto const collapsed = sector.Collapse();
  sector.Recenter();
  EXPECT_EQ(-1, sector.cell.x);
  EXPECT_EQ(-half, sector.local.coordinates().x);
  EXPECT_EQ(collapsed, sector.Collapse());
}

// `Collapse(Δq)` sums the small terms before the cell displacement, so a
// millimetre-scale Δq survives the collapse when the cell part vanishes in a
// difference — the collapse-last discipline of the gravity kernels.
TEST_F(SectorTest, CollapseLast) {
  SectorDisplacement<World> const sector{
      .cell = {0, 0, 0},
      .local = MakeDisplacement(1e3, 0, 0)};
  Displacement<World> const Δq = MakeDisplacement(1e-3, 0, 0);
  EXPECT_EQ(MakeDisplacement(1e3 + 1e-3, 0, 0), sector.Collapse(Δq));
}

}  // namespace physics
}  // namespace principia
