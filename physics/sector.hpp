#pragma once

#include <cstdint>

#include "geometry/space.hpp"
#include "numerics/double_precision.hpp"
#include "quantities/quantities.hpp"
#include "quantities/si.hpp"

namespace principia {
namespace physics {
namespace _sector {
namespace internal {

using namespace principia::geometry::_space;
using namespace principia::numerics::_double_precision;
using namespace principia::quantities::_quantities;
using namespace principia::quantities::_si;

// The spacing of the canonical sector lattice.  A power of two, so that
// `cell · sector_side` is exact for any cell index that fits the mantissa;
// 2^40 m ≈ 1.10e12 m keeps the local ULP at ~0.12 mm while a vessel at
// interstellar cruise (30 km/s) crosses a cell about once per game-year.
constexpr Length sector_side = 0x1p40 * Metre;

// The integer index of a sector on the canonical lattice.
struct SectorIndex final {
  std::int64_t x = 0;
  std::int64_t y = 0;
  std::int64_t z = 0;

  SectorIndex& operator+=(SectorIndex const& right);
  SectorIndex& operator-=(SectorIndex const& right);

  friend bool operator==(SectorIndex const& left,
                         SectorIndex const& right) = default;
};

SectorIndex operator+(SectorIndex const& left, SectorIndex const& right);
SectorIndex operator-(SectorIndex const& left, SectorIndex const& right);

// A displacement decomposed as `cell · sector_side + local` on the canonical
// lattice.  The two parts are never summed into a single displacement except
// through `Collapse`, which adds the small terms first; the difference of two
// sector displacements is exact in the cell part, so relative geometry
// between two nearby points is accurate to the ULP of `local` — sub-mm —
// regardless of how far both are from the origin.
template<typename Frame>
struct SectorDisplacement final {
  SectorIndex cell;
  Displacement<Frame> local;

  // Decomposes `displacement` into the nearest cell and the remainder.  The
  // remainder is exact (Sterbenz), so `Collapse()` returns `displacement` bit
  // for bit.  The decomposition is canonical: each component of the remainder
  // is in [−sector_side/2, sector_side/2).
  static SectorDisplacement Split(Displacement<Frame> const& displacement);

  // Same, but also folds in the `error` component.  Lossless when the exact
  // sum is representable at the magnitude of the remainder, which is in
  // particular the case for the output of `ToDoublePrecision`.
  static SectorDisplacement Split(
      DoublePrecision<Displacement<Frame>> const& displacement);

  // The exact displacement of the cell from the lattice origin,
  // `cell · sector_side`.
  static Displacement<Frame> FromCell(SectorIndex const& cell);

  // Collapses to a single displacement, `cell · sector_side + (local + Δq)`,
  // summing the small terms first so that the rounding happens once, at the
  // magnitude of the result.
  Displacement<Frame> Collapse(Displacement<Frame> const& Δq) const;
  Displacement<Frame> Collapse() const;

  // Moves whole cells out of `local` into `cell`, exactly; afterwards the
  // decomposition is canonical, each component of `local` in
  // [−sector_side/2, sector_side/2).
  void Recenter();

  // The value `cell · sector_side + local` as an exact double-precision sum;
  // for a canonical decomposition the round trip through `Split` recovers the
  // cell and the local part exactly.  Used for serialization.
  DoublePrecision<Displacement<Frame>> ToDoublePrecision() const;

  SectorDisplacement& operator+=(SectorDisplacement const& right);
  SectorDisplacement& operator-=(SectorDisplacement const& right);
  SectorDisplacement& operator+=(Displacement<Frame> const& right);
  SectorDisplacement& operator-=(Displacement<Frame> const& right);

  friend bool operator==(SectorDisplacement const& left,
                         SectorDisplacement const& right) = default;
};

template<typename Frame>
SectorDisplacement<Frame> operator+(SectorDisplacement<Frame> const& left,
                                    SectorDisplacement<Frame> const& right);
template<typename Frame>
SectorDisplacement<Frame> operator-(SectorDisplacement<Frame> const& left,
                                    SectorDisplacement<Frame> const& right);
template<typename Frame>
SectorDisplacement<Frame> operator+(SectorDisplacement<Frame> const& left,
                                    Displacement<Frame> const& right);
template<typename Frame>
SectorDisplacement<Frame> operator-(SectorDisplacement<Frame> const& left,
                                    Displacement<Frame> const& right);

}  // namespace internal

using internal::sector_side;
using internal::SectorDisplacement;
using internal::SectorIndex;

}  // namespace _sector
}  // namespace physics
}  // namespace principia

#include "physics/sector_body.hpp"
