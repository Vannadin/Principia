#pragma once

#include "physics/sector.hpp"

#include <cmath>

#include "geometry/r3_element.hpp"

namespace principia {
namespace physics {
namespace _sector {
namespace internal {

using namespace principia::geometry::_r3_element;

inline SectorIndex& SectorIndex::operator+=(SectorIndex const& right) {
  x += right.x;
  y += right.y;
  z += right.z;
  return *this;
}

inline SectorIndex& SectorIndex::operator-=(SectorIndex const& right) {
  x -= right.x;
  y -= right.y;
  z -= right.z;
  return *this;
}

inline SectorIndex operator+(SectorIndex const& left,
                             SectorIndex const& right) {
  SectorIndex result = left;
  result += right;
  return result;
}

inline SectorIndex operator-(SectorIndex const& left,
                             SectorIndex const& right) {
  SectorIndex result = left;
  result -= right;
  return result;
}

template<typename Frame>
SectorDisplacement<Frame> SectorDisplacement<Frame>::Split(
    Displacement<Frame> const& displacement) {
  R3Element<Length> const& coordinates = displacement.coordinates();
  SectorDisplacement result;
  result.cell = {std::llround(coordinates.x / sector_side),
                 std::llround(coordinates.y / sector_side),
                 std::llround(coordinates.z / sector_side)};
  // The nearest lattice point is within a factor of two of the displacement
  // (or the cell is zero and the subtrahend vanishes), so the remainder is
  // exact by the Sterbenz lemma.
  result.local = displacement - FromCell(result.cell);
  return result;
}

template<typename Frame>
SectorDisplacement<Frame> SectorDisplacement<Frame>::Split(
    DoublePrecision<Displacement<Frame>> const& displacement) {
  SectorDisplacement result = Split(displacement.value);
  // `result.local` is exact, so when the true sum `local + error` is
  // representable — as it is for the output of `ToDoublePrecision` — this
  // addition is exact too; otherwise it rounds at the magnitude of the
  // remainder, far below the ULP of `value`.
  result.local += displacement.error;
  return result;
}

template<typename Frame>
Displacement<Frame> SectorDisplacement<Frame>::FromCell(
    SectorIndex const& cell) {
  // The conversions are exact as long as the indices fit the mantissa (a
  // galaxy is ~2^30 cells), and the products are exact because `sector_side`
  // is a power of two.
  return Displacement<Frame>({static_cast<double>(cell.x) * sector_side,
                              static_cast<double>(cell.y) * sector_side,
                              static_cast<double>(cell.z) * sector_side});
}

template<typename Frame>
Displacement<Frame> SectorDisplacement<Frame>::Collapse(
    Displacement<Frame> const& Δq) const {
  return FromCell(cell) + (local + Δq);
}

template<typename Frame>
Displacement<Frame> SectorDisplacement<Frame>::Collapse() const {
  return FromCell(cell) + local;
}

template<typename Frame>
void SectorDisplacement<Frame>::Recenter() {
  R3Element<Length> const& coordinates = local.coordinates();
  SectorIndex const excess{std::llround(coordinates.x / sector_side),
                           std::llround(coordinates.y / sector_side),
                           std::llround(coordinates.z / sector_side)};
  if (excess == SectorIndex{}) {
    return;
  }
  cell += excess;
  // Exact for the same reason as in `Split`.
  local -= FromCell(excess);
}

template<typename Frame>
DoublePrecision<Displacement<Frame>>
SectorDisplacement<Frame>::ToDoublePrecision() const {
  return TwoSum(FromCell(cell), local);
}

template<typename Frame>
SectorDisplacement<Frame>& SectorDisplacement<Frame>::operator+=(
    SectorDisplacement const& right) {
  cell += right.cell;
  local += right.local;
  return *this;
}

template<typename Frame>
SectorDisplacement<Frame>& SectorDisplacement<Frame>::operator-=(
    SectorDisplacement const& right) {
  cell -= right.cell;
  local -= right.local;
  return *this;
}

template<typename Frame>
SectorDisplacement<Frame>& SectorDisplacement<Frame>::operator+=(
    Displacement<Frame> const& right) {
  local += right;
  return *this;
}

template<typename Frame>
SectorDisplacement<Frame>& SectorDisplacement<Frame>::operator-=(
    Displacement<Frame> const& right) {
  local -= right;
  return *this;
}

template<typename Frame>
SectorDisplacement<Frame> operator+(SectorDisplacement<Frame> const& left,
                                    SectorDisplacement<Frame> const& right) {
  SectorDisplacement<Frame> result = left;
  result += right;
  return result;
}

template<typename Frame>
SectorDisplacement<Frame> operator-(SectorDisplacement<Frame> const& left,
                                    SectorDisplacement<Frame> const& right) {
  SectorDisplacement<Frame> result = left;
  result -= right;
  return result;
}

template<typename Frame>
SectorDisplacement<Frame> operator+(SectorDisplacement<Frame> const& left,
                                    Displacement<Frame> const& right) {
  SectorDisplacement<Frame> result = left;
  result += right;
  return result;
}

template<typename Frame>
SectorDisplacement<Frame> operator-(SectorDisplacement<Frame> const& left,
                                    Displacement<Frame> const& right) {
  SectorDisplacement<Frame> result = left;
  result -= right;
  return result;
}

}  // namespace internal
}  // namespace _sector
}  // namespace physics
}  // namespace principia
