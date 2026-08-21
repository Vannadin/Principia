#pragma once

#include "physics/analytic_subsystem_motion.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "geometry/barycentre_calculator.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/space.hpp"
#include "numerics/elementary_functions.hpp"
#include "physics/massive_body.hpp"
#include "quantities/astronomy.hpp"

namespace principia {
namespace physics {
namespace _analytic_subsystem_motion {
namespace internal {

using namespace principia::geometry::_barycentre_calculator;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_space;
using namespace principia::numerics::_elementary_functions;
using namespace principia::physics::_massive_body;
using namespace principia::quantities::_astronomy;

template<typename Frame>
AnalyticSubsystemMotion<Frame>::AnalyticSubsystemMotion(
    std::vector<Member> const& members,
    Instant const& epoch,
    AnalyticSubsystemMotion const* const previous)
    : epoch_(epoch) {
  int const n = static_cast<int>(members.size());
  CHECK_GE(n, 1);
  parents_.reserve(n);
  for (auto const& member : members) {
    CHECK_GT(member.gravitational_parameter, GravitationalParameter{});
    parents_.push_back(member.parent);
  }
  if (previous != nullptr) {
    CHECK(parents_ == previous->parents_);
  }

  std::vector<std::vector<int>> children(n);
  int root = -1;
  for (int m = 0; m < n; ++m) {
    if (parents_[m].has_value()) {
      int const p = *parents_[m];
      CHECK_GE(p, 0);
      CHECK_LT(p, n);
      CHECK_NE(p, m);
      children[p].push_back(m);
    } else {
      CHECK_EQ(root, -1) << "several roots";
      root = m;
    }
  }
  CHECK_NE(root, -1) << "no root";

  for (int m = 0; m < n; ++m) {
    std::sort(children[m].begin(),
              children[m].end(),
              [&members, m](int const left, int const right) {
                auto const distance² = [&members, m](int const c) {
                  return (members[c].degrees_of_freedom.position() -
                          members[m].degrees_of_freedom.position()).Norm²();
                };
                auto const left_distance² = distance²(left);
                auto const right_distance² = distance²(right);
                // The index breaks ties to make the order reproducible.
                return left_distance² != right_distance²
                           ? left_distance² < right_distance²
                           : left < right;
              });
  }

  std::vector<GravitationalParameter> subtree_μ(n);
  std::vector<DegreesOfFreedom<Frame>> subtree_barycentre(
      n,
      DegreesOfFreedom<Frame>(Frame::origin, Frame::unmoving));
  int visited = 0;
  std::function<void(int)> const compute_subtree = [&](int const m) {
    ++visited;
    BarycentreCalculator<DegreesOfFreedom<Frame>, GravitationalParameter>
        calculator;
    calculator.Add(members[m].degrees_of_freedom,
                   members[m].gravitational_parameter);
    for (int const c : children[m]) {
      compute_subtree(c);
      calculator.Add(subtree_barycentre[c], subtree_μ[c]);
    }
    subtree_μ[m] = calculator.weight();
    subtree_barycentre[m] = calculator.Get();
  };
  compute_subtree(root);
  CHECK_EQ(visited, n) << "the parents do not form a tree";

  jacobi_.resize(n);
  for (int m = 0; m < n; ++m) {
    if (children[m].empty()) {
      continue;
    }
    BarycentreCalculator<DegreesOfFreedom<Frame>, GravitationalParameter>
        inner;
    inner.Add(members[m].degrees_of_freedom,
              members[m].gravitational_parameter);
    for (int const c : children[m]) {
      RelativeDegreesOfFreedom<Frame> const state =
          subtree_barycentre[c] - inner.Get();
      GravitationalParameter const pair_μ = inner.weight() + subtree_μ[c];
      std::optional<bool> previous_tier;
      if (previous != nullptr) {
        previous_tier = previous->jacobi_[c]->kepler.has_value();
      }
      JacobiVector v;
      v.state_at_epoch = state;
      v.weight = subtree_μ[c] / pair_μ;
      if (PairIsKeplerian(state, pair_μ, previous_tier)) {
        v.kepler.emplace(MassiveBody(MassiveBody::Parameters(inner.weight())),
                         MassiveBody(MassiveBody::Parameters(subtree_μ[c])),
                         state,
                         epoch);
      }
      jacobi_[c].emplace(std::move(v));
      inner.Add(subtree_barycentre[c], subtree_μ[c]);
    }
  }
  barycentre_at_epoch_ = subtree_barycentre[root];

  paths_.resize(n);
  for (int m = 0; m < n; ++m) {
    std::vector<int>& path = paths_[m];
    for (std::optional<int> a = m; a.has_value(); a = parents_[*a]) {
      path.push_back(*a);
    }
    std::reverse(path.begin(), path.end());
  }
  ordered_children_ = std::move(children);
}

template<typename Frame>
Instant const& AnalyticSubsystemMotion<Frame>::epoch() const {
  return epoch_;
}

template<typename Frame>
DegreesOfFreedom<Frame>
AnalyticSubsystemMotion<Frame>::EvaluateDegreesOfFreedom(
    int const member,
    Instant const& t) const {
  CHECK_GE(member, 0);
  CHECK_LT(member, static_cast<int>(paths_.size()));
  auto offset = RelativeDegreesOfFreedom<Frame>(Displacement<Frame>{},
                                                Velocity<Frame>{});
  auto const& path = paths_[member];
  int const path_size = static_cast<int>(path.size());
  for (int depth = 0; depth < path_size; ++depth) {
    int const node = path[depth];
    int const next = depth + 1 < path_size ? path[depth + 1] : -1;
    // At each node, with Jacobi vectors jᵢ and weights wᵢ accumulating over
    // the ordered children, the barycentre of the cluster interior to child i
    // is the node barycentre minus Σ_{l≥i} wₗ jₗ; the terms interior to the
    // path contribute nothing.
    bool before_next = next != -1;
    for (int const c : ordered_children_[node]) {
      if (before_next && c != next) {
        continue;
      }
      JacobiVector const& v = *jacobi_[c];
      RelativeDegreesOfFreedom<Frame> const j = EvaluateJacobiVector(v, t);
      if (before_next) {
        offset += (1 - v.weight) * j;
        before_next = false;
      } else {
        offset -= v.weight * j;
      }
    }
  }
  Time const Δt = t - epoch_;
  return DegreesOfFreedom<Frame>(
             barycentre_at_epoch_.position() +
                 barycentre_at_epoch_.velocity() * Δt,
             barycentre_at_epoch_.velocity()) +
         offset;
}

template<typename Frame>
bool AnalyticSubsystemMotion<Frame>::JacobiVectorIsKeplerian(
    int const member) const {
  CHECK_GE(member, 0);
  CHECK_LT(member, static_cast<int>(jacobi_.size()));
  CHECK(jacobi_[member].has_value()) << "member " << member << " is the root";
  return jacobi_[member]->kepler.has_value();
}

template<typename Frame>
bool AnalyticSubsystemMotion<Frame>::PairIsKeplerian(
    RelativeDegreesOfFreedom<Frame> const& state,
    GravitationalParameter const& pair_gravitational_parameter,
    std::optional<bool> const previous_tier_is_keplerian) {
  GravitationalParameter const& μ = pair_gravitational_parameter;
  Length const r = state.displacement().Norm();
  Speed const v = state.velocity().Norm();
  SpecificEnergy const ε = Pow<2>(v) / 2 - μ / r;
  auto const h = Wedge(state.displacement(), state.velocity()).Norm();
  double const e² = 1 + 2 * ε * Pow<2>(h) / Pow<2>(μ);
  double const e = Sqrt(std::max(0.0, e²));
  if (Abs(e - 1) < near_parabolic_band_) {
    return false;
  }
  if (previous_tier_is_keplerian.has_value() &&
      Abs(ε) <= hysteresis_band_ * μ / (2 * r)) {
    return *previous_tier_is_keplerian;
  }
  if (ε >= SpecificEnergy{}) {
    return false;
  }
  // Beyond this separation the two-body propagation is pointless and
  // ill-conditioned — the period exceeds ~3 × 10⁵ a, so any span this model
  // serves covers a vanishing arc, and a linear drift is already accurate
  // (measured: 0.12 AU over 1000 a at 1.2 × 10⁴ AU) — whereas below it the
  // orbit matters (61 Cyg, P ≈ 659 a).
  Length const wide_pair_threshold =
      4443 * AstronomicalUnit * Sqrt(μ / SolarGravitationalParameter);
  return r <= wide_pair_threshold;
}

template<typename Frame>
RelativeDegreesOfFreedom<Frame>
AnalyticSubsystemMotion<Frame>::EvaluateJacobiVector(JacobiVector const& v,
                                                     Instant const& t) const {
  if (v.kepler.has_value()) {
    return v.kepler->StateVectors(t);
  }
  return {v.state_at_epoch.displacement() +
              v.state_at_epoch.velocity() * (t - epoch_),
          v.state_at_epoch.velocity()};
}

}  // namespace internal
}  // namespace _analytic_subsystem_motion
}  // namespace physics
}  // namespace principia
