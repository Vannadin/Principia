#include "ksp_plugin/pile_up.hpp"

#include <algorithm>
#include <functional>
#include <future>
#include <iterator>
#include <list>
#include <map>
#include <optional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "base/map_util.hpp"
#include "geometry/identity.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/quaternion.hpp"
#include "geometry/rotation.hpp"
#include "geometry/signature.hpp"
#include "geometry/space.hpp"
#include "ksp_plugin/integrators.hpp"
#include "ksp_plugin/manœuvre.hpp"
#include "ksp_plugin/part.hpp"
#include "numerics/davenport_q_method.hpp"
#include "numerics/double_precision.hpp"
#include "physics/sector.hpp"
#include "quantities/si.hpp"

namespace principia {
namespace ksp_plugin {
namespace _pile_up {
namespace internal {

using namespace principia::base::_map_util;
using namespace principia::geometry::_identity;
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_quaternion;
using namespace principia::geometry::_rotation;
using namespace principia::geometry::_signature;
using namespace principia::geometry::_space;
using namespace principia::ksp_plugin::_integrators;
using namespace principia::ksp_plugin::_manœuvre;
using namespace principia::ksp_plugin::_part;
using namespace principia::numerics::_davenport_q_method;
using namespace principia::numerics::_double_precision;
using namespace principia::physics::_sector;
using namespace principia::quantities::_si;

const auto part_x = Vector<double, RigidPart>({1, 0, 0});
const auto part_y = Vector<double, RigidPart>({0, 1, 0});
const auto part_z = Vector<double, RigidPart>({0, 0, 1});

// A pile-up is rebased into another subsystem when the gravitational dominance
// μ/d² of that subsystem exceeds that of its current subsystem by this factor.
// The margin makes the boundary hysteretic: switching back requires the
// inverse ratio, so a pile-up weaving around the balance point does not
// oscillate between representations, and a transit that merely grazes a
// subsystem's region does not adopt it.
constexpr double rebase_dominance_margin = 3;

bool operator==(OnRailsBurn const& left, OnRailsBurn const& right) {
  return left.thrust == right.thrust &&
         left.specific_impulse == right.specific_impulse &&
         left.initial_mass == right.initial_mass &&
         left.direction == right.direction &&
         left.max_duration == right.max_duration;
}

bool operator!=(OnRailsBurn const& left, OnRailsBurn const& right) {
  return !(left == right);
}

PileUp::PileUp(
    std::list<not_null<Part*>> parts,
    Instant const& t,
    Ephemeris<Barycentric>::AdaptiveStepParameters adaptive_step_parameters,
    Ephemeris<Barycentric>::FixedStepParameters fixed_step_parameters,
    not_null<Ephemeris<Barycentric>*> const ephemeris,
    std::function<void()> deletion_callback)
    : lock_(make_not_null_unique<absl::Mutex>()),
      parts_(std::move(parts)),
      ephemeris_(ephemeris),
      adaptive_step_parameters_(std::move(adaptive_step_parameters)),
      fixed_step_parameters_(std::move(fixed_step_parameters)),
      history_(trajectory_.segments().begin()),
      deletion_callback_(std::move(deletion_callback)) {
  LOG(INFO) << "Constructing pile up at " << this;
  // Docking parts may hold representations in distinct subsystems: the rebase
  // runs only for unloaded vessels and its hysteresis leaves a void band where
  // the subsystem follows approach history.  The pile-up needs one
  // representation, so vote for the subsystem carrying the largest part mass
  // (smallest index breaks a tie) and reconcile the parts onto it.
  std::map<int, Mass> mass_by_subsystem;
  for (not_null<Part*> const part : parts_) {
    mass_by_subsystem[part->subsystem()] += part->mass();
  }
  subsystem_ = mass_by_subsystem.begin()->first;
  Mass target_mass = mass_by_subsystem.begin()->second;
  for (auto const& [subsystem, mass] : mass_by_subsystem) {
    if (mass > target_mass) {
      subsystem_ = subsystem;
      target_mass = mass;
    }
  }
  // The anchor of the first part already in the target subsystem: always
  // present, and natively valid, so it needs no folding.
  for (not_null<Part*> const part : parts_) {
    if (part->subsystem() == subsystem_) {
      anchor_ = part->anchor();
      break;
    }
  }
  MechanicalSystem<Barycentric, NonRotatingPileUp> mechanical_system;
  for (not_null<Part*> const part : parts_) {
    if (part->subsystem() != subsystem_ || part->anchor() != anchor_) {
      // A part from a vessel in another placement carries the donor's tags:
      // re-express its rigid motion in the pile-up's, consistently with the
      // retag below.
      auto const [displacement, velocity_offset] =
          ephemeris_->placement_conversion({part->subsystem(), part->anchor()},
                                           {subsystem_, anchor_},
                                           t);
      RigidMotion<Barycentric, Barycentric> const conversion_motion(
          RigidTransformation<Barycentric, Barycentric>(
              Barycentric::origin,
              Barycentric::origin + displacement,
              OrthogonalMap<Barycentric, Barycentric>::Identity()),
          Barycentric::nonrotating,
          -velocity_offset);
      part->set_rigid_motion(conversion_motion * part->rigid_motion());
      part->set_subsystem(subsystem_);
      part->set_anchor(anchor_);
    }
    mechanical_system.AddRigidBody(
        part->rigid_motion(), part->mass(), part->inertia_tensor());
  }
  auto const barycentre = mechanical_system.centre_of_mass();
  trajectory_.Append(t, barycentre).IgnoreError();

  angular_momentum_ = mechanical_system.AngularMomentum();

  RigidMotion<Barycentric, NonRotatingPileUp> const barycentric_to_pile_up =
      mechanical_system.LinearMotion().Inverse();
  for (not_null<Part*> const part : parts_) {
    actual_part_rigid_motion_.emplace(
        part, barycentric_to_pile_up * part->rigid_motion());
  }
  MakeEulerSolver(mechanical_system.InertiaTensor(), t);

  psychohistory_ = trajectory_.NewSegment();

  RecomputeFromParts();
}

PileUp::~PileUp() {
  LOG(INFO) << "Destroying pile up at " << this;
  if (deletion_callback_ != nullptr) {
    deletion_callback_();
  }
}

std::list<not_null<Part*>> const& PileUp::parts() const {
  return parts_;
}

Ephemeris<Barycentric>::FixedStepParameters const&
PileUp::fixed_step_parameters() const {
  return fixed_step_parameters_;
}

void PileUp::set_on_rails_burn(OnRailsBurn const& on_rails_burn) {
  absl::MutexLock l(lock_.get());
  on_rails_burn_ = on_rails_burn;
}

void PileUp::clear_on_rails_burn() {
  absl::MutexLock l(lock_.get());
  on_rails_burn_.reset();
  on_rails_burn_for_prediction_.reset();
}

std::optional<OnRailsBurn> PileUp::on_rails_burn() const {
  absl::MutexLock l(lock_.get());
  return on_rails_burn_;
}

std::optional<OnRailsBurn> PileUp::on_rails_burn_for_prediction() const {
  absl::MutexLock l(lock_.get());
  return on_rails_burn_for_prediction_;
}

void PileUp::SetPartApparentRigidMotion(
    not_null<Part*> const part,
    RigidMotion<RigidPart, Apparent> const& rigid_motion) {
  auto const [_, inserted] =
      apparent_part_rigid_motion_.emplace(part, rigid_motion);
  CHECK(inserted) << "Duplicate part " << part->ShortDebugString() << " at "
                  << rigid_motion;
}

absl::Status PileUp::DeformAndAdvanceTime(Instant const& t) {
  absl::MutexLock l(lock_.get());
  absl::Status status;
  if (psychohistory_->back().time < t) {
    DeformPileUpIfNeeded(t);
    status = AdvanceTime(t);
    NudgeParts();
  } else {
    // The burn arms a single catch-up, even one that has nothing to do.
    on_rails_burn_.reset();
  }
  return status;
}

void PileUp::RecomputeFromParts() {
  absl::MutexLock l(lock_.get());
  mass_ = Mass();
  intrinsic_force_ = Vector<Force, Barycentric>();
  intrinsic_torque_ = Bivector<Torque, NonRotatingPileUp>();
  angular_momentum_change_ = Bivector<AngularMomentum, NonRotatingPileUp>();

  for (not_null<Part*> const part : parts_) {
    mass_ += part->mass();

    intrinsic_force_ += part->intrinsic_force();

    RigidMotion<RigidPart, NonRotatingPileUp> const part_motion =
        FindOrDie(actual_part_rigid_motion_, part);
    DegreesOfFreedom<NonRotatingPileUp> const part_dof =
        part_motion({RigidPart::origin, RigidPart::unmoving});
    intrinsic_torque_ +=
        Wedge(part_dof.position() - NonRotatingPileUp::origin,
              Identity<Barycentric, NonRotatingPileUp>()(
                  part->intrinsic_force())) * Radian +
        Identity<Barycentric, NonRotatingPileUp>()(part->intrinsic_torque());

    AngularVelocity<NonRotatingPileUp> const part_angular_velocity =
        part_motion.angular_velocity_of<RigidPart>();
    InertiaTensor<NonRotatingPileUp> const part_inertia_tensor =
        part_motion.orthogonal_map()(part->inertia_tensor());
    if (part->is_solid_rocket_motor()) {
      // KSP makes the inertia tensor vary proportionally to the mass; this
      // corresponds to the body uniformly changing density.
      angular_momentum_change_ +=
          Wedge(part_dof.position() - NonRotatingPileUp::origin,
                part->mass_change() * part_dof.velocity()) *
              Radian +
          part->mass_change() / part->mass() *
              (part_inertia_tensor * part_angular_velocity);
    }
  }
}

void PileUp::WriteToMessage(not_null<serialization::PileUp*> message) const {
  for (not_null<Part*> const part : parts_) {
    message->add_part_id(part->part_id());
  }
  trajectory_.WriteToMessage(message->mutable_history(),
                             /*tracked=*/{history_, psychohistory_},
                             /*exact=*/{});
  for (auto const& [part, rigid_motion] : actual_part_rigid_motion_) {
    rigid_motion.WriteToMessage(&(
        (*message->mutable_actual_part_rigid_motion())[part->part_id()]));
  }
  for (auto const& [part, rigid_motion] : apparent_part_rigid_motion_) {
    rigid_motion.WriteToMessage(&(
        (*message->mutable_apparent_part_rigid_motion())[part->part_id()]));
  }
  for (auto const& [part, rigid_transformation] : rigid_pile_up_) {
    rigid_transformation.WriteToMessage(&(
        (*message->mutable_rigid_pile_up())[part->part_id()]));
  }
  if (euler_solver_.has_value()) {
    euler_solver_->WriteToMessage(message->mutable_euler_solver());
  }
  angular_momentum_.WriteToMessage(message->mutable_angular_momentum());
  adaptive_step_parameters_.WriteToMessage(
      message->mutable_adaptive_step_parameters());
  fixed_step_parameters_.WriteToMessage(
      message->mutable_fixed_step_parameters());
}

not_null<std::unique_ptr<PileUp>> PileUp::ReadFromMessage(
    serialization::PileUp const& message,
    std::function<not_null<Part*>(PartId)> const& part_id_to_part,
    not_null<Ephemeris<Barycentric>*> const ephemeris,
    std::function<void()> deletion_callback) {
  std::list<not_null<Part*>> parts;
  for (auto const part_id : message.part_id()) {
    parts.push_back(part_id_to_part(part_id));
  }

  bool const is_pre_cartan = !message.has_adaptive_step_parameters() ||
                             !message.has_fixed_step_parameters();
  bool const is_pre_cesàro = message.history().children().empty() &&
                             message.history().segment_size() == 0;
  bool const is_pre_frege = message.actual_part_degrees_of_freedom_size() > 0 ||
                            message.apparent_part_degrees_of_freedom_size() > 0;
  bool const is_pre_frobenius = message.rigid_pile_up().empty() ||
                                !message.has_angular_momentum();
  bool const is_pre_hamilton = message.history().segment_size() == 0;
  LOG_IF(WARNING, is_pre_hamilton)
      << "Reading pre-"
      << (is_pre_cartan      ? "Cartan"
          : is_pre_cesàro    ? "Cesàro"
          : is_pre_frege     ? "Frege"
          : is_pre_frobenius ? "Frobenius"
                             : "Hamilton") << " PileUp";

  std::unique_ptr<PileUp> pile_up;
  if (is_pre_cesàro) {
    if (is_pre_cartan) {
      pile_up = std::unique_ptr<PileUp>(
          new PileUp(std::move(parts),
                     DefaultPsychohistoryParameters(),
                     DefaultHistoryParameters(),
                     DiscreteTrajectory<Barycentric>::ReadFromMessage(
                         message.history(),
                         /*tracked=*/{}),
                     /*history=*/std::nullopt,
                     /*psychohistory=*/std::nullopt,
                     /*angular_momentum=*/{},
                     ephemeris,
                     std::move(deletion_callback)));
    } else {
      pile_up = std::unique_ptr<PileUp>(
          new PileUp(
              std::move(parts),
              Ephemeris<Barycentric>::AdaptiveStepParameters::ReadFromMessage(
                  message.adaptive_step_parameters()),
              Ephemeris<Barycentric>::FixedStepParameters::ReadFromMessage(
                  message.fixed_step_parameters()),
              DiscreteTrajectory<Barycentric>::ReadFromMessage(
                  message.history(),
                  /*tracked=*/{}),
              /*history=*/std::nullopt,
              /*psychohistory=*/std::nullopt,
              /*angular_momentum=*/{},
              ephemeris,
              std::move(deletion_callback)));
    }
    // Fork a psychohistory for compatibility if there is a non-authoritative
    // point.
    if (pile_up->history_->size() == 2) {
      DiscreteTrajectory<Barycentric> psychohistory;
      for (auto const& [time, degrees_of_freedom] : *pile_up->history_) {
        psychohistory.Append(time, degrees_of_freedom).IgnoreError();
      }
      pile_up->trajectory_.ForgetAfter(std::next(pile_up->history_->begin()));
      pile_up->psychohistory_ =
          pile_up->trajectory_.AttachSegments(std::move(psychohistory));
    } else {
      pile_up->psychohistory_ = pile_up->trajectory_.NewSegment();
    }
  } else {
    if (is_pre_frobenius) {
      DiscreteTrajectorySegmentIterator<Barycentric> psychohistory;
      auto trajectory = DiscreteTrajectory<Barycentric>::ReadFromMessage(
          message.history(),
          /*tracked=*/{&psychohistory});
      pile_up = std::unique_ptr<PileUp>(
          new PileUp(
              std::move(parts),
              Ephemeris<Barycentric>::AdaptiveStepParameters::ReadFromMessage(
                  message.adaptive_step_parameters()),
              Ephemeris<Barycentric>::FixedStepParameters::ReadFromMessage(
                  message.fixed_step_parameters()),
              std::move(trajectory),
              /*history=*/std::nullopt,
              psychohistory,
              /*angular_momentum=*/{},
              ephemeris,
              std::move(deletion_callback)));
    } else if (is_pre_hamilton) {
      DiscreteTrajectorySegmentIterator<Barycentric> psychohistory;
      auto trajectory = DiscreteTrajectory<Barycentric>::ReadFromMessage(
          message.history(),
          /*tracked=*/{&psychohistory});
      pile_up = std::unique_ptr<PileUp>(
          new PileUp(
              std::move(parts),
              Ephemeris<Barycentric>::AdaptiveStepParameters::ReadFromMessage(
                  message.adaptive_step_parameters()),
              Ephemeris<Barycentric>::FixedStepParameters::ReadFromMessage(
                  message.fixed_step_parameters()),
              std::move(trajectory),
              /*history=*/std::nullopt,
              psychohistory,
              Bivector<AngularMomentum, NonRotatingPileUp>::ReadFromMessage(
                  message.angular_momentum()),
              ephemeris,
              std::move(deletion_callback)));
    } else {
      DiscreteTrajectorySegmentIterator<Barycentric> history;
      DiscreteTrajectorySegmentIterator<Barycentric> psychohistory;
      auto trajectory = DiscreteTrajectory<Barycentric>::ReadFromMessage(
          message.history(),
          /*tracked=*/{&history, &psychohistory});
      pile_up = std::unique_ptr<PileUp>(
          new PileUp(
              std::move(parts),
              Ephemeris<Barycentric>::AdaptiveStepParameters::ReadFromMessage(
                  message.adaptive_step_parameters()),
              Ephemeris<Barycentric>::FixedStepParameters::ReadFromMessage(
                  message.fixed_step_parameters()),
              std::move(trajectory),
              history,
              psychohistory,
              Bivector<AngularMomentum, NonRotatingPileUp>::ReadFromMessage(
                  message.angular_momentum()),
              ephemeris,
              std::move(deletion_callback)));
    }
  }

  if (is_pre_frege) {
    for (auto const& [part_id, degrees_of_freedom] :
         message.actual_part_degrees_of_freedom()) {
      pile_up->actual_part_rigid_motion_.emplace(
          part_id_to_part(part_id),
          RigidMotion<RigidPart, NonRotatingPileUp>::MakeNonRotatingMotion(
              DegreesOfFreedom<NonRotatingPileUp>::ReadFromMessage(
                  degrees_of_freedom)));
    }
    for (auto const& [part_id, degrees_of_freedom] :
         message.apparent_part_degrees_of_freedom()) {
      pile_up->apparent_part_rigid_motion_.emplace(
          part_id_to_part(part_id),
          RigidMotion<RigidPart, Apparent>::MakeNonRotatingMotion(
              DegreesOfFreedom<Apparent>::ReadFromMessage(degrees_of_freedom)));
    }
  } else {
    for (auto const& [part_id, rigid_motion] :
         message.actual_part_rigid_motion()) {
      pile_up->actual_part_rigid_motion_.emplace(
          part_id_to_part(part_id),
          RigidMotion<RigidPart, NonRotatingPileUp>::ReadFromMessage(
              rigid_motion));
    }
    for (auto const& [part_id, rigid_motion] :
         message.apparent_part_rigid_motion()) {
      pile_up->apparent_part_rigid_motion_.emplace(
          part_id_to_part(part_id),
          RigidMotion<RigidPart, Apparent>::ReadFromMessage(rigid_motion));
    }
  }

  if (is_pre_frobenius) {
    MechanicalSystem<Barycentric, NonRotatingPileUp> mechanical_system;
    for (not_null<Part*> const part : pile_up->parts_) {
      mechanical_system.AddRigidBody(
          part->rigid_motion(), part->mass(), part->inertia_tensor());
    }
    pile_up->MakeEulerSolver(mechanical_system.InertiaTensor(),
                             pile_up->psychohistory_->back().time);
  } else {
    for (auto const& [part_id, rigid_transformation] :
         message.rigid_pile_up()) {
      pile_up->rigid_pile_up_.emplace(
          part_id_to_part(part_id),
          RigidTransformation<RigidPart, PileUpPrincipalAxes>::ReadFromMessage(
              rigid_transformation));
    }
    if (message.has_euler_solver()) {
      pile_up->euler_solver_.emplace(
          EulerSolver<NonRotatingPileUp, PileUpPrincipalAxes>::ReadFromMessage(
              message.euler_solver()));
    }
  }
  pile_up->RecomputeFromParts();

  return check_not_null(std::move(pile_up));
}

PileUp::PileUp(
    std::list<not_null<Part*>>&& parts,
    Ephemeris<Barycentric>::AdaptiveStepParameters adaptive_step_parameters,
    Ephemeris<Barycentric>::FixedStepParameters fixed_step_parameters,
    DiscreteTrajectory<Barycentric> trajectory,
    std::optional<DiscreteTrajectorySegmentIterator<Barycentric>> history,
    std::optional<DiscreteTrajectorySegmentIterator<Barycentric>> psychohistory,
    Bivector<AngularMomentum, NonRotatingPileUp> const& angular_momentum,
    not_null<Ephemeris<Barycentric>*> const ephemeris,
    std::function<void()> deletion_callback)
    : lock_(make_not_null_unique<absl::Mutex>()),
      parts_(std::move(parts)),
      ephemeris_(ephemeris),
      adaptive_step_parameters_(std::move(adaptive_step_parameters)),
      fixed_step_parameters_(std::move(fixed_step_parameters)),
      trajectory_(std::move(trajectory)),
      angular_momentum_(angular_momentum),
      deletion_callback_(std::move(deletion_callback)) {
  subsystem_ = parts_.front()->subsystem();
  anchor_ = parts_.front()->anchor();
  if (history.has_value()) {
    history_ = history.value();
  } else {
    history_ = trajectory_.segments().begin();
  }
  if (psychohistory.has_value()) {
    psychohistory_ = psychohistory.value();
  }
}

int PileUp::subsystem() const {
  return subsystem_;
}

std::optional<Ephemeris<Barycentric>::Anchor> const& PileUp::anchor() const {
  return anchor_;
}

void PileUp::Rebase(Displacement<Barycentric> const& displacement_at_epoch,
                    Velocity<Barycentric> const& velocity_offset,
                    Instant const& epoch,
                    int const subsystem,
                    std::optional<Ephemeris<Barycentric>::Anchor> const&
                        anchor) {
  trajectory_.Translate(displacement_at_epoch, velocity_offset, epoch);
  subsystem_ = subsystem;
  anchor_ = anchor;
  // The fixed instance, if any, holds integrator state in the previous
  // representation; it will be re-created as needed.
  fixed_instance_ = nullptr;
}

DiscreteTrajectory<Barycentric>::value_type const&
PileUp::PresentState() const {
  if (!psychohistory_->empty()) {
    return psychohistory_->back();
  }
  // Only a freshly constructed pile-up lands here: the psychohistory is forked
  // empty at the end of the single-point history.
  return trajectory_.back();
}

std::vector<PileUp::PlacementChange> PileUp::RebaseIfNeeded() {
  std::vector<PlacementChange> changes;
  int const number_of_subsystems = ephemeris_->number_of_subsystems();
  if (number_of_subsystems < 2 || trajectory_.empty()) {
    return changes;
  }
  // The decision point is the present state, which for a pile-up is the end of
  // its trajectory (it never carries a prediction).
  // Copies, not references: the translations below rebuild the timeline that
  // `back()` points into.
  Instant const t = PresentState().time;
  Position<Barycentric> const q = PresentState().degrees_of_freedom.position();
  // The true position, independent of the representation: the dominance
  // geometry below must not depend on whether the pile-up is anchored.  The
  // collapse rounds at the ULP of the void distance, which is harmless in a
  // μ/d² comparison.
  Position<Barycentric> const true_q =
      anchor_.has_value() ? q + anchor_->OffsetAt(t) : q;
  // The dominance μ/d² is computed from the subsystem masses and (linearly
  // extrapolated) barycentres, not from the acceleration field: far-field
  // damping zeroes the field precisely in the region where the boundary lies.
  // The comparisons are cross-multiplied so that a vanishing distance needs no
  // special-casing.
  auto const squared_distance_to_barycentre = [&](int const s) {
    Position<Barycentric> const q_in_s =
        true_q + ephemeris_->subsystem_conversion(subsystem_, s, t);
    return (q_in_s - ephemeris_->subsystem_barycentre(s, t)).Norm²();
  };
  auto const current_distance² =
      squared_distance_to_barycentre(subsystem_);
  GravitationalParameter const current_μ =
      ephemeris_->subsystem_gravitational_parameter(subsystem_);
  int dominant_subsystem = subsystem_;
  GravitationalParameter dominant_μ = current_μ;
  auto dominant_distance² = current_distance²;
  for (int s = 0; s < number_of_subsystems; ++s) {
    if (s == subsystem_) {
      continue;
    }
    auto const distance² = squared_distance_to_barycentre(s);
    GravitationalParameter const μ =
        ephemeris_->subsystem_gravitational_parameter(s);
    if (μ * dominant_distance² > dominant_μ * distance²) {
      dominant_subsystem = s;
      dominant_μ = μ;
      dominant_distance² = distance²;
    }
  }
  if (dominant_subsystem != subsystem_ &&
      dominant_μ * current_distance² >
          rebase_dominance_margin * current_μ * dominant_distance²) {
    // A pure gravity-grouping retag: with an anchor the subsystem conversion
    // is folded into the anchor exactly, so the representation — and with it
    // the mm-scale geometry of the parts — survives the retag bit for bit.
    changes.push_back(RebaseToSubsystem(dominant_subsystem, t));
  }

  // The representation invariant, void and star domain alike: the represented
  // coordinates and the anchor's affine term stay within the re-anchor bound,
  // so their ULP — and with it the World mapping of the parts — stays
  // sub-millimetre everywhere.  Dominance plays no part here: in particular,
  // entering a star's field KEEPS the anchor (unanchored subsystem coordinates
  // of ~1e15 m in a star's domain quantize the parts at 0.125 m, which is
  // owner-visible part-gap misalignment).  Re-anchoring also bounds the affine
  // term (a long coast) and the coordinates (a burn carries the pile-up away
  // from its anchor), keeping anchor *differences* — the relative geometry of
  // two anchored pile-ups — at the ULP of the local parts, sub-mm.
  Position<Barycentric> const q_now =
      PresentState().degrees_of_freedom.position();
  bool const coordinates_exceed_bound =
      (q_now - Barycentric::origin).Norm²() >
      re_anchor_bound_for_testing_ * re_anchor_bound_for_testing_;
  if (anchor_.has_value()) {
    if (coordinates_exceed_bound ||
        (anchor_->velocity * (t - anchor_->epoch)).Norm²() >
            re_anchor_bound_for_testing_ * re_anchor_bound_for_testing_) {
      changes.push_back(AdoptAnchorAtPresentState(t));
    }
  } else if (coordinates_exceed_bound) {
    changes.push_back(AdoptAnchorAtPresentState(t));
  }
  return changes;
}

PileUp::PlacementChange PileUp::RebaseToSubsystem(int const subsystem,
                                                  Instant const& t) {
  // The origins move relative to each other, so the re-expression is affine
  // in time: each point of each trajectory is translated at its own time.
  Displacement<Barycentric> const displacement =
      ephemeris_->subsystem_conversion(subsystem_, subsystem, t);
  Velocity<Barycentric> const velocity_offset =
      ephemeris_->subsystem_velocity_conversion(subsystem_, subsystem);
  LOG(INFO) << "Rebasing pile up at " << this << " from subsystem "
            << subsystem_ << " to subsystem " << subsystem;
  if (anchor_.has_value()) {
    // Fold the subsystem conversion into the anchor instead of translating the
    // timeline: the conversion is affine in time and so is the anchor, so
    // re-expressing the anchor at its own epoch is exact algebra —
    // new.OffsetAt(τ) = old.OffsetAt(τ) + conversion(τ) for all τ — and the
    // anchored coordinates (the trajectory, the parts) stay bit-for-bit
    // untouched.
    Displacement<Barycentric> const conversion_at_epoch =
        displacement - velocity_offset * (t - anchor_->epoch);
    SectorDisplacement<Barycentric> const split =
        SectorDisplacement<Barycentric>::Split(conversion_at_epoch);
    // The cells add exactly; the local parts accumulate through `TwoSum` and
    // the residual is folded back after the (exact) re-centring, mirroring the
    // re-anchor path so that the two ways of moving an anchor round
    // identically.
    DoublePrecision<Displacement<Barycentric>> const local =
        TwoSum(anchor_->offset.local, split.local);
    Ephemeris<Barycentric>::Anchor new_anchor{
        .offset = {.cell = anchor_->offset.cell + split.cell,
                   .local = local.value},
        .velocity = anchor_->velocity + velocity_offset,
        .epoch = anchor_->epoch};
    new_anchor.offset.Recenter();
    new_anchor.offset += local.error;
    // The trajectory does not move; folding into the anchor leaves the
    // represented coordinates intact.
    Rebase(Displacement<Barycentric>{}, Velocity<Barycentric>{}, t, subsystem,
           new_anchor);
    // The parts' rigid motions are valid numbers in the preserved
    // representation; only their tags change.
    for (not_null<Part*> const part : parts_) {
      part->set_subsystem(subsystem_);
      part->set_anchor(anchor_);
    }
    return {.displacement = Displacement<Barycentric>{},
            .velocity_offset = Velocity<Barycentric>{},
            .epoch = t,
            .subsystem = subsystem_,
            .anchor = anchor_};
  }
  Rebase(displacement, velocity_offset, t, subsystem, /*anchor=*/std::nullopt);
  // The parts' rigid motions are expressed in the old representation; keep them
  // consistent with their subsystem tag (they are regenerated from the rebased
  // pile-up trajectory at the next advance, but a docking merge before then
  // reads them directly).
  RigidMotion<Barycentric, Barycentric> const conversion_motion(
      RigidTransformation<Barycentric, Barycentric>(
          Barycentric::origin,
          Barycentric::origin + displacement,
          OrthogonalMap<Barycentric, Barycentric>::Identity()),
      Barycentric::nonrotating,
      -velocity_offset);
  for (not_null<Part*> const part : parts_) {
    part->set_subsystem(subsystem_);
    part->set_rigid_motion(conversion_motion * part->rigid_motion());
  }
  return {.displacement = displacement,
          .velocity_offset = velocity_offset,
          .epoch = t,
          .subsystem = subsystem_,
          .anchor = anchor_};
}

PileUp::PlacementChange PileUp::AdoptAnchorAtPresentState(Instant const& t) {
  // The anchor is seeded from the present state.  Value copies: the
  // translations rebuild the timeline.
  std::optional<Ephemeris<Barycentric>::Anchor> const old_anchor = anchor_;
  DegreesOfFreedom<Barycentric> const degrees_of_freedom =
      PresentState().degrees_of_freedom;
  Displacement<Barycentric> const displacement =
      degrees_of_freedom.position() - Barycentric::origin;
  Velocity<Barycentric> const velocity_offset = degrees_of_freedom.velocity();
  Ephemeris<Barycentric>::Anchor new_anchor{
      .offset = SectorDisplacement<Barycentric>::Split(displacement),
      .velocity = velocity_offset,
      .epoch = t};
  // The exact translation at first adoption; `Split` is exact, so the new
  // representation collapses back to `displacement` bit for bit.
  Displacement<Barycentric> translation = -displacement;
  if (anchor_.has_value()) {
    // Re-anchoring: the new anchor is displaced from the subsystem origin by
    // the old anchor as well as by the anchored coordinates.  The cells carry
    // over exactly; the small terms — the old local part, the affine term, and
    // the anchored coordinates — accumulate in double precision, the value is
    // re-centred (exact) and the residual is folded back at the magnitude of
    // the re-centred local part.  The affine term is reset to the new epoch.
    Displacement<Barycentric> const affine =
        anchor_->velocity * (t - anchor_->epoch);
    DoublePrecision<Displacement<Barycentric>> local =
        TwoSum(anchor_->offset.local, affine);
    local += displacement;
    new_anchor.offset = {.cell = anchor_->offset.cell, .local = local.value};
    new_anchor.offset.Recenter();
    new_anchor.offset += local.error;
    new_anchor.velocity += anchor_->velocity;
    // The translation that keeps the represented positions consistent with the
    // new anchor: the cells difference exactly (`Recenter` may have moved whole
    // cells) and the small terms are summed last.
    translation = (anchor_->offset - new_anchor.offset).Collapse(affine);
  }
  LOG(INFO) << "Pile up at " << this << " adopts an anchor";
  Rebase(translation, -velocity_offset, t, subsystem_, new_anchor);
  // Retag the parts by the same offset (the retag half of
  // `Vessel::TranslateParts`, called with `(translation, -velocity_offset)`;
  // its `conversion_motion` velocity is thus `velocity_offset`).
  RigidMotion<Barycentric, Barycentric> const conversion_motion(
      RigidTransformation<Barycentric, Barycentric>(
          Barycentric::origin,
          Barycentric::origin + translation,
          OrthogonalMap<Barycentric, Barycentric>::Identity()),
      Barycentric::nonrotating,
      velocity_offset);
  for (not_null<Part*> const part : parts_) {
    part->set_anchor(anchor_);
    part->set_rigid_motion(conversion_motion * part->rigid_motion());
  }
  return {.displacement = translation,
          .velocity_offset = -velocity_offset,
          .epoch = t,
          .subsystem = subsystem_,
          .anchor = anchor_};
}

void PileUp::MakeEulerSolver(
    InertiaTensor<NonRotatingPileUp> const& inertia_tensor,
    Instant const& t) {
  auto const eigensystem = inertia_tensor.Diagonalize<PileUpPrincipalAxes>();
  euler_solver_.emplace(eigensystem.form,
                        angular_momentum_,
                        eigensystem.rotation,
                        t);
  RigidTransformation<NonRotatingPileUp, PileUpPrincipalAxes> const
      to_pile_up_principal_axes(
          NonRotatingPileUp::origin,
          PileUpPrincipalAxes::origin,
          eigensystem.rotation.Inverse().Forget<OrthogonalMap>());
  rigid_pile_up_.clear();
  for (auto const& [part, actual_rigid_motion] : actual_part_rigid_motion_) {
    rigid_pile_up_.emplace(
        part,
        to_pile_up_principal_axes * actual_rigid_motion.rigid_transformation());
  }
}

void PileUp::DeformPileUpIfNeeded(Instant const& t) {
  if (apparent_part_rigid_motion_.empty()) {
    RigidMotion<PileUpPrincipalAxes, NonRotatingPileUp> const pile_up_motion =
        euler_solver_->MotionAt(
            t, {NonRotatingPileUp::origin, NonRotatingPileUp::unmoving});

    for (auto& [part, actual_rigid_motion] : actual_part_rigid_motion_) {
      actual_rigid_motion =
          pile_up_motion * RigidMotion<RigidPart, PileUpPrincipalAxes>(
                               rigid_pile_up_.at(part),
                               PileUpPrincipalAxes::nonrotating,
                               PileUpPrincipalAxes::unmoving);
    }
    return;
  }
  // A consistency check that `SetPartApparentDegreesOfFreedom` was called for
  // all the parts.
  // TODO(egg): I'd like to log some useful information on check failure, but I
  // need a clean way of getting the debug strings of all parts (rather than
  // giant self-evaluating lambdas).
  CHECK_EQ(parts_.size(), apparent_part_rigid_motion_.size());
  for (not_null<Part*> const part : parts_) {
    CHECK(apparent_part_rigid_motion_.contains(part));
  }

  Instant const& t0 = psychohistory_->back().time;
  Time const Δt = t - t0;

  MechanicalSystem<Apparent, ApparentPileUp> apparent_system;
  for (auto const& [part, apparent_part_rigid_motion] :
       apparent_part_rigid_motion_) {
    apparent_system.AddRigidBody(
        apparent_part_rigid_motion, part->mass(), part->inertia_tensor());
  }
  auto const apparent_angular_momentum = apparent_system.AngularMomentum();
  auto const apparent_inertia_tensor = apparent_system.InertiaTensor();
  auto const apparent_inertia_eigensystem =
      apparent_inertia_tensor.Diagonalize<PileUpPrincipalAxes>();

  Rotation<PileUpPrincipalAxes, ApparentPileUp> const apparent_attitude =
      apparent_inertia_eigensystem.rotation;

  // In a non-rigid body, the principal axes are not stable, and cannot be used
  // to determine attitude.  We treat this as a flexible body, and use the parts
  // to propagate the attitude: the part orientations at `t0` and `t` are used
  // as input to Davenport's method to figure out how the game rotated the pile-
  // up overall.  This is then used to determine the attitute at `t` based on
  // the principal axis at `t`.

  // Compute the canonical axes of all the parts using their apparent and actual
  // motions.
  std::vector<Vector<double, Apparent>> apparent_directions;
  std::vector<Vector<double, NonRotatingPileUp>> actual_directions;
  std::vector<Mass> masses;
  apparent_directions.reserve(parts_.size());
  actual_directions.reserve(parts_.size());
  for (not_null<Part*> const part : parts_) {
    auto const& apparent_part_orthogonal_map =
        apparent_part_rigid_motion_.at(part).orthogonal_map();
    auto const& actual_part_orthogonal_map =
        actual_part_rigid_motion_.at(part).orthogonal_map();
    apparent_directions.push_back(apparent_part_orthogonal_map(part_x));
    apparent_directions.push_back(apparent_part_orthogonal_map(part_y));
    apparent_directions.push_back(apparent_part_orthogonal_map(part_z));
    actual_directions.push_back(actual_part_orthogonal_map(part_x));
    actual_directions.push_back(actual_part_orthogonal_map(part_y));
    actual_directions.push_back(actual_part_orthogonal_map(part_z));
    for (int d = 1; d <= 3; ++d) {
      masses.push_back(part->mass());
    }
  }

  // Use Davenport's Q Method to figure out how the game rotated the pile-up
  // overall.  The parts are weighted by their masses, so if a tiny antenna
  // wiggles a bit it doesn't have much influence.
  Rotation<NonRotatingPileUp, Apparent> const davenport_rotation =
      DavenportQMethod(/*a=*/actual_directions,
                       /*b=*/apparent_directions,
                       /*weights=*/masses);

  // In order to prevent roundoff accumulation from eventually producing
  // noticeably non-unit quaternions, we normalize `initial_attitude`.
  Rotation<PileUpPrincipalAxes, NonRotatingPileUp> initial_attitude =
      davenport_rotation.Inverse() *
      apparent_system.LinearMotion().orthogonal_map().AsRotation() *
      apparent_attitude;
  initial_attitude = Rotation<PileUpPrincipalAxes, NonRotatingPileUp>(
      Normalize(initial_attitude.quaternion()));

  // We take into account the changes to `angular_momentum_` and to the moments
  // of inertia for the step from t0 to t before propagating the attitude from
  // t0 to t. This forms a splitting with the game, with the game changing
  // angular momentum and moment of inertia according to various physical
  // effects (engines, aerodynamics, internal dynamics, etc.) that depend, among
  // other things, on the attitude and angular velocity, and the Euler solver
  // changing attitude and angular velocity according to Euler’s equations.
  angular_momentum_ += intrinsic_torque_ * Δt + angular_momentum_change_;
  euler_solver_.emplace(apparent_inertia_eigensystem.form,
                        angular_momentum_,
                        initial_attitude,
                        t0);

  // This is where we compute our half of the splitting.
  RigidMotion<PileUpPrincipalAxes, NonRotatingPileUp> const
      actual_pile_up_motion = euler_solver_->MotionAt(
          t, {NonRotatingPileUp::origin, NonRotatingPileUp::unmoving});

  // The motion of a hypothetical rigid body with the same moment of inertia and
  // angular momentum as the apparent parts.
  RigidMotion<PileUpPrincipalAxes, ApparentPileUp> const
      apparent_pile_up_motion(
          RigidTransformation<PileUpPrincipalAxes, ApparentPileUp>(
              PileUpPrincipalAxes::origin,
              ApparentPileUp::origin,
              apparent_attitude.Forget<OrthogonalMap>()),
          apparent_angular_momentum / apparent_inertia_tensor,
          ApparentPileUp::unmoving);

  RigidMotion<ApparentPileUp, NonRotatingPileUp> const rotational_correction =
      actual_pile_up_motion * apparent_pile_up_motion.Inverse();
  RigidMotion<Apparent, NonRotatingPileUp> const correction =
      rotational_correction * apparent_system.LinearMotion().Inverse();

  // Now update the motions of the parts in the pile-up frame, and keep their
  // orientations with respect to the principal axes in case we warp.
  actual_part_rigid_motion_.clear();
  rigid_pile_up_.clear();
  for (auto const& [part, apparent_part_rigid_motion] :
       apparent_part_rigid_motion_) {
    RigidMotion<RigidPart, NonRotatingPileUp> const actual_rigid_motion =
        correction * apparent_part_rigid_motion;

    actual_part_rigid_motion_.emplace(part, actual_rigid_motion);
    rigid_pile_up_.emplace(
        part,
        actual_pile_up_motion.rigid_transformation().Inverse() *
            actual_rigid_motion.rigid_transformation());
  }
  apparent_part_rigid_motion_.clear();
}

absl::Status PileUp::AdvanceTime(Instant const& t) {
  absl::Status status;
  Instant const history_last = history_->back().time;
  bool const has_intrinsic_force =
      intrinsic_force_ != Vector<Force, Barycentric>{};
  on_rails_burn_for_prediction_.reset();
  if (!has_intrinsic_force && !on_rails_burn_.has_value()) {
    // Remove the fork.
    trajectory_.DeleteSegments(psychohistory_);
    if (fixed_instance_ == nullptr) {
      // A single placement carrying `subsystem_` and `anchor_`.  When the pile
      // up is unanchored (`anchor_` empty) `StoppableNewInstance` materializes
      // no anchor vector, keeping the anchorless fast path (an anchor vector
      // holding nullopt would leave `has_anchors` set and cost a per-pair check
      // on every step).
      fixed_instance_ = ephemeris_->NewInstance(
          {&trajectory_},
          Ephemeris<Barycentric>::NoIntrinsicAccelerations,
          fixed_step_parameters_,
          {{subsystem_, anchor_}});
    }
    CHECK_LT(history_->back().time, t);
    status = ephemeris_->FlowWithFixedStep(t, *fixed_instance_);
    psychohistory_ = trajectory_.NewSegment();
    if (history_->back().time < t) {
      // Do not clear the `fixed_instance_` here, we will use it for the next
      // fixed-step integration.
      status.Update(ephemeris_->FlowWithAdaptiveStep(
          &trajectory_,
          Ephemeris<Barycentric>::NoIntrinsicAcceleration,
          t,
          adaptive_step_parameters_,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps,
          {subsystem_, anchor_}));
    }
  } else {
    // Destroy the fixed instance, it wouldn't be correct to use it the next
    // time we go through this function.  It will be re-created as needed.
    fixed_instance_ = nullptr;
    // We make the `psychohistory_`, if any, authoritative, i.e. append it to
    // the end of the `history_`.  We integrate on top of it.  Note how we skip
    // the first point of the psychohistory, which is already present in the
    // `trajectory_`.
    auto const psychohistory_trajectory =
        trajectory_.DetachSegments(psychohistory_);
    CHECK(!psychohistory_trajectory.empty());
    for (auto it = std::next(psychohistory_trajectory.begin());
         it != psychohistory_trajectory.end();
         ++it) {
      trajectory_.Append(it->time, it->degrees_of_freedom).IgnoreError();
    }

    if (has_intrinsic_force) {
      // If the game hands us both an intrinsic force and an on-rails burn, the
      // force, which comes from real physics, wins; the burn is consumed
      // nonetheless.
      on_rails_burn_.reset();
      auto const intrinsic_acceleration =
          [a = intrinsic_force_ / mass_](Instant const& /*t*/) { return a; };
      status = ephemeris_->FlowWithAdaptiveStep(
          &trajectory_,
          intrinsic_acceleration,
          t,
          adaptive_step_parameters_,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps,
          {subsystem_, anchor_});
    } else {
      OnRailsBurn const& burn = *on_rails_burn_;
      Variation<Mass> const mass_flow = burn.thrust / burn.specific_impulse;
      Instant const initial_time = trajectory_.back().time;
      // Cut the burn off before it consumes the entire mass of the pile up,
      // and never let it flow backwards, in case the game hands us
      // inconsistent numbers.
      Time const duration =
          std::max(Time{},
                   std::min(burn.max_duration,
                            0.99 * burn.initial_mass / mass_flow));
      Instant const final_time = initial_time + duration;
      auto const intrinsic_acceleration =
          [burn, mass_flow, initial_time, final_time](Instant const& time) {
            return ThrustAcceleration(time,
                                      burn.direction,
                                      burn.thrust,
                                      burn.initial_mass,
                                      mass_flow,
                                      initial_time,
                                      final_time);
          };
      on_rails_burn_for_prediction_ = burn;
      on_rails_burn_.reset();
      // Integrate the burn exactly to its end, so that the flow never crosses
      // the acceleration discontinuity at the cutoff; if the propellant runs
      // out before `t`, coast the rest of the way.
      status = ephemeris_->FlowWithAdaptiveStep(
          &trajectory_,
          intrinsic_acceleration,
          std::min(final_time, t),
          adaptive_step_parameters_,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps,
          {subsystem_, anchor_});
      if (status.ok() && trajectory_.back().time < t) {
        status.Update(ephemeris_->FlowWithAdaptiveStep(
            &trajectory_,
            Ephemeris<Barycentric>::NoIntrinsicAcceleration,
            t,
            adaptive_step_parameters_,
            Ephemeris<Barycentric>::unlimited_max_ephemeris_steps,
            {subsystem_, anchor_}));
      }
    }
    psychohistory_ = trajectory_.NewSegment();
  }

  // Append the `history_` to the parts' history and the `psychohistory_` to the
  // parts' psychohistory.  Drop the history of the pile-up, we won't need it
  // anymore.
  auto const history_end = history_->end();
  auto const psychohistory_end = psychohistory_->end();
  for (auto it = trajectory_.upper_bound(history_last);
       it != history_end;
       ++it) {
    AppendToPart<&Part::AppendToHistory>(it);
  }
  for (auto it = history_end; it != psychohistory_end; ++it) {
    AppendToPart<&Part::AppendToPsychohistory>(it);
  }
  trajectory_.ForgetBefore(psychohistory_->front().time);

  return status;
}

void PileUp::NudgeParts() const {
  auto const actual_centre_of_mass = psychohistory_->back().degrees_of_freedom;

  RigidMotion<Barycentric, NonRotatingPileUp> const barycentric_to_pile_up{
      RigidTransformation<Barycentric, NonRotatingPileUp>{
          actual_centre_of_mass.position(),
          NonRotatingPileUp::origin,
          OrthogonalMap<Barycentric, NonRotatingPileUp>::Identity()},
      Barycentric::nonrotating,
      actual_centre_of_mass.velocity()};
  auto const pile_up_to_barycentric = barycentric_to_pile_up.Inverse();
  for (not_null<Part*> const part : parts_) {
    RigidMotion<RigidPart, Barycentric> const actual_part_rigid_motion =
        pile_up_to_barycentric * FindOrDie(actual_part_rigid_motion_, part);
    part->set_rigid_motion(actual_part_rigid_motion);
  }
}

template<PileUp::AppendToPartTrajectory append_to_part_trajectory>
void PileUp::AppendToPart(DiscreteTrajectory<Barycentric>::iterator it) const {
  auto const& pile_up_dof = it->degrees_of_freedom;
  RigidMotion<Barycentric, NonRotatingPileUp> const barycentric_to_pile_up(
      RigidTransformation<Barycentric, NonRotatingPileUp>(
          pile_up_dof.position(),
          NonRotatingPileUp::origin,
          OrthogonalMap<Barycentric, NonRotatingPileUp>::Identity()),
      Barycentric::nonrotating,
      pile_up_dof.velocity());
  auto const pile_up_to_barycentric = barycentric_to_pile_up.Inverse();
  for (not_null<Part*> const part : parts_) {
    DegreesOfFreedom<NonRotatingPileUp> const actual_part_degrees_of_freedom =
        FindOrDie(actual_part_rigid_motion_, part)({RigidPart::origin,
                                                    RigidPart::unmoving});
    (static_cast<Part*>(part)->*append_to_part_trajectory)(
        it->time,
        pile_up_to_barycentric(actual_part_degrees_of_freedom));
  }
}

PileUpFuture::PileUpFuture(not_null<PileUp const*> const pile_up,
                           std::future<absl::Status> future)
    : pile_up(pile_up),
      future(std::move(future)) {}

// While anchored in the void, a pile-up whose coordinates grow beyond this
// bound (under thrust), or whose anchor's affine term grows beyond it (a long
// coast), re-anchors, keeping the local ULP at ~0.2 mm.
Length PileUp::re_anchor_bound_for_testing_ = 1e12 * Metre;

}  // namespace internal
}  // namespace _pile_up
}  // namespace ksp_plugin
}  // namespace principia
