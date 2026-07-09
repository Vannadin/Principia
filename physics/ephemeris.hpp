#pragma once

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "base/concepts.hpp"
#include "base/not_null.hpp"
#include "base/recurring_thread.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/instant.hpp"
#include "geometry/space.hpp"
#include "google/protobuf/repeated_field.h"
#include "integrators/integrators.hpp"
#include "integrators/ordinary_differential_equations.hpp"
#include "numerics/double_precision.hpp"
#include "physics/apsides.hpp"
#include "physics/checkpointer.hpp"
#include "physics/clientele.hpp"
#include "physics/continuous_trajectory.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/far_field_damping.hpp"
#include "physics/geopotential.hpp"
#include "physics/sector.hpp"
#include "physics/integration_parameters.hpp"
#include "physics/massive_body.hpp"
#include "physics/tensors.hpp"
#include "quantities/named_quantities.hpp"
#include "quantities/quantities.hpp"
#include "serialization/ksp_plugin.pb.h"
#include "serialization/numerics.pb.h"
#include "serialization/physics.pb.h"

namespace principia {
namespace physics {
namespace _ephemeris {
namespace internal {

using namespace principia::base::_concepts;
using namespace principia::base::_not_null;
using namespace principia::base::_recurring_thread;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_instant;
using namespace principia::geometry::_space;
using namespace principia::integrators::_integrators;
using namespace principia::integrators::_ordinary_differential_equations;
using namespace principia::numerics::_double_precision;
using namespace principia::physics::_apsides;
using namespace principia::physics::_checkpointer;
using namespace principia::physics::_clientele;
using namespace principia::physics::_continuous_trajectory;
using namespace principia::physics::_degrees_of_freedom;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_far_field_damping;
using namespace principia::physics::_geopotential;
using namespace principia::physics::_integration_parameters;
using namespace principia::physics::_massive_body;
using namespace principia::physics::_sector;
using namespace principia::physics::_tensors;
using namespace principia::quantities::_named_quantities;
using namespace principia::quantities::_quantities;

// Partitions the given `positions` into subsystems by single-linkage
// clustering: two positions within `threshold` of each other end up in the
// same subsystem, as does any chain of such positions.  The subsystem indices
// are dense and numbered by order of first appearance.  Returns an empty
// vector if all the positions end up in a single subsystem, so that the result
// may be passed to the constructor of `Ephemeris` without altering the
// representation in that case.
template<typename Frame>
std::vector<int> ClusterSubsystems(
    std::vector<Position<Frame>> const& positions,
    Length const& threshold);

// Note on thread-safety: the integration functions (Prolong, FlowWithFixedStep,
// FlowWithAdaptiveStep) can be called concurrently as long as their parameters
// designate distinct objects.  No guarantee is offered for the other functions.
template<typename Frame>
class Ephemeris {
  static_assert(Frame::is_inertial, "Frame must be inertial");

 public:
  using IntrinsicAcceleration =
      std::function<Vector<Acceleration, Frame>(Instant const& time)>;
  static std::nullptr_t constexpr NoIntrinsicAcceleration = nullptr;
  using GeneralizedIntrinsicAcceleration =
      std::function<Vector<Acceleration, Frame>(
          Instant const& time,
          DegreesOfFreedom<Frame> const& degrees_of_freedom)>;
  using IntrinsicAccelerations = std::vector<IntrinsicAcceleration>;
  static IntrinsicAccelerations const NoIntrinsicAccelerations;
  static std::int64_t constexpr unlimited_max_ephemeris_steps =
      std::numeric_limits<std::int64_t>::max();

  using BodiesToPositions =
      absl::flat_hash_map<not_null<MassiveBody const*>, Position<Frame>>;
  using BodiesToVelocities =
      absl::flat_hash_map<not_null<MassiveBody const*>, Velocity<Frame>>;
  using BodiesToDegreesOfFreedom =
      absl::flat_hash_map<not_null<MassiveBody const*>,
                          DegreesOfFreedom<Frame>>;

  // The equations describing the motion of the `bodies_`.
  using NewtonianMotionEquation =
      SpecialSecondOrderDifferentialEquation<Position<Frame>>;
  using GeneralizedNewtonianMotionEquation =
      ExplicitSecondOrderOrdinaryDifferentialEquation<Position<Frame>>;

  using AdaptiveStepParameters =
      _integration_parameters::AdaptiveStepParameters<NewtonianMotionEquation>;
  using FixedStepParameters =
      _integration_parameters::FixedStepParameters<NewtonianMotionEquation>;
  using GeneralizedAdaptiveStepParameters =
      _integration_parameters::AdaptiveStepParameters<
          GeneralizedNewtonianMotionEquation>;

  // An affine origin relative to a subsystem's local origin: the positions of
  // a trajectory represented in this anchor are offset from subsystem-relative
  // positions by `offset + velocity * (t - epoch)`.  Used for vessels anchored
  // in the force-free void between stellar subsystems, whose subsystem-relative
  // coordinates would otherwise grow until they quantize rendezvous physics.
  // The offset lives on the canonical sector lattice, so the offsets of two
  // anchors difference exactly in their cell part: the relative geometry of
  // two anchored vessels is accurate to the ULP of the local parts — sub-mm —
  // not to the ULP of the void distance.
  struct Anchor {
    SectorDisplacement<Frame> offset;
    Velocity<Frame> velocity;
    Instant epoch;

    // Evaluates `offset + velocity * (t - epoch)`, collapsed to a single
    // displacement; all consumers must use this so that the rounding is
    // identical everywhere.  When *two* anchors are involved, use
    // `Conversion` instead of differencing two collapsed offsets.
    Displacement<Frame> OffsetAt(Instant const& t) const;

    // The affine offset (and velocity) to add to a point expressed under
    // anchor `from` to re-express it under anchor `to` at time `t`.  A
    // nullopt anchor is the zero offset, i.e., plain subsystem-relative
    // coordinates; both anchors are affine in time, so a single translation
    // at any epoch is exact.  The offsets are differenced before any
    // collapse — the cells subtract exactly — so for nearby anchors the
    // result is small and precise where the difference of two `OffsetAt`
    // results would carry the ULP of the void distance.  Parallels
    // `Ephemeris::subsystem_conversion` for the anchor degree of freedom.
    static std::pair<Displacement<Frame>, Velocity<Frame>> Conversion(
        std::optional<Anchor> const& from,
        std::optional<Anchor> const& to,
        Instant const& t);

    friend bool operator==(Anchor const& left, Anchor const& right) = default;

    void WriteToMessage(
        not_null<serialization::Ephemeris::Anchor*> message) const;
    static Anchor ReadFromMessage(
        serialization::Ephemeris::Anchor const& message);
  };

  // Bundles the two parameters that together specify how a massless
  // trajectory's positions are represented: the `subsystem` relative to whose
  // local origin they are expressed, and the per-vessel `anchor` (if any)
  // further displacing that representation in the force-free void.  They travel
  // together — and the constructor requires both — so that a call site naming a
  // subsystem cannot silently forget the anchor, the omission that made an
  // anchored vessel's trajectory plunge into its home star.  Use `Stock()` for
  // the ordinary { subsystem 0, no anchor } case; a `Stock()` at a call site
  // reads as an intentional anchorless placement rather than a mistake.
  struct SubsystemPlacement {
    SubsystemPlacement(int const subsystem, std::optional<Anchor> anchor)
        : subsystem(subsystem), anchor(std::move(anchor)) {}
    static SubsystemPlacement Stock() {
      return SubsystemPlacement(0, std::nullopt);
    }
    int subsystem;
    std::optional<Anchor> anchor;
  };

  class AccuracyParameters final {
   public:
    AccuracyParameters(Length const& fitting_tolerance,
                       double geopotential_tolerance);

    void WriteToMessage(
        not_null<serialization::Ephemeris::AccuracyParameters*> message) const;
    static AccuracyParameters ReadFromMessage(
        serialization::Ephemeris::AccuracyParameters const& message);

   private:
    Length fitting_tolerance_;
    double geopotential_tolerance_ = 0;
    friend class Ephemeris<Frame>;
  };

  // Constructs an Ephemeris that owns the `bodies`.  The elements of vectors
  // `bodies` and `initial_state` correspond to one another.  If `subsystems`
  // is nonempty, it must be parallel to `bodies` and partition them into
  // subsystems identified by dense indices starting at 0; the positions of the
  // bodies of each subsystem are then represented relative to a local origin
  // anchored at the initial position of its first body, which preserves
  // precision when the subsystems are very far apart.  If `subsystems` is
  // empty, all the bodies belong to subsystem 0 and the positions are
  // represented as given.  If `far_field_damping_floor` is strictly positive,
  // the point-mass potential of each body is damped to exactly zero (see
  // `FarFieldDamping`) beyond the distance where its gravitational
  // acceleration falls below that floor; this only affects the computations
  // on massless bodies, not the motion of the massive bodies.
  Ephemeris(std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies,
            std::vector<DegreesOfFreedom<Frame>> const& initial_state,
            Instant const& initial_time,
            AccuracyParameters const& accuracy_parameters,
            FixedStepParameters fixed_step_parameters,
            std::vector<int> const& subsystems = {},
            Acceleration const& far_field_damping_floor = {});

  virtual ~Ephemeris();

  // Returns the bodies in the order in which they were given at construction.
  virtual std::vector<not_null<MassiveBody const*>> const& bodies() const;

  // Returns the subsystem of the given `body`.
  virtual int subsystem_of_body(not_null<MassiveBody const*> body) const;

  // Returns the number of subsystems; 1 unless subsystems were given at
  // construction.
  virtual int number_of_subsystems() const;

  // Returns the displacement from the local origin of subsystem `s2` to that
  // of subsystem `s1`, at time `t`, as a sector displacement whose cell part
  // differences exactly against other sector displacements.
  SectorDisplacement<Frame> inter_subsystem_offset(
      int s1,
      int s2,
      Instant const& t) const;

  // Returns the displacement to add to a position represented relative to the
  // local origin of subsystem `s1` so that it becomes represented relative to
  // the local origin of subsystem `s2` at time `t`, rounded to a single
  // displacement.  Zero when `s1 == s2`.
  virtual Displacement<Frame> subsystem_conversion(int s1,
                                                   int s2,
                                                   Instant const& t) const;

  // Returns the velocity to add to a velocity represented relative to the
  // (moving) local origin of subsystem `s1` so that it becomes represented
  // relative to that of subsystem `s2`.  Zero when `s1 == s2`.
  virtual Velocity<Frame> subsystem_velocity_conversion(int s1, int s2) const;

  // Returns the total gravitational parameter of the bodies of subsystem `s`.
  // Must not be called unless subsystems were given at construction.
  virtual GravitationalParameter const& subsystem_gravitational_parameter(
      int s) const;

  // Returns the barycentre of the bodies of subsystem `s` at time `t`,
  // relative to the local origin of that subsystem.  The origin moves with
  // the barycentre, so this is the (constant) initial barycentre.  Must not
  // be called unless subsystems were given at construction.
  virtual Position<Frame> subsystem_barycentre(int s, Instant const& t) const;

  // Returns the velocity of the barycentre of the bodies of subsystem `s`.
  // Must not be called unless subsystems were given at construction.
  virtual Velocity<Frame> const& subsystem_barycentre_velocity(int s) const;

  // Returns true iff the far field of every massive body is damped to exactly
  // zero at the given `position` (represented relative to the local origin of
  // `subsystem`) — i.e., a massless body there coasts force-free.  Always
  // false if the far field is not damped.  The bodies of `subsystem` are
  // checked first, so a position near its own star exits on the first test.
  virtual bool FarFieldIsZero(Position<Frame> const& position,
                              int subsystem,
                              Instant const& t) const EXCLUDES(lock_);

  // Returns the trajectory for the given `body`.
  virtual not_null<ContinuousTrajectory<Frame> const*> trajectory(
      not_null<MassiveBody const*> body) const;

  // Returns true if at least one of the trajectories is empty.
  virtual bool empty() const;

  // The maximum of the `t_min`s of the trajectories.
  virtual Instant t_min() const EXCLUDES(lock_);
  // The mimimum of the `t_max`s of the trajectories.
  virtual Instant t_max() const EXCLUDES(lock_);

  virtual FixedStepSizeIntegrator<NewtonianMotionEquation> const&
  planetary_integrator() const;

  virtual absl::Status last_severe_integration_status() const;

  // Convenience methods to evaluate the positions/velocities for all bodies in
  // this ephemeris.
  BodiesToPositions EvaluateAllPositions(Instant const& t) const
      EXCLUDES(lock_);
  BodiesToVelocities EvaluateAllVelocities(Instant const& t) const
      EXCLUDES(lock_);
  BodiesToDegreesOfFreedom EvaluateAllDegreesOfFreedom(Instant const& t) const
      EXCLUDES(lock_);

  // Prolongs the ephemeris up to at least `t`.  Returns an error iff the thread
  // is stopped.  After a successful call with the second parameter defaulted,
  // `t_max() >= t`.
  virtual absl::Status Prolong(
      Instant const& t,
      std::int64_t max_ephemeris_steps = unlimited_max_ephemeris_steps)
      EXCLUDES(lock_);

  // Asks the reanimator thread to asynchronously reconstruct the past so that
  // the `t_min()` of the ephemeris ultimately ends up at or before
  // `desired_t_min`.
  void RequestReanimation(Instant const& desired_t_min);

  // Same as `RequestReanimation`, but synchronous.  This function blocks until
  // the `t_min()` of the ephemeris is at or before `desired_t_min`.
  void AwaitReanimation(Instant const& desired_t_min);

  // Creates an instance suitable for integrating the given `trajectories` with
  // their `intrinsic_accelerations` using a fixed-step integrator parameterized
  // by `parameters`.  If `placements` is nonempty, it must be parallel to
  // `trajectories` and give the `SubsystemPlacement` (subsystem local origin,
  // plus anchor if set) relative to which the positions of each trajectory are
  // represented; empty means `SubsystemPlacement::Stock()` for all the
  // trajectories.
  virtual not_null<
      std::unique_ptr<typename Integrator<NewtonianMotionEquation>::Instance>>
  NewInstance(
      std::vector<not_null<DiscreteTrajectory<Frame>*>> const& trajectories,
      IntrinsicAccelerations const& intrinsic_accelerations,
      FixedStepParameters const& parameters,
      std::vector<SubsystemPlacement> const& placements = {});

  // Same as above, but returns an error status if the thread is stopped.
  virtual absl::StatusOr<not_null<
      std::unique_ptr<typename Integrator<NewtonianMotionEquation>::Instance>>>
  StoppableNewInstance(
      std::vector<not_null<DiscreteTrajectory<Frame>*>> const& trajectories,
      IntrinsicAccelerations const& intrinsic_accelerations,
      FixedStepParameters const& parameters,
      std::vector<SubsystemPlacement> const& placements = {});

  // Integrates, until exactly `t` (except for timeouts or singularities), the
  // `trajectory` followed by a massless body in the gravitational potential
  // described by `*this`.  If `t > t_max()`, calls `Prolong(t)` beforehand.
  // Prolongs the ephemeris by at most `max_ephemeris_steps`.  Returns OK if and
  // only if `*trajectory` was integrated until `t`.  The positions of the
  // `trajectory` are represented as given by `placement` (relative to the local
  // origin of its subsystem, further displaced by its anchor if set).
  virtual absl::Status FlowWithAdaptiveStep(
      not_null<DiscreteTrajectory<Frame>*> trajectory,
      IntrinsicAcceleration intrinsic_acceleration,
      Instant const& t,
      AdaptiveStepParameters const& parameters,
      std::int64_t max_ephemeris_steps = unlimited_max_ephemeris_steps,
      SubsystemPlacement const& placement = SubsystemPlacement::Stock())
      EXCLUDES(lock_);

  // Same as above, but uses a generalized integrator.
  virtual absl::Status FlowWithAdaptiveStep(
      not_null<DiscreteTrajectory<Frame>*> trajectory,
      GeneralizedIntrinsicAcceleration intrinsic_acceleration,
      Instant const& t,
      GeneralizedAdaptiveStepParameters const& parameters,
      std::int64_t max_ephemeris_steps = unlimited_max_ephemeris_steps,
      SubsystemPlacement const& placement = SubsystemPlacement::Stock())
      EXCLUDES(lock_);

  // Integrates, until at most `t`, the trajectories followed by massless
  // bodies in the gravitational potential described by `*this`.  If
  // `t > t_max()`, calls `Prolong(t)` beforehand.  The trajectories and
  // integration parameters are given by the `instance`.
  virtual absl::Status FlowWithFixedStep(
      Instant const& t,
      typename Integrator<NewtonianMotionEquation>::Instance& instance)
      EXCLUDES(lock_);

  // Returns the Jacobian of the acceleration field exerted on the given `body`
  // by the rest of the system.
  JacobianOfAcceleration<Frame> ComputeJacobianOnMassiveBody(
      not_null<MassiveBody const*> body,
      Instant const& t) const EXCLUDES(lock_);

  // Returns the gravitational jerk on a massless body with the given
  // `degrees_of_freedom` at time `t`.  The position is represented relative to
  // the local origin of `subsystem`.
  Vector<Jerk, Frame> ComputeGravitationalJerkOnMasslessBody(
      DegreesOfFreedom<Frame> const& degrees_of_freedom,
      Instant const& t,
      int subsystem = 0) const EXCLUDES(lock_);

  // Returns the gravitational jerk on the massive `body` at time `t`.  `body`
  // must be one of the bodies of this object.
  Vector<Jerk, Frame> ComputeGravitationalJerkOnMassiveBody(
      not_null<MassiveBody const*> body,
      Instant const& t) const EXCLUDES(lock_);

  // Same as above, but for multiple bodies.  The degrees of freedom must have
  // been precomputed by `EvaluateAllDegreesOfFreedom` at time `t`.
  std::vector<Vector<Jerk, Frame>> ComputeGravitationalJerkOnMassiveBodies(
      std::vector<not_null<MassiveBody const*>> const& bodies,
      BodiesToDegreesOfFreedom const& bodies_to_degrees_of_freedom,
      Instant const& t) const;

  // Returns the gravitational acceleration on a massless body located at the
  // given `position` at time `t`.  The position is represented relative to the
  // local origin of `subsystem`.
  virtual Vector<Acceleration, Frame>
  ComputeGravitationalAccelerationOnMasslessBody(
      Position<Frame> const& position,
      Instant const& t,
      int subsystem = 0) const EXCLUDES(lock_);

  // Returns the gravitational acceleration on the massless body having the
  // given `trajectory` at time `t`.  `t` must be one of the times of the
  // `trajectory`.  The positions of the `trajectory` are represented relative
  // to the local origin of `subsystem`.
  virtual Vector<Acceleration, Frame>
  ComputeGravitationalAccelerationOnMasslessBody(
      not_null<DiscreteTrajectory<Frame>*> trajectory,
      Instant const& t,
      int subsystem = 0) const EXCLUDES(lock_);

  // Returns the gravitational acceleration on the massive `body` at time `t`.
  // `body` must be one of the bodies of this object.
  virtual Vector<Acceleration, Frame>
  ComputeGravitationalAccelerationOnMassiveBody(
      not_null<MassiveBody const*> body,
      Instant const& t) const EXCLUDES(lock_);

  // Same as above, but for multiple bodies.  The positions must have been
  // precomputed by `EvaluateAllPositions`.  The client must ensure that the
  // evaluation was for time `t`.
  virtual std::vector<Vector<Acceleration, Frame>>
  ComputeGravitationalAccelerationOnMassiveBodies(
      std::vector<not_null<MassiveBody const*>> const& bodies,
      BodiesToPositions const& bodies_to_positions,
      Instant const& t) const;

  // Returns the potential at the given `position` at time `t`.  The position
  // is represented relative to the local origin of `subsystem`.
  SpecificEnergy ComputeGravitationalPotential(
      Position<Frame> const& position,
      Instant const& t,
      int subsystem = 0) const EXCLUDES(lock_);

  // Computes the apsides of the relative trajectory of `body1` and `body2`.
  // Appends to the given out parameters two points for each apsis, one for
  // `body1` and one for `body2`.  The times of `apoapsides1` and `apoapsides2`
  // are identical (are similarly for `periapsides1` and `periapsides2`).  The
  // relative trajectory is that of the true positions even when the bodies
  // belong to different subsystems, but the output degrees of freedom are
  // represented relative to each body's own subsystem origin.
  virtual void ComputeApsides(not_null<MassiveBody const*> body1,
                              not_null<MassiveBody const*> body2,
                              DistinguishedPoints<Frame>& apoapsides1,
                              DistinguishedPoints<Frame>& periapsides1,
                              DistinguishedPoints<Frame>& apoapsides2,
                              DistinguishedPoints<Frame>& periapsides2);

  // Returns the index of the given body in the serialization produced by
  // `WriteToMessage` and read by the `Read...` functions.  This index is not
  // suitable for other uses.
  virtual int serialization_index_for_body(
      not_null<MassiveBody const*> body) const;

  virtual not_null<MassiveBody const*> body_for_serialization_index(
      int serialization_index) const;

  virtual void WriteToMessage(
      not_null<serialization::Ephemeris*> message) const EXCLUDES(lock_);
  // The parameter `desired_t_min` indicates that the ephemeris must be restored
  // at a checkpoint such that, once the ephemeris is prolonged, its `t_min()`
  // is at or before `desired_t_min`.
  static not_null<std::unique_ptr<Ephemeris>> ReadFromMessage(
      Instant const& desired_t_min,
      serialization::Ephemeris const& message)
    requires serializable<Frame>;

 protected:
  // For mocking purposes, leaves everything uninitialized and uses the given
  // `integrator`.
  explicit Ephemeris(FixedStepSizeIntegrator<typename Ephemeris<
                         Frame>::NewtonianMotionEquation> const& integrator);

 private:
  // Checkpointing support.
  void WriteToCheckpointIfNeeded(Instant const& time) const
      ABSL_SHARED_LOCKS_REQUIRED(lock_);
  Checkpointer<serialization::Ephemeris>::Writer MakeCheckpointerWriter();
  Checkpointer<serialization::Ephemeris>::Reader MakeCheckpointerReader();

  // Called on a stoppable thread to reconstruct the past state of the ephemeris
  // and its trajectories starting in such a way that `t_min()` is at or before
  // `desired_t_min`.  The member variable `oldest_reanimated_checkpoint_` tells
  // the reanimator where to stop.
  absl::Status Reanimate(Instant const& desired_t_min) EXCLUDES(lock_);

  // Reconstructs the past state of the ephemeris between `t_initial` and
  // `t_final` using the given checkpoint `message`.
  absl::Status ReanimateOneCheckpoint(
      serialization::Ephemeris::Checkpoint const& message,
      Instant const& t_initial,
      Instant const& t_final) EXCLUDES(lock_);

  bool DesiredTMinReachedOrFullyReanimated(Instant const& desired_t_min)
      REQUIRES_SHARED(lock_);

  // Callbacks for the integrators.
  void AppendMassiveBodiesState(
      typename NewtonianMotionEquation::State const& state)
      REQUIRES(lock_);
  template<typename ContinuousTrajectoryPtr>
  static std::vector<absl::Status> AppendMassiveBodiesStateToTrajectories(
      typename NewtonianMotionEquation::State const& state,
      std::vector<not_null<ContinuousTrajectoryPtr>> const& trajectories);
  static void AppendMasslessBodiesStateToTrajectories(
      typename NewtonianMotionEquation::State const& state,
      std::vector<not_null<DiscreteTrajectory<Frame>*>> const& trajectories);

  // Returns an equation suitable for the massive bodies contained in this
  // ephemeris.
  NewtonianMotionEquation MakeMassiveBodiesNewtonianMotionEquation();

  Instant instance_time_locked() const REQUIRES_SHARED(lock_);

  virtual Instant t_min_locked() const REQUIRES_SHARED(lock_);
  virtual Instant t_max_locked() const REQUIRES_SHARED(lock_);

  // Returns the far-field damping applicable to the pair of massive bodies at
  // indices `b1` and `b2` in `bodies_`: the one, of the dampings of the two
  // bodies, whose outer threshold is farther.  Damping the pair as a whole (a
  // single σ applied to the actions of both bodies) preserves Newton's third
  // law; using the farther threshold ensures that a small body keeps feeling
  // a large one beyond its own threshold.  Requires `far_field_damping_` to
  // be non-empty.
  FarFieldDamping const& PairFarFieldDamping(std::size_t b1,
                                             std::size_t b2) const;

  // Computes the Jacobian of the acceleration field between one body, `body1`
  // (with index `b1` in the `positions` and `jacobians` arrays) and the bodies
  // `bodies2` (with indices [b2_begin, b2_end[ in the `bodies2`, `positions`
  // and `jacobians` arrays).  This assumes that the bodies are point masses
  // (that is, it doesn't take the geopotential into account).  The far-field
  // damping, if any, is applied to each pair (see `PairFarFieldDamping`).  The
  // positions are relative to the local origin of the subsystem of each body,
  // as described by `subsystem_of_body_`.
  template<typename MassiveBodyConstPtr>
  void ComputeJacobianByMassiveBodyOnMassiveBodies(
      Instant const& t,
      MassiveBody const& body1,
      std::size_t b1,
      std::vector<not_null<MassiveBodyConstPtr>> const& bodies2,
      std::size_t b2_begin,
      std::size_t b2_end,
      std::vector<Position<Frame>> const& positions,
      std::vector<JacobianOfAcceleration<Frame>>& jacobians) const;

  // Computes the jerk between one body, `body1` (with index `b1` in the
  // `degrees_of_freedom` and `jerks` arrays) and the bodies `bodies2` (with
  // indices [b2_begin, b2_end[ in the `bodies2`, `degrees_of_freedom` and
  // `jerks` arrays).  This assumes that the bodies are point masses
  // (that is, it doesn't take the geopotential into account).  The far-field
  // damping, if any, is applied to each pair (see `PairFarFieldDamping`).  The
  // positions are relative to the local origin of the subsystem of each body,
  // as described by `subsystem_of_body_`.
  template<typename MassiveBodyConstPtr>
  void ComputeGravitationalJerkByMassiveBodyOnMassiveBodies(
      Instant const& t,
      MassiveBody const& body1,
      std::size_t b1,
      std::vector<not_null<MassiveBodyConstPtr>> const& bodies2,
      std::size_t b2_begin,
      std::size_t b2_end,
      std::vector<DegreesOfFreedom<Frame>> const& degrees_of_freedom,
      std::vector<Vector<Jerk, Frame>>& jerks) const;

  // Returns the gravitational acceleration on the massive `body` at time `t`.
  // The `positions` must be for all the bodies in this object, in the order of
  // `bodies_` and must have been evaluated at time `t`.
  Vector<Acceleration, Frame>
  ComputeGravitationalAccelerationOnMassiveBody(
      not_null<MassiveBody const*> body,
      std::vector<Position<Frame>> const& positions,
      Instant const& t) const;

  // The implementation of the above, on which see
  // `ComputeGravitationalAccelerationByMassiveBodyOnMassiveBodies` regarding
  // `has_far_field_damping`.
  template<bool has_far_field_damping>
  Vector<Acceleration, Frame>
  ComputeGravitationalAccelerationOnMassiveBody(
      not_null<MassiveBody const*> body,
      std::vector<Position<Frame>> const& positions,
      Instant const& t) const;

  // Returns the gravitational jerk on the massive `body`.  The
  // `degrees_of_freedom` must be for all the bodies in this object, in the
  // order of `bodies_` and must have been evaluated at time `t`.
  Vector<Jerk, Frame>
  ComputeGravitationalJerkOnMassiveBody(
      not_null<MassiveBody const*> body,
      std::vector<DegreesOfFreedom<Frame>> const& degrees_of_freedom,
      Instant const& t) const;

  // Computes the accelerations between one body, `body1` (with index `b1` in
  // the `positions` and `accelerations` arrays) and the bodies `bodies2` (with
  // indices [b2_begin, b2_end[ in the `bodies2`, `positions` and
  // `accelerations` arrays).  The template parameters specify what we know
  // about the bodies, and therefore what forces apply.  Works for both owning
  // and non-owning pointers thanks to the `MassiveBodyConstPtr` template
  // parameter.  The positions are relative to the local origin of the
  // subsystem of each body, as described by `subsystem_of_body_`; if
  // `has_subsystems` is false the (then trivial) subsystem handling is
  // compiled out of the loop.  Similarly, if `has_far_field_damping` is false
  // the (then trivial) damping of each pair (see `PairFarFieldDamping`) is
  // compiled out, as a per-pair runtime check would tax this hot kernel.
  template<bool has_far_field_damping,
           bool has_subsystems,
           bool body1_is_oblate,
           bool body2_is_oblate,
           typename MassiveBodyConstPtr>
  void ComputeGravitationalAccelerationByMassiveBodyOnMassiveBodies(
      Instant const& t,
      MassiveBody const& body1,
      std::size_t b1,
      std::vector<not_null<MassiveBodyConstPtr>> const& bodies2,
      std::size_t b2_begin,
      std::size_t b2_end,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations,
      std::vector<Geopotential<Frame>> const& geopotentials) const;

  // Adds to `accelerations` the mutual gravitational acceleration of the pair
  // of massive bodies `b1` and `b2`.  `b1`'s μ, subsystem, position and
  // acceleration accumulator are passed already dereferenced, so the caller
  // can hoist them across a run of `b2`; `b2` indexes `bodies2`, `positions`,
  // `accelerations` and `geopotentials`.  `b1` and `b2` are indices into this
  // ephemeris' own arrays and must be distinct (a self-pair would divide by
  // zero).  This is the body of the loop of
  // `ComputeGravitationalAccelerationByMassiveBodyOnMassiveBodies`, factored
  // out so that a pair can be processed with the exact same arithmetic (and
  // hence the exact same far-field σ) whether it is reached by that dense loop
  // or by another caller.  Additions into `accelerations` are not associative,
  // so a caller that needs a result identical to the dense loop must visit the
  // pairs of each body in the same ascending order.  The template parameters
  // carry the same meaning as there.
  template<bool has_far_field_damping,
           bool has_subsystems,
           bool body1_is_oblate,
           bool body2_is_oblate,
           typename MassiveBodyConstPtr>
  void AddMassiveBodyPairGravitationalAcceleration(
      Instant const& t,
      std::size_t b1,
      std::size_t b2,
      GravitationalParameter const& μ1,
      int s1,
      Position<Frame> const& position_of_b1,
      Vector<Acceleration, Frame>& acceleration_on_b1,
      std::vector<not_null<MassiveBodyConstPtr>> const& bodies2,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations,
      std::vector<Geopotential<Frame>> const& geopotentials) const;

  // Computes the accelerations due to one body, `body1` (with index `b1` in the
  // `bodies_` and `trajectories_` arrays) on massless bodies at the given
  // `positions`.  The template parameters specify what we know about the
  // massive body, and therefore what forces apply; `has_far_field_damping` is
  // a template parameter (dispatched once per call by the caller) because a
  // per-pair runtime check would tax this hot kernel measurably when the
  // damping is off.  Returns an integer for efficiency.  The massless
  // positions are represented relative to the local origins of the
  // `subsystems`, which must be parallel to `positions`; where `anchors` is
  // set (it is either empty or parallel to `positions`), the representation is
  // further displaced by the anchor.
  template<bool has_far_field_damping, bool body1_is_oblate>
  std::underlying_type_t<absl::StatusCode>
  ComputeGravitationalAccelerationByMassiveBodyOnMasslessBodies(
      Instant const& t,
      MassiveBody const& body1,
      std::size_t b1,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations,
      std::vector<int> const& subsystems,
      std::vector<std::optional<Anchor>> const& anchors) const
      REQUIRES_SHARED(lock_);

  // Computes the potential resulting from one body, `body1` (with index `b1` in
  // the `bodies_` and `trajectories_` arrays) at the given `positions`.  The
  // template parameter specifies what we know about the massive body, and
  // therefore what potential applies.  The positions are represented relative
  // to the local origins of the `subsystems`, which must be parallel to
  // `positions`.
  template<bool body1_is_oblate>
  void ComputeGravitationalPotentialsOfMassiveBody(
      Instant const& t,
      MassiveBody const& body1,
      std::size_t b1,
      std::vector<Position<Frame>> const& positions,
      std::vector<SpecificEnergy>& potentials,
      std::vector<int> const& subsystems) const
      REQUIRES_SHARED(lock_);

  // Computes the accelerations between all the massive bodies in `bodies_`.
  absl::Status ComputeGravitationalAccelerationBetweenAllMassiveBodies(
      Instant const& t,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations) const;

  // The implementation of the above, on which see
  // `ComputeGravitationalAccelerationByMassiveBodyOnMassiveBodies` regarding
  // `has_far_field_damping` and `has_subsystems`.
  template<bool has_far_field_damping, bool has_subsystems>
  absl::Status ComputeGravitationalAccelerationBetweenAllMassiveBodies(
      Instant const& t,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations) const;

  // Computes the acceleration exerted by the massive bodies in `bodies_` on
  // massless bodies.  The massless bodies are at the given `positions`,
  // represented relative to the local origins of the `subsystems`, which must
  // be parallel to `positions`, and further displaced by the `anchors` where
  // set (`anchors` is either empty or parallel to `positions`).  Returns an
  // error iff a collision occurred, i.e., the massless body is inside one of
  // the `bodies_`.
  absl::StatusCode
  ComputeGravitationalAccelerationByAllMassiveBodiesOnMasslessBodies(
      Instant const& t,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations,
      std::vector<int> const& subsystems,
      std::vector<std::optional<Anchor>> const& anchors) const
      EXCLUDES(lock_);

  // The implementation of the above, on which see
  // `ComputeGravitationalAccelerationByMassiveBodyOnMasslessBodies` regarding
  // `has_far_field_damping`.
  template<bool has_far_field_damping>
  absl::StatusCode
  ComputeGravitationalAccelerationByAllMassiveBodiesOnMasslessBodies(
      Instant const& t,
      std::vector<Position<Frame>> const& positions,
      std::vector<Vector<Acceleration, Frame>>& accelerations,
      std::vector<int> const& subsystems,
      std::vector<std::optional<Anchor>> const& anchors) const
      EXCLUDES(lock_);

  // Computes the potential resulting from the massive bodies in `bodies_`.  The
  // potentials are computed at the given `positions`, represented relative to
  // the local origins of the `subsystems`, which must be parallel to
  // `positions`.
  void ComputeGravitationalPotentialsOfAllMassiveBodies(
      Instant const& t,
      std::vector<Position<Frame>> const& positions,
      std::vector<SpecificEnergy>& potentials,
      std::vector<int> const& subsystems) const
      EXCLUDES(lock_);

  // Flows the given ODE with an adaptive step integrator.
  template<typename ODE>
  absl::Status FlowODEWithAdaptiveStep(
      typename ODE::RightHandSideComputation compute_acceleration,
      not_null<DiscreteTrajectory<Frame>*> trajectory,
      Instant const& t,
      _integration_parameters::AdaptiveStepParameters<ODE> const& parameters,
      std::int64_t max_ephemeris_steps) EXCLUDES(lock_);

  // Fills `inter_subsystem_offsets_` from `subsystem_origin_offset_`.
  void ComputeInterSubsystemOffsets();

  // Computes an estimate of the ratio `tolerance / error`.
  static double ToleranceToErrorRatio(
      Length const& length_integration_tolerance,
      Speed const& speed_integration_tolerance,
      Time const& current_step_size,
      typename NewtonianMotionEquation::State const& /*state*/,
      typename NewtonianMotionEquation::State::Error const& error);

  // The bodies in the order in which they were given at construction.
  std::vector<not_null<MassiveBody const*>> unowned_bodies_;

  // The indices of bodies in `unowned_bodies_`.
  absl::flat_hash_map<not_null<MassiveBody const*>,
                      int> unowned_bodies_indices_;

  // The oblate bodies precede the spherical bodies in this vector.  The system
  // state is indexed in the same order.
  std::vector<not_null<std::unique_ptr<MassiveBody const>>> bodies_;

  // The indices of bodies in `bodies_`.
  absl::flat_hash_map<not_null<MassiveBody const*>, int> bodies_indices_;

  // Only has entries for the oblate bodies, at the same indices as `bodies_`.
  std::vector<Geopotential<Frame>> geopotentials_;

  // The indices in `bodies_` correspond to those in `trajectories_`.
  std::vector<not_null<ContinuousTrajectory<Frame>*>> trajectories_;

  absl::flat_hash_map<not_null<MassiveBody const*>,
                      not_null<std::unique_ptr<ContinuousTrajectory<Frame>>>>
      bodies_to_trajectories_;

  AccuracyParameters const accuracy_parameters_;
  FixedStepParameters const fixed_step_parameters_;

  int number_of_oblate_bodies_ = 0;
  int number_of_spherical_bodies_ = 0;

  // The subsystem of each body, parallel to `bodies_`.  All the entries are 0
  // unless subsystems were given at construction.
  std::vector<int> subsystem_of_body_;

  // For each subsystem, the displacement from the local origin of subsystem 0
  // to its local origin, on the canonical sector lattice.  Indexed by
  // subsystem; the entry at index 0 is zero.
  std::vector<SectorDisplacement<Frame>> subsystem_origin_offset_;

  // The pairwise differences of the entries of `subsystem_origin_offset_`,
  // precomputed for the gravity kernels; see `inter_subsystem_offset`.
  std::vector<SectorDisplacement<Frame>> inter_subsystem_offsets_;

  // The total gravitational parameter of the bodies of each subsystem.
  // Indexed by subsystem; empty unless subsystems were given at construction.
  std::vector<GravitationalParameter> subsystem_gravitational_parameter_;

  // The degrees of freedom of the barycentre of the bodies of each subsystem
  // at `subsystem_barycentre_time_`, the position being relative to the local
  // origin of the subsystem.  Indexed by subsystem; empty unless subsystems
  // were given at construction.
  std::vector<DegreesOfFreedom<Frame>> subsystem_barycentre_;

  // The time at which `subsystem_barycentre_` was computed.
  Instant subsystem_barycentre_time_;

  // The floor given at construction; zero if the far field is not damped.
  Acceleration far_field_damping_floor_;

  // The far-field damping of each body, parallel to `bodies_`.  Empty if the
  // far field is not damped.
  std::vector<FarFieldDamping> far_field_damping_;

  not_null<
      std::unique_ptr<Checkpointer<serialization::Ephemeris>>> checkpointer_;

  // An ephemeris that is constructed de novo won't ever need reanimation, so
  // all the checkpoints are animate at birth.
  Instant oldest_reanimated_checkpoint_ ABSL_GUARDED_BY(lock_) = InfinitePast;

  // The techniques and terminology follow [Lov22].
  RecurringThread<Instant> reanimator_;
  Clientele<Instant> reanimator_clientele_;

  // The fields above this line are fixed at construction and therefore not
  // protected.  Note that `ContinuousTrajectory` is thread-safe.  `lock_` is
  // also used to protect sections where the trajectories are not mutually
  // consistent (e.g., during Prolong).
  mutable absl::Mutex lock_;

  // Parameter passed to the last call to `RequestReanimation`, if any.
  std::optional<Instant> last_desired_t_min_ ABSL_GUARDED_BY(lock_);

  std::unique_ptr<typename Integrator<NewtonianMotionEquation>::Instance>
      instance_ ABSL_GUARDED_BY(lock_);

  absl::Status last_severe_integration_status_ ABSL_GUARDED_BY(lock_);
};

}  // namespace internal

using internal::ClusterSubsystems;
using internal::Ephemeris;

}  // namespace _ephemeris
}  // namespace physics
}  // namespace principia

#include "physics/ephemeris_body.hpp"
