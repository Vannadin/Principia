using System;
using System.Collections.Generic;

namespace principia {
namespace ksp_plugin_adapter {

// Thrust under timewarp: while the active vessel is packed, stock KSP runs no
// physics, so the force census in `FashionablyLate` sees nothing and the
// vessel coasts.  This class harvests the ignited engines of the packed active
// vessel every frame and hands them to the plugin as an on-rails burn, which
// the catch-up integration applies to the vessel history with Tsiolkovsky mass
// depletion.  The burn is one-shot on the C++ side (consumed by every
// catch-up, even one with nothing to do), so `HandleWarpFrame` must run each
// frame BEFORE the catch-up futures are spawned in `Precalc`—the catch-up
// that actually advances a packed vessel.
// The adapter owns the mass bookkeeping (part masses are stale on the C++ side
// while packed) as well as the propellant drain, and stops the warp when the
// burn cannot proceed: propellant exhaustion or starvation, engine torque
// exceeding the attitude-control authority, or lack of electric charge.
internal class OnRailsBurner {
  // Fraction of the attitude-control authority that the net engine torque may
  // use up before the burn is deemed uncontrollable.
  private const double torque_authority_margin = 0.5;

  // Whether on-rails burns are enabled, from the `principia_flags` config
  // (`on_rails_burns = true`); off by default.
  public static bool enabled {
    get {
      enabled_ ??= GameDatabase.Instance.
          GetAtMostOneNode(PrincipiaPluginAdapter.principia_flags)?.
          GetAtMostOneValue("on_rails_burns") == "true";
      return enabled_.Value;
    }
  }

  // Invalidates the cached flag, so that it is reread after the game database
  // reloads; called where the plugin reloads the C++-side flags.
  public static void InvalidateFlagCache() {
    enabled_ = null;
  }

  // Rearms the warp-stop message; called on the frames where the burn branch
  // is not taken (vessel unpacked, warp dropped), so that the next incident
  // is reported once again.
  public void ResetWarpStopMessageLatch() {
    warp_stop_message_latched_ = false;
  }

  public void HandleWarpFrame(IntPtr plugin,
                              Vessel vessel,
                              string vessel_guid,
                              double Δt) {
    double throttle = FlightInputHandler.state.mainThrottle;

    // The true current mass and centre of mass of the packed vessel.
    double vessel_mass = 0;  // t.
    Vector3d centre_of_mass = Vector3d.zero;  // World.
    foreach (Part part in vessel.parts) {
      double part_mass = part.mass + part.GetResourceMass();
      vessel_mass += part_mass;
      centre_of_mass +=
          part_mass *
          (Vector3d)part.partTransform.TransformPoint(part.CoMOffset);
    }
    if (vessel_mass <= 0) {
      return;
    }
    centre_of_mass /= vessel_mass;

    // Harvest the ignited engines at the current throttle.  A packed vessel
    // under warp is on rails, effectively in vacuum, so vacuum thrust and
    // specific impulse are used throughout.
    double total_thrust = 0;  // Scalar sum, kN.
    double total_mass_flow = 0;  // t/s.
    Vector3d thrust = Vector3d.zero;  // World, kN.
    Vector3d thrust_torque = Vector3d.zero;  // World, kN m, about the CoM.
    // Drains are kept per (engine part, propellant): NO_FLOW propellants
    // (solid fuel) can only be drawn from the part that burns them, and the
    // flowing modes reach the same tanks from any engine part anyway.
    // The containers are fields reused across frames—a warp burn runs this
    // every frame—so they are cleared rather than reallocated.
    drains_.Clear();
    drain_rates_.Clear();
    foreach (Part part in vessel.parts) {
      foreach (PartModule module in part.Modules) {
        if (!(module is ModuleEngines engine) ||
            !engine.EngineIgnited ||
            !engine.isOperational) {
          continue;
        }
        double engine_throttle = engine.throttleLocked
            ? 1
            : engine.independentThrottle
                  ? engine.independentThrottlePercentage / 100
                  : throttle;
        double engine_thrust = engine.maxThrust *
                               (engine.thrustPercentage / 100) *
                               engine_throttle;
        double vacuum_specific_impulse = engine.atmosphereCurve.Evaluate(0);
        if (engine_thrust <= 0 || vacuum_specific_impulse <= 0) {
          continue;
        }
        // The mixture density converts the engine mass flow to per-propellant
        // unit rates using KSP's ratio bookkeeping; massless resources
        // (electric charge) drain by ratio without contributing to the
        // density, as in stock.  An engine whose whole mixture is massless
        // cannot deplete mass at its stated specific impulse: it is excluded
        // from the burn rather than granted free, undrained thrust.
        double mixture_density = 0;  // t per unit.
        foreach (Propellant propellant in engine.propellants) {
          mixture_density += propellant.ratio *
                             PartResourceLibrary.Instance.
                                 GetDefinition(propellant.id).density;
        }
        if (mixture_density <= 0) {
          continue;
        }
        // kN / (m/s) = t/s.
        double engine_mass_flow =
            engine_thrust / (vacuum_specific_impulse * engine.g);
        int transform_count = engine.thrustTransforms.Count;
        for (int i = 0; i < transform_count; ++i) {
          UnityEngine.Transform thrust_transform = engine.thrustTransforms[i];
          double multiplier =
              engine.thrustTransformMultipliers.Count == transform_count
                  ? engine.thrustTransformMultipliers[i]
                  : 1.0 / transform_count;
          // KSP thrust transforms point exhaust-ward.
          Vector3d force = -(Vector3d)thrust_transform.forward *
                           (engine_thrust * multiplier);
          thrust += force;
          thrust_torque += Vector3d.Cross(
              (Vector3d)thrust_transform.position - centre_of_mass,
              force);
        }
        double unit_flow = engine_mass_flow / mixture_density;  // units/s.
        foreach (Propellant propellant in engine.propellants) {
          double rate = unit_flow * propellant.ratio;
          drains_.Add(new PropellantDrain{
              part = part,
              resource_id = propellant.id,
              rate = rate,
              flow_mode = propellant.GetFlowMode()});
          drain_rates_.TryGetValue(propellant.id, out double total_rate);
          drain_rates_[propellant.id] = total_rate + rate;
        }
        total_thrust += engine_thrust;
        total_mass_flow += engine_mass_flow;
      }
    }

    // The burn is applied along a single direction, so the net force is the
    // vector sum: canted engines must not get their cancelled components
    // back.  The mass flow, in contrast, is the full scalar bookkeeping.
    double net_thrust = thrust.magnitude;  // kN.
    if (total_thrust <= 0 || total_mass_flow <= 0 ||
        net_thrust <= 1e-6 * total_thrust) {
      // Engines idle, or exactly cancelling.  Not setting the burn again is
      // enough to cut it: it is consumed by every catch-up.
      warp_stop_message_latched_ = false;
      return;
    }

    // Propellant availability bounds the burn duration; the C++ side cuts the
    // thrust at exactly that time, even in mid-step.  The vessel-wide totals
    // ignore the flow topology; the drain below detects starvation (reachable
    // amount less than the totals suggest) and stops the warp a frame later.
    double max_duration = double.PositiveInfinity;  // s.
    bool propellant_depleted = false;
    foreach (var drain_rate in drain_rates_) {
      double rate = drain_rate.Value;
      if (rate <= 0) {
        continue;
      }
      vessel.GetConnectedResourceTotals(drain_rate.Key,
                                        out double available,
                                        out double _);
      if (available <= 1e-9) {
        propellant_depleted = true;
      }
      max_duration = Math.Min(max_duration, available / rate);
    }
    if (propellant_depleted) {
      StopWarp("propellant depleted");
      return;
    }

    // The burn is only physical if the vessel could actually hold its
    // attitude: the net engine torque about the centre of mass must lie
    // within the attitude-control authority, and control must be powered.
    Vector3d control_authority = Vector3d.zero;  // Vessel axes, kN m.
    foreach (Part part in vessel.parts) {
      foreach (PartModule module in part.Modules) {
        if (module is ITorqueProvider torque_provider) {
          torque_provider.GetPotentialTorque(
              out UnityEngine.Vector3 positive,
              out UnityEngine.Vector3 negative);
          control_authority += new Vector3d(
              Math.Max(Math.Abs(positive.x), Math.Abs(negative.x)),
              Math.Max(Math.Abs(positive.y), Math.Abs(negative.y)),
              Math.Max(Math.Abs(positive.z), Math.Abs(negative.z)));
        }
      }
    }
    Vector3d vessel_torque =
        UnityEngine.QuaternionD.Inverse(
            (UnityEngine.QuaternionD)vessel.ReferenceTransform.rotation) *
        thrust_torque;
    // Allow a 1 mm effective lever arm of numerical tolerance, so that
    // perfectly balanced engines pass on a vessel with no attitude control.
    double tolerance = 1e-3 * total_thrust;
    if (Math.Abs(vessel_torque.x) >
            torque_authority_margin * control_authority.x + tolerance ||
        Math.Abs(vessel_torque.y) >
            torque_authority_margin * control_authority.y + tolerance ||
        Math.Abs(vessel_torque.z) >
            torque_authority_margin * control_authority.z + tolerance) {
      StopWarp("engine torque exceeds attitude control authority");
      return;
    }
    // Presence-only: stock `ModuleCommand.FixedUpdate` already drains command
    // electric charge on rails, so draining it here again would double-count;
    // the power cost of holding attitude with reaction wheels is not
    // modelled while packed.
    vessel.GetConnectedResourceTotals(PartResourceLibrary.ElectricityHashcode,
                                      out double electric_charge,
                                      out double _);
    if (electric_charge <= 1e-9) {
      StopWarp("out of electric charge");
      return;
    }

    // The C++ side clears a burn whose direction degenerates to zero (e.g.
    // prograde at rest, target at contact) without applying any thrust;
    // mirror that rejection here, so that no propellant is drained for a
    // frame that cannot burn.
    Vector3d direction = CommandedDirection(vessel, thrust);
    if (direction.sqrMagnitude == 0) {
      warp_stop_message_latched_ = false;
      return;
    }
    plugin.VesselSetOnRailsBurn(
        vessel_guid,
        net_thrust,
        net_thrust / total_mass_flow / 9.80665,
        vessel_mass,
        (XYZ)direction,
        max_duration);
    warp_stop_message_latched_ = false;

    // Drain the tanks for this frame; stock drains nothing while packed (and
    // conversely this branch never runs when the vessel is unpacked and stock
    // drains, so there is no double drain).  `Δt` is already warp-stretched.
    // A shortfall means the flow topology cannot actually feed the engines
    // (empty booster, crossfeed-blocked or locked tanks): stop the warp so
    // that the error stays bounded by this frame's burn.
    double burn_time = Math.Min(Δt, max_duration);
    bool starved = false;
    foreach (PropellantDrain drain in drains_) {
      double demand = drain.rate * burn_time;
      if (demand > 0) {
        double obtained = drain.part.RequestResource(drain.resource_id,
                                                     demand,
                                                     drain.flow_mode);
        starved |= obtained < 0.99 * demand;
      }
    }
    if (starved) {
      StopWarp("propellant starved (empty booster or blocked crossfeed)");
    }
  }

  // The burn direction is commanded, not attitude-derived (the residual spin
  // of the packed vessel must not steer a month-long burn): the guidance modes
  // are recomputed every frame from the current dynamical state—prograde is
  // the stock main-body-relative velocity—and the default holds the net
  // engine thrust axis as it is in World at this frame.
  private static Vector3d CommandedDirection(Vessel vessel,
                                             Vector3d engine_thrust) {
    ITargetable target = FlightGlobals.fetch.VesselTarget;
    Vector3d to_target = target == null
        ? Vector3d.zero
        : ((Vector3d)target.GetTransform().position -
           (Vector3d)vessel.ReferenceTransform.position).normalized;
    switch (vessel.Autopilot?.Mode) {
      case VesselAutopilot.AutopilotMode.Prograde:
        return vessel.obt_velocity.normalized;
      case VesselAutopilot.AutopilotMode.Retrograde:
        return -vessel.obt_velocity.normalized;
      case VesselAutopilot.AutopilotMode.Target when target != null:
        return to_target;
      case VesselAutopilot.AutopilotMode.AntiTarget when target != null:
        return -to_target;
      default:
        return engine_thrust;
    }
  }

  private void StopWarp(string reason) {
    // The burn is not rearmed after this frame, so the next catch-up coasts;
    // this method only reports and halts the warp.  The message is latched so
    // that it is posted once per crossing rather than every frame.
    TimeWarp.fetch.CancelAutoWarp();
    TimeWarp.SetRate(0, instant : true);
    if (!warp_stop_message_latched_) {
      warp_stop_message_latched_ = true;
      ScreenMessages.PostScreenMessage(
          "[Principia] Engine burn interrupted: " + reason,
          5f,
          ScreenMessageStyle.UPPER_CENTER);
    }
  }

  private struct PropellantDrain {
    public Part part;
    public int resource_id;
    public double rate;  // units/s.
    public ResourceFlowMode flow_mode;
  }

  // Reused across frames to avoid per-frame heap allocation during warp;
  // cleared at the start of `HandleWarpFrame`.
  private readonly List<PropellantDrain> drains_ = new List<PropellantDrain>();
  private readonly Dictionary<int, double> drain_rates_ =
      new Dictionary<int, double>();

  private static bool? enabled_;
  private bool warp_stop_message_latched_ = false;
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
