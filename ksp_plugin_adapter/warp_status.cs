namespace principia {
namespace ksp_plugin_adapter {

// The warp release channel.  This module is instantiated by KSP on every
// vessel; an FTL/warp mod that moves a vessel itself (Blueshift, KSP
// Interstellar Extended, …) declares so by setting `warpEngaged` every
// FixedUpdate while the cruise lasts—directly or by name over reflection,
// without compiling against Principia.  While the flag is up, Principia
// reports the vessel as unmanageable and releases it; when it clears, the
// vessel is re-adopted from its stock orbit around the destination through
// the existing first-insertion path.
// The flag is a dead man's switch: it decays after `grace_period` without
// reassertion, so a warp mod that was uninstalled or broke mid-cruise cannot
// strand a released vessel.  It survives save/load with a fresh grace, so
// that a save written mid-cruise is not re-adopted before the warp mod
// reasserts itself.
public class PrincipiaWarpStatus : VesselModule {
  // Unity physics-clock seconds without reassertion after which the flag
  // clears itself; the physics clock does not advance with timewarp, so this
  // is unaffected by the warp rate.
  private const double grace_period = 10;

  public bool warpEngaged {
    get {
      if (engaged_ &&
          UnityEngine.Time.fixedTime - last_asserted_ > grace_period) {
        engaged_ = false;
      }
      return engaged_;
    }
    set {
      engaged_ = value;
      last_asserted_ = UnityEngine.Time.fixedTime;
      if (value) {
        last_asserted_any_ = last_asserted_;
      }
    }
  }

  // Whether any vessel's flag was asserted within the grace period.  The
  // unmanageability census asks every vessel every frame; when no warp mod
  // is asserting at all—the common case—this lets the census skip the
  // per-vessel module scan wholesale.  It decays exactly like the per-vessel
  // flag: when this is false, every per-vessel flag has decayed too.
  public static bool any_recently_engaged =>
      UnityEngine.Time.fixedTime - last_asserted_any_ <= grace_period;

  protected override void OnSave(ConfigNode node) {
    base.OnSave(node);
    node.SetValue("warp_engaged", warpEngaged, createIfNotFound : true);
  }

  protected override void OnLoad(ConfigNode node) {
    base.OnLoad(node);
    bool engaged = false;
    node.TryGetValue("warp_engaged", ref engaged);
    if (engaged) {
      warpEngaged = true;
    }
  }

  private bool engaged_ = false;
  private double last_asserted_;
  private static double last_asserted_any_ = double.NegativeInfinity;
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
