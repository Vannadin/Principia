namespace principia {
namespace ksp_plugin_adapter {

// The warp release channel.  This module is instantiated by KSP on every
// vessel; an FTL/warp mod that moves a vessel itself (Blueshift, KSP
// Interstellar Extended, …) declares so by setting `warpEngaged`—directly or
// by name over reflection, without compiling against Principia.  While the
// flag is up, Principia reports the vessel as unmanageable and releases it;
// when it clears, the vessel is re-adopted from its stock orbit around the
// destination through the existing first-insertion path.  The flag survives
// save/load, so that a save written mid-cruise does not get its vessel
// re-adopted before the warp mod reasserts itself.
public class PrincipiaWarpStatus : VesselModule {
  public bool warpEngaged = false;

  protected override void OnSave(ConfigNode node) {
    base.OnSave(node);
    node.SetValue("warp_engaged", warpEngaged, createIfNotFound : true);
  }

  protected override void OnLoad(ConfigNode node) {
    base.OnLoad(node);
    node.TryGetValue("warp_engaged", ref warpEngaged);
  }
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
