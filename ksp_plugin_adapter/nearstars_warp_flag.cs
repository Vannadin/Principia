using System.Reflection;

namespace principia {
namespace ksp_plugin_adapter {

// Reads the NearStars warp flag over a stock channel—a `VesselModule` named
// `NearStarsWarpStatus` with a public boolean field `warpEngaged`, published
// by the NearStarsWarp plugin—so that Principia carries no compile-time
// dependency on NearStars.  A vessel flying under NearStars warp is moved by
// the NearStars cruise layer and must not be managed by Principia; on dropout
// the flag clears and the vessel is re-adopted from its stock orbit.
internal static class NearStarsWarpFlag {
  private const string module_type_name = "NearStarsWarpStatus";
  private const string field_name = "warpEngaged";

  public static bool IsUnderWarp(Vessel vessel) {
    if (module_absent_ || vessel.vesselModules == null) {
      return false;
    }
    foreach (VesselModule module in vessel.vesselModules) {
      if (module.GetType().Name == module_type_name) {
        // The field is looked up per concrete type, in case another mod ships
        // an unrelated module with the same short name.
        if (field_ == null ||
            !field_.DeclaringType.IsInstanceOfType(module)) {
          field_ = module.GetType().GetField(field_name);
          if (field_ == null) {
            Log.Error("Found " + module_type_name + " without a public " +
                      field_name + " field; ignoring the NearStars warp " +
                      "channel");
            module_absent_ = true;
            return false;
          }
          Log.Info("NearStars warp channel found (" +
                   module.GetType().AssemblyQualifiedName + ")");
        }
        return field_.GetValue(module) is bool warp_engaged && warp_engaged;
      }
    }
    // Vessel modules are instantiated on every vessel for every loaded
    // assembly, so a vessel that has its stock modules but no NearStars one
    // means that NearStars is not installed.  An empty list merely means that
    // the vessel is too young to have been given its modules.
    if (vessel.vesselModules.Count > 0) {
      module_absent_ = true;
    }
    return false;
  }

  private static bool module_absent_ = false;
  private static FieldInfo field_;
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
