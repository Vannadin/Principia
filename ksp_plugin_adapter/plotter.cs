using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace principia {
namespace ksp_plugin_adapter {

class Plotter {
  public Plotter(PrincipiaPluginAdapter adapter) {
    adapter_ = adapter;
  }

  public static double TanAngularResolution() {
    const double degree = Math.PI / 180;
    UnityEngine.Camera camera = PlanetariumCamera.Camera;
    float vertical_fov = camera.fieldOfView;
    float horizontal_fov =
        UnityEngine.Camera.VerticalToHorizontalFieldOfView(
            vertical_fov,
            camera.aspect);
    // The angle subtended by the pixel closest to the centre of the viewport.
    return Math.Min(
               Math.Tan(vertical_fov * degree / 2) / (camera.pixelHeight / 2),
               Math.Tan(horizontal_fov * degree / 2) /
               (camera.pixelWidth / 2));
  }

  public void PlotEquipotentials(DisposablePlanetarium planetarium) {
    int number_of_equipotentials = Plugin.EquipotentialCount();
    if (number_of_equipotentials == 0) {
      return;
    }
    for (int i = equipotential_meshes_.Count;
         i < number_of_equipotentials;
         ++i) {
      equipotential_meshes_.Add(MakeDynamicMesh());
    }
    var colour = adapter_.plotting_frame_selector_.Primary()
        .orbitDriver?.Renderer?.orbitColor ?? XKCDColors.SunshineYellow;
    CelestialBody equipotential_reference =
        adapter_.plotting_frame_selector_.Centre() ??
        adapter_.plotting_frame_selector_.Primary();
    for (int i = 0; i < number_of_equipotentials; ++i) {
      planetarium.PlanetariumPlotEquipotential(
          Plugin,
          i,
          VertexBuffer.data,
          VertexBuffer.size,
          out int vertex_count,
          out XYZ anchor);
      DrawLineMesh(equipotential_meshes_[i], vertex_count, anchor, colour,
                   GLLines.Style.Solid,
                   /*registration_reference_world=*/null,
                   equipotential_reference?.position);
    }
  }

  public void PlotTrajectories(DisposablePlanetarium planetarium,
                               string main_vessel_guid,
                               double history_length,
                               double? prediction_t_max,
                               double? flight_plan_t_max) {
    PlotCelestialTrajectories(planetarium, main_vessel_guid, history_length);
    PlotVesselTrajectories(planetarium, main_vessel_guid, history_length,
                           prediction_t_max, flight_plan_t_max);
  }

  private void PlotVesselTrajectories(DisposablePlanetarium planetarium,
                                      string main_vessel_guid,
                                      double history_length,
                                      double? prediction_t_max,
                                      double? flight_plan_t_max) {
    if (main_vessel_guid != null) {
      Vessel main_vessel =
          FlightGlobals.FindVessel(new Guid(main_vessel_guid));
      Vector3d? main_reference = main_vessel?.GetWorldPos3D();
      {
        planetarium.PlanetariumPlotPsychohistory(
            Plugin,
            main_vessel_guid,
            history_length,
            prediction_t_max,
            VertexBuffer.data,
            VertexBuffer.size,
            out int vertex_count,
            out XYZ anchor);
        DrawLineMesh(ref psychohistory_mesh_,
                     vertex_count,
                     anchor,
                     adapter_.history_colour,
                     adapter_.history_style,
                     main_reference);
      }
      {
        planetarium.PlanetariumPlotPrediction(Plugin,
                                              main_vessel_guid,
                                              prediction_t_max,
                                              VertexBuffer.data,
                                              VertexBuffer.size,
                                              out int vertex_count,
                                              out XYZ anchor);
        DrawLineMesh(ref prediction_mesh_,
                     vertex_count,
                     anchor,
                     adapter_.prediction_colour,
                     adapter_.prediction_style,
                     main_reference);
      }

      // Main vessel flight plan.
      if (Plugin.FlightPlanExists(main_vessel_guid)) {
        int number_of_segments =
            Plugin.FlightPlanNumberOfSegments(main_vessel_guid);
        for (int i = flight_plan_segment_meshes_.Count;
             i < number_of_segments;
             ++i) {
          flight_plan_segment_meshes_.Add(MakeDynamicMesh());
        }
        for (int i = 0; i < number_of_segments; ++i) {
          bool is_burn = i % 2 == 1;
          var colour = is_burn
                           ? adapter_.burn_colour
                           : adapter_.flight_plan_colour;
          planetarium.PlanetariumPlotFlightPlanSegment(
              Plugin,
              main_vessel_guid,
              i,
              flight_plan_t_max,
              VertexBuffer.data,
              VertexBuffer.size,
              out int vertex_count,
              out XYZ anchor);
          // No need for dynamic initialization, that was done above.
          DrawLineMesh(flight_plan_segment_meshes_[i],
                       vertex_count,
                       anchor,
                       colour,
                       is_burn
                           ? adapter_.burn_style
                           : adapter_.flight_plan_style,
                       main_reference);
        }
      }
    }

    // Target psychohistory and prediction.
    Vessel target_vessel = FlightGlobals.fetch.VesselTarget?.GetVessel();
    string target_id = target_vessel?.id.ToString();
    if (FlightGlobals.ActiveVessel != null &&
        !adapter_.plotting_frame_selector_.target_frame_selected &&
        target_id != null &&
        Plugin.HasVessel(target_id)) {
      Vector3d target_reference = target_vessel.GetWorldPos3D();
      {
        planetarium.PlanetariumPlotPsychohistory(
            Plugin,
            target_id,
            history_length,
            t_max: null,
            VertexBuffer.data,
            VertexBuffer.size,
            out int vertex_count,
            out XYZ anchor);
        DrawLineMesh(ref target_psychohistory_mesh_,
                     vertex_count,
                     anchor,
                     adapter_.target_history_colour,
                     adapter_.target_history_style,
                     target_reference);
      }
      {
        planetarium.PlanetariumPlotPrediction(
            Plugin,
            target_id,
            t_max: null,
            VertexBuffer.data,
            VertexBuffer.size,
            out int vertex_count,
            out XYZ anchor);
        DrawLineMesh(ref target_prediction_mesh_,
                     vertex_count,
                     anchor,
                     adapter_.target_prediction_colour,
                     adapter_.target_prediction_style,
                     target_reference);
      }
    }
  }

  private void PlotCelestialTrajectories(DisposablePlanetarium planetarium,
                                         string main_vessel_guid,
                                         double history_length) {
    PlotSubtreeTrajectories(planetarium, main_vessel_guid, history_length,
                            Planetarium.fetch.Sun, TanAngularResolution());
  }

  // Plots the trajectories of `root` and its natural satellites.
  private void PlotSubtreeTrajectories(DisposablePlanetarium planetarium,
                                       string main_vessel_guid,
                                       double history_length,
                                       CelestialBody root,
                                       double tan_angular_resolution) {
    if (!celestial_trajectory_meshes_.TryGetValue(
            root,
            out CelestialTrajectories trajectories)) {
      trajectories = celestial_trajectory_meshes_[root] =
          new CelestialTrajectories();
    }
    var colour = root.orbitDriver?.Renderer?.orbitColor ??
        XKCDColors.SunshineYellow;
    var camera_world_position = ScaledSpace.ScaledToLocalSpace(
        PlanetariumCamera.fetch.transform.position);
    double min_distance_from_camera =
        (root.position - camera_world_position).magnitude;
    if (!adapter_.plotting_frame_selector_.FixesBody(root) &&
        adapter_.show_celestial_trajectory(root)) {
      {
        planetarium.PlanetariumPlotCelestialPastTrajectory(
            Plugin,
            root.flightGlobalsIndex,
            history_length,
            VertexBuffer.data,
            VertexBuffer.size,
            out double min_past_distance,
            out int vertex_count,
            out XYZ anchor);
        min_distance_from_camera =
            Math.Min(min_distance_from_camera, min_past_distance);
        DrawLineMesh(ref trajectories.past,
                     vertex_count,
                     anchor,
                     colour,
                     GLLines.Style.Faded,
                     root.position);
      }

      if (main_vessel_guid != null) {
        planetarium.PlanetariumPlotCelestialFutureTrajectory(
            Plugin,
            root.flightGlobalsIndex,
            main_vessel_guid,
            VertexBuffer.data,
            VertexBuffer.size,
            out double min_future_distance,
            out int vertex_count,
            out XYZ anchor);
        min_distance_from_camera =
            Math.Min(min_distance_from_camera, min_future_distance);
        DrawLineMesh(ref trajectories.future,
                     vertex_count,
                     anchor,
                     colour,
                     GLLines.Style.Solid,
                     root.position);
      }
    }
    foreach (CelestialBody child in root.orbitingBodies) {
      // Plot the trajectory of an orbiting body if it could be separated from
      // that of its parent by a pixel of empty space, instead of merely making
      // the line wider; but always traverse the subtree if the current body is
      // hidden.
      if (!adapter_.show_celestial_trajectory(root) ||
          child.orbit.ApR / min_distance_from_camera >
              2 * tan_angular_resolution) {
        PlotSubtreeTrajectories(planetarium, main_vessel_guid, history_length,
                                child, tan_angular_resolution);
      }
    }
  }

  // An anchored mesh is emitted relative to the plot's own reference, which
  // is the plotted vessel at the present — the very position the scene draws
  // its icon at.  Drawing the mesh at the scene's own mapping of that
  // position registers the two exactly: neither side has to express the
  // reference in scaled space across the interstellar distance to the
  // plotting frame's origin, where that expression would round to a
  // kilometre and shift the whole line off the icon.
  private static Vector3d SceneReferenceTranslation(Vector3d reference_world,
                                                   XYZ camera_from_reference) {
    return (Vector3d)ScaledSpace.LocalToScaledSpace(reference_world) +
           (Vector3d)camera_from_reference;
  }

  // The scene's world-to-scaled transform and the affine map given to the
  // planetarium disagree by the float32 rounding of the transform state —
  // ~1e8 m at interstellar magnitudes.  Rebasing a mesh by this correction,
  // evaluated at a reference near its geometry, draws it through the scene's
  // own mapping there, so the disagreement is common-mode with the icons and
  // sprites the lines are compared against.  This is the convention of the
  // plots that carry no registration, whose anchor is the camera's absolute
  // scaled-space position.
  private static Vector3d SceneMappingCorrection(Vector3d reference_world) {
    return (Vector3d)ScaledSpace.LocalToScaledSpace(reference_world) -
           (reference_world - GLLines.current_scaled_space_origin) *
               ScaledSpace.InverseScaleFactor;
  }

  private void DrawLineMesh(ref UnityEngine.Mesh mesh,
                            int vertex_count,
                            XYZ anchor,
                            UnityEngine.Color colour,
                            GLLines.Style style,
                            Vector3d? registration_reference_world = null,
                            Vector3d? correction_reference_world = null) {
    // Construct the mesh on the first call because Unity doesn't want us to do
    // that at construction.
    if (mesh == null) {
      mesh = MakeDynamicMesh();
    }
    DrawLineMesh(mesh, vertex_count, anchor, colour, style,
                 registration_reference_world, correction_reference_world);
  }

  private void DrawLineMesh(UnityEngine.Mesh mesh,
                            int vertex_count,
                            XYZ anchor,
                            UnityEngine.Color colour,
                            GLLines.Style style,
                            Vector3d? registration_reference_world = null,
                            Vector3d? correction_reference_world = null) {
    if (vertex_count > VertexBuffer.size) {
      Log.Fatal("Trying to draw " +
                vertex_count +
                " vertices, maximum is " +
                VertexBuffer.size);
    }

    mesh.vertices = VertexBuffer.vertices;
    int index_count = style == GLLines.Style.Dashed ? vertex_count & ~1
                                                    : vertex_count;

    if (indices_ == null) {
      indices_ = new int[VertexBuffer.size];
      for (int i = 0; i < indices_.Length; ++i) {
        indices_[i] = i;
      }
    }

    if (style == GLLines.Style.Faded) {
      for (int i = 0; i < vertex_count; ++i) {
        var faded_colour = colour;
        // Fade from the opacity of `colour` (when i = 0) down to 20% of that
        // opacity.
        faded_colour.a *= 1 - 0.8f * (i / (float)vertex_count);
        colours_[i] = faded_colour;
      }
    } else {
      for (int i = 0; i < vertex_count; ++i) {
        colours_[i] = colour;
      }
    }

    mesh.colors = colours_;
    mesh.SetIndices(indices_,
                    indicesStart: 0,
                    indicesLength: index_count,
                    style == GLLines.Style.Dashed
                        ? UnityEngine.MeshTopology.Lines
                        : UnityEngine.MeshTopology.LineStrip,
                    submesh: 0);
    mesh.RecalculateBounds();
    // The vertices are relative to the camera, which bounds their float
    // rounding by the ULP of their distance from it, angularly sub-pixel from
    // any viewpoint.  An anchored plot reports the camera as a displacement
    // from the plot's own reference — the plotted vessel at the present, the
    // very position the scene draws its icon at — so drawing at the scene's
    // mapping of that position, plus the displacement — the camera's own
    // scene position — reassembles the plot through the scene's own
    // arithmetic: the camera term cancels and neither
    // side ever expresses a point in scaled space across the distance to the
    // plotting frame's origin, where it would round to a kilometre.  A zero
    // anchor is the stock bit-identical path, left untouched.
    Vector3d translation = (Vector3d)anchor;
    if (anchor.x != 0 || anchor.y != 0 || anchor.z != 0) {
      if (registration_reference_world.HasValue) {
        // The anchor is the camera as a displacement from the plot's own
        // registration point; the scene's mapping of that point plus the
        // displacement is the camera's own scene position.
        translation = SceneReferenceTranslation(
            registration_reference_world.Value, anchor);
      } else if (correction_reference_world.HasValue) {
        // The anchor is the camera's absolute scaled-space position.
        translation += SceneMappingCorrection(
            correction_reference_world.Value);
      } else {
        // A nonzero anchor is meaningless without the reference it was
        // reported against; drawing it raw would put the mesh at the
        // scaled-space origin.
        return;
      }
    }
    // If the lines are drawn in layer 31 (Vectors), which sounds more
    // appropriate, they vanish when zoomed out.  Layer 9 works; pay no
    // attention to its name.
    UnityEngine.Graphics.DrawMesh(
        mesh,
        (UnityEngine.Vector3)translation,
        UnityEngine.Quaternion.identity,
        GLLines.line_material,
        (int)PrincipiaPluginAdapter.UnityLayers.Atmosphere,
        PlanetariumCamera.Camera);
  }

  private static UnityEngine.Mesh MakeDynamicMesh() {
    var result = new UnityEngine.Mesh();
    result.MarkDynamic();
    return result;
  }

  private readonly PrincipiaPluginAdapter adapter_;

  private IntPtr Plugin => adapter_.Plugin();

  private static class VertexBuffer {
    public static IntPtr data => handle_.AddrOfPinnedObject();
    public static int size => vertices_.Length;

    public static UnityEngine.Vector3[] vertices => vertices_;

    private static readonly UnityEngine.Vector3[] vertices_ =
        new UnityEngine.Vector3[10_000];
    private static GCHandle handle_ =
        GCHandle.Alloc(vertices_, GCHandleType.Pinned);
  }

  private class CelestialTrajectories {
    public UnityEngine.Mesh future = MakeDynamicMesh();
    public UnityEngine.Mesh past = MakeDynamicMesh();
  }

  private readonly Dictionary<CelestialBody, CelestialTrajectories>
      celestial_trajectory_meshes_ =
      new Dictionary<CelestialBody, CelestialTrajectories>();
  private UnityEngine.Mesh psychohistory_mesh_;
  private UnityEngine.Mesh prediction_mesh_;
  private readonly List<UnityEngine.Mesh> flight_plan_segment_meshes_ =
      new List<UnityEngine.Mesh>();
  private readonly List<UnityEngine.Mesh> equipotential_meshes_ =
      new List<UnityEngine.Mesh>();
  private UnityEngine.Mesh target_psychohistory_mesh_;
  private UnityEngine.Mesh target_prediction_mesh_;
  private int[] indices_ = null;
  private UnityEngine.Color[] colours_ =
      new UnityEngine.Color[VertexBuffer.size];
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
