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
    for (int i = 0; i < number_of_equipotentials; ++i) {
      planetarium.PlanetariumPlotEquipotential(
          Plugin,
          i,
          VertexBuffer.data,
          VertexBuffer.size,
          out int vertex_count,
          out XYZ anchor);
      DrawLineMesh(equipotential_meshes_[i], vertex_count, anchor, colour,
                   GLLines.Style.Solid);
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
    main_prediction_vertex_count_ = 0;
    if (main_vessel_guid != null) {
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
                     adapter_.history_style);
      }
      {
        planetarium.PlanetariumPlotPrediction(Plugin,
                                              main_vessel_guid,
                                              prediction_t_max,
                                              VertexBuffer.data,
                                              VertexBuffer.size,
                                              out int vertex_count,
                                              out XYZ anchor);
        main_prediction_vertex_count_ = vertex_count;
        DrawLineMesh(ref prediction_mesh_,
                     vertex_count,
                     anchor,
                     adapter_.prediction_colour,
                     adapter_.prediction_style);
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
                           : adapter_.flight_plan_style);
        }
      }
    }

    // Target psychohistory and prediction.
    string target_id = FlightGlobals.fetch.VesselTarget?.GetVessel()?.id.
        ToString();
    if (FlightGlobals.ActiveVessel != null &&
        !adapter_.plotting_frame_selector_.target_frame_selected &&
        target_id != null &&
        Plugin.HasVessel(target_id)) {
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
                     adapter_.target_history_style);
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
                     adapter_.target_prediction_style);
      }
    }
  }

  // Plots the trajectories of every plugin-managed vessel other than
  // `main_vessel_guid`, in the target-vessel colours — the palette of a
  // vessel that is not the protagonist.  Used in the tracking station, which
  // surveys the whole fleet; `PlottedInTrackingStation` reports which
  // vessels actually drew, so that the caller only suppresses stock lines
  // that we replace.
  public void PlotFleetTrajectories(DisposablePlanetarium planetarium,
                                    string main_vessel_guid,
                                    double history_length) {
    fleet_plotted_.Clear();
    last_fleet_plot_frame_ = UnityEngine.Time.frameCount;
    var present = new HashSet<Guid>();
    foreach (Vessel vessel in FlightGlobals.Vessels) {
      string vessel_guid = vessel.id.ToString();
      if (!Plugin.HasVessel(vessel_guid)) {
        continue;
      }
      present.Add(vessel.id);
      if (vessel_guid == main_vessel_guid) {
        // Plotted by `PlotVesselTrajectories`, with its flight plan; the
        // same warm-up gate applies before its stock line goes.
        if (main_prediction_vertex_count_ >= 2) {
          fleet_plotted_.Add(vessel.id);
        }
        continue;
      }
      if (!fleet_trajectory_meshes_.TryGetValue(
              vessel.id,
              out FleetTrajectories trajectories)) {
        trajectories = fleet_trajectory_meshes_[vessel.id] =
            new FleetTrajectories();
      }
      {
        planetarium.PlanetariumPlotPsychohistory(
            Plugin,
            vessel_guid,
            history_length,
            t_max: null,
            VertexBuffer.data,
            VertexBuffer.size,
            out int vertex_count,
            out XYZ anchor);
        DrawLineMesh(trajectories.past, vertex_count, anchor,
                     adapter_.target_history_colour,
                     adapter_.target_history_style);
      }
      {
        planetarium.PlanetariumPlotPrediction(
            Plugin,
            vessel_guid,
            t_max: null,
            VertexBuffer.data,
            VertexBuffer.size,
            out int vertex_count,
            out XYZ anchor);
        DrawLineMesh(trajectories.future, vertex_count, anchor,
                     adapter_.target_prediction_colour,
                     adapter_.target_prediction_style);
        // Only report the vessel as covered once its orbit actually drew, so
        // that its stock line survives until the prognosticator delivers.
        if (vertex_count >= 2) {
          fleet_plotted_.Add(vessel.id);
        }
      }
    }
    List<Guid> stale = null;
    foreach (Guid vessel_id in fleet_trajectory_meshes_.Keys) {
      if (!present.Contains(vessel_id)) {
        if (stale == null) {
          stale = new List<Guid>();
        }
        stale.Add(vessel_id);
      }
    }
    if (stale != null) {
      foreach (Guid vessel_id in stale) {
        FleetTrajectories trajectories = fleet_trajectory_meshes_[vessel_id];
        UnityEngine.Object.Destroy(trajectories.past);
        UnityEngine.Object.Destroy(trajectories.future);
        fleet_trajectory_meshes_.Remove(vessel_id);
      }
    }
  }

  public bool PlottedInTrackingStation(Guid vessel_id) {
    // The set is one frame behind the plot; treat older data as absent so
    // that a scene re-entry never suppresses from a stale set.
    return UnityEngine.Time.frameCount - last_fleet_plot_frame_ <= 2 &&
           fleet_plotted_.Contains(vessel_id);
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
                     GLLines.Style.Faded);
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
                     GLLines.Style.Solid);
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

  private void DrawLineMesh(ref UnityEngine.Mesh mesh,
                            int vertex_count,
                            XYZ anchor,
                            UnityEngine.Color colour,
                            GLLines.Style style) {
    // Construct the mesh on the first call because Unity doesn't want us to do
    // that at construction.
    if (mesh == null) {
      mesh = MakeDynamicMesh();
    }
    DrawLineMesh(mesh, vertex_count, anchor, colour, style);
  }

  private void DrawLineMesh(UnityEngine.Mesh mesh,
                            int vertex_count,
                            XYZ anchor,
                            UnityEngine.Color colour,
                            GLLines.Style style) {
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
    // The vertices are relative to the anchor, whose single float rounding
    // here is common-mode over the mesh; drawing at the anchor reassembles
    // their scaled-space positions.
    // If the lines are drawn in layer 31 (Vectors), which sounds more
    // appropriate, they vanish when zoomed out.  Layer 9 works; pay no
    // attention to its name.
    UnityEngine.Graphics.DrawMesh(
        mesh,
        (UnityEngine.Vector3)anchor,
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

  private class FleetTrajectories {
    public UnityEngine.Mesh future = MakeDynamicMesh();
    public UnityEngine.Mesh past = MakeDynamicMesh();
  }

  private readonly Dictionary<CelestialBody, CelestialTrajectories>
      celestial_trajectory_meshes_ =
      new Dictionary<CelestialBody, CelestialTrajectories>();
  private readonly Dictionary<Guid, FleetTrajectories>
      fleet_trajectory_meshes_ = new Dictionary<Guid, FleetTrajectories>();
  private readonly HashSet<Guid> fleet_plotted_ = new HashSet<Guid>();
  private int last_fleet_plot_frame_ = -1;
  private int main_prediction_vertex_count_;
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
