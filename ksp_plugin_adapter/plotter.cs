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
    burn_start_scene_positions_.Clear();
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
            out XYZ anchor,
            out _);
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
                                              out XYZ anchor,
                                              out _);
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
          flight_plan_segment_tail_meshes_.Add(MakeDynamicMesh());
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
              out XYZ anchor,
              out int seam_vertex_count);
          if (is_burn) {
            Vector3d? translation = MeshTranslation(anchor,
                                                    main_reference,
                                                    null);
            burn_start_scene_positions_.Add(
                vertex_count > 0 && translation.HasValue
                    ? translation.Value + (Vector3d)VertexBuffer.vertices[0]
                    : (Vector3d?)null);
          }
          // No need for dynamic initialization, that was done above.
          DrawLineMesh(flight_plan_segment_meshes_[i],
                       seam_vertex_count,
                       anchor,
                       colour,
                       is_burn
                           ? adapter_.burn_style
                           : adapter_.flight_plan_style,
                       main_reference);
          if (seam_vertex_count < vertex_count) {
            // The vertices past the seam are the analytically extended tail,
            // set apart by colour: a plausible continuation, not an
            // integration.
            DrawLineMesh(flight_plan_segment_tail_meshes_[i],
                         vertex_count - seam_vertex_count,
                         anchor,
                         adapter_.flight_plan_tail_colour,
                         adapter_.flight_plan_tail_style,
                         main_reference,
                         first_vertex: seam_vertex_count);
          }
        }

        // The arrival ghost: where the targeted celestial sits when the plan
        // ends.
        if (FlightGlobals.fetch.VesselTarget is CelestialBody
                target_celestial) {
          planetarium.PlanetariumPlotArrivalGhost(
              Plugin,
              main_vessel_guid,
              target_celestial.flightGlobalsIndex,
              out bool plotted,
              out XYZ ghost,
              out XYZ ghost_anchor);
          if (plotted) {
            DrawArrivalGhost(ghost, ghost_anchor, main_reference);
          }
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
            out XYZ anchor,
            out _);
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
            out XYZ anchor,
            out _);
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
                            Vector3d? correction_reference_world = null,
                            int first_vertex = 0) {
    // Construct the mesh on the first call because Unity doesn't want us to do
    // that at construction.
    if (mesh == null) {
      mesh = MakeDynamicMesh();
    }
    DrawLineMesh(mesh, vertex_count, anchor, colour, style,
                 registration_reference_world, correction_reference_world,
                 first_vertex);
  }

  // Draws the `vertex_count` vertices of the buffer starting at
  // `first_vertex`.
  private void DrawLineMesh(UnityEngine.Mesh mesh,
                            int vertex_count,
                            XYZ anchor,
                            UnityEngine.Color colour,
                            GLLines.Style style,
                            Vector3d? registration_reference_world = null,
                            Vector3d? correction_reference_world = null,
                            int first_vertex = 0) {
    if (first_vertex + vertex_count > VertexBuffer.size) {
      Log.Fatal("Trying to draw " +
                (first_vertex + vertex_count) +
                " vertices, maximum is " +
                VertexBuffer.size);
    }

    if (indices_ == null) {
      indices_ = new int[4 * VertexBuffer.size];
      for (int i = 0; i < indices_.Length; ++i) {
        indices_[i] = i;
      }
    }

    if (style == GLLines.Style.Dashed && vertex_count >= 2) {
      // Dashes are cut by apparent arc along the plotted curve, not by
      // pairing raw vertices: a vertex pair is one adaptive-sampler segment,
      // whose length breathes with the local curvature and whose phase
      // reshuffles on every re-plot, so the dashes crawl.  The apparent arc
      // is a property of the curve as seen, which leaves the dashes still
      // and pixel-sized whatever the depth.
      bool camera_relative = anchor.x != 0 || anchor.y != 0 || anchor.z != 0;
      int dash_vertex_count = FillDashBuffers(first_vertex,
                                              vertex_count,
                                              colour,
                                              camera_relative);
      mesh.Clear();
      mesh.SetVertices(dash_vertices_);
      mesh.SetColors(dash_colours_);
      mesh.SetIndices(indices_,
                      indicesStart: 0,
                      indicesLength: dash_vertex_count,
                      UnityEngine.MeshTopology.Lines,
                      submesh: 0);
    } else {
      mesh.vertices = VertexBuffer.vertices;
      int index_count = style == GLLines.Style.Dashed ? vertex_count & ~1
                                                      : vertex_count;

      if (style == GLLines.Style.Faded) {
        for (int i = 0; i < vertex_count; ++i) {
          var faded_colour = colour;
          // Fade from the opacity of `colour` (when i = 0) down to 20% of
          // that opacity.
          faded_colour.a *= 1 - 0.8f * (i / (float)vertex_count);
          colours_[first_vertex + i] = faded_colour;
        }
      } else {
        for (int i = 0; i < vertex_count; ++i) {
          colours_[first_vertex + i] = colour;
        }
      }

      mesh.colors = colours_;
      mesh.SetIndices(indices_,
                      indicesStart: first_vertex,
                      indicesLength: index_count,
                      style == GLLines.Style.Dashed
                          ? UnityEngine.MeshTopology.Lines
                          : UnityEngine.MeshTopology.LineStrip,
                      submesh: 0);
    }
    mesh.RecalculateBounds();
    Vector3d? maybe_translation = MeshTranslation(anchor,
                                                  registration_reference_world,
                                                  correction_reference_world);
    if (!maybe_translation.HasValue) {
      return;
    }
    Vector3d translation = maybe_translation.Value;
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

  // Draws a small three-axis cross at the ghost vertex, in the tail's own
  // colour: the vertex and its anchor follow the anchored-plot conventions,
  // so the cross is assembled exactly as a line mesh would be.
  private void DrawArrivalGhost(XYZ ghost,
                                XYZ anchor,
                                Vector3d? registration_reference_world) {
    if (!registration_reference_world.HasValue) {
      return;
    }
    if (ghost_mesh_ == null) {
      ghost_mesh_ = MakeDynamicMesh();
    }
    Vector3d translation = SceneReferenceTranslation(
        registration_reference_world.Value, anchor);
    Vector3d position = translation + (Vector3d)ghost;
    Vector3d camera =
        PlanetariumCamera.fetch.transform.position;
    // A constant apparent size, from any distance.
    double size = (position - camera).magnitude * 0.01;
    Vector3d g = (Vector3d)ghost;
    ghost_vertices_[0] = (UnityEngine.Vector3)(g + new Vector3d(size, 0, 0));
    ghost_vertices_[1] = (UnityEngine.Vector3)(g - new Vector3d(size, 0, 0));
    ghost_vertices_[2] = (UnityEngine.Vector3)(g + new Vector3d(0, size, 0));
    ghost_vertices_[3] = (UnityEngine.Vector3)(g - new Vector3d(0, size, 0));
    ghost_vertices_[4] = (UnityEngine.Vector3)(g + new Vector3d(0, 0, size));
    ghost_vertices_[5] = (UnityEngine.Vector3)(g - new Vector3d(0, 0, size));
    ghost_mesh_.vertices = ghost_vertices_;
    for (int i = 0; i < ghost_colours_.Length; ++i) {
      ghost_colours_[i] = adapter_.flight_plan_tail_colour;
    }
    ghost_mesh_.colors = ghost_colours_;
    ghost_mesh_.SetIndices(ghost_indices_,
                           UnityEngine.MeshTopology.Lines,
                           submesh: 0);
    ghost_mesh_.RecalculateBounds();
    UnityEngine.Graphics.DrawMesh(
        ghost_mesh_,
        (UnityEngine.Vector3)translation,
        UnityEngine.Quaternion.identity,
        GLLines.line_material,
        (int)PrincipiaPluginAdapter.UnityLayers.Atmosphere,
        PlanetariumCamera.Camera);
  }

  // The scene translation at which a mesh with this anchor is drawn.  The
  // vertices are relative to the camera, which bounds their float rounding by
  // the ULP of their distance from it, angularly sub-pixel from any
  // viewpoint.  An anchored plot reports the camera as a displacement from
  // the plot's own reference — the plotted vessel at the present, the very
  // position the scene draws its icon at — so the scene's mapping of that
  // position, plus the displacement, reassembles the plot through the
  // scene's own arithmetic: the camera term cancels and neither side ever
  // expresses a point in scaled space across the distance to the plotting
  // frame's origin, where it would round to a kilometre.  A zero anchor is
  // the stock bit-identical path, left untouched.  A nonzero anchor without
  // its reference is meaningless — drawing it raw would put the mesh at the
  // scaled-space origin — and yields null.
  private Vector3d? MeshTranslation(XYZ anchor,
                                    Vector3d? registration_reference_world,
                                    Vector3d? correction_reference_world) {
    Vector3d translation = (Vector3d)anchor;
    if (anchor.x != 0 || anchor.y != 0 || anchor.z != 0) {
      if (registration_reference_world.HasValue) {
        translation = SceneReferenceTranslation(
            registration_reference_world.Value, anchor);
      } else if (correction_reference_world.HasValue) {
        translation += SceneMappingCorrection(
            correction_reference_world.Value);
      } else {
        return null;
      }
    }
    return translation;
  }

  // The scene position of the start of the manœuvre's burn segment, as the
  // last plot placed it; null when the burn was not plotted.  The manœuvre
  // markers take their position from here, so that they ride the same
  // jitter-free reassembly as the lines they sit on.
  public Vector3d? BurnStartScenePosition(int manœuvre_index) {
    return manœuvre_index < burn_start_scene_positions_.Count
               ? burn_start_scene_positions_[manœuvre_index]
               : null;
  }

  // Fills the dash buffers with dashes cut along the polyline's 3D arc — a
  // camera-independent parameter, so moving the viewpoint never slides the
  // boundaries — at the local period nearest a 16-pixel dash for that
  // point's camera distance, rounded to a power of two.  Powers of two with
  // a common origin nest, so a change of viewpoint or zoom only merges or
  // splits dashes in place.  Returns the number of vertices used.  On a
  // line plotted from a moving origin the pattern rides that origin.
  private int FillDashBuffers(int first_vertex,
                              int vertex_count,
                              UnityEngine.Color colour,
                              bool camera_relative) {
    if (dash_vertices_ == null) {
      dash_vertices_ = new List<UnityEngine.Vector3>();
      dash_colours_ = new List<UnityEngine.Color>();
    }
    dash_vertices_.Clear();
    dash_colours_.Clear();
    var vertices = VertexBuffer.vertices;
    UnityEngine.Vector3 camera =
        camera_relative
            ? UnityEngine.Vector3.zero
            : PlanetariumCamera.Camera.transform.position;
    double angular_period = 32 * TanAngularResolution();

    // Estimate the dash count to bound the buffers: a plot coiled by a
    // twisted frame can sweep an enormous apparent arc.
    double estimated_periods = 0;
    for (int i = 1; i < vertex_count; ++i) {
      UnityEngine.Vector3 p0 = vertices[first_vertex + i - 1];
      UnityEngine.Vector3 p1 = vertices[first_vertex + i];
      double r = Math.Max(
          0.5 * ((p0 - camera).magnitude + (p1 - camera).magnitude),
          minimal_camera_distance);
      estimated_periods += (p1 - p0).magnitude / (angular_period * r);
    }
    if (!(estimated_periods > 0) || double.IsInfinity(estimated_periods)) {
      return 0;
    }
    int extra_octaves = (int)Math.Max(
        0,
        Math.Ceiling(Math.Log(estimated_periods / max_dash_periods, 2)));

    int n = 0;
    double s = 0;  // 3D arc length from the first vertex.
    for (int i = 1;
         i < vertex_count && n < 8 * max_dash_periods;
         ++i) {
      UnityEngine.Vector3 p0 = vertices[first_vertex + i - 1];
      UnityEngine.Vector3 p1 = vertices[first_vertex + i];
      double ds = (p1 - p0).magnitude;
      if (!(ds > 0)) {
        continue;
      }
      double r0 = Math.Max((p0 - camera).magnitude, minimal_camera_distance);
      double r1 = Math.Max((p1 - camera).magnitude, minimal_camera_distance);
      double pos = 0;
      while (pos < ds && n < 8 * max_dash_periods) {
        double r = r0 + (r1 - r0) * (pos / ds);
        double period = Math.Pow(
            2,
            Math.Round(Math.Log(angular_period * r, 2)) + extra_octaves);
        double a = s + pos;
        double cell = Math.Floor(a / period);
        double next = Math.Min((cell + 1) * period - s, ds);
        if (next <= pos) {
          // A rounding stall at a cell boundary; step clear of it.
          next = Math.Min(pos + period / 2, ds);
        }
        if (cell % 2 == 0) {
          dash_vertices_.Add(
              UnityEngine.Vector3.Lerp(p0, p1, (float)(pos / ds)));
          dash_colours_.Add(colour);
          dash_vertices_.Add(
              UnityEngine.Vector3.Lerp(p0, p1, (float)(next / ds)));
          dash_colours_.Add(colour);
          n += 2;
        }
        pos = next;
      }
      s += ds;
    }
    return n;
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
  private readonly List<UnityEngine.Mesh> flight_plan_segment_tail_meshes_ =
      new List<UnityEngine.Mesh>();
  private readonly List<UnityEngine.Mesh> equipotential_meshes_ =
      new List<UnityEngine.Mesh>();
  private UnityEngine.Mesh target_psychohistory_mesh_;
  private UnityEngine.Mesh target_prediction_mesh_;
  private UnityEngine.Mesh ghost_mesh_;
  private readonly UnityEngine.Vector3[] ghost_vertices_ =
      new UnityEngine.Vector3[6];
  private readonly UnityEngine.Color[] ghost_colours_ =
      new UnityEngine.Color[6];
  private readonly int[] ghost_indices_ = { 0, 1, 2, 3, 4, 5 };
  private int[] indices_ = null;
  private UnityEngine.Color[] colours_ =
      new UnityEngine.Color[VertexBuffer.size];
  private readonly List<Vector3d?> burn_start_scene_positions_ =
      new List<Vector3d?>();
  private List<UnityEngine.Vector3> dash_vertices_ = null;
  private List<UnityEngine.Color> dash_colours_ = null;
  private const int max_dash_periods = 6000;
  private const double minimal_camera_distance = 1e-6;
}

}  // namespace ksp_plugin_adapter
}  // namespace principia
