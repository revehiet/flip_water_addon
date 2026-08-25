"""Property groups attached to Blender objects for the FLIP water addon."""

import bpy
from bpy.types import PropertyGroup
from bpy.props import (
    BoolProperty, FloatProperty, IntProperty, EnumProperty, StringProperty, PointerProperty,
)


def _refresh_surface_modifier(self, context):
    domain_obj = getattr(self, "id_data", None)
    if not isinstance(domain_obj, bpy.types.Object):
        return

    try:
        from . import operators
    except Exception:  # noqa: BLE001
        return

    ctx = context if context is not None else bpy.context
    scene = ctx.scene if ctx is not None else bpy.context.scene
    frame = scene.frame_current if scene is not None else 1
    operators.refresh_surface_preview(ctx, domain_obj, frame)


def _tag_redraw_node_editors(self, context):
    ctx = context if context is not None else bpy.context
    wm = getattr(ctx, "window_manager", None)
    if wm is None:
        return
    for window in wm.windows:
        for area in window.screen.areas:
            if area.type == 'NODE_EDITOR':
                area.tag_redraw()


def _refresh_particle_overlays(self, context):
    domain_obj = getattr(self, "id_data", None)
    if not isinstance(domain_obj, bpy.types.Object):
        return

    try:
        from . import operators
    except Exception:  # noqa: BLE001
        return

    ctx = context if context is not None else bpy.context
    scene = ctx.scene if ctx is not None else bpy.context.scene
    frame = scene.frame_current if scene is not None else 1
    operators.sync_seed_previews_from_node_graph(ctx)
    operators.update_baked_domain_overlay(domain_obj, frame)
    operators.update_whitewater_overlay(domain_obj, frame)
    operators.refresh_seed_preview_if_active(ctx, domain_obj.name)


def _refresh_domain_resolution_dependents(self, context):
    """When domain resolution changes, refresh systems that depend on cell size.

    This includes live surface modifier tuning and obstacle voxel previews.
    """
    _refresh_surface_modifier(self, context)

    domain_obj = getattr(self, "id_data", None)
    if not isinstance(domain_obj, bpy.types.Object):
        return

    try:
        from . import operators
    except Exception:  # noqa: BLE001
        return
    operators.refresh_all_obstacle_previews_for_domain(context, domain_obj)
    operators.refresh_domain_voxel_guide(context, domain_obj)


def _refresh_obstacle_preview(self, context):
    """Regenerates obstacle collision preview when settings change."""
    obj = getattr(self, "id_data", None)
    if not isinstance(obj, bpy.types.Object):
        return

    try:
        from . import operators
    except Exception:  # noqa: BLE001
        return

    if self.voxel_preview_enabled:
        ptype = getattr(self, 'collision_preview_type', 'VOXEL')
        if ptype == 'SDF':
            operators.refresh_obstacle_sdf_preview(context, obj)
            operators.clear_obstacle_voxel_preview(obj)
        else:
            operators.refresh_obstacle_voxel_preview(context, obj)
            operators.clear_obstacle_sdf_preview(obj)
    else:
        operators.clear_obstacle_voxel_preview(obj)
        operators.clear_obstacle_sdf_preview(obj)


_FLIP_PRESETS = {
    'WATER': dict(
        density=1000.0, flip_ratio=0.95,
        gravity=(0.0, 0.0, -9.81), cfl_number=8.0,
        viscosity_strength=0.0, surface_tension_strength=0.0),
    'HONEY': dict(
        density=1400.0, flip_ratio=0.05,
        gravity=(0.0, 0.0, -9.81), cfl_number=4.0,
        viscosity_strength=0.9, surface_tension_strength=10.0),
    'LAVA': dict(
        density=2600.0, flip_ratio=0.5,
        gravity=(0.0, 0.0, -9.81), cfl_number=4.0,
        viscosity_strength=0.6, surface_tension_strength=20.0),
    'SPLASH': dict(
        density=1000.0, flip_ratio=1.0,
        gravity=(0.0, 0.0, -9.81), cfl_number=8.0,
        viscosity_strength=0.0, surface_tension_strength=0.0),
    'ZERO_G': dict(
        density=1000.0, flip_ratio=0.95,
        gravity=(0.0, 0.0, 0.0), cfl_number=8.0,
        viscosity_strength=0.0, surface_tension_strength=40.0),
}


def _apply_flip_preset(self, context):
    """Enum update callback: applies a liquid preset's material settings
    (density, FLIP/PIC blend, gravity, CFL, viscosity, surface tension)."""
    preset = _FLIP_PRESETS.get(self.flip_preset)
    if preset is None:
        return
    self.density = preset["density"]
    self.flip_ratio = preset["flip_ratio"]
    self.gravity_override = True
    self.gravity = preset["gravity"]
    self.cfl_number = preset["cfl_number"]
    if "viscosity_strength" in preset:
        self.viscosity_strength = preset["viscosity_strength"]
    if "surface_tension_strength" in preset:
        self.surface_tension_strength = preset["surface_tension_strength"]


class FLIPWATER_DomainSettings(PropertyGroup):
    """Attached to any object marked as a FLIP fluid domain."""

    resolution: IntProperty(
        name="Resolution",
        description="Number of grid cells along the domain's longest axis. "
                    "Higher = more detail but much slower and more memory",
        default=48, min=8, max=512, update=_refresh_domain_resolution_dependents,
    )
    flip_ratio: FloatProperty(
        name="FLIP Ratio",
        description="Blend between FLIP (energetic, splashy, 1.0) and PIC "
                    "(stable, viscous-looking, 0.0). 0.9-0.97 is typical for water",
        default=0.95, min=0.0, max=1.0,
    )
    flip_preset: EnumProperty(
        name="Liquid Preset",
        description="One-click material presets that tune density, FLIP/PIC "
                    "blend, gravity and CFL for a typical look of each liquid. "
                    "Changing this overwrites those settings",
        items=[
            ('WATER', "Water", "Standard water — neutral, splashy"),
            ('HONEY', "Honey", "Thick and slow — heavy PIC damping"),
            ('LAVA', "Lava", "Dense and heavy — slow thick flow"),
            ('SPLASH', "Splash", "Maximum energy — pure FLIP, very splashy"),
            ('ZERO_G', "Zero-G", "No gravity — floating blobs"),
            ('CUSTOM', "Custom", "No preset — manually tuned values"),
        ],
        default='CUSTOM',
        update=_apply_flip_preset,
    )
    density: FloatProperty(
        name="Density (kg/m³)", default=1000.0, min=1.0, max=20000.0,
    )
    gravity_override: BoolProperty(
        name="Override Scene Gravity",
        description="Use a custom gravity vector instead of the scene's gravity",
        default=False,
    )
    gravity: bpy.props.FloatVectorProperty(
        name="Gravity", subtype='ACCELERATION', size=3, default=(0.0, 0.0, -9.81),
    )
    cfl_number: FloatProperty(
        name="Target CFL Number",
        description=(
            "Target max particle travel per physics grid step, in grid cells. "
            "With ST-FLIP enabled this can be pushed well beyond classic FLIP's "
            "~1-3 (the method's whole point is staying artifact-free at much "
            "larger steps) - try 8-16 for a solid speed/quality balance, up to "
            "~30 for maximum speed with more Monte-Carlo-style surface noise. "
            "Higher = fewer, larger physics steps = faster bakes"
        ),
        default=8.0, min=0.5, max=40.0,
    )
    st_flip_enabled: BoolProperty(
        name="ST-FLIP (Spatiotemporal Sampling)",
        description=(
            "Braun, Winchenbach, Bender & Thuerey, 'Spatiotemporal FLIP for Fast "
            "Free-Surface and Two-Phase Simulation With Very Large Time Steps' "
            "(ACM TOG 45(4), SIGGRAPH 2026). Treats particles as samples in "
            "space AND time, which avoids the aliasing/rippling artifacts "
            "classic FLIP gets at large time steps - letting you raise the "
            "target CFL number well beyond 1-3 for a large speedup at "
            "comparable quality. Turn off to fall back to classic instantaneous "
            "FLIP (useful for comparison, or if you need bit-for-bit determinism "
            "matching older bakes)"
        ),
        default=True,
    )
    jitter_strength: FloatProperty(
        name="Jitter Strength",
        description="ST-FLIP temporal jitter amount (the paper's gamma). 1.0 is "
                    "the paper's default; lower reduces surface noise on calm "
                    "water at the cost of some large-step artifact resistance",
        default=1.0, min=0.0, max=1.0,
    )
    max_substeps: IntProperty(name="Max Substeps / Frame", default=48, min=1, max=200)
    pressure_iterations: IntProperty(name="Pressure Solver Iterations", default=150, min=10, max=1000)
    particles_per_cell: IntProperty(
        name="Particles per Cell (per axis)",
        description="Particle seeding density. 2 means 2x2x2=8 particles per cell",
        default=2, min=1, max=4,
    )
    seeding_lattice: EnumProperty(
        name="Seeding Lattice",
        description="How emitter/tank particles are arranged at seed time. "
                    "BCC (body-centered cubic) interleaves two offset grids for a "
                    "more isotropic distribution (~30% less grid-aligned bias), "
                    "at the same particle count as the axis-aligned lattice",
        items=[
            ('AA', "Axis-Aligned", "Simple cubic lattice (classic FLIP seeding)"),
            ('BCC', "Body-Centered Cubic", "Interleaved offset lattices — more isotropic"),
        ],
        default='BCC',
    )
    max_particles: IntProperty(
        name="Max Particles", default=2000000, min=1000, max=20000000,
    )

    # ── Houdini FLIP Solver parity ─────────────────────────────────────────

    reseed_enabled: BoolProperty(
        name="Reseeding",
        description="Keep particle density inside the target range by topping up "
                    "under-sampled cells and removing excess particles from "
                    "over-sampled cells (Houdini FLIP reseeding)",
        default=False,
    )
    reseed_min_ratio: FloatProperty(
        name="Min Density",
        description="Cells holding fewer than this fraction of the nominal "
                    "particles-per-cell count get new particles",
        default=0.5, min=0.1, max=1.0,
    )
    reseed_max_ratio: FloatProperty(
        name="Max Density",
        description="Cells holding more than this fraction of the nominal "
                    "particles-per-cell count have excess particles removed",
        default=2.5, min=1.0, max=4.0,
    )

    viscosity_strength: FloatProperty(
        name="Viscosity",
        description="Velocity diffusion (XSPH) between neighbouring particles. "
                    "0 = inviscid water; higher = honey-like damping",
        default=0.0, min=0.0, max=1.0,
    )
    surface_tension_strength: FloatProperty(
        name="Surface Tension",
        description="Cohesion force pulling under-sampled surface particles "
                    "toward the liquid mass (acceleration, m/s^2). 0 = off; "
                    "~30-100 gives visible rounding/beading",
        default=0.0, min=0.0, max=500.0,
    )
    vorticity_confinement: FloatProperty(
        name="Vorticity Confinement",
        description="Reinjects dissipated rotational energy into the flow "
                    "(Fedkiw-style epsilon x (N x omega)). 0 = off; 0.05-0.5 "
                    "keeps splashes lively at large CFL steps",
        default=0.0, min=0.0, max=2.0, step=0.05,
    )

    pressure_warm_start: BoolProperty(
        name="Pressure Warm Start",
        description="Seed the pressure CG solve with the previous step's "
                    "pressure field - typically cuts solver iterations roughly "
                    "in half after the first frame",
        default=True,
    )
    adaptive_pressure_iterations: BoolProperty(
        name="Adaptive Iterations",
        description="Grow/shrink the pressure solver iteration cap based on how "
                    "quickly it converges, instead of always running the full "
                    "iteration budget",
        default=True,
    )

    air_incompressibility_enabled: BoolProperty(
        name="Air Incompressibility",
        description="Two-phase FLIP approximation: a band of air cells around "
                    "the liquid joins the pressure solve as a low-density second "
                    "phase, so trapped air pushes back instead of being pinned "
                    "at p=0 (CPU backend; CUDA falls back to CPU when enabled)",
        default=False,
    )
    air_band_cells: IntProperty(
        name="Air Band Width",
        description="Depth of the air band around the liquid surface (in cells) "
                    "that participates in the pressure solve",
        default=3, min=1, max=8,
    )
    air_density_ratio: FloatProperty(
        name="Air Density Ratio",
        description="Air density relative to water inside the active band "
                    "(~0.001 physical; higher values are more stable)",
        default=0.01, min=0.0001, max=1.0,
    )

    # ── Whitewater (dedicated secondary solver) ────────────────────────────

    whitewater_enabled: BoolProperty(
        name="Whitewater",
        description="Run the dedicated Whitewater solver after each FLIP frame: "
                    "emits foam/spray/bubble particles from the liquid's "
                    "vorticity near the surface and simulates them as a "
                    "separate particle system (Houdini Whitewater Solver style)",
        default=False,
    )
    whitewater_emission_amount: FloatProperty(
        name="Emission Amount",
        description="Multiplier on emitted whitewater (Houdini's Emission Amount)",
        default=1.0, min=0.0, max=100.0,
    )
    whitewater_scale: FloatProperty(
        name="Whitewater Scale",
        description="Target separation between whitewater particles in world "
                    "units (Houdini's Whitewater Scale; smaller = denser foam)",
        default=0.03, min=0.005, max=0.5, step=0.005,
    )
    whitewater_vorticity_threshold: FloatProperty(
        name="Vorticity Threshold",
        description="Minimum liquid vorticity magnitude (1/s) that triggers "
                    "whitewater emission at the surface",
        default=3.0, min=0.0, max=100.0,
    )
    whitewater_lifespan: FloatProperty(
        name="Lifespan",
        description="Average whitewater particle lifetime in seconds",
        default=3.0, min=0.1, max=30.0,
    )
    whitewater_aging_foam: FloatProperty(
        name="Foam Aging Rate",
        description="Multiplier on foam lifetime before it converts to bubbles",
        default=1.0, min=0.1, max=10.0,
    )
    whitewater_aging_bubble: FloatProperty(
        name="Bubble Aging Rate",
        description="Multiplier on bubble lifetime before it converts to spray",
        default=1.0, min=0.1, max=10.0,
    )
    whitewater_aging_spray: FloatProperty(
        name="Spray Aging Rate",
        description="Multiplier on spray lifetime before it dies",
        default=1.0, min=0.1, max=10.0,
    )
    whitewater_buoyancy: FloatProperty(
        name="Buoyancy",
        description="Upward acceleration applied to bubble particles (m/s^2)",
        default=9.81, min=0.0, max=100.0,
    )
    whitewater_noise: FloatProperty(
        name="Birth Noise",
        description="Random velocity noise added to newly born whitewater "
                    "particles (m/s)",
        default=0.5, min=0.0, max=10.0,
    )
    whitewater_advection_strength: FloatProperty(
        name="Advection Strength",
        description="How strongly whitewater follows the liquid velocity field "
                    "(Houdini's Base Advection Strength)",
        default=1.0, min=0.0, max=5.0,
    )
    whitewater_seed: IntProperty(
        name="Seed", description="Random seed for whitewater emission", default=12345,
    )
    whitewater_max_particles: IntProperty(
        name="Max Whitewater Particles", default=2000000, min=1000, max=20000000,
    )
    whitewater_overlay_enabled: BoolProperty(
        name="Viewport Preview",
        description="Draw whitewater particles in the viewport (foam = white, "
                    "spray = pale blue, bubbles = cyan)",
        default=True,
    )
    collision_mode: EnumProperty(
        name="Collision Mode",
        description="How solid obstacles are represented for collision during baking",
        items=[
            ('VOXEL', "Voxel Mask (Legacy)",
             "Binary per-cell solid/open mask - the original, fast method"),
            ('SDF', "Signed Distance Field (Experimental)",
             "Smoother sub-cell collision with particle penetration push-out; may "
             "better preserve thin or curved obstacle shapes"),
        ],
        default='SDF',
    )
    sdf_collision_margin: FloatProperty(
        name="SDF Collision Margin",
        description="Push-out distance as fraction of cell size. Lower = tighter fit but more likely to stick. "
                    "Higher = softer but more clearance (0.001-0.5)",
        default=0.01, min=0.001, max=0.5, soft_min=0.005, soft_max=0.1, step=0.005,
    )
    solver_backend: EnumProperty(
        name="Solver Backend",
        description="Pressure-solver implementation: CPU (OpenMP multi-core) or GPU (CUDA). "
                    "GPU typically 5-20× faster for grids above ~64³, especially on NVIDIA cards. "
                    "Falls back to CPU automatically if CUDA is unavailable at runtime",
        items=[
            ('CPU', "CPU (OpenMP)", "Multi-core CPU conjugate gradient"),
            ('CUDA', "GPU (CUDA)", "CUDA-accelerated conjugate gradient (NVIDIA only)"),
        ],
        default='CUDA',
    )
    particle_overlay_enabled: BoolProperty(
        name="Particle Overlay Preview",
        description="Draw sim particles in viewport as a GPU overlay during bake",
        default=True, update=_refresh_particle_overlays,
    )
    particle_overlay_max_points: IntProperty(
        name="Overlay Max Points",
        description="Maximum particle points drawn in overlay (subsampled if needed)",
        default=120000, min=1000, max=2000000, update=_refresh_particle_overlays,
    )
    particle_overlay_point_size: FloatProperty(
        name="Overlay Point Size",
        description="Viewport point size for particle overlays",
        default=2.5, min=1.0, max=12.0, update=_refresh_particle_overlays,
    )
    particle_overlay_render_style: EnumProperty(
        name="Point Style",
        description="How particle overlays are drawn in the viewport, always applied consistently "
                    "(live preview and during bake)",
        items=[
            ('SPHERES', "Spheres", "Render particles as low-poly spheres (clearer, more GPU cost)"),
            ('POINTS', "Points", "Render particles as GL points (fastest)"),
        ],
        default='POINTS', update=_refresh_particle_overlays,
    )
    show_domain_overlay: BoolProperty(
        name="Show Domain Overlay",
        description="Draw the domain bounding box and a 1-voxel guide cube as a GPU viewport overlay",
        default=True,
        update=_refresh_domain_resolution_dependents,
    )

    # ── UI collapse toggles for the FLIP Solver node ──
    show_advanced: BoolProperty(name="Show Advanced", default=False, update=_tag_redraw_node_editors)
    show_viewport: BoolProperty(name="Show Viewport", default=False, update=_tag_redraw_node_editors)
    show_collisions: BoolProperty(name="Show Collisions", default=False, update=_tag_redraw_node_editors)
    show_outflow: BoolProperty(name="Show Domain Wall Outflow", default=False, update=_tag_redraw_node_editors)
    show_performance: BoolProperty(name="Show Performance", default=False, update=_tag_redraw_node_editors)
    show_gravity: BoolProperty(name="Show Gravity Override", default=False, update=_tag_redraw_node_editors)
    show_reseeding: BoolProperty(name="Show Reseeding", default=False, update=_tag_redraw_node_editors)
    show_liquid_material: BoolProperty(name="Show Viscosity & Surface Tension", default=False, update=_tag_redraw_node_editors)
    show_vorticity: BoolProperty(name="Show Vorticity Confinement", default=False, update=_tag_redraw_node_editors)
    show_pressure_solve: BoolProperty(name="Show Pressure Solve", default=False, update=_tag_redraw_node_editors)
    show_air_phase: BoolProperty(name="Show Air Incompressibility", default=False, update=_tag_redraw_node_editors)
    show_whitewater: BoolProperty(name="Show Whitewater", default=False, update=_tag_redraw_node_editors)

    cache_dir: StringProperty(
        name="Cache Directory", subtype='DIR_PATH',
        description="Where baked per-frame particle caches are stored. Leave "
                    "empty to use a 'flip_cache' folder next to the .blend file",
        default="",
    )
    cache_compression: BoolProperty(
        name="Compress Cache Files",
        description="Store particle cache frames with fast zlib compression "
                    "(smaller files, negligible bake overhead)",
        default=True,
    )
    cache_velocity_half: BoolProperty(
        name="Half-Precision Velocities",
        description="Store velocities as float16 — about 17% smaller cache at a "
                    "minor precision loss (fine for playback and motion blur)",
        default=False,
    )
    cache_format: EnumProperty(
        name="Cache Format",
        description="Storage format for particle cache frames. FWC2 is the native "
                    "binary format (fastest). HDF5 writes each frame as a gzip-"
                    "compressed .h5 file that external pipelines can read "
                    "directly (needs h5py in Blender's Python — falls back to "
                    "FWC2 automatically if missing)",
        items=[
            ('FWC2', "FWC2 Binary", "Native binary format — fastest bake/playback"),
            ('HDF5', "HDF5 (.h5)", "Pipeline-friendly per-frame .h5 files (h5py required)"),
        ],
        default='FWC2',
    )

    frame_start: IntProperty(name="Start Frame", default=1)
    frame_end: IntProperty(name="End Frame", default=100)

    is_baking: BoolProperty(default=False)
    is_baked: BoolProperty(default=False)
    bake_progress: FloatProperty(default=0.0, min=0.0, max=1.0)
    bake_current_frame: IntProperty(default=0)
    bake_eta_seconds: FloatProperty(default=0.0, min=0.0)
    bake_particle_count: IntProperty(default=0, min=0)
    bake_elapsed_seconds: FloatProperty(default=0.0, min=0.0)
    bake_peak_particle_count: IntProperty(default=0, min=0)
    bake_solver_backend: StringProperty(
        name="Solver Backend Used",
        description="Which solver backend was actually used during the last bake (CPU, GPU, or fallback)",
        default="",
    )

    surface_object: PointerProperty(
        name="Surface Object", type=bpy.types.Object,
        description="Optional mesh object used to display reconstructed water surface",
    )

    surface_particle_radius_scale: FloatProperty(
        name="Influence Scale",
        description="How far particles interact, as a multiple of the particle "
                    "separation (Houdini's Influence Scale). Higher = smoother "
                    "surface but slower meshing",
        default=1.0, min=0.5, max=3.0, update=_refresh_surface_modifier,
    )
    surface_particle_separation: FloatProperty(
        name="Particle Separation",
        description="World-space distance between particles (Houdini's Particle "
                    "Separation). 0 = automatic, derived from the simulation "
                    "grid cell size divided by particles-per-cell",
        default=0.0, min=0.0, max=10.0, soft_max=1.0, step=0.005,
        precision=4, update=_refresh_surface_modifier,
    )
    surface_adaptivity: FloatProperty(
        name="Adaptivity",
        description="Mesh polygonization tolerance (Houdini's Adaptivity). 0 keeps "
                    "the mesh at full voxel resolution; higher values give fewer, "
                    "larger polygons with a less precise match (OpenVDB mesher "
                    "only)",
        default=0.0, min=0.0, max=1.0, update=_refresh_surface_modifier,
    )
    surface_max_particles: IntProperty(
        name="Max Surface Particles",
        description="Subsample particles before surface reconstruction to speed up meshing. "
                    "0 = use all particles. 50000-200000 gives good quality/speed balance",
        default=80000, min=0, max=2000000, update=_refresh_surface_modifier,
    )
    surface_smoothing_length: FloatProperty(
        name="Smoothing Length",
        description="SPH kernel smoothing length, as a multiple of the particle radius",
        default=2.0, min=1.0, max=6.0, update=_refresh_surface_modifier,
    )
    surface_cube_size_scale: FloatProperty(
        name="Voxel Scale",
        description="Marching-cubes voxel size as a multiple of the influence radius "
                    "(Houdini's Voxel Scale; smaller = more detail, slower). "
                    "Affects both OpenVDB and GPU meshers",
        default=0.5, min=0.05, max=3.0, update=_refresh_surface_modifier,
    )
    surface_threshold: FloatProperty(
        name="Isovalue",
        description="Surface level-set iso offset in OpenVDB mode / density-threshold "
                    "scale in GPU mode (higher = smaller, tighter surface). "
                    "~0.6 is a good default",
        default=0.6, min=0.01, max=2.0, update=_refresh_surface_modifier,
    )
    surface_smoothing_iterations: IntProperty(
        name="Mesh Smoothing Iterations",
        description="Weighted Laplacian smoothing iterations applied to the reconstructed surface "
                    "(0 disables smoothing)",
        default=0, min=0, max=100, update=_refresh_surface_modifier,
    )
    surface_mesh_cleanup: BoolProperty(
        name="Mesh Cleanup",
        description="Removes sliver triangles typically generated by marching cubes",
        default=True, update=_refresh_surface_modifier,
    )

    surface_mesher_mode: EnumProperty(
        name="Mesher",
        items=[
            ("OpenVDB", "OpenVDB", "CPU OpenVDB level-set meshing — smooth, high quality"),
            ("GPU", "GPU (CUDA)", "GPU marching-cubes meshing — fast, slightly faceted"),
        ],
        default="OpenVDB",
        update=_refresh_surface_modifier,
        description="Which surface reconstruction backend to use",
    )
    surface_gpu_iso: FloatProperty(
        name="GPU Iso Threshold",
        description="Isosurface threshold of the GPU density field. Lower = surface "
                    "further outside the particles (~0.2-0.35 is a good range)",
        default=0.25, min=0.01, max=2.0, update=_refresh_surface_modifier,
    )
    surface_use_obstacles: BoolProperty(
        name="Cut by Colliders",
        description="CSG-subtract collider meshes from the fluid surface "
                    "(OpenVDB mode). The water surface is cut at collider boundaries",
        default=True, update=_refresh_surface_modifier,
    )
    surface_droplet_scale: FloatProperty(
        name="Droplet Scale",
        description="Remove disconnected surface blobs smaller than this fraction "
                    "of the main body (Houdini's Droplet Scale separation). "
                    "0 = keep everything",
        default=0.0, min=0.0, max=0.5, step=0.01, update=_refresh_surface_modifier,
    )
    surface_preserve_bubbles: BoolProperty(
        name="Preserve Bubbles",
        description="Mesh enclosed air pockets inside the liquid as separate "
                    "bubble surfaces (Houdini's Preserve Bubbles; OpenVDB mode)",
        default=False, update=_refresh_surface_modifier,
    )

    is_surface_baked: BoolProperty(default=False)
    is_baking_surface: BoolProperty(default=False)
    surface_bake_progress: FloatProperty(default=0.0, min=0.0, max=1.0)
    surface_bake_current_frame: IntProperty(default=0)

    outflow_x_minus: BoolProperty(name="Outflow -X", default=False)
    outflow_x_plus: BoolProperty(name="Outflow +X", default=False)
    outflow_y_minus: BoolProperty(name="Outflow -Y", default=False)
    outflow_y_plus: BoolProperty(name="Outflow +Y", default=False)
    outflow_z_minus: BoolProperty(name="Outflow -Z", default=False)
    outflow_z_plus: BoolProperty(name="Outflow +Z", default=False)
    outflow_debug_enabled: BoolProperty(
        name="Debug: Show Removed Particles",
        description="Draws a GPU overlay (red points) of particles removed by Domain Wall Outflow this frame",
        default=False,
    )

    viz_mode: EnumProperty(
        name="Color By",
        description="Colors the particle overlay by a physical quantity instead of a flat color",
        items=[
            ('NONE', "Off", "Use a single flat overlay color"),
            ('VELOCITY', "Velocity", "Color particles by speed (blue = slow, red = fast)"),
            ('VORTICITY', "Vorticity (approx)", "Color particles by an approximate local vorticity magnitude, "
                                                  "estimated from a coarse velocity grid since the solver core "
                                                  "does not expose true vorticity"),
        ],
        default='VELOCITY', update=_refresh_particle_overlays,
    )


class FLIPWATER_EmitterSettings(PropertyGroup):
    emission_type: EnumProperty(
        name="Emission Type",
        items=[
            ('VOLUME_ONCE', "Initial Volume", "Fill the emitter's volume with particles once, at the start frame (dam break / splash)"),
            ('INFLOW', "Inflow", "Continuously emit particles from the emitter's volume every frame (faucet / stream)"),
        ],
        default='VOLUME_ONCE',
    )
    initial_speed: bpy.props.FloatVectorProperty(name="Initial Velocity", size=3, default=(0.0, 0.0, 0.0))
    sampling_mode: EnumProperty(
        name="Sampling",
        items=[
            ('BOUNDS', "Bounding Box", "Fast: seed particles in the emitter's axis-aligned bounding box"),
            ('MESH', "Mesh Volume", "Accurate: seed particles only inside the emitter's actual mesh volume (requires a closed/manifold mesh)"),
        ],
        default='MESH',
    )
    enabled: BoolProperty(name="Enabled", default=True)
    animated: BoolProperty(
        name="Animated",
        description=(
            "Enable ONLY if this emitter's position/shape actually changes over "
            "time (keyframes, drivers, animated modifiers/shape keys). Leave off "
            "for static emitters (the common case) - baking is significantly "
            "faster since Blender's full scene doesn't need to be re-evaluated "
            "every single frame, and the emitter only needs to be sampled once "
            "instead of every frame"
        ),
        default=False,
    )


class FLIPWATER_ObstacleSettings(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=True, update=_refresh_obstacle_preview)
    collision_preview_type: EnumProperty(
        name="Preview Type",
        description="Which collision representation to show in the viewport overlay",
        items=[
            ('VOXEL', "Voxel Mask", "Wireframe outline of which grid cells are solid"),
            ('SDF', "Signed Distance Field", "Color-coded SDF shell (red=inside, blue=outside)"),
        ],
        default='VOXEL',
        update=_refresh_obstacle_preview,
    )
    animated: BoolProperty(
        name="Animated",
        description=(
            "Enable if this collider's position/shape changes over time "
            "(keyframes, drivers, animated modifiers). When on, the collider "
            "is re-voxelized every frame during bake. Leave off for static "
            "colliders to avoid unnecessary per-frame computation"
        ),
        default=False,
    )
    voxel_padding_cells: IntProperty(
        name="Voxel Padding",
        description="Extra grid-cell padding around obstacle bounds during voxelization",
        default=1, min=0, max=8, update=_refresh_obstacle_preview,
    )
    voxel_dilation_steps: IntProperty(
        name="Voxel Dilation",
        description="Expand solid cells after voxelization to thicken thin obstacles",
        default=0, min=0, max=4, update=_refresh_obstacle_preview,
    )
    sdf_preview_band_cells: FloatProperty(
        name="SDF Band Width",
        description="How many grid cells around the surface to visualize (larger = wider colored shell)",
        default=2.5, min=0.5, max=8.0, soft_min=0.5, soft_max=5.0, step=0.5,
        update=_refresh_obstacle_preview,
    )
    sdf_preview_point_size: FloatProperty(
        name="SDF Point Size",
        description="Size of the overlay points for the SDF field preview",
        default=4.0, min=1.0, max=12.0, soft_min=2.0, soft_max=8.0, step=0.5,
        update=_refresh_obstacle_preview,
    )
    voxel_preview_enabled: BoolProperty(
        name="Preview Voxels",
        description="Show/update a generated voxel collision preview mesh",
        default=False,
        update=_refresh_obstacle_preview,
    )
    voxel_preview_object: PointerProperty(
        name="Voxel Preview Object",
        type=bpy.types.Object,
    )
    voxel_preview_domain_object: PointerProperty(
        name="Voxel Preview Domain",
        type=bpy.types.Object,
    )
    sdf_preview_enabled: BoolProperty(
        name="Preview SDF Field",
        description="Show a color-coded viewport overlay of the signed distance field near this "
                     "collider's surface (red = inside solid, blue = outside)",
        default=False,
    )


class FLIPWATER_SinkSettings(PropertyGroup):
    enabled: BoolProperty(name="Enabled", default=True)


class FLIPWATER_MpmBakeState(PropertyGroup):
    is_baking: BoolProperty(default=False)
    bake_current_frame: IntProperty(default=0)
    bake_progress: FloatProperty(default=0.0, min=0.0, max=1.0)


class FLIPWATER_WakeBakeState(PropertyGroup):
    is_baking: BoolProperty(default=False)
    bake_current_frame: IntProperty(default=0)
    bake_progress: FloatProperty(default=0.0, min=0.0, max=1.0)


_CLASSES = (
    FLIPWATER_DomainSettings,
    FLIPWATER_EmitterSettings,
    FLIPWATER_ObstacleSettings,
    FLIPWATER_SinkSettings,
    FLIPWATER_MpmBakeState,
    FLIPWATER_WakeBakeState,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Object.flip_water_is_domain = BoolProperty(default=False)
    bpy.types.Object.flip_water_is_emitter = BoolProperty(default=False)
    bpy.types.Object.flip_water_is_obstacle = BoolProperty(default=False)
    bpy.types.Object.flip_water_is_sink = BoolProperty(default=False)

    bpy.types.Object.flip_water_domain = PointerProperty(type=FLIPWATER_DomainSettings)
    bpy.types.Object.flip_water_emitter = PointerProperty(type=FLIPWATER_EmitterSettings)
    bpy.types.Object.flip_water_obstacle = PointerProperty(type=FLIPWATER_ObstacleSettings)
    bpy.types.Object.flip_water_sink = PointerProperty(type=FLIPWATER_SinkSettings)

    bpy.types.Scene.flip_water_mpm = PointerProperty(type=FLIPWATER_MpmBakeState)
    bpy.types.Scene.flip_water_wake = PointerProperty(type=FLIPWATER_WakeBakeState)


def unregister():
    del bpy.types.Scene.flip_water_wake
    del bpy.types.Scene.flip_water_mpm
    del bpy.types.Object.flip_water_sink
    del bpy.types.Object.flip_water_obstacle
    del bpy.types.Object.flip_water_emitter
    del bpy.types.Object.flip_water_domain
    del bpy.types.Object.flip_water_is_sink
    del bpy.types.Object.flip_water_is_obstacle
    del bpy.types.Object.flip_water_is_emitter
    del bpy.types.Object.flip_water_is_domain

    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
