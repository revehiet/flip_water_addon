import sys
import bpy
from bpy.types import AddonPreferences
from bpy.props import BoolProperty, StringProperty

from . import solver_bridge


def _tag_node_editors_redraw(_self, _context):
    """Preference update callback: refresh every node editor so the
    N-panel / on-node parameter switch takes effect immediately."""
    try:
        from . import panels
        if hasattr(panels, "apply_npanel_node_widths"):
            panels.apply_npanel_node_widths()
    except Exception:  # noqa: BLE001 - panels may be mid-(re)load
        pass
    try:
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'NODE_EDITOR':
                    area.tag_redraw()
    except Exception:  # noqa: BLE001 - redraw is best-effort
        pass


class FLIPWATER_AddonPreferences(AddonPreferences):
    bl_idname = __package__

    build_python_executable: StringProperty(
        name="Build Python Executable",
        description=(
            "Path to a standalone Python interpreter matching this Blender's "
            "bundled Python version, used ONLY to compile the solver (Blender's "
            "own bundled Python cannot compile C extensions - it ships without "
            "development headers). Install a matching version from python.org "
            "and point this at its python/python.exe"
        ),
        subtype='FILE_PATH',
        default="",
    )

    node_params_in_npanel: BoolProperty(
        name="Node Params in N-Panel",
        description=(
            "A/B experiment: draw node parameters and action buttons in the "
            "node editor's N-panel ('FLIP Water' category) instead of on the "
            "node bodies, leaving the nodes as pure input/output stubs"
        ),
        default=False,
        update=_tag_node_editors_redraw,
    )

    dsph_root: StringProperty(
        name="DualSPHysics Root",
        description=(
            "Folder containing the DualSPHysics executables (GenCase_win64.exe, "
            "DualSPHysics5.4_win64.exe or DualSPHysics5.4CPU_win64.exe, "
            "PartVTK_win64.exe) - either an official package or a local build "
            "with bin/windows. DualSPHysics is LGPL-licensed and runs as an "
            "external process; it is never shipped with this addon"
        ),
        subtype='DIR_PATH',
        default="",
    )

    def draw(self, context):
        layout = self.layout
        needed = f"{sys.version_info.major}.{sys.version_info.minor}"

        box = layout.box()
        col = box.column()
        col.label(text=f"This Blender is running Python {needed}.{sys.version_info.micro}", icon='INFO')

        module, err = solver_bridge.load()
        if module is not None:
            col.label(text="FLIP solver core: loaded ✓", icon='CHECKMARK')
            if getattr(module, "openmp_enabled", False):
                threads = getattr(module, "openmp_max_threads", 1)
                col.label(text=f"CPU (OpenMP): {threads} threads ✓", icon='CHECKMARK')
            if getattr(module, "cuda_enabled", False):
                col.label(text="GPU (CUDA): enabled ✓", icon='CHECKMARK')

        layout.separator()
        layout.prop(self, "build_python_executable")
        row = layout.row()
        row.scale_y = 1.3
        if module is not None:
            row.label(text="Solver is ready — no build needed", icon='CHECKMARK')
        else:
            row.operator("flip_water.build_solver", icon='SETTINGS')

        row = layout.row()
        row.scale_y = 1.1
        row.operator("flip_water.reload_scripts", icon='FILE_REFRESH')

        layout.separator()
        box = layout.box()
        box.label(text="Interface", icon='WINDOW')
        box.prop(self, "node_params_in_npanel")

        layout.separator()
        box = layout.box()
        box.label(text="DualSPHysics (SPH solver bridge)", icon='PHYSICS')
        box.prop(self, "dsph_root")
        root = self.dsph_root.strip()
        if not root:
            box.label(text="Set the path to a DualSPHysics install/build above",
                      icon='INFO')
        else:
            try:
                from . import dsph_bridge
                tools = dsph_bridge.find_install(root)
                labels = (("gencase", "GenCase"), ("gpu", "GPU Solver"),
                          ("cpu", "CPU Solver"), ("partvtk", "PartVTK"))
                for key, label in labels:
                    ok = bool(tools.get(key))
                    box.label(text=f"{label}: {'found' if ok else 'missing'}",
                              icon='CHECKMARK' if ok else 'X')
                if tools.get("gpu") or tools.get("cpu"):
                    box.label(text="DualSPHysics ready", icon='CHECKMARK')
            except Exception as e:  # noqa: BLE001 - status probe is best-effort
                box.label(text=f"Probe failed: {e}", icon='ERROR')

        layout.separator()
        col = layout.column(align=True)
        col.label(text="Setup steps:")
        col.label(text=f"1. Install a standalone Python {needed}.x (python.org) matching this Blender.")
        col.label(text="2. Point 'Build Python Executable' above at it.")
        col.label(text="3. Click 'Build FLIP Solver' (needs a C++ compiler + CMake on your system).")
        col.label(text="4. Add a FLIP Domain from the 3D Viewport's 'Add > Mesh' menu, or the sidebar panel.")
        col.label(text="5. During development, use 'Reload Addon Scripts' after Python edits (no reinstall needed).")


_CLASSES = (FLIPWATER_AddonPreferences,)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
