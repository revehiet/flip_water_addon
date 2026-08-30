bl_info = {
    "name": "FLIP Water Simulation (C++ Core)",
    "author": "FLIP Water Addon",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "Node Editor > FLIP Water",
    "description": "FLIP fluid simulation for water, powered by a compiled C++ solver core",
    "warning": "Requires a one-time build step - see Preferences > Add-ons > FLIP Water",
    "doc_url": "",
    "category": "Physics",
}

import bpy

if "properties" in locals():
    import importlib
    importlib.reload(cache_io)
    importlib.reload(domain_utils)
    importlib.reload(solver_bridge)
    importlib.reload(surface_reconstruction)
    importlib.reload(voxelize)
    importlib.reload(preview_overlay)
    importlib.reload(operators)
    importlib.reload(operators_dsph)
    importlib.reload(panels)
    importlib.reload(preferences)
    importlib.reload(properties)
    importlib.reload(handlers)
    importlib.reload(nodes_wake)
    importlib.reload(solver_wake)
    importlib.reload(evaluator_wake)
    importlib.reload(draw_wake)
    importlib.reload(viewport_ui)
    importlib.reload(wake_deformer)
else:
    from . import cache_io
    from . import domain_utils
    from . import solver_bridge
    from . import surface_reconstruction
    from . import voxelize
    from . import preview_overlay
    from . import properties
    from . import preferences
    from . import operators
    from . import operators_dsph
    from . import panels
    from . import handlers
    from . import nodes_wake
    from . import solver_wake
    from . import evaluator_wake
    from . import draw_wake
    from . import viewport_ui
    from . import wake_deformer

_MODULES = (
    properties, preferences, preview_overlay, operators, operators_dsph,
    panels, handlers,
    nodes_wake, evaluator_wake, draw_wake, viewport_ui, wake_deformer,
)


def register():
    for m in _MODULES:
        m.register()


def unregister():
    for m in reversed(_MODULES):
        m.unregister()


if __name__ == "__main__":
    register()
