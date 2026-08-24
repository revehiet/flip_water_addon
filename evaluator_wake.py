"""Evaluator — walks the WakePoints tree in topological order, calls each node's
evaluate(), caches results. Triggered by frame_change_post handler."""

import bpy
import numpy as np

# Module-level cache for the last evaluated point array (for the draw handler)
_last_points = None
_last_point_size = 6.0
_last_color = (1.0, 1.0, 1.0, 1.0)


def clear_cache():
    global _last_points
    _last_points = None


def get_last_points():
    """Called by draw handler. Returns (points_array, point_size, color) or None."""
    return _last_points, _last_point_size, _last_color


def _topological_sort(nodes):
    """Kahn's algorithm. Returns list of nodes in dependency order."""
    in_degree = {}
    adj = {}
    for node in nodes:
        in_degree[node] = 0
        adj[node] = []

    for node in nodes:
        for inp in node.inputs:
            for link in inp.links:
                src = link.from_node
                adj.setdefault(src, []).append(node)
                in_degree[node] = in_degree.get(node, 0) + 1

    queue = [n for n in nodes if in_degree.get(n, 0) == 0]
    result = []
    while queue:
        n = queue.pop(0)
        result.append(n)
        for downstream in adj.get(n, []):
            in_degree[downstream] -= 1
            if in_degree[downstream] == 0:
                queue.append(downstream)
    return result


def _ancestors(tree, node):
    """All nodes transitively upstream of `node` (by name)."""
    seen = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if n.name in seen:
            continue
        seen.add(n.name)
        for inp in n.inputs:
            for link in inp.links:
                if link.from_node is not None:
                    stack.append(link.from_node)
    seen.discard(node.name)
    return seen


def _serving_cache_nodes(tree, context):
    """CacheNodes whose Load-From-Disk toggle is on AND whose frame file
    exists — they short-circuit their upstream simulation."""
    from . import nodes_wake
    frame = context.scene.frame_current
    serving = []
    for node in tree.nodes:
        if node.bl_idname != "WakeCacheNode":
            continue
        if node.load_from_disk and nodes_wake.wake_cache_load(node, frame) is not None:
            serving.append(node)
    return serving


def evaluate_tree(tree, context):
    """Topologically evaluate all nodes in tree. Stores final point array.
    Load-From-Disk cache nodes short-circuit their upstream subgraph."""
    global _last_points, _last_point_size, _last_color

    nodes = [n for n in tree.nodes if n.bl_idname != 'NodeReroute']
    if not nodes:
        _last_points = None
        return

    # Skip upstream nodes behind a serving cache node (Houdini File Cache
    # semantics: loading from disk bypasses the upstream cook). `skip` holds
    # node names, so compare against node.name.
    skip = set()
    for serving in _serving_cache_nodes(tree, context):
        skip |= _ancestors(tree, serving)

    ordered = [n for n in _topological_sort(nodes) if n.name not in skip]
    results = {}  # (node, socket_identifier) -> data

    for node in ordered:
        # Nodes without an evaluate() (e.g. Wake Deformer, which only
        # provides live parameters through links) are skipped.
        if not hasattr(node, "evaluate"):
            continue
        # Gather inputs from upstream results
        inputs = {}
        for inp in node.inputs:
            if inp.is_linked:
                link = inp.links[0]
                upstream_data = results.get((link.from_node, link.from_socket.identifier))
                inputs[inp.identifier] = upstream_data
            else:
                inputs[inp.identifier] = None

        # Evaluate this node
        try:
            output = node.evaluate(context, inputs)
        except Exception as e:
            print(f"[Wake Eval] Error in {node.name}: {e}")
            output = {}

        # Store outputs
        for out in node.outputs:
            val = output.get(out.identifier) if output else None
            results[(node, out.identifier)] = val

        # If this is a DrawPoints node, capture its output for rendering
        if node.bl_idname == "WakeDrawPointsNode":
            pts = output.get("Points") if output else None
            if pts is not None and pts.shape[0] > 0:
                _last_points = pts[:, :2].astype(np.float32)  # XY positions
                _last_point_size = output.get("point_size", 6.0)
                _last_color = output.get("color", (1.0, 1.0, 1.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Handlers
# ═══════════════════════════════════════════════════════════════════════════

def on_frame_change(scene, depsgraph=None):
    """Called on frame change. Evaluate all WakePoints trees."""
    context = bpy.context
    for tree in bpy.data.node_groups:
        if tree.bl_idname == "WakePointsTreeType":
            evaluate_tree(tree, context)


_frame_handler = None


def register():
    global _frame_handler
    _frame_handler = bpy.app.handlers.frame_change_post.append(on_frame_change)


def unregister():
    global _frame_handler
    if _frame_handler is not None:
        bpy.app.handlers.frame_change_post.remove(on_frame_change)
        _frame_handler = None
    clear_cache()
