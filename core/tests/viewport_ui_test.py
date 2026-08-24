"""Headless validation of the viewport N-panel role assignment."""
import bpy
import sys

# ── 0. Register the addon straight from the repo (avoids extension-enable
#       state issues in headless runs) ──
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
import flip_water_addon  # noqa: E402
flip_water_addon.register()
print("addon registered from repo")

# ── 1. Fresh scene with three mesh objects ──
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 1))
domain = bpy.context.object
domain.name = "DomainBox"
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(2, 2, 2))
emitter = bpy.context.object
emitter.name = "EmitterA"
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.3, location=(-2, 2, 2))
emitter2 = bpy.context.object
emitter2.name = "EmitterB"
bpy.ops.mesh.primitive_cube_add(size=0.6, location=(0, 2, 2))
collider = bpy.context.object
collider.name = "ColliderA"
bpy.ops.mesh.primitive_cube_add(size=0.4, location=(2, 2, 2))
collider2 = bpy.context.object
collider2.name = "ColliderB"

def set_active(obj):
    bpy.context.view_layer.objects.active = obj

# ── 2. Assign roles via the new operators ──
set_active(domain)
res = bpy.ops.flip_water.vp_assign_role(role='DOMAIN')
print("assign DOMAIN:", res)

set_active(emitter)
res = bpy.ops.flip_water.vp_assign_role(role='EMITTER')
print("assign EMITTER A:", res)

# Single emitter must connect DIRECTLY to the solver - no Merge node yet.
tree = None
for ng in bpy.data.node_groups:
    if ng.bl_idname == "FLIPWATER_NodeTree":
        tree = ng
        break
assert tree is not None
solver = next(n for n in tree.nodes if n.bl_idname == "FLIPWATER_ND_solver")
assert all(l.from_node.bl_idname != "FLIPWATER_ND_merge"
           for l in solver.inputs["Points"].links), "single emitter must not use a Merge"
assert len(solver.inputs["Points"].links) == 1
assert solver.inputs["Points"].links[0].from_node.bl_idname == "FLIPWATER_ND_emitter"
assert len([n for n in tree.nodes if n.bl_idname == "FLIPWATER_ND_merge"]) == 0
print("single emitter direct link, no Merge ✓")

set_active(emitter2)
res = bpy.ops.flip_water.vp_assign_role(role='EMITTER')
print("assign EMITTER B:", res)

set_active(collider)
res = bpy.ops.flip_water.vp_assign_role(role='COLLIDER')
print("assign COLLIDER A:", res)

# Single collider must also connect directly - still only the Points merge.
assert all(l.from_node.bl_idname != "FLIPWATER_ND_merge"
           for l in solver.inputs["Obstacles"].links), "single collider must not use a Merge"
assert len(solver.inputs["Obstacles"].links) == 1
assert solver.inputs["Obstacles"].links[0].from_node.bl_idname == "FLIPWATER_ND_obstacle"
assert len([n for n in tree.nodes if n.bl_idname == "FLIPWATER_ND_merge"]) == 1
print("single collider direct link, no Merge ✓")

set_active(collider2)
res = bpy.ops.flip_water.vp_assign_role(role='COLLIDER')
print("assign COLLIDER B:", res)

# ── 3. Verify node tree wiring ──
domains, emitters, colliders, merges = [], [], [], []
for n in tree.nodes:
    if n.bl_idname == "FLIPWATER_ND_solver":
        solver = n
    elif n.bl_idname == "FLIPWATER_ND_domain" and n.domain_object:
        domains.append(n.domain_object.name)
    elif n.bl_idname == "FLIPWATER_ND_emitter" and n.emitter_object:
        emitters.append(n.emitter_object.name)
    elif n.bl_idname == "FLIPWATER_ND_obstacle" and n.obstacle_object:
        colliders.append(n.obstacle_object.name)
    elif n.bl_idname == "FLIPWATER_ND_merge":
        merges.append(n)

print("solver:", solver is not None)
print("domains:", domains)
print("emitters:", sorted(emitters))
print("colliders:", sorted(colliders))
print("merge nodes:", len(merges))

assert solver is not None
assert domains == ["DomainBox"], domains
assert sorted(emitters) == ["EmitterA", "EmitterB"], emitters
assert sorted(colliders) == ["ColliderA", "ColliderB"], colliders
assert len(merges) == 2, "expected one merge for Points + one for Obstacles"

# Merge wiring: both emitters through a Points merge, both colliders through
# an Obstacles merge (merges only exist because there are 2+ sources).
points_src = [l.from_node for l in solver.inputs["Points"].links]
obst_src = [l.from_node for l in solver.inputs["Obstacles"].links]
print("Points input sources:", [n.bl_idname for n in points_src])
print("Obstacles input sources:", [n.bl_idname for n in obst_src])
assert len(points_src) == 1 and points_src[0].bl_idname == "FLIPWATER_ND_merge"
assert len(obst_src) == 1 and obst_src[0].bl_idname == "FLIPWATER_ND_merge"

# Every emitter node output must reach the Points merge
for n in tree.nodes:
    if n.bl_idname == "FLIPWATER_ND_emitter" and n.emitter_object:
        assert n.outputs["Points"].links, f"{n.name} not linked"
        assert n.outputs["Points"].links[0].to_node.bl_idname == "FLIPWATER_ND_merge"

# ── 4. Object tags ──
assert domain.flip_water_is_domain
assert emitter.flip_water_is_emitter and emitter2.flip_water_is_emitter
assert collider.flip_water_is_obstacle and collider2.flip_water_is_obstacle
print("object tags OK")

# ── 4b. Role objects display as wireframe ──
assert domain.display_type == 'WIRE', domain.display_type
assert emitter.display_type == 'WIRE', emitter.display_type
assert collider.display_type == 'WIRE', collider.display_type
print("wireframe display OK")

# ── 5. Selection state works ──
bpy.ops.flip_water.vp_select_item(role='EMITTER', index=1)
assert bpy.context.scene.flip_water_vp.emitter_index == 1
print("select state OK")

# ── 6. Remove an emitter: Points merge must collapse to a direct link ──
res = bpy.ops.flip_water.vp_remove_role(role='EMITTER', obj_name='EmitterA')
print("remove EmitterA:", res)
assert not emitter.flip_water_is_emitter
remaining = [n.emitter_object.name for n in tree.nodes
             if n.bl_idname == "FLIPWATER_ND_emitter" and n.emitter_object]
assert remaining == ["EmitterB"], remaining
print("remaining emitters:", remaining)

points_src = [l.from_node for l in solver.inputs["Points"].links]
assert len(points_src) == 1 and points_src[0].bl_idname == "FLIPWATER_ND_emitter", \
    "EmitterB must connect directly after merge collapse"
assert len([n for n in tree.nodes if n.bl_idname == "FLIPWATER_ND_merge"]) == 1
print("Points merge collapsed to direct link ✓")

# ── 6b. Remove a collider: Obstacles merge must collapse too ──
res = bpy.ops.flip_water.vp_remove_role(role='COLLIDER', obj_name='ColliderA')
print("remove ColliderA:", res)
assert not collider.flip_water_is_obstacle
remaining_c = [n.obstacle_object.name for n in tree.nodes
               if n.bl_idname == "FLIPWATER_ND_obstacle" and n.obstacle_object]
assert remaining_c == ["ColliderB"], remaining_c
obst_src = [l.from_node for l in solver.inputs["Obstacles"].links]
assert len(obst_src) == 1 and obst_src[0].bl_idname == "FLIPWATER_ND_obstacle", \
    "ColliderB must connect directly after merge collapse"
assert len([n for n in tree.nodes if n.bl_idname == "FLIPWATER_ND_merge"]) == 0
print("Obstacles merge collapsed to direct link ✓")

# ── 7. Panels registered ──
import bl_ui  # noqa: F401
found = [p.bl_idname for p in (bpy.types.FLIPWATER_PT_setup,
                               bpy.types.FLIPWATER_PT_domain,
                               bpy.types.FLIPWATER_PT_emitters,
                               bpy.types.FLIPWATER_PT_colliders)]
print("panels registered:", found)

print("\nALL VIEWPORT PANEL CHECKS PASSED")
