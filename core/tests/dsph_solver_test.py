"""Guard test for the DualSPHysics (DSPH) solver node pipeline.

Validates the SPH-specific conventions that are easy to break silently:
  - Bake/Free live on the Cache node (like every other solver).
  - The Domain input is optional for SPH.
  - Seed preview + cache preview toggles exist and are wired into handlers.
  - 'Cache Every N Frames' is documented as an output stride.
  - Newly exposed solver params reach write_case.

Run headless (bpy-free source scan):
    python core/tests/dsph_solver_test.py
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def test_bake_buttons_moved_to_cache_node():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    cache = panels.split("class FLIPWATER_ND_cache")[1].split("class FLIPWATER_ND_tank")[0]
    solver = panels.split("class FLIPWATER_ND_dsph_solver")[1].split("class FLIPWATER_ND_mpm_solver")[0]
    assert 'row.operator("flip_water.bake_dsph"' in cache, "Cache node must host Bake DSPH"
    assert 'row.operator("flip_water.free_dsph_cache"' in cache, "Cache node must host Free DSPH"
    assert 'flip_water.bake_dsph' not in solver, "Solver node must not host Bake"
    assert 'flip_water.free_dsph_cache' not in solver, "Solver node must not host Free"


def test_domain_optional():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    ops = (_ROOT / "operators_dsph.py").read_text(encoding="utf-8")
    assert "return None, \"\"" in panels, "_resolve_dsph_solver_domain must allow no domain"
    assert "_padded_bounds" in ops
    assert "Connect an Emitter and/or Collider" in ops
    assert "_dsph_source_nodes" in ops


def test_workspace_build_is_discovered_without_preferences():
    ops = (_ROOT / "operators_dsph.py").read_text(encoding="utf-8")
    assert 'os.path.join(os.path.dirname(__file__), "third_party",' in ops
    assert '"DualSPHysics")' in ops


def test_seed_and_cache_previews_wired():
    ops = (_ROOT / "operators_dsph.py").read_text(encoding="utf-8")
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    handlers = (_ROOT / "handlers.py").read_text(encoding="utf-8")
    assert "dsph_seed_preview" in panels and "dsph_seed_preview" in ops
    assert "sync_dsph_seed_previews_from_node_graph" in ops
    assert "sync_dsph_seed_previews_from_node_graph(bpy.context)" in handlers
    assert "dsph_preview_enabled" in panels and "dsph_preview_enabled" in ops


def test_default_particle_spacing():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    m = re.search(r"dsph_dp:.*?default=([\d.]+)", panels, re.S)
    assert m and float(m.group(1)) > 0.0, "dsph_dp must have a real default > 0"


def test_frame_stride_label():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    m = re.search(r'dsph_frame_step:.*?name="([^"]+)"', panels, re.S)
    assert m and "Nth" not in m.group(1) and "Cache Every" in m.group(1)


def test_new_solver_params_exposed():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    for attr in ("dsph_density", "dsph_gravity", "dsph_kernel",
                 "dsph_visco_treatment", "dsph_cfl", "dsph_coefsound"):
        assert f"{attr}:" in panels, f"missing {attr}"


def test_new_params_reach_write_case():
    ops = (_ROOT / "operators_dsph.py").read_text(encoding="utf-8")
    for kw in ("gravity=", "rhop0=", "kernel=", "visco_treatment=",
               "cflnumber=", "coefsound="):
        assert kw in ops, f"write_case call missing {kw}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")
