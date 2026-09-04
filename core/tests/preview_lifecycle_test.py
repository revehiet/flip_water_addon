"""Guard the shared viewport-preview lifecycle across solver families.

Run headless:
    python core/tests/preview_lifecycle_test.py
"""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def test_all_solver_seed_previews_sync_from_the_node_graph():
    handlers = (_ROOT / "handlers.py").read_text(encoding="utf-8")
    for call in (
        "operators.sync_seed_previews_from_node_graph(bpy.context)",
        "operators.sync_mpm_seed_previews_from_node_graph(bpy.context)",
        "operators_dsph.sync_dsph_seed_previews_from_node_graph(bpy.context)",
        "operators_smoke.sync_smoke_seed_previews_from_node_graph(bpy.context)",
    ):
        assert call in handlers, f"missing preview synchronizer: {call}"


def test_loading_or_unregistering_clears_gpu_batches():
    handlers = (_ROOT / "handlers.py").read_text(encoding="utf-8")
    assert "def flip_water_load_post" in handlers
    assert "bpy.app.handlers.load_post.append(flip_water_load_post)" in handlers
    assert handlers.count("preview_overlay.clear_all()") >= 2


def test_each_solver_drops_stale_preview_bookkeeping():
    for filename in ("operators.py", "operators_dsph.py", "operators_smoke.py"):
        source = (_ROOT / filename).read_text(encoding="utf-8")
        assert "def reset_preview_state():" in source, f"missing reset hook: {filename}"


def test_baked_flip_preview_requires_a_live_solver_node():
    source = (_ROOT / "operators.py").read_text(encoding="utf-8")
    assert "def _domain_has_linked_flip_solver(domain_obj):" in source
    update_start = source.index("def update_baked_domain_overlay(domain_obj, frame):")
    update_source = source[update_start:source.index("def update_whitewater_overlay", update_start)]
    assert "if not _domain_has_linked_flip_solver(domain_obj):" in update_source


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\n{len(tests)} tests passed.")