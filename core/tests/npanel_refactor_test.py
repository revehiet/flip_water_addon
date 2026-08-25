"""Guard test: every node using draw_buttons must expose _draw_params.

The N-panel A/B feature routes rendering through `<Node>._draw_params`;
a future node that adds draw_buttons without the sibling method would
silently show an empty parameter panel. This bpy-free source scan keeps
that from happening:

    python core/tests/npanel_refactor_test.py
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _classes_of(path):
    """Yield (class_name, body_text) for each class block in a source file."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^class\s+(\w+).*?:\s*$", re.M)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        yield m.group(1), text[m.start():end]


def test_every_draw_buttons_has_draw_params():
    offenders = []
    found_pairs = 0
    for relpath in ("panels.py", "nodes_wake.py"):
        for cls_name, body in _classes_of(_ROOT / relpath):
            has_gate = "def draw_buttons(" in body
            has_params = "def _draw_params(" in body
            if has_gate and not has_params:
                offenders.append(f"{relpath}:{cls_name}")
            if has_gate:
                found_pairs += 1
    assert not offenders, "missing _draw_params: %s" % ", ".join(offenders)
    assert found_pairs >= 15, (
        f"expected >=15 gated nodes, found {found_pairs} — "
        "did a node lose its draw_buttons?")


def test_preference_toggle_exists():
    prefs = (_ROOT / "preferences.py").read_text(encoding="utf-8")
    assert "node_params_in_npanel" in prefs
    assert "_tag_node_editors_redraw" in prefs


def test_npanel_panel_registered():
    panels = (_ROOT / "panels.py").read_text(encoding="utf-8")
    assert "class FLIPWATER_PT_node_params" in panels
    assert re.search(r"_CLASSES\s*=\s*\([^)]*FLIPWATER_PT_node_params",
                     panels, re.S), "panel not registered in _CLASSES"


def test_gates_have_no_invalid_icon_and_use_width_hook():
    """The original gate used icon='UI' (not a valid enum -> crash) and the
    compact-stub width hook must be present in every gate. The on-node hint
    label was removed later (it forced Blender's minimum node width up)."""
    total_hooks = 0
    for relpath in ("panels.py", "nodes_wake.py"):
        text = (_ROOT / relpath).read_text(encoding="utf-8")
        assert ", icon='UI')" not in text, \
            f"{relpath}: invalid icon='UI' still present"
        assert "Params & actions" not in text, \
            f"{relpath}: on-node hint label must stay removed"
        total_hooks += text.count("_update_node_width_for_mode(self)")
    assert total_hooks >= 15, (
        f"width hook missing from some gates ({total_hooks}/15)")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(tests)} tests passed.")
