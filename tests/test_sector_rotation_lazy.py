"""Guard: the Sector Rotation page must render ONE panel per run.

Why this exists. The page used st.tabs. `with tab:` is a plain context manager,
so every one of the seven panel bodies executed on every rerun and Streamlit let
the browser hide six of them. Measured 2026-08-16 under streamlit's AppTest, one
page load cost 61.3s of which the panel actually on screen was 0.47s. The two
worst panels — Rotation Clock 25.7s and Operator Footprint 24.9s — were paid in
full while reading Market Next Month.

Streamlit exposes no way to read which st.tabs tab is active server-side, so the
fix had to own the selection. That makes this a structural property worth
pinning: if someone reintroduces st.tabs here the page silently goes back to
paying for all seven, and nothing else in the suite would notice.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.dashboard.views import sector_rotation as sr

_RENDERERS = (
    "_render_smart_money", "_render_forward_tilt", "_render_rotation_clock",
    "_render_relative_strength", "_render_month_seasonality",
    "_render_operator_footprint", "_render_market_next_month",
    "_render_index_largecap",
)


@pytest.fixture
def spy(monkeypatch):
    """Replace every panel renderer with a recorder, so nothing touches data."""
    called: list[str] = []
    for name in _RENDERERS:
        monkeypatch.setattr(
            sr, name,
            (lambda n: lambda *a, **k: called.append(n))(name))
    monkeypatch.setattr(sr.st, "subheader", lambda *a, **k: None)
    return called


def _select(monkeypatch, panel):
    monkeypatch.setattr(
        sr.st, "segmented_control", lambda *a, **k: panel)


def test_every_panel_has_a_distinct_renderer():
    assert len(_RENDERERS) == len(sr._PANELS)
    assert len(set(sr._PANELS)) == len(sr._PANELS), "duplicate panel label"


@pytest.mark.parametrize("panel", sr._PANELS)
def test_exactly_one_panel_renders(panel, spy, monkeypatch):
    _select(monkeypatch, panel)
    sr.render(date(2026, 8, 14), 1.0, all_dates=[date(2026, 8, 14)])
    assert len(spy) == 1, (
        f"selecting {panel!r} ran {len(spy)} panels ({spy}). The page must "
        "render only what is on screen — see this module's docstring.")


def test_every_panel_is_reachable(spy, monkeypatch):
    """No panel may be orphaned by a typo in its dispatch branch."""
    seen = set()
    for panel in sr._PANELS:
        spy.clear()
        _select(monkeypatch, panel)
        sr.render(date(2026, 8, 14), 1.0, all_dates=[date(2026, 8, 14)])
        assert spy, f"{panel!r} dispatched to nothing"
        seen.add(spy[0])
    assert seen == set(_RENDERERS), (
        f"unreachable renderers: {set(_RENDERERS) - seen}")


def test_deselecting_falls_back_instead_of_blanking(spy, monkeypatch):
    """segmented_control returns None when the active chip is clicked again."""
    _select(monkeypatch, None)
    sr.render(date(2026, 8, 14), 1.0, all_dates=[date(2026, 8, 14)])
    assert spy == ["_render_smart_money"], (
        "a stray second click on the active chip must fall back to the first "
        "panel, not render an empty page")


def test_every_index_largecap_column_has_a_tooltip():
    """Six tables on this panel, ~40 columns. Most shipped with no help text at
    all, which is how "Deliv z" and "Fut OI %" reached the screen unexplained."""
    import re
    from pathlib import Path
    src = Path("src/dashboard/views/sector_rotation.py").read_text(encoding="utf-8")
    start = src.index("def _render_index_largecap")
    nxt = re.search(r"\ndef ", src[start + 10:])
    body = src[start:] if nxt is None else src[start:start + 10 + nxt.start()]

    blocks = []
    for m in re.finditer(r"column_config=\{", body):
        i, depth = m.end(), 1
        while depth and i < len(body):
            depth += (body[i] == "{") - (body[i] == "}")
            i += 1
        blocks.append(body[m.end():i])
    assert len(blocks) >= 5, f"expected >=5 configured tables, found {len(blocks)}"

    missing = []
    for b in blocks:
        for col in re.findall(r'"([^"]+)":\s*st\.column_config\.\w+\(', b):
            seg = b[b.index(f'"{col}": st.column_config'):]
            nx = re.search(r'\n\s+"[^"]+":\s*st\.column_config', seg)
            if nx:
                seg = seg[:nx.start()]
            if "help=" not in seg:
                missing.append(col)
    assert not missing, f"columns rendered with no tooltip: {missing}"


def test_the_duplicate_analogue_expander_is_gone():
    """The session list superseded it; leaving both rendered the same 25 matches
    twice, the second time with nine unlabelled raw columns."""
    from pathlib import Path
    src = Path("src/dashboard/views/sector_rotation.py").read_text(encoding="utf-8")
    assert "When the flow looked like today" not in src
    assert src.count("When these numbers occurred before") == 1
