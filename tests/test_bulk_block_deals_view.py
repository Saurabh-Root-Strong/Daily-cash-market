"""Headless render test for the Bulk & Block Deals page.

WHY: the page died in the browser with
    AttributeError: module 'src.analytics.deal_signals' has no attribute 'entity_leaderboard'
for a function that existed on disk, imported fine in a shell, and passed every unit
test. Nothing in the suite executed the view, so nothing could catch a view calling an
analytics function that is not there.

`streamlit.testing.v1.AppTest` runs the real render path in-process, so a renamed or
removed analytics function fails here instead of in front of the user.
"""
from __future__ import annotations

import datetime as dt

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = st_testing.AppTest


def _render_page() -> None:
    """Entry point AppTest executes as a standalone script.

    Every import must live INSIDE this function: AppTest re-executes the function's
    source in a fresh module namespace, so module-level imports of this test file are
    not in scope there.
    """
    import datetime as _dt

    from src.dashboard.views import bulk_block_deals
    from src.data.repository import query_dataframe

    dates = query_dataframe("SELECT max(trade_date) AS d FROM bulk_block_deals", [])
    latest = dates["d"].iat[0] if not dates.empty and dates["d"].iat[0] is not None else None
    if latest is None:
        latest = _dt.date.today()
    elif not isinstance(latest, _dt.date):
        latest = _dt.date.fromisoformat(str(latest)[:10])
    bulk_block_deals.render(latest)


def _has_data() -> bool:
    try:
        from src.data.repository import query_dataframe
        n = query_dataframe("SELECT count(*) AS n FROM bulk_block_deals", [])
        return bool(n["n"].iat[0])
    except Exception:
        return False


@pytest.mark.skipif(not _has_data(), reason="bulk_block_deals table not populated")
def test_page_renders_without_exception():
    at = AppTest.from_function(_render_page, default_timeout=180).run()
    assert not at.exception, (
        "Bulk & Block Deals page raised: "
        + "; ".join(str(e.message) for e in at.exception)
    )


@pytest.mark.skipif(not _has_data(), reason="bulk_block_deals table not populated")
def test_view_only_calls_analytics_functions_that_exist():
    """Every ds.<name> the view references must exist on the analytics module.

    Cheap guard against the exact failure above: the view is the only consumer of
    several of these helpers, so a rename elsewhere is otherwise silent until render.
    """
    import ast
    import inspect

    from src.analytics import deal_signals
    from src.dashboard.views import bulk_block_deals

    tree = ast.parse(inspect.getsource(bulk_block_deals))
    referenced = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "ds"
    }
    missing = sorted(n for n in referenced if not hasattr(deal_signals, n))
    assert not missing, f"view calls deal_signals.{missing} which do not exist"
