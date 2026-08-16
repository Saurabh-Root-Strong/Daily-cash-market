"""Guards on the dashboard's caching policy.

The trap this pins: `@st.cache_data(persist="disk", ttl=...)` looks like a free
win but Streamlit's LocalDiskCacheStorageManager.check_context downgrades it to
a log line and drops the TTL —

    "The cached function '%s' has a TTL that will be ignored.
     Persistent cached functions currently don't support TTL."

so the entry never expires. A corrected trading session would keep serving
pre-correction numbers until .streamlit/cache was deleted by hand. That failure
is invisible: the page renders, the numbers look plausible, and nothing logs at
render time. Cheaper to fail the suite.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

_QUERIES = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "cache" / "queries.py"


def _cache_decorators():
    """Yield (function_name, {kwarg: literal_or_name}) for every st.cache_data."""
    tree = ast.parse(_QUERIES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            attr = dec.func
            if not (isinstance(attr, ast.Attribute) and attr.attr == "cache_data"):
                continue
            kw = {}
            for k in dec.keywords:
                if k.arg is None:
                    continue
                v = k.value
                if isinstance(v, ast.Constant):
                    kw[k.arg] = v.value
                elif isinstance(v, ast.Name):
                    kw[k.arg] = v.id
                else:
                    kw[k.arg] = "<expr>"
            yield node.name, kw


def test_the_file_actually_parses_some_caches():
    """If this ever finds nothing, the other tests here are vacuously passing."""
    found = list(_cache_decorators())
    assert len(found) > 50, f"only found {len(found)} cache_data decorators"


def test_no_cached_function_persists_to_disk():
    offenders = [n for n, kw in _cache_decorators() if kw.get("persist")]
    assert not offenders, (
        f"{offenders} use persist=. Streamlit silently ignores ttl on persisted "
        "caches, so these would never expire and would serve stale numbers "
        "across restarts. See this module's docstring.")


def test_every_cache_declares_a_ttl():
    """A cache with no ttl never expires either — same failure, different route."""
    missing = [n for n, kw in _cache_decorators() if "ttl" not in kw]
    assert not missing, f"{missing} declare no ttl and would never expire"


@pytest.mark.parametrize("fn", [
    "cached_rotation_clock_accuracy",
    "cached_rotation_clock_backtest",
    "cached_signal_backtest",
])
def test_walk_forward_backtests_use_the_backtest_ttl(fn):
    """These recompute a multi-window walk-forward and cost 8s+ each.

    cached_rotation_clock_accuracy alone is 24.6s of the Rotation Clock panel's
    29.6s (three windows). At the 300s default they were recomputed every five
    minutes despite depending only on closed history.
    """
    kw = dict(_cache_decorators()).get(fn)
    assert kw is not None, f"{fn} no longer exists or lost its cache decorator"
    assert kw.get("ttl") == "_TTL_BACKTEST", (
        f"{fn} uses ttl={kw.get('ttl')!r}; walk-forward backtests belong on "
        "_TTL_BACKTEST")
