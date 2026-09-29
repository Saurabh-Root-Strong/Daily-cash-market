"""backfill-fno --rederive-recent: a late-filled F&O day gets its predictions and
sector snapshot rebuilt; old (deep backfill) days and already-present days do not."""
from datetime import date
from types import SimpleNamespace

import pandas as pd

import src.analytics.index_prediction as ip
import src.analytics.sector_memory as sm
import src.data.repository as repo
import src.ingestion.orchestrator as orch
from src.cli import cmd_backfill_fno


def test_rederives_only_recent_newly_filled_days(monkeypatch):
    days = [date(2026, 9, d) for d in (21, 22, 23, 24, 25, 28)]
    state = {"dates": [days[0]] + days[2:5]}          # 22nd (old gap) and 28th (late) missing

    def fake_query(sql, params):
        return pd.DataFrame({"trade_date": pd.to_datetime(state["dates"])})

    def fake_backfill(days):                          # the retry fills both gaps
        state["dates"] = list(days_all)

    days_all = days
    calls = []
    monkeypatch.setattr(repo, "query_dataframe", fake_query)
    monkeypatch.setattr(orch, "run_fno_backfill", fake_backfill)
    monkeypatch.setattr(ip, "get_index_predictions", lambda d, persist: calls.append(("pred", d)) or [1, 2])
    monkeypatch.setattr(sm, "record_daily_snapshot", lambda d, replace: calls.append(("snap", d, replace)) or 24)

    cmd_backfill_fno(SimpleNamespace(days=5, rederive_recent=2))
    # only the 28th is both newly filled AND among the last 2 F&O sessions
    assert calls == [("pred", date(2026, 9, 28)), ("snap", date(2026, 9, 28), True)]


def test_no_rederive_without_flag(monkeypatch):
    monkeypatch.setattr(repo, "query_dataframe", lambda s, p: pd.DataFrame({"trade_date": []}))
    monkeypatch.setattr(orch, "run_fno_backfill", lambda days: None)
    called = []
    monkeypatch.setattr(ip, "get_index_predictions", lambda *a, **k: called.append(1))
    cmd_backfill_fno(SimpleNamespace(days=5, rederive_recent=0))
    assert not called
