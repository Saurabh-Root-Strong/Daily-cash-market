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

    def fake_backfill(days, **kw):                    # the retry fills both gaps
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
    monkeypatch.setattr(orch, "run_fno_backfill", lambda days, **kw: None)
    called = []
    monkeypatch.setattr(ip, "get_index_predictions", lambda *a, **k: called.append(1))
    cmd_backfill_fno(SimpleNamespace(days=5, rederive_recent=0))
    assert not called


# ── same-evening retry (26 Oct 2026: NSE's first F&O file lacked the November contracts,
# the sanity gate refused it, and the 23:30 run never asked again because the target list
# started at yesterday) ─────────────────────────────────────────────────────────────────
class _Clock:
    def __init__(self, when):
        self.when = when

    def now(self):
        return self.when


def _run(monkeypatch, when, include_today, present):
    from datetime import datetime
    asked = []
    monkeypatch.setattr(orch, "datetime", _Clock(datetime.fromisoformat(when)))
    monkeypatch.setattr(orch, "_weekdays_back", lambda n: [date(2026, 10, 23)])
    monkeypatch.setattr("src.data.schema.initialize_schema", lambda: None)
    fake_repo = SimpleNamespace(get_distinct_dates=lambda t: set(present),
                                upsert_fno_bhavcopy=lambda df: None)

    class _Fetcher:
        def __init__(self, client):
            pass

        def fetch(self, d):
            asked.append(d)
            return pd.DataFrame()

    monkeypatch.setattr(orch, "FNOBhavCopyFetcher", _Fetcher)
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    orch.run_fno_backfill(days=1, client=object(), repo=fake_repo, include_today=include_today)
    return asked


def test_the_night_run_retries_todays_refused_file(monkeypatch):
    assert _run(monkeypatch, "2026-10-26T23:30:00", True,
                [date(2026, 10, 23)]) == [date(2026, 10, 26)]


def test_today_is_not_asked_for_before_nse_publishes(monkeypatch):
    assert _run(monkeypatch, "2026-10-26T10:00:00", True, [date(2026, 10, 23)]) == []


def test_today_is_not_refetched_when_already_stored(monkeypatch):
    assert _run(monkeypatch, "2026-10-26T23:30:00", True,
                [date(2026, 10, 23), date(2026, 10, 26)]) == []


def test_plain_backfill_callers_are_unchanged(monkeypatch):
    assert _run(monkeypatch, "2026-10-26T23:30:00", False, [date(2026, 10, 23)]) == []


def test_the_cli_turns_the_retry_on(monkeypatch):
    seen = {}
    monkeypatch.setattr(repo, "query_dataframe", lambda s, p: pd.DataFrame({"trade_date": []}))
    monkeypatch.setattr(orch, "run_fno_backfill", lambda days, **kw: seen.update(kw))
    cmd_backfill_fno(SimpleNamespace(days=5, rederive_recent=0))
    assert seen.get("include_today") is True


def test_a_holiday_night_does_not_rewrite_the_previous_session(monkeypatch):
    """Today's request falls back to NSE's no-date 'latest' file, which on a holiday serves
    the PREVIOUS session; writing it would DELETE and re-insert a stored day."""
    from datetime import datetime
    written = []
    monkeypatch.setattr(orch, "datetime", _Clock(datetime.fromisoformat("2026-11-24T23:30:00")))
    monkeypatch.setattr(orch, "_weekdays_back", lambda n: [date(2026, 11, 23)])
    monkeypatch.setattr("src.data.schema.initialize_schema", lambda: None)
    fake_repo = SimpleNamespace(get_distinct_dates=lambda t: {date(2026, 11, 23)},
                                upsert_fno_bhavcopy=lambda df: written.append(df))

    class _Fetcher:
        def __init__(self, client):
            pass

        def fetch(self, d):
            return pd.DataFrame({"trade_date": [date(2026, 11, 23)], "symbol": ["X"]})

    monkeypatch.setattr(orch, "FNOBhavCopyFetcher", _Fetcher)
    monkeypatch.setattr(orch.time, "sleep", lambda s: None)
    orch.run_fno_backfill(days=1, client=object(), repo=fake_repo, include_today=True)
    assert written == []
