"""
Prediction Memory Dashboard — historical accuracy, similar days, signal calibration.

Sections:
  1. Accuracy Summary      — rolling 30/60/90D accuracy per symbol + direction
  2. Similar Days Panel    — top-7 most similar past days to today + outcomes
  3. Memory Signal         — what history says about today's market condition
  4. Calibration Chart     — confidence level vs actual hit rate
  5. Prediction Log Table  — raw history with outcomes
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.memory_engine import (
    get_accuracy_report,
    get_memory_signal,
    get_prediction_log,
    get_pending_predictions,
)
from src.dashboard.cache.queries import cached_index_predictions

_SYMBOLS    = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
_SYM_LABELS = {"NIFTY": "Nifty 50", "BANKNIFTY": "Bank Nifty",
               "FINNIFTY": "Fin Nifty", "MIDCPNIFTY": "Midcap Nifty"}

_DIR_COLOR = {"UP": "#4CAF50", "DOWN": "#EF5350", "SIDEWAYS": "#FFD600"}
_CONF_COLOR = {"HIGH": "#00C853", "MEDIUM": "#FFD600", "LOW": "#78909C"}
_BOOL_COLOR = {True: "#4CAF50", False: "#EF5350"}


def render(selected_date: date) -> None:
    st.subheader("🧠 Prediction Memory Engine")
    st.caption(
        "Adaptive pattern memory: every prediction is stored with a market fingerprint. "
        "When market conditions today resemble past days, historical outcomes calibrate confidence."
    )

    # Symbol selector
    col_sym, col_days = st.columns([2, 1])
    with col_sym:
        sym = st.selectbox("Index", _SYMBOLS,
                           format_func=lambda s: _SYM_LABELS[s],
                           key="mem_symbol")
    with col_days:
        window = st.selectbox("Accuracy Window", [30, 60, 90], index=1,
                              format_func=lambda d: f"Last {d} days",
                              key="mem_window")

    # Load
    report = get_accuracy_report(sym, days=window)
    preds  = cached_index_predictions(selected_date)
    today_pred = next((p for p in preds if p.fno_symbol == sym), None)

    # ── 1. Accuracy Summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Overall Accuracy")

    if report.total_predictions == 0:
        st.info(
            "No prediction history yet. The memory engine builds up from today onwards. "
            "Run `python -m src.cli backfill-predictions` to seed with historical data."
        )
        _render_empty_state()
        return

    _render_accuracy_kpis(report)
    st.divider()

    # ── 2. By-Direction Accuracy ──────────────────────────────────────────────
    if report.by_direction:
        st.markdown("#### Accuracy by Predicted Direction")
        _render_direction_accuracy(report)
        st.divider()

    # ── 3. By-Confidence Accuracy ─────────────────────────────────────────────
    if report.by_confidence:
        st.markdown("#### Accuracy by Confidence Level")
        _render_confidence_accuracy(report)
        st.divider()

    # ── 4. By-Regime Accuracy ─────────────────────────────────────────────────
    if report.by_regime:
        st.markdown("#### Accuracy by HMM Market Regime")
        _render_regime_accuracy(report)
        st.divider()

    # ── 5. Similar Days + Memory Signal ──────────────────────────────────────
    if today_pred and today_pred.data_available:
        st.markdown("#### 🔍 Today's Market Pattern — Similar Historical Days")
        mem = today_pred.mem_signal
        if mem is None or mem.error:
            st.info(
                f"Memory signal not yet available: "
                f"{mem.error if mem else 'insufficient history'}. "
                f"Needs ≥15 filled predictions."
            )
        else:
            _render_memory_signal(mem, today_pred.direction)
            st.divider()
            _render_similar_days(mem)
            st.divider()

    # ── 6. Raw Prediction Log ─────────────────────────────────────────────────
    st.markdown("#### 📋 Prediction History")
    _render_prediction_log(sym)


# ── Sub-renderers ─────────────────────────────────────────────────────────────

def _render_empty_state() -> None:
    st.markdown(
        "<div style='padding:24px;background:rgba(255,255,255,0.03);"
        "border-radius:10px;border:1px dashed #444;text-align:center'>"
        "<div style='font-size:2em'>🧠</div>"
        "<div style='font-size:1.1em;color:#aaa;margin:8px 0'>Memory Engine is empty</div>"
        "<div style='font-size:0.85em;color:#666'>Predictions are stored automatically each day.<br>"
        "After 15+ days, similar-pattern search activates.<br>"
        "Run <code>python -m src.cli backfill-predictions 60</code> to seed with history.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_accuracy_kpis(report) -> None:
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    acc_color = "#4CAF50" if report.overall_accuracy >= 0.55 else (
        "#FFD600" if report.overall_accuracy >= 0.45 else "#EF5350"
    )
    c1.metric("Overall Accuracy", f"{report.overall_accuracy:.0%}",
              help="Correct direction predictions / total predictions in the window.")
    c2.metric("Total Predictions", f"{report.total_predictions}")
    c3.metric("Correct", f"{report.correct}")
    c4.metric("30D Accuracy",
              f"{report.accuracy_30d:.0%}" if report.accuracy_30d else "—",
              help="Rolling 30-day hit rate.")
    c5.metric("Avg Return (Correct)",
              f"{report.avg_return_correct:+.2f}%",
              help="Average next-day index return on correctly predicted days.")
    c6.metric("Avg Return (Wrong)",
              f"{report.avg_return_incorrect:+.2f}%",
              help="Average next-day index return on incorrectly predicted days.")

    # Accuracy gauge bar
    pct = report.overall_accuracy
    col = "#4CAF50" if pct >= 0.55 else ("#FFD600" if pct >= 0.45 else "#EF5350")
    st.markdown(
        f"<div style='margin:8px 0 4px;font-size:0.75em;color:#888'>Accuracy gauge (50% = random)</div>"
        f"<div style='height:10px;background:#2a2a2a;border-radius:5px;overflow:hidden'>"
        f"<div style='height:100%;width:{pct*100:.1f}%;background:{col};'></div>"
        f"</div>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.7em;color:#555;margin-top:2px'>"
        f"<span>0%</span><span style='color:#FFD600'>50% (random)</span><span>100%</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_direction_accuracy(report) -> None:
    rows = []
    for d in ["UP", "DOWN", "SIDEWAYS"]:
        info = report.by_direction.get(d)
        if info:
            rows.append({
                "Direction":  d,
                "Count":      info["count"],
                "Correct":    info["correct"],
                "Accuracy":   f"{info['accuracy']:.0%}",
                "Avg Return": f"{info['avg_actual_return']:+.2f}%",
            })

    if not rows:
        return

    fig = go.Figure()
    dirs   = [r["Direction"] for r in rows]
    accs   = [report.by_direction[d]["accuracy"] for d in dirs if d in report.by_direction]
    colors = [_DIR_COLOR.get(d, "#9E9E9E") for d in dirs]

    fig.add_trace(go.Bar(
        x=dirs, y=accs,
        marker_color=colors,
        text=[f"{a:.0%}" for a in accs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#FFD600", line_width=1,
                  annotation_text="50% random", annotation_position="right",
                  annotation_font=dict(size=10, color="#FFD600"))
    fig.update_layout(
        height=220, template="plotly_dark",
        yaxis=dict(tickformat=".0%", range=[0, 1]),
        margin=dict(t=20, b=30, l=50, r=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="dir_accuracy")

    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)


def _render_confidence_accuracy(report) -> None:
    fig = go.Figure()
    confs = ["HIGH", "MEDIUM", "LOW"]
    accs  = [report.by_confidence.get(c, {}).get("accuracy", None) for c in confs]
    cnts  = [report.by_confidence.get(c, {}).get("count", 0)      for c in confs]
    colors= [_CONF_COLOR.get(c, "#9E9E9E") for c in confs]

    valid = [(c, a, n, col) for c, a, n, col in zip(confs, accs, cnts, colors) if a is not None]
    if not valid:
        return

    cs, aa, ns, ccs = zip(*valid)
    fig.add_trace(go.Bar(
        x=list(cs), y=list(aa),
        marker_color=list(ccs),
        text=[f"{a:.0%} ({n})" for a, n in zip(aa, ns)],
        textposition="outside",
    ))
    fig.add_hline(y=0.5, line_dash="dot", line_color="#FFD600", line_width=1)
    fig.update_layout(
        height=200, template="plotly_dark",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        margin=dict(t=20, b=30, l=50, r=20),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="conf_accuracy")
    st.caption("Calibration insight: HIGH confidence predictions should have the highest accuracy. "
               "If LOW > HIGH, the confidence scoring needs re-calibration.")


def _render_regime_accuracy(report) -> None:
    rows = []
    for regime in ["Bull", "Sideways", "Bear"]:
        info = report.by_regime.get(regime)
        if info and info["count"] >= 3:
            rows.append({
                "HMM Regime": regime,
                "Count":      info["count"],
                "Accuracy":   f"{info['accuracy']:.0%}",
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            "When HMM detects Bull/Bear regime, accuracy should be higher than Sideways. "
            "Sideways regime has no momentum — random walk performance expected."
        )


def _render_memory_signal(mem, today_direction: str) -> None:
    conf_text  = ("CONFIRMS" if mem.confirms_prediction is True else
                  "CONTRADICTS" if mem.confirms_prediction is False else "UNCERTAIN about")
    conf_color = ("#4CAF50" if mem.confirms_prediction is True else
                  "#EF5350" if mem.confirms_prediction is False else "#FFD600")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Similar Days Found", mem.similar_count,
              help=f"Days with market fingerprint closest to today (top-{mem.similar_count} from history).")
    c2.metric("UP%",       f"{mem.memory_up_pct:.0%}")
    c3.metric("DOWN%",     f"{mem.memory_dn_pct:.0%}")
    c4.metric("Hist. Accuracy", f"{mem.memory_accuracy:.0%}",
              help="Hit rate among similar days — how often prediction was correct in those sessions.")
    c5.metric("Avg Next-Day Return", f"{mem.avg_actual_return:+.2f}%",
              help="Average actual next-day return on the similar past days.")

    # Probability bar
    fig = go.Figure()
    for label, pct, color in [
        ("UP",       mem.memory_up_pct, "#4CAF50"),
        ("SIDEWAYS", mem.memory_sw_pct, "#FFD600"),
        ("DOWN",     mem.memory_dn_pct, "#EF5350"),
    ]:
        fig.add_trace(go.Bar(
            name=label, x=[pct], y=["Outcome distribution"],
            orientation="h", marker_color=color,
            text=f"<b>{label}</b> {pct:.0%}",
            textposition="inside" if pct > 0.10 else "outside",
            insidetextanchor="middle",
            showlegend=False,
        ))
    fig.update_layout(
        barmode="stack", height=60, template="plotly_dark",
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis=dict(visible=False),
        margin=dict(t=5, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, key="mem_outcome_bar")

    # Verdict block
    st.markdown(
        f"<div style='padding:12px 16px;border-left:4px solid {conf_color};"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;font-size:13px'>"
        f"<b style='color:{conf_color}'>Memory {conf_text} the current {today_direction} prediction</b><br>"
        f"<span style='color:rgba(255,255,255,0.6);font-size:12px'>{mem.memory_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_similar_days(mem) -> None:
    st.markdown("#### Most Similar Historical Days")
    st.caption(
        "These past trading days had the most similar market fingerprint to today "
        "(PCR · FII net · VIX · carry · breadth · Hurst · entropy · score). "
        "Their outcomes inform the memory signal."
    )

    if not mem.similar_days:
        st.info("No similar days to display.")
        return

    rows = []
    for sd in mem.similar_days:
        correct_icon = "✅" if sd.was_correct else "❌"
        rows.append({
            "Date":         sd.trade_date.strftime("%d %b %Y"),
            "Similarity":   f"{sd.similarity:.3f}",
            "Predicted":    sd.direction_pred,
            "Actual":       sd.direction_actual,
            "Correct":      correct_icon,
            "Return":       f"{sd.actual_return:+.2f}%",
            "Score":        f"{sd.composite_score:+.1f}",
            "HMM":          sd.hmm_state,
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Similarity": st.column_config.ProgressColumn(
                "Similarity", min_value=0, max_value=1, format="%.3f"
            ),
        },
    )


def _render_prediction_log(symbol: str) -> None:
    try:
        history  = get_prediction_log(symbol, limit=60)
        unfilled = get_pending_predictions(symbol)

        if history.empty and unfilled.empty:
            st.info("No predictions logged yet.")
            return

        all_rows = []

        for _, row in history.head(60).iterrows():
            td = row["trade_date"]
            if hasattr(td, "date"):
                td = td.date()
            correct_icon = "✅" if row.get("was_correct") else "❌"
            all_rows.append({
                "Date":       td.strftime("%d %b %Y"),
                "Pred":       str(row.get("direction_pred", "—")),
                "Conf":       str(row.get("confidence_pred", "—")),
                "Score":      f"{row.get('composite_score', 0):+.1f}",
                "Actual":     str(row.get("direction_actual", "—")),
                "Return":     f"{row.get('actual_return', 0):+.2f}%",
                "Correct":    correct_icon,
                "HMM":        str(row.get("hmm_state") or "—"),
                "Hurst":      f"{row.get('feat_hurst', 0):.3f}" if row.get("feat_hurst") else "—",
                "Status":     "✅ Filled",
            })

        for _, row in unfilled.iterrows():
            td = row["trade_date"]
            if hasattr(td, "date"):
                td = td.date()
            all_rows.append({
                "Date":    td.strftime("%d %b %Y"),
                "Pred":    str(row.get("direction_pred", "—")),
                "Conf":    str(row.get("confidence_pred", "—")),
                "Score":   f"{row.get('composite_score', 0):+.1f}",
                "Actual":  "—",
                "Return":  "—",
                "Correct": "⏳",
                "HMM":     str(row.get("hmm_state") or "—"),
                "Hurst":   f"{row.get('feat_hurst', 0):.3f}" if row.get("feat_hurst") else "—",
                "Status":  "⏳ Pending",
            })

        df = pd.DataFrame(all_rows)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption(
            f"Showing last {len(history)} filled + {len(unfilled)} pending predictions for {_SYM_LABELS.get(symbol, symbol)}. "
            "Pending = outcome not yet available (next trading day data not fetched yet)."
        )
    except Exception as exc:
        st.warning(f"Could not load prediction log: {exc}")
