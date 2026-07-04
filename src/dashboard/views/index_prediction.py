"""
Index Prediction Dashboard — tomorrow's directional forecast for all major F&O indices.

Layout per index:
  • Prediction card — direction badge, score, PCR, carry, FII net, VIX
  • Tab 1 Verdict & Signals  — verdict panel + signal breakdown by tier
  • Tab 2 Key Levels         — put wall / max pain / call wall + carry detail
  • Tab 3 Today vs Yesterday — comparison table
  • Tab 4 OI Structure       — call/put OI chart (options chain heatmap)
  • Tab 5 Market Context     — VIX trend, sector breadth, FII positioning table
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from src.analytics.index_prediction import IndexPrediction, IndexSignal, MarketContext
from src.dashboard.cache.queries import cached_index_predictions
from src.dashboard.constants import GRID_COLOR, PAPER_BG, PLOT_BG

# ── Colour palette ────────────────────────────────────────────────────────────
_DIR_EMOJI  = {"UP": "↑", "DOWN": "↓", "SIDEWAYS": "↔"}
_CONF_COLOR = {"HIGH": "#00C853", "MEDIUM": "#FFD600", "LOW": "#78909C"}
_CAT_COLOR  = {
    "Price Action":      "#9c27b0",
    "Futures OI":        "#2196f3",
    "Options OI":        "#ff9800",
    "Carry":             "#4caf50",
    "Institutional":     "#f44336",
    "Market Context":    "#00bcd4",
    "Sector":            "#8bc34a",
    "Statistical Regime": "#607d8b",
    "Memory":            "#795548",
}
_POS  = "#69f0ae"
_NEG  = "#ff5252"
_NEU  = "#78909C"


def _bull_meter(score: float) -> tuple[int, str, str]:
    """
    Convert raw composite score (−20 to +20) → intuitive 0–100 Bull/Bear Meter.

    0   = Maximum Bearish  (score ≤ −20)
    50  = Neutral          (score =    0)
    100 = Maximum Bullish  (score ≥ +20)

    Zones:
     0–20  → STRONG BEAR  (deep red)
    21–35  → BEAR         (red)
    36–45  → MILD BEAR    (orange)
    46–54  → NEUTRAL      (grey)
    55–64  → MILD BULL    (light green)
    65–80  → BULL         (green)
    81–100 → STRONG BULL  (bright green)
    """
    meter = max(0, min(100, round((score + 20) / 40 * 100)))
    if   meter <= 20: return meter, "STRONG BEAR", "#d50000"
    elif meter <= 35: return meter, "BEAR",        "#ff5252"
    elif meter <= 45: return meter, "MILD BEAR",   "#ff9100"
    elif meter <= 54: return meter, "NEUTRAL",     "#78909C"
    elif meter <= 64: return meter, "MILD BULL",   "#b9f6ca"
    elif meter <= 80: return meter, "BULL",        "#00c853"
    else:             return meter, "STRONG BULL", "#69f0ae"


def _meter_html(score: float, show_raw: bool = True) -> str:
    """
    Render a compact 0–100 Bull/Bear Meter as an HTML snippet.
    Includes: label, number/100, gradient bar, raw score (optional).
    """
    meter, label, color = _bull_meter(score)
    fill = meter                      # 0–100 fill%
    raw  = f" <span style='color:#555;font-size:0.8em'>({score:+.1f})</span>" if show_raw else ""
    bar  = (
        f"<div style='background:#2a2a2a;border-radius:4px;height:5px;margin-top:4px'>"
        f"<div style='width:{fill}%;height:5px;border-radius:4px;"
        f"background:linear-gradient(90deg,#d50000,#ff9100,#78909C,#00c853);'></div>"
        f"</div>"
    )
    return (
        f"<span style='color:{color};font-weight:700;font-size:0.9em'>{label}</span>"
        f"<span style='color:#888;font-size:0.8em;margin-left:5px'>{meter}/100</span>"
        f"{raw}{bar}"
    )


def _card_move_line(pred: "IndexPrediction") -> str:
    """Compact one-line expected-move summary for the prediction card."""
    if pred.expected_move_pts is None or pred.spot_close is None:
        return ""
    em  = pred.expected_move_pts
    lo  = pred.range_low
    hi  = pred.range_high
    tmv = pred.target_move_pts or 0
    tgt_color = _POS if tmv > 0 else (_NEG if tmv < 0 else _NEU)
    arrow     = "▲" if tmv > 0 else ("▼" if tmv < 0 else "↔")
    tgt_html  = (f"<b style='color:{_NEU}'>↔ range</b>" if abs(tmv) < 1
                 else f"<b style='color:{tgt_color}'>{arrow}{abs(tmv):,.0f}</b>")
    return (
        f"<div style='margin-top:5px;font-size:0.74em;color:#888'>"
        f"Range <b style='color:#fff'>{lo:,.0f}–{hi:,.0f}</b> "
        f"<span style='color:#666'>(±{em:,.0f})</span> "
        f"&nbsp;Tgt {tgt_html}"
        f"</div>"
    )


def _expected_move_box(pred: "IndexPrediction", no_edge: bool = False) -> str:
    """
    Prominent expected-move forecast: range, directional target, sideways band.
    Returns an HTML snippet (empty string if no data).

    The three magnitudes share one conviction axis (see _compute_expected_move), so they
    cannot contradict:  band ≤ |target| ≤ 1σ range. A sub-band conviction yields NO point
    target (target_move == 0) — we render "none → range-bound" rather than an arrow inside
    the band. When `no_edge` (measured hit-rate ≤ coin-flip) the target is greyed to a
    "lean", subordinate to the range, so it can't masquerade as a tradeable level.
    """
    if pred.expected_move_pts is None or pred.spot_close is None:
        return ""

    em   = pred.expected_move_pts
    pct  = pred.expected_move_pct or 0
    lo   = pred.range_low
    hi   = pred.range_high
    tgt  = pred.target_close
    tmv  = pred.target_move_pts or 0
    band = pred.sideways_band_pts or 0
    spot = pred.spot_close

    has_target = abs(tmv) >= 1                       # sub-band → target_move is 0 → no arrow
    arrow      = "▲" if tmv > 0 else ("▼" if tmv < 0 else "↔")
    # Under no measured edge, demote the target to a grey "lean" so it sits below the range.
    tgt_color  = "#9e9e9e" if no_edge else (_POS if tmv > 0 else (_NEG if tmv < 0 else _NEU))

    # Sub-band → no honest point target (validated: next-day move not predictable vs spot).
    # Show the structural MAGNET instead — where OI gravity pulls IF price drifts. Labeled
    # weak, because max-pain doesn't actually pin (49% vs 50% null) except very near expiry.
    mp  = getattr(pred.levels, "max_pain", None) if pred.levels else None
    dte = pred.days_to_expiry
    if not has_target:
        if mp:
            mp_pts   = mp - spot
            mp_arrow = "↓" if mp_pts < 0 else ("↑" if mp_pts > 0 else "↔")
            pull     = ("pull strengthens into expiry" if (dte is not None and dte <= 2)
                        else "weak pull — gravity, not a pin")
            tgt_label = "Magnet (OI gravity)"
            tgt_value = f"<span style='color:{_NEU}'>{mp_arrow} {mp:,.0f}</span>"
            tgt_sub   = f"{mp_pts:+,.0f} pts · max pain · {pull}"
        else:
            tgt_label = "Directional Target"
            tgt_value = f"<span style='color:{_NEU}'>— none</span>"
            tgt_sub   = f"move within ±{band:,.0f} pt band → range-bound"
        sub_color = "#888"
    else:
        tgt_label = "Directional Lean" if no_edge else "Directional Target"
        tgt_value = f"<span style='color:{tgt_color}'>{arrow} {tgt:,.0f}</span>"
        tgt_sub   = (f"{tmv:+,.0f} pts — no measured edge, trade the range" if no_edge
                     else f"{tmv:+,.0f} pts from {spot:,.0f}")
        sub_color = tgt_color

    # Magnitude verdict line — consistent with the verdict chip (band is the same boundary).
    if not has_target:
        # Sub-band = a SWING regime. Spell out that the lean (composite tilt), the magnet
        # (OI gravity) and the zero target are DIFFERENT axes — a mild-bull score sitting
        # beside a downward magnet is not a contradiction, it's an oscillation signature.
        _cs   = pred.composite_score or 0
        lean  = "bullish" if _cs > 1 else ("bearish" if _cs < -1 else "flat")
        mp_dir = ""
        if mp:
            mp_dir = " (pulling down)" if (mp - spot) < 0 else (" (pulling up)" if (mp - spot) > 0 else "")
        magnitude_verdict = (
            f"<span style='color:{_NEU}'>Range-bound → expect a <b>SWING inside the ±{band:,.0f} pt band</b> "
            f"(up-then-down or down-then-up), not a trend. The mild <b>{lean} lean</b> is only a slight tilt "
            f"and the <b>magnet</b>{mp_dir} is OI gravity — <b>different axes, not a contradiction</b>; the "
            f"directional target is <b>0</b>. Trade the band edges; a close beyond ±{band:,.0f} pts flips to "
            f"the breakout scenarios below.</span>"
        )
    else:
        dir_word = "upside" if tmv > 0 else "downside"
        magnitude_verdict = (
            f"<span style='color:{tgt_color}'>Clears the ±{band:,.0f} pt band → "
            f"{dir_word} {'lean' if no_edge else 'target'} of ~{abs(tmv):,.0f} pts "
            f"(between band edge and the 1σ range)</span>"
        )

    # EOD close anchor — the price every range/target/band is measured from, shown here
    # so it's readable without scrolling up to the price header.
    chg     = pred.day_change_pct
    chg_col = _POS if (chg or 0) > 0 else (_NEG if (chg or 0) < 0 else _NEU)
    chg_txt = f" <span style='color:{chg_col};font-size:0.82em'>({chg:+.2f}%)</span>" if chg is not None else ""
    asof    = pred.as_of_date.strftime("%d %b") if pred.as_of_date else ""
    close_anchor = (
        f"<div style='margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid #2a2a2a'>"
        f"<span style='color:#888;font-size:0.72em;text-transform:uppercase;letter-spacing:0.5px'>"
        f"{pred.display_name} EOD Close </span>"
        f"<b style='color:#fff;font-size:1.15em'>{spot:,.0f}</b>{chg_txt}"
        f"<span style='color:#666;font-size:0.72em'> &nbsp;· as of {asof} · ranges below are from here</span>"
        f"</div>"
    )

    return (
        f"<div style='background:#161616;border:1px solid #333;border-radius:8px;"
        f"padding:10px 14px;margin-bottom:10px'>"
        f"{close_anchor}"
        f"<div style='display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px'>"
        # Expected range
        f"<div style='min-width:160px'>"
        f"<div style='color:#888;font-size:0.72em;text-transform:uppercase;letter-spacing:0.5px'>"
        f"Expected Range (~68%)</div>"
        f"<div style='color:#fff;font-weight:700;font-size:1.0em'>"
        f"{lo:,.0f} – {hi:,.0f}</div>"
        f"<div style='color:#888;font-size:0.75em'>±{em:,.0f} pts (±{pct:.2f}%)</div>"
        f"</div>"
        # Directional target / lean
        f"<div style='min-width:150px'>"
        f"<div style='color:#888;font-size:0.72em;text-transform:uppercase;letter-spacing:0.5px'>"
        f"{tgt_label}</div>"
        f"<div style='font-weight:700;font-size:1.0em'>{tgt_value}</div>"
        f"<div style='color:{sub_color};font-size:0.75em'>{tgt_sub}</div>"
        f"</div>"
        # Sideways band
        f"<div style='min-width:150px'>"
        f"<div style='color:#888;font-size:0.72em;text-transform:uppercase;letter-spacing:0.5px'>"
        f"Sideways Band</div>"
        f"<div style='color:#fff;font-weight:700;font-size:1.0em'>±{band:,.0f} pts</div>"
        f"<div style='color:#888;font-size:0.75em'>{spot-band:,.0f} – {spot+band:,.0f}</div>"
        f"</div>"
        f"</div>"
        f"<div style='margin-top:6px;font-size:0.78em'>{magnitude_verdict}</div>"
        # Tail odds — symmetric 1σ split (NOT skewed by composite: it has ~0 next-day IC,
        # so tilting these probabilities would just relabel noise). Honest reach map.
        f"<div style='margin-top:3px;font-size:0.74em;color:#999'>"
        f"Reach odds: <b style='color:{_POS}'>~16%</b> &gt; {hi:,.0f} &nbsp;·&nbsp; "
        f"<b style='color:#bbb'>~68%</b> inside {lo:,.0f}–{hi:,.0f} &nbsp;·&nbsp; "
        f"<b style='color:{_NEG}'>~16%</b> &lt; {lo:,.0f}</div>"
        f"<div style='margin-top:3px;color:#666;font-size:0.72em'>{pred.move_basis}</div>"
        f"</div>"
    )


def _breakout_extension_box(pred: "IndexPrediction") -> str:
    """
    Second-range breakout extension box.

    Shows WHERE the index is likely headed if the first range is broken by more
    than the threshold (default ~50 pts on NIFTY, proportionally scaled for others).

    Layout:
      ↑ UPSIDE BREAK  >X  →  second range: X – Y   [source]
      ↓ DOWNSIDE BREAK <X →  second range: A – B   [source]
    """
    if pred.breakout_threshold_pts is None or pred.breakout_up_start is None:
        return ""

    thr  = pred.breakout_threshold_pts
    us   = pred.breakout_up_start
    ue   = pred.breakout_up_end
    ds   = pred.breakout_dn_start
    de   = pred.breakout_dn_end
    usrc = pred.breakout_up_src or "2σ statistical"
    dsrc = pred.breakout_dn_src or "2σ statistical"

    if ue is None or ds is None or de is None:
        return ""

    up_pts   = round(ue - us, 0)
    dn_pts   = round(abs(de - ds), 0)

    # Split source into the primary label (first part) and context notes (rest)
    def _split_src(src: str) -> tuple[str, str]:
        parts = src.split(" · ", 1)
        return parts[0], (" · " + parts[1]) if len(parts) > 1 else ""

    up_primary, up_context = _split_src(usrc)
    dn_primary, dn_context = _split_src(dsrc)

    return (
        f"<div style='background:#0d1117;border:1px solid #2a3a2a;border-radius:8px;"
        f"padding:8px 14px;margin-bottom:10px'>"
        f"<div style='color:#555;font-size:0.68em;text-transform:uppercase;letter-spacing:0.5px;"
        f"margin-bottom:6px'>Breakout Scenarios (next-day) — if the 1σ range breaks by &gt;{thr:.0f} pts · "
        f"horizon is NEXT-DAY, not the weekly/monthly bands above</div>"
        # Two columns: upside | downside
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>"
        # Upside breakout
        f"<div style='border-left:2px solid #69f0ae;padding-left:8px'>"
        f"<div style='color:#888;font-size:0.68em;text-transform:uppercase;letter-spacing:0.4px'>"
        f"If Upside Break (&gt;{us:,.0f})</div>"
        f"<div style='color:#69f0ae;font-weight:700;font-size:0.95em'>"
        f"{us:,.0f} — {ue:,.0f}</div>"
        f"<div style='color:#69f0ae;font-size:0.72em'>+{up_pts:,.0f} pts extension zone</div>"
        f"<div style='color:#888;font-size:0.70em;margin-top:2px'>{up_primary}</div>"
        f"<div style='color:#444;font-size:0.66em'>{up_context}</div>"
        f"</div>"
        # Downside breakout
        f"<div style='border-left:2px solid #ff5252;padding-left:8px'>"
        f"<div style='color:#888;font-size:0.68em;text-transform:uppercase;letter-spacing:0.4px'>"
        f"If Downside Break (&lt;{de:,.0f})</div>"
        f"<div style='color:#ff5252;font-weight:700;font-size:0.95em'>"
        f"{ds:,.0f} — {de:,.0f}</div>"
        f"<div style='color:#ff5252;font-size:0.72em'>{dn_pts:,.0f} pts extension zone</div>"
        f"<div style='color:#888;font-size:0.70em;margin-top:2px'>{dn_primary}</div>"
        f"<div style='color:#444;font-size:0.66em'>{dn_context}</div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )


_TOOLTIP_CSS = """<style>
.nse-tip{position:relative;display:inline-flex;align-items:center;gap:5px}
.nse-tip .ti{display:inline-flex;align-items:center;justify-content:center;
  width:15px;height:15px;border-radius:50%;background:#2a2a2a;border:1px solid #555;
  color:#777;font-size:9px;cursor:help;flex-shrink:0;font-weight:700}
.nse-tip .tb{visibility:hidden;opacity:0;position:absolute;z-index:9999;
  left:0;top:115%;background:#1a1a2e;color:#ccc;border:1px solid #444;
  border-radius:6px;padding:10px 14px;font-size:12px;width:320px;
  line-height:1.6;transition:opacity 0.15s;pointer-events:none;font-weight:400}
.nse-tip:hover .tb{visibility:visible;opacity:1}
</style>"""


def _h(text: str, tip: str, level: int = 4) -> None:
    """Render a heading with a hover ? tooltip explaining the section."""
    safe = tip.replace('"', "&quot;").replace("'", "&#39;")
    tag  = f"h{level}"
    st.markdown(
        f'{_TOOLTIP_CSS}<{tag} style="margin-bottom:2px" class="nse-tip">'
        f'{text}&nbsp;<span class="ti">?</span>'
        f'<span class="tb">{safe}</span>'
        f'</{tag}>',
        unsafe_allow_html=True,
    )


def _pct_color(v: Optional[float]) -> str:
    return _POS if (v or 0) >= 0 else _NEG


def _fmt(v: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    return "—" if v is None else f"{v:,.{decimals}f}{suffix}"


def _delta_html(today: Optional[float], yesterday: Optional[float],
                suffix: str = "", fmt: str = ".2f") -> str:
    if today is None:
        return "<span style='color:#888'>—</span>"
    val = f"{today:{fmt}}{suffix}"
    if yesterday is None:
        return f"<span style='color:#ccc'>{val}</span>"
    delta = today - yesterday
    color = _POS if delta > 0 else (_NEG if delta < 0 else _NEU)
    sign  = "+" if delta >= 0 else ""
    return (
        f"<span style='color:#ccc'>{val}</span> "
        f"<span style='color:{color};font-size:0.75em'>({sign}{delta:{fmt}}{suffix})</span>"
    )


# ── Prediction card ───────────────────────────────────────────────────────────

_MEMORY_COLOR = {"Trending": "#4CAF50", "Random Walk": "#9E9E9E", "Mean-Reverting": "#FF9800"}
_HMM_COLOR    = {"Bull": "#4CAF50", "Sideways": "#FFD600", "Bear": "#EF5350", "Unknown": "#9E9E9E"}
_ENT_COLOR    = {"Ordered": "#4CAF50", "Moderate": "#9E9E9E", "Chaotic": "#EF5350", "Unknown": "#9E9E9E"}


def _render_card(pred: IndexPrediction) -> None:
    dir_emoji  = _DIR_EMOJI.get(pred.direction, "↔")
    conf_color = _CONF_COLOR.get(pred.confidence, "#78909C")
    dir_color  = pred.direction_color
    ctx        = pred.market_context

    # FII net display — the fao_participant net is an AGGREGATE across all index
    # futures (~76% NIFTY). Only attribute it to indices with a material share of FII
    # index-futures OI; for thin ones (FinNifty/Midcap) FIIs barely trade futures, so
    # showing the pooled short there is misleading. Mirrors the engine's signal gate.
    fii_net   = ctx.fii_fut_idx_net if ctx else 0
    _oi_by    = (ctx.fii_oi_cr_latest or {}) if ctx else {}
    _oi_tot   = sum(float(v) for v in _oi_by.values() if v)
    _oi_this  = float(_oi_by.get(pred.fno_symbol, 0.0) or 0.0)
    _fii_material = (_oi_tot <= 0) or (_oi_this / _oi_tot >= 0.10)
    if ctx and ctx.fao_date and _fii_material:
        fii_str   = f"{fii_net:+,}"
        fii_emoji = "🐂" if fii_net > 80_000 else ("🐻" if fii_net < -80_000 else "⚪")
    else:
        fii_str   = "n/a · negligible FII fut OI" if (ctx and ctx.fao_date) else "—"
        fii_emoji = "⚪"

    # VIX display
    vix_str   = f"{ctx.vix_close:.1f}" if ctx and ctx.vix_close else "—"
    vix_trend = ""
    if ctx and ctx.vix_pct_chg is not None:
        vix_trend = f" ({'↓' if ctx.vix_pct_chg < 0 else '↑'}{abs(ctx.vix_pct_chg):.1f}%)"
    vix_color = (_POS if (ctx.vix_pct_chg or 0) < 0 else _NEG) if ctx and ctx.vix_pct_chg else _NEU

    # FII position change display (covering vs adding)
    fii_chg     = ctx.fii_net_change_1d if ctx else 0
    fii_chg_str = ""
    if ctx and ctx.fii_prev_fao_date and fii_chg != 0 and _fii_material:
        fii_chg_color = _POS if fii_chg > 0 else _NEG
        fii_chg_label = "COV" if fii_chg > 0 else "ADD"
        fii_chg_str   = (
            f"<span style='color:{fii_chg_color};font-size:0.70em'>"
            f" ({fii_chg_label} {abs(fii_chg):,})</span>"
        )

    # ── Statistical Regime badges ─────────────────────────────────────────────
    regime_html = ""
    r = pred.regime
    if r and not r.error:
        h_col  = _MEMORY_COLOR.get(r.memory_label, "#9E9E9E")
        h_abbr = {"Trending": "TREND", "Random Walk": "RND", "Mean-Reverting": "M-REV"}.get(r.memory_label, "?")

        hmm_col   = _HMM_COLOR.get(r.hmm_state, "#9E9E9E")
        hmm_label = r.hmm_state if r.hmm_state != "Unknown" else "—"

        ent_col   = _ENT_COLOR.get(r.entropy_label, "#9E9E9E")
        ent_abbr  = {"Ordered": "ORD", "Moderate": "MOD", "Chaotic": "CHAOS"}.get(r.entropy_label, "?")

        def _badge(label: str, value: str, color: str) -> str:
            return (
                f"<span style='display:inline-flex;flex-direction:column;align-items:center;"
                f"background:rgba(255,255,255,0.04);border:1px solid {color}44;"
                f"border-radius:5px;padding:2px 7px;margin-right:4px'>"
                f"<span style='font-size:0.60em;color:#888;letter-spacing:0.5px'>{label}</span>"
                f"<span style='font-size:0.78em;font-weight:700;color:{color}'>{value}</span>"
                f"</span>"
            )

        regime_html = (
            f"<div style='margin-top:7px;border-top:1px solid #2a2a2a;padding-top:6px'>"
            f"<div style='font-size:0.65em;color:#555;margin-bottom:3px;letter-spacing:0.5px'>STATISTICAL REGIME</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:2px'>"
            f"{_badge('HURST', f'{r.memory_avg:.3f} {h_abbr}', h_col)}"
            f"{_badge('HMM', hmm_label, hmm_col)}"
            f"{_badge('ENTROPY', ent_abbr, ent_col)}"
            f"</div>"
            f"</div>"
        )

    st.markdown(
        f"""
        <div style="
            background:#1e1e1e; border:1px solid {dir_color};
            border-radius:10px; padding:16px 18px; height:100%;
        ">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-size:1.05em;font-weight:700;color:#fff">{pred.display_name}</span>
                <span style="
                    background:{dir_color}22;color:{dir_color};
                    border:1px solid {dir_color};border-radius:6px;
                    padding:2px 10px;font-size:0.9em;font-weight:700;
                ">{dir_emoji} {pred.direction}</span>
            </div>
            <div style="font-size:1.4em;font-weight:700;color:#fff;margin-bottom:2px">
                {_fmt(pred.spot_close, 0) if pred.spot_close else '—'}
            </div>
            <div style="font-size:0.9em;color:{'#69f0ae' if (pred.day_change_pct or 0)>=0 else '#ff5252'};margin-bottom:10px">
                {'▲' if (pred.day_change_pct or 0)>=0 else '▼'} {abs(pred.day_change_pct or 0):.2f}%
                &nbsp;|&nbsp;
                <span style="color:{conf_color};font-size:0.85em">{pred.confidence} CONF</span>
            </div>
            <div style="margin-bottom:6px;font-size:0.8em">
                {_meter_html(pred.composite_score, show_raw=True)}
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:0.8em;color:#aaa">
                <div>PCR <b style="color:{'#fff' if getattr(pred, 'opt_chain_liquid', True) else '#997a3d'}">{_fmt(pred.pcr) if pred.pcr else '—'}</b>{'' if getattr(pred, 'opt_chain_liquid', True) else ' <span style="color:#ffae00;font-size:0.85em" title="Near-expiry chain below liquidity floor — PCR is noise, excluded from signals">⚠ thin</span>'}</div>
                <div>DTE <b style="color:#fff">{pred.days_to_expiry}d</b></div>
                <div>Carry <b style="color:#fff">{_fmt(pred.carry_pct_ann, 1, '% ann') if pred.carry_pct_ann is not None else '—'}</b></div>
                <div>VIX <b style="color:{vix_color}">{vix_str}{vix_trend}</b></div>
                <div style="grid-column:1/-1">FII <b style="color:#fff">{fii_emoji} {fii_str}</b>{fii_chg_str}</div>
            </div>
            <div style="margin-top:8px;font-size:0.75em;color:#888;border-top:1px solid #333;padding-top:6px">
                S: <b style="color:#69f0ae">{_fmt(pred.levels.top_put_strike, 0) if pred.levels.top_put_strike else '—'}</b>
                &nbsp;|&nbsp;
                MP: <b style="color:#FFD600">{_fmt(pred.levels.max_pain, 0) if pred.levels.max_pain else '—'}</b>
                &nbsp;|&nbsp;
                R: <b style="color:#ff5252">{_fmt(pred.levels.top_call_strike, 0) if pred.levels.top_call_strike else '—'}</b>
                {'' if getattr(pred, 'opt_chain_liquid', True) else '<span style="color:#ffae00" title="Near-expiry chain below liquidity floor — walls/max pain are noise, excluded from signals and ranges"> ⚠ thin chain</span>'}
            </div>
            {_card_move_line(pred)}
            {regime_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Today vs Yesterday ────────────────────────────────────────────────────────

def _render_comparison(pred: IndexPrediction) -> None:
    t = pred.today
    y = pred.yesterday
    _h("Today vs Yesterday", "Side-by-side comparison of today's key metrics vs yesterday. Use this to spot momentum shifts — e.g. PCR rising, FII net getting more short, or carry turning negative are early warning signs.")
    if t is None:
        st.caption("No snapshot data available.")
        return

    rows = [
        ("Spot Close",    t.spot_close,   y.spot_close if y else None,   ".0f",  ""),
        ("Day Change %",  t.pct_chg,      y.pct_chg if y else None,      ".2f",  "%"),
        ("Futures Settle",t.fut_settle,   y.fut_settle if y else None,   ".0f",  ""),
        ("Carry % ann",   t.carry_pct_ann,y.carry_pct_ann if y else None,".2f",  "%"),
        ("Carry Pts",     t.carry_pts,    y.carry_pts if y else None,    ".0f",  " pts"),
        ("Futures OI",    t.fut_oi,       y.fut_oi if y else None,       ",.0f", ""),
        ("Call OI (Near)",t.call_oi,      y.call_oi if y else None,      ",.0f", ""),
        ("Put OI (Near)", t.put_oi,       y.put_oi if y else None,       ",.0f", ""),
        ("PCR (Near)",    t.pcr,          y.pcr if y else None,          ".2f",  ""),
        ("Total Volume",  t.total_volume, y.total_volume if y else None, ",.0f", ""),
    ]

    td = t.trade_date.strftime("%d %b")
    yd = y.trade_date.strftime("%d %b") if y else "Yesterday"
    html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:0.85em">
        <thead>
            <tr style="border-bottom:1px solid #333;color:#888">
                <th style="text-align:left;padding:4px 8px">Metric</th>
                <th style="text-align:right;padding:4px 8px">{td}</th>
                <th style="text-align:right;padding:4px 8px">{yd}</th>
                <th style="text-align:right;padding:4px 8px">Change</th>
            </tr>
        </thead><tbody>
    """
    for label, tv, yv, fmt, suf in rows:
        ts = f"{tv:{fmt}}{suf}" if tv is not None else "—"
        ys = f"{yv:{fmt}}{suf}" if yv is not None else "—"
        if tv is not None and yv is not None:
            delta = tv - yv
            sign  = "+" if delta >= 0 else ""
            dc    = _POS if delta > 0 else (_NEG if delta < 0 else _NEU)
            ds    = f"<span style='color:{dc}'>{sign}{delta:{fmt}}{suf}</span>"
        else:
            ds = "—"
        html += f"""
        <tr style="border-bottom:1px solid #222">
            <td style="padding:5px 8px;color:#aaa">{label}</td>
            <td style="padding:5px 8px;text-align:right;color:#fff">{ts}</td>
            <td style="padding:5px 8px;text-align:right;color:#666">{ys}</td>
            <td style="padding:5px 8px;text-align:right">{ds}</td>
        </tr>"""
    html += "</tbody></table>"
    st.html(html.strip())


# ── OI Structure chart ────────────────────────────────────────────────────────

def _render_oi_chart(pred: IndexPrediction) -> None:
    expiry_label = "Weekly" if (pred.fno_symbol == "NIFTY" and pred.monthly_expiry) else "Near-Month"
    _h(f"OI Structure — {expiry_label} Options Chain", "Call OI (red, right) = resistance levels where sellers have written calls. Put OI (green, left) = support levels where sellers have written puts. The strike with highest combined OI is Max Pain — markets often gravitate here into expiry. Wide call wall above spot = ceiling; wide put wall below = floor.")
    if pred.near_expiry is None or pred.spot_close is None:
        st.caption("Options chain data not available.")
        return

    from src.dashboard.cache.queries import cached_options_chain, cached_index_options_chain
    chain_df = cached_options_chain(pred.as_of_date, pred.fno_symbol, pred.near_expiry, "OPTIDX")
    if chain_df is None or chain_df.empty:
        chain_df = cached_index_options_chain(pred.as_of_date, pred.fno_symbol, pred.near_expiry, n_strikes=20)
    if chain_df is None or chain_df.empty:
        st.caption("Options chain not available for this expiry.")
        return

    spot = pred.spot_close

    # Both chain functions return pivoted DataFrames: one row per strike,
    # columns ce_oi / pe_oi (not raw option_type rows).
    # Use dict lookup to avoid float-precision issues with .loc[]
    ce_oi_map = dict(zip(chain_df["strike_price"], chain_df["ce_oi"]))
    pe_oi_map = dict(zip(chain_df["strike_price"], chain_df["pe_oi"]))
    all_s     = sorted(float(s) for s in ce_oi_map)
    band      = spot * 0.08
    strikes   = [s for s in all_s if abs(s - spot) <= band] or all_s

    call_oi = [int(ce_oi_map.get(s, 0)) for s in strikes]
    put_oi  = [int(pe_oi_map.get(s, 0)) for s in strikes]

    # Use NUMERIC y-axis (not categorical strings) so add_hline works in Plotly 6.x.
    # Tick labels are set explicitly via tickvals/ticktext.
    fig = go.Figure()
    fig.add_trace(go.Bar(y=strikes, x=call_oi,
                         name="Call OI", orientation="h", marker_color="#ef5350", opacity=0.85))
    fig.add_trace(go.Bar(y=strikes, x=[-v for v in put_oi],
                         name="Put OI",  orientation="h", marker_color="#26a69a", opacity=0.85))

    if strikes:
        cls = min(strikes, key=lambda s: abs(s - spot))
        fig.add_hline(y=float(cls), line_color="#FFD600", line_width=2, line_dash="dash",
                      annotation=dict(text=f"Spot {spot:.0f}", font_color="#FFD600"))
    if pred.levels.max_pain and strikes:
        mp = min(strikes, key=lambda s: abs(s - pred.levels.max_pain))
        fig.add_hline(y=float(mp), line_color="#FF9800", line_width=2, line_dash="dot",
                      annotation=dict(text=f"MaxPain {pred.levels.max_pain:.0f}", font_color="#FF9800"))

    fig.update_layout(
        barmode="overlay", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        font=dict(color="#ccc", size=11),
        height=max(300, len(strikes) * 22),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(title="Put OI  <--  |  -->  Call OI", gridcolor=GRID_COLOR,
                   zeroline=True, zerolinecolor="#555", zerolinewidth=2, tickformat=","),
        yaxis=dict(title="Strike", gridcolor=GRID_COLOR,
                   tickvals=strikes, ticktext=[str(int(s)) for s in strikes]),
        legend=dict(orientation="h", x=0, y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Nifty: monthly expiry OI chain ────────────────────────────────────────
    if pred.fno_symbol == "NIFTY" and pred.monthly_expiry is not None and pred.spot_close is not None:
        spot_close = pred.spot_close
        m_dte = (pred.monthly_expiry - pred.as_of_date).days
        st.markdown(
            f"<div style='margin-top:16px;font-size:0.75em;color:#888;font-weight:600;"
            f"letter-spacing:0.05em'>── MONTHLY EXPIRY OI  "
            f"({pred.monthly_expiry.strftime('%d %b')} · {m_dte}d) ──</div>",
            unsafe_allow_html=True,
        )
        from src.dashboard.cache.queries import cached_options_chain, cached_index_options_chain
        m_chain = cached_options_chain(pred.as_of_date, pred.fno_symbol, pred.monthly_expiry, "OPTIDX")
        if m_chain is None or m_chain.empty:
            m_chain = cached_index_options_chain(pred.as_of_date, pred.fno_symbol, pred.monthly_expiry, n_strikes=20)
        if m_chain is not None and not m_chain.empty:
            ce_m = dict(zip(m_chain["strike_price"], m_chain["ce_oi"]))
            pe_m = dict(zip(m_chain["strike_price"], m_chain["pe_oi"]))
            all_ms   = sorted(float(s) for s in ce_m)
            band_m   = spot_close * 0.10
            strikes_m = [s for s in all_ms if abs(s - spot_close) <= band_m] or all_ms
            call_oi_m = [int(ce_m.get(s, 0)) for s in strikes_m]
            put_oi_m  = [int(pe_m.get(s, 0)) for s in strikes_m]

            fig_m = go.Figure()
            fig_m.add_trace(go.Bar(y=strikes_m, x=call_oi_m,
                                   name="Call OI (Monthly)", orientation="h",
                                   marker_color="#ef9a9a", opacity=0.80))
            fig_m.add_trace(go.Bar(y=strikes_m, x=[-v for v in put_oi_m],
                                   name="Put OI (Monthly)",  orientation="h",
                                   marker_color="#80cbc4", opacity=0.80))
            if strikes_m:
                cls_m = min(strikes_m, key=lambda s: abs(s - spot_close))
                fig_m.add_hline(y=float(cls_m), line_color="#FFD600", line_width=2,
                                line_dash="dash",
                                annotation=dict(text=f"Spot {spot_close:.0f}", font_color="#FFD600"))
            if pred.monthly_max_pain and strikes_m:
                mmp_s = min(strikes_m, key=lambda s: abs(s - pred.monthly_max_pain))
                fig_m.add_hline(y=float(mmp_s), line_color="#FF9800", line_width=2,
                                line_dash="dot",
                                annotation=dict(text=f"M-MaxPain {pred.monthly_max_pain:.0f}",
                                                font_color="#FF9800"))
            fig_m.update_layout(
                barmode="overlay", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
                font=dict(color="#ccc", size=11),
                height=max(280, len(strikes_m) * 22),
                margin=dict(l=10, r=30, t=10, b=10),
                xaxis=dict(title="Put OI  <--  |  -->  Call OI", gridcolor=GRID_COLOR,
                           zeroline=True, zerolinecolor="#555", zerolinewidth=2, tickformat=","),
                yaxis=dict(title="Strike", gridcolor=GRID_COLOR,
                           tickvals=strikes_m, ticktext=[str(int(s)) for s in strikes_m]),
                legend=dict(orientation="h", x=0, y=1.02),
            )
            st.plotly_chart(fig_m, use_container_width=True)
            if pred.monthly_pcr is not None:
                mpcr_clr = "#69f0ae" if pred.monthly_pcr > 1.0 else ("#ff5252" if pred.monthly_pcr < 0.75 else "#FFD600")
                st.caption(
                    f"Monthly PCR: **{pred.monthly_pcr:.2f}**  "
                    f"· Monthly Max Pain: **{pred.monthly_max_pain:.0f}**  "
                    f"· Call OI: {pred.monthly_call_oi:,}  · Put OI: {pred.monthly_put_oi:,}"
                )
        else:
            st.caption("Monthly options chain not available for this date.")


# ── Signal breakdown ──────────────────────────────────────────────────────────

def _render_signals(pred: IndexPrediction) -> None:
    n_total = len(pred.signals)
    _h(
        "Signal Breakdown",
        f"All {n_total} fired signals grouped by tier. Green = bullish, Red = bearish. "
        "Weighted composite score drives the final UP/DOWN/SIDEWAYS verdict. "
        "Institutional signals are capped at ±3.0 to prevent collinear FII data from "
        "dominating the score. Statistical Regime signals (Hurst, HMM, Entropy) also have "
        "full visualisations in the Statistical Regime tab.",
    )
    if not pred.signals:
        st.caption("No signals computed.")
        return

    # Group by tier for readability — includes Statistical Regime + Memory
    tier_order = ["Institutional", "Options OI", "Futures OI", "Carry",
                  "Price Action", "Market Context", "Sector",
                  "Statistical Regime", "Memory"]
    grouped: dict[str, list[IndexSignal]] = {}
    for s in pred.signals:
        grouped.setdefault(s.category, []).append(s)

    for tier in tier_order:
        if tier not in grouped:
            continue
        cat_color = _CAT_COLOR.get(tier, "#888")
        st.markdown(
            f"<div style='font-size:0.75em;color:{cat_color};font-weight:700;"
            f"letter-spacing:0.05em;margin:8px 0 4px'>── {tier.upper()} ──</div>",
            unsafe_allow_html=True,
        )
        for sig in grouped[tier]:
            arrow       = "↑" if sig.direction > 0 else ("↓" if sig.direction < 0 else "↔")
            score_color = _POS if sig.score > 0 else (_NEG if sig.score < 0 else _NEU)
            st.markdown(
                f"""
                <div style="
                    background:#1a1a1a;border-left:3px solid {cat_color};
                    border-radius:6px;padding:8px 12px;margin-bottom:6px;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="color:#fff;font-weight:600">{sig.emoji} {sig.headline}</span>
                        <span style="
                            background:{score_color}22;color:{score_color};
                            border:1px solid {score_color};border-radius:4px;
                            padding:1px 8px;font-weight:700;font-size:0.85em
                        ">{arrow} {sig.score:+.1f}</span>
                    </div>
                    <div style="color:#888;font-size:0.78em;margin-top:3px">
                        {sig.description}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Key levels ────────────────────────────────────────────────────────────────

def _render_key_levels(pred: IndexPrediction) -> None:
    lv = pred.levels
    has_oi_levels = any(v is not None for v in [lv.top_put_strike, lv.max_pain, lv.top_call_strike])
    has_price_sr  = bool(pred.sr_support or pred.sr_resistance)
    # Price-structure S/R comes from the index OHLC and must render even when the
    # options chain produced no walls — the two level sources are independent.
    if not has_oi_levels and not has_price_sr:
        return
    _h("Key Levels", "Put Wall = strong support (where put sellers will defend the market). Call Wall = strong resistance (where call sellers will suppress rallies). Max Pain = level where option writers lose the least — markets magnetically move toward this as expiry nears, especially in the last 5 trading days.")
    if not getattr(pred, "opt_chain_liquid", True):
        st.warning(
            "⚠ Thin options chain: near-expiry OI is below the engine's liquidity floor. "
            "The put/call walls and max pain below are raw-data context — the engine excludes "
            "them from signals and range structure because on this chain they are noise.",
            icon="⚠️",
        )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div style="background:#1a3d2e;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#69f0ae;font-size:0.8em;font-weight:600">SUPPORT (Put Wall)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.top_put_strike, 0) if lv.top_put_strike else '—'}</div>
                <div style="color:#888;font-size:0.75em">OI: {lv.put_oi_at_top:,}</div>
                {f'<div style="color:#555;font-size:0.7em">2nd: {lv.second_put_strike:.0f}</div>' if lv.second_put_strike else ''}
            </div>""", unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div style="background:#3d3320;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#FFD600;font-size:0.8em;font-weight:600">MAX PAIN (Expiry Pin)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.max_pain, 0) if lv.max_pain else '—'}</div>
                <div style="color:#888;font-size:0.75em">DTE: {pred.days_to_expiry}d</div>
                {f'<div style="color:#888;font-size:0.72em">Expiry: {pred.near_expiry.strftime("%d %b") if pred.near_expiry else ""}</div>' if pred.near_expiry else ''}
            </div>""", unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div style="background:#3d1a1a;border-radius:8px;padding:12px;text-align:center">
                <div style="color:#ff5252;font-size:0.8em;font-weight:600">RESISTANCE (Call Wall)</div>
                <div style="color:#fff;font-size:1.4em;font-weight:700">{_fmt(lv.top_call_strike, 0) if lv.top_call_strike else '—'}</div>
                <div style="color:#888;font-size:0.75em">OI: {lv.call_oi_at_top:,}</div>
                {f'<div style="color:#555;font-size:0.7em">2nd: {lv.second_call_strike:.0f}</div>' if lv.second_call_strike else ''}
            </div>""", unsafe_allow_html=True,
        )

    # ── Price-structure S/R (swing-pivot zones from daily OHLC — Signal 6b) ──
    if pred.sr_support or pred.sr_resistance:
        def _conf(price_lvl, wall1, wall2):
            """True when an OI wall sits within 0.6% of the price level (confluence)."""
            if not price_lvl:
                return False
            return any(w and abs(w - price_lvl) / price_lvl * 100 <= 0.60 for w in (wall1, wall2))
        sup_conf = _conf(pred.sr_support, lv.top_put_strike, lv.second_put_strike)
        res_conf = _conf(pred.sr_resistance, lv.top_call_strike, lv.second_call_strike)
        sup_txt = (f"<b style='color:#69f0ae'>{pred.sr_support:,.0f}</b> "
                   f"({pred.sr_support_touches} touches)"
                   + (" · 🤝 on the put wall" if sup_conf else "")
                   if pred.sr_support else "—")
        res_txt = (f"<b style='color:#ff5252'>{pred.sr_resistance:,.0f}</b> "
                   f"({pred.sr_resistance_touches} touches)"
                   + (" · 🤝 on the call wall" if res_conf else "")
                   if pred.sr_resistance else "—")
        st.markdown(
            f"""<div style="background:#16202b;border-radius:8px;padding:10px 14px;margin-top:8px;font-size:0.85em">
                <span style="color:#aaa;font-weight:600">PRICE-STRUCTURE S/R</span>
                <span style="color:#555">(swing-pivot zones, ~60D daily OHLC)</span> ·
                Support {sup_txt} &nbsp;|&nbsp; Resistance {res_txt}
                <div style="color:#555;font-size:0.85em;margin-top:2px">
                    Chart-derived levels the market actually traded at — independent of the
                    options chain. 🤝 = an OI wall defends the same line (strongest levels).
                    Range map, not a directional vote.</div>
            </div>""", unsafe_allow_html=True,
        )

    # ── Nifty 50 multi-expiry panel ───────────────────────────────────────────
    if pred.fno_symbol == "NIFTY" and pred.monthly_expiry is not None:
        st.markdown("---")
        _h(
            "Multi-Expiry Structure (Nifty 50 — Weekly + Monthly)",
            "Nifty 50 is the only index with both weekly and monthly expiry. "
            "This creates a two-layer OI structure: weekly gamma pins the market this week; "
            "monthly max pain dictates where spot drifts after weekly expiry resolves.",
        )
        me1, me2, me3, me4 = st.columns(4)

        w_dte = (pred.weekly_expiry  - pred.as_of_date).days if pred.weekly_expiry else "—"
        m_dte = (pred.monthly_expiry - pred.as_of_date).days

        with me1:
            pcr_color   = "#69f0ae" if (pred.weekly_pcr or 0) > 1.0 else ("#ff5252" if (pred.weekly_pcr or 0) < 0.75 else "#FFD600")
            wpcr_str    = f"{pred.weekly_pcr:.2f}" if pred.weekly_pcr is not None else "—"
            st.markdown(
                f"""<div style="background:#1e2a1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">WEEKLY PCR</div>
                    <div style="color:{pcr_color};font-size:1.5em;font-weight:700">{wpcr_str}</div>
                    <div style="color:#555;font-size:0.7em">{pred.weekly_expiry.strftime('%d %b') if pred.weekly_expiry else '—'}  ({w_dte}d)</div>
                </div>""", unsafe_allow_html=True,
            )
        with me2:
            mpcr_color  = "#69f0ae" if (pred.monthly_pcr or 0) > 1.0 else ("#ff5252" if (pred.monthly_pcr or 0) < 0.75 else "#FFD600")
            mpcr_str    = f"{pred.monthly_pcr:.2f}" if pred.monthly_pcr is not None else "—"
            st.markdown(
                f"""<div style="background:#1e1e2a;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">MONTHLY PCR</div>
                    <div style="color:{mpcr_color};font-size:1.5em;font-weight:700">{mpcr_str}</div>
                    <div style="color:#555;font-size:0.7em">{pred.monthly_expiry.strftime('%d %b')}  ({m_dte}d)</div>
                </div>""", unsafe_allow_html=True,
            )
        with me3:
            mmp = pred.monthly_max_pain
            if mmp and pred.spot_close:
                mmp_gap = (mmp - pred.spot_close) / pred.spot_close * 100
                mmp_color = "#69f0ae" if mmp_gap > 0 else "#ff5252"
                mmp_str = f"{mmp:.0f}  ({mmp_gap:+.1f}%)"
            else:
                mmp_color = "#888"; mmp_str = "—"
            st.markdown(
                f"""<div style="background:#2a2a1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">MONTHLY MAX PAIN</div>
                    <div style="color:{mmp_color};font-size:1.2em;font-weight:700">{mmp_str}</div>
                    <div style="color:#555;font-size:0.7em">Post-weekly gravitational pull</div>
                </div>""", unsafe_allow_html=True,
            )
        with me4:
            gr = pred.gamma_ratio
            if gr is not None:
                gr_pct = gr * 100
                gr_color = "#ff9800" if gr_pct > 58 else ("#40c4ff" if gr_pct < 28 else "#888")
                gr_label = "⚠ Weekly Pinning" if gr_pct > 58 else ("↔ Balanced" if gr_pct > 28 else "↗ Monthly Dominates")
            else:
                gr_pct = 0; gr_color = "#888"; gr_label = "—"
            st.markdown(
                f"""<div style="background:#2a1e1e;border-radius:8px;padding:12px;text-align:center">
                    <div style="color:#aaa;font-size:0.75em;font-weight:600">GAMMA RATIO</div>
                    <div style="color:{gr_color};font-size:1.5em;font-weight:700">{gr_pct:.0f}%</div>
                    <div style="color:{gr_color};font-size:0.68em">{gr_label}</div>
                    <div style="color:#555;font-size:0.65em">Weekly OI / Total OI</div>
                </div>""", unsafe_allow_html=True,
            )


# ── Verdict panel ─────────────────────────────────────────────────────────────

def _weekly_range_box(pred: "IndexPrediction") -> str:
    """
    Forward WEEKLY range to the next expiry — straddle-implied + σ√T, bounded by OI
    walls / max pain. Rolls to the next cycle on expiry day; recomputed each EOD.
    """
    if pred.wk_range_low is None or pred.wk_range_high is None or pred.spot_close is None:
        return ""
    lo, hi, mv = pred.wk_range_low, pred.wk_range_high, pred.wk_move_pts or 0
    exp = pred.wk_expiry.strftime("%d %b") if pred.wk_expiry else "—"
    kind = (pred.wk_kind or "").upper()
    dte = pred.wk_dte_cal
    pct = (mv / pred.spot_close * 100) if pred.spot_close else 0
    impl = (f"ATM straddle {pred.wk_straddle_pts:,.0f} (market-implied) + σ√T"
            if pred.wk_straddle_pts else "σ√T statistical")
    oi_bits = []
    if pred.wk_put_floor:  oi_bits.append(f"🛡️ put floor <b>{pred.wk_put_floor:,.0f}</b>")
    if pred.wk_max_pain:   oi_bits.append(f"🎯 max pain <b>{pred.wk_max_pain:,.0f}</b>")
    if pred.wk_call_cap:   oi_bits.append(f"🧱 call cap <b>{pred.wk_call_cap:,.0f}</b>")
    oi_line = " &nbsp;·&nbsp; ".join(oi_bits)
    return (
        f"<div style='background:#0d1117;border:1px solid #1e3a4a;border-radius:8px;"
        f"padding:8px 14px;margin-bottom:10px'>"
        f"<div style='color:#40c4ff;font-size:0.68em;text-transform:uppercase;letter-spacing:0.5px;"
        f"margin-bottom:4px'>📆 Weekly Range → expiry {exp} ({kind} · {dte}d) · updates daily</div>"
        f"<div style='color:#40c4ff;font-weight:700;font-size:1.05em'>{lo:,.0f} &nbsp;—&nbsp; {hi:,.0f}</div>"
        f"<div style='color:#7fdbff;font-size:0.74em'>±{mv:,.0f} pts ({pct:.1f}%) — where price can travel until expiry</div>"
        f"<div style='color:#888;font-size:0.70em;margin-top:3px'>implied by {impl}</div>"
        + (f"<div style='color:#aaa;font-size:0.72em;margin-top:2px'>{oi_line}</div>" if oi_line else "")
        + "<div style='color:#777;font-size:0.68em;margin-top:3px'>📊 validated: the CLOSE lands "
          "inside by expiry ~68% (NIFTY/Bank) — but price tests OUTSIDE intraday in ~60% of weeks. "
          "A settle-by-expiry zone, <b>not an intraday stop level</b>.</div>"
        + "</div>"
    )


def _monthly_range_box(pred: "IndexPrediction") -> str:
    """
    Forward MONTHLY range to the monthly expiry — NIFTY only, built from the MONTHLY
    chain's own ATM straddle + σ√T, bounded by monthly OI walls / max pain. A wider,
    slower scenario band beside the weekly. Empty when there is no distinct monthly
    (Bank/Fin/Midcap are monthly-only, or NIFTY's near expiry already IS the monthly).
    """
    if pred.mn_range_low is None or pred.mn_range_high is None or pred.spot_close is None:
        return ""
    lo, hi, mv = pred.mn_range_low, pred.mn_range_high, pred.mn_move_pts or 0
    exp = pred.mn_expiry.strftime("%d %b") if pred.mn_expiry else "—"
    dte = pred.mn_dte_cal
    pct = (mv / pred.spot_close * 100) if pred.spot_close else 0
    impl = (f"ATM straddle {pred.mn_straddle_pts:,.0f} (market-implied) + σ√T"
            if pred.mn_straddle_pts else "σ√T statistical")
    oi_bits = []
    if pred.mn_put_floor:  oi_bits.append(f"🛡️ put floor <b>{pred.mn_put_floor:,.0f}</b>")
    if pred.mn_max_pain:   oi_bits.append(f"🎯 max pain <b>{pred.mn_max_pain:,.0f}</b>")
    if pred.mn_call_cap:   oi_bits.append(f"🧱 call cap <b>{pred.mn_call_cap:,.0f}</b>")
    oi_line = " &nbsp;·&nbsp; ".join(oi_bits)
    return (
        f"<div style='background:#0d1117;border:1px solid #4a3a1e;border-radius:8px;"
        f"padding:8px 14px;margin-bottom:10px'>"
        f"<div style='color:#ffb74d;font-size:0.68em;text-transform:uppercase;letter-spacing:0.5px;"
        f"margin-bottom:4px'>🗓️ Monthly Range → expiry {exp} (MONTHLY · {dte}d) · updates daily</div>"
        f"<div style='color:#ffb74d;font-weight:700;font-size:1.05em'>{lo:,.0f} &nbsp;—&nbsp; {hi:,.0f}</div>"
        f"<div style='color:#ffcc80;font-size:0.74em'>±{mv:,.0f} pts ({pct:.1f}%) — where price can travel until the monthly expiry</div>"
        f"<div style='color:#888;font-size:0.70em;margin-top:3px'>implied by {impl} (monthly chain)</div>"
        + (f"<div style='color:#aaa;font-size:0.72em;margin-top:2px'>{oi_line}</div>" if oi_line else "")
        + "<div style='color:#777;font-size:0.68em;margin-top:3px'>📊 runs TIGHT: the close lands inside "
          "by expiry only ~57–61% (small sample, ~13 expiries) — read as a <b>typical-move</b> band, "
          "not a 68% band; awaiting more expiries before recalibration.</div>"
        + "</div>"
    )


def _range_cards(pred: "IndexPrediction") -> str:
    """Weekly + monthly to-expiry bands. Side-by-side (weekly left / monthly right)
    when both exist (NIFTY); otherwise the single weekly box full-width."""
    wk = _weekly_range_box(pred)
    mn = _monthly_range_box(pred)
    if wk and mn:
        return (f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;"
                f"align-items:stretch'>{wk}{mn}</div>")
    return wk or mn


def _confidence_phrase(pred: IndexPrediction) -> str:
    """
    Qualify the confidence label so it says confidence IN WHAT.

    The word "confidence" means different things per verdict:
      - UP / DOWN  → confidence in the DIRECTION.
      - SIDEWAYS   → confidence in the market staying RANGE-BOUND (e.g. an expiry
                     gamma pin), NOT a directional call. Showing a bare
                     "MEDIUM CONFIDENCE" next to a STRONG BEAR meter reads as a
                     contradiction; this disambiguates it.
    """
    conf = pred.confidence
    if pred.direction == "SIDEWAYS":
        return f"{conf} CONFIDENCE (range-bound, not directional)"
    return f"{conf} CONFIDENCE (directional)"


# Meter zones that are decisively directional (mild zones excluded to avoid noise).
_METER_BEAR = {"STRONG BEAR", "BEAR"}
_METER_BULL = {"STRONG BULL", "BULL"}


def _verdict_meter_note(pred: IndexPrediction) -> str:
    """
    Reconcile the headline verdict with the raw 0–100 Bull/Bear meter when they
    disagree — otherwise the user sees e.g. "SIDEWAYS" beside "STRONG BEAR 0/100"
    with nothing linking them.

    Two disagreement cases:
      1. SIDEWAYS verdict + directional meter  → the score is the BREAK direction
         if the range fails (commonly an expiry-day gamma pin suppressing it).
      2. Directional verdict OPPOSITE the meter → an override (mean-reversion /
         squeeze) is countering the raw trend signals → lower conviction.
    Returns "" when verdict and meter agree (or meter is only mildly directional).
    """
    meter, label, color = _bull_meter(pred.composite_score)
    meter_dir = -1 if label in _METER_BEAR else 1 if label in _METER_BULL else 0
    if meter_dir == 0:
        return ""   # meter near neutral — nothing to reconcile

    verdict_dir = {"UP": 1, "DOWN": -1, "SIDEWAYS": 0}.get(pred.direction, 0)
    if meter_dir == verdict_dir:
        return ""   # aligned — no note needed

    lean  = "downside" if meter_dir < 0 else "upside"
    arrow = "▼" if meter_dir < 0 else "▲"

    if verdict_dir == 0:
        is_expiry = (pred.days_to_expiry is not None and pred.days_to_expiry <= 1)
        reason = ("the expiry-day gamma pin" if is_expiry
                  else "a neutral-zone / mean-reversion override")
        msg = (f"{arrow} Raw multi-signal read is <b>{label} ({meter}/100)</b>, but it's suppressed by "
               f"{reason}. Treat it as the likely <b>break direction</b> ({lean}) if the range "
               f"fails — not today's base case.")
    else:
        msg = (f"{arrow} The <b>{pred.direction}</b> verdict <b>overrides</b> the raw multi-signal "
               f"score (<b>{label} {meter}/100</b>) — a mean-reversion / squeeze setup is countering "
               f"the trend signals. Lower conviction; use tight stops.")

    return (
        f"<div style='background:rgba(255,170,0,0.08);border-left:3px solid {color};"
        f"border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:10px;"
        f"color:#cfcfcf;font-size:0.8em;line-height:1.45'>{msg}</div>"
    )


def _measured_trust_box(acc: dict, display_name: str, cov: Optional[dict] = None) -> str:
    """Per-index realized track record, rendered ON the verdict card so the
    displayed confidence can never read higher than the measured next-day
    hit-rate. Pooling hides the worst case (NIFTY ≈ 22%), so this is symbol-
    specific. Returns "" when there is too little history to judge."""
    if not acc:
        return ""
    sh, ic, n = acc["sign_hit"], acc["ic"], acc["n_dir"]
    # Measured 1σ-band coverage (self-updating): keeps the "~68%" range claim
    # honest — in a calm-vol regime the band over-covers, and saying so beats
    # letting a hardcoded number drift from reality.
    cov_line = ""
    if cov:
        c90 = f" · last 90d <b>{cov['cov_90d']:.0f}%</b> (n={cov['n_90d']})" if cov.get("cov_90d") else ""
        note = ""
        c_ref = cov.get("cov_90d", cov["cov_all"])
        if c_ref >= 76:
            note = " — band currently runs WIDE (calm-vol regime): honest but conservative."
        elif c_ref <= 60:
            note = " — band currently runs TIGHT: treat edges as soft."
        cov_line = (
            f"<div style='color:#9fb3c8;margin-top:3px'>Range band measured: next close landed "
            f"inside the ±1σ range <b>{cov['cov_all']:.0f}%</b> of {cov['n']} days (target ~68%)"
            f"{c90}{note}</div>"
        )
    if sh >= 56:
        col, tag, msg = "#00c853", "has held an edge", \
            "the directional read has worked — still size with the range."
    elif sh >= 48:
        col, tag, msg = "#ffd600", "≈ coin-flip", \
            "no measured directional edge — trade the range, not the arrow."
    else:
        col, tag, msg = "#ff5252", "below coin-flip", \
            "the arrow has been WRONG more often than right — do NOT trade the " \
            "direction; use only the range / levels / risk flags below."
    return (
        f"<div style='background:rgba(255,255,255,0.03);border-left:3px solid {col};"
        f"border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:10px;font-size:0.8em;color:#ddd'>"
        f"<b style='color:{col}'>⚖ Reality check — {display_name}:</b> measured next-day "
        f"directional hit-rate <b style='color:{col}'>{sh:.0f}%</b> (n={n}) · IC {ic:+.3f} — "
        f"<b style='color:{col}'>{tag}</b>. {msg}{cov_line}</div>"
    )


def _thin_chain_banner(pred: IndexPrediction) -> str:
    """Warn when the near-expiry options chain failed the engine's liquidity gate.

    The engine already refuses to read PCR / max pain / walls from such a chain
    (signals suppressed, structure excluded from ranges), but the raw numbers are
    still printed on the card — without this banner a reader would trust e.g.
    FinNifty's PCR from a ~0.2M-OI chain the engine itself calls noise."""
    if getattr(pred, "opt_chain_liquid", True):
        return ""
    return (
        "<div style='background:rgba(255,170,0,0.10);border-left:3px solid #ffae00;"
        "border-radius:0 6px 6px 0;padding:6px 10px;margin-bottom:10px;"
        "color:#e6c98a;font-size:0.8em;line-height:1.45'>"
        "⚠ <b>Thin options chain</b> — near-expiry OI is below the liquidity floor, so "
        "PCR, max pain and OI walls on this index are <b>statistical noise</b>: the engine "
        "excludes them from signals and range structure; values shown are raw-data context only. "
        "Trust the volatility range and futures/FII reads instead.</div>"
    )


def _render_verdict(pred: IndexPrediction) -> None:
    dir_color  = pred.direction_color
    conf_color = _CONF_COLOR.get(pred.confidence, "#78909C")
    dir_emoji  = _DIR_EMOJI.get(pred.direction, "↔")
    # Per-index measured track record — gates how confident the card may LOOK.
    try:
        from src.analytics.index_prediction import get_engine_accuracy, get_range_coverage
        _acc = get_engine_accuracy(fno_symbol=pred.fno_symbol)
        _cov = get_range_coverage(fno_symbol=pred.fno_symbol)
    except Exception:
        _acc, _cov = {}, {}
    # When the measured directional hit-rate is at/below a coin-flip, strip the
    # confidence colour to a warning hue so "MEDIUM/HIGH CONFIDENCE" can't masquerade
    # as actionable next to a verdict that has not earned it.
    if _acc and _acc["sign_hit"] < 48:
        conf_color = "#ff5252" if _acc["sign_hit"] < 44 else "#ffae00"
    # Built by concatenation (no indented triple-quoted template): an optional
    # box returning "" must never leave a whitespace-only line, otherwise
    # CommonMark ends the HTML block early and renders the rest as a code block.
    html = (
        f"<div style='background:#1e1e1e;border:1px solid {dir_color};"
        f"border-radius:10px;padding:16px 20px;margin-bottom:12px'>"
        f"<div style='display:flex;align-items:center;gap:12px;margin-bottom:10px'>"
        f"<div style='background:{dir_color};color:#000;font-weight:700;"
        f"font-size:1.1em;padding:6px 16px;border-radius:8px'>{dir_emoji} {pred.direction}</div>"
        f"<div style='color:{conf_color};font-weight:600'>{_confidence_phrase(pred)}</div>"
        f"<div style='margin-left:auto;font-size:0.82em;text-align:right'>"
        f"{_meter_html(pred.composite_score, show_raw=True)}"
        f"<div style='color:#666;font-size:0.85em;margin-top:2px'>"
        f"{len(pred.signals)} signals &nbsp;|&nbsp; 0=Max Bear · 50=Neutral · 100=Max Bull</div>"
        f"</div>"
        f"</div>"
        f"<div style='color:#ddd;font-size:0.9em;margin-bottom:10px'>{pred.headline}</div>"
        f"{_thin_chain_banner(pred)}"
        f"{_measured_trust_box(_acc, pred.display_name, _cov)}"
        f"{_verdict_meter_note(pred)}"
        f"{_expected_move_box(pred, no_edge=bool(_acc and _acc['sign_hit'] < 48))}"
        f"{_range_cards(pred)}"
        f"{_breakout_extension_box(pred)}"
        f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.82em'>"
        f"<div>"
        f"<div style='color:#69f0ae;font-weight:600;margin-bottom:2px'>Key Driver</div>"
        f"<div style='color:#aaa'>{pred.key_driver}</div>"
        f"</div>"
        f"<div>"
        f"<div style='color:#ff5252;font-weight:600;margin-bottom:2px'>Key Risk</div>"
        f"<div style='color:#aaa'>{pred.key_risk}</div>"
        f"</div>"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Market Context tab ────────────────────────────────────────────────────────

_IDX_LABELS = {
    "NIFTY": "Nifty 50", "BANKNIFTY": "Bank Nifty",
    "FINNIFTY": "Fin Nifty", "MIDCPNIFTY": "Midcap Nifty",
}


def _render_regime_panel(pred: IndexPrediction) -> None:
    """
    Statistical Regime tab — Hurst R/S, DFA, HMM, Permutation Entropy.

    Shows the three advanced mathematical signals computed in
    src/analytics/regime_detection.py with full interpretation.
    """
    r = pred.regime
    if r is None:
        st.info("Statistical regime signals not computed for this prediction.")
        return
    if r.error:
        st.warning(f"Regime detection error: {r.error}")
        return

    st.markdown(
        "<div style='font-size:12px;color:rgba(255,255,255,0.45);margin-bottom:16px'>"
        "Advanced statistical tools that measure market MEMORY, hidden REGIME STATE, "
        "and return COMPLEXITY — dimensions that OI/price signals cannot capture. "
        f"Computed on {r.data_points} trading days of historical returns."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── 1. Hurst Exponent + DFA ───────────────────────────────────────────────
    _h("🌊 Market Memory — Hurst Exponent & DFA",
       "Hurst R/S (Hurst 1951) and DFA (Peng 1994) measure the autocorrelation "
       "decay law of daily returns. H > 0.58: returns are PERSISTENT — past direction "
       "predicts future direction (trend-following works). H < 0.42: returns are "
       "ANTI-PERSISTENT — mean-reversion works. H ≈ 0.50: random walk — no memory edge.")

    _HURST_COLORS = {
        "Trending":       "#4CAF50",
        "Random Walk":    "#9E9E9E",
        "Mean-Reverting": "#FF9800",
    }
    h_color = _HURST_COLORS.get(r.memory_label, "#9E9E9E")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hurst R/S",   f"{r.hurst_rs:.4f}",
              help="Rescaled Range estimator. Range: [0,1]. Above 0.5 = trending.")
    c2.metric("DFA Alpha",   f"{r.dfa_alpha:.4f}",
              help="Detrended Fluctuation Analysis exponent. More robust than R/S on non-stationary series.")
    c3.metric("Average H",   f"{r.memory_avg:.4f}",
              help="Mean of R/S and DFA — reduces estimation noise.")
    c4.metric("Signal Score", f"{r.memory_score:+.1f}",
              help="Contributes to the multi-signal composite. +2=strong bullish persistence, -2=strong bearish persistence.")

    st.markdown(
        f"<div style='padding:12px 16px;border-left:4px solid {h_color};"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;margin:8px 0'>"
        f"<div style='font-size:13px;font-weight:700;color:{h_color}'>"
        f"{r.memory_label}  (H = {r.memory_avg:.4f})</div>"
        f"<div style='font-size:12px;color:rgba(255,255,255,0.6);margin-top:4px'>"
        f"{r.memory_note}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Visual gauge: Hurst on 0→1 axis with zones
    fig_h = go.Figure()
    # Zones
    fig_h.add_shape(type="rect", x0=0, x1=0.42, y0=0, y1=1,
                    fillcolor="rgba(255,152,0,0.15)", line_width=0)
    fig_h.add_shape(type="rect", x0=0.42, x1=0.58, y0=0, y1=1,
                    fillcolor="rgba(158,158,158,0.10)", line_width=0)
    fig_h.add_shape(type="rect", x0=0.58, x1=1.0, y0=0, y1=1,
                    fillcolor="rgba(76,175,80,0.15)", line_width=0)
    # Labels
    for xpos, label, col in [(0.21, "Mean-Reverting", "#FF9800"),
                               (0.50, "Random Walk",    "#9E9E9E"),
                               (0.79, "Trending",       "#4CAF50")]:
        fig_h.add_annotation(x=xpos, y=0.5, text=label, showarrow=False,
                              font=dict(size=11, color=col))
    # Markers
    for val, name, color, sym in [
        (r.hurst_rs,  "R/S",  "#2196F3", "diamond"),
        (r.dfa_alpha, "DFA",  "#FF5722", "circle"),
        (r.memory_avg,"Avg",  "#FFD700", "star"),
    ]:
        fig_h.add_trace(go.Scatter(
            x=[val], y=[0.5], mode="markers+text",
            marker=dict(size=16, color=color, symbol=sym,
                        line=dict(width=2, color="white")),
            text=[f"<b>{name}</b><br>{val:.4f}"],
            textposition="top center",
            name=name,
            showlegend=True,
        ))
    fig_h.update_layout(
        height=160, template="plotly_dark",
        xaxis=dict(range=[0, 1], title="Hurst / DFA exponent", tickformat=".2f",
                   showgrid=False),
        yaxis=dict(visible=False),
        margin=dict(t=20, b=40, l=20, r=20),
        legend=dict(orientation="h", y=-0.6, x=0.5, xanchor="center"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_h, use_container_width=True,
                    key=f"hurst_{pred.fno_symbol}")

    st.divider()

    # ── 2. Hidden Markov Model Regime ─────────────────────────────────────────
    _h("🔮 Hidden Markov Model — Market Regime",
       "3-state Gaussian HMM fitted on 90 days of [daily_return, rolling_5D_vol]. "
       "Detects the latent regime the market is currently in. "
       "Bull: positive return + normal vol. Bear: negative return + HIGH vol. "
       "Sideways: near-zero return + LOW vol. Score = p_Bull×3 − p_Bear×3.")

    if r.hmm_state == "Unknown" or not r.hmm_probs:
        st.info("HMM not computed (insufficient data).")
    else:
        p_bull, p_side, p_bear = r.hmm_probs

        _STATE_COLOR = {"Bull": "#4CAF50", "Sideways": "#FFD600", "Bear": "#EF5350"}
        state_color  = _STATE_COLOR.get(r.hmm_state, "#9E9E9E")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Current State",  r.hmm_state,
                  help="Most probable hidden state at today's close.")
        h2.metric("State Prob",     f"{r.hmm_prob:.1%}",
                  help="Probability of the detected state. Higher = more confident regime ID.")
        h3.metric("HMM Score",      f"{r.hmm_score:+.2f}",
                  help="p_Bull×3 − p_Bear×3. Range ≈ [−3, +3]. Contributes to the multi-signal composite.")
        h4.metric("Data Window",    f"{min(r.data_points, 90)}D",
                  help="Number of trading days used to fit the HMM.")

        # Probability bar — Bull / Sideways / Bear
        fig_hmm = go.Figure()
        for label, prob, color in [
            ("Bull",     p_bull, "#4CAF50"),
            ("Sideways", p_side, "#FFD600"),
            ("Bear",     p_bear, "#EF5350"),
        ]:
            fig_hmm.add_trace(go.Bar(
                name=label, x=[prob], y=["State"],
                orientation="h",
                marker_color=color,
                text=f"<b>{label}</b> {prob:.1%}",
                textposition="inside" if prob > 0.08 else "outside",
                insidetextanchor="middle",
                hovertemplate=f"{label}: {prob:.3%}<extra></extra>",
            ))
        fig_hmm.update_layout(
            barmode="stack", height=80, template="plotly_dark",
            xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
            yaxis=dict(visible=False),
            margin=dict(t=10, b=30, l=10, r=10),
            legend=dict(orientation="h", y=-1.0, x=0.5, xanchor="center"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
        )
        st.plotly_chart(fig_hmm, use_container_width=True,
                        key=f"hmm_{pred.fno_symbol}")

        st.markdown(
            f"<div style='padding:10px 14px;border-left:4px solid {state_color};"
            f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;font-size:12px;"
            f"color:rgba(255,255,255,0.6)'>{r.hmm_note}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── 3. Permutation Entropy + Sample Entropy ───────────────────────────────
    _h("🌪️ Entropy — Market Complexity & Predictability",
       "Permutation Entropy (Bandt & Pompe 2002) measures the complexity of ordinal "
       "return patterns. PE → 0: highly ordered, repeating patterns — high predictability. "
       "PE → 1: maximum disorder — signals are less reliable, widen stops. "
       "Sample Entropy measures irregularity without self-matching bias. "
       "When PE > 0.72 (Chaotic), HIGH confidence verdicts are automatically capped at MEDIUM.")

    _ENT_COLORS = {"Ordered": "#4CAF50", "Moderate": "#FFD600", "Chaotic": "#EF5350", "Unknown": "#9E9E9E"}
    ent_color = _ENT_COLORS.get(r.entropy_label, "#9E9E9E")

    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Perm Entropy",    f"{r.perm_entropy:.4f}",
              help="Permutation Entropy on 60D returns (order=4). Range [0,1]. Low = ordered/predictable.")
    e2.metric("Sample Entropy",
              f"{r.samp_entropy:.4f}" if (r.samp_entropy is not None and
              r.samp_entropy == r.samp_entropy) else "—",
              help="SampEn on 60D returns (m=2). Low = regular/predictable. High = irregular/complex.")
    e3.metric("Complexity",      r.entropy_label,
              help="Ordered (<0.50), Moderate (0.50–0.72), Chaotic (>0.72).")
    e4.metric("Conf Multiplier", f"{r.entropy_conf:.2f}×",
              help="When Chaotic: HIGH verdict capped to MEDIUM. When Ordered: normal HIGH allowed.")

    # Entropy gauge
    fig_e = go.Figure()
    fig_e.add_shape(type="rect", x0=0, x1=0.50, y0=0, y1=1,
                    fillcolor="rgba(76,175,80,0.15)", line_width=0)
    fig_e.add_shape(type="rect", x0=0.50, x1=0.72, y0=0, y1=1,
                    fillcolor="rgba(255,213,0,0.10)", line_width=0)
    fig_e.add_shape(type="rect", x0=0.72, x1=1.0, y0=0, y1=1,
                    fillcolor="rgba(239,83,80,0.15)", line_width=0)
    for xpos, label, col in [(0.25, "Ordered", "#4CAF50"),
                               (0.61, "Moderate", "#FFD600"),
                               (0.86, "Chaotic",  "#EF5350")]:
        fig_e.add_annotation(x=xpos, y=0.5, text=label, showarrow=False,
                              font=dict(size=11, color=col))
    # Threshold lines
    for xv, xt in [(0.50, "0.50"), (0.72, "0.72")]:
        fig_e.add_vline(x=xv, line_dash="dot",
                        line_color="rgba(255,255,255,0.3)", line_width=1)
        fig_e.add_annotation(x=xv, y=0.05, text=xt, showarrow=False,
                              font=dict(size=9, color="rgba(255,255,255,0.4)"))
    fig_e.add_trace(go.Scatter(
        x=[r.perm_entropy], y=[0.5], mode="markers+text",
        marker=dict(size=18, color=ent_color, symbol="diamond",
                    line=dict(width=2, color="white")),
        text=[f"<b>PE = {r.perm_entropy:.4f}</b>"],
        textposition="top center",
        showlegend=False,
    ))
    fig_e.update_layout(
        height=160, template="plotly_dark",
        xaxis=dict(range=[0, 1], title="Permutation Entropy", tickformat=".2f",
                   showgrid=False),
        yaxis=dict(visible=False),
        margin=dict(t=20, b=40, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_e, use_container_width=True,
                    key=f"entropy_{pred.fno_symbol}")

    st.markdown(
        f"<div style='padding:10px 14px;border-left:4px solid {ent_color};"
        f"background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;font-size:12px;"
        f"color:rgba(255,255,255,0.6)'>{r.entropy_note}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── 4. Combined interpretation ────────────────────────────────────────────
    st.markdown("#### Combined Statistical Reading")

    h_label  = r.memory_label
    hmm_state = r.hmm_state
    ent_label = r.entropy_label

    bullets = []
    if h_label == "Trending" and r.memory_score > 0:
        bullets.append(("🌊", "#4CAF50",
                         f"Persistent uptrend (H={r.memory_avg:.3f}) — momentum signals have statistical backing."))
    elif h_label == "Trending" and r.memory_score < 0:
        bullets.append(("🌊", "#EF5350",
                         f"Persistent downtrend (H={r.memory_avg:.3f}) — bearish momentum has statistical backing."))
    elif h_label == "Mean-Reverting":
        bullets.append(("🔄", "#FF9800",
                         f"Anti-persistent returns (H={r.memory_avg:.3f}) — strong directional moves are likely to reverse."))
    else:
        bullets.append(("🎲", "#9E9E9E",
                         f"Near-random walk (H={r.memory_avg:.3f}) — no autocorrelation edge."))

    if hmm_state == "Sideways":
        bullets.append(("↔️", "#FFD600",
                         "HMM detects a sustained sideways/consolidation regime — "
                         "directional breakouts lack statistical regime support."))
    elif hmm_state == "Bull":
        bullets.append(("📈", "#4CAF50",
                         f"HMM confirms Bull regime (p={r.hmm_prob:.0%}) — "
                         "regime supports long positioning."))
    elif hmm_state == "Bear":
        bullets.append(("📉", "#EF5350",
                         f"HMM confirms Bear regime (p={r.hmm_prob:.0%}) — "
                         "regime supports caution / short positioning."))

    if ent_label == "Chaotic":
        bullets.append(("🌪️", "#EF5350",
                         f"High entropy (PE={r.perm_entropy:.4f}) — return patterns are disordered. "
                         "Reduce position size, widen stops, cap HIGH confidence verdicts at MEDIUM."))
    elif ent_label == "Ordered":
        bullets.append(("🎯", "#4CAF50",
                         f"Low entropy (PE={r.perm_entropy:.4f}) — patterns are predictable and repeating. "
                         "Directional signals have elevated reliability."))

    for emoji, color, text in bullets:
        st.markdown(
            f"<div style='padding:8px 12px;margin:4px 0;"
            f"background:rgba(255,255,255,0.03);border-left:3px solid {color};"
            f"border-radius:0 6px 6px 0;font-size:13px'>"
            f"{emoji} <span style='color:{color};font-weight:600'></span> "
            f"<span style='color:rgba(255,255,255,0.75)'>{text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_market_context(pred: IndexPrediction) -> None:
    ctx = pred.market_context
    if ctx is None:
        st.caption("Market context not available.")
        return

    # ── India VIX ─────────────────────────────────────────────────────────────
    _h("India VIX", "India VIX measures expected 30-day market volatility (fear gauge). VIX above 20 = high fear = options expensive = markets unstable. VIX below 13 = complacency = cheap options = potential for sharp moves. Contrarian signal: extreme high VIX often marks a bottom; extreme low VIX often marks a top.")
    v1, v2, v3, v4 = st.columns(4)
    vix_chg_str = f"{ctx.vix_pct_chg:+.2f}%" if ctx.vix_pct_chg is not None else "—"
    vix_5d_str  = f"{ctx.vix_5d_chg_pct:+.1f}%" if ctx.vix_5d_chg_pct is not None else "—"
    v1.metric("India VIX",    f"{ctx.vix_close:.2f}" if ctx.vix_close else "—")
    v2.metric("Today Change", vix_chg_str)
    v3.metric("5-Day Change", vix_5d_str)
    v4.metric("Regime",       ctx.vix_regime)

    if ctx.vix_close is not None:
        vix = ctx.vix_close
        chg = ctx.vix_pct_chg or 0
        if vix < 12:
            interp = "Complacency zone — be cautious of unexpected spikes"
            ic = _NEG
        elif vix < 18:
            interp = f"Normal range — {'fear clearing (bullish)' if chg < -3 else 'anxiety building (bearish)' if chg > 3 else 'stable'}"
            ic = _POS if chg < -3 else (_NEG if chg > 3 else _NEU)
        elif vix < 22:
            interp = f"Elevated — {'fear receding (relief rally likely)' if chg < -5 else 'uncertainty high, wide swings expected'}"
            ic = _POS if chg < -5 else _NEG
        else:
            interp = "Fear zone — contrarian buy signal; capitulation near"
            ic = _POS
        st.markdown(
            f"<div style='color:{ic};font-size:0.85em;margin-top:4px'>📊 {interp}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Sector Breadth ────────────────────────────────────────────────────────
    _h("Sector Breadth", "Fraction of NSE sectors showing above-average delivery volume today. High breadth (70%+) = broad market participation = rally has conviction. Low breadth (below 30%) = narrow market = only 1-2 sectors driving the index = fragile rally. Use this to confirm or doubt the index-level signal.")
    if ctx.breadth_pct_advancing is not None:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Advancing",     str(ctx.advancing_sectors))
        b2.metric("Declining",     str(ctx.declining_sectors))
        b3.metric("Total Sectors", str(ctx.total_sectors))
        b4.metric("% Advancing",   f"{ctx.breadth_pct_advancing:.0f}%")

        adv_pct   = ctx.breadth_pct_advancing
        bar_color = _POS if adv_pct >= 65 else (_NEG if adv_pct <= 35 else "#FFD600")
        st.markdown(
            f"""<div style="background:#333;border-radius:4px;height:12px;width:100%;margin:6px 0">
                <div style="background:{bar_color};border-radius:4px;height:12px;width:{adv_pct:.0f}%"></div>
            </div>
            <div style="font-size:0.78em;color:#888">
                {ctx.advancing_sectors} advancing &nbsp;·&nbsp; {ctx.declining_sectors} declining
                &nbsp;·&nbsp; {ctx.total_sectors - ctx.advancing_sectors - ctx.declining_sectors} flat
            </div>""",
            unsafe_allow_html=True,
        )

        if ctx.cyclical_avg_chg is not None and ctx.defensive_avg_chg is not None:
            st.markdown("**Sector Rotation**")
            r1, r2, r3 = st.columns(3)
            mode_color = _POS if ctx.risk_mode == "RISK-ON" else (_NEG if ctx.risk_mode == "RISK-OFF" else _NEU)
            r1.metric("Cyclicals (Bank/Auto/Metal)", f"{ctx.cyclical_avg_chg:+.2f}%")
            r2.metric("Defensives (FMCG/Pharma/IT)", f"{ctx.defensive_avg_chg:+.2f}%")
            r3.metric("Risk Mode", ctx.risk_mode)
            st.markdown(
                f"<div style='color:{mode_color};font-size:0.85em'>"
                f"{'⚡ Institutions rotating into growth sectors — RISK-ON' if ctx.risk_mode == 'RISK-ON' else '🛡️ Flight to defensives — RISK-OFF' if ctx.risk_mode == 'RISK-OFF' else '⚖️ Mixed rotation — no dominant theme'}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Sector breadth data not available for this date.")

    st.divider()

    # ── FII/DII/Client/Pro positioning ────────────────────────────────────────
    _h("Institutional Positioning (Index Futures OI)", "Net futures contracts held by each participant type. FII (Foreign Institutions) = smart money — follow their direction. DII (Domestic Institutions) = often contrarian to FII. Client = retail — contrarian indicator (when retail is very long, be cautious). Pro = proprietary desks. Key rule: FII net short but REDUCING position = covering = bullish price pressure.")
    if ctx.fao_date:
        lag = (ctx.trade_date - ctx.fao_date).days
        st.caption(
            f"Source: FAO Participant OI as of {ctx.fao_date.strftime('%d %b %Y')}"
            + (f" ({lag} day lag)" if lag > 0 else " (current)")
        )

        parts = [
            ("FII",    ctx.fii_fut_idx_net,    "Foreign Institutional Investors"),
            ("DII",    ctx.dii_fut_idx_net,    "Domestic Institutional Investors"),
            ("Client", ctx.client_fut_idx_net, "Retail / Client"),
            ("Pro",    ctx.pro_fut_idx_net,    "Proprietary Desks"),
        ]
        html = """
        <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
            <thead>
                <tr style="border-bottom:1px solid #333;color:#888">
                    <th style="text-align:left;padding:5px 8px">Participant</th>
                    <th style="text-align:right;padding:5px 8px">Net Contracts</th>
                    <th style="text-align:left;padding:5px 8px;color:#666">Interpretation</th>
                </tr>
            </thead><tbody>
        """
        for name, net, desc in parts:
            color   = _POS if net > 0 else (_NEG if net < 0 else _NEU)
            bias    = "LONG 🐂" if net > 50_000 else ("SHORT 🐻" if net < -50_000 else "NEUTRAL")
            bar_w   = min(abs(net) / 5_000, 100)
            bar_col = _POS if net > 0 else (_NEG if net < 0 else _NEU)
            html += f"""
            <tr style="border-bottom:1px solid #1a1a1a">
                <td style="padding:6px 8px;color:#ccc;font-weight:600">{name}</td>
                <td style="padding:6px 8px;text-align:right">
                    <span style="color:{color};font-weight:700">{net:+,}</span>
                </td>
                <td style="padding:6px 8px">
                    <div style="background:#333;border-radius:3px;height:6px;width:100px;display:inline-block;vertical-align:middle;margin-right:6px">
                        <div style="background:{bar_col};border-radius:3px;height:6px;width:{bar_w:.0f}px"></div>
                    </div>
                    <span style="color:#888;font-size:0.85em">{bias}</span>
                </td>
            </tr>"""
        html += "</tbody></table>"
        st.html(html.strip())

        # ── FII Position Change (Day-over-Day) ────────────────────────────────
        if ctx.fii_prev_fao_date:
            st.markdown("**FII Position Change (Day-over-Day)**")
            chg       = ctx.fii_net_change_1d
            chg_color = _POS if chg > 0 else (_NEG if chg < 0 else _NEU)
            action    = "COVERING SHORTS 🔄" if chg > 0 else ("ADDING SHORTS 🐻" if chg < 0 else "FLAT")
            interp    = (
                "FII buying back shorts — creates upward price pressure; watch for reversal trigger."
                if chg > 0 else
                "FII building more shorts — increasing bearish conviction. Doubling down on existing position."
                if chg < 0 else
                "FII position unchanged day-over-day."
            )
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric(f"Net {ctx.fii_prev_fao_date.strftime('%d %b')}", f"{ctx.fii_fut_idx_net_prev:+,}")
            pc2.metric(f"Net {ctx.fao_date.strftime('%d %b')}",           f"{ctx.fii_fut_idx_net:+,}")
            pc3.metric("1-Day Change",                                     f"{chg:+,}")
            st.markdown(
                f"<div style='color:{chg_color};font-size:0.85em;padding:4px 0'>"
                f"<b>{action}</b> — {interp}</div>",
                unsafe_allow_html=True,
            )

        # ── FII Stock Futures (market-neutral context) ─────────────────────────
        if ctx.fii_stock_fut_net_cr is not None:
            sfc       = ctx.fii_stock_fut_net_cr
            sfc_color = _POS if sfc > 0 else _NEG
            sfc_txt   = f"Rs{abs(sfc):,.0f}Cr {'Net BUY' if sfc > 0 else 'Net SELL'}"
            note = ""
            if sfc > 500 and ctx.fii_fut_idx_net < -50_000:
                note = " — Market-Neutral Pairs Trade: index short is a hedge, not pure directional bearish."
            elif sfc < -500 and ctx.fii_fut_idx_net < -50_000:
                note = " — Double bearish: selling both index futures AND stock futures."
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;background:#1a1a1a;"
                f"border-left:3px solid {sfc_color};border-radius:6px;font-size:0.85em'>"
                f"<b style='color:#ccc'>FII Stock Futures (Today):</b> "
                f"<span style='color:{sfc_color}'>{sfc_txt}</span>"
                f"<span style='color:#888'>{note}</span></div>",
                unsafe_allow_html=True,
            )

        # ── FII Options Book ──────────────────────────────────────────────────
        if ctx.fii_opt_delta != 0:
            st.markdown("**FII Options Book**")
            o1, o2, o3 = st.columns(3)
            delta_color = _POS if ctx.fii_opt_delta > 0 else _NEG
            o1.metric("FII Net Call OI", f"{ctx.fii_opt_call_net:+,}")
            o2.metric("FII Net Put OI",  f"{ctx.fii_opt_put_net:+,}")
            o3.metric("Options Delta",   f"{ctx.fii_opt_delta:+,}")
            bias_txt = (
                "FII positioned for upside via options (net long calls)"
                if ctx.fii_opt_delta > 0
                else "FII heavily hedged — net long puts (expect downside or volatility)"
            )
            st.markdown(
                f"<div style='color:{delta_color};font-size:0.82em;margin-top:4px'>{bias_txt}</div>",
                unsafe_allow_html=True,
            )
            opt_flow_cr = ctx.fii_opt_net_flows.get(pred.fno_symbol)
            if opt_flow_cr is not None:
                ofl_color = _POS if opt_flow_cr > 0 else _NEG
                ofl_txt   = f"Rs{abs(opt_flow_cr):,.0f}Cr {'Net BUY' if opt_flow_cr > 0 else 'Net SELL'}"
                st.markdown(
                    f"<div style='margin-top:6px;font-size:0.82em;color:#aaa'>"
                    f"FII Options Flow Today ({pred.fno_symbol}): "
                    f"<span style='color:{ofl_color};font-weight:600'>{ofl_txt}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── FII Today's Futures Flow ──────────────────────────────────────────
        flow_cr = ctx.fii_symbol_flows.get(pred.fno_symbol)
        if flow_cr is not None:
            flow_color = _POS if flow_cr > 0 else _NEG
            flow_txt   = f"Rs{abs(flow_cr):,.0f}Cr {'Net BUY' if flow_cr > 0 else 'Net SELL'}"
            st.markdown(
                f"<div style='margin-top:8px;padding:8px 12px;background:#1a1a1a;"
                f"border-left:3px solid {flow_color};border-radius:6px;"
                f"color:{flow_color};font-weight:600;font-size:0.9em'>"
                f"FII Net Futures Flow Today — {pred.fno_symbol}: {flow_txt}</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── FII 5-Day Cumulative Flow (All Indices) ───────────────────────────
        if ctx.fii_cumul_flows_5d:
            _h("FII 5-Day Cumulative Futures Flow (Rs Crore)", "Sum of FII net futures buying/selling over the last 5 trading days in Rupee Crore terms. Single-day flow can be noise (expiry, rebalancing). 5-day cumulative flow reveals the TRUE institutional trend. Positive = net accumulation; Negative = net distribution. Strong Accumulation above Rs 5,000 Cr is significant.")
            st.caption("5D rolling net — sustained institutional pressure; single-day flows can be noise")
            html5 = """
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
                <thead>
                    <tr style="border-bottom:1px solid #333;color:#888">
                        <th style="text-align:left;padding:5px 8px">Index</th>
                        <th style="text-align:right;padding:5px 8px">5D Net (Rs Cr)</th>
                        <th style="text-align:left;padding:5px 8px">Signal</th>
                    </tr>
                </thead><tbody>
            """
            for sym, label in _IDX_LABELS.items():
                cumul = ctx.fii_cumul_flows_5d.get(sym)
                if cumul is None:
                    continue
                c_col = _POS if cumul > 0 else (_NEG if cumul < 0 else _NEU)
                if cumul >= 5_000:    sig = "Strong Accumulation"
                elif cumul >= 2_000:  sig = "Accumulation"
                elif cumul > -2_000:  sig = "Neutral"
                elif cumul > -5_000:  sig = "Distribution"
                else:                 sig = "Heavy Distribution"
                hl = " background:#252015;" if sym == pred.fno_symbol else ""
                name_color = "#FFD600" if sym == pred.fno_symbol else "#ccc"
                name_weight = "700" if sym == pred.fno_symbol else "400"
                dir_arrow = "▲" if cumul >= 0 else "▼"
                html5 += (
                    f"<tr style='border-bottom:1px solid #1a1a1a;{hl}'>"
                    f"<td style='padding:6px 8px;color:{name_color};font-weight:{name_weight}'>{label}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:{c_col};font-weight:700'>"
                    f"{dir_arrow} Rs{abs(cumul):,.0f}</td>"
                    f"<td style='padding:6px 8px;color:#aaa'>{sig}</td>"
                    f"</tr>"
                )
            html5 += "</tbody></table>"
            st.html(html5.strip())

        # ── FII OI Buildup Trend (All Indices) ────────────────────────────────
        if ctx.fii_oi_cr_latest and ctx.fii_oi_cr_5d_ago:
            _h("FII OI Rs-Cr Buildup Trend (5-Day)", "Whether FII open interest is growing or shrinking over 5 days. Growing OI + FII net short = FIIs adding new short bets = increasing bearish conviction. Shrinking OI + FII net short = FIIs covering shorts = potential reversal signal. Growing OI + FII net long = building bullish conviction.")
            st.caption("Growing OI + net short = adding conviction | Shrinking OI + net short = covering (reversal risk)")
            html_oi = """
            <table style="width:100%;border-collapse:collapse;font-size:0.85em;margin-top:8px">
                <thead>
                    <tr style="border-bottom:1px solid #333;color:#888">
                        <th style="text-align:left;padding:5px 8px">Index</th>
                        <th style="text-align:right;padding:5px 8px">OI Now (Rs Cr)</th>
                        <th style="text-align:right;padding:5px 8px">OI 5D Ago</th>
                        <th style="text-align:right;padding:5px 8px">5D Chg</th>
                        <th style="text-align:left;padding:5px 8px">Signal</th>
                    </tr>
                </thead><tbody>
            """
            fii_net = ctx.fii_fut_idx_net
            for sym, label in _IDX_LABELS.items():
                now = ctx.fii_oi_cr_latest.get(sym)
                ago = ctx.fii_oi_cr_5d_ago.get(sym)
                if now is None or ago is None or ago == 0:
                    continue
                chg_pct = (now - ago) / ago * 100
                chg_col = _POS if chg_pct > 0 else _NEG
                if chg_pct >= 10 and fii_net < -50_000:    sig = "Building Shorts"
                elif chg_pct >= 10 and fii_net > 50_000:   sig = "Building Longs"
                elif chg_pct <= -10 and fii_net < -50_000: sig = "Covering Shorts"
                elif chg_pct <= -10 and fii_net > 50_000:  sig = "Reducing Longs"
                else:                                        sig = "Stable"
                hl = " background:#252015;" if sym == pred.fno_symbol else ""
                name_color  = "#FFD600" if sym == pred.fno_symbol else "#ccc"
                name_weight = "700" if sym == pred.fno_symbol else "400"
                html_oi += (
                    f"<tr style='border-bottom:1px solid #1a1a1a;{hl}'>"
                    f"<td style='padding:6px 8px;color:{name_color};font-weight:{name_weight}'>{label}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:#fff'>Rs{now:,.0f}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:#666'>Rs{ago:,.0f}</td>"
                    f"<td style='padding:6px 8px;text-align:right;color:{chg_col};font-weight:700'>{chg_pct:+.1f}%</td>"
                    f"<td style='padding:6px 8px;color:#aaa'>{sig}</td>"
                    f"</tr>"
                )
            html_oi += "</tbody></table>"
            st.html(html_oi.strip())

    else:
        st.caption("FAO participant data not available for this date.")

    # ── PE Valuation ─────────────────────────────────────────────────────────
    if ctx.nifty_pe is not None:
        st.divider()
        _h("Nifty 50 Valuation", "Price-to-Earnings ratio of Nifty 50. Historical mean is ~22x. Above 24x = expensive = lower margin of safety for bulls. Below 18x = cheap = strong long-term entry. PE alone does not predict short-term direction but sets the risk context — at PE 26x a 5% dip is healthy correction; at PE 18x the same dip is a buying opportunity.")
        pe = ctx.nifty_pe
        if pe < 16:
            pe_color = _POS; pe_label = "Cheap"
            pe_note  = "Below historical mean (~22x) — attractive entry; cushion on downside"
        elif pe < 22:
            pe_color = _POS; pe_label = "Fair Value (Low)"
            pe_note  = "Below mean — reasonable valuation, no major earnings headwind"
        elif pe < 28:
            pe_color = _NEU; pe_label = "Fair Value (High)"
            pe_note  = "Near/above mean — fully priced; requires earnings support to sustain"
        else:
            pe_color = _NEG; pe_label = "Expensive"
            pe_note  = "Above historical norms (>28x) — elevated correction risk on negative catalysts"
        pe1, pe2, pe3 = st.columns(3)
        pe1.metric("Nifty PE Ratio",   f"{pe:.1f}x")
        pe2.metric("Valuation Zone",   pe_label)
        pe3.metric("Historical Mean",  "~22x")
        st.markdown(
            f"<div style='color:{pe_color};font-size:0.85em;margin-top:4px'>📊 {pe_note}</div>",
            unsafe_allow_html=True,
        )


# ── Main render ───────────────────────────────────────────────────────────────

def render(selected_date: date) -> None:
    st.subheader("Index Prediction — Institutional Market-Structure Read")

    # ── Honest track record (radical transparency) ────────────────────────────
    try:
        from src.analytics.index_prediction import get_engine_accuracy, get_range_coverage
        _acc = get_engine_accuracy()
        _cov_all = get_range_coverage()
    except Exception:
        _acc, _cov_all = {}, {}
    if _acc:
        _sh = _acc["sign_hit"]; _ic = _acc["ic"]
        _edge = ("a real edge" if _sh >= 56 else "near coin-flip" if _sh >= 47 else "below coin-flip")
        _col = "#ff9100" if _sh < 56 else "#00c853"
        _cov_txt = ""
        if _cov_all:
            _c90 = f" ({_cov_all['cov_90d']:.0f}% last 90d)" if _cov_all.get("cov_90d") else ""
            _cov_txt = (f" The <b>range</b> is the trustworthy product: measured band coverage "
                        f"<b style='color:#40c4ff'>{_cov_all['cov_all']:.0f}%</b>{_c90} vs ~68% target.")
        st.markdown(
            f"<div style='border-left:4px solid {_col};background:rgba(255,255,255,0.03);"
            f"padding:8px 12px;margin:4px 0;border-radius:0 6px 6px 0;font-size:12.5px'>"
            f"<b>📊 Live track record:</b> next-day directional hit-rate "
            f"<b style='color:{_col}'>{_sh:.0f}%</b> (n={_acc['n_dir']}, since {_acc['since']}) · "
            f"composite→return IC {_ic:+.3f} — <b>{_edge}</b>.{_cov_txt} "
            f"Treat this page as a <b>market-structure map</b> (FII positioning, OI walls, PCR, "
            f"vol regime, ranges), <b>not</b> a precise forecast. Next-day index direction is near-random; "
            f"use the verdict as one input, sized accordingly.</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        "Index-F&O market-structure engine: OI-Price Matrix · PCR · OI-Premium Matrix · Max Pain · "
        "Price Mean-Reversion · FII Institutional · FII Options Delta · FII Flow · FII 5D Cumulative · "
        "FII OI Buildup · Pro/Client Participant · Short Squeeze · Multi-Expiry PCR · Dual Max Pain · "
        "Gamma Wall · Hurst · HMM. Carry, FII Position-Change & Memory shown as context (no vote). "
        "Pruned as non-index-F&O / no measured edge: VWAP · RSI · Sector Breadth · Cyclical-Defensive · "
        "PE Valuation · VIX-vote · constituent stock OI · Entropy. Live signal count shown per card."
    )

    with st.expander("📖 How to read this page", expanded=False):
        st.markdown("""
**Prediction Cards (top row)** — one card per index (Nifty 50, Bank Nifty, Fin Nifty, Midcap Nifty).
- **Direction badge** (🟢 UP / 🔴 DOWN / 🟡 SIDEWAYS) — tomorrow's expected move from the index-F&O signal engine (live signal count shown on each card)
- **Bull/Bear Meter (0–100)** — intuitive conviction scale replacing the raw ±20 score:
  - **0–20 = STRONG BEAR** 🔴🔴 — very high bearish conviction, multiple strong signals aligned
  - **21–35 = BEAR** 🔴 — clear bearish bias, institutions positioned short
  - **36–45 = MILD BEAR** 🟠 — slight bearish lean, mixed signals with a downward tilt
  - **46–54 = NEUTRAL** ⚖️ — no clear edge, range-bound likely
  - **55–64 = MILD BULL** 🟡 — slight bullish lean
  - **65–80 = BULL** 🟢 — clear bullish bias
  - **81–100 = STRONG BULL** 🟢🟢 — very high bullish conviction
  - Raw score shown in brackets for reference — 50 = neutral, each 10 points = one strong signal
- **Expected Move forecast** (in the Verdict & Signals tab) — answers *how many points*:
  - **Expected Range (~68%)** — the 1σ volatility band: tomorrow's close lands inside this ~68% of the time
    (validated coverage on the live prediction_log: NIFTY ~72% / others 63–70%; a 75% target failed OOS,
    so the band is an honest ~1σ, not 75%). Computed from realized 20-day volatility blended with India VIX
    (forward-looking implied vol), per-index vol-calibrated. Bank Nifty / Midcap ranges are naturally wider.
  - **Directional Target** — where the engine expects the close, skewed within the range by conviction
    (composite score). Strong bull → upper end / above; neutral → middle; strong bear → lower end / below.
  - **Sideways Band** — the data-driven ± threshold (40% of the 1σ move). If the predicted move stays
    inside this band → range-bound. Beyond it → genuine directional bias. (Replaces a fixed ±40 pts
    with a volatility-scaled band — wider on high-VIX days, tighter on calm days.)
  - On expiry day (DTE ≤ 1) the range is compressed ×0.7 for gamma pinning.
- **PCR** — Put-Call Ratio. Above 1.3 = peak fear (contrarian bullish). Below 0.70 = complacency (contrarian bearish)
- **Carry** — annualised futures premium vs spot. Above 8% = bullish demand. Below 3% or negative = smart money cautious
- **FII net** — Foreign Institutional net futures contracts today. Positive = net long, negative = net short
- **S / MP / R** — Support / Max Pain / Resistance for the nearest expiry (max pain = strike with least OI writer pain)

**Summary Bar (below cards)** — market-wide context shared across all 4 indices.
- **India VIX** — fear gauge. Below 13 = calm market. 13–17 = normal. 17–20 = elevated caution. Above 20 = high fear / sharp moves likely
- **Breadth** — how many of the tracked sectors are advancing today. Above 60% = broad rally. Below 40% = broad selloff. Mixed = rotation
- **FII Fut Net** — total FII net futures position (index futures, all indices combined). Persistent negative = institutions are hedging or shorting the market
- **FII 1D Chg** — change in FII futures position vs yesterday. COVERING = reducing shorts (bullish signal). ADDING = adding to shorts (bearish signal)
- **Risk Mode** — derived from VIX + breadth + FII positioning: RISK-ON (buy-the-dip) / RISK-OFF (reduce exposure) / NEUTRAL
- **VIX Regime** — CALM / NORMAL / ELEVATED / HIGH based on VIX level; determines how much signal weight to give to volatility signals

**Detail Tabs** — select an index from the dropdown, then explore:
- **Verdict & Signals** — full signal breakdown with individual scores and explanations
- **Key Levels** — put wall, call wall, max pain chart with cost-of-carry context
- **Today vs Yesterday** — spot changes in OI, carry, PCR, breadth to see what shifted overnight
- **OI Structure** — call/put OI chart showing where options writers have concentrated positions (these act as support/resistance)
- **Market Context** — India VIX trend, FII 5-day cumulative flow, sector breadth table, Nifty PE valuation

**Tip:** Use the Trading Date dropdown (sidebar) to select a past date and see what the engine predicted — then compare to what actually happened next day.
        """)

    preds = cached_index_predictions(selected_date)
    if not preds:
        st.warning("No prediction data available. Fetch F&O bhavcopy first.")
        return

    # ── Prediction cards ──────────────────────────────────────────────────────
    cols = st.columns(len(preds))
    for col, pred in zip(cols, preds):
        with col:
            _render_card(pred)

    st.divider()

    # ── Market-wide summary bar (breadth + VIX + FII) ─────────────────────────
    ctx = preds[0].market_context if preds else None
    if ctx:
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("India VIX",
                  f"{ctx.vix_close:.1f}" if ctx.vix_close else "—",
                  f"{ctx.vix_pct_chg:+.2f}%" if ctx.vix_pct_chg else None,
                  help="NSE India VIX (volatility index). Below 13 = calm. 13–17 = normal. 17–20 = cautious. Above 20 = high fear, sharp moves likely. Delta shown is today's % change.")
        m2.metric("Breadth",
                  f"{ctx.breadth_pct_advancing:.0f}%" if ctx.breadth_pct_advancing else "—",
                  f"{ctx.advancing_sectors}/{ctx.total_sectors} up",
                  help="Sector breadth: % of tracked sectors with positive price change today. Above 60% = broad rally. Below 40% = broad selloff. Mixed = rotation between sectors.")
        m3.metric("FII Fut Net", f"{ctx.fii_fut_idx_net:+,}" if ctx.fao_date else "—",
                  help="FII net index futures position today (contracts). Positive = net long (bullish), negative = net short (hedging or bearish). Sustained large negative = institutional caution.")
        if ctx.fii_prev_fao_date:
            _chg = ctx.fii_net_change_1d
            m4.metric("FII 1D Chg",
                      f"{_chg:+,}",
                      "COVERING" if _chg > 0 else ("ADDING" if _chg < 0 else "FLAT"),
                      help="Change in FII futures position vs previous F&O day. COVERING = reducing short positions (bullish signal). ADDING = increasing shorts (bearish signal). FLAT = no meaningful change.")
        else:
            m4.metric("FII 1D Chg", "—",
                      help="Change in FII futures position vs previous F&O day. Not available when prior day data is missing.")
        m5.metric("Risk Mode",   ctx.risk_mode,
                  help="Composite market mode derived from VIX level + sector breadth + FII positioning. RISK-ON = favourable for longs. RISK-OFF = reduce exposure, prefer cash/hedges. NEUTRAL = mixed signals.")
        m6.metric("VIX Regime",  ctx.vix_regime,
                  help="VIX classification: CALM (VIX < 13), NORMAL (13–17), ELEVATED (17–20), HIGH (> 20). Regime affects signal weights — in HIGH regime, mean-reversion signals get more weight than trend signals.")
        st.divider()

    # ── Weekly (5-day) mean-reversion outlook — contrarian complement ─────────
    try:
        from src.analytics.weekly_outlook import get_weekly_outlook, VALIDATION
        wk = get_weekly_outlook(selected_date)
    except Exception:
        wk, VALIDATION = [], ""
    if wk:
        st.markdown("#### 📅 Weekly Outlook — 5-Day Mean-Reversion Bias")
        st.caption(
            "A separate, **contrarian** lens to the next-day engine: how *extended* each index is "
            "(position in its 20-day range + RSI). The 5-day move mean-reverts — overbought fades, "
            "oversold bounces. Validated: " + VALIDATION + "."
        )
        _wc = {"UP": "#00c853", "DOWN": "#ff5252", "NEUTRAL": "#9e9e9e"}
        _wl = {"UP": "↑ Bounce likely", "DOWN": "↓ Pullback likely", "NEUTRAL": "→ Neutral"}
        wcols = st.columns(len(wk))
        for col, r in zip(wcols, wk):
            c = _wc[r["weekly_bias"]]
            col.markdown(
                f"<div style='border-left:3px solid {c};padding:4px 10px'>"
                f"<div style='font-size:12px;color:rgba(255,255,255,0.6)'>{r['label']}</div>"
                f"<div style='font-size:15px;font-weight:700;color:{c}'>{_wl[r['weekly_bias']]}</div>"
                f"<div style='font-size:10.5px;color:rgba(255,255,255,0.45)'>extension {r['extension']:.2f} "
                f"(pctl {r['extension_pctl']:.0f}%) · RSI {r['rsi']:.0f} · range-pos {r['pos_in_range']:.0f}%</div>"
                f"</div>", unsafe_allow_html=True,
            )
        _nb = [r for r in wk if r["weekly_bias"] != "NEUTRAL"]
        if _nb:
            st.caption(" · ".join(f"**{r['label']}**: {r['rationale']}" for r in _nb))
        st.divider()

    # ── Index selector ────────────────────────────────────────────────────────
    sym_options  = [p.fno_symbol for p in preds]
    selected_sym = st.selectbox(
        "Select Index for Detail",
        options=sym_options,
        format_func=lambda s: next(p.display_name for p in preds if p.fno_symbol == s),
    )
    detail = next((p for p in preds if p.fno_symbol == selected_sym), None)
    if detail is None:
        return
    if not detail.data_available:
        st.info(detail.note or "No F&O data available for this index on the selected date.")
        return

    # ── Detail tabs ───────────────────────────────────────────────────────────
    tab_verdict, tab_levels, tab_compare, tab_oi, tab_ctx, tab_regime = st.tabs([
        "Verdict & Signals", "Key Levels", "Today vs Yesterday",
        "OI Structure", "Market Context", "📐 Statistical Regime",
    ])

    with tab_verdict:
        _render_verdict(detail)
        _render_signals(detail)

    with tab_levels:
        _render_key_levels(detail)
        _h("Cost of Carry", "Annualised implied carry = (Futures price - Spot) / Spot / days_to_expiry * 365 * 100. At India repo rate ~6.5%, fair carry is 6-7%. Above 8% = market expects rise (bullish bias in futures). Below 3% = market expects fall or uncertainty. Negative carry = futures trading at discount = strong bearish signal from smart money.")
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Futures Price", _fmt(detail.futures_price, 0) if detail.futures_price else "—")
        cb.metric("Spot Close",    _fmt(detail.spot_close, 0)    if detail.spot_close    else "—")
        cc.metric("Carry (pts)",   _fmt(detail.carry_pts, 0, " pts") if detail.carry_pts is not None else "—")
        cd.metric("Carry % ann",   _fmt(detail.carry_pct_ann, 1, "%") if detail.carry_pct_ann is not None else "—")
        if detail.carry_label and detail.carry_label != "No Data":
            carry_color = (
                "#00C853" if "Bullish"  in detail.carry_label else
                "#FF5252" if "Bearish"  in detail.carry_label or "Backwardation" in detail.carry_label else
                "#FFD600"
            )
            st.markdown(
                f"<div style='color:{carry_color};font-weight:600;margin-top:4px'>"
                f"Carry Signal: {detail.carry_label}</div>",
                unsafe_allow_html=True,
            )

    with tab_compare:
        _render_comparison(detail)

    with tab_oi:
        _render_oi_chart(detail)

    with tab_ctx:
        _render_market_context(detail)

    with tab_regime:
        _render_regime_panel(detail)
