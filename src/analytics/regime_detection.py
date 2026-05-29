"""
Regime Detection — Hurst Exponent, DFA, Hidden Markov Model, Permutation Entropy.

Four complementary statistical tools that measure dimensions traditional OI/price
signals cannot reach: market MEMORY, hidden REGIME state, and return COMPLEXITY.

Mathematical foundations:
  Hurst R/S   : Hurst (1951). Rescaled Range. H>0.5 = trending, H<0.5 = mean-reverting.
  DFA         : Peng et al. (1994). Detrended Fluctuation Analysis. Robust to non-stationarity.
                Both R/S and DFA measure the autocorrelation decay law of the return series.
  HMM         : Rabiner (1989). 3-state Gaussian HMM with Baum-Welch (E-M) + scaled
                forward-backward. States: Bull / Sideways / Bear (labeled by mean return).
  PE          : Bandt & Pompe (2002). Permutation Entropy. Measures ordinal pattern
                complexity. Fast and noise-robust. Low PE = ordered/predictable.

No external libraries — pure NumPy + Python math.
Data sourced via the data layer (query_dataframe).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.repository import query_dataframe

__all__ = ["RegimeResult", "get_regime_signals"]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    # ── Hurst + DFA ───────────────────────────────────────────────────────────
    hurst_rs:     float = 0.5       # R/S Hurst exponent  [0,1]
    dfa_alpha:    float = 0.5       # DFA scaling exponent [0,1]
    memory_avg:   float = 0.5       # mean of the two
    memory_label: str   = "Random Walk"   # Trending | Random Walk | Mean-Reverting
    memory_score: float = 0.0       # −2 … +2 (direction-sensitive)
    memory_note:  str   = ""

    # ── Hidden Markov Model ───────────────────────────────────────────────────
    hmm_state:  str   = "Unknown"   # Bull | Sideways | Bear
    hmm_prob:   float = 0.0         # probability of current state
    hmm_score:  float = 0.0         # probability-weighted score [−3, +3]
    hmm_probs:  list  = field(default_factory=list)   # [p_bull, p_sideways, p_bear]
    hmm_note:   str   = ""

    # ── Permutation Entropy ───────────────────────────────────────────────────
    perm_entropy:  float = 1.0      # [0,1] — 0=perfectly ordered, 1=maximum chaos
    samp_entropy:  float = float("nan")
    entropy_label: str   = "Unknown"  # Ordered | Moderate | Chaotic
    entropy_conf:  float = 1.0      # confidence multiplier [0.7, 1.35]
    entropy_note:  str   = ""

    # ── Meta ──────────────────────────────────────────────────────────────────
    data_points: int            = 0
    error:       Optional[str]  = None


# ═══════════════════════════════════════════════════════════════════════════════
# MATH FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _hurst_rs(returns: np.ndarray) -> float:
    """
    Classic Hurst exponent via Rescaled Range (R/S) analysis.

    Computes R/S at multiple overlapping window sizes, then fits log(R/S) ~ H·log(n)
    via OLS. Overlapping windows reduce variance and improve small-sample stability.

    Returns H ∈ (0,1):  H>0.58 = trending, H<0.42 = mean-reverting, else random.
    """
    n = len(returns)
    if n < 20:
        return 0.5

    min_w = max(8, n // 10)
    max_w = n // 2
    if max_w <= min_w:
        return 0.5

    n_scales = min(14, max_w - min_w + 1)
    scales   = np.unique(
        np.logspace(np.log10(min_w), np.log10(max_w), n_scales).astype(int)
    )

    rs_means, valid_scales = [], []

    for w in scales:
        rs_list = []
        step = max(1, w // 3)
        for start in range(0, n - w + 1, step):
            sub  = returns[start:start + w]
            mean = sub.mean()
            dev  = np.cumsum(sub - mean)
            R    = dev.max() - dev.min()
            S    = sub.std(ddof=1)
            if S > 1e-10:
                rs_list.append(R / S)
        if len(rs_list) >= 2:
            rs_means.append(float(np.mean(rs_list)))
            valid_scales.append(w)

    if len(rs_means) < 4:
        return 0.5

    log_s = np.log(valid_scales)
    log_r = np.log(rs_means)

    # Drop outliers (> 2σ in log_r) before fitting
    mask = np.abs(log_r - log_r.mean()) < 2.0 * (log_r.std() + 1e-10)
    if mask.sum() < 4:
        mask = np.ones(len(log_r), dtype=bool)

    H = float(np.polyfit(log_s[mask], log_r[mask], 1)[0])
    return float(np.clip(H, 0.01, 0.99))


def _dfa(returns: np.ndarray, order: int = 1) -> float:
    """
    Detrended Fluctuation Analysis (DFA-1 with linear detrending).

    Integrates returns, splits into segments, fits a polynomial trend in each,
    then measures the RMS of residuals vs segment length. Slope of log-log fit
    gives the DFA exponent α (analogous to Hurst H for stationary processes).
    More robust than R/S when the series has non-stationary drift.
    """
    n = len(returns)
    if n < 20:
        return 0.5

    y = np.cumsum(returns - returns.mean())   # profile = integrated mean-centred series

    min_w = max(order + 3, 6)
    max_w = n // 4
    if max_w <= min_w:
        return 0.5

    scales  = np.unique(
        np.logspace(np.log10(min_w), np.log10(max_w), 12).astype(int)
    )

    f_list, s_list = [], []
    x_base = np.arange(max(scales) + 1, dtype=float)

    for w in scales:
        n_seg = n // w
        if n_seg < 2:
            continue
        x = x_base[:w]
        rms_sum = 0.0
        for s in range(n_seg):
            seg    = y[s * w:(s + 1) * w]
            coeffs = np.polyfit(x, seg, order)
            trend  = np.polyval(coeffs, x)
            rms_sum += float(np.mean((seg - trend) ** 2))
        rms = math.sqrt(rms_sum / n_seg)
        if rms > 1e-12:
            f_list.append(rms)
            s_list.append(w)

    if len(f_list) < 4:
        return 0.5

    alpha = float(np.polyfit(np.log(s_list), np.log(f_list), 1)[0])
    return float(np.clip(alpha, 0.01, 0.99))


def _permutation_entropy(series: np.ndarray, order: int = 4, delay: int = 1) -> float:
    """
    Permutation Entropy (Bandt & Pompe 2002).

    Encodes each length-`order` window as an ordinal permutation pattern.
    Computes the Shannon entropy of the pattern frequency distribution.
    Normalised to [0,1] by dividing by log₂(order!).

    Fast (O(n·order!)), robust to noise and amplitude, ideal for financial series.
    """
    n = len(series)
    if n < order * delay + 2:
        return 1.0

    counts: dict = {}
    total = 0
    for i in range(n - (order - 1) * delay):
        vec = np.array([series[i + j * delay] for j in range(order)])
        # Rank order = ordinal pattern
        pattern = tuple(np.argsort(np.argsort(vec)))
        counts[pattern] = counts.get(pattern, 0) + 1
        total += 1

    if total == 0:
        return 1.0

    probs = np.array(list(counts.values()), dtype=float) / total
    probs = probs[probs > 0]

    H     = float(-np.sum(probs * np.log2(probs)))
    H_max = math.log2(math.factorial(order)) if order <= 12 else math.log2(total)

    return float(np.clip(H / H_max if H_max > 0 else 1.0, 0.0, 1.0))


def _sample_entropy(series: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """
    Sample Entropy (Richman & Moorman 2000).

    Measures regularity without template self-matching bias.
    SampEn → 0: highly regular.  SampEn → ∞: maximally irregular.
    Capped to n=100 for computational tractability (O(n²)).
    """
    series = series.astype(float)
    n = min(len(series), 100)
    if n < 10:
        return float("nan")
    series = series[-n:]

    r = r_factor * series.std()
    if r < 1e-12:
        return 0.0

    def _count(length: int) -> int:
        cnt = 0
        for i in range(n - length):
            tmpl = series[i:i + length]
            for j in range(i + 1, n - length + 1):
                if np.max(np.abs(series[j:j + length] - tmpl)) <= r:
                    cnt += 1
        return cnt

    B = _count(m)
    A = _count(m + 1)

    if B == 0:
        return float("nan")
    if A == 0:
        return float("inf")
    return float(-math.log(A / B))


# ═══════════════════════════════════════════════════════════════════════════════
# GAUSSIAN HMM (Baum-Welch with scaled forward-backward)
# ═══════════════════════════════════════════════════════════════════════════════

class _GaussianHMM:
    """
    3-state diagonal-covariance multivariate Gaussian HMM.

    Accepts X of shape (T, D) — D features per time step.
    We use D=2: [daily_return, rolling_5D_vol].

    Why 2 features:
      • Returns alone collapse to "Sideways" in range-bound markets because
        all three state means converge near zero.
      • Adding rolling volatility gives the model a second axis:
          Bull     → positive return + NORMAL vol
          Sideways → near-zero return + LOW vol
          Bear     → negative return + HIGH vol
        This separates regimes even when directional returns are flat.

    Emission: p(x_t | state i) = prod_d N(x_d | μ_id, σ_id²)  [diagonal covariance]

    Algorithm: Baum-Welch (E-M) with scaled forward-backward.
    Decoding:  forward-only at last step (beta[-1] = 1 identically, so
               forward-backward = forward-only for the LAST timestep — no future data).

    Hyperparameters:
        n_states = 3  (Bull / Sideways / Bear — labeled by mean RETURN, not mean vol)
        n_iter   = 100 EM iterations
        tol      = 1e-5 log-likelihood convergence
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-5):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol

    def _emit_one(self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """Log-space diagonal Gaussian for one observation vector."""
        log_p = 0.0
        for d in range(len(mu)):
            s = max(float(sigma[d]), 1e-8)
            z = (float(x[d]) - float(mu[d])) / s
            log_p += -0.5 * z * z - math.log(s * 2.5066282746310002)
        return math.exp(max(log_p, -700))

    def _B(self, X: np.ndarray) -> np.ndarray:
        """Emission matrix: shape (T, n_states)."""
        T, n = len(X), self.n_states
        B = np.array([[self._emit_one(X[t], self.mu[i], self.sigma[i])
                       for i in range(n)] for t in range(T)])
        return np.maximum(B, 1e-300)

    def _forward(self, B: np.ndarray):
        T, n = B.shape
        alpha = np.empty((T, n))
        scale = np.empty(T)
        alpha[0] = self.pi * B[0]
        scale[0] = alpha[0].sum()
        alpha[0] /= max(scale[0], 1e-300)
        for t in range(1, T):
            alpha[t] = alpha[t-1] @ self.A * B[t]
            scale[t] = alpha[t].sum()
            alpha[t] /= max(scale[t], 1e-300)
        return alpha, scale

    def _backward(self, B: np.ndarray, scale: np.ndarray) -> np.ndarray:
        T, n = B.shape
        beta = np.ones((T, n))
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A * B[t+1] * beta[t+1]).sum(axis=1)
            beta[t] /= max(scale[t+1], 1e-300)
        return beta

    def fit(self, X: np.ndarray) -> "_GaussianHMM":
        X = np.asarray(X, dtype=float)   # shape (T, D)
        if X.ndim == 1:
            X = X[:, None]
        T, D  = X.shape
        n     = self.n_states

        # Percentile-based initialization on first feature (return)
        ret   = X[:, 0]
        s_ret = np.sort(ret)
        self.pi = np.full(n, 1.0 / n)
        # Moderate self-transition — allows regime switching in range-bound markets
        self.A  = np.full((n, n), 0.10 / max(n - 1, 1))
        np.fill_diagonal(self.A, 0.80)
        # Mu: per-state, per-feature
        self.mu = np.array([
            [float(s_ret[max(0, int((i + 0.5) * T / n))])] +
            [float(X[:, d].mean()) for d in range(1, D)]
            for i in range(n)
        ])   # shape (n, D)
        self.sigma = np.tile(
            np.maximum(X.std(axis=0) * 0.5, 1e-4),
            (n, 1)
        )   # shape (n, D)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            Bm           = self._B(X)
            alpha, scale = self._forward(Bm)
            beta         = self._backward(Bm, scale)

            gamma  = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

            xi = np.zeros((T - 1, n, n))
            for t in range(T - 1):
                xi[t] = (alpha[t:t+1].T * self.A * Bm[t+1] * beta[t+1])
                xi[t] /= max(xi[t].sum(), 1e-300)

            self.pi = gamma[0]
            xi_sum  = xi.sum(axis=0)
            self.A  = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-300)
            for i in range(n):
                g  = gamma[:, i]
                gs = max(g.sum(), 1e-10)
                for d in range(D):
                    self.mu[i, d]    = float(np.dot(g, X[:, d]) / gs)
                    self.sigma[i, d] = max(
                        math.sqrt(float(np.dot(g, (X[:, d] - self.mu[i, d])**2) / gs)),
                        1e-6,
                    )

            ll = float(np.log(np.maximum(scale, 1e-300)).sum())
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        return self

    def label_states(self) -> list[str]:
        """Label by MEAN RETURN (feature 0) — vol feature is discriminating not labeling."""
        order  = np.argsort(self.mu[:, 0])   # sort by mean return only
        labels = [""] * self.n_states
        for rank, idx in enumerate(order):
            labels[idx] = ["Bear", "Sideways", "Bull"][min(rank, 2)]
        return labels

    def state_probs_now(self, X: np.ndarray) -> np.ndarray:
        """
        Causal state distribution at the last time step.

        Note: beta[-1] = np.ones(n) identically (never modified in backward loop),
        so alpha[-1] * beta[-1] = alpha[-1]. The backward pass adds no future
        information for the final time step — this is purely forward/causal.
        """
        if X.ndim == 1:
            X = X[:, None]
        Bm    = self._B(X)
        alpha, _ = self._forward(Bm)
        p     = alpha[-1].copy()
        p    /= max(p.sum(), 1e-300)
        return p


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def get_regime_signals(
    fno_symbol: str,
    index_name: str,
    trade_date: date,
    lookback: int = 90,
) -> RegimeResult:
    """
    Fetch historical price returns for `fno_symbol` and compute:
      1. Hurst R/S + DFA  — market memory / autocorrelation
      2. 3-state HMM      — hidden Bull / Sideways / Bear regime
      3. Permutation Entropy + Sample Entropy — complexity / predictability

    `index_name` is the index_data.index_name label (e.g. "Nifty 50").
    `lookback` is the number of trading days to use (default 90 ≈ 4.5 months).
    """
    result = RegimeResult()
    try:
        from_date = trade_date - timedelta(days=lookback * 2)   # buffer for weekends

        df = query_dataframe("""
            SELECT trade_date, pct_chg
            FROM index_data
            WHERE index_name = ? AND trade_date BETWEEN ? AND ?
              AND pct_chg IS NOT NULL
            ORDER BY trade_date
        """, [index_name, from_date, trade_date])

        if df.empty or len(df) < 20:
            result.error = f"Insufficient history ({len(df)} rows) for {index_name}"
            return result

        returns          = df["pct_chg"].tail(lookback).dropna().values.astype(float)
        n                = len(returns)
        result.data_points = n

        if n < 20:
            result.error = "Less than 20 usable return observations"
            return result

        # ── 1. Hurst R/S + DFA ───────────────────────────────────────────────
        h_rs  = _hurst_rs(returns)
        h_dfa = _dfa(returns, order=1)
        h_avg = (h_rs + h_dfa) / 2.0

        result.hurst_rs   = round(h_rs,  4)
        result.dfa_alpha  = round(h_dfa, 4)
        result.memory_avg = round(h_avg, 4)

        if h_avg > 0.58:
            result.memory_label = "Trending"
        elif h_avg < 0.42:
            result.memory_label = "Mean-Reverting"
        else:
            result.memory_label = "Random Walk"

        # Direction: use sign of recent 5D cumulative return
        recent_5d  = float(returns[-5:].sum())
        trend_up   = recent_5d > 0

        if result.memory_label == "Trending":
            raw_score = 2.0 if h_avg > 0.65 else 1.0
            result.memory_score = raw_score if trend_up else -raw_score
            result.memory_note  = (
                f"H(R/S)={h_rs:.3f}  DFA-α={h_dfa:.3f}  Avg={h_avg:.3f}  "
                f"→ Persistent market — returns autocorrelated. "
                f"Recent 5D: {'+' if trend_up else ''}{recent_5d:.2f}%. "
                f"Persistence suggests continuation of {'upside' if trend_up else 'downside'}."
            )
        elif result.memory_label == "Mean-Reverting":
            raw_score = 1.5 if h_avg < 0.38 else 0.8
            # Anti-persistent: FADE the recent move
            result.memory_score = -raw_score if trend_up else raw_score
            result.memory_note  = (
                f"H(R/S)={h_rs:.3f}  DFA-α={h_dfa:.3f}  Avg={h_avg:.3f}  "
                f"→ Anti-persistent / mean-reverting series. Returns self-correct. "
                f"Recent 5D: {'+' if trend_up else ''}{recent_5d:.2f}% — "
                f"elevated probability of {'pullback' if trend_up else 'bounce'}."
            )
        else:
            result.memory_score = 0.0
            result.memory_note  = (
                f"H(R/S)={h_rs:.3f}  DFA-α={h_dfa:.3f}  Avg={h_avg:.3f}  "
                f"→ Near-random walk. No memory-based directional edge."
            )

        # ── 2. Hidden Markov Model — 2-feature (return, rolling vol) ────────────
        hmm_ret = returns[-min(n, 90):]
        T_hmm   = len(hmm_ret)

        if T_hmm >= 30:
            try:
                # Rolling 5D volatility — key discriminator for Bear vs Sideways.
                # Bear = negative return + HIGH vol; Sideways = near-zero + LOW vol.
                # Use min_periods=2 so early rows are valid.
                ret_series = pd.Series(hmm_ret)
                rolling_vol = ret_series.rolling(5, min_periods=2).std().fillna(
                    ret_series.std()
                ).values

                # Standardise both features to unit-variance before HMM fitting.
                # Prevents one feature dominating the diagonal covariance.
                ret_std = max(hmm_ret.std(), 1e-6)
                vol_std = max(rolling_vol.std(), 1e-6)
                hmm_X = np.column_stack([
                    hmm_ret    / ret_std,    # normalised return
                    rolling_vol / vol_std,   # normalised volatility
                ])

                hmm    = _GaussianHMM(n_states=3, n_iter=100, tol=1e-5)
                hmm.fit(hmm_X)

                labels = hmm.label_states()
                probs  = hmm.state_probs_now(hmm_X)

                bull_i = labels.index("Bull")
                side_i = labels.index("Sideways")
                bear_i = labels.index("Bear")

                p_bull = float(probs[bull_i])
                p_side = float(probs[side_i])
                p_bear = float(probs[bear_i])

                cur_state         = int(np.argmax(probs))
                result.hmm_state  = labels[cur_state]
                result.hmm_prob   = round(float(probs[cur_state]), 3)
                result.hmm_probs  = [round(p_bull, 3), round(p_side, 3), round(p_bear, 3)]
                result.hmm_score  = round(p_bull * 3.0 - p_bear * 3.0, 2)

                # Report means in original (un-normalised) units
                result.hmm_note = (
                    f"3-state HMM on {T_hmm}D data [return, rolling_vol]. "
                    f"Mean return: Bull={hmm.mu[bull_i,0]*ret_std:+.3f}%  "
                    f"Sideways={hmm.mu[side_i,0]*ret_std:+.3f}%  "
                    f"Bear={hmm.mu[bear_i,0]*ret_std:+.3f}%. "
                    f"Mean vol: Bull={hmm.mu[bull_i,1]*vol_std:.3f}  "
                    f"Sideways={hmm.mu[side_i,1]*vol_std:.3f}  "
                    f"Bear={hmm.mu[bear_i,1]*vol_std:.3f}. "
                    f"Current: {result.hmm_state} (p={result.hmm_prob:.1%}) | "
                    f"Bull={p_bull:.1%}  Sideways={p_side:.1%}  Bear={p_bear:.1%}."
                )
            except Exception as hmm_exc:
                result.hmm_note = f"HMM error: {hmm_exc}"
        else:
            result.hmm_note = f"HMM skipped — only {T_hmm} obs (min 30 required)"

        # ── 3. Permutation Entropy + Sample Entropy ───────────────────────────
        pe_data = returns[-min(n, 60):]

        result.perm_entropy = round(_permutation_entropy(pe_data, order=4, delay=1), 4)

        se = _sample_entropy(pe_data, m=2, r_factor=0.2)
        result.samp_entropy = round(se, 4) if math.isfinite(se) else float("nan")

        pe = result.perm_entropy
        if pe < 0.50:
            result.entropy_label = "Ordered"
            result.entropy_conf  = 1.30
            result.entropy_note  = (
                f"PE={pe:.4f} — highly ordered return structure ({len(pe_data)}D). "
                f"Market is exhibiting clear, non-random patterns. "
                f"Directional signals have higher reliability. Confidence ×{result.entropy_conf:.2f}."
            )
        elif pe < 0.72:
            result.entropy_label = "Moderate"
            result.entropy_conf  = 1.00
            result.entropy_note  = (
                f"PE={pe:.4f} — moderate complexity ({len(pe_data)}D). "
                f"Normal market structure. No confidence adjustment."
            )
        else:
            result.entropy_label = "Chaotic"
            result.entropy_conf  = 0.72
            result.entropy_note  = (
                f"PE={pe:.4f} — high entropy / disordered return structure ({len(pe_data)}D). "
                f"Market is in a chaotic, low-predictability state. "
                f"Reduce position sizing; tighten stops. Confidence ×{result.entropy_conf:.2f}."
            )

    except Exception as exc:
        result.error = str(exc)

    return result
