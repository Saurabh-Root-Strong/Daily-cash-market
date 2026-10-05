"""Check the Index & Large Cap buckets against NSE's live lists. Read-only.

    python scripts/refresh_index_buckets.py [--on 2026-10-06]

WHY THIS EXISTS. The buckets in src/analytics/index_largecap.py are frozen
membership, and NSE reconstitutes every March and September (plus ad-hoc
replacements for demergers and suspensions). A frozen list goes wrong quietly:
the panel keeps rendering, just for a basket the index no longer holds. This
script is the tripwire. Exit code 0 = coded baskets match NSE; 1 = drift found,
with the suggested tuples printed.

WHAT IT CHECKS, per index, against the basket in force on --on (default today):
  1 MEMBERSHIP   the niftyindices constituent CSV vs every coded bucket
  2 NIFTY RANKS  Top 10 / Next 10 vs NSE's own Nifty Top 10 / Top 20 Equal
                 Weight membership (6-month average free-float cap selection)
  3 CUTS         for Bank / Fin / Midcap Select, the factsheet's top-10 weights:
                 are the coded cut points still on the widest weight gaps? A
                 "gap" under 0.3pp is reported as NO NATURAL CUT (the boundary is
                 a layout choice, not structure) and is not counted as drift.
  4 FACTSHEET    its as-of date, so a PROVISIONAL basket is flagged the moment
                 NSE publishes weights for it

It never edits the module: a membership change needs a dated version in
_VERSIONS with its source, which is a human decision.
"""
import argparse, csv, io, re, sys, os
from datetime import date, datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analytics.index_largecap import INDEX_BUCKETS, index_meta   # noqa: E402

BASE = "https://www.niftyindices.com"
SRC = {
    "NIFTY": ("ind_nifty50list.csv", "ind_nifty50.pdf"),
    "BANKNIFTY": ("ind_niftybanklist.csv", "ind_nifty_bank.pdf"),
    "FINNIFTY": ("ind_niftyfinancelist.csv", "ind_nifty_financial_services.pdf"),
    "MIDCPNIFTY": ("ind_niftymidcapselect_list.csv", "Factsheet_NiftyMidcapSelect.pdf"),
}
NO_CUT = 0.3        # pp: a gap narrower than this is not a structural boundary
CLEAR_CUT = 1.0     # pp: both gaps this wide = the cut POSITIONS are structure

_S = requests.Session()
_S.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 Chrome/128 Safari/537.36")


def _csv(name):
    r = _S.get(f"{BASE}/IndexConstituent/{name}", timeout=30)
    r.raise_for_status()
    if not r.text.startswith("Company"):
        raise RuntimeError(f"{name}: not a constituent CSV")
    return list(csv.DictReader(io.StringIO(r.text)))


def _factsheet(name):
    """(as-of date, [(company name, weight)]) from a factsheet's top-10 table."""
    import pypdf
    r = _S.get(f"{BASE}/Factsheet/{name}", timeout=30)
    if not (r.ok and r.content[:4] == b"%PDF"):
        return None, []
    t = "".join(p.extract_text() for p in pypdf.PdfReader(io.BytesIO(r.content)).pages)
    m = re.search(r"(January|February|March|April|May|June|July|August|September|"
                  r"October|November|December)\s+(\d{1,2}),\s+(20\d\d)", t)
    asof = datetime.strptime(" ".join(m.groups()), "%B %d %Y").date() if m else None
    i = t.find("Top constituents by weightage")
    rows = []
    for line in t[i:].splitlines()[1:12]:
        mm = re.match(r"(.+?)\s+(\d+\.\d+)\s*$", line.strip())
        if mm:
            rows.append((mm.group(1).strip(), float(mm.group(2))))
    return asof, rows[:10]


def _gap_cuts(w):
    """Two cut points on the widest gaps, buckets >= 2 names, cuts inside top 10."""
    gaps = {k: w[k - 1] - w[k] for k in range(1, len(w))}      # gap AFTER rank k
    best = None
    for k in range(2, len(w)):
        for j in range(k + 2, len(w) + 1):
            if j >= len(w):
                continue
            score = min(gaps[k], gaps[j])
            if best is None or score > best[0]:
                best = (score, k, j)
    return gaps, best


def check(sym, on):
    meta = index_meta(sym, on)
    coded = meta["buckets"]
    coded_all = {s for v in coded.values() for s in v}
    drift = []
    print(f"\n=== {meta['display']}  (basket in force {on}: {meta['source']}"
          f"{', PROVISIONAL' if meta['provisional'] else ''})")

    rows = _csv(SRC[sym][0])
    live = {r["Symbol"] for r in rows}
    name2sym = {r["Company Name"].strip().rstrip("."): r["Symbol"] for r in rows}
    add, drop = sorted(live - coded_all), sorted(coded_all - live)
    print(f"  membership: NSE {len(live)} vs coded {len(coded_all)}"
          + ("  OK" if not (add or drop) else f"  NSE adds {add}  NSE drops {drop}"))
    if add or drop:
        drift.append("membership")

    if sym == "NIFTY":
        t10 = {r["Symbol"] for r in _csv("ind_niftytop10equalweight_list.csv")}
        t20 = {r["Symbol"] for r in _csv("ind_niftytop20equalweight_list.csv")}
        top, nxt = set(coded["Top 10"]), set(coded["Next 10"])
        ok = top == t10 and nxt == (t20 - t10)
        print("  Top 10 / Next 10 vs NSE Top10/Top20 EW: " + ("OK" if ok else
              f"Top10 +{sorted(t10 - top)} -{sorted(top - t10)}  "
              f"Next10 +{sorted((t20 - t10) - nxt)} -{sorted(nxt - (t20 - t10))}"))
        if not ok:
            drift.append("nifty ranks")
            print(f"  suggested: Top 10 {tuple(sorted(t10))}\n"
                  f"             Next 10 {tuple(sorted(t20 - t10))}")

    asof, top10 = _factsheet(SRC[sym][1])
    print(f"  factsheet as of {asof}" + (
        f"  <- NEWER than the provisional basket's {meta['effective']}: CONFIRM THE CUTS"
        if meta["provisional"] and asof and meta["effective"] and asof >= meta["effective"]
        else ""))
    if meta["provisional"] and asof and meta["effective"] and asof >= meta["effective"]:
        drift.append("provisional basket now has published weights")
    if sym != "NIFTY" and top10:
        syms = [name2sym.get(n.rstrip("."), n) for n, _ in top10]
        w = [x for _, x in top10]
        gaps, best = _gap_cuts(w)
        print("  top 10: " + "  ".join(f"{s} {x:.2f}" for s, x in zip(syms, w)))
        labels = list(coded)
        k_coded = len(coded[labels[0]])
        j_coded = k_coded + len(coded[labels[1]])
        order_ok = (set(syms[:k_coded]) == set(coded[labels[0]])
                    and set(syms[k_coded:j_coded]) == set(coded[labels[1]]))
        if best and best[0] >= NO_CUT:
            _, k, j = best
            same = (k, j) == (k_coded, j_coded)
            print(f"  widest-gap cuts: after rank {k} ({gaps[k]:.2f}pp) and {j} "
                  f"({gaps[j]:.2f}pp) -> Top {k} / Next {j - k} / Rest {len(live) - j}"
                  + ("  = coded" if same else f"  != coded {k_coded}/{j_coded - k_coded}"))
            # Coded buckets holding the wrong NAMES is always drift. Different
            # cut POSITIONS are drift only when both gaps are unmistakable --
            # Midcap Select's second "gap" is 0.31pp, a layout choice.
            clear = min(gaps[k], gaps[j]) >= CLEAR_CUT
            if not order_ok or (not same and clear):
                drift.append("cuts")
                print(f"  suggested: Top {k} {tuple(syms[:k])}\n"
                      f"             Next {j - k} {tuple(syms[k:j])}")
            elif not same:
                print(f"  (advisory only: gaps under {CLEAR_CUT}pp, coded names "
                      f"are the correct top {k_coded} / next {j_coded - k_coded})")
        else:
            print(f"  NO NATURAL CUT: widest usable gap {best[0]:.2f}pp < {NO_CUT}pp "
                  f"-- the Top/Next boundary is a layout choice (not drift)")
            if not set(coded[labels[0]]) <= set(syms):
                drift.append("heavyweight bucket holds names outside the top 10")
    return drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", default=None, help="check the basket in force on this date")
    a = ap.parse_args()
    on = date.fromisoformat(a.on) if a.on else date.today()
    bad = {}
    for sym in INDEX_BUCKETS:
        try:
            d = check(sym, on)
        except Exception as exc:                              # noqa: BLE001
            d = [f"could not check: {exc}"]
            print(f"  ERROR {exc}")
        if d:
            bad[sym] = d
    print("\nRESULT:", "all baskets match NSE" if not bad else bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
