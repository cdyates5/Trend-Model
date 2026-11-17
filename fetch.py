"""Stage 1: point-in-time membership, constituent prices, index and cash rate.

Two modes. A full download pulls every ticker that has ever been in the index over the
sample (about 1,200 names, eight minutes). A top-up refreshes only the current members
and splices the new bars onto the stored panel, which is what the scheduled job uses.
A top-up re-downloads any ticker whose overlap with the stored panel has moved, so a
split or a changed adjustment basis cannot leave the panel silently wrong.
"""
import io, os, time, pickle, datetime as dt
import numpy as np, pandas as pd, requests

CACHE = os.environ.get("MOM_CACHE", "cache")
START = "1996-01-01"
UA = {"User-Agent": "Mozilla/5.0 (research; momentum-trend-aggregator)"}
MEMBERS_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
               "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
KEYS = (("Close", "close"), ("Adj Close", "adj"))      # the model needs no highs or lows
SPLIT_TOL = 0.005                                       # overlap drift that forces a full refetch


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def get_membership():
    """Point-in-time membership: fja05680 history (1996->) rolled forward with Wikipedia."""
    txt = requests.get(MEMBERS_URL, headers=UA, timeout=60).text
    hist = pd.read_csv(io.StringIO(txt), parse_dates=["date"])
    hist["set"] = hist["tickers"].str.split(",").apply(lambda x: frozenset(t.strip() for t in x if t.strip()))
    hist = hist[["date", "set"]].sort_values("date").reset_index(drop=True)
    last_date, cur = hist["date"].iloc[-1], set(hist["set"].iloc[-1])
    notes = {"hist_last": str(last_date.date()), "wiki_applied": 0, "wiki_ok": False}
    try:
        html = requests.get(WIKI_URL, headers=UA, timeout=60).text
        tabs = pd.read_html(io.StringIO(html))
        cur_tab = next(t for t in tabs if "Symbol" in [str(c) for c in t.columns])
        current = set(cur_tab["Symbol"].astype(str).str.strip())
        dadd = pd.to_datetime(cur_tab.get("Date added"), errors="coerce")
        adds = sorted([(d, s) for d, s in zip(dadd, cur_tab["Symbol"].astype(str).str.strip())
                       if pd.notna(d) and d > last_date], key=lambda x: x[0])
        drops = sorted(cur - current)
        rows = []
        for k, (d, sym) in enumerate(adds):
            cur.add(sym)
            if k < len(drops):                 # index changes are paired: retire one per addition
                cur.discard(drops[k])
            rows.append({"date": d, "set": frozenset(cur)})
            notes["wiki_applied"] += 1
        today = pd.Timestamp(dt.datetime.now(dt.timezone.utc).date())
        rows.append({"date": today, "set": frozenset(current)})
        notes["gap_adds"] = len(adds); notes["gap_drops"] = len(drops)
        notes["recon_missing"] = len(current - cur); notes["recon_extra"] = len(cur - current)
        hist = pd.concat([hist, pd.DataFrame(rows)], ignore_index=True)
        notes["wiki_ok"] = True
    except Exception as e:  # the history file alone is still usable
        notes["wiki_err"] = repr(e)[:120]
    return hist, notes


def ysym(t):
    return t.replace(".", "-")


def _flat(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _dl(syms, tries=3, **kw):
    import yfinance as yf
    for a in range(tries):
        try:
            df = yf.download(syms, auto_adjust=False, progress=False, threads=True, timeout=40,
                             group_by="column", **kw)
            if df is not None and not df.empty:
                return df
        except Exception as e:
            log("retry", repr(e)[:70])
        time.sleep(4 * (a + 1))
    return None


def _index_series(out):
    for sym, key, st in (("^GSPC", "spx", "1927-01-01"), ("^IRX", "irx", "1960-01-01")):
        d = _dl(sym, start=st)
        if d is not None and len(d) > 100:
            out[key] = _flat(d)
        elif key not in out:
            raise SystemExit("could not download " + sym)
    return out


def get_prices(tickers, batch=60):
    """Full download of every ticker, used for the first build and the weekly rebuild."""
    os.makedirs(CACHE, exist_ok=True)
    tickers = sorted(tickers)
    F = {k: {} for _, k in KEYS}
    for i in range(0, len(tickers), batch):
        b = tickers[i:i + batch]
        df = _dl([ysym(t) for t in b], start=START)
        if df is None:
            continue
        for t in b:
            y = ysym(t)
            try:
                if df["Close"][y].dropna().shape[0] < 30:
                    continue
                for src, key in KEYS:
                    F[key][t] = df[src][y].dropna().astype("float32")
            except Exception:
                pass
        log(f"prices {min(i + batch, len(tickers))}/{len(tickers)} ok={len(F['close'])}")
        time.sleep(0.5)
    out = {k: pd.DataFrame(F[k]) for _, k in KEYS}
    return _index_series(out)


def top_up(px, tickers, days=400, batch=60):
    """Refresh recent bars for `tickers` and splice them onto the stored panel."""
    cols = {k: {c: px[k][c] for c in px[k].columns} for _, k in KEYS}
    have = set(px["close"].columns)
    fresh = sorted(t for t in tickers if t in have)
    new = sorted(t for t in tickers if t not in have)
    suspect, done = [], 0
    for i in range(0, len(fresh), batch):
        b = fresh[i:i + batch]
        df = _dl([ysym(t) for t in b], period=f"{days}d")
        if df is None:
            log("batch failed, keeping stored history for", len(b), "tickers"); continue
        for t in b:
            y = ysym(t)
            try:
                s = df["Close"][y].dropna()
            except Exception:
                continue
            if s.empty:
                continue
            old = cols["close"][t]
            ov = old.index.intersection(s.index)
            if len(ov) >= 20:
                r = float(np.nanmedian(s.reindex(ov).values / old.reindex(ov).values))
                if not np.isfinite(r) or abs(r - 1) > SPLIT_TOL:
                    suspect.append(t)              # split or re-based adjustment: refetch in full
                    continue
            for src, key in KEYS:
                try:
                    v = df[src][y].dropna().astype("float32")
                except Exception:
                    continue
                if not v.empty:
                    cols[key][t] = v.combine_first(cols[key][t]).astype("float32")
            done += 1
        time.sleep(0.4)
    refetch = sorted(set(suspect) | set(new))
    if refetch:
        log("full refetch for", len(refetch), "tickers", "(" + ", ".join(refetch[:8]) + ("..." if len(refetch) > 8 else "") + ")")
        for i in range(0, len(refetch), batch):
            b = refetch[i:i + batch]
            df = _dl([ysym(t) for t in b], start=START)
            if df is None:
                continue
            for t in b:
                y = ysym(t)
                try:
                    if df["Close"][y].dropna().shape[0] < 30:
                        continue
                    for src, key in KEYS:
                        cols[key][t] = df[src][y].dropna().astype("float32")
                except Exception:
                    pass
            time.sleep(0.4)
    out = {k: pd.DataFrame(cols[k]) for _, k in KEYS}
    log(f"top-up: {done} spliced, {len(suspect)} re-based, {len(new)} new, panel {out['adj'].shape}")
    return _index_series(out), {"spliced": done, "rebased": len(suspect), "new": len(new)}


if __name__ == "__main__":
    hist, notes = get_membership()
    log("membership rows", len(hist), notes)
    universe = set().union(*hist[hist["date"] >= "1995-12-01"]["set"])
    log("universe", len(universe))
    os.makedirs(CACHE, exist_ok=True)
    pickle.dump((hist, notes), open(os.path.join(CACHE, "members.pkl"), "wb"))
    px = get_prices(universe)
    pickle.dump(px, open(os.path.join(CACHE, "prices.pkl"), "wb"))
    log("done", px["adj"].shape,
        {k: (str(px[k].index[0].date()), str(px[k].index[-1].date())) for k in ("spx", "irx") if k in px})
