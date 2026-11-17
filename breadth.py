"""Stage 2: participation and internal-momentum series from point-in-time S&P 500 constituents."""
import numpy as np, pandas as pd


def build_breadth(px, hist, start="1997-01-01", price="close"):
    """price: 'close' (split-adjusted, chart convention) or 'adj' (split+dividend adjusted)."""
    spx = px["spx"]
    idx = spx.index[spx.index >= start]
    P = px[price].reindex(idx).astype("float64")
    P = P.where(P > 0)
    ret_src = px["adj"].reindex(idx).astype("float64")       # advance/decline uses total-return closes
    ret_src = ret_src.where(ret_src > 0)
    cols = P.columns

    h = hist.sort_values("date")
    mrows = np.zeros((len(h), len(cols)), dtype=bool)
    cpos = {c: i for i, c in enumerate(cols)}
    for r, s in enumerate(h["set"]):
        for t in s:
            j = cpos.get(t)
            if j is not None:
                mrows[r, j] = True
    pos = np.clip(np.searchsorted(h["date"].values, idx.values, side="right") - 1, 0, len(h) - 1)
    member = pd.DataFrame(mrows[pos], index=idx, columns=cols)
    n_index = pd.Series(np.array([len(s) for s in h["set"]])[pos], index=idx)

    ret = ret_src.pct_change(fill_method=None)
    ret = ret.where(ret.abs() < 0.9)
    valid = member & ret.notna()
    adv = (valid & (ret > 0)).sum(1); dec = (valid & (ret < 0)).sum(1); n = valid.sum(1)

    out = pd.DataFrame(index=idx)
    out["spx"] = spx["Close"].reindex(idx)
    out["n"] = n; out["coverage"] = n / n_index
    out["adv"] = adv; out["dec"] = dec

    # --- McClellan oscillator / summation (ratio-adjusted, Miekka closed form) ---
    rana = 1000 * (adv - dec) / (adv + dec)
    e19 = rana.ewm(span=19, adjust=False).mean(); e39 = rana.ewm(span=39, adjust=False).mean()
    out["osc"] = e19 - e39
    out["msi"] = 19 * e39 - 9 * e19

    # --- 200-day average: share above, share rising ---
    ma = P.rolling(200, min_periods=200).mean()
    ok = member & ma.notna() & P.notna()
    out["p200"] = 100 * (ok & (P > ma)).sum(1) / ok.sum(1)
    rise1 = ma.diff() > 0                                     # literal: today's MA above yesterday's
    ok1 = member & ma.notna() & ma.shift(1).notna()
    out["r200"] = 100 * (ok1 & rise1).sum(1) / ok1.sum(1)
    rise21 = ma.diff(21) > 0                                  # slower variant, for reference
    ok21 = member & ma.notna() & ma.shift(21).notna()
    out["r200_21"] = 100 * (ok21 & rise21).sum(1) / ok21.sum(1)

    # --- 52-week high/low: new highs, new lows, share in a correction ---
    hi252 = P.rolling(252, min_periods=252).max()
    lo252 = P.rolling(252, min_periods=252).min()
    okh = member & hi252.notna() & P.notna()
    den = okh.sum(1)
    out["nh"] = (okh & (P >= hi252 - 1e-9)).sum(1)
    out["nl"] = (okh & (P <= lo252 + 1e-9)).sum(1)
    out["nhnl"] = 100 * (out["nh"] - out["nl"]) / den          # net new highs, % of eligible members
    out["corr"] = 100 * (okh & (P < 0.90 * hi252)).sum(1) / den   # more than 10% below own 52w high
    out["bear"] = 100 * (okh & (P < 0.80 * hi252)).sum(1) / den   # more than 20% below, for reference
    out["den"] = den
    return out.dropna(subset=["spx"]), member


if __name__ == "__main__":
    import pickle, sys, time
    px = pickle.load(open("cache/prices.pkl", "rb"))
    hist, notes = pickle.load(open("cache/members.pkl", "rb"))
    pd.set_option("display.width", 250)
    t0 = time.time()
    out = {}
    for src in ("close", "adj"):
        b, member = build_breadth(px, hist, price=src)
        out[src] = b
        print(src, "secs", round(time.time() - t0), b.shape)
    # validation anchor from the SentimenTrader chart: SPX 5909.03, %>200DMA 51.49, % rising 200DMA 56.86
    sp = out["close"]["spx"]
    anchor = (sp - 5909.03).abs().idxmin()
    print("\nanchor date", anchor.date(), "spx", round(float(sp.loc[anchor]), 2))
    for src in ("close", "adj"):
        b = out[src]
        print(f"  {src:5s} p200 {b.loc[anchor,'p200']:.2f}  r200 {b.loc[anchor,'r200']:.2f}  r200_21 {b.loc[anchor,'r200_21']:.2f}  corr {b.loc[anchor,'corr']:.1f}  nhnl {b.loc[anchor,'nhnl']:.2f}")
    b = out[sys.argv[1] if len(sys.argv) > 1 else "close"]
    b.to_pickle("cache/breadth.pkl")
    print("\n", b.tail(4).round(2).to_string())
    print("\ncoverage by year\n", b["coverage"].resample("YE").mean().round(3).iloc[::3].to_string())
    for d in ["2018-12-24", "2020-03-23", "2022-06-16", "2025-04-08"]:
        if d in b.index.astype(str).values:
            print(d, b.loc[d, ["p200", "r200", "corr", "nhnl", "msi"]].round(1).to_dict())
    print(b[["p200", "r200", "corr", "nhnl", "msi"]].describe(percentiles=[.05, .5, .95]).round(1).to_string())
