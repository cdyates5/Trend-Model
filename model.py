"""Stage 3b: the aggregator, the risk-adjusted evidence and the JSON payload.

Every statistic uses the condition as known at the close of day t paired with the return of day t+1.
Published regime tables are usually contemporaneous (condition and return on the same day), which is
circular for breadth measures; the two are not comparable and the notes say so.
"""
import json, datetime as dt
import numpy as np, pandas as pd
import signals as S

EVID_START = "2000-01-03"
HZ = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
SPLIT = "2013-01-01"
SMOOTH = 21                     # sessions averaged before a state is called ("consistently")
STRONG_AT, BROKEN_AT = 7, 3     # of 9 conditions
WIN = {"1962": "1962-01-02", "2000": "2000-01-03"}   # 1962: first full year of the T-bill series
SMA_PERIODS = list(range(10, 310, 5))
FILT_K = [4, 5, 6, 7, 8]
MSI_LAGS = [1, 5, 10, 21, 42]


def r1(x, nd=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def maxdd(curve):
    peak = np.maximum.accumulate(curve)
    return r1(float((curve / peak - 1).min() * 100))


class Book:
    """Return and risk statistics for daily masks, all on the next-day convention."""

    def __init__(self, c, cash):
        self.c = c; self.D = c.index; self.n = len(c)
        self.ret = c.pct_change().shift(-1).values                     # state on t earns t+1
        self.cash = (cash.reindex(c.index).ffill().values / 100.0 / 252.0)
        self.ok = ~np.isnan(self.ret) & ~np.isnan(self.cash)
        self.exc = np.where(self.ok, self.ret - self.cash, np.nan)
        self.fwd = {k: (c.shift(-h) / c - 1).values for k, h in HZ.items()}

    def stats(self, mask, den=None):
        den = np.ones(self.n, bool) if den is None else den
        m = mask & den
        mo = m & self.ok
        r = {"pct": r1(m.sum() / max(1, den.sum()) * 100), "days": int(m.sum())}
        if mo.sum() > 60:
            rr = self.ret[mo]; ee = self.exc[mo]
            r["ann"] = r1(((1 + rr).prod() ** (252 / len(rr)) - 1) * 100)
            r["vol"] = r1(rr.std(ddof=1) * np.sqrt(252) * 100)
            sh = float(ee.mean() / ee.std(ddof=1) * np.sqrt(252))
            r["sharpe"] = r1(sh, 2); r["shr"] = r1(sh, 6)
            r["dd"] = maxdd(np.cumprod(1 + rr))
        else:
            r.update(ann=None, vol=None, sharpe=None, shr=None, dd=None)
        for k in HZ:
            v = self.fwd[k][m]; v = v[~np.isnan(v)]
            r[k] = r1(v.mean() * 100) if len(v) else None
            r["w" + k] = r1((v > 0).mean() * 100, 0) if len(v) else None
        return r

    def placebo(self, mask, den):
        """p-value for the Sharpe of the masked days against every circular shift of the mask."""
        sl = slice(int(den.argmax()), int(self.n - den[::-1].argmax()))
        a = self.exc[sl]; m = (mask & den)[sl] & ~np.isnan(a)
        a = np.nan_to_num(a)
        if m.sum() < 60:
            return None
        k = m.sum(); obs = a[m].mean() / a[m].std(ddof=1) * np.sqrt(252)
        nn = len(a)
        F = np.conj(np.fft.rfft(m.astype(float)))
        s1 = np.fft.irfft(F * np.fft.rfft(a), nn)          # sum of a over each shifted mask
        s2 = np.fft.irfft(F * np.fft.rfft(a * a), nn)      # sum of a^2
        mu = s1 / k; var = np.maximum(s2 / k - mu ** 2, 1e-18) * k / (k - 1)
        sh = mu / np.sqrt(var) * np.sqrt(252)
        return round(float((sh >= obs - 1e-12).mean()), 3)

    def curve(self, mask):
        g = np.where(mask & self.ok, np.nan_to_num(self.ret), 0.0)
        return 10000 * np.cumprod(1 + g)

    def filt(self, long_mask, den):
        """Long while the mask is true, otherwise in cash. The curve spans every day of den."""
        m = den & self.ok
        r = np.where(long_mask[m], self.ret[m], self.cash[m]); e = r - self.cash[m]
        full = np.zeros(int(den.sum())); full[m[den]] = r
        return {"ann": r1(((1 + r).prod() ** (252 / len(r)) - 1) * 100), "vol": r1(r.std(ddof=1) * np.sqrt(252) * 100),
                "sharpe": r1(e.mean() / e.std(ddof=1) * np.sqrt(252), 2), "dd": maxdd(np.cumprod(1 + r)),
                "exp": r1(long_mask[m].mean() * 100, 0)}, np.cumprod(1 + full)


def sma_study(c, cash, periods):
    r = c.pct_change().values
    cd = cash.reindex(c.index).ffill().values / 100.0 / 252.0
    ok = ~np.isnan(r) & ~np.isnan(cd)
    out = []
    for n in periods:
        sig = (c > c.rolling(n, min_periods=n).mean()).shift(1)
        m = ok & sig.notna().values
        s = sig.values[m].astype(bool)
        rr = np.where(s, r[m], cd[m]); e = rr - cd[m]
        out.append({"n": n, "sharpe": r1(e.mean() / e.std(ddof=1) * np.sqrt(252), 3),
                    "ann": r1(((1 + rr).prod() ** (252 / m.sum()) - 1) * 100),
                    "vol": r1(rr.std(ddof=1) * np.sqrt(252) * 100),
                    "dd": maxdd(np.cumprod(1 + rr)), "exp": r1(s.mean() * 100, 0)})
    e = (r - cd)[ok]
    bh = {"n": None, "sharpe": r1(e.mean() / e.std(ddof=1) * np.sqrt(252), 3),
          "ann": r1(((1 + r[ok]).prod() ** (252 / ok.sum()) - 1) * 100),
          "vol": r1(r[ok].std(ddof=1) * np.sqrt(252) * 100), "dd": maxdd(np.cumprod(1 + r[ok])), "exp": 100.0}
    return out, bh, str(c.index[ok.argmax()].date())


def build_model(spx, b, cash, notes, n_tickers, members_now):
    T, I = S.build_signals(spx, b)
    c = spx["Close"]; D = c.index; n = len(c); cv = c.values
    B = Book(c, cash)
    pos = lambda d: int(D.searchsorted(pd.Timestamp(d)))
    ev0 = pos(EVID_START); in_ev = np.arange(n) >= ev0
    h1 = in_ev & np.asarray(D < SPLIT); h2 = in_ev & np.asarray(D >= SPLIT)
    keys = [x[0] for x in S.BOOK]
    bases = {w: B.stats(np.ones(n, bool), np.arange(n) >= pos(d)) for w, d in WIN.items()}

    # ---------------- per-condition evidence: on against off ----------------
    legs = []
    for key, name, short, fam, tier, rule, src, st in S.BOOK:
        t = T[key].reindex(D, fill_value=False).values
        era = np.arange(n) >= pos(WIN[st])
        on, off = B.stats(t, era), B.stats(~t, era)
        pl = B.placebo(t, era)
        mid = D[pos(WIN[st])] + (D[-1] - D[pos(WIN[st])]) / 2
        half = {}
        for lab, hm in (("a", np.asarray(D < mid)), ("b", np.asarray(D >= mid))):
            ha, hb = B.stats(t, era & hm), B.stats(~t, era & hm)
            half["on_" + lab], half["off_" + lab] = ha["sharpe"], hb["sharpe"]
            half["r_on_" + lab], half["r_off_" + lab] = ha["shr"], hb["shr"]
        ds = lambda a, bq: (a is not None and bq is not None and a > bq)
        beats = ds(on["shr"], off["shr"])
        halves = all(ds(half["r_on_" + l], half["r_off_" + l]) for l in "ab")
        sig = pl is not None and pl < 0.10
        enough = on["days"] >= 250 and off["days"] >= 250
        grade = "pass" if (beats and halves and sig and enough) else ("weak" if beats and (halves or sig) and enough else "none")
        v = t; i = n - 1
        while i > 0 and v[i - 1] == v[-1]:
            i -= 1
        legs.append({"key": key, "name": name, "short": short, "family": fam, "tier": tier, "rule": rule, "src": src,
                     "since": st, "on": on, "off": off, "pl": pl, "half": half, "beats": beats, "halves": halves,
                     "sig": sig, "grade": grade, "state": bool(v[-1]), "state_since": str(D[i].date()),
                     "d_sharpe": r1((on["sharpe"] - off["sharpe"]) if beats or (on["sharpe"] is not None and off["sharpe"] is not None) else None, 2),
                     "d_ann": r1((on["ann"] - off["ann"]) if (on["ann"] is not None and off["ann"] is not None) else None)})
    core = [l["key"] for l in legs if l["grade"] == "pass"]
    core_rule = "pass"
    if len(core) < 3:
        core = [l["key"] for l in legs if l["grade"] in ("pass", "weak")]; core_rule = "pass_weak"

    # ---------------- aggregator ----------------
    on_df = pd.DataFrame({k: T[k].reindex(D, fill_value=False).values for k in keys}, index=D)

    def agg(ks, hi, lo):
        cnt = on_df[ks].sum(1)
        sm = cnt.rolling(SMOOTH, min_periods=SMOOTH).mean()
        st = pd.Series(np.where(sm >= hi, 1, np.where(sm <= lo, -1, 0)), index=D)
        st[sm.isna()] = 0
        out = {"hi": hi, "lo": lo, "n": len(ks)}
        for v, lab in ((1, "strong"), (0, "mixed"), (-1, "broken")):
            m = (st == v).values
            out[lab] = {"all": B.stats(m, in_ev), "h1": B.stats(m, h1), "h2": B.stats(m, h2),
                        "pl": B.placebo(m, in_ev)}
        out["bycount"] = [dict(B.stats((cnt == k).values, in_ev), count=int(k))
                          for k in range(len(ks) + 1) if ((cnt == k) & in_ev).sum() > 60]
        return cnt, sm, st, out

    cnt_all, sm_all, st_all, ev_all = agg(keys, STRONG_AT, BROKEN_AT)
    price_k = [l["key"] for l in legs if l["family"] == "Price trend"]
    br_k = [l["key"] for l in legs if l["family"] != "Price trend"]
    sc = lambda ks, t: max(1, int(round(t * len(ks) / len(keys))))
    cnt_pr, sm_pr, st_pr, ev_pr = agg(price_k, sc(price_k, STRONG_AT), sc(price_k, BROKEN_AT))
    cnt_br, sm_br, st_br, ev_br = agg(br_k, sc(br_k, STRONG_AT), sc(br_k, BROKEN_AT))
    MODES = {"all": {"keys": keys, "ev": ev_all, "label": "All 9"},
             "price": {"keys": price_k, "ev": ev_pr, "label": "Price trend 4"},
             "breadth": {"keys": br_k, "ev": ev_br, "label": "Participation 5"}}

    # ---------------- the score as a risk filter ----------------
    wkk = D[ev0:].to_period("W-FRI")
    wlast = pd.Series(np.arange(n - ev0), index=D[ev0:]).groupby(wkk).last().values
    smp = lambda curve: [r1(float(x), 0) for x in curve[wlast]]
    rows = []; curves = {}
    bh_stat, bh_curve = B.filt(np.ones(n, bool), in_ev)
    curves["hold"] = smp(10000 * bh_curve)
    prev = sm_all.shift(1)
    for k in FILT_K:
        lm = (prev >= k).fillna(False).values
        s, cu = B.filt(lm, in_ev)
        rows.append(dict(s, k=k)); curves["k%d" % k] = smp(10000 * cu)
    lm = (st_all.shift(1) >= 0).fillna(False).values          # out only while the trend is broken
    s, cu = B.filt(lm, in_ev)
    rows.append(dict(s, k="state")); curves["state"] = smp(10000 * cu)
    filt = {"rows": rows, "bh": bh_stat, "curves": curves}

    # ---------------- McClellan Summation regimes ----------------
    msi = I["msi"].reindex(D); rising = msi.diff() > 0
    same_ret = c.pct_change().values                      # contemporaneous, for the circularity check

    def annm(m):
        x = B.ret[m & B.ok]
        return ((1 + x).prod() ** (252 / len(x)) - 1) * 100 if len(x) > 60 else None

    def annsame(m):
        x = same_ret[m & ~np.isnan(same_ret)]
        return ((1 + x).prod() ** (252 / len(x)) - 1) * 100 if len(x) > 60 else None

    msi_reg = []
    for lab, m in (("Above zero and rising", (msi > 0) & rising), ("Above zero and declining", (msi > 0) & ~rising),
                   ("Below zero and rising", (msi <= 0) & rising), ("Below zero and declining", (msi <= 0) & ~rising)):
        mm = (m & msi.notna()).fillna(False).values
        s = B.stats(mm, in_ev)
        cu = B.curve(mm & in_ev)[ev0:]
        rr = B.ret[mm & in_ev & B.ok]
        top = np.sort(rr)[-10:]
        s_ex = ((1 + np.sort(rr)[:-10]).prod() ** (252 / max(1, len(rr) - 10)) - 1) * 100
        msi_reg.append({"label": lab, **s, "eq": smp(cu), "end": r1(float(cu[-1]), 0),
                        "ann_ex10": r1(s_ex), "best10": r1(float(top.sum() * 100)),
                        "lag": [r1(annm((m & msi.notna()).shift(L).fillna(False).astype(bool).values & in_ev)) for L in MSI_LAGS],
                        "same": r1(annsame((m & msi.notna()).fillna(False).values & in_ev))})

    # ---------------- thin participation near a 5-year high ----------------
    thin = I["thin"].reindex(D, fill_value=False).values
    nearhi = T["hi5y"].reindex(D, fill_value=False).values
    f2m = (c.shift(-42) / c - 1).values
    warn = {}
    for k, m in (("thin", thin), ("near_ok", nearhi & ~thin), ("near", nearhi), ("base", np.ones(n, bool))):
        s = B.stats(m, in_ev)
        v = f2m[m & in_ev]; v = v[~np.isnan(v)]
        s["2m"] = r1(v.mean() * 100) if len(v) else None
        s["w2m"] = r1((v > 0).mean() * 100, 0) if len(v) else None
        warn[k] = s
    warn["pl"] = B.placebo(thin, in_ev)

    # ---------------- moving-average study ----------------
    sma_rows, sma_bh, sma_start = sma_study(c, cash, SMA_PERIODS)
    best = max(sma_rows, key=lambda r: (r["sharpe"] is not None, r["sharpe"]))
    band = [r for r in sma_rows if 130 <= r["n"] <= 200]
    short_ = [r for r in sma_rows if r["n"] <= 50]
    sma = {"rows": sma_rows, "bh": sma_bh, "start": sma_start, "best": best,
           "band_mean": r1(float(np.mean([r["sharpe"] for r in band])), 3),
           "band_dd": r1(float(np.mean([r["dd"] for r in band]))),
           "short_mean": r1(float(np.mean([r["sharpe"] for r in short_])), 3),
           "all_mean": r1(float(np.mean([r["sharpe"] for r in sma_rows])), 3),
           "beat": int(sum(1 for r in sma_rows if r["sharpe"] > sma_bh["sharpe"])),
           "band_beat": int(sum(1 for r in band if r["sharpe"] > sma_bh["sharpe"])), "band_n": len(band)}

    # ---------------- state-change log ----------------
    sv = st_all.values; log = []
    for i in range(ev0 + 1, n):
        if sv[i] != sv[i - 1]:
            j = i - 1
            while j > 0 and sv[j - 1] == sv[i - 1]:
                j -= 1
            log.append({"d": str(D[i].date()), "to": int(sv[i]), "frm": int(sv[i - 1]), "days": int(i - j),
                        "px": r1(cv[i], 0), "cnt": int(cnt_all.iloc[i]),
                        **{"f" + k: r1(B.fwd[k][i] * 100) for k in HZ}})
    log.sort(key=lambda r: r["d"], reverse=True)

    # ---------------- current status ----------------
    def last(s, nd=2):
        s = s.dropna(); return r1(s.iloc[-1], nd) if len(s) else None

    sw_hi, sw_lo, sw_ph, sw_pl = (last(I[k], 0) for k in ("sw_hi", "sw_lo", "sw_ph", "sw_pl"))
    status = {
        "hhhl": f"Last swing high {sw_hi:,.0f} against {sw_ph:,.0f} before it, last swing low {sw_lo:,.0f} against {sw_pl:,.0f}",
        "ma": f"Close {last(I['gap200'], 1):+.1f}% against the 200-day average, {last(I['gap150'], 1):+.1f}% against the 150-day",
        "slope": f"200-day average {last(I['ma200_chg'], 1):+.1f}% over 21 sessions, {last(I['ma200_chg1'], 3):+.3f}% on the day",
        "hi5y": f"{abs(last(I['below5y'], 1)):.1f}% below the 5-year high",
        "p200": f"{last(I['p200'], 1):.1f}% of members above their 200-day average",
        "r200": f"{last(I['r200'], 1):.1f}% of members with a rising 200-day average",
        "corr": f"{last(I['corr'], 1):.1f}% of members more than 10% below their 52-week high, {last(I['bear'], 1):.1f}% more than 20% below",
        "msi": f"Summation Index {last(I['msi'], 0):,.0f}, {last(I['msi_chg'], 0):+,.0f} over 21 sessions",
        "nhnl": f"Net new 52-week highs {last(I['nhnl'], 2):+.2f}% of members on a 10-session average",
    }
    for l in legs:
        l["status"] = status[l["key"]]

    # ---------------- payload ----------------
    sl = slice(ev0, n); idx = D[sl]

    def arr(s, nd=1):
        v = np.asarray(s)[sl]
        return [None if (x is None or (isinstance(x, float) and np.isnan(x))) else (int(x) if nd == 0 else round(float(x), nd)) for x in v]

    def since(series):
        v = series.values; i = len(v) - 1
        while i > 0 and v[i - 1] == v[-1]:
            i -= 1
        return str(series.index[i].date())

    wd = [str(idx[i].date()) for i in wlast]

    def wk(s, nd=2):
        v = pd.Series(s).reindex(D).values[sl][wlast].astype(float)
        return [None if np.isnan(x) else round(float(x), nd) for x in v]

    C_ = {"a": "#7A4A66", "b": "#0E756C", "c": "#A67A22", "d": "#4E6577"}
    detail = {
        "hhhl": {"series": [["Last confirmed swing high", wk(I["sw_hi"], 0), C_["b"]], ["Last confirmed swing low", wk(I["sw_lo"], 0), C_["a"]]], "refs": [], "step": 1},
        "ma": {"series": [["Close against the 200-day average, %", wk(I["gap200"], 2), C_["a"]], ["Close against the 150-day average, %", wk(I["gap150"], 2), C_["d"]]], "refs": [[0, "Above the average"]]},
        "slope": {"series": [["200-day average, % change over 21 sessions", wk(I["ma200_chg"], 2), C_["a"]]], "refs": [[0, "Rising above this line"]]},
        "hi5y": {"series": [["Below the 5-year high, %", wk(I["below5y"], 2), C_["a"]]], "refs": [[-3, "Counts inside 3%"], [0, "At the high"]]},
        "p200": {"series": [["Members above their 200-day average, %", wk(I["p200"], 1), C_["a"]]], "refs": [[52, "Counts above 52%"], [60, "Checklist's 60% variant"]]},
        "r200": {"series": [["Members with a rising 200-day average, %", wk(I["r200"], 1), C_["a"]]], "refs": [[57, "Counts above 57%"]]},
        "corr": {"series": [["More than 10% below their 52-week high, %", wk(I["corr"], 1), C_["a"]], ["More than 20% below, %", wk(I["bear"], 1), C_["d"]]], "refs": [[40, "Counts below 40%"]]},
        "msi": {"series": [["McClellan Summation Index", wk(I["msi"], 0), C_["a"]]], "refs": [[0, "Above zero"]]},
        "nhnl": {"series": [["Net 52-week highs, % of members (10-session average)", wk(I["nhnl"], 2), C_["a"]]], "refs": [[0, "More highs than lows"]]},
    }
    for k in detail:
        detail[k]["on"] = [int(x) for x in T[k].reindex(D, fill_value=False).values[sl][wlast]]

    P = {
        "meta": {"asof": str(D[-1].date()), "built": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                 "evid_start": str(idx[0].date()), "split": SPLIT[:4], "smooth": SMOOTH, "strong_at": STRONG_AT,
                 "broken_at": BROKEN_AT, "n_legs": len(keys), "core": core, "price_k": price_k, "br_k": br_k,
                 "modes": {k: {"keys": v["keys"], "label": v["label"], "hi": v["ev"]["hi"], "lo": v["ev"]["lo"]} for k, v in MODES.items()},
                 "msi_lags": MSI_LAGS,
                 "core_rule": core_rule, "members_now": int(members_now), "tickers_priced": int(n_tickers),
                 "membership": notes, "pivot_k": S.PIVOT_K, "price_start": WIN["1962"][:4],
                 "cov_2000": r1(b["coverage"].loc["2000"].mean() * 100, 0), "cov_2010": r1(b.loc["2010", "coverage"].mean() * 100, 0),
                 "cov_2020": r1(b.loc["2020", "coverage"].mean() * 100, 0), "cov_last": r1(b["coverage"].iloc[-21:].mean() * 100, 0),
                 "breadth_start": str(b.index[0].date()), "cash_start": sma_start},
        "d": [str(x.date()) for x in idx], "spx": arr(c, 2),
        "cnt": arr(cnt_all, 0), "st": arr(st_all, 0), "sm": arr(sm_all, 2),
        "cntp": arr(cnt_pr, 0), "stp": arr(st_pr, 0), "cntb": arr(cnt_br, 0), "stb": arr(st_br, 0),
        "on": {k: arr(on_df[k].astype(int), 0) for k in keys},
        "thin": arr(pd.Series(thin.astype(int), index=D), 0),
        "wd": wd, "wspx": [round(float(x), 1) for x in cv[sl][wlast]], "detail": detail,
        "cov": wk(b["coverage"].reindex(D) * 100, 1),
        "now": {"spx": r1(cv[-1], 2), "cnt": int(cnt_all.iloc[-1]), "sm": r1(float(sm_all.iloc[-1]), 2),
                "st": int(st_all.iloc[-1]), "st_since": since(st_all), "cntp": int(cnt_pr.iloc[-1]), "stp": int(st_pr.iloc[-1]), "stp_since": since(st_pr),
                "cntb": int(cnt_br.iloc[-1]), "stb": int(st_br.iloc[-1]), "stb_since": since(st_br), "thin": bool(thin[-1]),
                "thin_since": since(pd.Series(thin, index=D)), "below5y": r1(float(I["below5y"].iloc[-1]), 1),
                "p200": r1(float(I["p200"].iloc[-1]), 1), "r200": r1(float(I["r200"].iloc[-1]), 1),
                "corr": r1(float(I["corr"].iloc[-1]), 1), "msi": r1(float(I["msi"].iloc[-1]), 0),
                "dd": r1(float((cv[-1] / c.rolling(252, min_periods=126).max().iloc[-1] - 1) * 100)),
                "primary": int(sum(1 for l in legs if l["tier"] == "Primary" and l["state"])),
                "n_primary": int(sum(1 for l in legs if l["tier"] == "Primary")),
                "last_change": log[0] if log else None},
        "legs": legs, "bases": bases, "ev_all": ev_all, "ev_price": ev_pr, "ev_breadth": ev_br, "msi_reg": msi_reg, "warn": warn,
        "sma": sma, "filt": filt, "log": log[:60],
    }
    return P


def run(cache="cache"):
    import pickle, os
    px = pickle.load(open(os.path.join(cache, "prices.pkl"), "rb"))
    b = pd.read_pickle(os.path.join(cache, "breadth.pkl"))
    hist, notes = pickle.load(open(os.path.join(cache, "members.pkl"), "rb"))
    spx = px["spx"]; spx = spx[spx.index >= "1959-01-01"]
    cash = px["irx"]["Close"] if "irx" in px else pd.Series(0.0, index=spx.index)
    return build_model(spx, b, cash, notes, px["adj"].shape[1], len(hist["set"].iloc[-1]))


if __name__ == "__main__":
    import warnings, os; warnings.filterwarnings("ignore")
    P = run()
    json.dump(P, open("cache/payload.json", "w"), separators=(",", ":"))
    print("payload", round(os.path.getsize("cache/payload.json") / 1e3), "kB")
    print(json.dumps(P["now"])); print("core:", P["meta"]["core"], P["meta"]["core_rule"])
    print(f"{'condition':34s} {'grade':5s} {'on%':>5s} {'annOn':>6s} {'annOff':>6s} {'shOn':>5s} {'shOff':>5s} {'volOn':>5s} {'volOff':>6s} {'ddOn':>6s} {'ddOff':>6s} {'p':>5s} halves")
    for l in P["legs"]:
        o, f = l["on"], l["off"]
        print(f"{l['name'][:34]:34s} {l['grade']:5s} {o['pct']:5.1f} {o['ann']:6} {f['ann']:6} {o['sharpe']:5} {f['sharpe']:5} {o['vol']:5} {f['vol']:6} {o['dd']:6} {f['dd']:6} {str(l['pl']):>5s} {l['half']['on_a']}v{l['half']['off_a']} {l['half']['on_b']}v{l['half']['off_b']}")
    for nm, E in (("ALL", P["ev_all"]), ("PRICE", P["ev_price"]), ("BREADTH", P["ev_breadth"])):
        print(nm, "hi", E["hi"], "lo", E["lo"], "n", E["n"])
        for k in ("strong", "mixed", "broken"):
            s = E[k]["all"]; print(f"   {k:7s} {s['pct']:5}% ann {s['ann']:6} sharpe {s['sharpe']:5} vol {s['vol']:5} dd {s['dd']:7} 12m {s['12m']:5} ({s['w12m']}%) p {E[k]['pl']} | h1 {E[k]['h1']['sharpe']} h2 {E[k]['h2']['sharpe']}")
        print("   counts:", [(g["count"], g["pct"], g["ann"], g["sharpe"], g["12m"]) for g in E["bycount"]])
    print("bases", {k: {x: v[x] for x in ("ann", "vol", "sharpe", "dd", "12m")} for k, v in P["bases"].items()})
    for r in P["msi_reg"]:
        print(f"MSI {r['label']:26s} {r['pct']:5}% ann {r['ann']:6} sharpe {r['sharpe']:5} end {r['end']:8} ex10 {r['ann_ex10']:6} same {r['same']:7} lags {r['lag']}")
    print("warn", {k: ({x: v[x] for x in ("pct", "ann", "sharpe", "1m", "2m", "3m", "12m")} if isinstance(v, dict) else v) for k, v in P["warn"].items()})
    print("sma best", P["sma"]["best"], "\n    bh", P["sma"]["bh"], "\n    band", P["sma"]["band_mean"], "band_dd", P["sma"]["band_dd"], "short", P["sma"]["short_mean"], "beat", P["sma"]["beat"], "band_beat", P["sma"]["band_beat"], "from", P["sma"]["start"])
    print("filter:", P["filt"]["bh"])
    for r in P["filt"]["rows"]: print("   ", r)
    print("log", P["log"][:2])
