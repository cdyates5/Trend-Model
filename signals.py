"""Stage 3a: the momentum/trend conditions. Every function returns a boolean daily Series (True while the condition holds)."""
import numpy as np, pandas as pd

PIVOT_K = 21          # sessions either side of a swing pivot (confirmed PIVOT_K sessions late)
NHNL_SMOOTH = 10      # sessions averaged for net new highs


def swing_state(c, k=PIVOT_K):
    """Higher highs and higher lows from confirmed swing pivots on closing prices.
    A pivot at i is only known at i+k, so the state at t uses pivots confirmed by t (no look-ahead).
    Returns (state, last_high, prev_high, last_low, prev_low) as daily Series."""
    v = c.values; n = len(v)
    roll_max = pd.Series(v).rolling(2 * k + 1, center=True).max().values
    roll_min = pd.Series(v).rolling(2 * k + 1, center=True).min().values
    is_hi = np.zeros(n, bool); is_lo = np.zeros(n, bool)
    ok = ~np.isnan(roll_max)
    is_hi[ok] = v[ok] >= roll_max[ok] - 1e-12
    is_lo[ok] = v[ok] <= roll_min[ok] + 1e-12
    st = np.zeros(n, np.int8); LH = np.full(n, np.nan); PH = np.full(n, np.nan)
    LL = np.full(n, np.nan); PL = np.full(n, np.nan)
    his, los = [], []
    for t in range(n):
        j = t - k                                   # pivots at j become known now
        if j >= 0:
            if is_hi[j] and (not his or j - his[-1][0] >= k):
                his.append((j, v[j]))
            elif is_hi[j] and his and v[j] >= his[-1][1]:
                his[-1] = (j, v[j])                 # extend an adjacent plateau
            if is_lo[j] and (not los or j - los[-1][0] >= k):
                los.append((j, v[j]))
            elif is_lo[j] and los and v[j] <= los[-1][1]:
                los[-1] = (j, v[j])
        if len(his) >= 2:
            LH[t], PH[t] = his[-1][1], his[-2][1]
        if len(los) >= 2:
            LL[t], PL[t] = los[-1][1], los[-2][1]
        if len(his) >= 2 and len(los) >= 2:
            # Read as a chartist would, so neither side waits the k sessions a pivot takes to confirm:
            # a close above the last confirmed swing high is itself a higher high, and a close under the
            # last confirmed swing low breaks the run of higher lows at once.
            hh = his[-1][1] > his[-2][1] or v[t] > his[-1][1]
            hl = los[-1][1] > los[-2][1] and v[t] > los[-1][1]
            st[t] = 1 if (hh and hl) else 0
    f = lambda a: pd.Series(a, index=c.index)
    return f(st.astype(bool)), f(LH), f(PH), f(LL), f(PL)


def build_signals(spx, b):
    """spx: daily index OHLC; b: constituent breadth frame. Returns (conditions, indicator series)."""
    D = spx.index; c = spx["Close"]
    T, I = {}, {}

    # --- price trend (index level) ---
    hh, LH, PH, LL, PL = swing_state(c)
    T["hhhl"] = hh
    I.update(sw_hi=LH, sw_lo=LL, sw_ph=PH, sw_pl=PL)

    ma150 = c.rolling(150, min_periods=150).mean(); ma200 = c.rolling(200, min_periods=200).mean()
    T["ma"] = ((c > ma150) & (c > ma200)).fillna(False) & ma200.notna()
    I["gap150"] = 100 * (c / ma150 - 1); I["gap200"] = 100 * (c / ma200 - 1)

    T["slope"] = (ma200.diff() > 0).fillna(False) & ma200.notna()
    I["ma200_chg"] = 100 * (ma200 / ma200.shift(21) - 1)
    I["ma200_chg1"] = 100 * (ma200 / ma200.shift(1) - 1)

    hi5 = c.rolling(1260, min_periods=252).max()
    I["below5y"] = 100 * (c / hi5 - 1)
    T["hi5y"] = (I["below5y"] >= -3).fillna(False) & hi5.notna()

    # --- participation and internal momentum (constituents) ---
    bb = b.reindex(D)
    T["p200"] = (bb["p200"] > 52).fillna(False)
    T["r200"] = (bb["r200"] > 57).fillna(False)
    T["corr"] = (bb["corr"] < 40).fillna(False) & bb["corr"].notna()
    msi = bb["msi"]
    T["msi"] = ((msi > 0) & (msi.diff() > 0)).fillna(False)
    nn = bb["nhnl"].rolling(NHNL_SMOOTH, min_periods=NHNL_SMOOTH).mean()
    T["nhnl"] = (nn > 0).fillna(False)
    I.update(p200=bb["p200"], r200=bb["r200"], corr=bb["corr"], msi=msi, msi_chg=msi.diff(21), nhnl=nn,
             nhnl_raw=bb["nhnl"], bear=bb["bear"], r200_21=bb["r200_21"])

    # --- warning overlay: near a 5-year high on thin long-term participation (not a leg) ---
    I["thin"] = (T["hi5y"] & (bb["r200"] < 57) & bb["r200"].notna()).fillna(False)
    return T, I


BOOK = [  # key, name, short, family, tier, rule, source, history-start
    ("hhhl", "Higher highs and higher lows", "HH/HL", "Price trend", "Primary",
     "The last confirmed swing high is above the one before it and the last confirmed swing low is above the one before it, on 21-session closing pivots",
     "Momentum/trend checklist", "1962"),
    ("ma", "Price above the 150–200 day average", "150–200DMA", "Price trend", "Primary",
     "The close is above both the 150-day and the 200-day simple average", "Checklist; 3Fourteen Research SMA study", "1962"),
    ("slope", "200-day average rising", "200DMA up", "Price trend", "Primary",
     "The 200-day average is above its level one session earlier", "Momentum/trend checklist", "1962"),
    ("hi5y", "Within 3% of a 5-year high", "5y high", "Price trend", "Context",
     "The close is no more than 3% below the highest close of the past five years", "SentimenTrader", "1962"),
    ("p200", "Members above their 200-day average", "%>200DMA", "Participation", "Primary",
     "More than 52% of index members close above their own 200-day average", "SentimenTrader", "2000"),
    ("r200", "Members with a rising 200-day average", "Rising 200", "Participation", "Primary",
     "More than 57% of index members have a 200-day average above its level one session earlier", "SentimenTrader", "2000"),
    ("corr", "Few members in a correction", "In correction", "Participation", "Secondary",
     "Fewer than 40% of index members are more than 10% below their own 52-week high", "Momentum/trend checklist", "2000"),
    ("msi", "Summation Index above zero and rising", "Summation", "Internal momentum", "Secondary",
     "The ratio-adjusted McClellan Summation Index is above zero and above its level one session earlier", "SentimenTrader", "2000"),
    ("nhnl", "52-week highs beat lows", "NH − NL", "Internal momentum", "Secondary",
     "More members at 52-week highs than lows, on a 10-session average of net new highs", "Momentum/trend checklist", "2000"),
]
