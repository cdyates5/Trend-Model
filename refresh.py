#!/usr/bin/env python3
"""Refresh the S&P 500 Momentum/Trend Aggregator and write site/index.html.

    python3 refresh.py              # top up prices if the cache is warm, else full download
    python3 refresh.py --full       # force a full history download (~8 minutes)
    python3 refresh.py --no-fetch   # recompute from the cached prices only
    python3 refresh.py --check      # run the sanity gate against the built payload

The scheduled job runs the default path on weekdays and --full once a week; a full run
also happens automatically when the cache is missing or more than FULL_EVERY days old,
which is what repairs any split adjustment the top-up could not splice cleanly.
"""
import argparse, datetime as dt, json, os, pickle, shutil, sys, warnings
warnings.filterwarnings("ignore")

CACHE = os.environ.get("MOM_CACHE", "cache")
SITE = os.environ.get("MOM_SITE", "site")
FULL_EVERY = 7          # days before a top-up is replaced by a full download
MAX_STALE = 5           # calendar days the latest close may lag today before the gate fails
MIN_MEMBERS = 450       # priced members required on the latest session
MIN_COVERAGE = 90.0     # percent of the index priced on the latest session


def log(*a):
    print(dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S"), *a, flush=True)


def meta_path():
    return os.path.join(CACHE, "meta.json")


def read_meta():
    try:
        return json.load(open(meta_path()))
    except Exception:
        return {}


def write_meta(m):
    json.dump(m, open(meta_path(), "w"), indent=1)


def need_full(args):
    if args.full:
        return True, "forced"
    if not os.path.exists(os.path.join(CACHE, "prices.pkl")):
        return True, "no cached prices"
    lf = read_meta().get("last_full")
    if not lf:
        return True, "no record of a full download"
    age = (dt.date.today() - dt.date.fromisoformat(lf)).days
    if age >= FULL_EVERY:
        return True, f"last full download was {age} days ago"
    return False, f"last full download was {age} days ago"


def do_fetch(args):
    import fetch
    fetch.CACHE = CACHE
    os.makedirs(CACHE, exist_ok=True)
    hist, notes = fetch.get_membership()
    log("membership:", len(hist), "rows,", len(hist["set"].iloc[-1]), "members now,", notes)
    pickle.dump((hist, notes), open(os.path.join(CACHE, "members.pkl"), "wb"))

    full, why = need_full(args)
    log(("full download" if full else "top-up") + ":", why)
    pp = os.path.join(CACHE, "prices.pkl")
    m = read_meta()
    if full:
        universe = set().union(*hist[hist["date"] >= "1995-12-01"]["set"])
        px = fetch.get_prices(universe)
        m["last_full"] = dt.date.today().isoformat()
    else:
        old = pickle.load(open(pp, "rb"))
        px, info = fetch.top_up(old, sorted(hist["set"].iloc[-1]))
        m["last_topup_info"] = info
        # never replace a good panel with a worse one
        if px["close"].shape[1] < old["close"].shape[1] - 5:
            sys.exit("top-up lost columns (%d -> %d); keeping the cached panel"
                     % (old["close"].shape[1], px["close"].shape[1]))
        if px["spx"].index[-1] < old["spx"].index[-1]:
            sys.exit("top-up went backwards in time; keeping the cached panel")
    pickle.dump(px, open(pp, "wb"))
    m["last_run"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    m["last_close"] = str(px["spx"].index[-1].date())
    m["tickers"] = int(px["adj"].shape[1])
    write_meta(m)
    log("prices to", m["last_close"], "|", m["tickers"], "tickers")


def do_build(args):
    import pandas as pd, breadth, model, build
    px = pickle.load(open(os.path.join(CACHE, "prices.pkl"), "rb"))
    hist, notes = pickle.load(open(os.path.join(CACHE, "members.pkl"), "rb"))
    b, _ = breadth.build_breadth(px, hist, price=args.price)
    b.to_pickle(os.path.join(CACHE, "breadth.pkl"))
    log("breadth", b.shape, "| coverage now", round(float(b["coverage"].iloc[-1]) * 100, 1), "%")
    P = model.run(CACHE)
    json.dump(P, open(os.path.join(CACHE, "payload.json"), "w"), separators=(",", ":"))
    log("model: score", P["now"]["cnt"], "of", P["meta"]["n_legs"],
        "| state", P["now"]["st"], "| as of", P["meta"]["asof"])
    os.makedirs(SITE, exist_ok=True)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    out, size = build.build(os.path.join(CACHE, "payload.json"), "template.html",
                            os.path.join(SITE, "index.html"))
    log("wrote", out, round(size / 1024), "kB")


def do_check(args):
    """Block a deploy on stale, thin or impossible data."""
    bad = []
    pf = os.path.join(CACHE, "payload.json")
    if not os.path.exists(pf):
        sys.exit("gate: no payload")
    P = json.load(open(pf))
    M, N = P["meta"], P["now"]
    asof = dt.date.fromisoformat(M["asof"])
    stale = (dt.date.today() - asof).days
    if stale > MAX_STALE:
        bad.append(f"latest close {M['asof']} is {stale} days old (limit {MAX_STALE})")
    if M["members_now"] < MIN_MEMBERS:
        bad.append(f"only {M['members_now']} members in the index set")
    if (M["cov_last"] or 0) < MIN_COVERAGE:
        bad.append(f"priced coverage {M['cov_last']}% is below {MIN_COVERAGE}%")
    if not 0 <= N["cnt"] <= M["n_legs"]:
        bad.append(f"score {N['cnt']} outside 0-{M['n_legs']}")
    for k, lo, hi in (("p200", 0, 100), ("r200", 0, 100), ("corr", 0, 100), ("below5y", -100, 0.001)):
        v = N.get(k)
        if v is None or not lo <= v <= hi:
            bad.append(f"{k} = {v} outside [{lo}, {hi}]")
    if len(P["d"]) != len(P["spx"]) or P["d"][-1] != M["asof"]:
        bad.append("date series does not line up with the latest close")
    if sorted(P["d"]) != P["d"]:
        bad.append("date series is not sorted")
    if len(P["legs"]) != M["n_legs"]:
        bad.append("condition count does not match the book")
    ip = os.path.join(SITE, "index.html")
    size = os.path.getsize(ip) if os.path.exists(ip) else 0
    if size < 300_000:
        bad.append(f"index.html is {size} bytes, which is too small to be the built page")
    if bad:
        print("GATE FAILED:")
        for x in bad:
            print("  -", x)
        sys.exit(1)
    log(f"gate passed: {M['asof']}, score {N['cnt']}/{M['n_legs']}, "
        f"coverage {M['cov_last']}%, page {round(size/1024)} kB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="force a full history download")
    ap.add_argument("--no-fetch", action="store_true", help="recompute from the cached prices")
    ap.add_argument("--check", action="store_true", help="run the sanity gate and exit")
    ap.add_argument("--price", choices=["close", "adj"], default="close",
                    help="price series for member breadth; close (split-adjusted) matches the published charts")
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if args.check:
        return do_check(args)
    if not args.no_fetch:
        do_fetch(args)
    do_build(args)
    do_check(args)


if __name__ == "__main__":
    main()
