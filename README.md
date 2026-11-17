# S&P 500 Momentum/Trend Aggregator

Nine trend, participation and internal-momentum conditions scored as states, with the
evidence for each judged on risk-adjusted returns. Rebuilds itself after every US close
and publishes to GitHub Pages.

**Live site:** `https://<your-username>.github.io/<repo-name>/`

---

## File map

| File | Purpose |
|---|---|
| `refresh.py` | The driver. Fetches, recomputes, writes `site/index.html`, runs the sanity gate. |
| `fetch.py` | Index membership, constituent prices, `^GSPC` and `^IRX`. Full download and top-up modes. |
| `breadth.py` | Member breadth from point-in-time constituents: %>200DMA, % rising, % in correction, new highs and lows, McClellan Summation. |
| `signals.py` | The nine conditions and the swing-pivot logic. Edit this to change a threshold. |
| `model.py` | Aggregator, risk-adjusted evidence, placebo tests, the SMA study, the JSON payload. |
| `build.py` | Injects the payload into the template. |
| `template.html` | Layout, styling and all chart code. Edit this for presentation changes. |
| `site/index.html` | The built dashboard. Generated — never edit it directly. |
| `.github/workflows/refresh.yml` | The scheduled job. |
| `requirements.txt` | pandas, numpy, requests, yfinance, lxml, html5lib. |

Nothing else is needed: no API keys, no secrets, no paid data.

---

## Deploy in about ten minutes

### 1. Create the repository

Create a new repository on GitHub. **Public** is simplest — Pages on a private repo
needs a paid plan. Don't let GitHub add a README, you have one.

### 2. Upload the files

```bash
git init
git add .
git commit -m "Momentum/Trend Aggregator"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

> If you use the drag-and-drop uploader instead, note that it silently skips dot-folders,
> so `.github/` will not make it. After uploading everything else, click
> **Add file → Create new file**, type `.github/workflows/refresh.yml` as the filename
> (typing the slashes creates the folders), paste the workflow in, and commit.

### 3. Turn on Pages

**Settings → Pages → Build and deployment → Source: GitHub Actions.**

That is the whole setting — don't pick a branch. The workflow uploads the built page
as an artifact and deploys it, so no data is ever committed back to the repo.

### 4. Allow the workflow to commit

**Settings → Actions → General → Workflow permissions → Read and write permissions → Save.**

Only the monthly keepalive commit needs this (see *Schedule pauses* below).

### 5. First run

**Actions → Refresh Momentum/Trend Aggregator → Run workflow**, and tick
**Force a full history download** for this first one. It takes about eight minutes:
roughly 1,200 tickers of history, most of which are delisted names needed for
point-in-time breadth. Every later run takes about 90 seconds.

When it finishes green, the site is live at the URL above.

---

## How the refresh works

Each run does four things: rolls membership forward, tops up prices, recomputes
everything, and publishes — but only if the result passes a sanity gate.

**Prices.** A full download pulls every name that has ever been in the index over the
sample. A top-up pulls the last 400 days for the 500 current members and splices them
onto the stored panel, which is what the weekday runs do. The panel lives in the Actions
cache (about 49 MB), not in the repo.

**Splits are handled.** A top-up compares the overlap between new and stored bars for
each ticker; if the median ratio has moved by more than 0.5% — a split, or a re-based
adjustment — that ticker is re-downloaded in full instead of spliced. The Saturday run
rebuilds the whole panel anyway, so nothing can drift for more than a week.

**The sanity gate** blocks the deploy, leaving the previous page live, if the latest
close is more than 5 days old, fewer than 450 members are in the index set, priced
coverage is under 90%, the score is outside 0–9, any breadth reading is outside its
possible range, or the built page is suspiciously small. Run it yourself with
`python refresh.py --check`.

### Schedule

| When (UTC) | What |
|---|---|
| 22:30 Tue–Fri | Top-up, about 90 seconds. Covers the Mon–Thu closes. |
| 22:30 Sat | Full history rebuild, about 8 minutes. Covers the Friday close. |
| 03:00 on the 1st | Keepalive commit only. |

22:30 UTC is roughly 90 minutes after the US close, which is enough for Yahoo to settle.
In Melbourne that lands at 9:30am AEDT the following morning.

To change the cadence, edit the `cron` lines in the workflow (always UTC):

| Cadence | cron |
|---|---|
| Weekdays after the close | `30 22 * * 2-6` |
| Twice daily | `30 22 * * 2-6` and `30 11 * * 1-5` |
| Mondays only | `30 22 * * 1` |

### Schedule pauses

GitHub disables scheduled workflows on repositories with no commits for 60 days. Because
this workflow deploys an artifact rather than committing, the monthly keepalive commit on
the 1st exists purely to reset that clock. Any push or manual run also re-arms it.

---

## Running it locally

```bash
pip install -r requirements.txt
python refresh.py --full        # first time, ~8 minutes
python refresh.py               # later, ~90 seconds
open site/index.html
```

| Command | Effect |
|---|---|
| `python refresh.py` | Top up if the cache is warm, otherwise full download |
| `python refresh.py --full` | Force the full history download |
| `python refresh.py --no-fetch` | Recompute and rebuild the page from cached prices |
| `python refresh.py --check` | Sanity gate only |
| `python refresh.py --price adj` | Build member breadth on dividend-adjusted closes |

`--no-fetch` is the one to use while editing `template.html` or a threshold in
`signals.py`: it rebuilds the page in a couple of seconds.

---

## Changing the model

**A threshold.** `signals.py`, in `build_signals`. The published levels are 52% above a
200-day average, 57% with a rising one, 40% in a correction and 3% from a 5-year high.
The checklist's 60% variant is already drawn on the input chart for comparison, so
switching `T["p200"] = (bb["p200"] > 52)` to `> 60` is a one-character edit.

**A condition.** Add the state to `build_signals` and an entry to `BOOK` at the bottom of
`signals.py` (key, name, short label, family, tier, rule text, source, history start), then
add a `detail` entry in `model.py` so it appears in the inputs chart. Everything else —
lanes, board, evidence table, scoring — picks it up automatically.

**The state thresholds.** `STRONG_AT` and `BROKEN_AT` at the top of `model.py`, currently
7 and 3 of 9, with `SMOOTH = 21` sessions.

After any change run `python refresh.py --no-fetch`.

---

## Caveats worth keeping in mind

**Yahoo and datacenter IPs.** Yahoo rate-limits aggressively and sometimes blocks cloud
IP ranges outright. The fetcher retries with backoff and a top-up only makes about a
dozen requests, so this is usually fine, but if a run fails with 429s the gate keeps the
previous page live rather than publishing something broken. If it becomes persistent,
run `python refresh.py --full` locally and use the commit-based workflow below.

**Member coverage.** Yahoo drops delisted names, so priced members cover 47% of the index
in 2000, 66% in 2010, 87% in 2020 and 100% now. Early breadth is tilted towards survivors
and reads too strong before about 2010. The by-halves columns in the evidence table are
the check on this.

**Membership roll-forward.** Wikipedia moved its index-changes table, so membership comes
from the fja05680 history (currently to 18 Aug 2026) rolled forward using the dated
additions in Wikipedia's current-constituents table, pairing each addition with a removal.
The current set is exact; which specific name left on which day inside the gap is
approximate, which matters to breadth by well under a percentage point.

**Alignment.** Every statistic reads a condition at the close and pairs it with the *next*
session's return. Published regime tables are usually contemporaneous. The Summation panel
shows both so the difference is visible.

---

## Alternative: commit the page instead of deploying an artifact

Useful if Pages-via-Actions is blocked on your account, or if you want the page's history
in git. Set **Settings → Pages → Source: Deploy from a branch → `main` / `/docs`**, then
replace the `upload-pages-artifact` step and the whole `deploy` job with:

```yaml
      - name: Commit the built page
        run: |
          mkdir -p docs && cp site/index.html docs/index.html && touch docs/.nojekyll
          git config user.name  "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add docs
          git diff --cached --quiet && echo "no change" && exit 0
          git commit -m "Refresh ${{ steps.refresh.outputs.asof }}"
          git push
```

The commit doubles as the keepalive, so you can drop the monthly cron and the keepalive
step. The cost is about 730 kB of repo growth per trading day.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Workflow green, site unchanged | Pages source is set to a branch. Set it to **GitHub Actions**. |
| `GATE FAILED: latest close … days old` | Yahoo returned stale or empty data. Re-run; if it persists, run with **Force a full history download**. |
| `GATE FAILED: priced coverage …` | A batch of tickers failed to download. Re-run with the full download. |
| Run takes 8 minutes on a weekday | The Actions cache expired (7 days without a hit) and it fell back to a full download. Harmless. |
| `top-up lost columns … keeping the cached panel` | Yahoo returned partial data. The run stops deliberately rather than poisoning the cache. Re-run. |
| Schedule stopped firing | 60 days without a commit. Push anything, or run the workflow manually. |
| Want yesterday's page back | Actions → the previous successful run → re-run the `deploy` job. |
