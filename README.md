# WIDS — Wireless Intrusion Detection with machine learning

Detects three 802.11 attacks (deauthentication flood, rogue access point, website spoofing)
from the supplied AWID3-derived CSV files. Offline analysis only: nothing captures or transmits.

## Files

| File | What it is |
|---|---|
| `wids.py` | Whole pipeline: load → baseline → features → train → alert → validate |
| `detect.py` | **Live demo**: replays a capture second by second and raises ALERTs, logging them to `alerts.log` |
| `figures.py` | Charts, confusion matrices and permutation feature importance (run after `wids.py`) |
| `results/results.json` | Every metric quoted in the report |
| `results/<attack>_detector.joblib` | Trained model + frame threshold + window threshold k |
| `results/<attack>_windows.csv` | Per-second windows: flagged-frame count, ground truth, split |
| `results/<attack>_scores.parquet` | Per-frame scores (for re-checking any number) |
| `results/fig_*.png` | Figures used in the report |

## Reproduce

```bash
pip install pandas numpy scikit-learn matplotlib joblib pyarrow
# put the six CSVs (deauth_train.csv ... website_spoofing_test.csv) in ./data
python wids.py --data data --out results      # ~10 min on 2 CPU cores, < 8 GB RAM
python figures.py --data data --out results   # ~5 min
python detect.py                               # live demo: deauth flood alert at 424 s
```
On Windows use `py` instead of `python`, and install with `py -m pip install ...`.

### Demo options (`detect.py`)
| Command | What it shows |
|---|---|
| `py detect.py` | Replays from 380 s at 10x speed; the alert fires at 424 s (~5 s after start) |
| `py detect.py --held-out --start 0 --speed 50` | Only seconds the model never trained on; prints the confusion matrix + accuracy at the end |
| `py detect.py --speed 2` | Slower, easier to narrate in the video |
| `py detect.py --attack rogue_ap` | The rogue-AP detector |
| `py detect.py --csv some.csv` | Any CSV with the same columns |
Random seed is fixed (42), so the numbers are repeatable.

## Design in one paragraph

The supplied `*_train.csv` / `*_test.csv` are a random split of the same ten-minute capture, so the
two files are re-joined and put back in capture order. Time/sequence columns are never used as features
(each attack starts at a fixed time, so time is a stand-in for the label). The capture is cut into 30-second
blocks: block mod 3 = 0 → fit, 1 → validation (choose thresholds), 2 → test; the first second of each block is
purged. The first 90 s (attack-free) are the site's **baseline**. Each frame gets its own header/radio fields,
**rate/context** over the previous 1 s (deauth count, beacon count, retry fraction, mean signal, deauth count
above the baseline maximum) and **baseline rarity** (how unusual this signal level / frame length is for this
frame type at this site). A gradient-boosted tree classifier (scikit-learn `HistGradientBoostingClassifier`,
class-balanced) scores each frame. **One sample at the alert stage = one 1-second window**: an alert fires
when at least k frames in the window are flagged, with k set so no clean validation window alerts.

## Headline results (held-out 30-s time blocks, one sample = one 1-second window)

| Attack | TP | FP | TN | FN | TPR | FPR | Precision |
|---|---|---|---|---|---|---|---|
| Deauthentication flood | 57 | 0 | 118 | 1 | 98.3% | 0.0% | 100% |
| Rogue AP | 52 | 0 | 125 | 0 | 100% | 0.0% | 100% |
| Website spoofing | 71 | 3 | 98 | 23 | 75.5% | 3.0% | 95.9% |

Read the report before quoting these: rogue AP depends on beacon length (recall falls to 42.7% without it),
and the rogue-AP capture contains ~62k deauth frames labelled "normal".

## Ethics
Supplied dataset only, analysed offline. No capture, no transmission, no automatic containment.
