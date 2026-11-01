"""
detect.py - live-replay demo of the WIDS.

Replays a capture second by second as if it were arriving from a monitor-mode sensor,
scores every frame with the trained detector, and raises an ALERT when an attack starts.
Every alerting second is written to alerts.log.

Examples
  py detect.py                          # deauth demo: replay from 380 s at 10x speed
  py detect.py --start 0 --speed 50     # whole capture, faster
  py detect.py --held-out               # only seconds the model never trained on
  py detect.py --attack rogue_ap        # the other detectors work too
  py detect.py --csv mytraffic.csv      # any CSV with the same columns (Label optional)

Run  py wids.py --data data --out results  first, so the trained detector exists.
Offline replay of supplied data only: nothing is captured or transmitted.
"""
import argparse, os, sys, time, datetime
import numpy as np, pandas as pd, joblib
import wids

NAMES = {"deauth": "DEAUTHENTICATION FLOOD", "rogue_ap": "ROGUE ACCESS POINT",
         "website_spoofing": "WEBSITE SPOOFING"}

# terminal colours (works in VS Code / Windows Terminal / Linux / macOS)
os.system("")
RED, GRN, YEL, DIM, BOLD, END = "\033[91m", "\033[92m", "\033[93m", "\033[2m", "\033[1m", "\033[0m"


def load_frames(files):
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d.sort_values("frame.number").reset_index(drop=True)
    d["t"] = np.maximum.accumulate(d["frame.time_relative"].astype(float).values)
    d["block"] = (d["t"] // wids.BLOCK_S).astype(int)
    d["role"] = np.select([d.block % 3 == 0, d.block % 3 == 1], ["fit", "val"], "test")
    d["purged"] = (d["t"] % wids.BLOCK_S) < wids.PURGE_S
    d["window"] = (d["t"] // wids.WIN_S).astype(int)
    return d


def main():
    ap = argparse.ArgumentParser(description="WIDS live-replay demo")
    ap.add_argument("--attack", default="deauth", choices=wids.ATTACKS)
    ap.add_argument("--data", default="data", help="folder with the supplied CSVs")
    ap.add_argument("--csv", nargs="*", help="replay these CSV file(s) instead")
    ap.add_argument("--models", default="results", help="folder with <attack>_detector.joblib")
    ap.add_argument("--start", type=float, default=380, help="capture second to start replay from")
    ap.add_argument("--speed", type=float, default=10, help="capture seconds per real second (0 = no delay)")
    ap.add_argument("--held-out", action="store_true", help="replay only held-out test seconds")
    ap.add_argument("--log", default="alerts.log")
    a = ap.parse_args()

    det_path = os.path.join(a.models, f"{a.attack}_detector.joblib")
    if not os.path.exists(det_path):
        sys.exit(f"No trained detector at {det_path}. Run:  py wids.py --data data --out results")
    det = joblib.load(det_path)
    model, thr, k, feats = det["model"], det["threshold"], det["k"], det["features"]

    files = a.csv or [os.path.join(a.data, f"{a.attack}_{s}.csv") for s in ("train", "test")]
    print(f"{BOLD}WIDS sensor (replay mode){END}  detector: {NAMES[a.attack]}")
    print(f"Loading {', '.join(os.path.basename(f) for f in files)} ...", flush=True)
    d = load_frames(files)

    # 1. BASELINE: learn this site's normal behaviour from the first 90 s
    base = wids.build_baseline(d, d.t < wids.BASELINE_END_S)
    print(f"Baseline learnt from first {wids.BASELINE_END_S:.0f} s: {base['n_frames']:,} frames, "
          f"normal max {base['deauth_per_s_max']} deauth/disassoc frames per second")
    # 2-3. OBSERVE + FEATURES, 4. SCORE every frame
    d = wids.add_features(d, base)
    d["flag"] = model.predict_proba(d[feats])[:, 1] >= thr
    d["is_deauth"] = (d["wlan.fc.type"] == 0) & d["wlan.fc.subtype"].isin([10, 12])
    has_truth = "Label" in d.columns
    print(f"Alert rule: a 1-second window raises an alert when >= {k} frames are flagged "
          f"(frame score >= {thr:.6f})\n")

    sec = d.groupby("window").agg(frames=("flag", "size"), flagged=("flag", "sum"),
                                  deauth=("is_deauth", "sum"),
                                  truth=("Label", "max") if has_truth else ("flag", "size"),
                                  role=("role", "first"), purged=("purged", "all"))
    sig = d[d.flag].groupby("window")["wlan_radio.signal_dbm"].median()
    sec = sec[sec.index >= int(a.start)]
    if a.held_out:
        sec = sec[(sec.role == "test") & ~sec.purged]
        print(f"{DIM}Held-out mode: only seconds from test blocks (never used for training).{END}\n")

    new_log = not os.path.exists(a.log)
    log = open(a.log, "a", encoding="utf-8")
    if new_log:
        log.write("wall_clock,capture_second,attack,flagged_frames,deauth_frames,median_signal_dbm\n")

    print(f"{'time':>7}  {'frames':>6}  {'deauth':>6}  {'flagged':>7}  status")
    in_alert, alert_start, alert_secs, y_true, y_pred = False, None, 0, [], []
    for w, r in sec.iterrows():
        alert = r.flagged >= k
        y_pred.append(int(alert))
        if has_truth:
            y_true.append(int(r.truth > 0))
        truth = ""
        if has_truth:
            truth = f"{DIM}  [truth: {'attack' if r.truth > 0 else 'normal'}]{END}"
        if alert and not in_alert:
            in_alert, alert_start = True, w
            print(f"\n{RED}{BOLD}{'!' * 60}\n  ALERT  {NAMES[a.attack]} DETECTED at {w} s\n"
                  f"  {int(r.flagged)} suspicious frames this second "
                  f"(normal max: {base['deauth_per_s_max']} deauth/s)\n"
                  f"  transmitter signal ~{sig.get(w, float('nan')):.0f} dBm\n{'!' * 60}{END}")
        elif not alert and in_alert:
            in_alert = False
            print(f"{YEL}{BOLD}  Attack activity stopped at {w} s (lasted {w - alert_start} s){END}\n")
        if alert:
            alert_secs += 1
            log.write(f"{datetime.datetime.now().isoformat(timespec='seconds')},{w},{a.attack},"
                      f"{int(r.flagged)},{int(r.deauth)},{sig.get(w, float('nan')):.0f}\n")
            status = f"{RED}ALERT{END}"
        else:
            status = f"{GRN}normal{END}"
        print(f"{w:>6}s  {int(r.frames):>6}  {int(r.deauth):>6}  {int(r.flagged):>7}  {status}{truth}", flush=True)
        if a.speed > 0:
            time.sleep(1.0 / a.speed)
    log.close()

    print(f"\n{BOLD}Replay finished:{END} {len(sec)} seconds, {alert_secs} alert seconds, logged to {a.log}")
    if has_truth and y_true:
        yt, yp = np.array(y_true, bool), np.array(y_pred, bool)
        tp, fp = int((yt & yp).sum()), int((~yt & yp).sum())
        fn, tn = int((yt & ~yp).sum()), int((~yt & ~yp).sum())
        print(f"\nConfusion matrix for the replayed seconds (one sample = one 1-second window):")
        print(f"                   predicted normal   predicted attack")
        print(f"  actual normal    {tn:>16}   {fp:>16}")
        print(f"  actual attack    {fn:>16}   {tp:>16}")
        print(f"  accuracy {(tp + tn) / len(yt):.2%}   TPR {tp / max(1, tp + fn):.2%}   "
              f"FPR {fp / max(1, fp + tn):.2%}")
        if not a.held_out:
            print(f"{DIM}  Note: this replay includes seconds used for training. "
                  f"Use --held-out for an honest test.{END}")


if __name__ == "__main__":
    main()
