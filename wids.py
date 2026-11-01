"""
WIDS - Wireless Intrusion Detection System trained on supplied 802.11 frame data
(AWID3-derived CSVs: deauthentication, rogue AP, website spoofing).

Pipeline (lecture / Lab 3):  BASELINE -> OBSERVE -> EXTRACT FEATURES -> ALERT -> VALIDATE

Run:  python wids.py --data <folder with the six CSVs> --out results
Ethics: offline analysis of a supplied dataset only. Nothing here captures or transmits.
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import joblib

# ----------------------------------------------------------------------------- config
ATTACKS = ["deauth", "rogue_ap", "website_spoofing"]
SEED = 42
BLOCK_S = 30.0        # time-blocked split: 30-second blocks
PURGE_S = 1.0         # drop the first second of every block (context window crosses the boundary)
CTX_S = 1.0           # look-back window for rate/context features (causal: past only)
WIN_S = 1.0           # ONE SAMPLE at the alert stage = one 1-second window
BASELINE_END_S = 90.0 # first 90 s of every capture are attack-free -> "normal" baseline

# Columns that identify WHEN a frame was captured. The attack in every capture
# starts after ~200-420 s, so these columns are a proxy for the label, not a
# behaviour. They are used only to order frames / build windows, never as features.
LEAKY = ["frame.number", "frame.time_epoch", "frame.time_relative",
         "radiotap.timestamp.ts", "frame.time_delta_displayed"]

# Per-frame header / radio features available in the dataset (no MACs, SSIDs,
# BSSIDs or reason codes exist in this data -> see limitations).
FRAME_FEATS = ["frame.len", "frame.time_delta", "radiotap.length", "wlan.duration",
               "wlan.fc.ds", "wlan.fc.frag", "wlan.fc.order", "wlan.fc.moredata",
               "wlan.fc.protected", "wlan.fc.pwrmgt", "wlan.fc.type", "wlan.fc.retry",
               "wlan.fc.subtype", "wlan_radio.duration", "wlan_radio.data_rate",
               "wlan_radio.signal_dbm", "wlan_radio.phy"]

CTX_FEATS = ["ctx_frames", "ctx_deauth", "ctx_disassoc", "ctx_beacon", "ctx_mgmt_frac",
             "ctx_retry_frac", "ctx_deauth_frac", "ctx_mean_signal", "ctx_deauth_excess"]
BASE_FEATS = ["sig_rarity", "profile_rarity"]
ALL_FEATS = FRAME_FEATS + CTX_FEATS + BASE_FEATS


# ----------------------------------------------------------------------------- data
def load_capture(folder, attack):
    """Re-join the supplied train/test files (a random split of ONE capture) and
    restore capture order. We keep which file each frame came from only to
    reproduce the naive split for comparison."""
    parts = []
    for split in ["train", "test"]:
        d = pd.read_csv(os.path.join(folder, f"{attack}_{split}.csv"))
        d["orig_split"] = split
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).sort_values("frame.number").reset_index(drop=True)
    d["t"] = d["frame.time_relative"].astype(float)
    d["t"] = np.maximum.accumulate(d["t"].values)     # guarantee monotone time
    d["block"] = (d["t"] // BLOCK_S).astype(int)
    d["role"] = np.select([d.block % 3 == 0, d.block % 3 == 1], ["fit", "val"], "test")
    d["purged"] = (d["t"] % BLOCK_S) < PURGE_S
    d["window"] = (d["t"] // WIN_S).astype(int)
    return d


# ----------------------------------------------------------------------------- baseline
def build_baseline(d, mask):
    """Profile of NORMAL behaviour from the attack-free start of the capture.
    Uses frames only (no labels): this is what a sensor learns at install time."""
    b = d[mask]
    is_deauth = (b["wlan.fc.type"] == 0) & (b["wlan.fc.subtype"].isin([10, 12]))
    per_sec = is_deauth.groupby(b["window"]).sum()
    base = {
        "deauth_per_s_max": int(per_sec.max()) if len(per_sec) else 0,
        "deauth_per_s_p99": float(per_sec.quantile(0.99)) if len(per_sec) else 0.0,
        # how often each (type, subtype, signal) / (type, subtype, length) was seen
        "sig": b.groupby(["wlan.fc.type", "wlan.fc.subtype", "wlan_radio.signal_dbm"]).size(),
        "sig_tot": b.groupby(["wlan.fc.type", "wlan.fc.subtype"]).size(),
        "len": b.groupby(["wlan.fc.type", "wlan.fc.subtype", "frame.len"]).size(),
        "n_frames": int(len(b)),
    }
    return base


def _rarity(d, counts, totals, keycols):
    """-log P(value | frame type) with add-one smoothing: high = never seen in baseline."""
    idx = pd.MultiIndex.from_frame(d[keycols])
    c = counts.reindex(idx).fillna(0).values
    tot = totals.reindex(pd.MultiIndex.from_frame(d[keycols[:2]])).fillna(0).values
    return -np.log((c + 1.0) / (tot + 50.0))


# ----------------------------------------------------------------------------- features
def add_features(d, base):
    t = d["t"].values
    left = np.searchsorted(t, t - CTX_S, side="left")
    n = np.arange(len(d)) + 1 - left                      # frames in the last CTX_S seconds

    def roll(x):                                           # causal rolling sum (past only)
        cs = np.concatenate([[0.0], np.cumsum(x, dtype=float)])
        return cs[np.arange(len(x)) + 1] - cs[left]

    ty, st = d["wlan.fc.type"].values, d["wlan.fc.subtype"].values
    deauth = ((ty == 0) & (st == 12)).astype(float)
    disas = ((ty == 0) & (st == 10)).astype(float)
    d["ctx_frames"] = n
    d["ctx_deauth"] = roll(deauth)
    d["ctx_disassoc"] = roll(disas)
    d["ctx_beacon"] = roll(((ty == 0) & (st == 8)).astype(float))
    d["ctx_mgmt_frac"] = roll((ty == 0).astype(float)) / n
    d["ctx_retry_frac"] = roll(d["wlan.fc.retry"].values.astype(float)) / n
    d["ctx_deauth_frac"] = (d["ctx_deauth"] + d["ctx_disassoc"]) / n
    d["ctx_mean_signal"] = roll(d["wlan_radio.signal_dbm"].values.astype(float)) / n
    # rate compared with the site's own baseline (not a textbook number)
    d["ctx_deauth_excess"] = d["ctx_deauth"] + d["ctx_disassoc"] - base["deauth_per_s_max"]
    # how unusual is this frame's signal level / length for its frame type at this site?
    d["sig_rarity"] = _rarity(d, base["sig"], base["sig_tot"],
                              ["wlan.fc.type", "wlan.fc.subtype", "wlan_radio.signal_dbm"])
    pr = _rarity(d, base["len"], base["sig_tot"], ["wlan.fc.type", "wlan.fc.subtype", "frame.len"])
    d["profile_rarity"] = np.where(ty == 2, 0.0, pr)   # inventory proxy: mgmt/control only
    return d


# ----------------------------------------------------------------------------- metrics
def confusion(y, p):
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    tpr = tp / (tp + fn) if tp + fn else float("nan")
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) and not np.isnan(prec) else 0.0
    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, TPR=tpr, FPR=fpr, precision=prec, F1=f1, accuracy=acc)


def best_threshold(y, s):
    """Threshold that maximises F1 on VALIDATION data (never on test)."""
    best, thr = -1, 0.5
    for q in np.unique(np.quantile(s, np.linspace(0.5, 0.99999, 400))):
        m = confusion(y, s >= q)
        if m["F1"] > best:
            best, thr = m["F1"], float(q)
    return thr


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=SEED)


def fit_detector(d, feats):
    """fit on 'fit' blocks -> choose threshold on 'val' blocks -> refit on fit+val."""
    keep = ~d.purged
    fit, val = keep & (d.role == "fit"), keep & (d.role == "val")
    m = make_model().fit(d.loc[fit, feats], d.loc[fit, "Label"])
    thr = best_threshold(d.loc[val, "Label"].values, m.predict_proba(d.loc[val, feats])[:, 1])
    final = make_model().fit(d.loc[fit | val, feats], d.loc[fit | val, "Label"])
    return final, thr


# ----------------------------------------------------------------------------- alert stage
def window_table(d, flagged):
    """Aggregate frame decisions into 1-second windows (one window = one sample)."""
    w = pd.DataFrame({"window": d["window"].values, "flag": flagged.astype(int),
                      "attack": d["Label"].values, "role": d["role"].values,
                      "purged": d["purged"].values})
    w = w[~w.purged]
    g = w.groupby("window").agg(n_flag=("flag", "sum"), n_attack=("attack", "sum"),
                                role=("role", "first"))
    g["y"] = (g.n_attack > 0).astype(int)
    return g


def choose_k(val_windows):
    """Smallest flagged-frame count k that raises NO alert on clean validation windows."""
    clean = val_windows[val_windows.y == 0].n_flag
    return int(clean.max()) + 1 if len(clean) else 1


# ----------------------------------------------------------------------------- main
def run(folder, out):
    os.makedirs(out, exist_ok=True)
    results, models = {}, {}
    for atk in ATTACKS:
        rpath = os.path.join(out, f"results_{atk}.json")
        if os.path.exists(rpath):                         # already trained: reuse (saves time)
            results[atk] = json.load(open(rpath))
            det = joblib.load(os.path.join(out, f"{atk}_detector.joblib"))
            models[atk] = (det["model"], det["threshold"], det["k"])
            continue
        t0 = time.time()
        d = load_capture(folder, atk)
        base = build_baseline(d, (d.t < BASELINE_END_S) & (d.role != "test"))
        d = add_features(d, base)
        R = {"rows": int(len(d)), "attack_frames": int(d.Label.sum()),
             "attack_start_s": float(d.loc[d.Label == 1, "t"].min()),
             "baseline": {"deauth_per_s_max": base["deauth_per_s_max"],
                          "deauth_per_s_p99": base["deauth_per_s_p99"],
                          "frames": base["n_frames"]},
             "duplicate_rows_pct": float(d[FRAME_FEATS + ["Label"]].duplicated().mean() * 100)}

        test = (d.role == "test") & ~d.purged
        # --- 1. main detector
        model, thr = fit_detector(d, ALL_FEATS)
        s = model.predict_proba(d[ALL_FEATS])[:, 1]
        d["score"] = s
        flag = s >= thr
        R["frame_threshold"] = thr
        R["frame_test"] = confusion(d.loc[test, "Label"], flag[test.values])
        R["frame_test"]["PR_AUC"] = float(average_precision_score(d.loc[test, "Label"], s[test.values]))
        W = window_table(d, flag)
        k = choose_k(W[W.role == "val"])
        W["alert"] = (W.n_flag >= k).astype(int)
        Wt = W[W.role == "test"]
        R["window_k"] = k
        R["window_test"] = confusion(Wt.y, Wt.alert)
        R["window_test"]["n_windows"] = int(len(Wt))
        # first alert after attack start (detection delay, test blocks)
        W.to_csv(os.path.join(out, f"{atk}_windows.csv"))
        models[atk] = (model, thr, k)
        joblib.dump({"model": model, "threshold": thr, "k": k, "features": ALL_FEATS},
                    os.path.join(out, f"{atk}_detector.joblib"))

        # --- 2. ablation: without frame.len / profile_rarity (attacker controls frame length)
        abl = [f for f in ALL_FEATS if f not in ("frame.len", "profile_rarity")]
        m2, thr2 = fit_detector(d, abl)
        s2 = m2.predict_proba(d.loc[test, abl])[:, 1]
        R["ablation_no_length"] = confusion(d.loc[test, "Label"], s2 >= thr2)
        R["ablation_no_length"]["PR_AUC"] = float(average_precision_score(d.loc[test, "Label"], s2))

        # --- 3. ablation: per-frame header fields only (no rate/context, no baseline)
        m3, thr3 = fit_detector(d, FRAME_FEATS)
        s3 = m3.predict_proba(d.loc[test, FRAME_FEATS])[:, 1]
        R["ablation_frame_only"] = confusion(d.loc[test, "Label"], s3 >= thr3)
        R["ablation_frame_only"]["PR_AUC"] = float(average_precision_score(d.loc[test, "Label"], s3))

        # --- 4. the traps: supplied random split, and a time-only "model"
        tr, te = d.orig_split == "train", d.orig_split == "test"
        mn = make_model().fit(d.loc[tr, FRAME_FEATS], d.loc[tr, "Label"])
        R["naive_random_split"] = confusion(d.loc[te, "Label"], mn.predict(d.loc[te, FRAME_FEATS]))
        h = pd.util.hash_pandas_object(d[FRAME_FEATS], index=False).values   # row fingerprint
        R["naive_test_rows_also_in_train_pct"] = float(np.isin(h[te.values], h[tr.values]).mean() * 100)
        mt = make_model().fit(d.loc[tr, ["frame.time_relative"]], d.loc[tr, "Label"])
        R["time_only_random_split"] = confusion(d.loc[te, "Label"],
                                                mt.predict(d.loc[te, ["frame.time_relative"]]))

        # --- 5. rule baselines (signatures) at window level, test blocks
        dd = d[~d.purged]
        is_de = ((dd["wlan.fc.type"] == 0) & dd["wlan.fc.subtype"].isin([10, 12])).astype(int)
        per_w = is_de.groupby(dd.window).sum().reindex(W.index).fillna(0)
        rules = {"any_deauth_in_window": per_w >= 1,
                 "deauth_rate_above_baseline_max": per_w > base["deauth_per_s_max"]}
        is_rb = ((dd["wlan.fc.type"] == 0) & (dd["wlan.fc.subtype"] == 8) &
                 (dd["profile_rarity"] > 5)).astype(int)
        rules["beacon_profile_not_in_baseline"] = is_rb.groupby(dd.window).sum().reindex(W.index).fillna(0) >= 1
        R["rules_window_test"] = {name: confusion(Wt.y, r[W.role == "test"]) for name, r in rules.items()}

        results[atk] = R
        with open(rpath, "w") as f:
            json.dump(R, f, indent=2, default=float)
        d[["t", "window", "role", "purged", "Label", "score"]].to_parquet(os.path.join(out, f"{atk}_scores.parquet"))
        print(f"[{atk}] done in {time.time()-t0:.0f}s  frame={R['frame_test']}  window={R['window_test']}", flush=True)
        del d

    # --- 6. cross-capture clean test: run each detector on the OTHER captures.
    # Sensor re-baselines on site (first 90 s); we score attack-free windows 90 s .. 190 s
    # and report false alerts. Also report what the deauth detector says about the
    # rogue-AP capture, whose ~62k deauth frames are labelled 'normal'.
    cross = {}
    for dst in ATTACKS:                                   # one capture in memory at a time
        dd = load_capture(folder, dst)
        dd = add_features(dd, build_baseline(dd, dd.t < BASELINE_END_S))
        for src, (model, thr, k) in models.items():
            if dst == src:
                continue
            flag = model.predict_proba(dd[ALL_FEATS])[:, 1] >= thr
            per_w = pd.Series(flag.astype(int)).groupby(dd.window.values).sum()
            clean = per_w[(per_w.index >= BASELINE_END_S) & (per_w.index < 190)]
            late = per_w[per_w.index >= 430]
            cross[f"{src}_on_{dst}"] = {
                "clean_windows": int(len(clean)), "false_alerts_clean": int((clean >= k).sum()),
                "alerts_after_430s": int((late >= k).sum()), "windows_after_430s": int(len(late))}
        del dd
    results["cross_capture"] = cross
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print_summary(results)
    return results


def print_summary(results):
    """Readable results table in the terminal."""
    pct = lambda v: "  n/a " if v != v else f"{v*100:6.2f}%"
    for level, key, unit in [("FRAME level (one sample = one frame)", "frame_test", "frames"),
                             ("ALERT level (one sample = one 1-second window)", "window_test", "windows")]:
        print(f"\n=== {level}, held-out test blocks ===")
        print(f"{'Attack':<18}{'TP':>9}{'FP':>8}{'TN':>9}{'FN':>8}{'TPR':>9}{'FPR':>9}{'Precision':>11}")
        for atk in ATTACKS:
            m = results[atk][key]
            print(f"{atk:<18}{m['TP']:>9}{m['FP']:>8}{m['TN']:>9}{m['FN']:>8}"
                  f"{pct(m['TPR']):>9}{pct(m['FPR']):>9}{pct(m['precision']):>11}")
    print("\nFull details: results/results.json   Charts: run  python figures.py --data data --out results")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    run(a.data, a.out)
