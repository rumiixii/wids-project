"""Figures + permutation feature importance for the WIDS report.
Run after wids.py:  python figures.py --data <csv folder> --out results"""
import argparse, json, os
import numpy as np, pandas as pd, joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import wids

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#d03b3b"
NAMES = {"deauth": "Deauthentication flood", "rogue_ap": "Rogue AP",
         "website_spoofing": "Website spoofing"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF})


def timeline(out, res):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=False)
    for ax, atk in zip(axes, wids.ATTACKS):
        W = pd.read_csv(os.path.join(out, f"{atk}_windows.csv"), index_col=0)
        W = W[W.role == "test"]
        k = res[atk]["window_k"]
        ax.axvspan(res[atk]["attack_start_s"], W.index.max() + 1, color=ORANGE, alpha=0.12, lw=0)
        ax.vlines(W.index, 0.8, W.n_flag.clip(lower=0.8), color=BLUE, lw=1.4)
        ax.axhline(k, color=RED, lw=1.2, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.8, max(10, W.n_flag.max() * 1.5))
        ax.text(0.005, 0.92, NAMES[atk], transform=ax.transAxes, color=INK, fontsize=11, weight="bold", va="top")
        ax.text(40, k * 1.2, f"alert threshold k = {k}", color=RED, ha="left", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.set_ylabel("flagged frames\nper 1-s window")
    axes[-1].set_xlabel("capture time (s)  - test blocks only; gaps are training/validation blocks")
    fig.suptitle("Flagged frames per 1-second window (blue) vs ground-truth attack period (shaded), held-out time blocks",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_timeline.png"), dpi=160); plt.close(fig)


def leak_chart(out, res):
    rows = [("Supplied random split\n(frame fields only)", "naive_random_split"),
            ("Time column only\n(random split)", "time_only_random_split"),
            ("Leak-free: frame fields only", "ablation_frame_only"),
            ("Leak-free: + rate/context\n+ baseline (final)", "frame_test"),
            ("Leak-free final,\nno frame length", "ablation_no_length")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)
    for ax, atk in zip(axes, wids.ATTACKS):
        y = np.arange(len(rows))
        f1 = [res[atk][key]["F1"] for _, key in rows]
        cols = [INK2 if i < 2 else BLUE for i in range(len(rows))]
        ax.barh(y, f1, color=cols, height=0.6)
        for yi, v in zip(y, f1):
            ax.text(min(v + 0.02, 0.98), yi, f"{v:.3f}", va="center", color=INK, fontsize=9,
                    ha="left" if v < 0.85 else "right", **({"color": SURF} if v >= 0.85 else {}))
        ax.set_yticks(y, [r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_title(NAMES[atk], color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        ax.set_xlabel("frame-level F1 on test data")
    fig.suptitle("Grey = flawed evaluation (leaks time / duplicates). Blue = time-blocked, leak-free evaluation.",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_evaluation_setups.png"), dpi=160); plt.close(fig)


def importance(folder, out):
    imp = {}
    for atk in wids.ATTACKS:
        d = wids.load_capture(folder, atk)
        base = wids.build_baseline(d, (d.t < wids.BASELINE_END_S) & (d.role != "test"))
        d = wids.add_features(d, base)
        det = joblib.load(os.path.join(out, f"{atk}_detector.joblib"))
        te = d[(d.role == "test") & ~d.purged]
        pos, neg = te[te.Label == 1], te[te.Label == 0]
        s = pd.concat([pos.sample(min(len(pos), 20000), random_state=1),
                       neg.sample(min(len(neg), 60000), random_state=1)])
        r = permutation_importance(det["model"], s[det["features"]], s.Label, scoring="average_precision",
                                   n_repeats=3, random_state=1, n_jobs=1)
        imp[atk] = dict(sorted(zip(det["features"], r.importances_mean.round(4).tolist()),
                               key=lambda kv: -kv[1]))
    with open(os.path.join(out, "feature_importance.json"), "w") as f:
        json.dump(imp, f, indent=2)
    imp_plot(out, imp)


def imp_plot(out, imp):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, atk in zip(axes, wids.ATTACKS):
        top = [kv for kv in imp[atk].items() if kv[1] > 0.0005][:8]
        ax.barh([t[0] for t in top][::-1], [t[1] for t in top][::-1], color=BLUE, height=0.6)
        ax.set_title(NAMES[atk], color=INK, fontsize=11, loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.set_xlabel("drop in PR-AUC when shuffled")
    fig.suptitle("Permutation importance (held-out test frames): which evidence the detector relies on",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_importance.png"), dpi=160); plt.close(fig)


def confusion_fig(out, res):
    """2x2 confusion matrices: rows = actual, columns = predicted. One sample = one 1-second window."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#1c5cab"])
    for level, key, unit, fname in [("1-second window", "window_test", "windows", "fig_confusion.png"),
                                    ("frame", "frame_test", "frames", "fig_confusion_frames.png")]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
        for ax, atk in zip(axes, wids.ATTACKS):
            m = res[atk][key]
            M = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
            ax.imshow(np.log10(M + 1), cmap=cmap, vmin=0, vmax=np.log10(M.max() + 1))
            names = [["True negative", "False positive"], ["False negative", "True positive"]]
            for i in range(2):
                for j in range(2):
                    dark = np.log10(M[i, j] + 1) > 0.6 * np.log10(M.max() + 1)
                    ax.text(j, i - 0.12, f"{M[i, j]:,}", ha="center", va="center", fontsize=17, weight="bold",
                            color=SURF if dark else INK)
                    ax.text(j, i + 0.2, names[i][j], ha="center", va="center", fontsize=9,
                            color=SURF if dark else INK2)
            ax.set_xticks([0, 1], ["normal", "attack"]); ax.set_yticks([0, 1], ["normal", "attack"])
            ax.set_xlabel("predicted"); ax.set_ylabel("actual")
            ax.set_title(f"{NAMES[atk]}\naccuracy {m['accuracy']:.2%} | TPR {m['TPR']:.1%} | FPR {m['FPR']:.2%}",
                         color=INK, fontsize=10.5, loc="left")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(length=0)
        fig.suptitle(f"Confusion matrices on held-out test data (one sample = one {level})",
                     color=INK, fontsize=12, x=0.01, ha="left")
        fig.tight_layout(); fig.savefig(os.path.join(out, fname), dpi=160); plt.close(fig)


def poster_figs(out, res):
    """High-resolution deauth figures for the A1 poster."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", "#1c5cab"])
    m = res["deauth"]["window_test"]
    M = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(np.log10(M + 1), cmap=cmap, vmin=0, vmax=np.log10(M.max() + 1))
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    for i in range(2):
        for j in range(2):
            dark = np.log10(M[i, j] + 1) > 0.6 * np.log10(M.max() + 1)
            ax.text(j, i - 0.1, f"{M[i, j]}", ha="center", va="center", fontsize=30, weight="bold",
                    color=SURF if dark else INK)
            ax.text(j, i + 0.24, names[i][j], ha="center", va="center", fontsize=11, color=SURF if dark else INK2)
    ax.set_xticks([0, 1], ["normal", "attack"], fontsize=12); ax.set_yticks([0, 1], ["normal", "attack"], fontsize=12)
    ax.set_xlabel("predicted", fontsize=12); ax.set_ylabel("actual", fontsize=12)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(); fig.savefig(os.path.join(out, "poster_cm_deauth.png"), dpi=300, facecolor="white"); plt.close(fig)

    W = pd.read_csv(os.path.join(out, "deauth_windows.csv"), index_col=0)
    W = W[W.role == "test"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axvspan(res["deauth"]["attack_start_s"], W.index.max() + 1, color=ORANGE, alpha=0.13, lw=0)
    ax.vlines(W.index, 0.8, W.n_flag.clip(lower=0.8), color=BLUE, lw=1.6)
    ax.axhline(res["deauth"]["window_k"], color=RED, lw=1.3, ls="--")
    ax.set_yscale("log"); ax.set_ylim(0.8, 600); ax.set_xlim(30, 610)
    ax.text(40, 1.25, "alert line: 1 flagged frame", color=RED, fontsize=10)
    ax.text(res["deauth"]["attack_start_s"] + 4, 420, "deauth flood (ground truth)", color="#b4471b", fontsize=10, va="top")
    ax.set_xlabel("capture time (s), held-out test seconds only", fontsize=11)
    ax.set_ylabel("flagged frames\nper second", fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_facecolor("white")
    fig.tight_layout(); fig.savefig(os.path.join(out, "poster_timeline_deauth.png"), dpi=300, facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data"); ap.add_argument("--out", default="results")
    ap.add_argument("--recompute", action="store_true"); a = ap.parse_args()
    res = json.load(open(os.path.join(a.out, "results.json")))
    timeline(a.out, res)
    confusion_fig(a.out, res)
    poster_figs(a.out, res)
    if os.path.exists(os.path.join(a.out, "feature_importance.json")) and not a.recompute:
        imp_plot(a.out, json.load(open(os.path.join(a.out, "feature_importance.json"))))
    else:
        importance(a.data, a.out)
    print("figures written")
