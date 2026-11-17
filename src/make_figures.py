"""Step 5: figures and aggregate tables from the outputs of steps 1-4.

Reads per-episode / per-window tables from data/derived (local) and writes only
aggregate figures and tables to outputs/ (safe to publish).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import DETECTORS, JUMP_M, setup_stdout  # noqa: E402

COND_LABELS = {
    "full": "all train (100%)",
    "random_half": "random 50%",
    "top_half": "cleanest 50%",
    "bottom_half": "noisiest 50%",
    "random_half_inject10": "random 50% + 10% corrupted",
    "random_half_inject30": "random 50% + 30% corrupted",
}


def fig_quality(ep: pd.DataFrame, summary: dict, findings: dict, path: Path) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    fv = findings["wrist_conf0_vs_view"]
    cats = [("conf_pos_in_view", "in view, tracked", "#4C72B0"), ("conf0_in_view", "in view, lost (conf 0)", "#C44E52"),
            ("conf_pos_out_of_view", "out of view, tracked", "#8DA0CB"), ("conf0_out_of_view", "out of view, conf 0", "#DD8452")]
    edge = np.zeros(2)
    for key, lab, col in cats:
        vals = np.array([100.0 * fv[s][key] / fv[s]["frames"] for s in ("left", "right")])
        ax[0, 0].barh([0, 1], vals, left=edge, color=col, label=lab)
        for i, v in enumerate(vals):
            if v > 4:
                ax[0, 0].text(edge[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        edge += vals
    lost = [100.0 * fv[s]["conf0_in_view"] / fv[s]["frames"] for s in ("left", "right")]
    ax[0, 0].set_yticks([0, 1])
    ax[0, 0].set_yticklabels(["left wrist", "right wrist"])
    ax[0, 0].set_xlim(0, 100)
    ax[0, 0].set_xlabel(f"% of frames (episodes with confidences); in view but lost: "
                        f"left {lost[0]:.2f}%, right {lost[1]:.2f}%", fontsize=8)
    ax[0, 0].set_title("Is the wrist inside the RGB view, and is it tracked?")
    ax[0, 0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    jit = np.concatenate([ep["left_jitter_mm"], ep["right_jitter_mm"]])
    ax[0, 1].hist(np.clip(jit, 0, np.nanpercentile(jit, 99.5)), bins=60, color="#55A868")
    ax[0, 1].set_title("Wrist jitter per episode\n(RMS distance to smoothed trajectory)")
    ax[0, 1].set_xlabel("mm (clipped at p99.5)")
    ax[0, 1].set_ylabel("hands")
    spd = np.concatenate([ep["left_max_speed_mps"], ep["right_max_speed_mps"]])
    ax[1, 0].hist(np.clip(spd, 0, 10), bins=60, color="#C44E52")
    ax[1, 0].axvline(JUMP_M * 30, color="k", ls="--", lw=1)
    ax[1, 0].text(JUMP_M * 30 + 0.1, 0.78, "jump rule\n(10 cm/frame)", fontsize=8,
                  transform=ax[1, 0].get_xaxis_transform())
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_title("Max wrist speed per episode")
    ax[1, 0].set_xlabel("m/s (clipped at 10)")
    ax[1, 0].set_ylabel("hands (log scale)")
    flags = {**summary["episode_flags_pct_of_all_episodes"], **summary["episode_flags_pct_of_episodes_with_confidences"]}
    # A fixed 0.5 bar on hand-mean confidence flags most episodes (ARKit finger confidences sit near 0.5),
    # so that rule is uncalibrated and left out of the chart; it stays in quality_summary.json.
    flags.pop("hand_lowconf_gt_10pct_frames", None)
    names = list(flags)
    ax[1, 1].barh(range(len(names)), [flags[n] for n in names], color="#8172B2")
    ax[1, 1].set_yticks(range(len(names)))
    ax[1, 1].set_yticklabels([n.replace("_", " ") for n in names], fontsize=8)
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlabel("% of episodes")
    ax[1, 1].set_title("Episodes flagged by each rule")
    for i, n in enumerate(names):
        ax[1, 1].text(flags[n] + 0.3, i, f"{flags[n]:.1f}", va="center", fontsize=8)
    fig.suptitle("EgoDex test set: hand-tracking health check", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tasks(tv: pd.DataFrame, findings: dict, path: Path) -> None:
    t = tv[tv["episodes"] >= 5]
    fv = findings["wrist_conf0_vs_view"]
    frames = sum(fv[s]["frames"] for s in ("left", "right"))
    out_all = 100.0 * sum(fv[s]["conf0_out_of_view"] + fv[s]["conf_pos_out_of_view"] for s in ("left", "right")) / frames
    vis = sum(fv[s]["conf0_in_view"] + fv[s]["conf_pos_in_view"] for s in ("left", "right"))
    lost_all = 100.0 * sum(fv[s]["conf0_in_view"] for s in ("left", "right")) / vis
    panels = (
        ("out_of_view_pct", out_all, "#8DA0CB",
         "Wrist outside the RGB image\n(% of hand-frames; camera framing, not a tracker failure)"),
        ("in_view_lost_pct_of_in_view", lost_all, "#C44E52",
         "Wrist inside the image area but not tracked (confidence 0)\n(% of in-image hand-frames; occlusion or tracking loss)"),
    )
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    for k, (col, overall, color, title) in enumerate(panels):
        top = t.sort_values(col, ascending=False).head(15)
        ax[k].barh(range(len(top)), top[col], color=color)
        ax[k].set_yticks(range(len(top)))
        ax[k].set_yticklabels([f"{a} (n={b})" for a, b in zip(top["task"], top["episodes"])], fontsize=8)
        ax[k].invert_yaxis()
        ax[k].axvline(overall, color="k", ls="--", lw=1, label=f"all tasks: {overall:.2f}%")
        for i, v in enumerate(top[col]):
            ax[k].text(v, i, f" {v:.1f}", va="center", fontsize=7)
        ax[k].set_xlabel("%")
        ax[k].set_title(title, fontsize=10)
        ax[k].legend(loc="lower right", fontsize=8)
    fig.suptitle("Top 15 tasks (tasks with >= 5 episodes that have confidences; n = episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_injection(sm: pd.DataFrame, path: Path) -> None:
    s = sm.copy()
    s["label"] = [f"{d} {m}".strip() for d, m in zip(s["defect"], s["magnitude"].astype(str))]
    cols = [f"{d}_rate" for d in DETECTORS] + ["any_rate"]
    mat = s[cols].astype(float).to_numpy() * 100
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(s) + 1.8))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_rate", "") for c in cols], rotation=20)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["label"], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i, j] > 60 else "black")
    ax.set_title("Synthetic defects caught by each rule (% of injected windows)\n"
                 "threshold = 95th percentile on real windows; units: jitter sigma mm, spike mm, freeze frames, drift mm")
    fig.colorbar(im, ax=ax, fraction=0.03, label="% detected")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def stat(runs: pd.DataFrame, exp: str, cond: str, col: str) -> tuple[float, float, int]:
    v = runs[(runs["experiment"] == exp) & (runs["condition"] == cond)][col]
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, int(len(v))


def paired_delta(runs: pd.DataFrame, a: tuple, b: tuple, col: str) -> tuple[float, float]:
    """Mean and std over seeds of (run a - run b) for the same seed."""
    ra = runs[(runs["experiment"] == a[0]) & (runs["condition"] == a[1])].set_index("seed")[col]
    rb = runs[(runs["experiment"] == b[0]) & (runs["condition"] == b[1])].set_index("seed")[col]
    d = (ra - rb).dropna()
    return float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else 0.0


def fig_predictor(runs: pd.DataFrame, meta: dict, path: Path) -> None:
    conds = list(COND_LABELS)
    x = np.arange(len(conds))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    for k, (col, title) in enumerate((("avg_mm", "all test samples"), ("clean_avg_mm", "cleanest half of test samples"))):
        m = [stat(runs, "quality", c, col) for c in conds]
        ax[k].bar(x, [a for a, _, _ in m], yerr=[s for _, s, _ in m], color="#4C72B0", capsize=4)
        base = meta["baselines_test"]["zero_velocity"]["avg_mm" if col == "avg_mm" else "clean_avg_mm"]
        ax[k].axhline(base, color="#C44E52", ls="--", lw=1.2, label=f"'hand stays still' baseline: {base:.1f}")
        lo = min(min(a - s for a, s, _ in m), base)
        hi = max(max(a + s for a, s, _ in m), base)
        ax[k].set_ylim(lo - (hi - lo) * 0.25, hi + (hi - lo) * 0.15)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels([COND_LABELS[c] for c in conds], rotation=25, ha="right", fontsize=8)
        ax[k].set_ylabel("mean L2 error over the next 1 s (mm)")
        ax[k].set_title(f"Test error: {title}")
        ax[k].legend(fontsize=8, loc="upper left")
        for i, (a, s, _) in enumerate(m):
            ax[k].text(i, a + s + (hi - lo) * 0.01, f"{a:.1f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("Hand-motion predictor (joints only, MLP): training-data quality (mean +- std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_quantity(runs: pd.DataFrame, path: Path) -> None:
    rand = [("quantity", "random_eighth"), ("quantity", "random_quarter"), ("quality", "random_half"), ("quality", "full")]
    top = [("quantity", "top_quarter"), ("quality", "top_half")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, col, title in ((axes[0], "avg_mm", "all test samples"), (axes[1], "clean_avg_mm", "cleanest half of test samples")):
        for pts, color, label in ((rand, "#4C72B0", "random subset"), (top, "#DD8452", "cleanest subset (quality score)")):
            xs = [stat(runs, e, c, "n_train")[0] for e, c in pts]
            ms = [stat(runs, e, c, col) for e, c in pts]
            ax.errorbar(xs, [a for a, _, _ in ms], yerr=[s for _, s, _ in ms], marker="o", color=color, capsize=4, label=label)
            dy = 6 if color == "#4C72B0" else -14
            for xv, (a, _, _) in zip(xs, ms):
                ax.annotate(f"{a:.1f}", (xv, a), textcoords="offset points", xytext=(6, dy), fontsize=8, color=color)
        b = stat(runs, "quality", "bottom_half", col)
        bx = stat(runs, "quality", "bottom_half", "n_train")[0]
        ax.errorbar([bx], [b[0]], yerr=[b[1]], marker="s", color="#C44E52", capsize=4, ls="none", label="noisiest half (quality score)")
        ax.annotate(f"{b[0]:.1f}", (bx, b[0]), textcoords="offset points", xytext=(6, 6), fontsize=8, color="#C44E52")
        ax.set_xscale("log")
        ax.set_xlabel("training samples (log scale)")
        ax.set_ylabel("mean L2 error over the next 1 s (mm)")
        ax.set_title(f"Test error: {title}")
        ax.legend(fontsize=8)
    fig.suptitle("More data or cleaner data? (mean +- std over 3 seeds; the quality score is confounded with speed, see fig 8)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_defect_harm(runs: pd.DataFrame, inj: pd.DataFrame, path: Path) -> None:
    """x: share caught by the step-3 rules (same defect size); y: extra test error when 30% of training samples have it."""
    match = {"jitter": ("jitter", "10"), "spike": ("spike", "100"), "freeze": ("freeze", "15"),
             "swap": ("swap", "gap>=10cm"), "drift": ("drift", "100")}
    labels = {"jitter": "jitter 10 mm", "spike": "spike 10 cm", "freeze": "freeze 15 frames",
              "swap": "left/right swap", "drift": "drift 10 cm / 2 s"}
    rows = []
    for d, (kind, mag) in match.items():
        r = inj[(inj["defect"] == kind) & (inj["magnitude"].astype(str) == mag)]
        caught = float(r["any_rate"].iloc[0]) * 100 if len(r) else np.nan
        dm, ds = paired_delta(runs, ("defect_type", f"{d}30"), ("quality", "random_half"), "avg_mm")
        rows.append((labels[d], dm, ds, caught))
    rows.sort(key=lambda r: r[1], reverse=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(rows))
    colors = ["#4C72B0" if c >= 50 else "#C44E52" for *_, c in rows]
    ax.barh(y, [r[1] for r in rows], xerr=[r[2] for r in rows], color=colors, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    hi = max(r[1] for r in rows)
    for i, (_, dm, _, c) in enumerate(rows):
        dtxt = f"{dm:+.1f}" if abs(dm) >= 1 else f"{dm:+.2f}"
        ctxt = f"{c:.0f}%" if c >= 1 else f"{c:.1f}%"
        ax.text(max(dm, 0) + hi * 0.03, i, f"{dtxt} mm   (caught by the rules: {ctxt})", va="center", fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlim(min(-1.0, min(r[1] for r in rows) - 0.5), hi * 1.75)
    ax.set_xlabel("extra test error vs. the clean random half (mm, all test samples, paired by seed, mean +- std)")
    ax.set_title("30% of training samples carry one defect: harm to the predictor vs. detectability\n"
                 "(blue: caught by the step-3 rules; red: missed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


V2_LABELS = {
    "random_half": "random",
    "top_half": "cleanest\n(score)",
    "bottom_half": "noisiest\n(score)",
    "top_half_speed_matched": "cleanest\n(matched)",
    "bottom_half_speed_matched": "noisiest\n(matched)",
    "conf_top_half": "high\nconfidence",
    "conf_bottom_half": "low\nconfidence",
}


def fig_quality_v2(runs: pd.DataFrame, meta: dict, path: Path) -> None:
    conds = [c for c in V2_LABELS if len(runs[(runs["experiment"] == "quality_v2") & (runs["condition"] == c)])]
    if not conds:
        return
    x = np.arange(len(conds))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    m = [stat(runs, "quality_v2", c, "avg_mm") for c in conds]
    sp = [stat(runs, "quality_v2", c, "train_speed_mean")[0] for c in conds]
    ax[0].bar(x, [a for a, _, _ in m], yerr=[s for _, s, _ in m], capsize=4, color="#4C72B0")
    for i, ((a, s, _), v) in enumerate(zip(m, sp)):
        ax[0].text(i, a + s + 0.08, f"{a:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax[0].text(i, min(a for a, _, _ in m) * 0.9 + 0.1, f"train\n{v * 100:.1f} cm/s", ha="center", va="bottom",
                   fontsize=7, color="white")
    ax[0].set_ylim(min(a for a, _, _ in m) * 0.9, max(a + s for a, s, _ in m) * 1.03)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([V2_LABELS[c] for c in conds], fontsize=8)
    ax[0].set_ylabel("mean L2 error, all test samples (mm)")
    ax[0].set_title("All test samples (each training half: about 23,260 samples)")
    w = 0.26
    cols = (("slow_avg_mm", "slow third", "#8DA0CB"), ("mid_avg_mm", "middle third", "#4C72B0"), ("fast_avg_mm", "fast third", "#1F3A68"))
    for j, (col, lab, color) in enumerate(cols):
        mm = [stat(runs, "quality_v2", c, col) for c in conds]
        ax[1].bar(x + (j - 1) * w, [a for a, _, _ in mm], w, yerr=[s for _, s, _ in mm], capsize=2, color=color, label=lab)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([V2_LABELS[c] for c in conds], fontsize=8)
    ax[1].set_ylabel("mean L2 error (mm)")
    cuts = meta.get("test_speed_tertile_cuts_mps", [])
    ax[1].set_title("Test samples split by motion speed" + (f" (cuts {cuts[0] * 100:.1f} / {cuts[1] * 100:.1f} cm/s)" if cuts else ""))
    ax[1].legend(fontsize=8)
    rho = meta.get("spearman_quality_vs_speed")
    fig.suptitle("Separating data quality from motion speed (mean +- std over 3 seeds; matched = chosen within each speed decile)"
                 + (f"\nSpearman(quality score, speed) = {rho:.2f}" if rho is not None else ""))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_qc_filter(runs: pd.DataFrame, path: Path) -> None:
    col = "avg_mm"
    bars = [(("quality", "random_half"), "clean random half"),
            (("quality", "random_half_inject30"), "30% corrupted,\nno filter"),
            (("qc_filter", "inject30_rule_filter"), "30% corrupted,\nrule filter"),
            (("qc_filter", "inject30_oracle_filter"), "30% corrupted,\noracle filter")]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, ((e, c), lab) in enumerate(bars):
        m, s, _ = stat(runs, e, c, col)
        n = stat(runs, e, c, "n_train")[0]
        ax.bar(i, m, yerr=s, capsize=4, color=["#4C72B0", "#C44E52", "#55A868", "#8172B2"][i])
        ax.text(i, m + s + 0.05, f"{m:.1f} mm\nn={n:,.0f}", ha="center", va="bottom", fontsize=8)
    r = runs[(runs["experiment"] == "qc_filter") & (runs["condition"] == "inject30_rule_filter")]
    if len(r):
        recall = 100 * r["n_removed_corrupted"].sum() / max(r["n_corrupted"].sum(), 1)
        precision = 100 * r["n_removed_corrupted"].sum() / max(r["n_removed"].sum(), 1)
        ax.set_xlabel(f"rule filter removed {r['n_removed'].mean():,.0f} samples per run: "
                      f"{recall:.0f}% of the corrupted ones; {precision:.0f}% of removed samples were corrupted", fontsize=8)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[1] for b in bars], fontsize=8)
    lo = min(stat(runs, e, c, col)[0] for (e, c), _ in bars)
    ax.set_ylim(lo * 0.95, None)
    ax.set_ylabel("mean L2 error, all test samples (mm)")
    ax.set_title("Does rule-based QC filtering recover the damage? (mean +- std over 3 seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--derived-dir", default="data/derived")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    derived, out = Path(args.derived_dir), Path(args.out_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    ep = pd.read_csv(derived / "episode_metrics.csv")
    summary = json.loads((out / "quality_summary.json").read_text(encoding="utf-8"))
    findings = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    fig_quality(ep, summary, findings, figs / "fig1_quality_overview.png")
    fig_tasks(pd.read_csv(out / "task_view_tracking.csv"), findings, figs / "fig2_task_view_tracking.png")
    print("wrote fig1, fig2")
    if (out / "injection_summary.csv").exists():
        fig_injection(pd.read_csv(out / "injection_summary.csv").fillna(""), figs / "fig3_injection_detection.png")
        print("wrote fig3")
    if (out / "predictor_runs.csv").exists() and (out / "predictor_meta.json").exists():
        meta = json.loads((out / "predictor_meta.json").read_text(encoding="utf-8"))
        runs = pd.read_csv(out / "predictor_runs.csv")
        runs = runs[runs["config"] == runs["config"].iloc[-1]]
        fig_predictor(runs, meta, figs / "fig4_predictor_quality.png")
        fig_quantity(runs, figs / "fig5_quantity_vs_quality.png")
        fig_defect_harm(runs, pd.read_csv(out / "injection_summary.csv"), figs / "fig6_defect_harm_vs_detection.png")
        fig_qc_filter(runs, figs / "fig7_qc_filter.png")
        print("wrote fig4-fig7")
        if "quality_v2" in set(runs["experiment"]):
            fig_quality_v2(runs, meta, figs / "fig8_quality_vs_speed.png")
            print("wrote fig8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
