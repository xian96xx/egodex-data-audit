"""Step 3: inject synthetic defects into clean windows and check whether the rules catch them.

Thresholds: for each detector, the 95th percentile over all real windows (a 5% alarm budget).
Detection rates here only measure sensitivity to *synthetic* defects; they are not real error rates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import (  # noqa: E402
    DETECTORS, JUMP_M, SIDES, WIN, auroc, list_episodes, load_hands, setup_stdout, window_scores,
)

INJECT = {
    "jitter": [2, 5, 10],     # Gaussian noise sigma (mm) on all joints of one hand, whole window
    "spike": [30, 100, 300],  # offset (mm) on one hand for 2 frames
    "freeze": [5, 15, 30],    # frames held at the previous pose, one hand
    "swap": [0],              # left/right hands swapped from a random frame to the window end
    "drift": [20, 50, 100],   # linear drift (mm) reached at the window end, one hand
}


def inject(h: np.ndarray, kind: str, mag: int, rng: np.random.Generator):
    h2 = h.copy()
    t = h.shape[1]
    k = int(rng.integers(2))
    info = {"side": SIDES[k], "gap_mm": np.nan}
    if kind == "jitter":
        h2[k] += rng.normal(0.0, mag / 1000.0, size=h2[k].shape)
    elif kind == "spike":
        f = int(rng.integers(10, t - 12))
        d = rng.normal(size=3)
        h2[k, f:f + 2] += d / np.linalg.norm(d) * mag / 1000.0
    elif kind == "freeze":
        f = int(rng.integers(5, t - mag - 1))
        h2[k, f:f + mag] = h2[k, f - 1]
    elif kind == "swap":
        f = int(rng.integers(15, 46))
        h2[0, f:], h2[1, f:] = h[1, f:].copy(), h[0, f:].copy()
        info = {"side": "both", "gap_mm": 1000.0 * float(np.linalg.norm(h[0, f, 0] - h[1, f, 0]))}
    elif kind == "drift":
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        h2[k] += np.linspace(0.0, mag / 1000.0, t)[:, None, None] * d[None, None, :]
    else:
        raise ValueError(kind)
    return h2, info


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--derived-dir", default="data/derived")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--n-clean", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    derived, out = Path(args.derived_dir), Path(args.out_dir)
    win = pd.read_csv(derived / "window_metrics.csv")
    thr = {d: float(np.nanpercentile(win[d], 95)) for d in DETECTORS}
    real_alarm = {d: float((win[d] > thr[d]).mean()) for d in DETECTORS}
    real_any = float((win[list(DETECTORS)] > pd.Series(thr)).any(axis=1).mean())

    # Clean pool: both wrists detected in every frame, hand confidence at or above the median window
    # (ARKit hand-mean confidence sits around 0.5, so a fixed 0.9 bar selects nothing), no jump,
    # no freeze beyond the 1-frame duplicate at the start of most episodes, jitter <= 75th percentile.
    jit75 = float(win["jitter_mm"].quantile(0.75))
    wc = win[win["has_conf"]]
    conf_med = float(pd.concat([wc["left_conf_mean"], wc["right_conf_mean"]]).median())
    pool = win[
        win["has_conf"]
        & (win["left_conf_mean"] >= conf_med) & (win["right_conf_mean"] >= conf_med)
        & (win["left_conf0_ratio"] == 0) & (win["right_conf0_ratio"] == 0)
        & (win["max_step_mm"] < 1000 * JUMP_M) & (win["freeze_run"] < 2) & (win["jitter_mm"] <= jit75)
    ]
    criteria = {"hand_conf_mean_min_median_of_windows": round(conf_med, 4), "jitter_mm_max_p75": round(jit75, 3),
                "pool_size": int(len(pool))}
    sample = pool.sample(n=min(args.n_clean, len(pool)), random_state=args.seed).reset_index(drop=True)
    print(f"clean pool {len(pool)} windows, sampled {len(sample)}; criteria {criteria}")

    paths = {e["id"]: e["path"] for e in list_episodes(args.data_dir)}
    rng = np.random.default_rng(args.seed)
    rows = []
    cache_id, cache = None, None
    for wid, r in sample.sort_values(["id", "start"]).iterrows():
        if r["id"] != cache_id:
            cache_id, cache = r["id"], load_hands(paths[r["id"]])
        s = int(r["start"])
        h = cache["pos"][:, s:s + WIN]
        conf = cache["conf"][:, s:s + WIN] if cache["conf"] is not None else None
        base = {"wid": wid, "id": r["id"], "start": s}
        sc0 = window_scores(h, conf)
        rows.append({**base, "defect": "none", "magnitude": 0, "side": "", "gap_mm": np.nan,
                     **{d: sc0[d] for d in DETECTORS}})
        for kind, mags in INJECT.items():
            for mag in mags:
                h2, info = inject(h, kind, mag, rng)
                sc = window_scores(h2, conf)
                rows.append({**base, "defect": kind, "magnitude": mag, **info, **{d: sc[d] for d in DETECTORS}})
    df = pd.DataFrame(rows)
    df.to_csv(derived / "injection_scores.csv", index=False, encoding="utf-8")

    none = df[df["defect"] == "none"]
    thr_s = pd.Series(thr)
    summary = []
    clean_row = {"defect": "none (clean sample)", "magnitude": "", "n": len(none)}
    for d in DETECTORS:
        clean_row[f"{d}_rate"] = round(float((none[d] > thr[d]).mean()), 3)
        clean_row[f"{d}_auroc"] = ""
    clean_row["any_rate"] = round(float((none[list(DETECTORS)] > thr_s).any(axis=1).mean()), 3)
    summary.append(clean_row)
    inj = df[df["defect"] != "none"].copy()
    inj["group"] = inj["magnitude"].astype(str)
    swap = inj["defect"] == "swap"
    inj.loc[swap, "group"] = np.where(inj.loc[swap, "gap_mm"] < 100, "gap<10cm", "gap>=10cm")
    for (kind, grp), g in inj.groupby(["defect", "group"], sort=False):
        row = {"defect": kind, "magnitude": grp, "n": len(g)}
        for d in DETECTORS:
            row[f"{d}_rate"] = round(float((g[d] > thr[d]).mean()), 3)
            row[f"{d}_auroc"] = round(auroc(g[d], none[d]), 3)
        row["any_rate"] = round(float((g[list(DETECTORS)] > thr_s).any(axis=1).mean()), 3)
        summary.append(row)
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "injection_summary.csv", index=False, encoding="utf-8")
    meta = {
        "thresholds_p95_of_real_windows": {k: round(v, 3) for k, v in thr.items()},
        "real_window_alarm_rate": {k: round(v, 4) for k, v in real_alarm.items()},
        "real_window_alarm_rate_any_detector": round(real_any, 4),
        "n_real_windows": int(len(win)),
        "clean_pool": criteria,
        "n_clean_windows_injected": int(len(sample)),
        "inject_settings": INJECT,
        "seed": args.seed,
        "note": "Synthetic defects only. Detection rates are sensitivity to injected defects, not real error rates.",
    }
    (out / "injection_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(sm.to_string(index=False))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
