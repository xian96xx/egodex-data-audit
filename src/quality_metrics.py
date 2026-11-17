"""Step 2: hand-trajectory health check on every episode and every 2-second window.

Per-episode and per-window tables go to data/derived/ (local only).
Aggregate counts go to outputs/quality_summary.json (safe to publish).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import (  # noqa: E402
    CAM_JUMP_M, FREEZE_EPS_M, JUMP_M, LOW_CONF, SG_POLY, SG_WIN, SIDES, WIN,
    episode_metrics, list_episodes, load_hands, setup_stdout, window_scores,
)


def pct(x) -> float:
    return round(100.0 * float(np.mean(x)), 2)


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--derived-dir", default="data/derived")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N episodes (debug)")
    args = ap.parse_args()

    eps = list_episodes(args.data_dir)
    if args.limit:
        eps = eps[: args.limit]
    derived, out = Path(args.derived_dir), Path(args.out_dir)
    derived.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    ep_rows, win_rows, errors = [], [], []
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            d = load_hands(ep["path"])
        except Exception as e:
            errors.append({"id": ep["id"], "error": repr(e)})
            continue
        row = {"id": ep["id"], "task": ep["task"]}
        row.update(episode_metrics(d))
        ep_rows.append(row)
        n = d["pos"].shape[1]
        for s in range(0, n - WIN + 1, WIN):
            conf = d["conf"][:, s:s + WIN] if d["conf"] is not None else None
            w = {"id": ep["id"], "task": ep["task"], "start": s, "has_conf": conf is not None}
            w.update(window_scores(d["pos"][:, s:s + WIN], conf))
            win_rows.append(w)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(eps)} episodes, {time.time() - t0:.0f}s")

    ep_df, win_df = pd.DataFrame(ep_rows), pd.DataFrame(win_rows)
    ep_df.to_csv(derived / "episode_metrics.csv", index=False, encoding="utf-8")
    win_df.to_csv(derived / "window_metrics.csv", index=False, encoding="utf-8")

    c = ep_df[ep_df["has_conf"]]
    either = lambda col, cond: np.logical_or(cond(ep_df[f"left_{col}"]), cond(ep_df[f"right_{col}"]))  # noqa: E731
    either_c = lambda col, cond: np.logical_or(cond(c[f"left_{col}"]), cond(c[f"right_{col}"]))  # noqa: E731
    flags = {
        "no_confidences": pct(~ep_df["has_conf"]),
        "nan_positions": pct(ep_df["nan_frames"] > 0),
        "wrist_jump_gt_10cm_per_frame": pct(either("jumps", lambda x: x > 0)),
        "frozen_hand_ge_5_frames": pct(either("max_freeze_run", lambda x: x >= 5)),
        "camera_jump_gt_10cm_per_frame": pct(ep_df["cam_jumps"] > 0),
        "wrist_jitter_gt_2mm": pct(either("jitter_mm", lambda x: x > 2)),
        "wrist_jitter_gt_5mm": pct(either("jitter_mm", lambda x: x > 5)),
    }
    flags_conf = {
        "wrist_conf0_gt_10pct_frames": pct(either_c("conf0_ratio", lambda x: x > 0.10)),
        "wrist_conf0_any_frame": pct(either_c("conf0_ratio", lambda x: x > 0)),
        "hand_lowconf_gt_10pct_frames": pct(either_c("lowconf_ratio", lambda x: x > 0.10)),
    }
    q = lambda s: {k: round(float(s.quantile(v)), 4) for k, v in [("p10", .1), ("p50", .5), ("p90", .9), ("p99", .99)]}  # noqa: E731
    summary = {
        "n_episodes": int(len(ep_df)),
        "n_windows": int(len(win_df)),
        "read_errors": len(errors),
        "parameters": {
            "jump_m_per_frame": JUMP_M, "camera_jump_m_per_frame": CAM_JUMP_M, "freeze_eps_m": FREEZE_EPS_M,
            "low_conf": LOW_CONF, "savgol_window": SG_WIN, "savgol_poly": SG_POLY, "window_frames": WIN,
        },
        "episode_flags_pct_of_all_episodes": flags,
        "episode_flags_pct_of_episodes_with_confidences": flags_conf,
        "distributions": {
            **{f"{s}_jitter_mm": q(ep_df[f"{s}_jitter_mm"]) for s in SIDES},
            **{f"{s}_tip_jitter_mm": q(ep_df[f"{s}_tip_jitter_mm"]) for s in SIDES},
            **{f"{s}_max_speed_mps": q(ep_df[f"{s}_max_speed_mps"]) for s in SIDES},
            **{f"{s}_conf_mean": q(c[f"{s}_conf_mean"]) for s in SIDES},
            **{f"{s}_conf0_ratio": q(c[f"{s}_conf0_ratio"]) for s in SIDES},
            **{f"{s}_bone_cv_median": q(ep_df[f"{s}_bone_cv_median"]) for s in SIDES},
            "cam_max_speed_mps": q(ep_df["cam_max_speed_mps"]),
        },
        "frozen_frames_total": {s: int((ep_df[f"{s}_freeze_ratio"] * ep_df["n_frames"]).round().sum()) for s in SIDES},
        "frozen_frames_with_wrist_conf0": {s: int(c[f"{s}_frozen_and_conf0"].sum()) for s in SIDES},
        "window_detector_p95": {k: round(float(win_df[k].quantile(0.95)), 3) for k in
                                ("jitter_mm", "max_step_mm", "freeze_run", "swap_mm")},
    }
    (out / "quality_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        pd.DataFrame(errors).to_csv(derived / "quality_errors.csv", index=False, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
