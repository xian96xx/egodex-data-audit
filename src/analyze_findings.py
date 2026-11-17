"""Step 2b: follow-up questions raised by the health check.

1. Frozen frames: are they whole-frame duplicates (both hands and the camera unchanged)?
2. Wrist confidence = 0: is the hand outside the camera view, or lost while visible?
3. Language metadata: how often is the second description empty, and is the selector present?

Projection follows the official visualize_2d.py: joints are moved into the camera frame with
inv(camera extrinsic) and projected with the OpenCV pinhole model (Z forward).
Writes aggregate numbers to outputs/findings.json (safe to publish).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from egodex_common import FREEZE_EPS_M, SIDES, clean_attr, frozen, list_episodes, load_hands, setup_stdout, steps  # noqa: E402

K = np.array([[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]])
W, H = 1920, 1080


def in_view(p_world: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """(N, 3) world points, (N, 4, 4) camera extrinsics -> (N,) bool inside the image."""
    inv = np.linalg.inv(cam)
    pc = np.einsum("nij,nj->ni", inv[:, :3, :3], p_world) + inv[:, :3, 3]
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * pc[:, 0] / z + K[0, 2]
        v = K[1, 1] * pc[:, 1] / z + K[1, 2]
    return (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)


def runs(mask: np.ndarray) -> list[int]:
    m = np.asarray(mask, dtype=np.int8)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    return list((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).tolist())


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    eps = list_episodes(args.data_dir)
    dup = {"whole_frame_duplicates": 0, "hand_frozen_camera_moving": 0, "episodes_with_duplicates": 0,
           "frames_total": 0}
    dup_runs, dup_pos = Counter(), Counter()
    tab = {s: Counter() for s in SIDES}            # (conf0, in_view) counts per hand
    speed = {s: {"conf0": [], "conf_pos": []} for s in SIDES}
    conf_in, conf_out = [], []
    lang = Counter()
    per_task: dict[str, Counter] = {}
    for ep in eps:
        d = load_hands(ep["path"])
        pos, cam, conf = d["pos"], d["cam"], d["conf"]
        n = pos.shape[1]
        dup["frames_total"] += n
        fr_l, fr_r = frozen(pos[0]), frozen(pos[1])
        cam_still = np.zeros(n, dtype=bool)
        if n > 1:
            cam_still[1:] = np.abs(np.diff(cam, axis=0)).reshape(n - 1, -1).max(axis=1) < FREEZE_EPS_M
        whole = fr_l & fr_r & cam_still
        dup["whole_frame_duplicates"] += int(whole.sum())
        dup["hand_frozen_camera_moving"] += int(((fr_l | fr_r) & ~cam_still).sum())
        if whole.any():
            dup["episodes_with_duplicates"] += 1
            dup_runs.update(runs(whole))
            for i in np.flatnonzero(whole):
                dup_pos["first_10_frames" if i < 10 else ("last_10_frames" if i >= n - 10 else "middle")] += 1
        if conf is not None:
            pt = per_task.setdefault(ep["task"], Counter())
            pt["episodes"] += 1
            for k, s in enumerate(SIDES):
                c0 = conf[k][:, 0] <= 0
                vis = in_view(pos[k][:, 0], cam)
                pt["hand_frames"] += n
                pt["out_of_view"] += int((~vis).sum())
                pt["in_view"] += int(vis.sum())
                pt["in_view_lost"] += int((vis & c0).sum())
                for a in (True, False):
                    for b in (True, False):
                        tab[s][(a, b)] += int((c0 == a).__and__(vis == b).sum())
                sp = steps(pos[k][:, 0]) * 30
                speed[s]["conf0"].append(sp[c0])
                speed[s]["conf_pos"].append(sp[~c0])
                conf_in.append(conf[k][:, 0][vis & ~c0])
                conf_out.append(conf[k][:, 0][~vis & ~c0])
        with h5py.File(ep["path"], "r") as f:
            a = {k: clean_attr(v) for k, v in f.attrs.items()}
        d2 = a.get("llm_description2")
        has2 = isinstance(d2, str) and d2.strip() != ""
        lang["desc2_nonempty" if has2 else "desc2_empty"] += 1
        has_w = "which_llm_description" in a
        lang[f"desc2_{'nonempty' if has2 else 'empty'}__which_{'present' if has_w else 'missing'}"] += 1
        if has_w:
            lang[f"which_value_{a['which_llm_description']}"] += 1

    def share(c: Counter, s: str) -> dict:
        tot = sum(c.values())
        names = {(True, True): "conf0_in_view", (True, False): "conf0_out_of_view",
                 (False, True): "conf_pos_in_view", (False, False): "conf_pos_out_of_view"}
        out = {names[k]: v for k, v in c.items()}
        out["frames"] = tot
        c0 = c[(True, True)] + c[(True, False)]
        out["conf0_share_of_frames_pct"] = round(100 * c0 / max(tot, 1), 2)
        out["conf0_frames_out_of_view_pct"] = round(100 * c[(True, False)] / max(c0, 1), 1)
        vis = c[(False, True)] + c[(True, True)]
        out["in_view_frames_with_conf0_pct"] = round(100 * c[(True, True)] / max(vis, 1), 2)
        return out

    med = lambda xs: round(float(np.median(np.concatenate(xs))), 3) if xs and sum(len(x) for x in xs) else None  # noqa: E731
    findings = {
        "duplicate_frames": {**dup,
                             "whole_frame_duplicates_pct_of_frames": round(100 * dup["whole_frame_duplicates"] / dup["frames_total"], 3),
                             "run_lengths": {str(k): v for k, v in sorted(dup_runs.items())},
                             "where": dict(dup_pos)},
        "wrist_conf0_vs_view": {s: share(tab[s], s) for s in SIDES},
        "median_wrist_speed_mps": {s: {"when_conf0": med(speed[s]["conf0"]), "when_conf_pos": med(speed[s]["conf_pos"])} for s in SIDES},
        "wrist_conf_median_when_detected": {"in_view": med(conf_in), "out_of_view": med(conf_out)},
        "language_metadata": dict(lang),
        "projection": "OpenCV pinhole with the official intrinsics, after inv(camera extrinsic), as in visualize_2d.py",
    }
    out = Path(args.out_dir)
    (out / "findings.json").write_text(json.dumps(findings, indent=2), encoding="utf-8")
    rows = []
    for task, c in sorted(per_task.items()):
        rows.append({
            "task": task, "episodes": c["episodes"], "hand_frames": c["hand_frames"],
            "out_of_view_pct": round(100 * c["out_of_view"] / max(c["hand_frames"], 1), 2),
            "in_view_lost_pct_of_in_view": round(100 * c["in_view_lost"] / max(c["in_view"], 1), 3),
        })
    import pandas as pd  # local import keeps the module light when only findings are needed

    pd.DataFrame(rows).to_csv(out / "task_view_tracking.csv", index=False, encoding="utf-8")
    print(json.dumps(findings, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
