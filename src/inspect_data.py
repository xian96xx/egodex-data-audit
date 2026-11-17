"""Step 1: inventory the extracted EgoDex test set and check the file structure.

Writes per-episode details to data/derived/inventory.csv (local only) and
aggregate numbers to outputs/inventory_summary.json (safe to publish).
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
import pandas as pd  # noqa: E402

from egodex_common import BONES, SIDES, clean_attr, hand_joints, list_episodes, setup_stdout  # noqa: E402

EXPECTED_JOINTS = 68


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--derived-dir", default="data/derived")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    eps = list_episodes(args.data_dir)
    if not eps:
        print(f"No .hdf5 files under {args.data_dir}")
        return 1
    derived, out = Path(args.derived_dir), Path(args.out_dir)
    derived.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    hand_names = [j for s in SIDES for j in hand_joints(s)]
    rows, tf_counts, conf_counts, attr_keys, dtypes, errors = [], Counter(), Counter(), Counter(), Counter(), []
    missing_hand_joints = Counter()
    bone_mm, wrist_abs_m = [], []
    for i, ep in enumerate(eps):
        try:
            with h5py.File(ep["path"], "r") as f:
                tf = f["transforms"]
                keys = set(tf.keys())
                n = int(tf["leftHand"].shape[0])
                dtypes[str(tf["leftHand"].dtype)] += 1
                tf_counts[len(keys)] += 1
                for j in hand_names:
                    if j not in keys:
                        missing_hand_joints[j] += 1
                has_conf = "confidences" in f
                conf_counts[len(f["confidences"].keys()) if has_conf else 0] += 1
                attrs = {k: clean_attr(v) for k, v in f.attrs.items()}
                attr_keys.update(attrs.keys())
                shapes_ok = all(tf[k].shape == (n, 4, 4) for k in keys)
                if i % 25 == 0:  # unit sanity check on a subset: bone lengths and wrist distances
                    lh = np.stack([tf[j][()][:, :3, 3] for j in hand_joints("left")], axis=1)
                    bone_mm.append(float(np.median([np.linalg.norm(lh[:, b] - lh[:, a], axis=-1).mean() for a, b in BONES]) * 1000))
                    wrist_abs_m.append(float(np.median(np.linalg.norm(lh[:, 0], axis=-1))))
        except Exception as e:  # keep going, report at the end
            errors.append({"id": ep["id"], "error": repr(e)})
            continue
        desc = attrs.get("llm_description")
        rows.append({
            "id": ep["id"],
            "task": ep["task"],
            "n_frames": n,
            "duration_s": n / 30.0,
            "n_transforms": len(keys),
            "shapes_ok": shapes_ok,
            "has_conf": has_conf,
            "mp4_exists": ep["mp4"].exists(),
            "mp4_mb": ep["mp4"].stat().st_size / 1e6 if ep["mp4"].exists() else np.nan,
            "has_desc": isinstance(desc, str) and len(desc) > 0,
            "has_desc2": "llm_description2" in attrs,
            "which_desc": attrs.get("which_llm_description"),
        })
        if (i + 1) % 500 == 0:
            print(f"  scanned {i + 1}/{len(eps)}")

    df = pd.DataFrame(rows)
    df.to_csv(derived / "inventory.csv", index=False, encoding="utf-8")
    per_task = df.groupby("task").size()
    summary = {
        "n_episodes": int(len(df)),
        "n_tasks": int(df["task"].nunique()),
        "total_hours": round(float(df["duration_s"].sum() / 3600), 2),
        "episode_seconds": {q: round(float(df["duration_s"].quantile(v)), 1) for q, v in
                            [("min", 0), ("p10", .1), ("median", .5), ("p90", .9), ("max", 1)]},
        "episodes_per_task": {"min": int(per_task.min()), "median": float(per_task.median()), "max": int(per_task.max())},
        "transform_key_counts": {str(k): v for k, v in sorted(tf_counts.items())},
        "confidence_key_counts": {str(k): v for k, v in sorted(conf_counts.items())},
        "episodes_with_confidences": int(df["has_conf"].sum()),
        "episodes_without_confidences": int((~df["has_conf"]).sum()),
        "all_shapes_ok": bool(df["shapes_ok"].all()),
        "transform_dtypes": dict(dtypes),
        "missing_hand_joints": dict(missing_hand_joints),
        "attr_keys": dict(attr_keys),
        "episodes_with_description": int(df["has_desc"].sum()),
        "episodes_with_second_description": int(df["has_desc2"].sum()),
        "which_description_values": {str(k): int(v) for k, v in df["which_desc"].value_counts(dropna=False).items()},
        "mp4_paired": int(df["mp4_exists"].sum()),
        "unit_check": {
            "median_hand_bone_mm_on_sample": round(float(np.median(bone_mm)), 1) if bone_mm else None,
            "median_wrist_distance_from_origin_m_on_sample": round(float(np.median(wrist_abs_m)), 2) if wrist_abs_m else None,
            "sampled_episodes": len(bone_mm),
        },
        "expected_joints_plus_camera": EXPECTED_JOINTS + 1,
        "read_errors": len(errors),
    }
    (out / "inventory_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    if errors:
        pd.DataFrame(errors).to_csv(derived / "read_errors.csv", index=False, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
