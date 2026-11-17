"""Shared constants, loaders and quality scores for the EgoDex data audit.

Joint names follow utils/skeleton_tfs.py of apple-aiml-research/ml-egodex.
Positions are in meters, in the ARKit world frame of each episode.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter

FPS = 30
SIDES = ("left", "right")
FINGERS = ("Little", "Ring", "Middle", "Index")
FINGER_SEGS = ("Metacarpal", "Knuckle", "IntermediateBase", "IntermediateTip", "Tip")
THUMB_SEGS = ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")

# Fixed check parameters. They are physical bounds chosen before looking at the data.
JUMP_M = 0.10         # wrist moves more than 10 cm between two frames (> 3 m/s): tracking jump
CAM_JUMP_M = 0.10     # same bound for the head-mounted camera
FREEZE_EPS_M = 1e-6   # no hand joint moves more than 1 micrometre: frozen frame
LOW_CONF = 0.5
SG_WIN, SG_POLY = 9, 2  # Savitzky-Golay smoothing used for the jitter residual
WIN = 60              # 2-second windows

DETECTORS = ("jitter_mm", "max_step_mm", "freeze_run", "swap_mm")


def hand_joints(side: str) -> list[str]:
    """25 joints of one hand, wrist first."""
    names = [f"{side}Hand"]
    for finger in FINGERS:
        names += [f"{side}{finger}Finger{s}" for s in FINGER_SEGS]
    names += [f"{side}Thumb{s}" for s in THUMB_SEGS]
    return names


def _bones(side: str) -> list[tuple[int, int]]:
    idx = {n: i for i, n in enumerate(hand_joints(side))}
    bones = []
    for finger in FINGERS:
        chain = [f"{side}Hand"] + [f"{side}{finger}Finger{s}" for s in FINGER_SEGS]
        bones += [(idx[a], idx[b]) for a, b in zip(chain[:-1], chain[1:])]
    chain = [f"{side}Hand"] + [f"{side}Thumb{s}" for s in THUMB_SEGS]
    bones += [(idx[a], idx[b]) for a, b in zip(chain[:-1], chain[1:])]
    return bones


def _keypoints(side: str) -> list[str]:
    """Wrist + 5 fingertips: the 6 points per hand used by the official EgoDex metric."""
    return [f"{side}Hand", f"{side}ThumbTip"] + [
        f"{side}{f}FingerTip" for f in ("Index", "Middle", "Ring", "Little")
    ]


BONES = _bones("left")  # same indices for both hands (24 bones)
KP_IDX = [hand_joints("left").index(k) for k in _keypoints("left")]  # 6 indices
TIP_IDX = KP_IDX[1:]


def setup_stdout() -> None:
    """Scripts run with python -I, which ignores PYTHONIOENCODING; force UTF-8 output."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def list_episodes(data_dir) -> list[dict]:
    data_dir = Path(data_dir)
    out = []
    for h5 in sorted(data_dir.rglob("*.hdf5")):
        out.append({
            "id": f"{h5.parent.name}/{h5.stem}",
            "task": h5.parent.name,
            "path": h5,
            "mp4": h5.with_suffix(".mp4"),
        })
    return out


def clean_attr(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def load_hands(path) -> dict:
    """Load both hands (2, N, 25, 3), confidences (2, N, 25) or None, camera (N, 4, 4), attrs."""
    with h5py.File(path, "r") as f:
        tf = f["transforms"]
        cam = np.asarray(tf["camera"][()], dtype=np.float64)
        pos = np.stack([
            np.stack([np.asarray(tf[j][()], dtype=np.float64)[:, :3, 3] for j in hand_joints(s)], axis=1)
            for s in SIDES
        ])
        conf = None
        if "confidences" in f:
            c = f["confidences"]
            if all(j in c for s in SIDES for j in hand_joints(s)):
                conf = np.stack([
                    np.stack([np.asarray(c[j][()], dtype=np.float64) for j in hand_joints(s)], axis=1)
                    for s in SIDES
                ])
        attrs = {k: clean_attr(v) for k, v in f.attrs.items()}
    return {"pos": pos, "conf": conf, "cam": cam, "attrs": attrs}


def steps(p: np.ndarray) -> np.ndarray:
    """Per-frame displacement (m) of an (N, 3) trajectory; the first frame is 0."""
    s = np.zeros(len(p))
    if len(p) > 1:
        s[1:] = np.linalg.norm(np.diff(p, axis=0), axis=-1)
    return s


def jitter_residual(p: np.ndarray) -> np.ndarray:
    """Per-frame distance (m) between an (N, 3) trajectory and its smoothed version."""
    if len(p) < SG_WIN:
        return np.full(len(p), np.nan)
    sm = savgol_filter(p, SG_WIN, SG_POLY, axis=0)
    return np.linalg.norm(p - sm, axis=-1)


def frozen(hand: np.ndarray) -> np.ndarray:
    """(N, 25, 3) -> (N,) True where no hand joint moved since the previous frame."""
    m = np.zeros(len(hand), dtype=bool)
    if len(hand) > 1:
        m[1:] = np.linalg.norm(np.diff(hand, axis=0), axis=-1).max(axis=1) < FREEZE_EPS_M
    return m


def longest_run(mask) -> int:
    m = np.asarray(mask, dtype=np.int8)
    if m.size == 0 or not m.any():
        return 0
    d = np.diff(np.concatenate([[0], m, [0]]))
    return int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max())


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.sqrt(np.nanmean(x ** 2)))


def window_scores(hands: np.ndarray, conf: np.ndarray | None = None) -> dict:
    """Detector scores for one window. hands: (2, T, 25, 3); conf: (2, T, 25) or None."""
    out = {}
    st = []
    for k, side in enumerate(SIDES):
        w = hands[k][:, 0]
        s = steps(w)
        st.append(s)
        out[f"{side}_jitter_mm"] = 1000 * rms(jitter_residual(w))
        out[f"{side}_max_step_mm"] = 1000 * float(s.max())
        out[f"{side}_freeze_run"] = longest_run(frozen(hands[k]))
        if conf is not None:
            out[f"{side}_conf_mean"] = float(conf[k].mean())
            out[f"{side}_conf0_ratio"] = float((conf[k][:, 0] <= 0).mean())
    out["swap_mm"] = 1000 * float(np.minimum(st[0], st[1]).max())
    for name in ("jitter_mm", "max_step_mm", "freeze_run"):
        out[name] = max(out[f"left_{name}"], out[f"right_{name}"])
    return out


def episode_metrics(d: dict) -> dict:
    pos, conf, cam = d["pos"], d["conf"], d["cam"]
    n = pos.shape[1]
    row = {"n_frames": n, "duration_s": n / FPS, "has_conf": conf is not None}
    row["nan_frames"] = int((~np.isfinite(pos).all(axis=(0, 2, 3))).sum())
    for k, side in enumerate(SIDES):
        hand = pos[k]
        w = hand[:, 0]
        s = steps(w)
        fr = frozen(hand)
        tip_r = np.nanmean(np.stack([jitter_residual(hand[:, i]) for i in TIP_IDX]), axis=0)
        lengths = np.stack([np.linalg.norm(hand[:, b] - hand[:, a], axis=-1) for a, b in BONES], axis=1)
        cv = lengths.std(axis=0) / np.maximum(lengths.mean(axis=0), 1e-9)
        row[f"{side}_jitter_mm"] = 1000 * rms(jitter_residual(w))
        row[f"{side}_tip_jitter_mm"] = 1000 * rms(tip_r)
        row[f"{side}_max_speed_mps"] = float(s.max() * FPS)
        row[f"{side}_jumps"] = int((s > JUMP_M).sum())
        row[f"{side}_freeze_ratio"] = float(fr.mean())
        row[f"{side}_max_freeze_run"] = longest_run(fr)
        row[f"{side}_bone_cv_median"] = float(np.median(cv))
        if conf is not None:
            c = conf[k]
            row[f"{side}_conf_mean"] = float(c.mean())
            row[f"{side}_conf0_ratio"] = float((c[:, 0] <= 0).mean())
            row[f"{side}_lowconf_ratio"] = float((c.mean(axis=1) < LOW_CONF).mean())
            row[f"{side}_frozen_and_conf0"] = int((fr & (c[:, 0] <= 0)).sum())
        else:
            for key in ("conf_mean", "conf0_ratio", "lowconf_ratio", "frozen_and_conf0"):
                row[f"{side}_{key}"] = np.nan
    cs = steps(cam[:, :3, 3])
    row["cam_max_speed_mps"] = float(cs.max() * FPS)
    row["cam_jumps"] = int((cs > CAM_JUMP_M).sum())
    return row


def auroc(pos_scores, neg_scores) -> float:
    """Probability that an injected window scores higher than a clean one (ties count half)."""
    from scipy.stats import rankdata

    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos_scores, neg_scores]))
    rp = r[: len(pos_scores)].sum()
    return float((rp - len(pos_scores) * (len(pos_scores) + 1) / 2) / (len(pos_scores) * len(neg_scores)))
