"""Step 4: does training-data quality matter for a small hand-motion predictor?

Task: from the past 1 s (30 frames) of 12 keypoints (2 wrists + 10 fingertips), predict the next 1 s,
expressed in the camera frame at the current frame. Joints only, no images.
Metric follows the EgoDex paper's best-of-K distance with K = 1: mean L2 over keypoints and time
("avg") and at the last predicted frame ("final"), in millimetres.

Experiments (all on the same episode split, 3 seeds each):
  quality      equal-size training sets: random half, cleanest half, noisiest half (by a quality score),
               random half with 10% / 30% of samples corrupted by synthetic defects, and all data
  quantity     random 1/8 and 1/4 (with random half / all from "quality") and the cleanest 1/4
  defect_type  random half with 30% of samples corrupted by ONE defect type
  qc_filter    the 30%-corrupted random half after removing samples flagged by the step-3 rules,
               and after removing exactly the corrupted samples (oracle)
  quality_v2   speed controls: halves chosen by tracking confidence only, or by the quality score
               within each speed decile (same speed distribution as the whole training set);
               test error is also reported for slow / medium / fast thirds of the test samples

Results come from the 7.65-hour EgoDex *test set* split by episode, so they are not comparable
with the paper's benchmark. Every finished run is appended to outputs/predictor_runs.csv at once,
so an interrupted job can be restarted and continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from torch import nn  # noqa: E402

from egodex_common import (  # noqa: E402
    KP_IDX, frozen, jitter_residual, list_episodes, load_hands, longest_run, rms, setup_stdout, steps,
)

PAST, FUT = 30, 30
SPAN = PAST + FUT
NKP = 12
DEFECTS = ("jitter", "spike", "freeze", "swap", "drift")
RULES = ("jitter_mm", "max_step_mm", "freeze_run", "swap_mm")

SPECS = (
    [("quality", c) for c in ("full", "random_half", "top_half", "bottom_half",
                              "random_half_inject10", "random_half_inject30")]
    + [("quantity", c) for c in ("random_eighth", "random_quarter", "top_quarter")]
    + [("defect_type", f"{d}30") for d in DEFECTS]
    + [("qc_filter", c) for c in ("inject30_rule_filter", "inject30_oracle_filter")]
    + [("quality_v2", c) for c in ("random_half", "top_half", "bottom_half", "conf_top_half", "conf_bottom_half",
                                   "top_half_speed_matched", "bottom_half_speed_matched")]
)


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- data
def episode_samples(d: dict, stride: int):
    pos, conf, cam = d["pos"], d["conf"], d["cam"]
    n = pos.shape[1]
    if n < SPAN:
        return None
    kp = np.concatenate([pos[0][:, KP_IDX], pos[1][:, KP_IDX]], axis=1)  # (N, 12, 3) world
    t = np.arange(PAST - 1, n - FUT, stride)
    if len(t) == 0:
        return None
    idx = t[:, None] + np.arange(-PAST + 1, FUT + 1)[None]  # (S, 60)
    inv = np.linalg.inv(cam[t])
    seq = np.einsum("sij,sfkj->sfki", inv[:, :3, :3], kp[idx]) + inv[:, None, None, :3, 3]
    cur = seq[:, PAST - 1]
    x_rel, y = seq[:, :PAST] - cur[:, None], seq[:, PAST:] - cur[:, None]

    starts = t - PAST + 1
    step = np.stack([steps(pos[k][:, 0]) for k in range(2)]).max(axis=0)
    win_step = sliding_window_view(step, SPAN).max(axis=1)[starts]
    res2 = np.nan_to_num(np.stack([jitter_residual(pos[k][:, 0]) for k in range(2)]) ** 2)
    cs = np.concatenate([np.zeros((2, 1)), np.cumsum(res2, axis=1)], axis=1)
    jit = np.sqrt((cs[:, starts + SPAN] - cs[:, starts]) / SPAN).max(axis=0)
    fr = np.stack([frozen(pos[k]) for k in range(2)]).any(axis=0).astype(np.float64)
    cf = np.concatenate([[0.0], np.cumsum(fr)])
    frz = cf[starts + SPAN] - cf[starts]
    if conf is not None:
        cc = np.concatenate([[0.0], np.cumsum(conf.mean(axis=(0, 2)))])
        cmean = (cc[starts + SPAN] - cc[starts]) / SPAN
    else:
        cmean = np.full(len(t), np.nan)
    ok = np.isfinite(seq).all(axis=(1, 2, 3))
    return {
        "x_rel": x_rel[ok].astype(np.float32), "cur": cur[ok].astype(np.float32), "y": y[ok].astype(np.float32),
        "conf": cmean[ok], "jit_mm": 1000 * jit[ok], "step_mm": 1000 * win_step[ok], "frozen": frz[ok],
    }


def build_dataset(data_dir: str, cache: Path, stride: int) -> dict:
    eps = list_episodes(data_dir)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if int(z["stride"]) == stride and int(z["n_episodes"]) == len(eps):
            log(f"loaded cache {cache}")
            return {k: z[k] for k in z.files}
    parts, ep_idx = [], []
    t0 = time.time()
    for i, ep in enumerate(eps):
        s = episode_samples(load_hands(ep["path"]), stride)
        if s is not None and len(s["y"]):
            parts.append(s)
            ep_idx.append(np.full(len(s["y"]), i))
        if (i + 1) % 500 == 0:
            log(f"  built {i + 1}/{len(eps)} episodes, {time.time() - t0:.0f}s")
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    data["episode"] = np.concatenate(ep_idx)
    data["stride"] = np.array(stride)
    data["n_episodes"] = np.array(len(eps))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    return data


def sample_speed(data: dict, chunk: int = 8192) -> np.ndarray:
    """Mean wrist speed (m/s, both hands) over each sample's 60-frame span. Differences of the
    current-frame-relative coordinates equal differences of the camera-frame coordinates."""
    n = len(data["y"])
    out = np.empty(n)
    for s in range(0, n, chunk):
        seq = np.concatenate([data["x_rel"][s:s + chunk][:, :, [0, 6]], data["y"][s:s + chunk][:, :, [0, 6]]],
                             axis=1).astype(np.float64)
        out[s:s + chunk] = np.linalg.norm(np.diff(seq, axis=1), axis=-1).mean(axis=(1, 2)) * 30.0
    return out


def speed_matched(tr: np.ndarray, q: np.ndarray, speed: np.ndarray, top: bool, bins: int = 10) -> np.ndarray:
    """Half of the training samples, chosen by quality score within each speed decile,
    so the selected half has the same speed distribution as the whole training set."""
    edges = np.quantile(speed[tr], np.linspace(0, 1, bins + 1))
    b = np.clip(np.searchsorted(edges, speed[tr], side="right") - 1, 0, bins - 1)
    keep = []
    for k in range(bins):
        members = tr[b == k]
        members = members[np.argsort(-q[members] if top else q[members], kind="stable")]
        keep.append(members[: len(members) // 2])
    return np.concatenate(keep)


def quality_score(data: dict) -> np.ndarray:
    """Rank-average of four checks (higher = cleaner). Missing confidences count as the median."""
    parts = [
        pd.Series(data["conf"]).rank(pct=True),
        1 - pd.Series(data["jit_mm"]).rank(pct=True),
        1 - pd.Series(data["step_mm"]).rank(pct=True),
        1 - pd.Series(data["frozen"]).rank(pct=True),
    ]
    return pd.concat(parts, axis=1).fillna(0.5).mean(axis=1).to_numpy()


# ---------------------------------------------------------------- defects and rules
def corrupt(x_rel, cur, y, rng, kind: str):
    """Apply one synthetic defect over the 60-frame span (meters, camera frame). Returns new arrays."""
    full = np.concatenate([x_rel + cur[None], y + cur[None]], axis=0)
    if kind == "mixed":
        kind = DEFECTS[int(rng.integers(len(DEFECTS)))]
    k = int(rng.integers(2))
    sl = slice(6 * k, 6 * k + 6)
    if kind == "jitter":
        full[:, sl] += rng.normal(0.0, 0.010, size=full[:, sl].shape)
    elif kind == "spike":
        f = int(rng.integers(1, SPAN - 2))
        d = rng.normal(size=3)
        full[f:f + 2, sl] += (d / np.linalg.norm(d) * 0.10).astype(np.float32)
    elif kind == "freeze":
        f = int(rng.integers(1, SPAN - 15))
        full[f:f + 15, sl] = full[f - 1, sl]
    elif kind == "swap":
        f = int(rng.integers(15, 46))
        left = full[f:, 0:6].copy()
        full[f:, 0:6] = full[f:, 6:12]
        full[f:, 6:12] = left
    elif kind == "drift":
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        full[:, sl] += (np.linspace(0.0, 0.10, SPAN)[:, None, None] * d[None, None, :]).astype(np.float32)
    else:
        raise ValueError(kind)
    c2 = full[PAST - 1]
    return full[:PAST] - c2, c2, full[PAST:] - c2


def sample_rules(x_rel, cur, y) -> dict:
    """The step-3 rule scores on one 60-frame sample (distances are unchanged by the camera transform)."""
    full = np.concatenate([x_rel + cur[None], y + cur[None]], axis=0).astype(np.float64)
    st, jit, frz = [], [], []
    for k in range(2):
        w = full[:, 6 * k]
        st.append(steps(w))
        jit.append(1000 * rms(jitter_residual(w)))
        frz.append(longest_run(frozen(full[:, 6 * k:6 * k + 6])))
    return {"jitter_mm": max(jit), "max_step_mm": 1000 * max(float(s.max()) for s in st),
            "freeze_run": max(frz), "swap_mm": 1000 * float(np.minimum(st[0], st[1]).max())}


# ---------------------------------------------------------------- model
def to_xy(x_rel, cur, y):
    x = np.concatenate([x_rel.reshape(len(x_rel), -1), cur.reshape(len(cur), -1)], axis=1) * 100.0  # cm
    return x.astype(np.float32), (y.reshape(len(y), -1) * 100.0).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int = 512, p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, dout),
        )

    def forward(self, x):
        return self.net(x)


def train_one(xtr, ytr, xva, yva, seed, epochs, batch, lr, patience):
    torch.manual_seed(seed)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xt, yt = torch.from_numpy((xtr - mu) / sd), torch.from_numpy(ytr)
    xv, yv = torch.from_numpy((xva - mu) / sd), torch.from_numpy(yva)
    model = MLP(xt.shape[1], yt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, ep_run = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            loss = F.mse_loss(model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.mse_loss(model(xv), yv))
        ep_run = ep + 1
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, mu, sd, ep_run, best


def l2_mm(pred_cm: np.ndarray, y_cm: np.ndarray) -> tuple[float, float]:
    d = np.linalg.norm(pred_cm.reshape(-1, FUT, NKP, 3) - y_cm.reshape(-1, FUT, NKP, 3), axis=-1) * 10.0
    return float(d.mean()), float(d[:, -1].mean())


def per_sample_mm(pred_cm: np.ndarray, y_cm: np.ndarray) -> np.ndarray:
    return (np.linalg.norm(pred_cm.reshape(-1, FUT, NKP, 3) - y_cm.reshape(-1, FUT, NKP, 3), axis=-1) * 10.0).mean(axis=(1, 2))


def predict(model, mu, sd, x):
    with torch.no_grad():
        return model(torch.from_numpy((x - mu) / sd)).numpy()


# ---------------------------------------------------------------- training sets
def make_training_set(cond: str, seed: int, tr: np.ndarray, orders: dict, data: dict, thr: dict):
    """Return (x_rel, cur, y, idx, info) for one condition. Corruption is seeded per (seed, kind, fraction),
    so 'random_half_inject30' and the qc_filter runs see exactly the same corrupted samples."""
    perm = np.random.default_rng(1000 + seed).permutation(tr)
    n = len(tr)
    order, conf_order = orders["quality"], orders["conf"]
    subsets = {
        "full": tr, "random_half": perm[: n // 2], "random_quarter": perm[: n // 4], "random_eighth": perm[: n // 8],
        "top_half": order[: n // 2], "bottom_half": order[-(n // 2):], "top_quarter": order[: n // 4],
        "conf_top_half": conf_order[: n // 2], "conf_bottom_half": conf_order[-(n // 2):],
        "top_half_speed_matched": orders["speed_top"], "bottom_half_speed_matched": orders["speed_bottom"],
    }
    info = {"n_corrupted": 0, "n_removed": 0, "n_removed_corrupted": 0}
    if cond in subsets:
        idx = subsets[cond]
        return data["x_rel"][idx], data["cur"][idx], data["y"][idx], idx, info

    # corrupted variants of the random half
    if cond.startswith("random_half_inject"):
        kind, frac = "mixed", int(cond[-2:]) / 100.0
    elif cond.startswith("inject30_"):
        kind, frac = "mixed", 0.30
    else:  # defect_type, e.g. "spike30"
        kind, frac = cond[:-2], int(cond[-2:]) / 100.0
    idx = subsets["random_half"]
    x_rel, cur, y = data["x_rel"][idx].copy(), data["cur"][idx].copy(), data["y"][idx].copy()
    code = {"mixed": 0, **{d: i + 1 for i, d in enumerate(DEFECTS)}}[kind]
    crng = np.random.default_rng(2000 + 100 * seed + 10 * code + int(frac * 10))
    bad = crng.choice(len(idx), size=int(frac * len(idx)), replace=False)
    for j in bad:
        x_rel[j], cur[j], y[j] = corrupt(x_rel[j], cur[j], y[j], crng, kind)
    info["n_corrupted"] = len(bad)
    if cond.endswith("_filter"):
        is_bad = np.zeros(len(idx), dtype=bool)
        is_bad[bad] = True
        if cond.endswith("oracle_filter"):
            keep = ~is_bad
        else:
            scores = [sample_rules(x_rel[j], cur[j], y[j]) for j in range(len(idx))]
            flagged = np.array([any(s[r] > thr[r] for r in RULES) for s in scores])
            keep = ~flagged
        info["n_removed"] = int((~keep).sum())
        info["n_removed_corrupted"] = int((~keep & is_bad).sum())
        x_rel, cur, y, idx = x_rel[keep], cur[keep], y[keep], idx[keep]
    return x_rel, cur, y, idx, info


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--derived-dir", default="data/derived")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--split-seed", type=int, default=20261117)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    args = ap.parse_args()
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))

    derived, out = Path(args.derived_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_dataset(args.data_dir, derived / f"predictor_samples_stride{args.stride}.npz", args.stride)
    q = quality_score(data)
    thr = json.loads((out / "injection_meta.json").read_text(encoding="utf-8"))["thresholds_p95_of_real_windows"]

    eps = np.unique(data["episode"])
    rng = np.random.default_rng(args.split_seed)
    rng.shuffle(eps)
    n_tr, n_va = int(0.7 * len(eps)), int(0.1 * len(eps))
    split = {"train": eps[:n_tr], "val": eps[n_tr:n_tr + n_va], "test": eps[n_tr + n_va:]}
    sel = {k: np.flatnonzero(np.isin(data["episode"], v)) for k, v in split.items()}
    tr, va, te = sel["train"], sel["val"], sel["test"]
    te_clean = te[q[te] >= np.median(q[te])]
    clean_mask = np.isin(te, te_clean)
    speed = sample_speed(data)
    conf_rank = pd.Series(data["conf"]).rank(pct=True).fillna(0.5).to_numpy()
    orders = {
        "quality": tr[np.argsort(-q[tr], kind="stable")],
        "conf": tr[np.argsort(-conf_rank[tr], kind="stable")],
        "speed_top": speed_matched(tr, q, speed, top=True),
        "speed_bottom": speed_matched(tr, q, speed, top=False),
    }
    cuts = np.quantile(speed[te], [1 / 3, 2 / 3])
    tert = np.digitize(speed[te], cuts)  # 0 slow, 1 medium, 2 fast
    from scipy.stats import spearmanr
    rho_q_speed = float(spearmanr(q, speed)[0])
    log(f"samples: all {len(q)}, train {len(tr)}, val {len(va)}, test {len(te)} (clean half {len(te_clean)}); "
        f"spearman(quality, speed) = {rho_q_speed:.3f}; test speed tertile cuts {np.round(cuts, 3).tolist()} m/s")

    xva, yva = to_xy(data["x_rel"][va], data["cur"][va], data["y"][va])
    xte, yte = to_xy(data["x_rel"][te], data["cur"][te], data["y"][te])

    zero = np.zeros_like(yte)
    v = (data["x_rel"][te][:, -1] - data["x_rel"][te][:, -6]) / 5.0
    cv = (v[:, None] * np.arange(1, FUT + 1)[None, :, None, None]).reshape(len(te), -1) * 100.0
    baselines = {}
    for name, pred in (("zero_velocity", zero), ("constant_velocity", cv)):
        a, f = l2_mm(pred, yte)
        ac, fc = l2_mm(pred[clean_mask], yte[clean_mask])
        baselines[name] = {"avg_mm": round(a, 2), "final_mm": round(f, 2), "clean_avg_mm": round(ac, 2), "clean_final_mm": round(fc, 2)}
    log("baselines " + json.dumps(baselines))

    config = f"stride={args.stride};epochs={args.epochs};batch={args.batch};lr={args.lr};patience={args.patience};split={args.split_seed}"
    runs_csv = out / "predictor_runs.csv"
    done = set()
    if runs_csv.exists():
        prev = pd.read_csv(runs_csv)
        prev = prev[prev["config"] == config]
        done = set(zip(prev["experiment"], prev["condition"], prev["seed"]))
        log(f"resuming: {len(done)} runs already done with this config")

    specs = [s for s in SPECS if args.only is None or s[0] in args.only]
    t0 = time.time()
    for seed in args.seeds:
        for exp, cond in specs:
            if (exp, cond, seed) in done:
                continue
            x_rel, cur, y, idx, info = make_training_set(cond, seed, tr, orders, data, thr)
            xtr, ytr = to_xy(x_rel, cur, y)
            model, mu, sd, n_ep, best_val = train_one(xtr, ytr, xva, yva, seed, args.epochs, args.batch, args.lr, args.patience)
            pred = predict(model, mu, sd, xte)
            a, f = l2_mm(pred, yte)
            ac, fc = l2_mm(pred[clean_mask], yte[clean_mask])
            ps = per_sample_mm(pred, yte)
            row = {
                "experiment": exp, "condition": cond, "seed": seed, "n_train": len(xtr), **info,
                "train_quality_mean": round(float(q[idx].mean()), 4),
                "train_speed_mean": round(float(speed[idx].mean()), 4), "epochs_run": n_ep,
                "val_mse_cm2": round(best_val, 4), "avg_mm": round(a, 2), "final_mm": round(f, 2),
                "clean_avg_mm": round(ac, 2), "clean_final_mm": round(fc, 2),
                "slow_avg_mm": round(float(ps[tert == 0].mean()), 2), "mid_avg_mm": round(float(ps[tert == 1].mean()), 2),
                "fast_avg_mm": round(float(ps[tert == 2].mean()), 2), "config": config,
            }
            new = pd.DataFrame([row])
            if runs_csv.exists():  # rewrite the whole table so newly added columns stay aligned
                new = pd.concat([pd.read_csv(runs_csv), new], ignore_index=True)
            new.to_csv(runs_csv, index=False, encoding="utf-8")
            log(f"  seed {seed} {exp:11s} {cond:24s} n={len(xtr):6d} avg {a:6.2f} final {f:6.2f} "
                f"clean avg {ac:6.2f} ({n_ep} ep, {time.time() - t0:.0f}s)")

    res = pd.read_csv(runs_csv)
    res = res[res["config"] == config]
    metrics = [m for m in ("avg_mm", "final_mm", "clean_avg_mm", "clean_final_mm", "slow_avg_mm", "mid_avg_mm",
                           "fast_avg_mm", "train_speed_mean") if m in res.columns]
    summ = res.groupby(["experiment", "condition"], sort=False).agg(
        n_train=("n_train", "mean"), seeds=("seed", "nunique"),
        **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")},
        n_removed=("n_removed", "mean"), n_removed_corrupted=("n_removed_corrupted", "mean"),
        n_corrupted=("n_corrupted", "mean"),
    ).round(2).reset_index()
    summ.to_csv(out / "predictor_summary.csv", index=False, encoding="utf-8")
    meta = {
        "task": "predict next 30 frames (1 s) of 12 keypoints from the past 30 frames, camera frame at current time",
        "samples": {"all": int(len(q)), "train": int(len(tr)), "val": int(len(va)), "test": int(len(te)),
                    "test_clean_half": int(len(te_clean))},
        "episodes": {k: int(len(v)) for k, v in split.items()},
        "stride_frames": args.stride,
        "model": "MLP 1116-512-512-1080, GELU, dropout 0.1, AdamW, MSE in cm, early stop on validation",
        "training": {"epochs_max": args.epochs, "batch": args.batch, "lr": args.lr, "patience": args.patience,
                     "seeds": args.seeds, "split_seed": args.split_seed},
        "quality_score": "rank-average of window confidence (higher better), wrist jitter, max wrist step, frozen frames",
        "defects": "per corrupted sample, one hand: jitter 10 mm sigma; spike 10 cm for 2 frames; 15-frame freeze; "
                   "left/right swap from a random frame; linear drift reaching 10 cm over the 2-s span. "
                   "'mixed' picks one of the five at random.",
        "rule_filter_thresholds": thr,
        "spearman_quality_vs_speed": round(rho_q_speed, 3),
        "test_speed_tertile_cuts_mps": [round(float(c), 4) for c in cuts],
        "baselines_test": baselines,
        "note": "Split of the EgoDex test set by episode; not comparable with the paper's benchmark numbers.",
    }
    (out / "predictor_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        log(summ.to_string(index=False))
    log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
