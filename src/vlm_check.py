"""Experiment 2b: a VLM checks the language descriptions, scored with errors we create ourselves.

For each episode on the review list (data/review/review_items.csv, 80 episodes) the model sees
K frames in time order and judges whether a description matches the video. Besides the original
description it is asked about versions with errors injected on purpose, so the correct answer
for those is known by construction:
  direction_swap  reversible tasks only: the dataset's other candidate (e.g. insert <-> remove)
  cross_task      the description of an episode from another task
  detail_edit     one changed detail: a count, the hand (left/right) or a colour
  extra_step      an invented step appended at the end

The prompt and frame count are chosen on a separate dev set (--dev N: episodes that are not on
the review list), then frozen before the review-list run, so tuning never sees the evaluated
episodes.

Limits (repeated in the report): detection rates only show how sensitive the model is to errors
we made up; they are not the real error rate of the GPT-4 descriptions. The originals are assumed
correct but nobody has verified them, and direction_swap relies on which_llm_description.

Files that contain description text stay under data/derived/ (EgoDex is CC BY-NC-ND);
outputs/ only receives counts, rates and a figure. Needs an Ollama server with a vision model
(default: the GPU workstation through the local tunnel on port 11435).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402

EXPECTED_TYPE = {"direction_swap": {"direction"}, "cross_task": {"object", "action", "other"},
                 "extra_step": {"extra_step"}, "count": {"count"}, "hand": {"hand"}, "color": {"color"}}
ERROR_TYPES = ["none", "direction", "object", "action", "count", "hand", "color", "extra_step", "other"]
JSON_TAIL = """Reply as JSON: verdict = match / mismatch / unsure; error_type = the main problem if it is a mismatch
(direction, object, action, count, hand, color, extra_step, other), otherwise none;
reason = one short sentence."""

PROMPTS = {
    # v1: checks everything strictly. In the smoke test it called almost everything a mismatch.
    "strict": ("""You are checking a text description of a first-person video of a person's hands doing a task.
The {k} images are frames from that video in time order: image 1 is the start, image {k} is the end.

Description: "{desc}"

Decide from what is visible whether the description matches the video. Check the objects, the action,
the direction (for example putting in vs taking out, assembling vs disassembling), counts, which hand,
colours, and whether every step it mentions actually happens. Ignore small wording or spelling issues.
If the frames do not show enough to decide, answer "unsure".

""" + JSON_TAIL, False),
    # v2: mismatch only on a clear contradiction; describe the frames first; compare start and end.
    "calibrated": ("""You are checking a text description of a first-person video of a person's hands doing a task.
The {k} images are frames sampled from that video in time order: image 1 is the start, image {k} is the end.
Frames are sparse, so short moments of the action can fall between them.

Description: "{desc}"

First write what you see: the objects, what the hands do, and the state at the start (image 1)
compared with the end (image {k}). Then judge.
Answer "mismatch" only if the frames clearly contradict the description, for example:
- the main object or action is different;
- the direction is reversed (putting in vs taking out, assembling vs disassembling; compare start and end);
- a count, the hand (left/right) or a colour is clearly wrong;
- the description includes a step that clearly never happens (an object or person it needs never appears).
If the description is consistent with the frames, even if not every moment is visible, answer "match".
If the frames are too unclear to tell, answer "unsure". Ignore small wording or spelling issues.

Reply as JSON: observation = what you see (one or two sentences); verdict = match / mismatch / unsure;
error_type = the main problem if it is a mismatch (direction, object, action, count, hand, color,
extra_step, other), otherwise none; reason = one short sentence.""", True),
}

NUM_SWAP = {"two": "four", "three": "five", "four": "two", "five": "three", "twelve": "six", "nineteen": "seven"}
HAND_SWAP = {"right hand": "left hand", "left hand": "right hand"}
COLOR_SWAP = {"white": "black", "black": "white", "red": "blue", "blue": "red", "green": "yellow",
              "yellow": "green", "pink": "green", "hotpink": "blue", "gray": "yellow", "brown": "white",
              "lavender": "orange", "navy": "red", "clear": "black", "purple": "orange", "orange": "purple"}
EXTRA_STEPS = [("then hand it to another person", ("person", "hand it")),
               ("then throw it into a trash can", ("trash", "bin")),
               ("then pour water over it", ("water", "faucet", "wash", "wet", "clean", "dish"))]

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never route localhost via a proxy


def schema(with_observation: bool) -> dict:
    props = {"verdict": {"type": "string", "enum": ["match", "mismatch", "unsure"]},
             "error_type": {"type": "string", "enum": ERROR_TYPES}, "reason": {"type": "string"}}
    if with_observation:
        props = {"observation": {"type": "string"}, **props}
    return {"type": "object", "properties": props, "required": list(props)}


def prompt_id(name: str) -> str:
    return name + "-" + hashlib.sha256(PROMPTS[name][0].encode("utf-8")).hexdigest()[:8]


def detail_edit(desc: str) -> tuple[str | None, str | None]:
    """Change exactly one detail: the first count word, else the hand, else the first colour."""
    for kind, table in (("count", NUM_SWAP), ("hand", HAND_SWAP), ("color", COLOR_SWAP)):
        words = sorted(table, key=len, reverse=True)
        m = re.search(r"\b(" + "|".join(map(re.escape, words)) + r")\b", desc, flags=re.I)
        if m:
            new = table[m.group(1).lower()]
            if m.group(1)[0].isupper():
                new = new[0].upper() + new[1:]
            return kind, desc[:m.start()] + new + desc[m.end():]
    return None, None


def extra_step(desc: str, rng: random.Random) -> str | None:
    low = desc.lower()
    options = [s for s, avoid in EXTRA_STEPS if not any(a in low for a in avoid)]
    if not options:
        return None
    return desc.rstrip().rstrip(".") + ", " + rng.choice(options) + "."


def build_cases(items: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for it in items:
        base = {"id": it["id"], "task": it["task"]}
        cases.append({**base, "variant": "original", "kind": "", "desc": it["desc"]})
        if it["reversible"] and it["desc_alt"]:
            cases.append({**base, "variant": "direction_swap", "kind": "", "desc": it["desc_alt"]})
        other = rng.choice([o for o in items if o["task"] != it["task"]])
        cases.append({**base, "variant": "cross_task", "kind": other["id"], "desc": other["desc"]})
        kind, edited = detail_edit(it["desc"])
        if edited:
            cases.append({**base, "variant": "detail_edit", "kind": kind, "desc": edited})
        added = extra_step(it["desc"], rng)
        if added:
            cases.append({**base, "variant": "extra_step", "kind": "", "desc": added})
    return cases


def review_items(path: str) -> list[dict]:
    items = pd.read_csv(path, encoding="utf-8-sig", keep_default_na=False).to_dict("records")
    for it in items:
        it["reversible"] = str(it["reversible"]) == "True"
    return items


def dev_items(data_dir: Path, exclude_ids: set[str], exclude_tasks: set[str], n: int, seed: int) -> list[dict]:
    """n episodes that are not on the review list, one per task, preferring tasks not on it either."""
    import h5py
    from egodex_common import clean_attr, list_episodes

    eps = [e for e in list_episodes(data_dir) if e["mp4"].exists() and e["id"] not in exclude_ids]
    by_task: dict[str, list] = {}
    for e in eps:
        by_task.setdefault(e["task"], []).append(e)
    rng = random.Random(seed)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    tasks.sort(key=lambda t: t in exclude_tasks)  # stable: unseen tasks first
    items = []
    for t in tasks[:n]:
        e = rng.choice(by_task[t])
        with h5py.File(e["path"], "r") as f:
            attrs = {k: clean_attr(v) for k, v in f.attrs.items()}
        d1, d2 = attrs.get("llm_description", ""), attrs.get("llm_description2")
        which = attrs.get("which_llm_description")
        reversible = which is not None and isinstance(d2, str) and d2.strip() not in ("", "None")
        use2 = reversible and str(which) == "2"
        items.append({"id": e["id"], "task": e["task"], "reversible": reversible,
                      "desc": d2 if use2 else d1, "desc_alt": (d1 if use2 else d2) if reversible else ""})
    return items


def sample_frames(mp4: Path, k: int, width: int) -> list[str]:
    """k frames evenly spaced from the first to the last frame, as base64 JPEG."""
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if n > 1 else [0]
    out, i, j = [], 0, 0
    while j < len(want):
        ok, frame = cap.read()
        if not ok:
            break
        if i == want[j]:
            h = int(round(frame.shape[0] * width / frame.shape[1]))
            small = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            ok2, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok2:
                out.append(base64.b64encode(jpg.tobytes()).decode("ascii"))
            j += 1
        i += 1
    cap.release()
    return out


def call(api: str, path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(api + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with _opener.open(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def judge(api: str, model: str, prompt: str, desc: str, images: list[str], num_ctx: int) -> tuple[dict, dict]:
    text, with_obs = PROMPTS[prompt]
    payload = {"model": model, "stream": False, "think": False, "format": schema(with_obs),
               "options": {"temperature": 0, "num_predict": 300, "num_ctx": num_ctx},
               "messages": [{"role": "user", "content": text.format(k=len(images), desc=desc), "images": images}]}
    t0 = time.time()
    r = call(api, "/api/chat", payload)
    stats = {"wall_s": round(time.time() - t0, 2), "prompt_tokens": r.get("prompt_eval_count"),
             "output_tokens": r.get("eval_count")}
    content = r.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        parsed = {k: parsed.get(k, "") for k in ("observation", "verdict", "error_type", "reason")}
    except json.JSONDecodeError:
        parsed = {"observation": "", "verdict": "unparsed", "error_type": "", "reason": content[:300]}
    return parsed, stats


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(df: pd.DataFrame) -> list[dict]:
    groups = [("original", df[df.variant == "original"], set())]
    for v in ("direction_swap", "cross_task"):
        groups.append((v, df[df.variant == v], EXPECTED_TYPE[v]))
    for kind in ("count", "hand", "color"):
        groups.append((f"detail_edit:{kind}", df[(df.variant == "detail_edit") & (df.kind == kind)], EXPECTED_TYPE[kind]))
    groups.append(("extra_step", df[df.variant == "extra_step"], EXPECTED_TYPE["extra_step"]))
    orig = df[df.variant == "original"].set_index("id").verdict
    rows = []
    for name, sub, expected in groups:
        n = len(sub)
        mm = int((sub.verdict == "mismatch").sum())
        ma = int((sub.verdict == "match").sum())
        lo, hi = wilson(mm, n)
        typed = int(sub[sub.verdict == "mismatch"].error_type.isin(expected).sum()) if expected else None
        # Paired view: the model accepted the original of the same episode AND rejected this version.
        pair_ok = None if name == "original" else int(sum(
            1 for i, v in zip(sub.id, sub.verdict) if v == "mismatch" and orig.get(i) == "match"))
        rows.append({"group": name, "n": n, "mismatch": mm, "match": ma, "unsure_or_unparsed": n - mm - ma,
                     "mismatch_rate": round(mm / n, 4) if n else None,
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                     "mismatch_with_expected_type": typed, "original_match_and_this_mismatch": pair_ok})
    return rows


def direction_pairs(df: pd.DataFrame) -> tuple[dict, list[str]]:
    """Reversible episodes: compare the verdict on the data's chosen description with the other candidate."""
    o = df[df.variant == "original"].set_index("id").verdict
    s = df[df.variant == "direction_swap"].set_index("id").verdict
    cells = {"consistent_with_data": 0, "model_says_data_reversed": 0, "both_match": 0, "both_mismatch": 0, "other": 0}
    suspects = []
    for i in s.index.intersection(o.index):
        a, b = o[i], s[i]
        if a == "match" and b == "mismatch":
            cells["consistent_with_data"] += 1
        elif a == "mismatch" and b == "match":
            cells["model_says_data_reversed"] += 1
            suspects.append(i)
        elif a == b == "match":
            cells["both_match"] += 1
        elif a == b == "mismatch":
            cells["both_mismatch"] += 1
        else:
            cells["other"] += 1
    return cells, suspects


def balance(rows: list[dict]) -> float:
    """Selection score used on the dev set: mean detection over injected-error groups minus the flag rate on originals."""
    det = [r["mismatch_rate"] for r in rows if r["group"] != "original" and r["n"]]
    orig = next(r["mismatch_rate"] for r in rows if r["group"] == "original")
    return round(sum(det) / len(det) - orig, 4)


def make_figure(rows: list[dict], pairs: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"original": "original description\n(flagged: false alarm, or a real error)",
              "direction_swap": "direction swapped\n(the other candidate)", "cross_task": "another task's description",
              "detail_edit:count": "one count changed", "detail_edit:hand": "left/right hand swapped",
              "detail_edit:color": "one colour changed", "extra_step": "invented extra step added"}
    rows = [r for r in rows if r["n"]]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 0.62 * len(rows) + 3.0),
                                  gridspec_kw={"height_ratios": [len(rows), 1.6]})
    parts = [("consistent_with_data", "fits the data's choice", "#55A868"),
             ("model_says_data_reversed", "model: the other one fits", "#C44E52"),
             ("both_match", "both 'match'", "#CCB974"), ("both_mismatch", "both 'mismatch'", "#8C8C8C"),
             ("other", "unsure", "#DDDDDD")]
    total, left = sum(pairs.values()), 0
    for key, lab, col in parts:
        v = pairs.get(key, 0)
        if v:
            ax2.barh(0, v, left=left, color=col, label=f"{lab}: {v}")
            ax2.text(left + v / 2, 0, str(v), ha="center", va="center", fontsize=9, color="white" if key != "other" else "k")
            left += v
    ax2.set_xlim(0, max(total, 1))
    ax2.set_yticks([])
    ax2.set_xlabel(f"reversible episodes (n={total}): verdict on the data's description vs the other candidate")
    ax2.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.55), frameon=False)
    for i, r in enumerate(rows):
        p = 100 * r["mismatch_rate"]
        color = "#8C8C8C" if r["group"] == "original" else "#4C72B0"
        ax.barh(i, p, color=color)
        ax.errorbar(p, i, xerr=[[p - 100 * r["ci95_low"]], [100 * r["ci95_high"] - p]], fmt="none", ecolor="k", capsize=3, lw=1)
        ax.text(min(100 * r["ci95_high"] + 1.5, 101), i, f"{p:.0f}%  ({r['mismatch']}/{r['n']})", va="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([labels.get(r["group"], r["group"]) for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("% judged 'mismatch' by the VLM (bars: 95% Wilson interval)")
    ax.set_title(title, fontsize=10)
    fig.text(0.01, 0.01, "Grey: original GPT-4 descriptions (not human-verified). Blue: errors we injected; the rate is "
             "sensitivity to these injected errors, not the real error rate.", fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report(results: Path, outputs: Path | None, derived: Path, args: argparse.Namespace) -> None:
    df = pd.read_json(results, lines=True)
    df["kind"] = df["kind"].fillna("")
    rows = summarize(df)
    pairs, suspects = direction_pairs(df)
    dev = []
    for p in sorted(derived.glob("vlm_dev_*.jsonl")):
        d = pd.read_json(p, lines=True)
        d["kind"] = d["kind"].fillna("")
        r = summarize(d)
        dev.append({"config": p.stem.replace("vlm_dev_", ""), "episodes": int(d.id.nunique()), "judgments": int(len(d)),
                    "selection_score": balance(r), "original_flag_rate": r[0]["mismatch_rate"]})
    meta = {"model": args.model, "prompt": prompt_id(args.prompt), "frames_per_episode": args.frames,
            "seed": args.seed, "episodes": int(df.id.nunique()), "judgments": int(len(df)),
            "selection_score": balance(rows), "direction_pairs": pairs,
            "dev_selection": dev,
            "mean_seconds_per_judgment": round(float(df.wall_s.mean()), 2),
            "total_minutes": round(float(df.wall_s.sum()) / 60, 1),
            "note": "合成错误的检出率只说明模型对人为制造的错误有多敏感，不是 GPT-4 描述的真实错误率；"
                    "原描述没有经过人工核对，被模型报错的原描述可能是误报，也可能是真错；"
                    "方向错依赖数据自带的 which_llm_description 字段，这个字段同样没有人工核对。"}
    print(json.dumps(meta, ensure_ascii=False))
    print(pd.DataFrame(rows).to_string(index=False))
    if outputs is None:
        return
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "vlm_check_summary.json").write_text(json.dumps({"meta": meta, "rows": rows}, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
    pd.DataFrame(rows).to_csv(outputs / "vlm_check_summary.csv", index=False, encoding="utf-8-sig")
    (outputs / "figures").mkdir(exist_ok=True)
    make_figure(rows, pairs, outputs / "figures" / "fig9_vlm_check.png",
                f"VLM description checker: {args.model}, {args.frames} frames, {meta['episodes']} episodes")
    # Local only (they hold model text about the data): the short list a human should check.
    flagged = df[(df.variant == "original") & (df.verdict == "mismatch")][["id", "task", "error_type", "observation", "reason"]]
    flagged.insert(2, "direction_suspect", flagged.id.isin(suspects))
    flagged.to_csv(derived / "vlm_flagged_originals.csv", index=False, encoding="utf-8-sig")
    print(f"flagged originals: {len(flagged)} (of which direction suspects: {len(suspects)}) -> "
          f"{derived / 'vlm_flagged_originals.csv'}")


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/review/review_items.csv")
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--results", default="data/derived/vlm_check.jsonl")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--api", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="gemma4:12b-it-q4_K_M")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="calibrated")
    ap.add_argument("--frames", type=int, default=16)  # chosen on the dev set (calibrated prompt, 16 frames)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=20261123)
    ap.add_argument("--dev", type=int, default=0, help="use N episodes that are NOT on the review list (prompt tuning)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (smoke test)")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--print-only", action="store_true", help="print the summary, write nothing to outputs/")
    args = ap.parse_args()

    results, derived = Path(args.results), Path(args.results).parent
    derived.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        items = review_items(args.items)
        if args.dev:
            items = dev_items(Path(args.data_dir), {it["id"] for it in items}, {it["task"] for it in items},
                              args.dev, args.seed)
        cases = build_cases(items, args.seed)
        if args.limit:
            keep = {it["id"] for it in items[:args.limit]}
            cases = [c for c in cases if c["id"] in keep]
        done = set()
        if results.exists():
            for line in results.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if r.get("prompt_id") != prompt_id(args.prompt) or r.get("frames") != args.frames:
                        print("results file was made with other settings; use another --results path")
                        return 2
                    done.add((r["id"], r["variant"]))
        todo = [c for c in cases if (c["id"], c["variant"]) not in done]
        print(f"{len(cases)} judgments planned, {len(cases) - len(todo)} already done, {len(todo)} to run", flush=True)
        frames_cache: dict[str, list[str]] = {}
        fails, t_start = 0, time.time()
        with results.open("a", encoding="utf-8") as f:
            for n_done, c in enumerate(todo, 1):
                if c["id"] not in frames_cache:
                    frames_cache.clear()
                    task, num = c["id"].split("/")
                    frames_cache[c["id"]] = sample_frames(Path(args.data_dir) / task / f"{num}.mp4", args.frames, args.width)
                imgs = frames_cache[c["id"]]
                for attempt in range(2):
                    try:
                        parsed, stats = judge(args.api, args.model, args.prompt, c["desc"], imgs, args.num_ctx)
                        fails = 0
                        break
                    except Exception as e:  # network or server error: retry once, stop after 3 in a row
                        err = e
                        time.sleep(5)
                else:
                    fails += 1
                    print(f"FAILED {c['id']} {c['variant']}: {err}", flush=True)
                    if fails >= 3:
                        print("3 failures in a row, stopping; rerun to resume", flush=True)
                        return 1
                    continue
                rec = {**c, **parsed, **stats, "n_frames": len(imgs), "model": args.model, "frames": args.frames,
                       "seed": args.seed, "prompt_id": prompt_id(args.prompt), "set": "dev" if args.dev else "review"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n_done % 20 == 0 or n_done == len(todo):
                    print(f"  {n_done}/{len(todo)} done, {time.time() - t_start:.0f}s", flush=True)
    report(results, None if (args.print_only or args.dev) else Path(args.outputs), derived, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
