"""External check of the VLM description checker on EPIC-KITCHENS-100, with human narrations as the reference.

For each sampled segment (src/epic_make_sample.py) the same frames from its clip (src/epic_fetch_clips.py) are
judged three times with the EgoDex settings of vlm_check.py (same model, calibrated prompt, 16 frames, width 640):
- true: the segment's own label (verb + noun)                                    -> right answer "match";
- swapped: the verb replaced by its opposite (take/put, open/close, insert/remove, turn-on/off) -> "mismatch";
- other: another sampled segment's label (different verb pair, noun class, video)  -> "mismatch".
The labels come from EPIC's human annotation, so unlike on EgoDex the reference is neither the author's own
judgment nor GPT-4 output. EPIC still has some label noise, which pushes the "true" mismatch rate up a little, and
its labels are short ("open drawer"), unlike EgoDex's sentences, so the rates are a reference point for reading the
EgoDex numbers, not a transfer of them.

Local: data/derived/epic_vlm_check.jsonl (one judgment per line, resumable; it quotes labels).
Public: outputs/epic_vlm_summary.json (aggregates, with the EgoDex numbers next to them) and
outputs/epic_vlm_verdicts.csv (narration id, verb, kind, verdict, error type). EPIC-KITCHENS-100 annotations are
CC BY-NC 4.0 (Damen et al., IJCV 2022); attribution in the README.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402
from epic_fetch_clips import length_ok, probe  # noqa: E402
from vlm_check import EXPECTED_TYPE, judge, prompt_id, sample_frames, wilson  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
KINDS = {"true": "desc_true", "swapped": "desc_swapped", "other": "desc_other"}
EXPECT = {"true": "match", "swapped": "mismatch", "other": "mismatch"}
TYPED = {"swapped": EXPECTED_TYPE["direction_swap"], "other": EXPECTED_TYPE["cross_task"]}
PAIR_ORDER = ["take/put", "open/close", "insert/remove", "turn-on/turn-off"]


def binom_two_sided(k: int, n: int) -> float | None:
    """Exact two-sided sign test against 50/50 (used on the clips where the label and its reverse got different
    verdicts: did the model side with the label more often than chance?)."""
    if n == 0:
        return None
    tail = sum(comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def rate(sub: pd.DataFrame, verdict: str = "mismatch") -> dict:
    n = len(sub)
    k = int((sub["verdict"] == verdict).sum())
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": round(k / n, 4) if n else None,
            "ci95": [round(lo, 4), round(hi, 4)] if n else None}


def report(results: Path, sample: pd.DataFrame, pid: str, model: str, frames: int, width: int,
           summary_path: Path, verdicts_path: Path, egodex_summary: Path) -> None:
    rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines() if x.strip()]
    df = pd.DataFrame(rows)
    df = df[(df["prompt_id"] == pid) & (df["model"] == model) & (df["frames"] == frames) & (df["width"] == width)
            & (df["narration_id"].isin(sample["narration_id"]))]
    df = df.drop_duplicates(["narration_id", "kind"], keep="last")
    segs = df["narration_id"].nunique()

    groups = []
    for kind in KINDS:
        sub = df[df["kind"] == kind]
        g = {"group": kind, "expected": EXPECT[kind], **{f"n_{v}": int((sub["verdict"] == v).sum())
                                                         for v in ("match", "mismatch", "unsure")},
             "mismatch": rate(sub)}
        g["n_unparsed"] = int(len(sub) - g["n_match"] - g["n_mismatch"] - g["n_unsure"])
        g["mismatch_error_types"] = {str(k): int(v) for k, v in
                                     sub[sub["verdict"] == "mismatch"]["error_type"].value_counts().items()}
        if kind in TYPED:
            mm = sub[sub["verdict"] == "mismatch"]
            g["mismatch_with_expected_type"] = int(mm["error_type"].isin(TYPED[kind]).sum())
            true_v = df[df["kind"] == "true"].set_index("narration_id")["verdict"]
            g["true_match_and_this_mismatch"] = int(sum(1 for i, v in zip(sub["narration_id"], sub["verdict"])
                                                        if v == "mismatch" and true_v.get(i) == "match"))
        groups.append(g)

    by_pair = []
    for p in PAIR_ORDER:
        sp = df[df["pair"] == p]
        by_pair.append({"pair": p, "segments": int(sp["narration_id"].nunique()),
                        **{f"{k}_mismatch": rate(sp[sp["kind"] == k]) for k in KINDS}})
    by_verb = []
    for v in sample["verb_key"].drop_duplicates():
        sv = df[df["verb_key"] == v]
        by_verb.append({"verb_in_clip": v, "true_mismatch": rate(sv[sv["kind"] == "true"]),
                        "swapped_mismatch": rate(sv[sv["kind"] == "swapped"])})

    t = df[df["kind"] == "true"].set_index("narration_id")["verdict"]
    s = df[df["kind"] == "swapped"].set_index("narration_id")["verdict"]
    cells = {"consistent_with_label": 0, "model_says_label_reversed": 0, "both_match": 0, "both_mismatch": 0,
             "other": 0}
    for i in s.index.intersection(t.index):
        a, b = t[i], s[i]
        if a == "match" and b == "mismatch":
            cells["consistent_with_label"] += 1
        elif a == "mismatch" and b == "match":
            cells["model_says_label_reversed"] += 1
        elif a == b == "match":
            cells["both_match"] += 1
        elif a == b == "mismatch":
            cells["both_mismatch"] += 1
        else:
            cells["other"] += 1
    right, wrong = cells["consistent_with_label"], cells["model_says_label_reversed"]
    paired = sum(cells.values())
    cells["same_verdict_share"] = round((cells["both_match"] + cells["both_mismatch"]) / paired, 4) if paired else None
    cells["sign_test_p_two_sided"] = binom_two_sided(right, right + wrong)

    egodex = None
    if egodex_summary.exists():
        e = json.loads(egodex_summary.read_text(encoding="utf-8"))
        er = {r["group"]: r for r in e.get("rows", [])}
        egodex = {"source": "outputs/vlm_check_summary.json (EgoDex test set, 80 episodes; GPT-4 descriptions, "
                            "not human-checked)",
                  "original_mismatch": {k: er["original"][k] for k in ("n", "mismatch", "mismatch_rate")},
                  "direction_swap_mismatch": {k: er["direction_swap"][k] for k in ("n", "mismatch", "mismatch_rate")},
                  "cross_task_mismatch": {k: er["cross_task"][k] for k in ("n", "mismatch", "mismatch_rate")},
                  "direction_pairs": e.get("meta", {}).get("direction_pairs")}

    summary = {
        "what": "VLM description checker on EPIC-KITCHENS-100 validation clips, judged against EPIC's human labels",
        "data": {"source": "EPIC-KITCHENS-100 validation split (Damen et al., IJCV 2022), CC BY-NC 4.0; "
                           "only short clips of the sampled segments were fetched, none are published",
                 "segments": segs, "videos": int(sample[sample["narration_id"].isin(df["narration_id"])]["video_id"].nunique()),
                 "participants": int(sample[sample["narration_id"].isin(df["narration_id"])]["participant_id"].nunique()),
                 "per_verb": {v: int((sample[sample["narration_id"].isin(df["narration_id"])]["verb_key"] == v).sum())
                              for v in sample["verb_key"].drop_duplicates()},
                 "selection": "src/epic_make_sample.py, seed 7: 1-6 s, no overlap with the opposite verb on the same "
                              "noun, noun present in the narration and not a pronoun object, turn-on/off on "
                              "appliances only, at most 2 per video and verb"},
        "setup": {"model": model, "prompt": pid, "frames": frames, "width": width,
                  "descriptions": "true = verb + noun from the label; swapped = opposite verb + same noun; other = "
                                  "another sampled segment's verb + noun (different pair, noun class, video)",
                  "judgments": int(len(df)),
                  "mean_seconds_per_judgment": round(float(df["wall_s"].mean()), 2) if len(df) else None},
        "groups": groups, "by_pair": by_pair, "by_verb_in_clip": by_verb, "true_vs_swapped": cells,
        "egodex_same_settings": egodex,
        "notes": [
            "true mismatch = the model rejected a human label (false alarm, up to EPIC's own label noise). Its "
            "reasons are mostly 'the main action is something else': EPIC labels are atomic and overlap with what "
            "else the hands do (turning on the tap while washing), so part of this is label granularity.",
            "same_verdict_share = clips where the label and its exact reverse got the same verdict; the sign test "
            "asks whether, on the other clips, the model sided with the label more often than chance.",
            "swapped mismatch = direction errors caught; other mismatch = unrelated labels caught.",
            "Labels are short kitchen actions, EgoDex descriptions are long sentences: compare directions of "
            "effects, not exact rates.",
        ],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    df[["narration_id", "video_id", "pair", "verb_key", "kind", "verdict", "error_type"]].sort_values(
        ["narration_id", "kind"]).to_csv(verdicts_path, index=False, encoding="utf-8")

    print(f"{segs} segments, {len(df)} judgments, {summary['setup']['mean_seconds_per_judgment']} s each")
    for g in groups:
        m = g["mismatch"]
        extra = (f"; expected type {g['mismatch_with_expected_type']}, with true=match {g['true_match_and_this_mismatch']}"
                 if "mismatch_with_expected_type" in g else "")
        print(f"  {g['group']:8s} mismatch {m['k']}/{m['n']} = {m['rate']:.1%} (CI {m['ci95'][0]:.0%}-{m['ci95'][1]:.0%})"
              f"; unsure {g['n_unsure']}, unparsed {g['n_unparsed']}{extra}")
    for p in by_pair:
        print(f"  {p['pair']:17s} " + "  ".join(f"{k} {p[k + '_mismatch']['k']}/{p[k + '_mismatch']['n']}"
                                                  for k in KINDS))
    print("  swapped caught by verb in clip: " + ", ".join(
        f"{v['verb_in_clip']} {v['swapped_mismatch']['k']}/{v['swapped_mismatch']['n']}" for v in by_verb))
    print(f"  true vs swapped: {cells}")
    if egodex:
        print(f"  EgoDex (same settings): original {egodex['original_mismatch']['mismatch']}/"
              f"{egodex['original_mismatch']['n']}, direction swap {egodex['direction_swap_mismatch']['mismatch']}/"
              f"{egodex['direction_swap_mismatch']['n']}, cross task {egodex['cross_task_mismatch']['mismatch']}/"
              f"{egodex['cross_task_mismatch']['n']}")
    print(f"wrote {summary_path.relative_to(ROOT)} and {verdicts_path.relative_to(ROOT)}")


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "derived" / "epic_sample.csv")
    ap.add_argument("--clips", type=Path, default=ROOT / "data" / "external" / "epic100" / "clips")
    ap.add_argument("--results", type=Path, default=ROOT / "data" / "derived" / "epic_vlm_check.jsonl")
    ap.add_argument("--summary", type=Path, default=ROOT / "outputs" / "epic_vlm_summary.json")
    ap.add_argument("--verdicts", type=Path, default=ROOT / "outputs" / "epic_vlm_verdicts.csv")
    ap.add_argument("--egodex-summary", type=Path, default=ROOT / "outputs" / "vlm_check_summary.json")
    ap.add_argument("--api", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="gemma4:12b-it-q4_K_M")
    ap.add_argument("--prompt", default="calibrated")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=0, help="only the first N segments (smoke test)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    sample = pd.read_csv(args.sample)
    pid = prompt_id(args.prompt)
    if not args.report_only:
        have = []
        for r in sample.itertuples():
            clip = args.clips / f"{r.narration_id}.mp4"
            if clip.exists():
                n, d = probe(clip)
                if length_ok(n, d, r.dur):
                    have.append(r)
        print(f"{len(have)} of {len(sample)} segments have a clip", flush=True)
        if args.limit:
            have = have[:args.limit]
        done = set()
        if args.results.exists():
            for x in args.results.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    j = json.loads(x)
                    if (j.get("prompt_id"), j.get("model"), j.get("frames"), j.get("width")) == \
                            (pid, args.model, args.frames, args.width) and j.get("verdict") != "error":
                        done.add((j["narration_id"], j["kind"]))
        todo = [r for r in have if any((r.narration_id, k) not in done for k in KINDS)]
        print(f"{len(todo)} segments still to judge", flush=True)
        args.results.parent.mkdir(parents=True, exist_ok=True)
        fails, t0 = 0, time.time()
        with args.results.open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                imgs = sample_frames(args.clips / f"{r.narration_id}.mp4", args.frames, args.width)
                for kind, col in KINDS.items():
                    if (r.narration_id, kind) in done:
                        continue
                    desc = getattr(r, col)
                    for attempt in range(3):
                        try:
                            parsed, stats = judge(args.api, args.model, args.prompt, desc, imgs, args.num_ctx)
                            fails = 0
                            break
                        except Exception as e:  # noqa: BLE001 - network hiccup on the tunnel; retry, then give up
                            fails += 1
                            print(f"  call failed ({e}); retry {attempt + 1}", flush=True)
                            time.sleep(5)
                    else:
                        print("three failures in a row on one call; stopping (results so far are kept)")
                        return 2
                    rec = {"narration_id": r.narration_id, "video_id": r.video_id,
                           "participant_id": r.participant_id, "pair": r.pair, "verb_key": r.verb_key,
                           "kind": kind, "desc": desc, **parsed, **stats, "n_frames": len(imgs),
                           "model": args.model, "prompt_id": pid, "frames": args.frames, "width": args.width,
                           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} segments, {time.time() - t0:.0f}s", flush=True)
    if not args.results.exists():
        print("no results yet")
        return 1
    report(args.results, sample, pid, args.model, args.frames, args.width, args.summary, args.verdicts,
           args.egodex_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
