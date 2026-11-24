"""Draw the EPIC-KITCHENS-100 sample for the external VLM check and estimate what fetching it costs.

For each of the 8 direction verbs (take/put, open/close, insert/remove, turn-on/turn-off) draw --per-side usable
segments from the videos whose size was looked up (data/derived/epic_video_sizes.csv), at most --per-video-side
per video and verb so no single kitchen dominates, and only segments with no overlap at all with the opposite verb
on the same noun class and whose narration contains the parsed noun (turn-on/off: appliances only). Each segment gets three descriptions in the same terse EPIC style, built from its human
labels: the true one (verb + noun), the verb swapped to its opposite, and another sampled segment's verb + noun
(different verb pair, noun class and video). Downloads nothing; the byte estimate assumes only the segment's own
seconds (plus a keyframe pre-roll and the file index) are fetched with HTTP range requests.

Inputs (local): data/derived/epic_direction_segments.csv, data/derived/epic_video_sizes.csv,
data/derived/epic_video_candidates.csv. Output (local; quotes narrations): data/derived/epic_sample.csv.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DER = ROOT / "data" / "derived"
SIDES = ["take", "put", "open", "close", "insert", "remove", "turn-on", "turn-off"]
EPIC55_VIDEOS = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos"


def noun_text(noun: str) -> str:
    """EPIC writes compound nouns head first ('bowl:washing:up' = 'washing up bowl'); turn that into plain words."""
    parts = str(noun).split(":")
    return " ".join(parts[1:] + parts[:1])


def label_consistent(narration: str, noun: str) -> bool:
    """The noun's head word appears in the narration (plural endings allowed). Drops parsing slips such as the
    narration 'remove slice' parsed with the noun 'bowl:salad', and narrations without a noun ('take out',
    'open it'), where the 'true' description would rest on a guess. The verb is not checked: EPIC maps
    inflections and synonyms on purpose ('closing' -> close, 'open tap' -> turn-on). Narrations whose object is a
    pronoun ('put it on plate') are dropped too: there the parsed noun is the place, not the object moved."""
    text = str(narration).lower()
    if set(text.replace(",", " ").split()) & {"it", "them", "this", "that", "these", "those"}:
        return False
    head = str(noun).split(":")[0].lower()
    return head[:max(3, len(head) - 2)] in text


def phrase(verb_key: str, noun: str) -> str:
    return f"{verb_key.replace('-', ' ')} {noun_text(noun)}"


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--segments", type=Path, default=DER / "epic_direction_segments.csv")
    ap.add_argument("--sizes", type=Path, default=DER / "epic_video_sizes.csv")
    ap.add_argument("--videos", type=Path, default=DER / "epic_video_candidates.csv")
    ap.add_argument("--nouns", type=Path,
                    default=ROOT / "data" / "external" / "epic100" / "annotations" / "EPIC_100_noun_classes.csv")
    ap.add_argument("--out", type=Path, default=DER / "epic_sample.csv")
    ap.add_argument("--per-side", type=int, default=20)
    ap.add_argument("--per-video-side", type=int, default=2)
    ap.add_argument("--max-dur", type=float, default=6.0)
    ap.add_argument("--preroll-s", type=float, default=1.0, help="keyframe pre-roll assumed in the byte estimate")
    ap.add_argument("--index-mb", type=float, default=1.0, help="file index (moov) assumed fetched per clip")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    seg = pd.read_csv(args.segments)
    sizes = pd.read_csv(args.sizes)
    sizes = sizes[(sizes["status"] == "ok") & (sizes["bytes"] > 0)]
    vids = pd.read_csv(args.videos)[["video_id", "last_stop_s"]]
    sizes = sizes.merge(vids, on="video_id", how="left")
    # Bitrate from the file size over the last annotated second: the real video is a bit longer, so this
    # slightly overestimates bytes per second (a conservative estimate).
    sizes["bytes_per_s"] = sizes["bytes"] / sizes["last_stop_s"]

    pool = seg[(seg["usable"]) & (seg["opp_overlap"] == 0) & (seg["dur"] <= args.max_dur)
               & (seg["verb_key"].isin(SIDES)) & (seg["video_id"].isin(sizes["video_id"]))].copy()
    ok = [label_consistent(n, o) for n, o in zip(pool["narration"], pool["noun"])]
    print(f"label check: {len(pool) - sum(ok)} of {len(pool)} pool segments dropped "
          "(noun head not in the narration, or the object is a pronoun)")
    pool = pool[ok].copy()
    # turn-on / turn-off only on appliances: the few others are misparses such as 'turn lid off' (unscrewing).
    category = dict(zip(pd.read_csv(args.nouns)["id"], pd.read_csv(args.nouns)["category"]))
    switch = pool["verb_key"].isin(["turn-on", "turn-off"])
    not_appliance = switch & (pool["noun_class"].map(category) != "appliances")
    print(f"turn-on/off not on an appliance: {int(not_appliance.sum())} dropped")
    pool = pool[~not_appliance].copy()
    rng = random.Random(args.seed)
    pool["r"] = [rng.random() for _ in range(len(pool))]
    picked = []
    short = {}
    for side in SIDES:
        cand = pool[pool["verb_key"] == side].sort_values("r")
        per_video: dict[str, int] = {}
        got = 0
        for r in cand.itertuples():
            if per_video.get(r.video_id, 0) >= args.per_video_side:
                continue
            per_video[r.video_id] = per_video.get(r.video_id, 0) + 1
            picked.append(r.Index)
            got += 1
            if got == args.per_side:
                break
        if got < args.per_side:
            short[side] = got
    s = pool.loc[picked].drop(columns="r").copy()
    s = s.merge(sizes[["video_id", "split", "bytes", "bytes_per_s"]], on="video_id", how="left")

    s["desc_true"] = [phrase(v, n) for v, n in zip(s["verb_key"], s["noun"])]
    s["desc_swapped"] = [phrase(o, n) for o, n in zip(s["opposite_key"], s["noun"])]
    # Another sampled segment's description: different verb pair, noun class and video.
    others, other_ids = [], []
    rows = list(s.itertuples())
    for r in rows:
        opts = [o for o in rows if o.pair != r.pair and o.noun_class != r.noun_class and o.video_id != r.video_id]
        o = rng.choice(opts)
        others.append(phrase(o.verb_key, o.noun))
        other_ids.append(o.narration_id)
    s["desc_other"] = others
    s["other_narration_id"] = other_ids
    s["url"] = [f"{EPIC55_VIDEOS}/{sp}/{p}/{v}.MP4" for sp, p, v in zip(s["split"], s["participant_id"], s["video_id"])]
    s["est_bytes"] = (s["bytes_per_s"] * (s["dur"] + args.preroll_s) + args.index_mb * 1024 * 1024).round().astype(int)

    cols = ["narration_id", "video_id", "participant_id", "split", "url", "start_timestamp", "stop_timestamp",
            "start_s", "stop_s", "dur", "pair", "verb_key", "opposite_key", "noun", "noun_class", "narration",
            "desc_true", "desc_swapped", "desc_other", "other_narration_id", "est_bytes"]
    s = s[cols].sort_values(["video_id", "start_s"])
    s.to_csv(args.out, index=False, encoding="utf-8")

    print(f"pool (usable, no opposite overlap, <= {args.max_dur}s, in {sizes['video_id'].nunique()} sized videos): "
          + ", ".join(f"{k} {int((pool['verb_key'] == k).sum())}" for k in SIDES))
    print(f"sample: {len(s)} segments from {s['video_id'].nunique()} videos, "
          f"{s['participant_id'].nunique()} participants; per side "
          + ", ".join(f"{k} {int((s['verb_key'] == k).sum())}" for k in SIDES))
    if short:
        print("not enough segments for:", short)
    print(f"duration s: total {s['dur'].sum():.0f}, median {s['dur'].median():.2f}, max {s['dur'].max():.2f}")
    est = s["est_bytes"].sum()
    whole = sizes[sizes["video_id"].isin(s["video_id"])]["bytes"].sum()
    print(f"estimated transfer with range requests: {est / 1e9:.2f} GB "
          f"(whole files of the same videos: {whole / 1e9:.1f} GB)")
    print("examples:")
    for r in s.sample(6, random_state=args.seed).itertuples():
        print(f"  {r.narration_id} [{r.dur:.1f}s] narration={r.narration!r} -> true={r.desc_true!r} "
              f"swapped={r.desc_swapped!r} other={r.desc_other!r}")
    colon = sorted({n for n in s["noun"] if ":" in str(n)})
    if colon:
        print("compound nouns rewritten:", ", ".join(f"{n} -> {noun_text(n)}" for n in colon[:12]))
    print(f"wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
