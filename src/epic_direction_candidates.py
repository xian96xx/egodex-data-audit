"""Rank EPIC-KITCHENS-100 validation videos by usable opposite-direction segments (external check of the VLM).

EPIC narrations are human annotations, so they give the description checker a reference it does not get on
EgoDex: a clip shown with its own narration should be judged a match, and the same clip with the verb swapped
to its opposite (take <-> put, open <-> close, ...) should be judged a mismatch. This script only reads the
annotation tables and counts; it downloads nothing.

Usable segment: verb class in one of the opposite pairs below, between --min-dur and --max-dur seconds long,
and not overlapping (by more than --max-opp-overlap of its length) a segment of the opposite verb on the same
noun class -- otherwise the swapped narration could also be true inside the clip.

Inputs (local only): data/external/epic100/annotations/EPIC_100_validation.csv and EPIC_100_verb_classes.csv.
Outputs (local only; they quote narrations): data/derived/epic_direction_segments.csv (one row per direction
segment, with a `usable` flag and the reason when not) and data/derived/epic_video_candidates.csv (one row per
video).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "external" / "epic100" / "annotations"

# Opposite-direction pairs by EPIC-100 verb class key. The pairing is ours (the class table has no direction
# column). The first four are the main pairs; the rest are small and only reported.
PAIRS = [
    ("take", "put"),
    ("open", "close"),
    ("insert", "remove"),
    ("turn-on", "turn-off"),
    ("fill", "empty"),
    ("screw", "unscrew"),
    ("wrap", "unwrap"),
    ("lock", "unlock"),
    ("lift", "lower"),
]
MAIN_PAIRS = PAIRS[:4]


def to_sec(ts: str) -> float:
    h, m, s = str(ts).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--val", type=Path, default=ANN / "EPIC_100_validation.csv")
    ap.add_argument("--verbs", type=Path, default=ANN / "EPIC_100_verb_classes.csv")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--max-opp-overlap", type=float, default=0.3)
    ap.add_argument("--segments-out", type=Path, default=ROOT / "data" / "derived" / "epic_direction_segments.csv")
    ap.add_argument("--videos-out", type=Path, default=ROOT / "data" / "derived" / "epic_video_candidates.csv")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    verbs = pd.read_csv(args.verbs)
    key_to_id = {str(k): int(i) for k, i in zip(verbs["key"], verbs["id"])}
    missing = [k for p in PAIRS for k in p if k not in key_to_id]
    if missing:
        print("verb keys missing from the class table:", missing)
        return 2
    opposite: dict[int, int] = {}
    pair_of: dict[int, str] = {}
    key_of: dict[int, str] = {}
    for a, b in PAIRS:
        ia, ib = key_to_id[a], key_to_id[b]
        opposite[ia], opposite[ib] = ib, ia
        pair_of[ia] = pair_of[ib] = f"{a}/{b}"
        key_of[ia], key_of[ib] = a, b

    df = pd.read_csv(args.val)
    need = {"narration_id", "video_id", "participant_id", "start_timestamp", "stop_timestamp",
            "narration", "verb", "verb_class", "noun", "noun_class"}
    if not need.issubset(df.columns):
        print("validation table is missing columns:", sorted(need - set(df.columns)))
        return 2
    if df["narration_id"].duplicated().any():
        print("duplicate narration_id in validation table")
        return 2
    df["start_s"] = df["start_timestamp"].map(to_sec)
    df["stop_s"] = df["stop_timestamp"].map(to_sec)
    df["dur"] = (df["stop_s"] - df["start_s"]).round(3)

    seg = df[df["verb_class"].isin(opposite.keys())].copy()
    seg["pair"] = seg["verb_class"].map(pair_of)
    seg["verb_key"] = seg["verb_class"].map(key_of)
    seg["opposite_key"] = seg["verb_class"].map(lambda c: key_of[opposite[c]])

    # Largest overlap (as a share of this segment's length) with the opposite verb on the same noun class.
    opp_frac = []
    by_video = {v: g for v, g in df.groupby("video_id")}
    for r in seg.itertuples():
        g = by_video[r.video_id]
        o = g[(g["verb_class"] == opposite[r.verb_class]) & (g["noun_class"] == r.noun_class)]
        best = 0.0
        if r.dur > 0 and len(o):
            ov = (o["stop_s"].clip(upper=r.stop_s) - o["start_s"].clip(lower=r.start_s)).clip(lower=0)
            best = float(ov.max()) / r.dur
        opp_frac.append(round(best, 3))
    seg["opp_overlap"] = opp_frac

    def why_not(r) -> str:
        if r.dur < args.min_dur:
            return "too_short"
        if r.dur > args.max_dur:
            return "too_long"
        if r.opp_overlap > args.max_opp_overlap:
            return "opposite_overlap"
        return ""

    seg["drop_reason"] = seg.apply(why_not, axis=1)
    seg["usable"] = seg["drop_reason"] == ""

    vids = df.groupby("video_id").agg(participant_id=("participant_id", "first"),
                                      n_segments=("narration_id", "size"),
                                      last_stop_s=("stop_s", "max")).reset_index()
    vids["extension"] = vids["video_id"].map(lambda v: len(str(v).split("_")[1]) == 3)
    use = seg[seg["usable"]]
    counts = use.pivot_table(index="video_id", columns="verb_key", values="narration_id",
                             aggfunc="count", fill_value=0)
    for a, b in PAIRS:
        for k in (a, b):
            if k not in counts.columns:
                counts[k] = 0
    counts = counts[[k for p in PAIRS for k in p]]
    vids = vids.merge(counts, left_on="video_id", right_index=True, how="left").fillna(0)
    for a, b in PAIRS:
        vids[a] = vids[a].astype(int)
        vids[b] = vids[b].astype(int)
    vids["usable_main"] = sum(vids[a] + vids[b] for a, b in MAIN_PAIRS)
    vids["balanced_main"] = sum(2 * vids[[a, b]].min(axis=1) for a, b in MAIN_PAIRS)
    vids["usable_all"] = sum(vids[a] + vids[b] for a, b in PAIRS)
    vids = vids.sort_values(["balanced_main", "usable_main"], ascending=False)

    args.segments_out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["narration_id", "video_id", "participant_id", "start_timestamp", "stop_timestamp", "start_s",
            "stop_s", "dur", "narration", "verb", "verb_class", "verb_key", "opposite_key", "pair", "noun",
            "noun_class", "opp_overlap", "usable", "drop_reason"]
    seg[cols].sort_values(["video_id", "start_s"]).to_csv(args.segments_out, index=False, encoding="utf-8")
    vids.to_csv(args.videos_out, index=False, encoding="utf-8")

    print(f"validation: {len(df)} segments, {df['video_id'].nunique()} videos "
          f"({int(vids['extension'].sum())} extension, {int((~vids['extension']).sum())} EPIC-55 ids)")
    print(f"all segments duration s: median {df['dur'].median():.2f}, "
          f"p10 {df['dur'].quantile(.1):.2f}, p90 {df['dur'].quantile(.9):.2f}")
    print(f"direction-verb segments: {len(seg)}; usable {int(seg['usable'].sum())}; dropped "
          + ", ".join(f"{k} {v}" for k, v in seg.loc[~seg['usable'], 'drop_reason'].value_counts().items()))
    print("\nper verb (all -> usable), usable median duration:")
    for a, b in PAIRS:
        parts = []
        for k in (a, b):
            s = seg[seg["verb_key"] == k]
            u = s[s["usable"]]
            med = f"{u['dur'].median():.1f}s" if len(u) else "-"
            parts.append(f"{k} {len(s)}->{len(u)} ({med}, {u['video_id'].nunique()} videos)")
        print("  " + " | ".join(parts))
    print(f"\ntop {args.top} videos by balanced main-pair segments "
          "(balanced = 2*min of each pair, summed over take/put, open/close, insert/remove, turn-on/off):")
    show = ["video_id", "extension", "n_segments", "last_stop_s"] + [k for p in MAIN_PAIRS for k in p] + \
        ["balanced_main", "usable_main", "usable_all"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(vids[show].head(args.top).to_string(index=False))
    print(f"\nwrote {args.segments_out.relative_to(ROOT)} and {args.videos_out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
