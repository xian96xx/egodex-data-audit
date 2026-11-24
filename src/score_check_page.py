"""Score the blind spot check (data/review/review_check.html) against the VLM's verdicts.

Joins the reviewer's export with the answer key from make_check_page.py and the VLM results, then reports:
- of the original descriptions the VLM flagged, how many the reviewer also judged wrong or partly wrong;
- the same share among the unflagged controls (a rough look at what the VLM lets through);
- for reversible tasks, whether the data picked the right one of its two descriptions, split by what the VLM
  said about the two candidates (including the episodes it suspected were reversed).
Only the reviewer's judgments decide what is right; nothing here replaces or corrects them. Rows whose answers
leave the reason open (description judged wrong with no error type ticked, a direction error ticked while the
direction pick is judged right, or the direction question left blank) are listed so the reviewer can look again.

Public: outputs/human_check_summary.json (aggregates) and outputs/human_check_judgments.csv (episode id and the
reviewer's judgment only). Local: data/derived/check_page_scored.csv (joined with the model's fields and reasons,
which quote the description, so it stays under data/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402
from vlm_check import wilson  # noqa: E402

VERDICTS = ["correct", "partial", "wrong", "unclear"]
WHICH = ["right", "wrong", "unclear"]
ERR_TYPES = ["object", "action", "order_direction", "missing_step", "hallucinated", "other"]
PAIRS = ["consistent_with_data", "model_says_data_reversed", "both_match", "both_mismatch", "other"]


def pair_category(a: str | None, b: str | None) -> str:
    """Same cells as vlm_check.direction_pairs: verdict on the data's description vs the other candidate."""
    if a == "match" and b == "mismatch":
        return "consistent_with_data"
    if a == "mismatch" and b == "match":
        return "model_says_data_reversed"
    if a == b == "match":
        return "both_match"
    if a == b == "mismatch":
        return "both_mismatch"
    return "other"


def rate(k: int, n: int) -> dict:
    if not n:
        return {"k": k, "n": n, "rate": None, "ci95": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 3), "ci95": [round(lo, 3), round(hi, 3)]}


def counts(s: pd.Series, levels: list[str]) -> dict:
    out = {v: int((s == v).sum()) for v in levels}
    blank = int((s == "").sum())
    if blank:
        out["blank"] = blank
    return out


def second_look(r) -> str:
    errs = [e for e in r.errors.split("|") if e]
    out = []
    if not r.verdict:
        out.append("verdict_blank")
    if r.verdict in ("wrong", "partial") and not errs and r.which_check != "wrong":
        out.append("no_reason")  # wrong, but no error type and the direction pick is not judged wrong either
    if r.reversible and r.which_check == "right" and "order_direction" in errs:
        out.append("direction_ticked_but_pick_right")
    if r.reversible and r.which_check == "wrong" and r.verdict == "correct":
        out.append("pick_wrong_but_desc_correct")
    if r.reversible and not r.which_check:
        out.append("which_blank")
    return "|".join(out)


def fmt(d: dict) -> str:
    return f"{d['k']}/{d['n']}" + (f" = {d['rate']:.0%} (95% CI {d['ci95'][0]:.0%}-{d['ci95'][1]:.0%})" if d["n"] else "")


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="CSV exported from data/review/review_check.html")
    ap.add_argument("--key", default="data/derived/check_page_key.csv")
    ap.add_argument("--items", default="data/review/review_items.csv")
    ap.add_argument("--results", default="data/derived/vlm_check.jsonl")
    ap.add_argument("--summary", default="outputs/human_check_summary.json")
    ap.add_argument("--judgments", default="outputs/human_check_judgments.csv")
    ap.add_argument("--scored", default="data/derived/check_page_scored.csv")
    args = ap.parse_args()

    raw = Path(args.export).read_bytes()
    ex = pd.read_csv(args.export, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    key = pd.read_csv(args.key, encoding="utf-8-sig", keep_default_na=False, dtype=str)
    items = pd.read_csv(args.items, encoding="utf-8-sig", keep_default_na=False, dtype=str).set_index("id")
    res = pd.read_json(args.results, lines=True)
    if "set" in res.columns:
        res = res[res["set"] == "review"]
    o = res[res.variant == "original"].set_index("id")
    s = res[res.variant == "direction_swap"].set_index("id").verdict

    problems = []
    if ex.id.duplicated().any() or key.id.duplicated().any():
        problems.append("duplicate ids")
    if set(ex.id) != set(key.id):
        problems.append(f"export and key differ: {sorted(set(ex.id) ^ set(key.id))}")
    for r in ex.itertuples():
        if r.id not in items.index:
            problems.append(f"{r.id}: not in review_items.csv")
            continue
        rev = items.loc[r.id, "reversible"] == "True"
        if (r.reversible == "true") != rev or (rev and r.which_in_data != items.loc[r.id, "which"]):
            problems.append(f"{r.id}: page fields differ from review_items.csv")
    if problems:
        print("cannot score:\n  " + "\n  ".join(problems))
        return 2

    df = key.merge(ex[["id", "verdict", "which_check", "errors", "note", "reviewed_at"]], on="id", how="left")
    df["reversible"] = df.id.map(lambda i: items.loc[i, "reversible"] == "True")
    df["which_in_data"] = [items.loc[i, "which"] if rev else "" for i, rev in zip(df.id, df.reversible)]
    df["model_verdict"] = df.id.map(o.verdict)
    expected = df.group.map({"flagged": "mismatch", "control": "match"})
    if (df.model_verdict != expected).any():
        print("cannot score: key groups do not match the model verdicts in", args.results)
        return 2
    df["model_pair"] = [pair_category(o.verdict.get(i), s.get(i)) if rev else "" for i, rev in zip(df.id, df.reversible)]
    df["model_observation"] = df.id.map(o.observation).fillna("")
    df["model_reason"] = df.id.map(o.reason).fillna("")
    df["second_look"] = [second_look(r) for r in df.itertuples()]

    decided = df[df.verdict.isin(["correct", "partial", "wrong"])]

    def share(sub: pd.DataFrame, levels: tuple[str, ...]) -> dict:
        return rate(int(sub.verdict.isin(levels).sum()), len(sub))

    groups = ("flagged", "control")
    rev = df[df.reversible]
    rev_dec = rev[rev.which_check.isin(["right", "wrong"])]
    bad = df[df.verdict.isin(["wrong", "partial"])]
    err_lists = bad.errors.str.split("|")
    flagged_dec = decided[decided.group == "flagged"]
    looks = {}
    for r in df.sort_values("order", key=lambda c: c.astype(int)).itertuples():
        for tag in filter(None, r.second_look.split("|")):
            looks.setdefault(tag, []).append(f"#{r.order} {r.id}")  # #N is the card number on the page

    summary = {
        "what": ("Blind human spot check of the VLM's verdicts on the original EgoDex descriptions: every original "
                 "the VLM flagged plus unflagged controls, shuffled, model verdict hidden from the reviewer."),
        "reviewer": "the author; one person, one pass",
        "export_file": Path(args.export).name,
        "export_sha256": hashlib.sha256(raw).hexdigest(),
        "design": {"flagged": int((df.group == "flagged").sum()), "controls": int((df.group == "control").sum()),
                   "unflagged_pool": int((o.verdict == "match").sum()), "originals_checked_by_vlm": len(o),
                   "reversible_in_sample": len(rev)},
        "verdicts": {g: counts(df[df.group == g].verdict, VERDICTS) for g in groups},
        "wrong_or_partial": {g: share(decided[decided.group == g], ("wrong", "partial")) for g in groups},
        "wrong_only": {g: share(decided[decided.group == g], ("wrong",)) for g in groups},
        "direction": {
            "which_check": {g: counts(rev[rev.group == g].which_check, WHICH) for g in groups},
            "data_pick_judged_wrong": {"all": rate(int((rev_dec.which_check == "wrong").sum()), len(rev_dec)),
                                       **{g: rate(int((rev_dec[rev_dec.group == g].which_check == "wrong").sum()),
                                                  int((rev_dec.group == g).sum())) for g in groups}},
            # Split by the value of which_llm_description (1 = llm_description, 2 = llm_description2).
            "data_pick_judged_wrong_by_which_value": {
                w: rate(int((rev_dec[rev_dec.which_in_data == w].which_check == "wrong").sum()),
                        int((rev_dec.which_in_data == w).sum())) for w in sorted(set(rev_dec.which_in_data))},
            "by_model_pair": {p: counts(rev[rev.model_pair == p].which_check, WHICH)
                              for p in PAIRS if (rev.model_pair == p).any()},
            "model_direction_suspects": counts(rev[rev.direction_suspect == "True"].which_check, WHICH),
        },
        "human_error_types_in_wrong_or_partial": {
            **{e: int(err_lists.apply(lambda x, e=e: e in x).sum()) for e in ERR_TYPES},
            "none_ticked": int((bad.errors == "").sum()),
            "none_ticked_but_direction_pick_wrong": int(((bad.errors == "") & (bad.which_check == "wrong")).sum()),
        },
        "flagged_by_model_error_type": {
            t: {"decided": len(sub), "wrong_or_partial": int(sub.verdict.isin(["wrong", "partial"]).sum())}
            for t, sub in flagged_dec.groupby("model_error_type")},
        "needs_second_look": {k: len(v) for k, v in looks.items()},
        "notes": [
            "Shares count only episodes judged correct, partial or wrong; 'unclear' is left out.",
            "One reviewer, one pass. With 8 controls the control share is a rough look, not an estimate for all 80.",
            "The sample is chosen by the model's verdict, so none of these shares is an error rate of EgoDex descriptions.",
            "data_pick_judged_wrong: the reviewer judged that the data picked the wrong one of the two candidate "
            "descriptions of a reversible task (which_llm_description).",
        ],
    }

    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    df.sort_values("id")[["id", "verdict", "which_check", "errors"]].to_csv(
        args.judgments, index=False, encoding="utf-8-sig")
    cols = ["order", "id", "group", "direction_suspect", "model_error_type", "model_pair", "reversible", "which_in_data",
            "verdict", "which_check", "errors", "note", "second_look", "reviewed_at", "model_observation", "model_reason"]
    Path(args.scored).parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("order", key=lambda c: c.astype(int))[cols].to_csv(args.scored, index=False, encoding="utf-8-sig")

    d = summary["design"]
    print(f"{len(df)} episodes: {d['flagged']} flagged by the VLM, {d['controls']} controls "
          f"(from {d['unflagged_pool']} unflagged), {d['reversible_in_sample']} reversible")
    for g in groups:
        print(f"  {g:8s} verdicts {summary['verdicts'][g]}")
        print(f"           wrong or partial {fmt(summary['wrong_or_partial'][g])}; wrong only {fmt(summary['wrong_only'][g])}")
    dr = summary["direction"]
    print(f"reversible: data's pick judged wrong {fmt(dr['data_pick_judged_wrong']['all'])}; "
          f"flagged {fmt(dr['data_pick_judged_wrong']['flagged'])}; control {fmt(dr['data_pick_judged_wrong']['control'])}")
    print("  by which_llm_description value: " + "; ".join(
        f"{w}: {fmt(v)}" for w, v in dr["data_pick_judged_wrong_by_which_value"].items()))
    print(f"  by model pair: {dr['by_model_pair']}")
    print(f"  VLM direction suspects: {dr['model_direction_suspects']}")
    print(f"error types ticked in wrong/partial: {summary['human_error_types_in_wrong_or_partial']}")
    print(f"flagged, by model error type: {summary['flagged_by_model_error_type']}")
    for tag, ids in looks.items():
        print(f"second look [{tag}]: {', '.join(ids)}")
    print(f"-> {args.summary}, {args.judgments}; local {args.scored}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
