"""Blind spot-check page: the originals the VLM flagged, mixed with unflagged controls.

Builds data/review/review_check.html from the existing review page template and review items.
The page never shows the model's verdict and does not say which episodes were flagged; the
order is shuffled. It shares the browser storage key with review.html, so verdicts made on
either page are kept together. The answer key (which episode is flagged / control) goes to
data/derived/check_page_key.csv and stays local, like everything else that holds dataset text.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402
from prepare_language_audit import PAGE  # noqa: E402


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise SystemExit(f"page template changed, cannot find exactly one: {old[:40]!r}")
    return text.replace(old, new)


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-dir", default="data/review")
    ap.add_argument("--flagged", default="data/derived/vlm_flagged_originals.csv")
    ap.add_argument("--results", default="data/derived/vlm_check.jsonl")
    ap.add_argument("--key", default="data/derived/check_page_key.csv")
    ap.add_argument("--controls", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261124)
    ap.add_argument("--out", default="review_check.html")
    args = ap.parse_args()

    review = Path(args.review_dir)
    items = pd.read_csv(review / "review_items.csv", encoding="utf-8-sig", keep_default_na=False).to_dict("records")
    by_id = {}
    for it in items:
        it["reversible"] = str(it["reversible"]) == "True"
        it["which"] = str(it["which"]) if it["reversible"] and str(it["which"]) else None
        by_id[it["id"]] = it

    flagged = pd.read_csv(args.flagged, encoding="utf-8-sig", keep_default_na=False)
    res = pd.read_json(args.results, lines=True)
    originals = res[res.variant == "original"]
    unflagged = sorted(set(originals[originals.verdict == "match"].id))
    rng = random.Random(args.seed)
    # Controls mirror the share of reversible episodes among the flagged ones.
    n_rev = round(args.controls * sum(by_id[i]["reversible"] for i in flagged.id) / max(len(flagged), 1))
    pool_rev = [i for i in unflagged if by_id[i]["reversible"]]
    pool_one = [i for i in unflagged if not by_id[i]["reversible"]]
    controls = rng.sample(pool_rev, min(n_rev, len(pool_rev)))
    controls += rng.sample(pool_one, min(args.controls - len(controls), len(pool_one)))

    key = [{"id": i, "group": "flagged", "direction_suspect": str(s) == "True", "model_error_type": t}
           for i, s, t in zip(flagged.id, flagged.direction_suspect, flagged.error_type)]
    key += [{"id": i, "group": "control", "direction_suspect": False, "model_error_type": ""} for i in controls]
    rng.shuffle(key)
    for n, k in enumerate(key, 1):
        k["order"] = n

    page_items = [by_id[k["id"]] for k in key]  # template fields only, no model output
    page = PAGE.replace("__ITEMS__", json.dumps(page_items, ensure_ascii=False).replace("</", "<\\/"))
    page = replace_once(page, "<title>EgoDex 语言标注复核</title>", f"<title>EgoDex 描述抽查（{len(key)} 段）</title>")
    page = replace_once(page, "<header><b>EgoDex 语言标注复核</b>", "<header><b>EgoDex 描述抽查</b>")
    page = replace_once(page, '<p class="help">每段先看视频，',
                        f'<p class="help"><b>这一页共 {len(key)} 段，顺序是打乱的。</b>其中有的被模型报过错，有的没有，'
                        '页面上不会告诉你是哪些，请只按你自己看到的判断。判定和总复核页（review.html）存在同一个地方。</p>\n'
                        '<p class="help">每段先看视频，')
    page = replace_once(page, "a.download = 'language_review_'", "a.download = 'language_check_'")
    (review / args.out).write_text(page, encoding="utf-8")

    Path(args.key).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key)[["order", "id", "group", "direction_suspect", "model_error_type"]].to_csv(
        args.key, index=False, encoding="utf-8-sig")
    print(f"{len(key)} episodes ({len(flagged)} flagged, {len(controls)} controls, "
          f"{sum(by_id[k['id']]['reversible'] for k in key)} reversible) -> {review / args.out}; key -> {args.key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
