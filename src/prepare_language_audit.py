"""Experiment 2 prep: sample episodes for a human review of the LLM-written descriptions.

Writes a local review page (data/review/review.html) that plays each video next to its
description and exports the reviewer's verdicts as CSV. Everything here contains dataset
content, so it stays under data/ and is never published. Human verdicts must come from a
real reviewer; nothing in this script fills them in.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import clean_attr, list_episodes, setup_stdout  # noqa: E402

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EgoDex 语言标注复核</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --muted:#6b6b70; --line:#e2e2e0; --accent:#2f6fde; --ok:#2e8b57; --bad:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#161618; --card:#1f1f22; --ink:#ececf0; --muted:#9a9aa2; --line:#333338; --accent:#6b9cff; --ok:#5cc28a; --bad:#ff7b6b; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 "Microsoft YaHei","PingFang SC",sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; gap:12px; flex-wrap:wrap; align-items:center; }
header b { font-size:16px; }
button, label.tog { border:1px solid var(--line); background:var(--card); color:var(--ink); border-radius:6px; padding:5px 10px; cursor:pointer; font:inherit; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1100px; margin:0 auto; padding:16px; }
.help { color:var(--muted); font-size:13px; margin:0 0 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:14px; display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:14px; }
.card.done { border-left:4px solid var(--ok); }
@media (max-width:800px) { .card { grid-template-columns:1fr; } }
video { width:100%; border-radius:6px; background:#000; }
.meta { color:var(--muted); font-size:12px; }
.desc { font-size:15px; padding:8px 10px; border-left:3px solid var(--accent); background:color-mix(in srgb, var(--accent) 8%, transparent); margin:6px 0; }
.alt { font-size:13px; color:var(--muted); }
fieldset { border:1px solid var(--line); border-radius:6px; margin:8px 0; padding:6px 10px; }
legend { font-size:13px; color:var(--muted); }
fieldset label { margin-right:12px; white-space:nowrap; display:inline-block; }
input[type=text] { width:100%; padding:6px; border:1px solid var(--line); border-radius:6px; background:var(--bg); color:var(--ink); font:inherit; }
.hidden { display:none; }
</style></head><body>
<header><b>EgoDex 语言标注复核</b><span id="progress"></span>
<label class="tog"><input type="checkbox" id="onlyTodo"> 只看未复核</label>
<button class="primary" id="export">导出 CSV</button></header>
<main>
<p class="help">每段先看视频，再判断蓝框里的描述（数据实际使用的那条）对不对。可逆任务还要判断「数据选的是哪条」选得对不对。结果自动存在这个浏览器里，做完点「导出 CSV」。看不清就选「看不清」，不要猜。</p>
<div id="list"></div></main>
<script id="items" type="application/json">__ITEMS__</script>
<script>
const ITEMS = JSON.parse(document.getElementById('items').textContent);
const KEY = 'egodex_language_review_v1';
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { state = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} render_progress(); }
const VERDICTS = [['correct','正确'],['partial','部分正确'],['wrong','错误'],['unclear','看不清']];
const ERRS = [['object','物体错'],['action','动作错'],['order_direction','顺序或方向错'],['missing_step','漏了步骤'],['hallucinated','写了没发生的'],['other','其他']];
const WHICH = [['right','选对了'],['wrong','选错了'],['unclear','看不出']];
function esc(s) { return String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function card(it, i) {
  const st = state[it.id] || {};
  const radios = (name, opts, cur) => opts.map(([v, t]) => `<label><input type="radio" name="${name}-${i}" value="${v}" ${cur === v ? 'checked' : ''}> ${t}</label>`).join('');
  const errs = ERRS.map(([v, t]) => `<label><input type="checkbox" value="${v}" ${(st.errors || []).includes(v) ? 'checked' : ''}> ${t}</label>`).join('');
  const which = it.reversible ? `<fieldset data-k="which"><legend>可逆任务：数据选的是第 ${esc(it.which)} 条，选对了吗？</legend>${radios('which', WHICH, st.which)}</fieldset>
     <div class="alt">另一条候选：${esc(it.desc_alt)}</div>` : '';
  return `<div class="card ${st.verdict ? 'done' : ''}" data-id="${esc(it.id)}">
    <div><video controls preload="metadata" src="${esc(it.video)}"></video>
      <div class="meta">#${i + 1} · ${esc(it.id)} · ${esc(it.seconds)} 秒</div></div>
    <div><div class="meta">任务：${esc(it.task)}</div>
      <div class="desc">${esc(it.desc)}</div>${which}
      <fieldset data-k="verdict"><legend>描述对吗？</legend>${radios('verdict', VERDICTS, st.verdict)}</fieldset>
      <fieldset data-k="errors"><legend>错在哪（可多选）</legend>${errs}</fieldset>
      <input type="text" data-k="note" placeholder="备注（可选）" value="${esc(st.note)}"></div></div>`;
}
function render() {
  const only = document.getElementById('onlyTodo').checked;
  document.getElementById('list').innerHTML = ITEMS.map((it, i) => (only && (state[it.id] || {}).verdict) ? '' : card(it, i)).join('');
  render_progress();
}
function render_progress() {
  const n = ITEMS.filter(it => (state[it.id] || {}).verdict).length;
  document.getElementById('progress').textContent = `已复核 ${n} / ${ITEMS.length}`;
}
document.getElementById('list').addEventListener('change', ev => {
  const c = ev.target.closest('.card'); if (!c) return;
  const id = c.dataset.id; const st = state[id] = state[id] || {};
  const k = ev.target.closest('[data-k]').dataset.k;
  if (k === 'errors') st.errors = [...c.querySelectorAll('[data-k=errors] input:checked')].map(x => x.value);
  else if (k === 'note') st.note = ev.target.value;
  else st[k] = ev.target.value;
  st.reviewed_at = new Date().toISOString();
  if (st.verdict) c.classList.add('done');
  save();
});
document.getElementById('onlyTodo').addEventListener('change', render);
document.getElementById('export').addEventListener('click', () => {
  const head = ['id','task','reversible','which_in_data','verdict','which_check','errors','note','reviewed_at'];
  const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
  const rows = ITEMS.map(it => { const s = state[it.id] || {};
    return [it.id, it.task, it.reversible, it.which ?? '', s.verdict, s.which, (s.errors || []).join('|'), s.note, s.reviewed_at].map(q).join(','); });
  const blob = new Blob(['﻿' + [head.join(','), ...rows].join('\r\n')], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
  a.download = 'language_review_' + new Date().toISOString().slice(0, 10) + '.csv'; a.click();
});
render();
</script></body></html>
"""


def transcode(src: Path, dst: Path, width: int = 640, every: int = 2) -> bool:
    """EgoDex MP4s are MPEG-4 Part 2 (FourCC FMP4), which browsers cannot play.
    Write a small VP8 WebM copy (640 px wide, every 2nd frame) for the review page."""
    import cv2

    if dst.exists() and dst.stat().st_size > 0:
        return True
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            h = int(round(frame.shape[0] * width / frame.shape[1] / 2) * 2)
            small = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
            if writer is None:
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"VP80"), fps / every, (width, h))
            writer.write(small)
        i += 1
    cap.release()
    if writer is not None:
        writer.release()
    return dst.exists() and dst.stat().st_size > 0


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/egodex_test/test")
    ap.add_argument("--review-dir", default="data/review")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--no-transcode", action="store_true", help="link the original MP4s (browsers cannot play them)")
    args = ap.parse_args()

    data_dir, review = Path(args.data_dir).resolve(), Path(args.review_dir).resolve()
    review.mkdir(parents=True, exist_ok=True)
    eps = [e for e in list_episodes(data_dir) if e["mp4"].exists()]
    rng = np.random.default_rng(args.seed)
    by_task: dict[str, list] = {}
    for e in eps:
        by_task.setdefault(e["task"], []).append(e)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    picked, used = [], set()
    while len(picked) < min(args.n, len(eps)):  # one episode per task per round
        progressed = False
        for t in tasks:
            pool = [e for e in by_task[t] if e["id"] not in used]
            if not pool:
                continue
            e = pool[int(rng.integers(len(pool)))]
            picked.append(e)
            used.add(e["id"])
            progressed = True
            if len(picked) >= args.n:
                break
        if not progressed:
            break

    items = []
    vid_dir = review / "videos"
    vid_dir.mkdir(exist_ok=True)
    for n_done, e in enumerate(picked, 1):
        if args.no_transcode:
            video = Path(os.path.relpath(e["mp4"], review)).as_posix()
        else:
            webm = vid_dir / f"{e['task']}__{e['mp4'].stem}.webm"
            if not transcode(e["mp4"], webm):
                print(f"transcode failed: {e['id']}")
                return 1
            video = Path(os.path.relpath(webm, review)).as_posix()
            if n_done % 10 == 0:
                print(f"  transcoded {n_done}/{len(picked)}", flush=True)
        with h5py.File(e["path"], "r") as f:
            attrs = {k: clean_attr(v) for k, v in f.attrs.items()}
            n = int(f["transforms"]["leftHand"].shape[0])
        d1, d2 = attrs.get("llm_description", ""), attrs.get("llm_description2")
        which = attrs.get("which_llm_description")
        # Only reversible tasks carry which_llm_description; the others store the string "None" as desc2.
        reversible = which is not None and isinstance(d2, str) and d2.strip() not in ("", "None")
        use2 = reversible and str(which) == "2"
        items.append({
            "id": e["id"], "task": e["task"], "llm_type": attrs.get("llm_type"), "seconds": round(n / 30.0, 1),
            "video": video, "reversible": reversible, "which": which if reversible else None,
            "desc": d2 if use2 else d1, "desc_alt": (d1 if use2 else d2) if reversible else "",
        })
    page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
    (review / "review.html").write_text(page, encoding="utf-8")
    pd.DataFrame(items).to_csv(review / "review_items.csv", index=False, encoding="utf-8-sig")
    print(f"{len(items)} episodes from {len({i['task'] for i in items})} tasks; "
          f"{sum(i['reversible'] for i in items)} reversible -> {html.escape(str(review / 'review.html'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
