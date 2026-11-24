"""Cut the sampled EPIC-KITCHENS-100 segments out of the remote videos with HTTP range requests (ffmpeg).

Only the seconds each segment needs are fetched (plus the file index and a keyframe pre-roll); the clip is
re-encoded (H.264, --width wide, no audio) to data/external/epic100/clips/<narration_id>.mp4. Local only: the clips
are EPIC data (CC BY-NC 4.0) and never go into the public repo. Resumable: a clip that already exists and has the
expected length is skipped.

Network use is checked as it goes: ffmpeg's own read counter ("Statistics: N bytes read", the bytes it consumed
from the server; data in flight when it drops a connection is not counted) is compared with the sample's
estimate, and no new clip is started once the total is over --max-factor times the estimate of the clips done.

Speed: the server gave 0.1-0.47 MB/s per connection from here (2026-11-24, measured with curl range requests),
and 8 connections at once added up roughly linearly, so clips are fetched by --workers ffmpeg processes in
parallel. The proxy is bypassed unless --use-proxy (direct measured faster: 0.20 vs 0.12 MB/s).

Log: data/derived/epic_clips_log.csv (one row per attempt).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATS = re.compile(r"Statistics: (\d+) bytes read, (\d+) seeks")
LOG_COLS = ["at", "narration_id", "video_id", "verb_key", "start_s", "dur", "est_bytes", "status", "code",
            "wall_s", "bytes_read", "seeks", "out_bytes", "out_frames", "out_dur", "err_tail"]


def probe(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    return n, (n / fps if fps else 0.0)


def length_ok(n: int, out_dur: float, dur: float) -> bool:
    return n >= 16 and abs(out_dur - dur) <= max(0.2, 0.1 * dur)


def fetch(r, out: Path, ffmpeg: str, width: int, timeout: int, env: dict) -> dict:
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-v", "verbose", "-y",
           "-ss", f"{r.start_s:.3f}", "-i", r.url, "-t", f"{r.dur:.3f}",
           "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", f"scale={width}:-2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(tmp)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        code, err = p.returncode, p.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        code, err = -9, (e.stderr or b"").decode("utf-8", "replace") + "\nTIMEOUT"
    wall = round(time.time() - t0, 2)
    stats = STATS.findall(err)
    rec = {"code": code, "wall_s": wall, "bytes_read": sum(int(b) for b, _ in stats),
           "seeks": sum(int(s) for _, s in stats), "out_bytes": 0, "out_frames": 0, "out_dur": 0.0,
           "err_tail": ""}
    if code == 0 and tmp.exists():
        n, d = probe(tmp)
        rec.update(out_frames=n, out_dur=round(d, 3))
        if length_ok(n, d, r.dur):
            tmp.replace(out)
            rec.update(status="ok", out_bytes=out.stat().st_size)
            return rec
        rec["status"] = "bad_length"
    else:
        rec["status"] = "failed"
    tail = [ln for ln in err.strip().splitlines() if ln.strip()][-3:]
    rec["err_tail"] = " | ".join(tail)[:400]
    tmp.unlink(missing_ok=True)
    return rec


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=Path, default=ROOT / "data" / "derived" / "epic_sample.csv")
    ap.add_argument("--clips", type=Path, default=ROOT / "data" / "external" / "epic100" / "clips")
    ap.add_argument("--log", type=Path, default=ROOT / "data" / "derived" / "epic_clips_log.csv")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"), help="ffmpeg binary (or env FFMPEG)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--use-proxy", action="store_true")
    ap.add_argument("--max-factor", type=float, default=2.0)
    ap.add_argument("--pilot", action="store_true", help="only the first segment of each verb (8 clips)")
    args = ap.parse_args()

    s = pd.read_csv(args.sample)
    if args.pilot:
        s = s.groupby("verb_key", sort=False).head(1)
    args.clips.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if not args.use_proxy:
        env = {k: v for k, v in env.items() if k.lower() not in {"http_proxy", "https_proxy", "all_proxy"}}

    todo, already = [], 0
    for r in s.itertuples():
        out = args.clips / f"{r.narration_id}.mp4"
        if out.exists():
            n, d = probe(out)
            if length_ok(n, d, r.dur):
                already += 1
                continue
        todo.append(r)
    print(f"{len(s)} in sample, {already} already there, {len(todo)} to fetch with {args.workers} workers", flush=True)

    def run_one(r) -> list[dict]:
        out = args.clips / f"{r.narration_id}.mp4"
        attempts = []
        for _ in range(args.retries + 1):
            rec = fetch(r, out, args.ffmpeg, args.width, args.timeout, env)
            rec["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            attempts.append(rec)
            if rec["status"] == "ok":
                break
        return attempts

    new_log = not args.log.exists()
    logf = args.log.open("a", newline="", encoding="utf-8")
    w = csv.DictWriter(logf, fieldnames=LOG_COLS)
    if new_log:
        w.writeheader()
    fetched = failed = 0
    read_total = est_total = 0
    stopped = False
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    futures = {pool.submit(run_one, r): r for r in todo}
    for k, fut in enumerate(as_completed(futures), 1):
        r = futures[fut]
        if fut.cancelled():
            continue
        try:
            attempts = fut.result()
        except Exception as e:  # noqa: BLE001 - keep the run going, log the clip as failed
            attempts = [{"status": "error", "code": -1, "wall_s": 0.0, "bytes_read": 0, "seeks": 0, "out_bytes": 0,
                         "out_frames": 0, "out_dur": 0.0, "err_tail": repr(e)[:400],
                         "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}]
        for rec in attempts:
            w.writerow({"narration_id": r.narration_id, "video_id": r.video_id, "verb_key": r.verb_key,
                        "start_s": r.start_s, "dur": r.dur, "est_bytes": r.est_bytes, **rec})
            read_total += rec["bytes_read"]
        logf.flush()
        est_total += int(r.est_bytes)
        rec = attempts[-1]
        if rec["status"] == "ok":
            fetched += 1
        else:
            failed += 1
        print(f"{k}/{len(todo)} {r.narration_id} {rec['status']} (tries {len(attempts)}) "
              f"read {sum(a['bytes_read'] for a in attempts) / 1e6:.1f} MB (est {r.est_bytes / 1e6:.1f}) "
              f"{sum(a['wall_s'] for a in attempts):.0f}s out {rec['out_frames']} fr / {rec['out_dur']:.2f}s "
              f"(want {r.dur:.2f}); total {read_total / 1e9:.2f} GB, {time.time() - t_start:.0f}s"
              + (f" ERR {rec['err_tail']}" if rec["status"] != "ok" else ""), flush=True)
        if not stopped and est_total and read_total > args.max_factor * est_total:
            stopped = True
            print(f"STOP: read {read_total / 1e6:.0f} MB > {args.max_factor} x estimate {est_total / 1e6:.0f} MB; "
                  "no new clips will start", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
    pool.shutdown(wait=True)
    logf.close()
    print(f"done: {fetched} fetched, {already} already there, {failed} failed; read {read_total / 1e9:.3f} GB "
          f"vs estimate {est_total / 1e9:.3f} GB for the clips tried; {time.time() - t_start:.0f}s", flush=True)
    if stopped:
        return 3
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
