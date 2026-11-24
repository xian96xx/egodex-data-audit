"""Look up the size of candidate EPIC-KITCHENS-100 videos with HTTP HEAD requests (nothing is downloaded).

Takes the validation videos with at least --min-balanced balanced main-pair direction segments
(data/derived/epic_video_candidates.csv from epic_direction_candidates.py), ranks them by those segments per
annotated minute, and asks the EPIC server for the size of the top --top videos. All validation videos have EPIC-55
ids, so they sit under videos/test/ or videos/train/ of the EPIC-55 dataset; test is tried first. The sizes give the
video pool and the byte estimate of epic_make_sample.py, and "Accept-Ranges: bytes" says whether clips can be cut
remotely with range requests.

Output (local): data/derived/epic_video_sizes.csv.
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from egodex_common import setup_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPIC55_VIDEOS = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos"


def head(url: str, timeout: int) -> tuple[int, dict]:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        return e.code, {}
    except (urllib.error.URLError, TimeoutError) as e:
        return -1, {"error": str(e)[:200]}


def main() -> int:
    setup_stdout()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos", type=Path, default=ROOT / "data" / "derived" / "epic_video_candidates.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "derived" / "epic_video_sizes.csv")
    ap.add_argument("--min-balanced", type=int, default=20)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=30)
    args = ap.parse_args()

    v = pd.read_csv(args.videos)
    v = v[v["balanced_main"] >= args.min_balanced].copy()
    v["per_min"] = (v["balanced_main"] / v["last_stop_s"] * 60).round(2)
    v = v.sort_values("per_min", ascending=False, kind="stable").head(args.top)

    rows = []
    for r in v.itertuples():
        hit, last = None, ""
        for split in ("test", "train"):
            code, h = head(f"{EPIC55_VIDEOS}/{split}/{r.participant_id}/{r.video_id}.MP4", args.timeout)
            last = f"http {code}" + (f" {h['error']}" if "error" in h else "")
            if code == 200 and "content-length" in h:
                hit = {"split": split, "bytes": int(h["content-length"]),
                       "accept_ranges": h.get("accept-ranges", ""), "status": "ok"}
                break
        if hit is None:
            hit = {"split": "?", "bytes": -1, "accept_ranges": "", "status": last}
        rows.append({"video_id": r.video_id, **hit, "minutes": round(r.last_stop_s / 60, 1),
                     "balanced_main": int(r.balanced_main), "per_min": r.per_min})
        print(f"{r.video_id} {hit['split']} {hit['bytes'] / 1e6:.0f} MB ranges={hit['accept_ranges'] or '-'} "
              f"{hit['status']}", flush=True)

    out = pd.DataFrame(rows)[["video_id", "split", "bytes", "accept_ranges", "minutes", "balanced_main", "per_min",
                              "status"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8")
    ok = out[out["status"] == "ok"]
    print(f"{len(ok)} of {len(out)} sized, {ok['bytes'].sum() / 1e9:.1f} GB in all; "
          f"range requests supported by {int((ok['accept_ranges'] == 'bytes').sum())}; "
          f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0 if len(ok) == len(out) else 1


if __name__ == "__main__":
    raise SystemExit(main())
