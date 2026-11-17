"""Step 0: verify and extract the EgoDex test.zip.

Every member's CRC-32 is checked while extracting (zipfile raises on a mismatch).
Member names are checked for absolute paths and '..' before extracting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

EXPECTED_BYTES = 17_304_529_397  # Content-Length of https://ml-site.cdn-apple.com/datasets/egodex/test.zip (2026-11-17)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/test.zip")
    ap.add_argument("--dest", default="data/egodex_test")
    ap.add_argument("--recheck", action="store_true", help="re-verify even if a manifest exists")
    args = ap.parse_args()

    zp, dest = Path(args.zip), Path(args.dest)
    if (dest / "extract_manifest.json").exists() and not args.recheck:
        print(f"already extracted and CRC-checked: {dest / 'extract_manifest.json'} (use --recheck to verify again)")
        return 0
    size = zp.stat().st_size
    print(f"zip size {size} bytes (expected {EXPECTED_BYTES}): {'OK' if size == EXPECTED_BYTES else 'MISMATCH'}")
    if size != EXPECTED_BYTES:
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with zipfile.ZipFile(zp) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        bad = [i.filename for i in infos
               if PurePosixPath(i.filename).is_absolute() or ".." in PurePosixPath(i.filename).parts]
        if bad:
            print(f"refusing to extract: {len(bad)} unsafe member names, e.g. {bad[:3]}")
            return 3
        ext = Counter(PurePosixPath(i.filename).suffix.lower() for i in infos)
        total = sum(i.file_size for i in infos)
        print(f"{len(infos)} files, {total / 1e9:.2f} GB uncompressed, by type {dict(ext)}")
        done_bytes = 0
        for n, info in enumerate(infos, 1):
            target = dest / info.filename
            if target.exists() and target.stat().st_size == info.file_size:
                # already extracted earlier: re-check its CRC from the archive without writing
                with z.open(info) as src:
                    while src.read(1 << 22):
                        pass
            else:
                z.extract(info, dest)  # reads the whole member, CRC checked at the end
            done_bytes += info.file_size
            if n % 500 == 0 or n == len(infos):
                el = time.time() - t0
                print(f"  {n}/{len(infos)} files, {done_bytes / 1e9:.1f} GB, {el:.0f}s")
    top = sorted({PurePosixPath(i.filename).parts[0] for i in infos})
    manifest = {"zip_bytes": size, "files": len(infos), "uncompressed_bytes": total,
                "by_extension": dict(ext), "top_level": top, "crc_checked": True,
                "seconds": round(time.time() - t0)}
    (dest / "extract_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
