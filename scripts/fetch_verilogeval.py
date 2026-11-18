"""Fetch VerilogEval v2 spec-to-rtl (156 tasks) at a pinned commit into data/ (gitignored).

    uv run python scripts/fetch_verilogeval.py            # -> data/verilogeval_v2/dataset_spec-to-rtl

Only `dataset_spec-to-rtl/*.{txt,sv}` regular files are extracted, with path checks, and the
content is verified against a committed digest so every machine grades the same tasks.
License: MIT, NVlabs (third_party/LICENSE.verilog-eval).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

COMMIT = "c498220d0a52248f8e3fdffe279075215bde2da6"  # NVlabs/verilog-eval main, 2025-02-07
URL = f"https://codeload.github.com/NVlabs/verilog-eval/tar.gz/{COMMIT}"
SUBDIR = "dataset_spec-to-rtl"
# sha256 over sorted "name\0sha256(content)\n" lines of the extracted files.
DIGEST = "8a9a9aea974dfc96af0e31a9aab85aede8b11d5c3c8c4c5912b475a08b04e5e4"
ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "verilogeval_v2"


def content_digest(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.iterdir()):
        h.update(f"{f.name}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=DEST)
    a = ap.parse_args(argv)
    out = a.dest / SUBDIR
    if out.exists() and content_digest(out) == DIGEST:
        print(f"already present and verified: {out}")
        return 0
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=120) as r:
        blob = r.read()
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        for m in tf.getmembers():
            parts = PurePosixPath(m.name).parts
            if len(parts) != 3 or parts[1] != SUBDIR or not m.isreg():
                continue
            name = parts[2]
            if not name.endswith((".txt", ".sv")) or "/" in name or name.startswith("."):
                continue
            f = tf.extractfile(m)
            assert f is not None
            (out / name).write_bytes(f.read())
            n += 1
    got = content_digest(out)
    print(f"extracted {n} files to {out}\ndigest {got}")
    if got != DIGEST:
        print(f"digest mismatch: expected {DIGEST}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
