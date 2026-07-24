#!/usr/bin/env python3
"""Download a ROBOTIS R2 share manifest.

Usage:
  python3 download_manifest.py Task_101010_lerobot_download_manifest.json ./Task_101010_lerobot
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def safe_output_path(root: Path, relpath: str) -> Path:
    value = relpath.strip().replace("\\", "/")
    if not value or value.startswith("/") or value.endswith("/"):
        raise ValueError(f"invalid file path: {relpath!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe file path: {relpath!r}")
    return root.joinpath(*parts)


def download_one(root: Path, item: dict) -> tuple[str, int]:
    relpath = item.get("path") or ""
    url = item.get("url") or ""
    if not url:
        raise ValueError(f"missing URL for {relpath}")
    output = safe_output_path(root, relpath)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    size = 0
    try:
        with os.fdopen(fd, "wb") as fp:
            with urllib.request.urlopen(url, timeout=300) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fp.write(chunk)
                    size += len(chunk)
        os.replace(tmp_name, output)
        return relpath, size
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    manifest_path = Path(sys.argv[1])
    output_root = Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files") or []
    if not isinstance(files, list) or not files:
        raise SystemExit("manifest has no files")
    storage_mode = str(manifest.get("storageMode") or "expanded_files")
    if storage_mode == "archive_zip":
        archive = manifest.get("archive") or {}
        print(f"Archive dataset: downloading {archive.get('path') or 'zip file'}; unzip manually if needed")
    output_root.mkdir(parents=True, exist_ok=True)
    workers = min(8, max(1, len(files)))
    total = len(files)
    done = 0
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, output_root, item) for item in files]
        for future in as_completed(futures):
            relpath, size = future.result()
            done += 1
            bytes_done += size
            print(f"[{done}/{total}] {relpath} ({size} bytes)")
    print(f"Downloaded {done} files, {bytes_done} bytes -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
