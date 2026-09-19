#!/usr/bin/env python3
"""Evict a dataset from the page cache before a cold run (plan-doc experiments/sf10-bench §6.4).

    scripts/evict-cache.py DIR [DIR...] [--drop-caches]

posix_fadvise(POSIX_FADV_DONTNEED) on every regular file under the directories: no root
needed, drops the clean pages of exactly those files (a benchmark box has nothing else
worth keeping cached anyway). With --drop-caches, also `sync; echo 3 > /proc/sys/vm/drop_caches`
through passwordless sudo when available (the ClickBench way: page cache, dentries and inodes
of everything), silently skipped otherwise. Prints how many files and bytes were evicted;
a reader can verify with `fincore` or /proc/meminfo Cached.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def evict(path: Path) -> int:
    with path.open("rb") as handle:
        fd = handle.fileno()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        return os.fstat(fd).st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dirs", nargs="+", type=Path, help="dataset roots (symlinks are followed)")
    parser.add_argument("--drop-caches", action="store_true", help="also drop the whole page cache via sudo -n")
    args = parser.parse_args()

    files = 0
    total = 0
    for root in args.dirs:
        root = root.resolve()
        if root.is_file():
            candidates = [root]
        else:
            candidates = sorted(p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            try:
                total += evict(path)
                files += 1
            except OSError as exc:
                print(f"evict-cache: {path}: {exc}", file=sys.stderr)
    print(f"evict-cache: fadvise(DONTNEED) on {files} file(s), {total / 2**30:.2f} GiB")

    if args.drop_caches:
        if shutil.which("sudo") and subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
            subprocess.run(["sync"], check=True)
            subprocess.run(["sudo", "-n", "tee", "/proc/sys/vm/drop_caches"], input=b"3\n", check=True, stdout=subprocess.DEVNULL)
            print("evict-cache: dropped the page cache (echo 3 > /proc/sys/vm/drop_caches)")
        else:
            print("evict-cache: no passwordless sudo, page cache not dropped globally (fadvise only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
