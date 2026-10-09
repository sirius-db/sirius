#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Run configuration consumers with CUDA driver stubs, including on CPU CI runners."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

prefix = Path(os.environ["CONDA_PREFIX"])
stub_dirs = [prefix / "lib/stubs", *prefix.glob("targets/*/lib/stubs")]
with tempfile.TemporaryDirectory(prefix="sirius-driver-stubs-") as temporary:
    for library in ("libcuda", "libnvidia-ml"):
        stub = next(
            (p / f"{library}.so" for p in stub_dirs if (p / f"{library}.so").is_file()),
            None,
        )
        if stub is None:
            raise SystemExit(f"Missing {library} stub under {prefix}")
        (Path(temporary) / f"{library}.so.1").symlink_to(stub.resolve())
    library_dirs = [
        temporary,
        str(prefix / "lib"),
        *map(str, prefix.glob("targets/*/lib")),
    ]
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join([*library_dirs, env.get("LD_LIBRARY_PATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    sys.exit(
        subprocess.call(
            [
                "ctest",
                "--test-dir",
                sys.argv[1],
                "--output-on-failure",
                "--no-tests=error",
            ],
            env=env,
        )
    )
