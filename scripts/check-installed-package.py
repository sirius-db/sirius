#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Check installation, relocation, and consumption of the Sirius CMake package."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix="sirius-installed-package-") as temporary:
        temporary = Path(temporary)
        original = temporary / "original"
        relocated = temporary / "relocated"
        run(
            "cmake",
            "--install",
            args.build.resolve(),
            "--prefix",
            original,
            "--component",
            "sirius_library",
        )
        original.rename(relocated)
        for metadata in relocated.rglob("*.cmake"):
            contents = metadata.read_text()
            if str(root) in contents or str(original) in contents:
                raise RuntimeError(f"Non-relocatable package metadata: {metadata}")

        for mode in ("Release", "Debug"):
            consumer = temporary / mode
            configure = [
                "cmake",
                "--no-warn-unused-cli",
                "-S",
                str(root / "test/cmake/installed_consumer"),
                "-B",
                str(consumer),
                "-G",
                "Ninja",
                f"-DCMAKE_BUILD_TYPE={mode}",
                f"-DCMAKE_PREFIX_PATH={relocated}",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE",
            ]
            run(*configure)
            run("cmake", "--build", consumer)
            print(f"{mode}: relocated package consumer built successfully")


if __name__ == "__main__":
    main()
