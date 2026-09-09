#!/usr/bin/env python3
"""Regression probes for experimental/starrocks/tools/compare.py.

Run with: pixi run python docs/onboarding/starrocks-stack/research/benchmark-validation.py
from the repository root. The script owns its fixtures and changes no product files.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile


COMPARE = pathlib.Path(__file__).parents[4] / "experimental/starrocks/tools/compare.py"


def write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def invoke(
    sirius: pathlib.Path, oracle: pathlib.Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(COMPARE), str(sirius), str(oracle)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sirius-compare-validation-") as tmp:
        root = pathlib.Path(tmp)
        oracle = root / "oracle"
        sirius = root / "sirius"
        oracle.mkdir()
        sirius.mkdir()

        write(oracle / "q01.tsv", "v\n1.0\n")
        write(sirius / "q01.r0.out", "v\n1.0\n")
        baseline = invoke(sirius, oracle)
        assert baseline.returncode == 0 and "MATCH" in baseline.stdout, baseline

        write(sirius / "q01.r0.out", "v\n2.0\n")
        write(sirius / "q01.r1.out", "v\n1.0\n")
        cold_warm = invoke(sirius, oracle)
        assert (
            cold_warm.returncode == 1 and "VALUES-DIFFER" in cold_warm.stdout
        ), cold_warm
        assert "r1" in cold_warm.stdout and "MATCH" in cold_warm.stdout, cold_warm
        (sirius / "q01.r1.out").unlink()

        write(sirius / "q01.r0.out", "v\nnan\n")
        nan = invoke(sirius, oracle)
        assert nan.returncode == 0 and "MATCH" in nan.stdout, nan

        write(sirius / "q01.r0.out", "v\n1.0\n")
        write(oracle / "q02.tsv", "v\n2.0\n")
        missing = invoke(sirius, oracle)
        assert (
            missing.returncode == 0 and "1/1 queries match" in missing.stdout
        ), missing

        (sirius / "q01.r0.out").unlink()
        empty = invoke(sirius, oracle)
        assert empty.returncode != 0 and "ValueError: max()" in empty.stderr, empty

        print("baseline=PASS")
        print("wrong_cold_correct_warm=REJECTED")
        print("nan_vs_finite=FALSE_MATCH")
        print("oracle_only_query=SUBSET_POLICY")
        print("empty_result_directory=CRASH")


if __name__ == "__main__":
    main()
