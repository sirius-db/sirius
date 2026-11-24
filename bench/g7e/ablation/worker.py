# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Use the pinned engine harness with the corrected GB300 pinning protocol."""
import importlib
import json
from pathlib import Path
import re
import sys

GUIDE = Path(__file__).resolve().parents[1] / "guide-26b5caba"
sys.path.insert(0, str(GUIDE))
import worker as observed_worker  # noqa: E402


def pin_policy(sql, compressed):
    """Make every pin explicit; compress only the two GB300 plan tables."""
    statements = []
    for statement in sql.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        match = re.search(r"\bname='([^']+)'", statement)
        if not match or "CALL pin_table(" not in statement:
            raise ValueError("Unexpected pin SQL")
        if re.search(r"\bcompression\s*=", statement):
            raise ValueError("Pin compression was already specified")
        enabled = compressed and match[1] in ("lineitem", "orders")
        statements.append(
            statement.rstrip()[:-1] + f", compression={str(enabled).lower()});"
        )
    return "\n".join(statements) + "\n"


def main():
    engine = Path(sys.argv[sys.argv.index("--engine-root") + 1])
    config = Path(sys.argv[sys.argv.index("--config") + 1])
    # The stage spec is separate from YAML: no unknown engine parameters.
    stage = next(p for p in config.parents if (p / "stage-spec.json").is_file())
    spec = json.loads((stage / "stage-spec.json").read_text())
    compressed = spec["compressed_pin"]
    sys.path.insert(0, str(engine / "test/tpch_performance"))
    harness = importlib.import_module("performance_test")
    original_pin = harness.emit_pin
    harness.emit_pin = lambda *a, **kw: pin_policy(original_pin(*a, **kw), compressed)
    original_main = harness.main

    def measured_main():
        # No mode/profile override: default grouped ordering has no between-query
        # cache clears and never derives/replaces YAML. Startup clears once.
        index = sys.argv.index("--mode")
        del sys.argv[index : index + 2]
        if compressed:
            sys.argv.extend(
                [
                    "--compression-plan-dir",
                    str(engine / "src/compression/simpatico_codegen/plans/tpch_sf1000"),
                ]
            )
        observed_worker.event(
            "ablation_runtime",
            argv=sys.argv,
            stage=spec,
            fresh_process_per_stage=True,
            cache_clear="startup only",
            pin="host per query",
        )
        return original_main()

    harness.main = measured_main
    observed_worker.main()


if __name__ == "__main__":
    main()
