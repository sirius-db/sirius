"""Parse the `=== Pipeline Overview ===` block into a structured DAG.

Input: the raw lines of a single query segment (output of segmenter).
Output: dict ready for json.dump with the schema documented in README.

Assumptions (see README for full list):
  * The Pipeline Overview block starts with the literal header
    `=== Pipeline Overview ===` and ends at the first blank line OR at
    the next section marker (`=== Query Plan DAG ===`).
  * Each pipeline starts with `Pipeline #N: <op chain>`.
  * Indented lines starting with `Input:`, `Dependencies:`, `Output:`
    belong to the most recently seen pipeline.
  * Pipelines without any `Input:` line are leaves.
  * The pipeline whose Output: target is RESULT_COLLECTOR feeds the
    root; the highest-numbered RESULT_COLLECTOR pipeline is treated as
    the root for completeness.
"""

from collections import Counter
from typing import List, Optional

from . import patterns
from .validators import FormatWarnings


SCAN_OP_TYPES = {
    "DUCKDB_SCAN",
    "TABLE_SCAN",
    "GPU_PARQUET_SCAN",
    "PARQUET_SCAN",
}


def _extract_overview_lines(lines: List[str]) -> Optional[List[str]]:
    """Return the lines inside the Pipeline Overview block, or None if absent."""
    start = None
    for i, line in enumerate(lines):
        if patterns.PIPELINE_OVERVIEW_HEADER in line:
            start = i + 1
            break
    if start is None:
        return None
    block: List[str] = []
    for line in lines[start:]:
        s = line.rstrip("\n")
        if not s.strip():
            break
        if any(marker in s for marker in patterns.PIPELINE_OVERVIEW_END_MARKERS):
            break
        block.append(s)
    return block


def parse_plan(lines: List[str], warnings: FormatWarnings) -> Optional[dict]:
    block = _extract_overview_lines(lines)
    if block is None:
        return None

    pipelines: List[dict] = []
    current: Optional[dict] = None

    for raw in block:
        header_match = patterns.PIPELINE_HEADER_RE.match(raw)
        if header_match:
            if current is not None:
                pipelines.append(current)
            pid = int(header_match.group(1))
            chain = header_match.group(2)
            operators = [
                {"type": t, "id": int(i)}
                for (t, i) in patterns.OPERATOR_RE.findall(chain)
            ]
            if not operators:
                warnings.record_drift("pipeline_header_operators", raw)
            current = {
                "pipeline_num": pid,
                "operators": operators,
                "inputs": [],
                "dependencies": [],
                "output": None,
            }
            continue

        if current is None:
            # Stray line before any pipeline header; skip.
            continue

        m = patterns.PIPELINE_INPUT_RE.match(raw)
        if m:
            current["inputs"].append(
                {
                    "from_pipeline": int(m.group(1)),
                    "on_operator": {"type": m.group(2), "id": int(m.group(3))},
                    "port": m.group(4),
                    "barrier": m.group(5),
                }
            )
            continue

        m = patterns.PIPELINE_DEPS_RE.match(raw)
        if m:
            deps = []
            for tok in m.group(1).split(","):
                tok = tok.strip().lstrip("Pipeline #")
                if tok.isdigit():
                    deps.append(int(tok))
            current["dependencies"] = deps
            continue

        m = patterns.PIPELINE_OUTPUT_RE.match(raw)
        if m:
            current["output"] = {
                "to_operator": m.group(1),
                "port": m.group(2),
                "barrier": m.group(3),
            }
            continue

        # Line didn't match anything we know — record drift.
        warnings.record_drift("pipeline_overview_unknown_line", raw)

    if current is not None:
        pipelines.append(current)

    # Build derived indices.
    operator_index: dict = {}
    op_type_counts: Counter = Counter()
    scan_count = 0
    for p in pipelines:
        for op in p["operators"]:
            operator_index[str(op["id"])] = p["pipeline_num"]
            op_type_counts[op["type"]] += 1
            if op["type"] in SCAN_OP_TYPES:
                scan_count += 1

    leaves = [p["pipeline_num"] for p in pipelines if not p["inputs"]]

    # Root = highest pipeline_num whose Output is RESULT_COLLECTOR; if absent,
    # the pipeline that contains RESULT_COLLECTOR as its operator chain.
    root_pipeline = None
    for p in sorted(pipelines, key=lambda x: -x["pipeline_num"]):
        if any(op["type"] == "RESULT_COLLECTOR" for op in p["operators"]):
            root_pipeline = p["pipeline_num"]
            break
    if root_pipeline is None:
        for p in sorted(pipelines, key=lambda x: -x["pipeline_num"]):
            if p["output"] and p["output"]["to_operator"] == "RESULT_COLLECTOR":
                root_pipeline = p["pipeline_num"]
                break

    return {
        "pipelines": pipelines,
        "operator_index": operator_index,
        "counts": {
            "pipelines": len(pipelines),
            "operators": sum(op_type_counts.values()),
            "scans": scan_count,
            "operator_types": dict(op_type_counts),
        },
        "leaves": leaves,
        "root_pipeline": root_pipeline,
    }
