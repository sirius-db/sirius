# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Quent query-window extraction for prep_analysis_bundles, on synthetic captures."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from prep_analysis_bundles import quent_query_windows  # noqa: E402


def write(ctx, entity, records):
    directory = ctx / entity
    directory.mkdir(parents=True)
    with open(directory / "0.ndjson", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def init(qid, ts, label, group={"target": "g"}):
    return {
        "id": qid,
        "timestamp": ts,
        "data": {"Init": {"instance_name": label, "query_group_id": group}},
    }


def finished(qid, ts):
    return {"id": qid, "timestamp": ts, "data": {"Finished": {}}}


def details(qid, revision, label):
    metadata = {"instance_name": label} if label else {"disposition": "gpu"}
    return {
        "id": f"d-{qid}-{revision}",
        "timestamp": 0,
        "data": {
            "Update": {
                "query_id": {"target": qid},
                "revision": revision,
                "metadata": metadata,
            }
        },
    }


class QuentQueryWindows(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="prep-bundles-")
        self.addCleanup(temporary.cleanup)
        self.ctx = Path(temporary.name)

    def windows(self):
        return {
            key: (qid, w)
            for qid, key, w in quent_query_windows(self.ctx, {"g": "power"})
        }

    def test_late_label_from_latest_query_details_revision(self):
        write(self.ctx, "Query", [init("a", 10, "unnamed_query"), finished("a", 20)])
        # Out of order on disk; the highest revision with a label wins.
        write(
            self.ctx,
            "QueryDetails",
            [details("a", 3, "q7"), details("a", 1, None), details("a", 2, "stale")],
        )
        self.assertEqual(self.windows(), {("power", "q7"): ("a", (10, 20))})

    def test_init_label_is_the_fallback_for_older_captures(self):
        write(self.ctx, "query", [init("b", 5, "q1", group="g"), finished("b", 9)])
        self.assertEqual(self.windows(), {("power", "q1"): ("b", (5, 9))})

    def test_unlabeled_queries_do_not_shadow_labeled_ones(self):
        write(
            self.ctx,
            "Query",
            [
                init("a", 10, "unnamed_query"),
                finished("a", 20),
                init("c", 30, "unnamed_query"),
                finished("c", 40),
            ],
        )
        write(self.ctx, "QueryDetails", [details("a", 1, "q3")])
        windows = self.windows()
        self.assertEqual(windows[("power", "q3")], ("a", (10, 20)))
        self.assertEqual(windows[("power", "unnamed_query")], ("c", (30, 40)))


if __name__ == "__main__":
    unittest.main()
