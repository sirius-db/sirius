#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
import copy, json, sys, unittest, subprocess, tempfile
from pathlib import Path
from validate import validate, load, SCENARIOS, EVENTS

ROOT = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else None


def causal_mutations(root):
    cases = []

    def original(scenario):
        return load(root / (scenario + ".jsonl"))

    rows = original("reuse")
    for event in rows:
        if event["kind"] in EVENTS and event["id"] == 64:
            for field in ("current", "selected"):
                if event[field] == 1:
                    event[field] = 99
    cases.append(("unpublished_P99", "reuse", rows))
    for name, scenario, counter in [
        ("seed_counter_zero", "reuse", "seeds"),
        ("refresh_counter_zero", "stale", "refreshes"),
    ]:
        rows = original(scenario)
        for event in rows:
            if event["kind"] in EVENTS:
                event[counter] = 0
        cases.append((name, scenario, rows))
    rows = original("reuse")
    next(e for e in rows if e["kind"] == "seed_begin")["seeds"] = 1
    cases.append(("seed_increment_early", "reuse", rows))
    rows = original("reuse")
    next(e for e in rows if e["kind"] == "seed_end")["seeds"] = 0
    cases.append(("seed_increment_late", "reuse", rows))
    rows = original("stale")
    next(e for e in rows if e["kind"] == "invalidated" and e["id"] == 128)[
        "refreshes"
    ] = 1
    cases.append(("refresh_increment_early", "stale", rows))
    rows = original("stale")
    start = next(
        i
        for i, e in enumerate(rows)
        if e["kind"] == "select_begin"
        and e["id"] == 128
        and e["refreshed"]
        and not e["delegated"]
    )
    end = next(
        i
        for i, e in enumerate(rows)
        if i > start
        and e["kind"] == "select_begin"
        and e["id"] == 128
        and e["delegated"]
    )
    for event in rows[start:end]:
        if event["kind"] in EVENTS:
            event["refreshes"] -= 1
    cases.append(("refresh_increment_late", "stale", rows))
    rows = original("off")
    start = next(i for i, e in enumerate(rows) if e["kind"] == "seed_end")
    for event in rows[start:]:
        if event["kind"] in EVENTS:
            event["seeds"] += 1
    cases.append(("failed_seed_increment", "off", rows))
    return cases


class NegativeReceipts(unittest.TestCase):

    def test_mutations(self):
        base = load(ROOT / "stale.jsonl")
        validate(base, "stale")
        cases = []

        def edit(name, kind, field, value, id_=64):
            rows = copy.deepcopy(base)
            e = next((x for x in rows if x["kind"] == kind and x.get("id") == id_))
            e[field] = value
            cases.append((name, rows))

        for field, value in [
            ("physical", 999),
            ("open", 2),
            ("context", 2),
            ("selected", 2),
            ("current", 0),
            ("attempt", 2),
            ("reuse", 0),
            ("strong", 0),
            ("hits", 99),
        ]:
            edit("selection_" + field, "select_end", field, value)
        for field, value in [
            ("selected", 2),
            ("attempt", 3),
            ("delegated", 0),
            ("status", 206),
            ("bytes", 99),
        ]:
            edit("finish_" + field, "finish", field, value)
        edit("wrong_classifier", "classification", "result", 0)
        edit("drop_new_publication", "invalidated", "current", 0)
        edit("wrong_invalidated_identity", "invalidated", "selected", 2)
        edit("double_completion", "terminal", "completions", 2)
        edit("wrong_terminal", "terminal", "result", "object_changed")
        rows = copy.deepcopy(base)
        rows[0]["failed"] = 1
        cases.append(("observer_failure", rows))
        for kind in [
            "select_begin",
            "submitted",
            "finish",
            "invalidated",
            "seed_end",
            "terminal",
            "wire",
        ]:
            rows = copy.deepcopy(base)
            idx = next((i for (i, x) in enumerate(rows) if x["kind"] == kind))
            rows.pop(idx)
            if kind not in ["terminal", "wire"]:
                rows[0]["events"] -= 1
            cases.append(("missing_" + kind, rows))
        rows = copy.deepcopy(base)
        rows.append(copy.deepcopy(next((x for x in rows if x["kind"] == "terminal"))))
        cases.append(("duplicate_terminal", rows))
        rows = copy.deepcopy(base)
        next((x for x in rows if x["kind"] == "wire"))["range"] = "bytes=999-1030"
        cases.append(("wrong_wire_operation", rows))
        for name, rows in cases:
            with self.subTest(name=name):
                with self.assertRaises((ValueError, KeyError, StopIteration)):
                    validate(rows, "stale")
        print(json.dumps({"negative_mutations_rejected": len(cases)}))

    def test_cli_failure(self):
        tool = Path(__file__).with_name("validate.py")
        cases = []
        for name in ["observer_failure", "observer_overflow", "observer_exception"]:
            cases.append((name, load(ROOT / (name + ".jsonl")), "reuse"))
        bad = load(ROOT / "stale.jsonl")
        next((e for e in bad if e["kind"] == "invalidated" and e["id"] == 64))[
            "current"
        ] = 0
        cases.append(("wrong_publication", bad, "stale"))
        bad = load(ROOT / "reuse.jsonl")
        bad.pop(1)
        bad[0]["events"] -= 1
        cases.append(("missing_event", bad, "reuse"))
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows, scenario in cases:
                f = Path(tmp) / (name + ".jsonl")
                f.write_text("".join((json.dumps(x) + "\n" for x in rows)))
                result = subprocess.run(
                    [
                        sys.executable,
                        str(tool),
                        "--single",
                        str(f),
                        "--scenario",
                        scenario,
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, name)
        print(json.dumps({"cli_negative_rejections": len(cases)}))

    def test_causal_chain_and_cli(self):
        tool = Path(__file__).with_name("validate.py")
        mutations = causal_mutations(ROOT)
        with tempfile.TemporaryDirectory(prefix="d1-causal-negative-") as tmp:
            directory = Path(tmp)
            sides = [directory / "reference", directory / "candidate"]
            names = list(SCENARIOS) + [
                "observer_failure",
                "observer_overflow",
                "observer_exception",
            ]
            for side in sides:
                side.mkdir()
                for name in names:
                    (side / (name + ".jsonl")).write_bytes(
                        (ROOT / (name + ".jsonl")).read_bytes()
                    )
            pair = [
                sys.executable,
                str(tool),
                str(sides[0]),
                str(sides[1]),
                "--out",
                str(directory / "comparison.json"),
            ]
            positive = subprocess.run(pair, capture_output=True, text=True, timeout=10)
            self.assertEqual(positive.returncode, 0, positive.stderr)
            for name, scenario, rows in mutations:
                with self.subTest(case=name):
                    original = load(ROOT / (scenario + ".jsonl"))
                    self.assertNotEqual(
                        rows, original, "negative control did not mutate anything"
                    )
                    with self.assertRaises(ValueError):
                        validate(rows, scenario)
                    payload = "".join(json.dumps(row) + "\n" for row in rows)
                    candidate = sides[1] / (scenario + ".jsonl")
                    candidate.write_text(payload)
                    single = [
                        sys.executable,
                        str(tool),
                        "--single",
                        str(candidate),
                        "--scenario",
                        scenario,
                    ]
                    for mode, command in [
                        ("single", single),
                        ("one-sided-pair", pair),
                        ("common-error-pair", pair),
                    ]:
                        if mode == "common-error-pair":
                            (sides[0] / (scenario + ".jsonl")).write_text(payload)
                        result = subprocess.run(
                            command, capture_output=True, text=True, timeout=10
                        )
                        self.assertEqual(
                            result.returncode, 1, (name, mode, result.stderr)
                        )
                        self.assertIn("ValueError:", result.stderr)
                    for side in sides:
                        (side / (scenario + ".jsonl")).write_bytes(
                            (ROOT / (scenario + ".jsonl")).read_bytes()
                        )
        print(
            json.dumps(
                {
                    "causal_negative_rejections": len(mutations),
                    "causal_cli_rejections": len(mutations) * 3,
                    "paired_positive_controls": 1,
                    "cases": [name for name, _, _ in mutations],
                }
            )
        )

    def test_real_valid(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                validate(load(ROOT / (scenario + ".jsonl")), scenario)


if __name__ == "__main__":
    unittest.main()
