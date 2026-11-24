# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def read(name):
    spec = importlib.util.spec_from_file_location(
        "ablation_" + name, HERE / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PinControls(unittest.TestCase):
    def test_corrected_type_first_order_matches_gb300(self):
        run = read("run")
        stages = [run.stage_spec(i, "type-first") for i in range(6)]
        self.assertEqual(
            [s["label"] for s in stages],
            [
                "all_off",
                "type_narrow",
                "build_probe",
                "dynamic_filter",
                "compressed_pin",
                "fused_decode",
            ],
        )
        self.assertEqual([s["gb300_stage"] for s in stages], list(range(15, 21)))
        self.assertTrue(stages[1]["type_narrow"])
        self.assertFalse(stages[1]["build_probe"])
        self.assertFalse(stages[2]["dynamic_filter"])
        self.assertTrue(stages[3]["dynamic_filter"])
        self.assertFalse(stages[3]["compressed_pin"])
        self.assertTrue(stages[4]["compressed_pin"])
        self.assertFalse(stages[4]["fused_decode"])

    def test_native_coverage_paths_identify_the_logical_table(self):
        run = read("run")
        self.assertEqual(
            run.coverage_table("/data/sf1000/lineitem/part.0.parquet"), "lineitem"
        )
        self.assertEqual(run.coverage_table("orders"), "orders")

    def test_raw_stages_explicitly_disable_every_pin(self):
        worker = read("worker")
        sql = "CALL pin_table('/x', tier='host', name='lineitem', cols=['l_orderkey']);"
        self.assertIn("compression=false", worker.pin_policy(sql, False))
        self.assertNotIn("compression=true", worker.pin_policy(sql, False))

    def test_compressed_stages_do_not_enable_other_tables(self):
        worker = read("worker")
        sql = "\n".join(
            f"CALL pin_table('/x', tier='host', name='{t}', cols=['x']);"
            for t in ["lineitem", "orders", "customer"]
        )
        calls = worker.pin_policy(sql, True).splitlines()
        self.assertIn("compression=true", calls[0])
        self.assertIn("compression=true", calls[1])
        self.assertIn("compression=false", calls[2])
        with self.assertRaises(ValueError):
            worker.pin_policy(
                "CALL pin_table('/x', name='lineitem', compression=true);", False
            )

    def test_only_study_parameters_change_the_saved_preset(self):
        run = read("run")
        base = {"sirius": {"memory": {"host": {"initial_number_pools": 1600}}}}
        off = run.stage_config(base, run.stage_spec(0))
        on = run.stage_config(base, run.stage_spec(5))
        self.assertEqual(
            base, {"sirius": {"memory": {"host": {"initial_number_pools": 1600}}}}
        )
        self.assertEqual(off["sirius"]["memory"], on["sirius"]["memory"])
        self.assertEqual(
            off["sirius"]["operator_params"]["max_build_hash_table_bytes"], 0
        )
        self.assertNotIn("max_build_hash_table_bytes", on["sirius"]["operator_params"])
        self.assertTrue(on["sirius"]["operator_params"]["enable_fused_scan_filter"])


if __name__ == "__main__":
    unittest.main()
