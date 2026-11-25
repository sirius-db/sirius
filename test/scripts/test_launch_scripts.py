"""Tests of scripts/compare_answers.py, sirius_distributed_config.py, launcher_env.sh and
sirius_net.sh. No GPU."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import compare_answers  # noqa: E402

TIME = "Run Time (s): real 0.5 user 0.1 sys 0.1"


def output(*runs):
    """A client output. Each run is (query, run, lines); lines None is a run that never ended."""
    text = []
    for query, run, lines in runs:
        text += ["marker", f'"Q{query} run {run}"', TIME]
        text += lines + [TIME] if lines is not None else []
    return "\n".join(text) + "\n"


class CompareAnswers(unittest.TestCase):
    def compare(self, run, ref, *args):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "run.txt").write_text(run)
            Path(tmp, "ref.txt").write_text(ref)
            done = subprocess.run(
                [sys.executable, SCRIPTS / "compare_answers.py", "run.txt"]
                + ["--ref-client", "ref.txt", *args],
                cwd=tmp,
                capture_output=True,
                text=True,
            )
        return done.returncode, done.stdout

    def test_cells(self):
        match = compare_answers.cells_match
        self.assertTrue(match("0.04999417093607474", "0.04999417093607475"))
        self.assertTrue(match("38236.12", "38236.1169843049"))
        self.assertFalse(match("0.05", "0.06"))
        self.assertFalse(match("12314853116.0563", "12314855616.0563"))
        self.assertFalse(match("inf", "1.25"))

    def test_rows_match_in_any_order(self):
        ref = [["0.10"], ["0.11"]]
        for got in ([["0.105"], ["0.1049"]], [["0.1049"], ["0.105"]]):
            self.assertEqual(compare_answers.differing_rows(got, ref), ([], []))

    def test_match(self):
        ref = output((1, 1, ["a", "1"]), (1, 2, ["a", "1"]))
        self.assertEqual(self.compare(ref, ref)[0], 0)

    def test_run_that_stopped_early(self):
        ref = output((1, 1, ["a", "1"]), (1, 2, ["a", "1"]), (6, 1, ["a", "2"]))
        code, table = self.compare(output((1, 1, ["a", "1"])), ref)
        self.assertEqual(code, 1)
        self.assertIn("| Q1 | 2 | - | - | - | NOT RUN |", table)
        self.assertIn("| Q6 | 1 | - | - | - | NOT RUN |", table)

    def test_reference_that_never_ended(self):
        code, table = self.compare(output((1, 1, ["a"])), output((1, 1, None)))
        self.assertEqual(code, 1)
        self.assertIn("NO REFERENCE", table)

    def test_limit_tie(self):
        run = output((3, 1, ["k,v", "1,9", "2,5"]))
        ref = output((3, 1, ["k,v", "1,9", "3,5"]))
        self.assertEqual(self.compare(run, ref)[0], 1)
        self.assertEqual(self.compare(run, ref, "--ties", "3:v")[0], 0)


class DistributedConfig(unittest.TestCase):
    def read(self, yaml=None, env=None):
        with tempfile.TemporaryDirectory() as tmp:
            if yaml is not None:
                Path(tmp, "sirius.yaml").write_text(yaml)
            return subprocess.run(
                [sys.executable, SCRIPTS / "sirius_distributed_config.py"],
                cwd=tmp,
                env={"HOME": tmp, **(env or {})},
                capture_output=True,
                text=True,
            )

    def test_exports(self):
        done = self.read(
            "sirius:\n  distributed:\n    cluster: a b\n    node: { gpus: [0, 1] }\n"
        )
        self.assertEqual(done.returncode, 0)
        self.assertEqual(
            done.stdout.splitlines(),
            [
                "export SIRIUS_CLUSTER=${SIRIUS_CLUSTER:-'a b'}",
                "export SIRIUS_NODE_GPUS=${SIRIUS_NODE_GPUS:-0,1}",
            ],
        )

    def test_group_with_no_keys(self):
        done = self.read(
            "sirius:\n  distributed:\n    cluster: x\n    gateway:\n      # port: 1\n"
        )
        self.assertEqual(done.returncode, 0)
        self.assertEqual(
            done.stdout.splitlines(), ["export SIRIUS_CLUSTER=${SIRIUS_CLUSTER:-x}"]
        )

    def test_rails(self):
        done = self.read("sirius:\n  distributed:\n    ucx: { rails: true }\n")
        self.assertEqual(
            done.stdout, "export SIRIUS_UCX_RAILS=${SIRIUS_UCX_RAILS:-true}\n"
        )

    def test_no_file(self):
        done = self.read()
        self.assertEqual((done.returncode, done.stdout), (0, ""))

    def test_errors_print_nothing(self):
        for yaml in (
            "sirius:\n  distributed:\n    cluster: x\n    bogus: 1\n",
            "sirius:\n  distributed:\n    cluster: { a: [{ b: 1 }] }\n",
            "sirius:\n  distributed: [1, 2]\n",
            "sirius: [\n",
        ):
            done = self.read(yaml)
            self.assertNotEqual(done.returncode, 0, yaml)
            self.assertEqual(done.stdout, "", yaml)

    def test_missing_explicit_file(self):
        fallback = "sirius:\n  distributed:\n    cluster: other\n"
        done = self.read(fallback, {"SIRIUS_CONFIG_FILE": "/nonexistent/sirius.yaml"})
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")


class LauncherEnv(unittest.TestCase):
    def source(self, rails, env):
        """Sources launcher_env.sh on a host whose rail NICs are `rails` ("" is none)."""
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy(SCRIPTS / "launcher_env.sh", tmp)
            found = f"echo {rails}" if rails else "return 1"
            Path(tmp, "sirius_net.sh").write_text(
                f"sirius_rail_devices() {{ {found}; }}\n"
            )
            show = "$UCX_TLS $UCX_NET_DEVICES ${SIRIUS_GATEWAY_EXCHANGE_HOST_STAGING:-unset}"
            return subprocess.run(
                [
                    "bash",
                    "-euc",
                    f'source "$1/launcher_env.sh" && echo "{show}"',
                    "-",
                    tmp,
                ],
                env={"PATH": "/usr/bin:/bin", **env},
                capture_output=True,
                text=True,
            )

    def test_rails_write_over_nixl(self):
        done = self.source("mlx5_0:1,mlx5_1:1", {"SIRIUS_UCX_RAILS": "1"})
        self.assertEqual(done.stdout, "all mlx5_0:1,mlx5_1:1 never\n", done.stderr)

    def test_rails_keep_a_set_host_staging(self):
        env = {
            "SIRIUS_UCX_RAILS": "true",
            "SIRIUS_GATEWAY_EXCHANGE_HOST_STAGING": "auto",
        }
        done = self.source("mlx5_0:1,mlx5_1:1", env)
        self.assertEqual(done.stdout, "all mlx5_0:1,mlx5_1:1 auto\n", done.stderr)

    def test_tcp_leaves_host_staging_to_the_gateway(self):
        done = self.source("mlx5_0:1,mlx5_1:1", {"UCX_NET_DEVICES": "bond0"})
        self.assertEqual(
            done.stdout, "cuda_copy,cuda_ipc,tcp,self bond0 unset\n", done.stderr
        )

    def test_rails_keep_set_ucx_values(self):
        env = {"SIRIUS_UCX_RAILS": "1", "UCX_TLS": "rc", "UCX_NET_DEVICES": "mlx5_0:1"}
        done = self.source("mlx5_0:1,mlx5_1:1", env)
        self.assertEqual(done.stdout, "rc mlx5_0:1 never\n", done.stderr)

    def test_rails_takes_0_or_1(self):
        done = self.source("mlx5_0:1,mlx5_1:1", {"SIRIUS_UCX_RAILS": "rails"})
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")

    def test_rails_without_rail_nics_fail(self):
        done = self.source("", {"SIRIUS_UCX_RAILS": "1"})
        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(done.stdout, "")


class RailDevices(unittest.TestCase):
    def rails(self, ports):
        """sirius_rail_devices on a host with these (device, port, interface, state, speed, bonded)."""
        with tempfile.TemporaryDirectory() as tmp:
            for dev, port, net, state, speed, bonded in ports:
                ib = Path(tmp, "class/infiniband", dev)
                (ib / "ports" / str(port)).mkdir(parents=True)
                (ib / "ports" / str(port) / "state").write_text(f"{state}\n")
                (ib / "device/net" / net).mkdir(parents=True)
                (ib / "device/net" / net / "dev_port").write_text(f"{port - 1}\n")
                Path(tmp, "class/net", net).mkdir(parents=True)
                Path(tmp, "class/net", net, "speed").write_text(f"{speed}\n")
                if bonded:
                    Path(tmp, "class/net", net, "master").touch()
            return subprocess.run(
                [
                    "bash",
                    "-euc",
                    'source "$1" && sirius_rail_devices "$2"',
                    "-",
                    SCRIPTS / "sirius_net.sh",
                    tmp,
                ],
                capture_output=True,
                text=True,
            )

    def test_one_port_per_device(self):
        done = self.rails(
            [
                ("mlx5_0", 1, "rail0", "4: ACTIVE", 400000, False),
                ("mlx5_1", 1, "rail1", "4: ACTIVE", 400000, False),
                ("mlx5_2", 1, "eth0", "4: ACTIVE", 200000, False),
                ("mlx5_3", 1, "eth1", "4: ACTIVE", 400000, True),
                ("mlx5_4", 1, "rail4", "1: DOWN", 400000, False),
            ]
        )
        self.assertEqual(done.stdout, "mlx5_0:1,mlx5_1:1\n", done.stderr)

    def test_a_port_other_than_1(self):
        done = self.rails(
            [
                ("mlx4_0", 1, "ib0", "1: DOWN", 400000, False),
                ("mlx4_0", 2, "ib1", "4: ACTIVE", 400000, False),
                ("mlx4_1", 1, "ib2", "4: ACTIVE", 400000, False),
                ("mlx4_1", 2, "ib3", "4: ACTIVE", 400000, True),
            ]
        )
        self.assertEqual(done.stdout, "mlx4_0:2,mlx4_1:1\n", done.stderr)

    def test_fewer_than_two_rails_is_none(self):
        done = self.rails([("mlx5_0", 1, "rail0", "4: ACTIVE", 400000, False)])
        self.assertEqual((done.returncode, done.stdout), (1, ""))


if __name__ == "__main__":
    unittest.main()
