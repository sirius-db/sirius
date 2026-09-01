# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================
"""TPC-H driver for StarRocks FE + Sirius CNs.

Same CLI shape as performance_test.py (grouped/sequential, --pin gpu,
--pin-after-iteration) but the engine is a cluster: FILES() SQL to the FE,
ADMIN EXECUTE pin_table on every CN. Launch is subprocess of the existing
shell scripts, not a reimplementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from tpch_pin_columns import (
    emit_admin_pin,
    emit_admin_pin_all,
    emit_admin_unpin,
    emit_admin_unpin_all,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR_DIR = os.path.join(REPO_ROOT, "experimental", "starrocks")
QUERIES_DIR = os.path.join(SR_DIR, "benchmarks", "tpch", "queries")
PINNED_DIR = os.path.join(SR_DIR, "benchmarks", "pinned")
DEFAULT_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
TWO_HOST_SCALE_FACTORS = {1000, 3000, 10000}
PIN_RETRY_VALIDATE = 5
PIN_RETRY_SLEEP_S = 15
VALIDATION_ABS_TOL = 1e-10


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}", flush=True)


def _git_capture(args):
    try:
        out = subprocess.run(
            args,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_git_info():
    return _git_capture(["git", "rev-parse", "HEAD"]), _git_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"]
    )


def parse_query_spec(spec):
    if spec is None:
        return list(range(1, 23))
    nums = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            nums.extend(range(int(lo), int(hi) + 1))
        else:
            nums.append(int(token))
    return nums


def parse_hosts(spec):
    hosts = [h.strip() for h in spec.split(",") if h.strip()]
    if not hosts:
        raise SystemExit("--hosts must list at least one host")
    if len(hosts) > 2:
        raise SystemExit(
            "this fleet is two GB200 hosts; --hosts length 1 (local) or 2 "
            "(FE host, remote CN host)"
        )
    return hosts


def infer_scale_factor(input_dir, explicit):
    if explicit is not None:
        return explicit
    joined = os.path.abspath(input_dir)
    m = re.search(r"sf(\d+)", joined, re.IGNORECASE)
    if not m:
        raise SystemExit(
            "could not infer scale factor from --input; pass --scale-factor"
        )
    return int(m.group(1))


def q11_fraction(scale_factor):
    return f"{0.0001 / scale_factor:.12f}"


def load_query_sql(qnum, data_dir, scale_factor):
    path = os.path.join(QUERIES_DIR, f"q{qnum:02d}.sql")
    if not os.path.isfile(path):
        raise SystemExit(f"StarRocks query file not found: {path}")
    sql = Path(path).read_text()
    sql = sql.replace("__TPCH_DATA__", os.path.abspath(data_dir))
    if qnum == 11:
        sql = sql.replace("0.0001000000", q11_fraction(scale_factor))
    return sql


def setup_benchmark_dir(
    output_root,
    mode,
    iterations,
    queries,
    pin,
    name=None,
    pin_compression=False,
    compression_plan_dir=None,
    gpus=None,
    hosts=None,
    scale_factor=None,
    config_path=None,
    duckdb_results_source=None,
):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if name:
        benchmark_name = f"tpch_{ts}_{name}"
    else:
        benchmark_name = f"tpch_{ts}_{mode}_starrocks_iter{iterations}"
    benchmark_dir = os.path.join(output_root, benchmark_name)
    csv_dir = os.path.join(benchmark_dir, "csv")
    log_dir = os.path.join(benchmark_dir, "log_dir")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    runtime_csv = os.path.join(csv_dir, "runtimes.csv")
    if config_path and os.path.isfile(config_path):
        shutil.copy2(config_path, os.path.join(benchmark_dir, "config.yml"))
    commit, branch = get_git_info()
    metadata = {
        "commit": commit,
        "branch_name": branch,
        "date": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "iterations": iterations,
        "engine": "starrocks",
        "data_source": "parquet",
        "queries": [f"q{q}" for q in queries],
        "pin": pin,
        "pin_compression": pin_compression,
        "compression_plan_dir": compression_plan_dir,
        "gpus_per_host": gpus,
        "hosts": hosts,
        "scale_factor": scale_factor,
        "runtime_file": os.path.relpath(runtime_csv, benchmark_dir),
        "duckdb_results_source": duckdb_results_source,
    }
    with open(os.path.join(benchmark_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    return benchmark_dir, runtime_csv, log_dir


class MysqlClient:
    def __init__(self, host, port):
        self.host = host
        self.port = int(port)

    def _cmd(self, sql, extra=None):
        cmd = [
            "pixi",
            "run",
            "-e",
            "client",
            "mysql",
            "-h",
            self.host,
            "-P",
            str(self.port),
            "-uroot",
            "--batch",
            "--connect-timeout=5",
            "-e",
            sql,
        ]
        if extra:
            cmd[5:5] = extra
        return cmd

    def run(self, sql, timeout=None):
        cmd = self._cmd(sql)
        if timeout:
            cmd = ["timeout", "--signal=KILL", str(int(timeout))] + cmd
        proc = subprocess.run(
            cmd,
            cwd=SR_DIR,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out

    def execute(self, sql, timeout=None):
        rc, out = self.run(sql, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"mysql failed (rc={rc}): {out.strip()[-2000:]}")
        return out

    def alive_rows(self):
        rc, out = self.run("SHOW COMPUTE NODES;")
        if rc != 0:
            return []
        lines = [ln for ln in out.splitlines() if ln.strip()]
        if len(lines) < 2:
            return []
        header = lines[0].split("\t")
        try:
            alive_i = header.index("Alive")
            id_i = 0
        except ValueError:
            return []
        rows = []
        for ln in lines[1:]:
            cols = ln.split("\t")
            if len(cols) <= alive_i:
                continue
            if cols[alive_i] == "true":
                rows.append(cols[id_i])
        return rows

    def alive_count(self):
        return len(self.alive_rows())


def _is_pin_timeout(text):
    t = text.lower()
    return (
        "rpc failed" in t
        or "timeout" in t
        or "timed out" in t
        or "executecommand" in t and "fail" in t
    )


def _is_validate_error(text):
    return "unable to validate object" in text.lower()


class Cluster:
    def __init__(
        self,
        mysql: MysqlClient,
        *,
        gpus,
        hosts,
        gpu_mem,
        staging,
        host_mem,
        pipeline_dop,
        pin_compression,
        compression_plan_dir,
        config_dir,
        scale_factor,
        query_timeout,
        pin_enabled,
    ):
        self.mysql = mysql
        self.gpus = gpus
        self.hosts = hosts
        self.gpu_mem = gpu_mem
        self.staging = staging
        self.host_mem = host_mem
        self.pipeline_dop = pipeline_dop
        self.pin_compression = pin_compression
        self.compression_plan_dir = compression_plan_dir
        self.config_dir = config_dir
        self.scale_factor = scale_factor
        self.query_timeout = query_timeout
        self.pin_enabled = pin_enabled
        self.expected = gpus * len(hosts)
        self._up_proc = None

    def _env(self, extra=None):
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["NUM_CNS"] = str(self.expected if len(self.hosts) > 1 else self.gpus)
        env["NUM_CNS_PER_HOST"] = str(self.gpus)
        env["GPU_MEM"] = self.gpu_mem
        env["HOST_MEM"] = self.host_mem
        env["STAGING"] = self.staging
        env["SIRIUS_EXCHANGE_STAGING_BYTES"] = self.staging
        if self.pipeline_dop is not None:
            env["PIPELINE_DOP"] = str(self.pipeline_dop)
        env["SCALE_FACTOR"] = str(self.scale_factor)
        env["TPCH_DATA"] = env.get("TPCH_DATA", "")
        env["CN_GPU"] = os.environ.get("CN_GPU", "")
        if extra:
            env.update(extra)
        return env

    def gen_config(self):
        os.makedirs(self.config_dir, exist_ok=True)
        env = self._env(
            {
                "NUM_CNS": str(self.gpus),
                "OUT_DIR": self.config_dir,
                "ENABLE_PIN_COMPRESSION": "1" if self.pin_compression else "0",
            }
        )
        if self.compression_plan_dir:
            env["PLAN_DIR"] = self.compression_plan_dir
        script = os.path.join(PINNED_DIR, "gen-config.sh")
        log(f"generating CN YAML in {self.config_dir}")
        subprocess.run([script], cwd=SR_DIR, env=env, check=True)

    def wait_alive(self, timeout_s=300):
        deadline = time.time() + timeout_s
        prev = -1
        while time.time() < deadline:
            n = self.mysql.alive_count()
            if n == self.expected and n == prev:
                log(f"cluster: {n} alive compute nodes")
                return
            prev = n
            time.sleep(2)
        raise RuntimeError(
            f"cluster did not reach {self.expected} alive CNs "
            f"(last count {self.mysql.alive_count()})"
        )

    def fe_set(self):
        timeout = max(int(self.query_timeout), 1800)
        stmts = [f"SET GLOBAL query_timeout = {timeout};"]
        if self.pipeline_dop is not None:
            stmts.append("SET GLOBAL enable_pipeline_engine = true;")
            stmts.append(f"SET GLOBAL pipeline_dop = {int(self.pipeline_dop)};")
        if self.pin_enabled:
            stmts.append(
                'ADMIN SET FRONTEND CONFIG ("files_query_whole_file_ranges" = "true");'
            )
        sql = " ".join(stmts)
        log(f"FE SET: {sql}")
        self.mysql.execute(sql)

    def _teardown_1host(self):
        subprocess.run(
            ["pkill", "-f", "[s]irius-starrocks-cn"],
            capture_output=True,
        )
        subprocess.run(
            ["pkill", "-f", "[S]tarRocksFE"],
            capture_output=True,
        )
        for _ in range(30):
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            if not proc.stdout.strip():
                break
            time.sleep(2)
        time.sleep(2)
        self._up_proc = None

    def _teardown_2host(self):
        stop = os.path.join(SR_DIR, "benchmarks", "stop-cn-2host.sh")
        subprocess.run([stop], cwd=SR_DIR, check=False)
        remote = os.environ.get("REMOTE_HOST", "presto-gb200-gcn-09")
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                remote,
                f"cd {SR_DIR} && ./benchmarks/stop-cn-2host.sh",
            ],
            check=False,
        )

    def teardown(self):
        log("tearing down cluster")
        if len(self.hosts) == 1:
            self._teardown_1host()
        else:
            self._teardown_2host()

    def _launch_1host(self):
        self._teardown_1host()
        self.gen_config()
        env = self._env(
            {
                "NUM_CNS": str(self.gpus),
                "CONFIG_DIR": self.config_dir,
                "STAGING": self.staging,
            }
        )
        up = os.path.join(PINNED_DIR, "up.sh")
        log_path = "/tmp/starrocks-perf-cluster.log"
        log(f"starting FE + {self.gpus} CNs via up.sh (log {log_path})")
        self._up_proc = subprocess.Popen(
            ["nohup", up],
            cwd=SR_DIR,
            env=env,
            stdout=open(log_path, "ab"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait_alive()
        self.fe_set()

    def _launch_2host(self):
        if self.scale_factor not in TWO_HOST_SCALE_FACTORS:
            raise SystemExit(
                f"2-host launch needs --scale-factor in {sorted(TWO_HOST_SCALE_FACTORS)} "
                f"(bench/gb200-8gpu/sf*/env.sh); got {self.scale_factor}. "
                "Use --hosts 127.0.0.1 for other scales."
            )
        host = socket.gethostname()
        if "gcn-18" not in host:
            raise SystemExit(
                f"2-host relaunch.sh must run on gcn-18 (this host is {host})"
            )
        env = self._env()
        env["FE_HOST"] = self.hosts[0]
        env["CN09_HOST"] = self.hosts[1]
        if self.pin_compression:
            self.gen_config()
            env["SIRIUS_CONFIG_DIR"] = self.config_dir
        log(f"starting 2-host cluster via relaunch.sh ({self.expected} CNs)")
        subprocess.run(
            ["./configs/gb200-8gpu/relaunch.sh"],
            cwd=SR_DIR,
            env=env,
            check=True,
        )
        self.wait_alive(timeout_s=60)
        self.fe_set()

    def launch(self):
        if len(self.hosts) == 1:
            self._launch_1host()
        else:
            self._launch_2host()

    def restart(self):
        log("restarting cluster after refuse")
        self.launch()


def admin_execute_on(mysql: MysqlClient, node_id, script, timeout=620):
    sql = f"ADMIN EXECUTE ON {node_id} '{script.rstrip()}'"
    last = ""
    for attempt in range(1, PIN_RETRY_VALIDATE + 1):
        rc, out = mysql.run(sql, timeout=timeout)
        last = out
        if rc == 0:
            return out
        if _is_pin_timeout(out):
            raise RuntimeError(
                f"ADMIN EXECUTE ON {node_id} timed out (do not retry; "
                f"watch CN log for pin_table finished): {out.strip()[-1500:]}"
            )
        if _is_validate_error(out) or rc != 0:
            log(
                f"PIN CN{node_id} try {attempt} failed: {out.strip()[:300]}"
            )
            if attempt == PIN_RETRY_VALIDATE:
                break
            time.sleep(PIN_RETRY_SLEEP_S)
            continue
    raise RuntimeError(
        f"ADMIN EXECUTE ON {node_id} failed after {PIN_RETRY_VALIDATE} tries: "
        f"{last.strip()[-1500:]}"
    )


def pin_all_cns(mysql: MysqlClient, script):
    ids = mysql.alive_rows()
    if not ids:
        raise RuntimeError("no alive CNs to pin")
    t0 = time.time()

    def _one(nid):
        admin_execute_on(mysql, nid, script)
        return nid

    with ThreadPoolExecutor(max_workers=len(ids)) as pool:
        futs = {pool.submit(_one, nid): nid for nid in ids}
        for fut in as_completed(futs):
            nid = futs[fut]
            fut.result()
            log(f"PIN CN{nid}: {time.time() - t0:.0f}s ok=1")
    log(f"pinned {len(ids)} CNs in {time.time() - t0:.1f}s")


def unpin_all_cns(mysql: MysqlClient, script):
    ids = mysql.alive_rows()

    def _one(nid):
        admin_execute_on(mysql, nid, script, timeout=120)

    with ThreadPoolExecutor(max_workers=max(len(ids), 1)) as pool:
        list(pool.map(_one, ids))


def _query_failed(rc, out):
    if rc != 0:
        return True
    if not out.strip():
        return True
    first = out.splitlines()[0] if out.splitlines() else ""
    return "ERROR" in first


def _write_result(benchmark_dir, qnum, mysql_out):
    qdir = os.path.join(benchmark_dir, "starrocks", f"q{qnum}")
    os.makedirs(qdir, exist_ok=True)
    lines = mysql_out.splitlines()
    # mysql --batch: first line is header
    body = lines[1:] if len(lines) > 1 else []
    with open(os.path.join(qdir, "result.txt"), "w") as f:
        f.write("\n".join(body))
        if body:
            f.write("\n")
    return max(len(body), 0)


def _record(writer, qnum, it, runtime_s):
    writer.writerow(["starrocks", f"q{qnum}", it, f"{runtime_s:.6f}"])
    log(f"[starrocks] q{qnum} iter{it}: {runtime_s:.4f}s")


def run_one_query(mysql, sql, timeout):
    t0 = time.perf_counter()
    rc, out = mysql.run(sql, timeout=timeout)
    elapsed = time.perf_counter() - t0
    return rc, out, elapsed


def run_grouped(
    mysql,
    cluster,
    queries,
    iterations,
    writer,
    *,
    benchmark_dir,
    data_dir,
    scale_factor,
    pin,
    pin_after_iteration,
    query_timeout,
):
    pin_enabled = pin != "none"
    for qnum in queries:
        sql = load_query_sql(qnum, data_dir, scale_factor)
        pinned = False
        try:
            for it in range(iterations):
                if pin_enabled and not pinned and it >= pin_after_iteration:
                    log(f"  Pinning tables for q{qnum} (from iter{it})")
                    pin_all_cns(
                        mysql, emit_admin_pin(qnum, data_dir, tier=pin)
                    )
                    pinned = True
                log(f"--- q{qnum} iter{it} engine=starrocks ---")
                rc, out, elapsed = run_one_query(mysql, sql, query_timeout)
                if _query_failed(rc, out):
                    log(f"  q{qnum} iter{it} REFUSED: {out.strip()[:500]}")
                    cluster.restart()
                    break
                rows = _write_result(benchmark_dir, qnum, out)
                _record(writer, qnum, it, elapsed)
                log(f"  q{qnum} fetched {rows} rows in {elapsed:.4f}s")
        finally:
            if pinned:
                log(f"  Unpinning tables for q{qnum}")
                try:
                    unpin_all_cns(mysql, emit_admin_unpin(qnum))
                except Exception as e:
                    log(f"  unpin failed (non-fatal): {e}")


def run_sequential(
    mysql,
    cluster,
    queries,
    iterations,
    writer,
    *,
    benchmark_dir,
    data_dir,
    scale_factor,
    pin,
    pin_after_iteration,
    query_timeout,
):
    pin_enabled = pin != "none"
    pinned = False
    try:
        for it in range(iterations):
            if pin_enabled and not pinned and it >= pin_after_iteration:
                log(f"  Pinning all tables (from pass {it})")
                pin_all_cns(
                    mysql, emit_admin_pin_all(data_dir, tier=pin)
                )
                pinned = True
            for qnum in queries:
                sql = load_query_sql(qnum, data_dir, scale_factor)
                log(f"--- q{qnum} iter{it} engine=starrocks ---")
                rc, out, elapsed = run_one_query(mysql, sql, query_timeout)
                if _query_failed(rc, out):
                    log(f"  q{qnum} iter{it} REFUSED: {out.strip()[:500]}")
                    cluster.restart()
                    pinned = False
                    continue
                rows = _write_result(benchmark_dir, qnum, out)
                _record(writer, qnum, it, elapsed)
                log(f"  q{qnum} fetched {rows} rows in {elapsed:.4f}s")
    finally:
        if pinned:
            log("  Unpinning all tables")
            try:
                unpin_all_cns(mysql, emit_admin_unpin_all())
            except Exception as e:
                log(f"  unpin failed (non-fatal): {e}")


def _duckdb_row_count(path):
    if not os.path.isfile(path):
        return None
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_validation_csv(benchmark_dir, queries, duckdb_results_dir):
    out = os.path.join(benchmark_dir, "validation.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "starrocks_rows", "duckdb_rows", "status"])
        for qnum in queries:
            sr = os.path.join(benchmark_dir, "starrocks", f"q{qnum}", "result.txt")
            dr = os.path.join(duckdb_results_dir, f"q{qnum}", "result.txt")
            sc = _duckdb_row_count(sr)
            dc = _duckdb_row_count(dr)
            if sc is None or dc is None:
                status = "missing"
            elif sc == dc:
                status = "row_count_match"
            else:
                status = "row_count_mismatch"
            w.writerow([f"q{qnum}", sc if sc is not None else "", dc if dc is not None else "", status])
            log(f"validation q{qnum}: {status} (starrocks={sc} duckdb={dc})")
    log(
        "validation.csv compares row counts only. StarRocks FILES() SQL is not "
        "queries.py (q11 fraction, q08/q09 FROM order)."
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="TPC-H performance test against StarRocks FE + Sirius CNs"
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        help="TPC-H parquet directory (<table>/*.parquet)",
    )
    p.add_argument(
        "--data-source",
        choices=("parquet", "duckdb"),
        default="parquet",
        help="Must be parquet (StarRocks CNs scan FILES()). duckdb is rejected.",
    )
    p.add_argument(
        "--mode",
        choices=("grouped", "sequential", "isolated", "nsys-profile"),
        default="grouped",
        help="grouped: per-query iterations; sequential: union-pin then round-robin",
    )
    p.add_argument("--iterations", type=int, default=1)
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--engine",
        choices=("gpu", "cpu", "both"),
        default="gpu",
        help="StarRocks CNs are GPU-only; cpu/both are rejected",
    )
    p.add_argument(
        "--validation",
        action="store_true",
        help="Row-count compare vs --duckdb-results (SQL is not queries.py)",
    )
    p.add_argument("--queries", type=str, default=None)
    p.add_argument(
        "--config",
        type=str,
        default="",
        help="Unused for launch (CNs get generated YAML); copied into the run dir if it exists",
    )
    p.add_argument("--pin", choices=("none", "gpu", "host"), default="none")
    p.add_argument("--pin-after-iteration", type=int, default=0)
    p.add_argument(
        "--pin-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--compression-plan-dir", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument(
        "--duckdb-profiling",
        action="store_true",
        help="Rejected: no in-process DuckDB on this path",
    )
    p.add_argument(
        "--query-timeout",
        type=int,
        default=600,
        help="Per-query mysql timeout in seconds (default 600)",
    )
    p.add_argument("--duckdb-results", type=str, default=None)
    p.add_argument("--gpus", type=int, default=4, help="CNs / GPUs per host")
    p.add_argument(
        "--hosts",
        type=str,
        default="127.0.0.1",
        help="Comma list. First host runs the FE. Default 127.0.0.1",
    )
    p.add_argument("--gpu-mem", type=str, default="110GiB")
    p.add_argument("--staging", type=str, default="16GiB")
    p.add_argument("--host-mem", type=str, default="200GiB")
    p.add_argument("--pipeline-dop", type=int, default=None)
    p.add_argument("--fe-host", type=str, default=None)
    p.add_argument("--fe-port", type=int, default=9030)
    p.add_argument(
        "--keep-cluster",
        action="store_true",
        help="Leave FE+CNs running after the sweep",
    )
    p.add_argument("--scale-factor", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.engine != "gpu":
        raise SystemExit("--engine must be gpu (StarRocks CNs have no CPU path)")
    if args.data_source != "parquet":
        raise SystemExit("--data-source must be parquet (FILES() scans)")
    if args.mode in ("isolated", "nsys-profile"):
        raise SystemExit(f"--mode {args.mode} is not supported on the StarRocks path")
    if args.duckdb_profiling:
        raise SystemExit("--duckdb-profiling is standalone-DuckDB only")
    if not os.path.isdir(args.input):
        raise SystemExit(f"--input is not a parquet directory: {args.input}")
    if args.pin_compression and args.pin == "none":
        raise SystemExit("--pin-compression needs --pin gpu|host")
    if args.validation and not args.duckdb_results:
        raise SystemExit("--validation requires --duckdb-results")
    if args.gpus < 1:
        raise SystemExit("--gpus must be >= 1")

    hosts = parse_hosts(args.hosts)
    fe_host = args.fe_host or hosts[0]
    queries = parse_query_spec(args.queries)
    scale_factor = infer_scale_factor(args.input, args.scale_factor)
    plan_dir = args.compression_plan_dir
    if args.pin_compression:
        plan_dir = os.path.abspath(
            plan_dir
            or os.path.join(
                REPO_ROOT,
                "src",
                "compression",
                "simpatico_codegen",
                "plans",
                "tpch_sf1000",
            )
        )
        if not os.path.isdir(plan_dir):
            raise SystemExit(f"--compression-plan-dir not found: {plan_dir}")

    duckdb_results_dir = None
    if args.duckdb_results:
        candidate = args.duckdb_results
        if os.path.isdir(os.path.join(candidate, "duckdb")):
            candidate = os.path.join(candidate, "duckdb")
        if not os.path.isdir(candidate):
            raise SystemExit(f"--duckdb-results directory not found: {candidate}")
        duckdb_results_dir = candidate

    output_root = args.output or DEFAULT_OUTPUT_ROOT
    os.makedirs(output_root, exist_ok=True)
    config_dir = os.path.join(PINNED_DIR, "generated")
    benchmark_dir, runtime_csv, log_dir = setup_benchmark_dir(
        output_root,
        args.mode,
        args.iterations,
        queries,
        args.pin,
        name=args.name,
        pin_compression=args.pin_compression,
        compression_plan_dir=plan_dir,
        gpus=args.gpus,
        hosts=hosts,
        scale_factor=scale_factor,
        config_path=args.config or None,
        duckdb_results_source=duckdb_results_dir,
    )
    log(f"Benchmark dir: {benchmark_dir}")
    log(f"Runtime CSV:   {runtime_csv}")

    mysql = MysqlClient(fe_host, args.fe_port)
    cluster = Cluster(
        mysql,
        gpus=args.gpus,
        hosts=hosts,
        gpu_mem=args.gpu_mem,
        staging=args.staging,
        host_mem=args.host_mem,
        pipeline_dop=args.pipeline_dop,
        pin_compression=args.pin_compression,
        compression_plan_dir=plan_dir,
        config_dir=config_dir,
        scale_factor=scale_factor,
        query_timeout=args.query_timeout,
        pin_enabled=args.pin != "none",
    )

    try:
        cluster.launch()
        with open(runtime_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["engine", "query", "iteration", "runtime_s"])
            common = dict(
                benchmark_dir=benchmark_dir,
                data_dir=os.path.abspath(args.input),
                scale_factor=scale_factor,
                pin=args.pin,
                pin_after_iteration=args.pin_after_iteration,
                query_timeout=args.query_timeout,
            )
            if args.mode == "grouped":
                run_grouped(
                    mysql,
                    cluster,
                    queries,
                    args.iterations,
                    writer,
                    **common,
                )
            else:
                run_sequential(
                    mysql,
                    cluster,
                    queries,
                    args.iterations,
                    writer,
                    **common,
                )
            f.flush()
        yaml0 = os.path.join(config_dir, "cn0.yaml")
        if os.path.isfile(yaml0):
            shutil.copy2(yaml0, os.path.join(benchmark_dir, "config.yml"))
        if args.validation or duckdb_results_dir:
            write_validation_csv(benchmark_dir, queries, duckdb_results_dir)
        log("Benchmark run complete")
        print(f"\nResults: {benchmark_dir}")
    finally:
        if not args.keep_cluster:
            cluster.teardown()
        else:
            log("--keep-cluster: leaving FE+CNs running")


if __name__ == "__main__":
    main()
