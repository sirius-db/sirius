#!/usr/bin/env python3
"""Start the current FE and one or two engine-linked CNs on GB10's single GPU."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SESSION = (
    "SET single_node_exec_plan=true; SET pipeline_dop=1; SET new_planner_agg_stage=1; "
)


def cn_ports(count):
    if count == 1:
        return [dict(heartbeat=9050, thrift=9060, http=8040, brpc=8060, starlet=9070)]
    return [
        dict(
            heartbeat=base,
            thrift=base + 1,
            brpc=base + 2,
            http=base + 3,
            starlet=base + 4,
        )
        for base in (9100, 9110)
    ]


def alive_nodes(status):
    rows = [line.split("\t") for line in status.strip().splitlines()]
    if not rows:
        return []
    columns = {name.lower(): index for index, name in enumerate(rows[0])}
    if "alive" not in columns:
        raise RuntimeError("Missing Alive column in FE topology response")
    return [
        dict(zip(columns, row))
        for row in rows[1:]
        if row[columns["alive"]].lower() == "true"
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Check a FILES query against DuckDB, then stop",
    )
    parser.add_argument("--run-dir", type=Path, default=ROOT / "build/local-gb10")
    parser.add_argument("--cn-count", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--cn-binary", type=Path, help="Exact CN artifact for an isolated A/B run"
    )
    parser.add_argument(
        "--engine-library-dir",
        type=Path,
        help="Directory containing the matching libsirius.so",
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=ROOT,
        help="Sirius checkout containing the built engine and compute node",
    )
    parser.add_argument(
        "--cn-engine-dir",
        type=Path,
        help="Engine working directory; with two CNs, each uses its own cn0/cn1 subdirectory",
    )
    parser.add_argument(
        "--sirius-config", type=Path, default=Path(__file__).with_name("sirius.yaml")
    )
    parser.add_argument(
        "--timeout", type=int, default=180, help="Startup timeout in seconds"
    )
    args = parser.parse_args()
    ports = cn_ports(args.cn_count)
    engine_root = args.engine_root.resolve()
    starrocks = engine_root / "experimental/starrocks"
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "launcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    package = starrocks / "starrocks/output/fe"
    cn = (args.cn_binary or starrocks / "target/release/sirius-starrocks-cn").resolve()
    mysql = Path(
        os.environ.get("MYSQL_BIN", str(starrocks / ".pixi/envs/default/bin/mysql"))
    )
    java_home = Path(
        os.environ.get("GB10_JAVA_HOME", str(starrocks / ".pixi/envs/fe/lib/jvm"))
    )
    for path in (
        package / "bin/start_fe.sh",
        cn,
        mysql,
        java_home / "bin/java",
        args.sirius_config,
    ):
        if not path.exists():
            parser.error(f"Missing {path}; see scripts/local-gb10/README.md")
    for port in (
        8030,
        9010,
        9020,
        9030,
        *(port for node in ports for port in node.values()),
    ):
        with socket.socket() as check:
            check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            check.bind(("0.0.0.0", port))

    fe = run / "fe"
    for directory in ("bin", "conf"):
        shutil.copytree(package / directory, fe / directory, dirs_exist_ok=True)
    lib = fe / "lib"
    if not lib.exists():
        lib.symlink_to(package / "lib", target_is_directory=True)
    for directory in ("log", "meta"):
        (fe / directory).mkdir(exist_ok=True)
    conf = fe / "conf/fe.conf"
    text = (starrocks / "conf/fe.conf").read_text()
    text = re.sub(r"-Xmx\S+", "-Xmx2048m", text)
    text += f"\nmeta_dir = {fe / 'meta'}\nPID_DIR = {fe / 'bin'}\n"
    conf.write_text(text)

    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            (
                str(
                    (
                        args.engine_library_dir
                        or engine_root / "build/release/extension/sirius"
                    ).resolve()
                ),
                env.get("LD_LIBRARY_PATH"),
                str(engine_root / ".pixi/envs/default/lib"),
            ),
        )
    )
    env["SIRIUS_LOG_BACKEND"] = "spdlog"
    env["SIRIUS_LOG_DIR"] = str(run / "engine-log")
    env.setdefault("RUST_LOG", "info")
    env.pop("SIRIUS_DISABLE", None)
    env.pop("SIRIUS_CN_TRANSLATE_ONLY", None)
    if args.cn_count > 1:
        env.setdefault("SIRIUS_EXCHANGE_STAGING_BYTES", "2GiB")
        env.setdefault("SIRIUS_CN_NIXL_WARMUP", "on")
        env.setdefault("SIRIUS_CN_NIXL_WARMUP_EXPECT_PEERS", str(args.cn_count - 1))
        env.setdefault(
            "SIRIUS_CN_NIXL_WARMUP_PEERS",
            ",".join(f"127.0.0.1:{node['brpc']}" for node in ports),
        )
        env.setdefault("SIRIUS_CN_NIXL_WARMUP_TIMEOUT_SECS", str(args.timeout))
        env.setdefault("SIRIUS_CN_RPC_TIMEOUT_SECS", "600")
        env.setdefault("SIRIUS_QUERY_WATCHDOG_SECS", "600")

    client = [
        str(mysql),
        "--no-defaults",
        "--protocol=TCP",
        "--host=127.0.0.1",
        "--port=9030",
        "--user=root",
        "--connect-timeout=2",
        "--ssl-mode=DISABLED",
        "--batch",
        "--raw",
    ]

    def query(sql, headers=True):
        command = (
            client + ([] if headers else ["--skip-column-names"]) + ["--execute", sql]
        )
        return subprocess.run(
            command, env=env, text=True, capture_output=True, timeout=60, check=True
        ).stdout

    children = []
    logs = []

    def stop_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        (run / "launcher.pid").write_text(f"{os.getpid()}\n")
        commands = [("fe", [str(fe / "bin/start_fe.sh"), "--logconsole"], env, run)]
        for index, node_ports in enumerate(ports):
            name = "cn" if args.cn_count == 1 else f"cn{index}"
            command = [
                str(cn),
                "--gpu-device",
                "0",
                "--bind-host",
                "127.0.0.1",
                "--advertise-host",
                "127.0.0.1",
                "--fe-host",
                "127.0.0.1",
                "--sirius-config",
                str(args.sirius_config.resolve()),
            ]
            for label, port in node_ports.items():
                command.extend([f"--{label}-port", str(port)])
            engine_dir = run
            if args.cn_engine_dir or args.cn_count > 1:
                engine_dir = (args.cn_engine_dir or run / "engine").resolve()
                if args.cn_count > 1:
                    engine_dir /= name
                engine_dir.mkdir(parents=True, exist_ok=True)
                command.extend(["--engine-dir", str(engine_dir)])
            (engine_dir / "spill").mkdir(exist_ok=True)
            cn_env = env | {"SIRIUS_LOG_DIR": str(run / f"{name}-engine-log")}
            fragment_dir = run / f"{name}-fragments"
            fragment_dir.mkdir(exist_ok=True)
            cn_env["SIRIUS_CN_DUMP_FRAGMENTS"] = str(fragment_dir)
            commands.append((name, command, cn_env, engine_dir))
        (run / "launch.json").write_text(
            json.dumps(
                {
                    "cn_count": args.cn_count,
                    "gpu": 0,
                    "engine_root": str(engine_root),
                    "sirius_config": str(args.sirius_config.resolve()),
                    "commands": {name: command for name, command, _, _ in commands},
                    "working_directories": {
                        name: str(cwd) for name, _, _, cwd in commands
                    },
                    "transport_environment": {
                        key: value
                        for key, value in env.items()
                        if key.startswith(
                            (
                                "NIXL_",
                                "UCX_",
                                "SIRIUS_EXCHANGE_",
                                "SIRIUS_CN_",
                                "SIRIUS_QUERY_",
                            )
                        )
                    },
                },
                indent=2,
            )
            + "\n"
        )
        for name, command, child_env, cwd in commands:
            output = (run / f"{name}.log").open("w")
            logs.append(output)
            children.append(
                subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=child_env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
            )
        (run / "processes.json").write_text(
            json.dumps(
                {name: child.pid for (name, _, _, _), child in zip(commands, children)},
                indent=2,
            )
            + "\n"
        )
        (run / "process-start-ticks.json").write_text(
            json.dumps(
                {
                    str(child.pid): Path(f"/proc/{child.pid}/stat")
                    .read_text()
                    .rsplit(")", 1)[1]
                    .split()[19]
                    for child in children
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Starting FE and {args.cn_count} Sirius CN(s) on GPU 0; logs: {run}",
            flush=True,
        )
        deadline = time.monotonic() + args.timeout
        last_error = "FE/CN not ready"
        while time.monotonic() < deadline:
            for child in children:
                if child.poll() is not None:
                    raise RuntimeError(
                        f"Stack process {child.pid} exited ({child.returncode}); inspect {run}"
                    )
            try:
                status = query("SHOW COMPUTE NODES;")
                alive = alive_nodes(status)
                backends = alive_nodes(query("SHOW BACKENDS;"))
                if len(alive) > args.cn_count or backends:
                    raise RuntimeError(
                        f"Unexpected topology: {len(alive)} alive CNs, {len(backends)} alive BEs"
                    )
                expected = {str(node["heartbeat"]) for node in ports}
                actual = {node["heartbeatport"] for node in alive}
                if len(alive) == args.cn_count and actual == expected:
                    # Heartbeats alone do not prove remote exchange sessions are ready.
                    warm = (
                        args.cn_count == 1
                        or env.get("SIRIUS_CN_NIXL_WARMUP") == "off"
                        or all(
                            "nixl session warmup complete: every peer session is established"
                            in (run / f"cn{index}.log").read_text(errors="replace")
                            for index in range(args.cn_count)
                        )
                    )
                    if warm:
                        (run / "ready-compute-nodes.tsv").write_text(status)
                        print(status, end="", flush=True)
                        break
                    last_error = (
                        "CNs alive; waiting for NIXL peer sessions in both directions"
                    )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                last_error = getattr(error, "stderr", None) or str(error)
            time.sleep(1)
        else:
            raise RuntimeError(f"Startup timed out: {last_error}")

        if args.smoke:
            parquet = run / "tiny.parquet"
            fixture = """
import duckdb, json, sys
connection = duckdb.connect()
connection.execute("COPY (SELECT i::BIGINT AS value FROM range(1, 1001) t(i)) TO ? (FORMAT PARQUET)", [sys.argv[1]])
rows = connection.execute("SELECT value + 1 FROM read_parquet(?) WHERE value <= 10 ORDER BY 1", [sys.argv[1]]).fetchall()
print(json.dumps([row[0] for row in rows]))
"""
            oracle_env = env | {"SIRIUS_DISABLE": "1"}
            oracle = json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", fixture, str(parquet)],
                    env=oracle_env,
                    text=True,
                )
            )
            path = parquet.as_uri().replace("'", "''")
            sql = (
                SESSION
                + f"SELECT value + 1 AS shifted FROM FILES('path'='{path}', 'format'='parquet') WHERE value <= 10;"
            )
            (run / "smoke.sql").write_text(sql + "\n")
            (run / "smoke-plan.txt").write_text(
                query(SESSION + "EXPLAIN " + sql[len(SESSION) :])
            )
            actual = sorted(
                int(value) for value in query(sql, headers=False).splitlines()
            )
            (run / "smoke-result.txt").write_text(
                f"DuckDB: {oracle}\nStarRocks + Sirius: {actual}\n"
            )
            if actual != oracle or oracle != list(range(2, 12)):
                raise RuntimeError(
                    f"Unexpected FILES result: Sirius={actual!r}, DuckDB={oracle!r}"
                )
            print(
                f"PASS: StarRocks FILES -> Sirius filter/projection matches DuckDB ({len(actual)} rows)",
                flush=True,
            )
        else:
            print(
                "Ready on MySQL 127.0.0.1:9030. Ctrl-C stops this FE and CN.",
                flush=True,
            )
            while all(child.poll() is None for child in children):
                time.sleep(1)
            raise RuntimeError(f"A stack process exited; inspect {run}")
    except KeyboardInterrupt:
        print("Stopping FE and CN.", flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for output in logs:
            output.close()
        (run / "launcher.pid").unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        sys.exit(getattr(error, "stderr", None) or str(error))
