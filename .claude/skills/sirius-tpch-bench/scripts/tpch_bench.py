#!/usr/bin/env python3
"""Plan, build, smoke-test and benchmark TPC-H on Distributed Sirius through its gateway.

One run folder (--out) holds everything: plan.json, the resolved Sirius config, the generated
scripts and SQL, the client output and the reports. Steps, in order:

  configs               list the Sirius config presets shipped with this skill
  plan ...              record and validate the inputs (asks for missing ones at a terminal)
  show      --out DIR   print the plan and the Sirius configuration that will be used
  preflight --out DIR   check the tools, the nodes, the dataset and the build
  build     --out DIR   clone the branch if needed and build it
  smoke     --out DIR   a few queries once each, with pass/fail checks
  full      --out DIR   the benchmark queries, --runs times each
  report    --out DIR   print the results again; also works while a run is in progress

Steps that use nodes ask for confirmation at a terminal. Without a terminal they need --yes,
which an agent passes only after the user has approved that step. Standard library only.
"""

import argparse
import csv
import json
import re
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from decimal import Context, Decimal, InvalidOperation
from pathlib import Path

SKILL = Path(__file__).resolve().parent.parent
PRESETS = SKILL / "configs"
TABLES = "customer lineitem nation orders part partsupp region supplier".split()
DEFAULT_REPO = "https://github.com/sirius-db/sirius.git"
# The submodules the branch's build instructions name; those a branch does not have are skipped.
DEFAULT_SUBMODULES = (
    "duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools"
)
QUERIES_SUBDIR = "test/tpch_performance/tpch_queries/orig"
PIXI_PATH = 'export PATH="${PIXI_HOME:-$HOME/.pixi}/bin:$PATH"'
q = shlex.quote


def say(*parts):
    print("[tpch-bench]", *parts, flush=True)


def die(message, code=1):
    print(f"[tpch-bench] ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(code)


def confirm(args, what):
    """Every step that uses nodes passes through here."""
    if getattr(args, "yes", False):
        return
    if not sys.stdin.isatty():
        die(
            f"{what}: this needs the user's approval. Show them the output of `show`, "
            "and rerun with --yes once they agree.",
            2,
        )
    if input(f"[tpch-bench] {what}. Proceed? [y/N] ").strip().lower() not in (
        "y",
        "yes",
    ):
        die("stopped: not confirmed", 2)


def load(out):
    path = Path(out) / "plan.json"
    if not path.is_file():
        die(f"no plan in {out}: run `plan` first")
    return json.loads(path.read_text())


def save(plan):
    (Path(plan["out"]) / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


# ---------------------------------------------------------------------------- inputs


def parse_queries(text):
    """'1-22' or '1,6,12' -> [1, 6, 12]."""
    numbers = []
    for part in str(text).split(","):
        lo, _, hi = part.strip().partition("-")
        numbers += range(int(lo), int(hi or lo) + 1)
    if not numbers or min(numbers) < 1 or max(numbers) > 22:
        raise ValueError("queries are numbers from 1 to 22, such as 1-22 or 1,6,12")
    return numbers


def check_sf(text):
    try:
        return None if Decimal(text) > 0 else "the scale factor must be positive"
    except InvalidOperation:
        return "the scale factor must be a number, such as 100 or 10000"


def check_abs(text):
    return None if text.startswith("/") else "give an absolute path"


def check_config(text):
    try:
        read_config(text)
    except ValueError as error:
        return str(error)
    return None


def read_config(choice):
    """A preset name, a path to a YAML file, or 'engine-defaults'. Returns (label, text, env)."""
    if choice == "engine-defaults":
        return "engine defaults (no config file)", None, {}
    preset = PRESETS / f"{choice}.yaml"
    path = preset if preset.is_file() else Path(choice)
    if not path.is_file():
        names = ", ".join(sorted(p.stem for p in PRESETS.glob("*.yaml")))
        raise ValueError(
            f"'{choice}' is not a preset ({names}), a readable YAML file or 'engine-defaults'"
        )
    env = {}
    env_file = path.with_suffix(".env")
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip()
    label = f"preset {choice}" if path == preset else f"file {path}"
    return label, path.read_text(), env


def cmd_configs(_args):
    print("engine-defaults\n    No config file: every engine key at its default.")
    for path in sorted(PRESETS.glob("*.yaml")):
        lines = [
            l[1:].strip() for l in path.read_text().splitlines() if l.startswith("#")
        ]
        print(path.stem + "".join(f"\n    {line}" for line in lines))
    print(
        "Or give the path of your own YAML file. '@SPILL@' in it becomes --spill-dir."
    )


def cmd_plan(args):
    missing = []

    def need(name, label, check=None, default=None):
        value = getattr(args, name)
        if value in (None, "") and default is not None:
            value = default
        if value in (None, ""):
            if not sys.stdin.isatty():
                missing.append("--" + name.replace("_", "-"))
                return None
            while value in (None, ""):
                value = input(f"[tpch-bench] {label}: ").strip()
        while check and check(str(value)):
            if not sys.stdin.isatty():
                die(f"--{name.replace('_', '-')} {value}: {check(str(value))}")
            print(f"  {check(str(value))}")
            value = input(f"[tpch-bench] {label}: ").strip()
        return value

    branch = need("branch", "Branch to test")
    checkout = need(
        "checkout", "Checkout folder, existing or to create (absolute path)", check_abs
    )
    data = need(
        "data",
        "Dataset folder, one subfolder of Parquet files per table (absolute path)",
        check_abs,
    )
    sf = need("sf", "Scale factor of that dataset", check_sf)
    mode = need(
        "mode",
        "Execution mode, 'single' (one node, all its GPUs) or 'multi' (several nodes)",
        lambda v: None if v in ("single", "multi") else "answer single or multi",
    )
    launcher = args.launcher
    if launcher == "auto":
        launcher = "slurm" if shutil.which("sbatch") else "local"
    nodes = [n for n in (args.nodes or "").split(",") if n]
    count = args.num_nodes or len(nodes) or (1 if mode == "single" else None)
    if mode == "multi" and not count:
        answer = need(
            "nodes",
            "Nodes, comma-separated"
            + (
                ", or a number of nodes for Slurm to pick"
                if launcher == "slurm"
                else ""
            ),
        )
        if answer and answer.isdigit():
            count = int(answer)
        elif answer:
            nodes = [n for n in answer.split(",") if n]
            count = len(nodes)
    config = need(
        "config",
        "Sirius config: a preset from `configs`, a YAML path, or engine-defaults",
        check_config,
    )
    out = need(
        "out",
        "Run folder, new, on a file system the nodes also mount (absolute path)",
        check_abs,
    )
    if missing:
        die(
            "missing "
            + " ".join(missing)
            + ". Ask the user for each; do not guess them.",
            2,
        )

    if mode == "single" and count != 1:
        die("single mode is one node; use --mode multi for several")
    if mode == "multi" and count < 2:
        die("multi mode needs at least two nodes")
    if launcher == "local" and mode == "multi" and len(nodes) != count:
        die("without Slurm, name the machines with --nodes (the first is this one)")
    label, text, env = read_config(config)
    spill = args.spill_dir
    if text and "@SPILL@" in text:
        if not spill and sys.stdin.isatty():
            spill = input(
                "[tpch-bench] Spill folder, on a disk with free space (absolute path): "
            ).strip()
        if not spill:
            die("this config has @SPILL@: give --spill-dir", 2)
        text = text.replace("@SPILL@", spill)
    for item in args.env or []:
        key, sep, value = item.partition("=")
        if not sep:
            die(f"--env {item}: expected KEY=VALUE")
        env[key] = value
    unset = [key for key, value in env.items() if value == "?"]
    if unset:
        die(
            "this config needs " + ", ".join(f"--env {key}=<value>" for key in unset), 2
        )
    try:
        full, smoke = parse_queries(args.queries), parse_queries(args.smoke_queries)
    except ValueError as error:
        die(str(error))

    out_dir = Path(out)
    if (out_dir / "plan.json").exists() and not args.replace:
        die(f"{out} already holds a plan: pick a new run folder, or pass --replace")
    out_dir.mkdir(parents=True, exist_ok=True)
    if text:
        (out_dir / "config.yaml").write_text(text)
    plan = {
        "branch": branch,
        "repo": args.repo,
        "checkout": checkout.rstrip("/"),
        "data": data.rstrip("/"),
        "sf": str(sf),
        "mode": mode,
        "launcher": launcher,
        "nodes": nodes,
        "num_nodes": count,
        "partition": args.partition,
        "gpus_per_node": args.gpus_per_node,
        "sbatch_args": args.sbatch_args,
        "time_limit": args.time_limit,
        "client_on": args.client_on,
        "config": config,
        "config_label": label,
        "config_file": str(out_dir / "config.yaml") if text else None,
        "env": env,
        "statement_timeout": args.statement_timeout,
        "queries_dir": args.queries_dir,
        "setup_sql": args.setup_sql,
        "queries": full,
        "runs": args.runs,
        "smoke_queries": smoke,
        "big_dir": args.big_dir,
        "submodules": args.submodules,
        "spill_dir": spill,
        "out": str(out_dir),
    }
    save(plan)
    show(plan)
    if sys.stdin.isatty() and not args.yes:
        if input(
            "[tpch-bench] Are these inputs correct? [y/N] "
        ).strip().lower() not in ("y", "yes"):
            (out_dir / "plan.json").unlink()
            die("plan discarded: run `plan` again", 2)
    say(f"plan saved in {out_dir}/plan.json. Next: preflight --out {out_dir}")


def where(plan):
    names = ",".join(plan["nodes"]) or (
        "nodes chosen by Slurm" if plan["launcher"] == "slurm" else "this machine"
    )
    return f"{plan['num_nodes']} node(s): {names}"


def show(plan):
    ranks = plan.get("ranks")
    rows = [
        ("branch", plan["branch"]),
        ("repository", plan["repo"]),
        ("checkout", plan["checkout"]),
        ("dataset", plan["data"]),
        ("scale factor", plan["sf"]),
        ("mode", f"{plan['mode']}, {where(plan)}"),
        (
            "launcher",
            plan["launcher"]
            + (f", partition {plan['partition']}" if plan["partition"] else ""),
        ),
        (
            "ranks",
            (
                f"{plan['gpus_per_node']} per node, {ranks} in all"
                if ranks
                else "one per GPU; counted in preflight"
            ),
        ),
        (
            "smoke test",
            f"queries {','.join(map(str, plan['smoke_queries']))}, once each",
        ),
        (
            "full benchmark",
            f"{len(plan['queries'])} queries, {plan['runs']} run(s) each",
        ),
        (
            "statement timeout",
            (
                f"{plan['statement_timeout']} s"
                if plan["statement_timeout"]
                else "the gateway's default"
            ),
        ),
        ("build", plan.get("build", {}).get("commit", "not checked yet")),
        ("run folder", plan["out"]),
    ]
    print("Plan")
    for key, value in rows:
        print(f"  {key:18s} {value}")
    print(f"Sirius configuration: {plan['config_label']}")
    if plan["config_file"]:
        print(f"  SIRIUS_CONFIG_FILE={plan['config_file']}")
        print("  " + "-" * 60)
        for line in Path(plan["config_file"]).read_text().rstrip().splitlines():
            print("  " + line)
        print("  " + "-" * 60)
    for key, value in sorted(plan["env"].items()):
        print(f"  environment: {key}={value}")
    print("  Keys the config does not set keep the engine's defaults.")


def cmd_show(args):
    show(load(args.out))


# ---------------------------------------------------------------------------- nodes


def node_sh(plan, script, node=None):
    """Runs a shell snippet where the nodes' file systems are visible."""
    if plan["launcher"] == "slurm":
        cmd = ["srun", "-N1", "-n1", "--time=5"]
        if plan["partition"]:
            cmd += ["-p", plan["partition"]]
        if node:
            cmd += ["-w", node]
        cmd += ["bash", "-c", script]
    elif node and node != plan["nodes"][0]:
        cmd = ["ssh", node, "bash -c " + q(script)]
    else:
        cmd = ["bash", "-c", script]
    return subprocess.run(cmd, capture_output=True, text=True)


def node_facts(plan, node=None):
    c, d, o = q(plan["checkout"]), q(plan["data"]), q(plan["out"])
    script = "\n".join(
        [
            "echo HOST=$(hostname)",
            *[
                f"echo TABLE_{t}=$(ls {d}/{t}/*.parquet 2> /dev/null | wc -l)"
                for t in TABLES
            ],
            f"[ -e {c}/build/release/libsirius.so ] && [ -x {c}/gateway/build/release/duckdb ] && echo BUILD=yes || echo BUILD=no",
            f"[ -d {c}/.git ] && echo \"COMMIT=$(git -C {c} log -1 --format='%h %s')\" && "
            f"echo BRANCH=$(git -C {c} rev-parse --abbrev-ref HEAD) && "
            f"echo DIRTY=$(git -C {c} status --porcelain --untracked-files=no | wc -l)",
            f"[ -d {o} ] && echo OUT=yes || echo OUT=no",
            "echo GPUS=$(nvidia-smi -L 2> /dev/null | wc -l)",
            "echo CPUS=$(nproc 2> /dev/null)",
        ]
    )
    result = node_sh(plan, script, node)
    facts = dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )
    if result.returncode != 0 or "HOST" not in facts:
        die(
            f"could not run a check on {node or 'a node'}: {result.stderr.strip()[-400:]}"
        )
    return facts


def slurm_gpus(plan, node):
    """GPUs per node from Slurm's own records: a step without a GPU may not see any."""
    if node:
        text = subprocess.run(
            ["scontrol", "show", "node", node], capture_output=True, text=True
        ).stdout
    else:
        cmd = ["sinfo", "-h", "-o", "%G"] + (
            ["-p", plan["partition"]] if plan["partition"] else []
        )
        text = subprocess.run(cmd, capture_output=True, text=True).stdout
    found = re.search(r"gpu(?::[^:,()\s]+)?:(\d+)", text)
    return int(found.group(1)) if found else None


def cmd_preflight(args):
    plan = load(args.out)
    slurm = plan["launcher"] == "slurm"
    confirm(
        args,
        f"Preflight runs short checks on {where(plan)}"
        + (" through srun" if slurm else ""),
    )
    problems = []
    tools = (
        ["git", "sbatch", "srun", "squeue", "scancel"]
        if slurm
        else ["git", "nvidia-smi", "ss"]
    )
    problems += [
        f"`{tool}` is not on PATH here" for tool in tools if not shutil.which(tool)
    ]
    if problems:
        die("; ".join(problems))
    gpus = set()
    for node in plan["nodes"] or [None]:
        facts = node_facts(plan, node)
        host = facts["HOST"]
        empty = [t for t in TABLES if facts.get(f"TABLE_{t}", "0") == "0"]
        if empty:
            problems.append(
                f"{host}: no Parquet files in {plan['data']}/<table>/ for {', '.join(empty)}"
            )
        if facts.get("OUT") != "yes":
            problems.append(
                f"{host}: the run folder {plan['out']} is not visible there"
            )
        count = plan["gpus_per_node"] or (
            slurm_gpus(plan, node) if slurm else int(facts.get("GPUS") or 0)
        )
        if not count:
            problems.append(
                f"{host}: GPU count unknown; pass --gpus-per-node to `plan`"
            )
        gpus.add(count)
        say(
            f"{host}: dataset {'incomplete' if empty else 'ok'}, {count or '?'} GPU(s), "
            f"build {'present at ' + facts.get('COMMIT', '?') if facts.get('BUILD') == 'yes' else 'absent'}"
        )
        if facts.get("BRANCH") and facts["BRANCH"] != plan["branch"]:
            say(
                f"  note: the checkout is on '{facts['BRANCH']}', the plan says '{plan['branch']}'"
            )
    if len(gpus) > 1:
        problems.append(f"the nodes have different GPU counts: {sorted(gpus)}")
    if problems:
        plan["preflight"] = "failed"
        save(plan)
        die("preflight failed:\n  " + "\n  ".join(problems))
    plan.update(
        preflight="ok",
        gpus_per_node=gpus.pop(),
        build_present=facts.get("BUILD") == "yes",
    )
    plan["ranks"] = plan["gpus_per_node"] * plan["num_nodes"]
    save(plan)
    say(
        f"preflight ok: {plan['ranks']} ranks ({plan['gpus_per_node']} per node). Next: build --out {plan['out']}"
    )


# ---------------------------------------------------------------------------- build


def cmd_build(args):
    plan = load(args.out)
    if plan.get("preflight") != "ok":
        die("run `preflight` first")
    out, c = Path(plan["out"]), q(plan["checkout"])
    facts = node_facts(plan, (plan["nodes"] or [None])[0])
    if (
        facts.get("BRANCH")
        and facts["BRANCH"] != plan["branch"]
        and not args.use_existing_checkout
    ):
        die(
            f"the checkout is on '{facts['BRANCH']}', not on '{plan['branch']}'. Switch it yourself, give "
            "another --checkout, or pass --use-existing-checkout to test it as it is."
        )
    if facts.get("BUILD") == "yes" and not args.rebuild:
        plan["build"] = {
            "commit": facts.get("COMMIT", "?"),
            "dirty_files": facts.get("DIRTY", "?"),
            "reused": True,
        }
        save(plan)
        say(
            f"reusing the build at {plan['checkout']}: {plan['build']['commit']} "
            f"({plan['build']['dirty_files']} modified tracked files). Pass --rebuild to build again."
        )
        return
    script = out / "build.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# Generated by tpch_bench.py: clone if absent, then build Sirius and the gateway.",
                "set -euo pipefail",
                PIXI_PATH,
                f"[ -d {c}/.git ] || git clone --branch {q(plan['branch'])} {q(plan['repo'])} {c}",
                f"cd {c}",
                "subs=",
                f"for s in {plan['submodules']}; do",
                '  git config -f .gitmodules --get-regexp \'^submodule\\..*\\.path$\' | grep -qx ".* $s" && subs="$subs $s"',
                "done",
                "git submodule update --init --depth=1 --jobs 5 $subs",
                "pixi run build-distributed",
                "ls -d build/release/libsirius.so gateway/build/release/duckdb",
                f"git log -1 --format='%h %s' > {q(str(out / 'BUILD_OK'))}",
                "",
            ]
        )
    )
    if args.dry_run:
        say(f"dry run: wrote {script}")
        return
    confirm(
        args,
        f"Build branch {plan['branch']} at {plan['checkout']}"
        + (
            " as a Slurm job on one exclusive node"
            if plan["launcher"] == "slurm"
            else " on this machine"
        ),
    )
    (out / "BUILD_OK").unlink(missing_ok=True)
    if plan["launcher"] == "slurm":
        cmd = [
            "sbatch",
            "--wait",
            "-N1",
            "--exclusive",
            "-J",
            "sirius-build",
            "-o",
            str(out / "build.out"),
        ]
        cmd += ["-p", plan["partition"]] if plan["partition"] else []
        cmd += ["-w", plan["nodes"][0]] if plan["nodes"] else []
        cmd += shlex.split(plan["sbatch_args"] or "") + [
            "--wrap",
            f"bash {q(str(script))}",
        ]
        say(f"building in a Slurm job; output in {out}/build.out")
        subprocess.call(cmd)
    else:
        with open(out / "build.out", "w") as log:
            say(f"building; output in {out}/build.out")
            subprocess.call(["bash", str(script)], stdout=log, stderr=subprocess.STDOUT)
    if not (out / "BUILD_OK").is_file():
        die(f"the build did not finish: see {out}/build.out")
    plan["build"] = {"commit": (out / "BUILD_OK").read_text().strip(), "reused": False}
    save(plan)
    say(
        f"built {plan['build']['commit']}. Next: show --out {out}, then smoke --out {out}"
    )


# ---------------------------------------------------------------------------- SQL


def q11_fraction(sf):
    return format(Context(prec=4).divide(Decimal("0.0001"), Decimal(sf)), "f")


def fetch_queries(plan, numbers):
    """Copies the query texts into the run folder, through a node if this host cannot see them."""
    cache = Path(plan["out"]) / "queries"
    cache.mkdir(exist_ok=True)
    wanted = [n for n in sorted(set(numbers)) if not (cache / f"q{n}.sql").is_file()]
    if not wanted:
        return cache
    source = plan["queries_dir"] or f"{plan['checkout']}/{QUERIES_SUBDIR}"
    script = f"cd {q(source)} || exit 1\n" + "\n".join(
        f"echo '@@Q{n}'; cat q{n}.sql 2> /dev/null || cat q{n:02d}.sql 2> /dev/null || cat {n}.sql"
        for n in wanted
    )
    if Path(source).is_dir():
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    else:
        result = node_sh(plan, script, (plan["nodes"] or [None])[0])
    if result.returncode != 0:
        die(
            f"could not read the queries in {source} (give --queries-dir to `plan`): {result.stderr.strip()[-300:]}"
        )
    for block in result.stdout.split("@@Q")[1:]:
        number, _, text = block.partition("\n")
        if not text.strip():
            die(f"query {number} in {source} is empty")
        (cache / f"q{int(number)}.sql").write_text(text)
    return cache


def write_sql(plan, kind, run_dir):
    numbers = plan["smoke_queries"] if kind == "smoke" else plan["queries"]
    runs = 1 if kind == "smoke" else plan["runs"]
    cache = fetch_queries(plan, numbers)
    lines = [".mode csv", ".timer on"]
    if plan["setup_sql"]:
        lines.append(Path(plan["setup_sql"]).read_text().strip())
    data = plan["data"].replace("'", "''")
    lines += [
        f"CREATE OR REPLACE VIEW {t} AS FROM read_parquet('{data}/{t}/*.parquet');"
        for t in TABLES
    ]
    notes = []
    for number in numbers:
        text = (cache / f"q{number}.sql").read_text().strip().rstrip(";").strip()
        if number == 11:
            literals = set(re.findall(r"\*\s*(0\.0+[1-9]\d*)\b", text))
            if len(literals) == 1:
                fraction = q11_fraction(plan["sf"])
                text = text.replace(literals.pop(), fraction)
                notes.append(f"Q11 fraction set to {fraction} (0.0001 / SF)")
            else:
                notes.append(
                    "Q11: fraction not found in the query text, so it was left as written. Check it is 0.0001 / SF"
                )
        for run in range(1, runs + 1):
            lines += [f"SELECT 'Q{number} run {run}' AS marker;", text + ";"]
    if kind == "smoke":
        lines.append("SELECT 'SFCHECK' AS marker, count(*) AS suppliers FROM supplier;")
    lines.append("FROM sirius_gateway_status();")
    (run_dir / "run.sql").write_text("\n".join(lines) + "\n")
    return numbers, runs, notes


# ---------------------------------------------------------------------------- launch


def exports(plan, run_dir):
    env = dict(plan["env"], SIRIUS_RANK_LOG_ROOT=str(run_dir / "rank-logs"))
    if plan["config_file"]:
        env["SIRIUS_CONFIG_FILE"] = plan["config_file"]
    if plan["statement_timeout"]:
        env["SIRIUS_GATEWAY_STATEMENT_TIMEOUT"] = str(plan["statement_timeout"])
    return env


def slurm_session(plan, run_dir, big):
    c, r, b = q(plan["checkout"]), q(str(run_dir)), q(str(big))
    sbatch = [
        "sbatch",
        "--parsable",
        f"-N{plan['num_nodes']}",
        f"--ntasks-per-node={plan['gpus_per_node']}",
    ]
    sbatch += ["-p", q(plan["partition"])] if plan["partition"] else []
    sbatch += ["-w", q(",".join(plan["nodes"]))] if plan["nodes"] else []
    sbatch += [f"--time={plan['time_limit']}"] if plan["time_limit"] else []
    sbatch += [plan["sbatch_args"]] if plan["sbatch_args"] else []
    sbatch += ["-o", f"{r}/job.out", "--job-name=sirius", "--gpus-per-task=1"]
    # Wrapped, so that it works when the checkout and the spill folder are visible on the nodes
    # only. The engine does not create its spill folder.
    spill = f"mkdir -p {q(plan['spill_dir'])} && " if plan.get("spill_dir") else ""
    inner = (
        f"{spill}cd {c} && SLURM_SUBMIT_DIR={c} exec bash scripts/sirius_slurm.sbatch"
    )
    submit = " ".join(sbatch) + f" --wrap {q(inner)}"
    client = q(
        f"{PIXI_PATH}; cd {c} && git log -1 --oneline > {r}/env.txt; "
        f"pixi run sirius-client -bail < {r}/run.sql > {b}/client.txt 2>&1"
    )
    run_client = (
        f'srun --jobid="$job" --overlap -N1 -n1 --gpus=0 bash -c {client}'
        if plan["client_on"] == "job"
        else f"bash -c {client}"
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by tpch_bench.py. One run is one Slurm job with fresh ranks: start the gateway",
            "# and one rank per GPU (scripts/sirius_slurm.sbatch), send run.sql, cancel the job.",
            "set -uo pipefail",
            PIXI_PATH,
            *[
                f"export {key}={q(value)}"
                for key, value in sorted(exports(plan, run_dir).items())
            ],
            f"job=$({submit}) || {{ echo 'sbatch failed'; exit 1; }}",
            "job=${job%%;*}",
            f'echo "$job" > {r}/job.id',
            "cleanup()",
            "{",
            '  scancel "$job" 2> /dev/null',
            "  # The job's own exit removes its gateway folder, but can leave it behind half-removed.",
            '  for _ in $(seq 30); do squeue -j "$job" -h -o %T 2> /dev/null | grep -q . || break; sleep 2; done',
            '  rm -rf "$HOME/.sirius/gateways/$job"',
            "}",
            "trap cleanup EXIT",
            'echo "job $job submitted; waiting for the gateway"',
            'until [ -s "$HOME/.sirius/gateways/$job/gateway.conf" ]; do',
            f'  squeue -j "$job" -h -o %T | grep -q . || {{ echo "job $job ended before the gateway was up: see {r}/job.out"; exit 1; }}',
            "  sleep 5",
            "done",
            f'echo "gateway up; sending {r}/run.sql"',
            'export SIRIUS_CLUSTER="$job"    # the client finds the gateway by this name',
            run_client,
            "rc=$?",
            "sleep 5    # the ranks flush their logs every 3 s",
            "exit $rc",
            "",
        ]
    )


def local_session(plan, run_dir, big):
    c, r, b, d = q(plan["checkout"]), q(str(run_dir)), q(str(big)), q(plan["data"])
    others = " ".join(plan["nodes"][1:])
    env = dict(
        exports(plan, run_dir),
        SIRIUS_CLUSTER="tpch-bench",
        SIRIUS_GATEWAY_JOIN_WAIT="120",
        SIRIUS_GATEWAY_MIN_RANKS=str(plan["ranks"]),
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by tpch_bench.py, after docs/super-sirius/distributed-experiments-no-slurm.md:",
            "# a gateway, one rank per GPU here and on the other machines, the client, then stop it all.",
            "# Only for machines that Slurm does not schedule.",
            "set -uo pipefail",
            *[f"export {key}={q(value)}" for key, value in sorted(env.items())],
            f'NODES="{others}"',
            "state=$HOME/.sirius/gateways/$SIRIUS_CLUSTER",
            "if ss -ltn | grep -E ':(9494|950[0-9]|2700[0-9]) '; then echo 'ports in use'; exit 1; fi",
            f"cd {c} && git log -1 --oneline > {r}/env.txt",
            *(
                [
                    f"mkdir -p {q(plan['spill_dir'])}",
                    f'for n in $NODES; do ssh "$n" mkdir -p {q(plan["spill_dir"])}; done',
                ]
                if plan.get("spill_dir")
                else []
            ),
            'rm -rf "$state"',
            f"on_node=\"cd {c} && env $(env | grep -E '^(SIRIUS_|UCX_)' | tr '\\n' ' ') $(command -v pixi) run scripts/sirius_node.sh\"",
            "stop()",
            "{",
            f"  for n in $NODES; do ssh \"$n\" \"pkill -u \\$USER -f '{plan['checkout']}/script[s]/'\"; done",
            '  kill $(jobs -p) 2> /dev/null; wait; rm -rf "$state"',
            "}",
            "trap 'exit 1' INT TERM",
            "trap stop EXIT",
            f"scripts/sirius_gateway.sh --serve > {r}/gateway.out 2>&1 &",
            "gateway=$!",
            'until [ -s "$state/gateway.conf" ]; do',
            f'  kill -0 $gateway 2> /dev/null || {{ echo "no gateway: see {r}/gateway.out"; exit 1; }}',
            "  sleep 1",
            "done",
            f"scripts/sirius_node.sh --check --data {d} || exit 1",
            f'for n in $NODES; do ssh "$n" "$on_node --check --data {plan["data"]}" || exit 1; done',
            f"scripts/sirius_node.sh > {r}/node.out 2>&1 &",
            f'for n in $NODES; do ssh "$n" "$on_node" > {r}/node-$n.out 2>&1 & done',
            f"scripts/sirius_client.sh -bail < {r}/run.sql > {b}/client.txt 2>&1 &",
            "wait $!",
            "rc=$?",
            "sleep 5    # the ranks flush their logs every 3 s",
            "exit $rc",
            "",
        ]
    )


def cmd_run(args, kind):
    plan = load(args.out)
    if plan.get("preflight") != "ok" or not plan.get("build"):
        die("run `preflight` and `build` first")
    out = Path(plan["out"])
    if kind == "full" and not (out / "smoke" / "PASS").is_file() and not args.force:
        die(
            "the smoke test has not passed for this plan: run `smoke` first (or pass --force)"
        )
    run_dir = out / kind
    if run_dir.exists():
        if not args.rerun:
            die(f"{run_dir} exists: pass --rerun to replace it")
        shutil.rmtree(run_dir)
    run_dir.mkdir()
    big = Path(plan["big_dir"]) / kind if plan["big_dir"] else run_dir
    big.mkdir(parents=True, exist_ok=True)
    numbers, runs, notes = write_sql(plan, kind, run_dir)
    slurm = plan["launcher"] == "slurm"
    script = run_dir / "session.sh"
    script.write_text((slurm_session if slurm else local_session)(plan, run_dir, big))
    (run_dir / "meta.json").write_text(
        json.dumps({"queries": numbers, "runs": runs, "notes": notes, "big": str(big)})
    )
    what = (
        f"{kind} run: {len(numbers)} queries x {runs} run(s) on {plan['ranks']} ranks, {where(plan)}, "
        f"config {plan['config_label']}"
    )
    say(what)
    for note in notes:
        say(note)
    say(f"script: {script}   SQL: {run_dir}/run.sql")
    if args.dry_run:
        say("dry run: nothing started")
        return
    confirm(args, what[0].upper() + what[1:])
    command = (
        ["bash", str(script)]
        if slurm
        else [
            "bash",
            "-c",
            f"{PIXI_PATH}; cd {q(plan['checkout'])} && pixi run bash {q(str(script))}",
        ]
    )
    rc = subprocess.call(command)
    (run_dir / "rc").write_text(f"{rc}\n")
    ok = report(plan, kind)
    if ok:
        (run_dir / "PASS").write_text("")
    if kind == "smoke" and ok:
        say(f"smoke test passed. Next, after the user agrees: full --out {out}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------- results


def parse_client(path):
    """The second 'Run Time' after a marker is the query's; the first is the marker's own."""
    times, errors, status_rows, sf_rows = {}, [], 0, None
    current, seen = None, 0
    with open(path, encoding="utf-8", errors="replace") as file:
        for raw in file:
            line = raw.rstrip("\r\n")
            marker = re.fullmatch(r'"?Q(\d+) run (\d+)"?', line.strip())
            if marker:
                current, seen = (int(marker.group(1)), int(marker.group(2))), 0
                continue
            timed = re.match(r"Run Time \(s\): real ([0-9.]+)", line)
            if timed and current:
                seen += 1
                if seen == 2:
                    times[current] = float(timed.group(1))
                    current = None
                continue
            if re.search(r"\bError\b", line):
                errors.append(
                    (
                        (
                            f"Q{current[0]} run {current[1]}"
                            if current
                            else "outside a query"
                        ),
                        line.strip()[:300],
                    )
                )
            if ",up,ready," in line:
                status_rows += 1
            check = re.match(r'"?SFCHECK"?,(\d+)', line)
            if check:
                sf_rows = int(check.group(1))
    return times, errors, status_rows, sf_rows


def scan_rank_logs(run_dir):
    paths, held, limit = Counter(), 0, 500 * 1024 * 1024
    logs = (
        [p for p in (run_dir / "rank-logs").rglob("*") if p.is_file()]
        if (run_dir / "rank-logs").is_dir()
        else []
    )
    if not logs or sum(p.stat().st_size for p in logs) > limit:
        return None
    for log in logs:
        with open(log, encoding="utf-8", errors="replace") as file:
            for line in file:
                path = re.search(
                    r"exchange: rank \d+ .*-> rank \d+ .*: (.+)$", line.rstrip()
                )
                if path:
                    paths[path.group(1)] += 1
                held += "[gds] held:" in line
    return paths, held


def report(plan, kind):
    run_dir = Path(plan["out"]) / kind
    meta = json.loads((run_dir / "meta.json").read_text())
    client = Path(meta["big"]) / "client.txt"
    done = (run_dir / "rc").is_file()
    if not client.is_file():
        say(f"{kind}: no client output yet ({client})")
        return False
    times, errors, ready, sf_rows = parse_client(client)
    problems = []
    if done and (run_dir / "rc").read_text().strip() != "0":
        problems.append(
            f"the client or the launcher exited with {(run_dir / 'rc').read_text().strip()}"
        )
    problems += [f"{where_}: {text}" for where_, text in errors[:10]]
    expected = [(n, r) for n in meta["queries"] for r in range(1, meta["runs"] + 1)]
    lost = [f"Q{n} run {r}" for n, r in expected if (n, r) not in times]
    if done and lost:
        problems.append("no time for " + ", ".join(lost))
    if done and ready != plan["ranks"]:
        problems.append(
            f"{ready} of {plan['ranks']} ranks were up and ready in the final status"
        )
    sf_line = None
    if kind == "smoke":
        want = Decimal(plan["sf"]) * 10000
        sf_line = f"supplier has {sf_rows} rows; SF {plan['sf']} expects {want:f}"
        if done and (sf_rows is None or Decimal(sf_rows) != want):
            problems.append(
                sf_line + ": the dataset does not look like this scale factor"
            )
    logs = scan_rank_logs(run_dir)

    with open(run_dir / "timings.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "run", "seconds"])
        writer.writerows([n, r, times.get((n, r), "")] for n, r in expected)
    state = "in progress" if not done else ("FAIL" if problems else "PASS")
    lines = [
        f"# TPC-H {kind}: {state}",
        "",
        f"- Branch `{plan['branch']}`, build `{plan['build']['commit']}`",
        f"- {plan['ranks']} ranks, {where(plan)}, launcher {plan['launcher']}",
        f"- Dataset `{plan['data']}`, scale factor {plan['sf']}",
        f"- Sirius config: {plan['config_label']}"
        + (f" (`{plan['config_file']}`)" if plan["config_file"] else ""),
        *[f"- {note}" for note in meta["notes"]],
        "",
        "| Query | "
        + " | ".join(f"run {r} (s)" for r in range(1, meta["runs"] + 1))
        + " | best (s) |",
        "|---|" + "---|" * (meta["runs"] + 1),
    ]
    totals = [0.0] * (meta["runs"] + 1)
    complete = [True] * (meta["runs"] + 1)
    for n in meta["queries"]:
        row = [times.get((n, r)) for r in range(1, meta["runs"] + 1)]
        row.append(min((t for t in row if t is not None), default=None))
        for i, value in enumerate(row):
            totals[i] += value or 0
            complete[i] &= value is not None
        lines.append(
            f"| Q{n} | "
            + " | ".join("" if v is None else f"{v:.2f}" for v in row)
            + " |"
        )
    lines.append(
        "| **Total** | "
        + " | ".join(
            f"**{t:.1f}**" if c else "incomplete" for t, c in zip(totals, complete)
        )
        + " |"
    )
    lines += [
        "",
        "## Checks",
        f"- Ranks up and ready in the final status: {ready} of {plan['ranks']}",
        f"- Lines with an error in the client output: {len(errors)}",
    ]
    if sf_line:
        lines.append(f"- Scale factor check: {sf_line}")
    if logs:
        lines.append(
            "- Exchange paths in the rank logs: "
            + ("; ".join(f"{k} ({v})" for k, v in logs[0].items()) or "none logged")
        )
        lines.append(f"- GDS held-read lines in the rank logs: {logs[1]}")
    lines.append(
        '- Answers were not compared with a reference: see SKILL.md, "Check the answers".'
    )
    if problems:
        lines += ["", "## Problems", *[f"- {p}" for p in problems]]
    text = "\n".join(lines) + "\n"
    (run_dir / "summary.md").write_text(text)
    print(text)
    say(
        f"written: {run_dir}/summary.md, {run_dir}/timings.csv; client output: {client}"
    )
    return done and not problems


def cmd_report(args):
    plan = load(args.out)
    kinds = [
        k for k in ("smoke", "full") if (Path(plan["out"]) / k / "meta.json").is_file()
    ]
    if not kinds:
        die("nothing has run yet in this folder")
    for kind in kinds:
        report(plan, kind)


# ---------------------------------------------------------------------------- command line


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("configs", help="list the config presets").set_defaults(
        run=cmd_configs
    )

    plan = sub.add_parser("plan", help="record and validate the inputs")
    plan.add_argument("--branch", help="branch to test")
    plan.add_argument(
        "--checkout", help="checkout folder: used if it exists, cloned there if not"
    )
    plan.add_argument(
        "--data", help="dataset folder with one subfolder of Parquet files per table"
    )
    plan.add_argument("--sf", help="scale factor of the dataset")
    plan.add_argument(
        "--mode",
        choices=["single", "multi"],
        help="one node with all its GPUs, or several nodes",
    )
    plan.add_argument("--config", help="preset name, YAML path, or engine-defaults")
    plan.add_argument(
        "--out", help="new run folder, visible on this host and on the nodes"
    )
    plan.add_argument(
        "--nodes",
        help="node names, comma-separated (with the local launcher the first is this machine)",
    )
    plan.add_argument(
        "--num-nodes", type=int, help="with Slurm: how many nodes, for Slurm to pick"
    )
    plan.add_argument(
        "--launcher",
        choices=["auto", "slurm", "local"],
        default="auto",
        help="slurm on nodes Slurm schedules; local only on machines it does not (default: slurm if sbatch exists)",
    )
    plan.add_argument("--partition", help="Slurm partition")
    plan.add_argument(
        "--gpus-per-node", type=int, help="GPUs per node, if preflight cannot find out"
    )
    plan.add_argument(
        "--sbatch-args", help="extra sbatch options for the site, as one string"
    )
    plan.add_argument(
        "--time-limit", help="Slurm time limit of a run, as sbatch --time takes it"
    )
    plan.add_argument(
        "--client-on",
        choices=["job", "submit"],
        default="job",
        help="run the SQL client inside the job (default) or on this host",
    )
    plan.add_argument(
        "--spill-dir", help="spill folder; replaces @SPILL@ in the config"
    )
    plan.add_argument(
        "--env",
        action="append",
        metavar="KEY=VALUE",
        help="extra environment for gateway and ranks",
    )
    plan.add_argument(
        "--statement-timeout",
        type=int,
        help="seconds before the gateway cancels a statement",
    )
    plan.add_argument(
        "--queries-dir",
        help=f"folder with q1.sql ... q22.sql (default: <checkout>/{QUERIES_SUBDIR})",
    )
    plan.add_argument(
        "--setup-sql",
        help="file with statements to run before the views, such as SET lines",
    )
    plan.add_argument(
        "--queries", default="1-22", help="queries of the full benchmark (default 1-22)"
    )
    plan.add_argument(
        "--runs",
        type=int,
        default=2,
        help="runs of each query in a row (default 2: cold, warm)",
    )
    plan.add_argument(
        "--smoke-queries",
        default="1,6,12",
        help="queries of the smoke test (default 1,6,12)",
    )
    plan.add_argument(
        "--big-dir",
        help="folder for the client output, which can reach gigabytes (default: the run folder)",
    )
    plan.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="repository to clone if the checkout does not exist",
    )
    plan.add_argument(
        "--submodules", default=DEFAULT_SUBMODULES, help="submodules the build needs"
    )
    plan.add_argument(
        "--replace", action="store_true", help="replace the plan already in --out"
    )
    plan.add_argument(
        "--yes", action="store_true", help="do not ask whether the inputs are correct"
    )
    plan.set_defaults(run=cmd_plan)

    for name, function, text in [
        ("show", cmd_show, "print the plan and the Sirius configuration"),
        ("preflight", cmd_preflight, "check tools, nodes, dataset and build"),
        ("build", cmd_build, "clone if needed and build"),
        ("smoke", lambda a: cmd_run(a, "smoke"), "run the smoke test"),
        ("full", lambda a: cmd_run(a, "full"), "run the full benchmark"),
        ("report", cmd_report, "print the results"),
    ]:
        step = sub.add_parser(name, help=text)
        step.add_argument("--out", required=True, help="the run folder given to `plan`")
        if name in ("preflight", "build", "smoke", "full"):
            step.add_argument(
                "--yes", action="store_true", help="the user has approved this step"
            )
        if name in ("build", "smoke", "full"):
            step.add_argument(
                "--dry-run",
                action="store_true",
                help="write the scripts, start nothing",
            )
        if name == "build":
            step.add_argument(
                "--rebuild",
                action="store_true",
                help="build even if a build is present",
            )
            step.add_argument(
                "--use-existing-checkout",
                action="store_true",
                help="accept a checkout on another branch",
            )
        if name in ("smoke", "full"):
            step.add_argument(
                "--rerun",
                action="store_true",
                help="replace the results of an earlier run",
            )
        if name == "full":
            step.add_argument(
                "--force", action="store_true", help="run without a passed smoke test"
            )
        step.set_defaults(run=function)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
