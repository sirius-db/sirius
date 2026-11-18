#!/usr/bin/env python3
"""Plan, build, smoke-test and benchmark a query set on Distributed Sirius through its gateway.

The query set is TPC-H by default (--benchmark tpch), or any dataset with a folder of .sql files
(--benchmark custom).

One run folder (--out) holds everything: plan.json, the resolved Sirius config, the generated
scripts and SQL, the client output and the reports. Steps, in order:

  configs               list the Sirius config presets shipped with this skill
  reference             list the runs of the branch's Reference numbers table
  plan ...              record and validate the inputs (asks for missing ones at a terminal);
                        --reference NAME takes them from the record of a reported run
  show      --out DIR   print the plan and the Sirius configuration that will be used
  preflight --out DIR   check the tools, the nodes, the dataset and the build
  build     --out DIR   clone the branch if needed and build it
  smoke     --out DIR   a few queries once each, with pass/fail checks
  full      --out DIR   the benchmark queries, --runs times each
  report    --out DIR   print the results again; also works while a run is in progress
  check     --out DIR   compare the answers of a TPC-H run with expected answers

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
REFERENCE = SKILL / "reference" / "runs.json"
TPCH_TABLES = "customer lineitem nation orders part partsupp region supplier".split()
DEFAULT_REPO = "https://github.com/sirius-db/sirius.git"
# The submodules the branch's build instructions name; those a branch does not have are skipped.
DEFAULT_SUBMODULES = (
    "duckdb substrait cucascade sirius-duckdb/duckdb sirius-duckdb/extension-ci-tools"
)
QUERIES_SUBDIR = "test/tpch_performance/tpch_queries/orig"
PIXI_PATH = 'export PATH="${PIXI_HOME:-$HOME/.pixi}/bin:$PATH"'
q = shlex.quote


def say(*parts):
    print("[bench]", *parts, flush=True)


def die(message, code=1):
    print(f"[bench] ERROR: {message}", file=sys.stderr, flush=True)
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
    if input(f"[bench] {what}. Proceed? [y/N] ").strip().lower() not in (
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


def tpch_ids(text):
    """'1-22' or '1,6,12' -> ['Q1', 'Q6', 'Q12']."""
    numbers = []
    for part in str(text).split(","):
        lo, _, hi = part.strip().partition("-")
        numbers += range(int(lo), int(hi or lo) + 1)
    if not numbers or min(numbers) < 1 or max(numbers) > 22:
        raise ValueError(
            "TPC-H queries are numbers from 1 to 22, such as 1-22 or 1,6,12"
        )
    return [f"Q{n}" for n in numbers]


def natural(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def pick(selection, available, what):
    """Query names chosen from the folder: 'all', names, or numbers and ranges (7, 1-5)."""
    if not selection or selection == "all":
        return list(available)
    picked = []
    for part in (p.strip() for p in selection.split(",") if p.strip()):
        if part in available:
            picked.append(part)
            continue
        lo, _, hi = part.partition("-")
        if not (lo.isdigit() and (hi.isdigit() or not hi)):
            die(f"{what}: no query named '{part}' among {', '.join(available[:12])}")
        for n in range(int(lo), int(hi or lo) + 1):
            match = [a for a in available if re.fullmatch(rf"[A-Za-z_]*0*{n}", a)]
            if len(match) != 1:
                die(
                    f"{what}: query number {n} matches {match or 'no file'} in the queries folder"
                )
            picked += match
    return picked


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


def with_gds(text, gds):
    """Puts the GDS scan settings of a reference run under the config's scan_manager key."""
    lines = text.splitlines()
    at = [i for i, line in enumerate(lines) if line.strip() == "scan_manager:"]
    if len(at) != 1:
        die("the GDS settings need one 'scan_manager:' line of its own in the config")
    pad = " " * (len(lines[at[0]]) - len(lines[at[0]].lstrip()) + 2)
    settings = ", ".join(f"{key}: {value}" for key, value in gds.items())
    lines[at[0] + 1 : at[0] + 1] = [
        f"{pad}backend: gds",
        f"{pad}gds: {{ {settings} }}",
    ]
    return "\n".join(lines) + "\n"


def on_site(recorded):
    run = recorded.get("accel_etl_run")
    return f"accel-etl run {run}" if run else "not on accel-etl"


def reference_runs():
    doc = json.loads(REFERENCE.read_text())
    return doc["defaults"], doc["runs"]


def cmd_reference(_args):
    _, runs = reference_runs()
    print(
        "Runs of the Reference numbers table, docs/super-sirius/distributed-experiments.md:"
    )
    for name, run in sorted(runs.items(), key=lambda item: natural(item[0])):
        recorded = run["recorded"]
        print(
            f"  {name:22s} SF{run['sf']:7s} {run['scans']:9s} {run['nodes']} node(s) "
            f"{recorded['total_s']:8.1f} s   {on_site(recorded)}"
        )
    print("Set one up with: plan --reference NAME --data ... --checkout ... --out ...")


def reference_inputs(args):
    """Fills the inputs the user did not give from the record of a reported run."""
    defaults, runs = reference_runs()
    run = runs.get(args.reference)
    if not run:
        die(f"no reference run '{args.reference}': `reference` lists them")
    if args.benchmark != "tpch":
        die("a reference run is a TPC-H run")
    named = len([n for n in (args.nodes or "").split(",") if n])
    if (named or args.num_nodes or run["nodes"]) != run["nodes"]:
        die(f"{args.reference} ran on {run['nodes']} node(s)")
    values = dict(defaults, **run)
    values["mode"] = "single" if run["nodes"] == 1 else "multi"
    values["num_nodes"] = None if named else run["nodes"]
    for key in ("setup_sql", "queries_dir"):
        if values.get(key):
            values[key] = str(SKILL / values[key])
    if values.get("sessions"):
        values["sessions"] = ";".join(values["sessions"])
    if args.queries or args.sessions:  # the user's selection replaces the record's
        values["queries"] = values["sessions"] = None
    gds = None if args.config else run.get("gds")
    for key in (
        "branch sf mode num_nodes commit config queries_dir setup_sql queries sessions "
        "runs order results exclusive statement_timeout"
    ).split():
        if getattr(args, key) is None and values.get(key) is not None:
            setattr(args, key, values[key])
    return {
        "name": args.reference,
        "dataset": run["dataset"],
        "gds": gds,
        "recorded": run["recorded"],
    }


def cmd_plan(args):
    missing = []
    reference = reference_inputs(args) if args.reference else None

    def need(name, label, check=None, default=None):
        value = getattr(args, name)
        if value in (None, "") and default is not None:
            value = default
        if value in (None, ""):
            if not sys.stdin.isatty():
                missing.append("--" + name.replace("_", "-"))
                return None
            while value in (None, ""):
                value = input(f"[bench] {label}: ").strip()
        while check and check(str(value)):
            if not sys.stdin.isatty():
                die(f"--{name.replace('_', '-')} {value}: {check(str(value))}")
            print(f"  {check(str(value))}")
            value = input(f"[bench] {label}: ").strip()
        return value

    tpch = args.benchmark == "tpch"
    branch = need("branch", "Branch to test")
    checkout = need(
        "checkout", "Checkout folder, existing or to create (absolute path)", check_abs
    )
    data = args.data
    if not args.views_sql:
        data = need(
            "data",
            "Dataset folder, one subfolder of Parquet files per table (absolute path)",
            check_abs,
        )
    if tpch:
        sf = need("sf", "Scale factor of that dataset", check_sf)
    else:
        sf = args.sf
        if sf and check_sf(sf):
            die(f"--sf {sf}: {check_sf(sf)}")
        args.queries_dir = need(
            "queries_dir",
            "Folder of .sql files, one query per file (absolute path)",
            check_abs,
        )
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
    if reference and reference["gds"]:
        text = with_gds(text, reference["gds"])
        label += f", with the GDS scan settings of {reference['name']}"
    spill = args.spill_dir
    if text and "@SPILL@" in text:
        if not spill and sys.stdin.isatty():
            spill = input(
                "[bench] Spill folder, on a disk with free space (absolute path): "
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
    for name in ("views_sql", "setup_sql"):
        if getattr(args, name) and not Path(getattr(args, name)).is_file():
            die(
                f"--{name.replace('_', '-')}: {getattr(args, name)} is not a readable file here"
            )
    if args.queries and args.sessions:
        die("give --queries or --sessions, not both")
    sessions = None
    if tpch:
        try:
            if args.sessions:
                sessions = [tpch_ids(group) for group in args.sessions.split(";")]
                full = [qid for group in sessions for qid in group]
            else:
                full = tpch_ids(args.queries or "1-22")
            smoke = tpch_ids(args.smoke_queries or "1,6,12")
        except ValueError as error:
            die(str(error))
        tables = None if args.views_sql else TPCH_TABLES
    else:  # resolved in preflight, where the queries folder and the dataset are visible
        full, smoke = None, None
        tables = [t for t in (args.tables or "auto").split(",") if t]
        tables = None if args.views_sql and not args.tables else tables

    out_dir = Path(out)
    if (out_dir / "plan.json").exists() and not args.replace:
        die(f"{out} already holds a plan: pick a new run folder, or pass --replace")
    out_dir.mkdir(parents=True, exist_ok=True)
    if text:
        (out_dir / "config.yaml").write_text(text)
    plan = {
        "reference": reference,
        "branch": branch,
        "commit": args.commit,
        "repo": args.repo,
        "checkout": checkout.rstrip("/"),
        "benchmark": args.benchmark,
        "name": args.name or ("TPC-H" if tpch else Path(data or args.queries_dir).name),
        "data": (data or "").rstrip("/"),
        "tables": tables,
        "views_sql": args.views_sql,
        "sf": str(sf) if sf else None,
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
        "query_selection": args.queries,
        "smoke_selection": args.smoke_queries,
        "queries": full,
        "session_selection": args.sessions,
        "sessions": sessions,
        "runs": args.runs or 2,
        "order": args.order or "query",
        "results": args.results or "csv",
        "exclusive": bool(args.exclusive),
        "smoke_queries": smoke,
        "big_dir": args.big_dir,
        "submodules": args.submodules,
        "spill_dir": spill,
        "out": str(out_dir),
    }
    save(plan)
    show(plan)
    if sys.stdin.isatty() and not args.yes:
        if input("[bench] Are these inputs correct? [y/N] ").strip().lower() not in (
            "y",
            "yes",
        ):
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
    reference = plan.get("reference")
    shape = (
        ", all queries once, then all again" if plan.get("order") == "pass" else ""
    ) + (
        f", in {len(plan['session_selection'].split(';'))} sessions, each with new ranks"
        if plan.get("session_selection")
        else ""
    )
    rows = [
        *(
            [
                (
                    "reference",
                    f"{reference['name']}: {reference['recorded']['total_s']} s recorded "
                    f"({reference['recorded']['total_of']}), "
                    f"{on_site(reference['recorded'])}",
                ),
                ("its dataset", reference["dataset"]),
            ]
            if reference
            else []
        ),
        ("branch", plan["branch"]),
        (
            "commit",
            plan.get("commit")
            or "the checkout's; for a new clone the head of the branch",
        ),
        ("repository", plan["repo"]),
        ("checkout", plan["checkout"]),
        ("benchmark", f"{plan['name']} ({plan['benchmark']})"),
        ("dataset", plan["data"] or "defined by the views file"),
        ("tables", tables_text(plan)),
        (
            "views",
            plan["views_sql"]
            or "one per table, read_parquet(<dataset>/<table>/*.parquet)",
        ),
        ("queries folder", plan["queries_dir"] or f"<checkout>/{QUERIES_SUBDIR}"),
        ("scale factor", plan["sf"] or "not given"),
        ("mode", f"{plan['mode']}, {where(plan)}"),
        (
            "launcher",
            plan["launcher"]
            + (f", partition {plan['partition']}" if plan["partition"] else "")
            + (", exclusive nodes" if plan.get("exclusive") else ""),
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
            f"{query_text(plan['smoke_queries'], plan['smoke_selection'] or 'the first 3')}, once each",
        ),
        (
            "full benchmark",
            f"{query_text(plan['queries'], plan['query_selection'] or plan.get('session_selection') or 'all')}, "
            f"{plan['runs']} run(s) each{shape}",
        ),
        (
            "results",
            (
                "Parquet files, written by a second client"
                if plan.get("results") == "parquet"
                else "CSV, through the client"
            ),
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


def tables_text(plan):
    if plan["tables"] is None:
        return "defined by the views file"
    if plan["tables"] == ["auto"]:
        return "every subfolder of the dataset with Parquet files (found in preflight)"
    return ", ".join(plan["tables"])


def query_text(ids, selection):
    if ids is None:
        return f"{selection} (resolved in preflight)"
    head = ", ".join(ids[:8]) + (f" ... ({len(ids)} in all)" if len(ids) > 8 else "")
    return f"{len(ids)} queries: {head}"


def cmd_show(args):
    show(load(args.out))


# ---------------------------------------------------------------------------- nodes


def node_sh(plan, script, node=None, minutes=5):
    """Runs a shell snippet where the nodes' file systems are visible."""
    if plan["launcher"] == "slurm":
        cmd = ["srun", "-N1", "-n1", f"--time={minutes}"]
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
    c, d, o = q(plan["checkout"]), q(plan["data"] or "/nonexistent"), q(plan["out"])
    tables = plan["tables"] or []
    checks = []
    if tables == ["auto"]:
        checks.append(
            f'for t in {d}/*/; do ls "$t"*.parquet > /dev/null 2>&1 && echo "FOUND_TABLE=$(basename "$t")"; done'
        )
    else:
        checks += [
            f"echo TABLE_{t}=$(ls {d}/{q(t)}/*.parquet 2> /dev/null | wc -l)"
            for t in tables
        ]
    if plan["queries_dir"] and plan["benchmark"] != "tpch":
        checks.append(
            f'for f in {q(plan["queries_dir"])}/*.sql; do [ -f "$f" ] && echo "QUERY_FILE=$(basename "$f" .sql)"; done'
        )
    script = "\n".join(
        [
            "echo HOST=$(hostname)",
            *checks,
            f"[ -e {c}/build/release/libsirius.so ] && [ -x {c}/gateway/build/release/duckdb ] && echo BUILD=yes || echo BUILD=no",
            f"[ -e {c}/.git ] && echo \"COMMIT=$(git -C {c} log -1 --format='%h %s')\" && "
            f"echo BRANCH=$(git -C {c} rev-parse --abbrev-ref HEAD) && "
            f"echo DIRTY=$(git -C {c} status --porcelain --untracked-files=no | wc -l) && "
            f"echo HEAD=$(git -C {c} rev-parse HEAD)",
            f"[ -d {o} ] && echo OUT=yes || echo OUT=no",
            "echo GPUS=$(nvidia-smi -L 2> /dev/null | wc -l)",
            "echo CPUS=$(nproc 2> /dev/null)",
        ]
    )
    result = node_sh(plan, script, node)
    lines = [line.split("=", 1) for line in result.stdout.splitlines() if "=" in line]
    facts = dict(lines)
    facts["FOUND_TABLES"] = sorted(v for k, v in lines if k == "FOUND_TABLE")
    facts["QUERY_FILES"] = sorted(
        (v for k, v in lines if k == "QUERY_FILE"), key=natural
    )
    if result.returncode != 0 or "HOST" not in facts:
        die(
            f"could not run a check on {node or 'a node'}: {result.stderr.strip()[-400:]}"
        )
    return facts


def wrong_commit(plan, facts):
    """What to tell the user if the checkout exists at another commit than the plan's."""
    want, have = plan.get("commit"), facts.get("HEAD")
    if want and not have and facts.get("BUILD") == "yes":
        return "git cannot read the checkout's commit, so it cannot be checked against the plan's"
    if want and have and not have.startswith(want):
        return (
            f"the checkout is at {have[:9]}, the plan says {want[:9]}. Give a checkout at "
            "that commit: the script does not switch one"
        )
    return None


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
    gpus, discovered, query_files = set(), None, None
    for node in plan["nodes"] or [None]:
        facts = node_facts(plan, node)
        host = facts["HOST"]
        if plan["tables"] == ["auto"]:
            found = facts["FOUND_TABLES"]
            empty = [] if found else ["any table"]
            if found and not discovered:
                discovered = found
            if not found:
                problems.append(
                    f"{host}: no subfolder of {plan['data']} holds Parquet files"
                )
        else:
            empty = [
                t for t in plan["tables"] or [] if facts.get(f"TABLE_{t}", "0") == "0"
            ]
            if empty:
                problems.append(
                    f"{host}: no Parquet files in {plan['data']}/<table>/ for {', '.join(empty)}"
                )
        if plan["benchmark"] != "tpch" and not query_files:
            query_files = facts["QUERY_FILES"]
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
        if wrong_commit(plan, facts):
            problems.append(f"{host}: {wrong_commit(plan, facts)}")
        elif (
            not plan.get("commit")
            and facts.get("BRANCH")
            and facts["BRANCH"] != plan["branch"]
        ):
            say(
                f"  note: the checkout is on '{facts['BRANCH']}', the plan says '{plan['branch']}'"
            )
    if len(gpus) > 1:
        problems.append(f"the nodes have different GPU counts: {sorted(gpus)}")
    if discovered:
        plan["tables"] = discovered
        say(f"tables found: {', '.join(discovered)}")
    if plan["benchmark"] != "tpch":
        bad = [f for f in query_files or [] if not re.fullmatch(r"[A-Za-z0-9_.-]+", f)]
        if not query_files:
            problems.append(f"no .sql files in {plan['queries_dir']}")
        elif bad:
            problems.append(
                f"query file names may hold letters, digits, '_', '.', '-' only: {bad}"
            )
        else:
            if plan.get("session_selection"):
                plan["sessions"] = [
                    pick(group, query_files, "--sessions")
                    for group in plan["session_selection"].split(";")
                ]
                plan["queries"] = [qid for group in plan["sessions"] for qid in group]
            else:
                plan["queries"] = pick(
                    plan["query_selection"], query_files, "--queries"
                )
            plan["smoke_queries"] = (
                pick(plan["smoke_selection"], query_files, "--smoke-queries")
                if plan["smoke_selection"]
                else plan["queries"][:3]
            )
            say(
                f"queries: {query_text(plan['queries'], '')}; smoke test: {', '.join(plan['smoke_queries'])}"
            )
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
    if wrong_commit(plan, facts):
        die(wrong_commit(plan, facts))
    if (
        not plan.get("commit")  # a checkout at the plan's commit is right on any branch
        and facts.get("BRANCH")
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
    clone = f"git clone --branch {q(plan['branch'])} {q(plan['repo'])} {c}"
    if plan.get("commit"):
        clone += f" && git -C {c} checkout --detach {q(plan['commit'])}"
    script = out / "build.sh"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# Generated by bench.py: clone if absent, then build Sirius and the gateway.",
                "set -euo pipefail",
                PIXI_PATH,
                f"[ -e {c}/.git ] || {{ {clone}; }}",
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


def query_files(plan, qid):
    """Candidate file names of a query: TPC-H accepts q7.sql, q07.sql or 7.sql."""
    if plan["benchmark"] == "tpch":
        n = int(qid[1:])
        return [f"q{n}.sql", f"q{n:02d}.sql", f"{n}.sql"]
    return [f"{qid}.sql"]


def fetch_queries(plan, ids):
    """Copies the query texts into the run folder, through a node if this host cannot see them."""
    cache = Path(plan["out"]) / "queries"
    cache.mkdir(exist_ok=True)
    wanted = [i for i in dict.fromkeys(ids) if not (cache / f"{i}.sql").is_file()]
    if not wanted:
        return cache
    source = plan["queries_dir"] or f"{plan['checkout']}/{QUERIES_SUBDIR}"
    script = f"cd {q(source)} || exit 1\n" + "\n".join(
        f"echo '@@{i}'; "
        + " || ".join(f"cat {q(f)} 2> /dev/null" for f in query_files(plan, i))
        for i in wanted
    )
    if Path(source).is_dir():
        result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    else:
        result = node_sh(plan, script, (plan["nodes"] or [None])[0])
    if result.returncode != 0:
        die(
            f"could not read the queries in {source} (give --queries-dir to `plan`): {result.stderr.strip()[-300:]}"
        )
    for block in result.stdout.split("@@")[1:]:
        qid, _, text = block.partition("\n")
        if not text.strip():
            die(f"query {qid} in {source} is empty or missing")
        (cache / f"{qid}.sql").write_text(text)
    return cache


def client2(plan, run_dir, big):
    """The second client of --results parquet."""
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by bench.py (--results parquet). This client attaches the gateway but does not",
        "# CONNECT: its statements run here and reach the gateway through quack_query_by_name, so",
        "# COPY writes the Parquet files on this side. The init file holds the token: owner only.",
        "set -euo pipefail",
        "state=$HOME/.sirius/gateways/$SIRIUS_CLUSTER",
        "gateway='' gateway_token='' gateway_fingerprint=''",
        "# shellcheck disable=SC1091",
        'source "$state/gateway.conf"',
        "(",
        "  umask 077",
        "  {",
        '    echo "SET GLOBAL http_timeout = 86400;"',
        "    echo \"ATTACH 'quack:$gateway' AS sirius (TOKEN '$gateway_token'"
        "${gateway_fingerprint:+, SSL_FINGERPRINT '$gateway_fingerprint'});\"",
        '  } > "$state/client2.init.sql"',
        ")",
        f"mkdir -p {q(str(big))}/answers",
        f"cd {q(plan['checkout'])}",
        "exec pixi run gateway/build/release/duckdb -unsigned -no-agent -bail "
        f'-init "$state/client2.init.sql" < {q(str(run_dir))}/run2.sql',
    ]
    return "\n".join(lines) + "\n"


def write_sql(plan, kind, run_dir, ids, runs, big):
    """Writes the SQL of one session and returns the notes for the report."""
    cache = fetch_queries(plan, ids)
    tpch = plan["benchmark"] == "tpch"
    parquet = plan.get("results") == "parquet"
    lines = [".mode csv", ".timer on"]
    if plan["setup_sql"]:
        lines.append(Path(plan["setup_sql"]).read_text().strip())
    if plan["views_sql"]:
        lines.append(Path(plan["views_sql"]).read_text().strip())
    else:
        data = plan["data"].replace("'", "''")
        lines += [
            f"CREATE OR REPLACE VIEW \"{t}\" AS FROM read_parquet('{data}/{t}/*.parquet');"
            for t in plan["tables"]
        ]
    notes, texts = [], {}
    for qid in ids:
        text = (cache / f"{qid}.sql").read_text().strip().rstrip(";").strip()
        if tpch and qid == "Q11":
            literals = set(re.findall(r"\*\s*(0\.0+[1-9]\d*)\b", text))
            if len(literals) == 1:
                fraction = q11_fraction(plan["sf"])
                text = text.replace(literals.pop(), fraction)
                notes.append(f"Q11 fraction set to {fraction} (0.0001 / SF)")
            else:
                notes.append(
                    "Q11: fraction not found in the query text, so it was left as written. Check it is 0.0001 / SF"
                )
        texts[qid] = text
    passes = range(1, runs + 1)
    if plan.get("order") == "pass":
        order = [(qid, run) for run in passes for qid in ids]
    else:
        order = [(qid, run) for qid in ids for run in passes]
    answers = str(big).replace("'", "''") + "/answers"
    body = []
    for qid, run in order:
        body.append(f"SELECT '{qid} run {run}' AS marker;")
        if not parquet:
            body.append(texts[qid] + ";")
            continue
        if "$q$" in texts[qid]:
            die(f"{qid}: --results parquet cannot send a query that contains $q$")
        # The statement's time is the query, the fetch and the write of the file.
        body.append(
            f"COPY (FROM quack_query_by_name('sirius', $q${texts[qid]}$q$)) "
            f"TO '{answers}/{qid}-run{run}.parquet';"
        )
    check = (
        ["SELECT 'SFCHECK' AS marker, count(*) AS suppliers FROM supplier;"]
        if kind == "smoke" and tpch
        else []
    )
    status = "FROM sirius_gateway_status();"
    if parquet:
        # The connected client makes the views; the second one runs the queries.
        (run_dir / "run.sql").write_text("\n".join(lines + check) + "\n")
        (run_dir / "run2.sql").write_text(
            "\n".join([".mode csv", ".timer on"] + body) + "\n"
        )
        (run_dir / "status.sql").write_text(f".mode csv\n{status}\n")
        (run_dir / "client2.sh").write_text(client2(plan, run_dir, big))
    else:
        (run_dir / "run.sql").write_text(
            "\n".join(lines + body + check + [status]) + "\n"
        )
    return notes


# ---------------------------------------------------------------------------- launch


def exports(plan, run_dir):
    env = dict(plan["env"], SIRIUS_RANK_LOG_ROOT=str(run_dir / "rank-logs"))
    if plan["config_file"]:
        env["SIRIUS_CONFIG_FILE"] = plan["config_file"]
    if plan["statement_timeout"]:
        env["SIRIUS_GATEWAY_STATEMENT_TIMEOUT"] = str(plan["statement_timeout"])
    return env


def client_steps(plan, r, b, client):
    """The shell commands that send the SQL; their exit code is the client's."""
    steps = f"{client} -bail < {r}/run.sql > {b}/client.txt 2>&1"
    if plan.get("results") == "parquet":
        steps += (
            f" && bash {r}/client2.sh > {b}/client2.txt 2>&1; rc=$?; "
            f"{client} < {r}/status.sql >> {b}/client.txt 2>&1; (exit $rc)"
        )
    return steps


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
    sbatch += ["--exclusive"] if plan.get("exclusive") else []
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
        + client_steps(plan, r, b, "pixi run sirius-client")
    )
    run_client = (
        f'srun --jobid="$job" --overlap -N1 -n1 --gpus=0 bash -c {client}'
        if plan["client_on"] == "job"
        else f"bash -c {client}"
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by bench.py. One run is one Slurm job with fresh ranks: start the gateway",
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
        SIRIUS_CLUSTER="sirius-bench",
        SIRIUS_GATEWAY_JOIN_WAIT="120",
        SIRIUS_GATEWAY_MIN_RANKS=str(plan["ranks"]),
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "# Generated by bench.py, after docs/super-sirius/distributed-experiments-no-slurm.md:",
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
            f"( {client_steps(plan, r, b, 'scripts/sirius_client.sh')} ) &",
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
    if big != run_dir and big.exists():
        if not args.rerun:
            die(f"{big} exists: pass --rerun to replace it")
        shutil.rmtree(big)
    numbers = plan["smoke_queries"] if kind == "smoke" else plan["queries"]
    runs = 1 if kind == "smoke" else plan["runs"]
    sessions = plan.get("sessions") if kind == "full" else None
    # A session is one job with new ranks. Several of them get a subfolder each: s1, s2, ...
    parts = (
        [(f"s{k}", group) for k, group in enumerate(sessions, 1)]
        if sessions
        else [("", numbers)]
    )
    slurm = plan["launcher"] == "slurm"
    notes = []
    for name, group in parts:
        (run_dir / name).mkdir(exist_ok=True)
        (big / name).mkdir(parents=True, exist_ok=True)
        notes += write_sql(plan, kind, run_dir / name, group, runs, big / name)
        (run_dir / name / "session.sh").write_text(
            (slurm_session if slurm else local_session)(
                plan, run_dir / name, big / name
            )
        )
    notes = list(dict.fromkeys(notes))
    script = run_dir / parts[0][0] / "session.sh"
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "queries": numbers,
                "runs": runs,
                "notes": notes,
                "big": str(big),
                "parts": [name for name, _ in parts],
            }
        )
    )
    what = (
        f"{kind} run: {len(numbers)} queries x {runs} run(s) on {plan['ranks']} ranks, {where(plan)}, "
        f"config {plan['config_label']}"
        + (f", in {len(parts)} sessions" if sessions else "")
    )
    say(what)
    for note in notes:
        say(note)
    say(f"script: {script}   SQL: {script.parent}/run.sql")
    if args.dry_run:
        say("dry run: nothing started")
        return
    confirm(args, what[0].upper() + what[1:])
    for name, group in parts:
        script = run_dir / name / "session.sh"
        if sessions:
            say(f"session {name}: {', '.join(group)}")
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
        if rc:  # do not start the next session on nodes where this one failed
            break
    (run_dir / "rc").write_text(f"{rc}\n")
    ok = report(plan, kind)
    if ok:
        (run_dir / "PASS").write_text("")
    if kind == "smoke" and ok:
        say(f"smoke test passed. Next, after the user agrees: full --out {out}")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------- results


def parse_client(path, ids):
    """The second 'Run Time' after a marker is the query's; the first is the marker's own."""
    times, errors, status_rows, sf_rows = {}, [], 0, None
    current, seen = None, 0
    with open(path, encoding="utf-8", errors="replace") as file:
        for raw in file:
            line = raw.rstrip("\r\n")
            marker = re.fullmatch(r'"?(.+?) run (\d+)"?', line.strip())
            if marker and marker.group(1) in ids:
                current, seen = (marker.group(1), int(marker.group(2))), 0
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
                            f"{current[0]} run {current[1]}"
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


def scan_rank_logs(run_dir, parts):
    paths, held, limit = Counter(), 0, 500 * 1024 * 1024
    logs = [
        p
        for name in parts
        for p in (run_dir / name / "rank-logs").rglob("*")
        if p.is_file()
    ]
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
    parts = meta.get("parts", [""])
    client = Path(meta["big"]) / parts[0] / "client.txt"
    done = (run_dir / "rc").is_file()
    if not client.is_file():
        say(f"{kind}: no client output yet ({client})")
        return False
    # With --results parquet the times are in the second client's output.
    times, errors, ups, sf_rows = {}, [], {}, None
    for name in parts:
        files = ["client.txt"] + (
            ["client2.txt"] if plan.get("results") == "parquet" else []
        )
        for file in files:
            path = Path(meta["big"]) / name / file
            if not path.is_file():
                continue
            found = parse_client(path, set(meta["queries"]))
            times.update(found[0])
            errors += found[1]
            ups[name] = ups.get(name, 0) + found[2]
            sf_rows = found[3] if sf_rows is None else sf_rows
    ready = min(ups.values())
    short = [name for name, up in ups.items() if up != plan["ranks"]]
    problems = []
    if done and (run_dir / "rc").read_text().strip() != "0":
        problems.append(
            f"the client or the launcher exited with {(run_dir / 'rc').read_text().strip()}"
        )
    problems += [f"{where_}: {text}" for where_, text in errors[:10]]
    expected = [(n, r) for n in meta["queries"] for r in range(1, meta["runs"] + 1)]
    lost = [f"{n} run {r}" for n, r in expected if (n, r) not in times]
    if done and lost:
        problems.append("no time for " + ", ".join(lost))
    if done and short:
        problems.append(
            f"{ready} of {plan['ranks']} ranks were up and ready in the final status"
            + (f" of session {', '.join(short)}" if short[0] else "")
        )
    sf_line = None
    if kind == "smoke" and plan["benchmark"] == "tpch":
        want = Decimal(plan["sf"]) * 10000
        sf_line = f"supplier has {sf_rows} rows; SF {plan['sf']} expects {want:f}"
        if done and (sf_rows is None or Decimal(sf_rows) != want):
            problems.append(
                sf_line + ": the dataset does not look like this scale factor"
            )
    logs = scan_rank_logs(run_dir, parts)

    with open(run_dir / "timings.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["query", "run", "seconds"])
        writer.writerows([n, r, times.get((n, r), "")] for n, r in expected)
    state = "in progress" if not done else ("FAIL" if problems else "PASS")
    lines = [
        f"# {plan['name']} {kind}: {state}",
        "",
        f"- Branch `{plan['branch']}`, build `{plan['build']['commit']}`",
        f"- {plan['ranks']} ranks, {where(plan)}, launcher {plan['launcher']}",
        f"- Dataset `{plan['data'] or plan['views_sql']}`"
        + (f", scale factor {plan['sf']}" if plan["sf"] else ""),
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
            f"| {n} | "
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
    reference = plan.get("reference")
    if reference and kind == "full":
        recorded = reference["recorded"]
        lines += [
            "",
            f"Recorded for {reference['name']}: **{recorded['total_s']} s** "
            f"({recorded['total_of']}), {on_site(recorded)}. "
            "Compare it with the best total of this run.",
        ]
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
    check = run_dir / "check"
    if (check / "rc").is_file():
        same = (check / "rc").read_text().strip() == "0"
        lines.append(
            f"- Answers compared with {(check / 'source').read_text().strip()}: "
            + ("all match" if same else "DIFFERENCES or missing answers")
            + f". See `{check}/result.txt`"
        )
    else:
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


# ---------------------------------------------------------------------------- answers

TPCH_TIES = ["3:revenue,o_orderdate", "18:o_totalprice,o_orderdate"]
# Run on the node by `check` for --results parquet, before the comparison.
ALIGN = r'''#!/usr/bin/env python3
"""Written by bench.py check. In a Parquet answer an integer sum can come back as a DOUBLE and
print as 123.0. Where the reference column holds only integers, this prints the run's whole
numbers the same way, so that scripts/compare_answers.py compares them as integers."""
import csv
import io
import re
import sys


def blocks(lines):
    """{(query, run): (first, end)}: the lines of each answer, header first."""
    found, current, state, first = {}, None, None, 0
    for i, line in enumerate(lines):
        marker = re.fullmatch(r'"?Q(\d+) run (\d+)"?', line.strip())
        if marker:
            current, state = (int(marker.group(1)), int(marker.group(2))), "marker"
        elif current and line.startswith("Run Time (s): real"):
            if state == "marker":
                state, first = "body", i + 1
            else:
                found[current], current = (first, i), None
    return found


def rows(lines):
    return [row for row in csv.reader(io.StringIO("\n".join(lines))) if row]


def read(path):
    with open(path, encoding="utf-8", errors="replace") as file:
        return file.read().splitlines()


client, kind, ref = sys.argv[1:4]
lines = read(client)
ref_lines = read(ref) if kind == "--ref-client" else []
ref_blocks = blocks(ref_lines)
changed = 0
for (query, _), (first, end) in sorted(blocks(lines).items(), key=lambda b: -b[1][0]):
    if kind == "--ref-client":
        a, b = ref_blocks.get((query, 1), (0, 0))
        want = rows(ref_lines[a:b])[1:]
    else:
        try:
            want = rows(read(f"{ref}/q{query:02d}.csv"))[1:]
        except OSError:
            want = []
    got = rows(lines[first:end])
    width = len(got[0]) if got else 0
    if len(got) < 2 or not want or any(len(row) != width for row in got + want):
        continue
    whole = [
        j
        for j in range(width)
        if all(re.fullmatch(r"-?\d+", row[j]) for row in want)
        and all(re.fullmatch(r"-?\d+(\.0+)?", row[j]) for row in got[1:])
    ]
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(got[0])
    for row in got[1:]:
        writer.writerow(
            [re.sub(r"\.0+$", "", cell) if j in whole else cell for j, cell in enumerate(row)]
        )
    new = text.getvalue().splitlines()
    if new != lines[first:end]:
        lines[first:end] = new
        changed += 1
with open(client, "w", encoding="utf-8") as file:
    file.write("\n".join(lines) + "\n")
print(f"whole numbers printed as integers in {changed} answer(s)", file=sys.stderr)
'''


def cmd_check(args):
    """Compares a finished run's answers with expected ones, through the checkout's own tool."""
    plan = load(args.out)
    out = Path(plan["out"])
    if plan["benchmark"] != "tpch":
        die(
            "`check` is for TPC-H runs: for another query set, run "
            "<checkout>/scripts/compare_answers.py on the client output yourself"
        )
    if bool(args.expected) == bool(args.ref_client):
        die("give one of --expected and --ref-client")
    kind = args.kind or ("full" if (out / "full" / "rc").is_file() else "smoke")
    run_dir = out / kind
    if not (run_dir / "rc").is_file():
        die(f"no finished {kind} run in {out}")
    meta = json.loads((run_dir / "meta.json").read_text())
    parts, big = meta.get("parts", [""]), Path(meta["big"])
    groups = plan["sessions"] if len(parts) > 1 or parts[0] else [meta["queries"]]
    check = run_dir / "check"
    if check.exists():
        shutil.rmtree(check)
    check.mkdir()
    o = q(str(check))
    duckdb = "pixi run gateway/build/release/duckdb -unsigned -no-agent"
    lines = [
        "#!/usr/bin/env bash",
        "# Generated by bench.py check: one client output with every session's answers, the",
        "# expected answers as one CSV file per query, then scripts/compare_answers.py.",
        "set -uo pipefail",
        PIXI_PATH,
        *[
            f"export {key}={q(value)}"
            for key, value in sorted(plan["env"].items())
            if key.startswith("PIXI_")
        ],
        f"cd {q(plan['checkout'])} || exit 1",
    ]
    if plan.get("results") == "parquet":
        # Read back, an answer file looks to the tool like the client's output of that query.
        sql = [".mode csv", ".timer on"]
        for name, group in zip(parts, groups):
            answers = str(big / name / "answers").replace("'", "''")
            for qid in group:
                for run in range(1, meta["runs"] + 1):
                    sql += [
                        f"SELECT '{qid} run {run}' AS marker;",
                        f"FROM read_parquet('{answers}/{qid}-run{run}.parquet');",
                    ]
        (check / "answers.sql").write_text("\n".join(sql) + "\n")
        lines.append(f"{duckdb} < {o}/answers.sql > {o}/client.txt 2>&1")
    else:
        clients = " ".join(q(str(big / name / "client.txt")) for name in parts)
        lines.append(f"cat {clients} > {o}/client.txt")
    if args.expected:
        # Only the queries of this run: the tool counts a reference without an answer as not run.
        e = q(args.expected.rstrip("/"))
        lines += [f"mkdir {o}/expected", f": > {o}/expected.sql"]
        for n in sorted({int(qid[1:]) for qid in meta["queries"]}):
            stem, target = args.pattern.format(n=n), f"{o}/expected/q{n:02d}.csv"
            lines += [
                f"if [ -f {e}/{stem}.parquet ]; then",
                f"  echo \"COPY (FROM read_parquet('{e}/{stem}.parquet')) TO '{target}' (FORMAT csv, HEADER);\" >> {o}/expected.sql",
                f"elif [ -f {e}/{stem}.csv ]; then",
                f"  cp {e}/{stem}.csv {target}",
                "fi",
            ]
        lines.append(
            f"[ ! -s {o}/expected.sql ] || {duckdb} -bail < {o}/expected.sql > {o}/expected.out 2>&1 "
            f"|| {{ cat {o}/expected.out; exit 2; }}"
        )
        reference = f"--ref-dir {o}/expected --ref-pattern 'q{{n:02d}}.csv'"
        source = f"the expected answers in {args.expected}"
    else:
        reference = f"--ref-client {q(args.ref_client)}"
        source = f"the reference output {args.ref_client}"
    ties = " ".join(f"--ties {tie}" for tie in TPCH_TIES)
    if plan.get("results") == "parquet":
        (check / "align.py").write_text(ALIGN)
        lines.append(
            f"python3 {o}/align.py {o}/client.txt "
            + (f"--ref-dir {o}/expected" if args.expected else reference)
        )
    lines += [
        f"python3 scripts/compare_answers.py {o}/client.txt {reference} {ties} > {o}/result.txt 2>&1",
        "rc=$?",
        f"cat {o}/result.txt",
        "exit $rc",
    ]
    script = check / "check.sh"
    script.write_text("\n".join(lines) + "\n")
    (check / "source").write_text(source + "\n")
    slurm = plan["launcher"] == "slurm"
    confirm(
        args,
        f"Compare the answers of the {kind} run with {source}"
        + (", in a short step on a node without a GPU" if slurm else ""),
    )
    result = node_sh(
        plan, f"bash {q(str(script))}", (plan["nodes"] or [None])[0], minutes=30
    )
    print(result.stdout, end="")
    if result.returncode not in (0, 1) or not (check / "result.txt").is_file():
        die(f"the comparison did not run: {result.stderr.strip()[-400:]}")
    (check / "rc").write_text(f"{result.returncode}\n")
    say(
        ("the answers match" if result.returncode == 0 else "the answers DIFFER")
        + f": {check}/result.txt. `report` now says so in summary.md"
    )
    sys.exit(result.returncode)


# ---------------------------------------------------------------------------- command line


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("configs", help="list the config presets").set_defaults(
        run=cmd_configs
    )
    sub.add_parser(
        "reference", help="list the runs of the Reference numbers table"
    ).set_defaults(run=cmd_reference)

    plan = sub.add_parser("plan", help="record and validate the inputs")
    plan.add_argument(
        "--benchmark",
        choices=["tpch", "custom"],
        default="tpch",
        help="tpch (default): the 22 TPC-H queries; custom: any dataset and a folder of .sql files",
    )
    plan.add_argument(
        "--name",
        help="name of the query set in the reports (default: TPC-H, or the dataset folder's name)",
    )
    plan.add_argument(
        "--reference",
        metavar="NAME",
        help="a run of the Reference numbers table (`reference` lists them): its record gives "
        "the commit, config, queries, order and timeout; a flag given here replaces its value",
    )
    plan.add_argument("--branch", help="branch to test")
    plan.add_argument(
        "--commit",
        help="commit to test: checked out in a new clone; an existing checkout must be at it",
    )
    plan.add_argument(
        "--checkout", help="checkout folder: used if it exists, cloned there if not"
    )
    plan.add_argument(
        "--data", help="dataset folder with one subfolder of Parquet files per table"
    )
    plan.add_argument(
        "--tables",
        help="custom: the tables (subfolders of --data), comma-separated; default auto, every subfolder with Parquet files",
    )
    plan.add_argument(
        "--views-sql",
        help="file whose statements create the tables or views, instead of one read_parquet view per table (other formats, other layouts)",
    )
    plan.add_argument("--sf", help="scale factor of the dataset (required for tpch)")
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
        help=f"folder of .sql files, one query per file; required for custom (tpch default: <checkout>/{QUERIES_SUBDIR})",
    )
    plan.add_argument(
        "--setup-sql",
        help="file with statements to run before the views, such as SET lines",
    )
    plan.add_argument(
        "--queries",
        help="queries of the full benchmark: tpch numbers and ranges (default 1-22); custom file names without .sql, numbers, or all (default)",
    )
    plan.add_argument(
        "--sessions",
        help="instead of --queries: groups of queries separated by ';', each group one job "
        "with new ranks, such as '1-8;9;10-22'",
    )
    plan.add_argument(
        "--runs",
        type=int,
        help="runs of each query (default 2: cold, warm)",
    )
    plan.add_argument(
        "--order",
        choices=["query", "pass"],
        help="query (default): the runs of a query in a row; pass: all queries once, then all again",
    )
    plan.add_argument(
        "--results",
        choices=["csv", "parquet"],
        help="csv (default): the client prints the rows; parquet: a second client writes each "
        "result as a Parquet file under --big-dir, so no time goes to printing rows",
    )
    plan.add_argument(
        "--exclusive",
        action="store_true",
        default=None,
        help="with Slurm: the job takes its nodes exclusively, so no other job shares them",
    )
    plan.add_argument(
        "--smoke-queries",
        help="queries of the smoke test (tpch default 1,6,12; custom default: the first three)",
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
        ("check", cmd_check, "compare the answers with expected answers"),
    ]:
        step = sub.add_parser(name, help=text)
        step.add_argument("--out", required=True, help="the run folder given to `plan`")
        if name in ("preflight", "build", "smoke", "full", "check"):
            step.add_argument(
                "--yes", action="store_true", help="the user has approved this step"
            )
        if name == "check":
            step.add_argument(
                "--expected",
                help="folder with one expected answer per query, Parquet or CSV with a header",
            )
            step.add_argument(
                "--pattern",
                default="q{n:02d}",
                help="file name of a query's expected answer, without its extension (default q{n:02d})",
            )
            step.add_argument(
                "--ref-client",
                help="instead of --expected: a client output with the same markers, such as "
                "the same run.sql through another engine",
            )
            step.add_argument(
                "--kind",
                choices=["smoke", "full"],
                help="which run to check (default: the full run if there is one)",
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
