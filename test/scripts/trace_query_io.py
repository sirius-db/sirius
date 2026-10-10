#!/usr/bin/env python3
"""Save Linux file-read syscall evidence separately from untraced performance samples."""

import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import re
import subprocess


READ = re.compile(r"(read)\(\d+<([^>]+)>, .*, (\d+)\)$")
PREAD = re.compile(r"(pread64)\(\d+<([^>]+)>, .*, (\d+), -?\d+\)$")
OPEN = re.compile(r'openat\(([^,]+), "([^"\\]*)", .*\)$')
LINE = re.compile(r"(\d+\.\d+) (.*) = (-?\d+)(.*?) <([\d.]+)>$")


def joined_lines(lines, counts):
    pending = None
    for line in lines:
        line = line.strip()
        if line.endswith("<unfinished ...>"):
            if pending:
                counts["unparsed"] += 1
            pending = line.removesuffix("<unfinished ...>")
        elif " resumed>" in line:
            resumed = re.search(r"<\.\.\. (\w+) resumed>", line)
            if pending and resumed and f" {resumed[1]}(" in pending:
                yield pending + line.split(" resumed>", 1)[1]
                pending = None
            else:
                counts["unparsed"] += 2 if pending else 1
                pending = None
        else:
            yield line
    if pending:
        counts["unparsed"] += 1


def parse_call(line):
    match = LINE.fullmatch(line)
    if not match:
        return None
    timestamp, call, returned, tail, duration = match.groups()
    returned = int(returned)
    read = READ.fullmatch(call) or PREAD.fullmatch(call)
    opened = OPEN.fullmatch(call)
    if read:
        name, path, requested = read.groups()
    elif opened:
        directory, path = opened.groups()
        name, requested = "openat", "0"
        if returned >= 0:
            resolved = re.search(r"<([^>]+)>", tail)
            if not resolved:
                return None
            path = resolved[1]
        elif not path.startswith("/"):
            resolved = re.search(r"<([^>]+)>", directory)
            if not resolved:
                return None
            path = str(Path(resolved[1]) / path)
    else:
        return None
    if not path.startswith("/") or "\\" in path:
        return None
    start = int(Decimal(timestamp) * 1000000)
    return {
        "start_us": start,
        "end_us": start + int(Decimal(duration) * 1000000),
        "call": name,
        "path": path,
        "requested_bytes": int(requested),
        "returned_bytes": max(returned, 0) if read else 0,
        "status": "failed" if returned < 0 else "ok",
    }


def query_windows(log):
    windows = []
    for line in log.splitlines():
        if "PREPARATION_COST_SAMPLE " not in line:
            continue
        fields = dict(
            token.split("=", 1)
            for token in line.split("PREPARATION_COST_SAMPLE ", 1)[1].split()
        )
        start, end = int(fields["wall_begin_us"]), int(fields["wall_end_us"])
        plan = fields.get("wall_plan_end_us", "unobserved")
        plan = None if plan == "unobserved" else int(plan)
        if end < start or (plan is not None and not start <= plan <= end):
            raise ValueError("invalid query phase timestamps")
        windows.append((start, end, plan, fields["observations"], fields["sample"]))
    return windows


def attribute_call(call, windows):
    matches = [
        w for w in windows if w[0] <= call["start_us"] and call["end_us"] <= w[1]
    ]
    if len(matches) != 1:
        return {
            "sample": "unobserved",
            "observations": "unobserved",
            "phase": "unobserved",
        }
    start, end, plan, mode, sample = matches[0]
    phase = "unobserved"
    if plan is not None:
        phase = (
            "planning"
            if call["end_us"] <= plan
            else "execution_window" if call["start_us"] >= plan else "cross_boundary"
        )
    return {"sample": sample, "observations": mode, "phase": phase}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-prefix", action="append", required=True)
    parser.add_argument(
        "--reparse", type=Path, help="existing trace directory; do not run a command"
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="-- command [arguments]"
    )
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if bool(command) == bool(args.reparse) or any(
        not p.startswith("/") for p in args.path_prefix
    ):
        parser.error(
            "provide either a command or --reparse, and absolute --path-prefix values"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    trace = ["strace", "-ff", "-ttt", "-T", "-yy", "-s", "0"] + [
        "-o",
        str(args.output / "trace"),
        "-e",
        "trace=openat,read,pread64,readv,preadv,preadv2,mmap,io_uring_setup,io_uring_enter",
    ]
    source = args.reparse or args.output
    returncode = None
    if not args.reparse:
        with (args.output / "run.log").open("w") as log:
            result = subprocess.run(
                trace + command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
            returncode = result.returncode
    elif (source / "summary.json").exists():
        previous = json.loads((source / "summary.json").read_text())
        returncode = previous.get("returncode")
        command = previous.get("command", [])
    windows = query_windows((source / "run.log").read_text())
    totals = Counter()
    phases = {}
    file_kinds = {}
    files = sorted(source.glob("trace.*"))
    if not files:
        raise RuntimeError(
            "no trace files; see run.log; missing evidence cannot be counted as zero"
        )
    with (args.output / "calls.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames="pid sample observations phase file_kind start_us end_us call path requested_bytes returned_bytes status".split(),
        )
        writer.writeheader()
        for file in files:
            with file.open() as lines:
                for line in joined_lines(lines, totals):
                    syscall = re.search(
                        r"\b(openat|read|pread64|readv|preadv|preadv2|mmap|io_uring_\w+)\(",
                        line,
                    )
                    if not syscall:
                        continue
                    call = parse_call(line)
                    if call is None:
                        totals["unparsed"] += 1
                        totals[f"unparsed_{syscall[1]}"] += 1
                        continue
                    if not any(
                        call["path"] == prefix.rstrip("/")
                        or call["path"].startswith(prefix.rstrip("/") + "/")
                        for prefix in args.path_prefix
                    ):
                        totals["outside_filter"] += 1
                        continue
                    attribution = attribute_call(call, windows)
                    suffix = Path(call["path"]).suffix.lower()
                    kind = {
                        ".puffin": "puffin",
                        ".parquet": "parquet",
                        ".avro": "metadata",
                        ".json": "metadata",
                        ".text": "metadata",
                    }.get(suffix, "other")
                    writer.writerow(
                        dict(call, pid=file.suffix[1:], file_kind=kind, **attribution)
                    )
                    key = ":".join(
                        attribution[k] for k in ("observations", "sample", "phase")
                    )
                    phase = phases.setdefault(key, Counter())
                    by_kind = file_kinds.setdefault(f"{key}:{kind}", Counter())
                    by_kind[call["call"]] += 1
                    by_kind[call["status"]] += 1
                    by_kind["requested_bytes"] += call["requested_bytes"]
                    by_kind["returned_bytes"] += call["returned_bytes"]
                    phase[call["call"]] += 1
                    phase[call["status"]] += 1
                    phase["requested_bytes"] += call["requested_bytes"]
                    phase["returned_bytes"] += call["returned_bytes"]
                    totals[call["call"]] += 1
                    totals[call["status"]] += 1
                    totals["requested_bytes"] += call["requested_bytes"]
                    totals["returned_bytes"] += call["returned_bytes"]
    report = {
        "command": command,
        "returncode": returncode,
        "trace_source": str(source.resolve()),
        "path_prefixes": args.path_prefix,
        "file_syscalls": dict(totals),
        "backend_physical_io": "unobserved",
        "file_syscalls_by_sample_phase": phases,
        "file_syscalls_by_sample_phase_file_kind": file_kinds,
        "file_kind_attribution": "filename suffix only; does not identify the caller or I/O backend",
        "phase_attribution": "query windows; preparation/execution overlap is not separated",
        "coverage": "openat/read/pread64 only; file syscalls are not storage-device reads",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if returncode:
        raise SystemExit(
            f"traced command failed ({returncode}); retain logs for diagnostics only"
        )


if __name__ == "__main__":
    main()
