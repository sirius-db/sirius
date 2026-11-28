#!/usr/bin/env python3
"""Prints the CPUs one Sirius rank runs on: how many, and the list for taskset -c.

  sirius_rank_cpus.py auto|all <the rank's GPU, as nvidia-smi -i takes it>

all: every CPU the rank was started with. auto: a rank started with fewer CPUs than the host has
(Slurm bound its task, taskset, a container) keeps them: whoever started it chose its CPUs, and
ranks started together inside one such set share it. Otherwise the rank takes its GPU's share of
the cores next to that GPU (sysfs local_cpulist of the PCI device), divided in order among the
NVIDIA GPUs with the same local CPUs. Ranks that have the whole host then never share a core,
whoever starts them and whichever GPUs get a rank.
"""

import glob
import os
import subprocess
import sys

HINT = "; cpus=all (SIRIUS_NODE_CPUS) runs the rank on every CPU it was started with"


def cpulist(text):
    cpus = []
    for part in text.strip().split(","):
        if part:
            low, _, high = part.partition("-")
            cpus.extend(range(int(low), int(high or low) + 1))
    return cpus


def ranges(cpus):
    spans = []
    for cpu in sorted(cpus):
        if spans and spans[-1][1] == cpu - 1:
            spans[-1][1] = cpu
        else:
            spans.append([cpu, cpu])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in spans)


def read(path):
    with open(path) as file:
        return file.read().strip()


def core(cpu):
    return cpulist(
        read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
    )[0]


def gpu_share(gpu, allowed):
    query = ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", gpu]
    try:
        smi = subprocess.run(query, capture_output=True, text=True)
    except OSError as error:
        sys.exit(f"cannot run nvidia-smi: {error}" + HINT)
    if smi.returncode != 0 or ":" not in smi.stdout:
        sys.exit(
            f"nvidia-smi does not show GPU '{gpu}': {(smi.stdout + smi.stderr).strip()}"
            + HINT
        )
    # nvidia-smi prints an 8-digit PCI domain, sysfs 4.
    domain, rest = smi.stdout.strip().lower().split(":", 1)
    mine = f"{int(domain, 16):04x}:{rest}"
    gpus = {}
    for device in glob.glob("/sys/bus/pci/devices/*"):
        try:
            if read(device + "/vendor") == "0x10de" and read(
                device + "/class"
            ).startswith("0x03"):
                gpus[os.path.basename(device)] = read(device + "/local_cpulist")
        except OSError:
            pass
    if mine not in gpus:
        sys.exit(f"no NVIDIA GPU at {mine} in /sys/bus/pci/devices" + HINT)
    siblings = sorted(gpu for gpu, cpus in gpus.items() if cpus == gpus[mine])
    cores = {}
    for cpu in cpulist(gpus[mine]):
        if cpu in allowed:
            cores.setdefault(core(cpu), []).append(cpu)
    share = len(cores) // len(siblings)
    if share == 0:
        sys.exit(
            f"{len(cores)} cores next to GPU {mine} for {len(siblings)} GPUs" + HINT
        )
    first = siblings.index(mine) * share
    return [cpu for c in sorted(cores)[first : first + share] for cpu in cores[c]]


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("auto", "all"):
        sys.exit(f"cpus is auto or all, not '{' '.join(sys.argv[1:2])}'")
    allowed = os.sched_getaffinity(0)
    if sys.argv[1] == "all" or len(allowed) < os.cpu_count():
        cpus = allowed
    else:
        cpus = gpu_share(sys.argv[2], allowed)
    print(len(cpus), ranges(cpus))


if __name__ == "__main__":
    main()
