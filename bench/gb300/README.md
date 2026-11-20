# GB300 benchmarks

Use the [shared replay entry point](../README.md) with this hardware's
[configs](configs/) and [selected results](results/README.md).

```bash
bash bench/gb300/replay.sh 1000 cold --dry-run
bash bench/gb300/replay.sh 1000 hot --dry-run
bash bench/gb300/replay.sh 1000 pinned --dry-run
```

GB300 replay requires CDMM (`driver`) mode. New cold runs start a fresh Sirius
process for every query iteration and clear both the local dataset-file OS
cache and Sirius cache before timing. They follow suite order, Q1(1) through
Q22(1), then Q1(2) through Q22(2). SF10000 cold applies its separate Q10 config
in that order without grouping Q10's iterations in one process.

Hot keeps one process for the suite and groups iterations by query; SF10000
hot runs Q10 separately with its own config. The selected historical cold
measurements reused one process and are unchanged: the new cold protocol is
not a rerun of those timings. The pinned protocol uses per-query host pinning inside one
session, with explicit compression plans only for `lineitem` and `orders`.
There is no successful selected SF10000 pinned run.

The existing GB300 material was moved from `bench/gb300-*` into this directory.
Configs, timing values, figures and measured source revisions are preserved.
Manifest repository paths and script paths now use the new layout; historical
on-machine paths and original hashes remain provenance.

Additional studies:

- [Host-cap spill sweep](spill-sweep/README.md)
- [GPU-cap spill sweep](device-spill-sweep/README.md)
- [Cumulative ablation](ablation/README.md)
- [Scale figures](scale-figures/README.md)
- [Concurrency and ablation figures](analysis-figures/README.md)
- [Spill figures](spill-figures/README.md)
