# GB300 benchmarks

Use the [shared replay entry point](../README.md) with this hardware's
[configs](configs/) and [selected results](results/README.md).

```bash
bash bench/gb300/replay.sh 1000 cold --dry-run
bash bench/gb300/replay.sh 1000 hot --dry-run
bash bench/gb300/replay.sh 1000 pinned --dry-run
```

GB300 replay requires CDMM (`driver`) mode. The saved cold/hot protocol uses
one process for the full query suite, with two iterations. Cold resets the
cache before each measurement; hot groups iterations by query. These process
lifetimes preserve the existing GB300 replay and differ from G7e's protocol.
SF10000 uses a separate process and config for Q10, reproducing the saved
composite suite. The pinned protocol uses per-query host pinning inside one
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
