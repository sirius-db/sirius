# GB300 device-memory spill sweep

Run one unpinned TPC-H query per process using the saved GB300 cold config.
Set a fixed host cap of **384 GiB**, then sweep
`sirius.memory.gpu.usage_limit_bytes` and the Quent output path. The GPU caps
are **decimal GB**: 32, 64, 128, and 256 billion bytes. Decimal units allow
the 256 GB point to fit on the GB300's approximately 250.7 GiB of physical
device memory; 256 GiB would exceed it. The GPU
reservation fraction remains 0.9, downgrade trigger/stop remain 0.8/0.6,
and spill compression remains enabled. The host cap is fixed for all GPU caps.
An initial attempt with the original 450 GiB host cap
drove physical host RAM near exhaustion at GPU 16 GB Q3, and the tmux session
disappeared twice. At a fixed 384 GiB host cap, GPU 16 GB Q3 spilled heavily
and was stopped before completion. The 384 GiB limit leaves headroom while
keeping the GPU cap as the swept variable. The
host initial pool count scales from 225 to 192 to retain the same preallocation
ratio. No `--profile` override is used.

From the repository root:

```bash
tmux new-session -d -s gb300-device-spill-sf10k \
  'bash bench/gb300-device-spill-sweep/run_sf.sh 10000'
```

Progress is in
`test_datasets/gb300-device-spill-sweep/host384Gi/sf10000/progress.log`.
An `exit-code` file is written at completion. The runner saves each effective
YAML, command, benchmark output, runner log, raw Quent trace, and result under
`test_datasets/gb300-device-spill-sweep/host384Gi/sf10000/gpu<CAP>GB/q<NN>/`.
It updates `per-query.csv` and `totals.csv` after every query, skips completed
queries on rerun, and accepts `--retry-failed` to retry failures. Each process
starts with an empty Sirius cache. The benchmark requests that the OS discard
cached dataset pages, but a fully cold page cache is not guaranteed.

The three counters are cumulative one-way DataBatch conversions during each
query: GPU→HOST, HOST→DISK, and GPU→DISK. A move is counted at the destination's
representation size, so compression can change its byte count. They are not
peak memory occupancy. Failed queries have no movement totals and are excluded
from the aggregate; compare successful-query counts before comparing caps.
The measured results and per-query files are linked from [RESULTS.md](RESULTS.md).
