# 8× GB200 (2 hosts × 4 GPUs)

Two `presto-gb200-gcn-*` boxes, 4 Sirius CNs each, FE on gcn-18.

Per-scale knobs are the source of truth:

| SF | Folder | GPU_MEM / STAGING / HOST_MEM | Status |
|---|---|---|---|
| 1000 | [`sf1000/`](sf1000/) | 128 / 32 / 112 | measured 22/22, 1.44× vs 4-GPU GPFS |
| 3000 | [`sf3000/`](sf3000/) | 120 / 36 / 16 | extrapolated from 4-CN SF3000 + SF1000 8-CN staging lesson |
| 10000 | [`sf10000/`](sf10000/) | 112 / 44 / 16 | extrapolated; 4-CN was probe-only |

```bash
SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh   # gcn-18, SSHs to 09
./bench/gb200-8gpu/sweep.sh 3000
```

Do not launch with `cluster4-numa.sh` or `benchmarks/cluster8.sh`. Method:
[`SIRIUS-TUNING-RUNBOOK.md`](SIRIUS-TUNING-RUNBOOK.md).
