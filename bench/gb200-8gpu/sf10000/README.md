# 8-CN SF10000 on gcn-09 + gcn-18. Source of truth: `env.sh`. Extrapolated.

Occupancy 156.8 / 184 GiB. Dataset `/scratch/sirius/datasets/tpch_sf10000` is 3.7 T
on GPFS. 4-CN SF10000 is 112/44/16. 8-CN SF1000 did not get to cut staging (32 GiB
required), so this arm keeps 44 GiB. Next arm if arena-exhausted: there is no more
HBM. Drop GPU_MEM and raise STAGING only if occupancy stays ≤ 160 GiB, e.g.
`GPU_MEM=100GiB STAGING=56GiB`.

The 4-CN runbook calls SF10000 probe-only. 8 CN is more total HBM and less shuffle
per node. It is not a promise the 22-query suite closes.

| Knob | Value | Why |
|---|---|---|
| GPU_MEM | 112GiB | 4-CN SF10000 pool |
| STAGING | 44GiB | 4-CN SF10000 arena; do not halve |
| HOST_MEM | 16GiB | 3.7 T >> 957 GiB LPDDR |
| pipeline_dop | 18 | 36-core CN formula |
| datasource | uring (`true`) | GPFS |
| arena | fabric | required cross-host |
| UCX_TLS | cuda_copy,cuda_ipc,tcp,self | no `rc_mlx5` |
| watchdog / RPC | 1800 / 3600 | 4-CN watchdog; RPC registry max |
| warmup | 900 s | 112+44 GiB alloc + 7 peers |
| QUERY_TIMEOUT / COLD | 18000 / 60000 | `1.8×10000` / `6×10000` |
| FE query_timeout | 60000 | must be ≥ cold timeout |
| q11 FRACTION | 0.000000010000 | 0.0001/10000 |
| TPCH_DATA | `/scratch/sirius/datasets/tpch_sf10000` | Scale, both hosts |

```bash
SCALE_FACTOR=10000 ./configs/gb200-8gpu/relaunch.sh
./bench/gb200-8gpu/sweep.sh 10000
```

Restart both hosts after any arena refuse. One cold q21 can run 16 hours at the
harness cap; do not treat a 60000 s cut as a query bug until the FE timeout and
RPC bound are at these values.
