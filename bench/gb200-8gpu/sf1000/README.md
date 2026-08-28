# 8-CN SF1000 on gcn-09 + gcn-18. Source of truth: `env.sh`.

Occupancy `GPU_MEM + STAGING + 0.76 GiB` = 160.8 / 184 GiB. Dataset 264.5 GiB decimal
on `/scratch/sirius/datasets/tpch_sf1000`. Closed 22/22 at 20260828T035635Z.

| Knob | Value | Why |
|---|---|---|
| GPU_MEM | 128GiB | 4-CN SF1000 pool; HASH_JOIN |
| STAGING | 32GiB | 16 GiB died at q05/q07 |
| HOST_MEM | 112GiB | 4×112 leaves ~509 GiB cache vs 264 GiB tree |
| pipeline_dop | 18 | `avgCores/2` for 36-core CNs. 36 was 2× over |
| datasource | uring (`true`) | match 4-GPU GPFS arm |
| arena | fabric | required cross-host |
| UCX_TLS | cuda_copy,cuda_ipc,tcp,self | no `rc_mlx5` |
| watchdog / RPC | 300 / 300 | 4-CN SF1000 preset |
| QUERY_TIMEOUT / COLD | 1800 / 6000 | `max(90, 1.8×SF)` / `max(300, 6×SF)` |
| q11 FRACTION | 0.000000100000 | 0.0001/1000 |

```bash
SCALE_FACTOR=1000 ./configs/gb200-8gpu/relaunch.sh   # on gcn-18
# then
./bench/gb200-8gpu/sf1000/sweep.sh                   # on gcn-18, cluster already up
```
