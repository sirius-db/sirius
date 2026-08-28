# 8-CN SF10000 on gcn-09 + gcn-18

Live knobs: [`env.sh`](env.sh). What we measured:
[`TUNING-DISCOVERY.md`](TUNING-DISCOVERY.md).

Dataset `/scratch/sirius/datasets/tpch_sf10000` is 3.7 T on GPFS. Occupancy
112+64+0.76 = 176.8 / 185 GiB ≈ 96 %. **14/22 with times.** q03/q05/q08/q09/q10/q17/q18/q21
are empty windows. q07 needs 96/80; everything else that closed used 112/64 dop=9
(q01/q02 at dop=12).

| Knob | Common (most) | q07 |
|---|---|---|
| GPU_MEM | 112GiB | 96GiB |
| STAGING | 64GiB | 80GiB |
| HOST_MEM | 16GiB | 16GiB |
| pipeline_dop | 9 | 9 |
| datasource | uring (`true`) | same |
| arena | fabric | same |
| UCX_TLS | cuda_copy,cuda_ipc,tcp,self | same |

```bash
SCALE_FACTOR=10000 GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=9 \
  ./configs/gb200-8gpu/relaunch.sh
GPU_MEM=112GiB STAGING=64GiB PIPELINE_DOP=9 \
  ./bench/gb200-8gpu/sweep.sh 10000 q01 q02 q04 q06 q12 q13 q14 q16 q19 q20 q22
```

Pass explicit memory knobs. Persistent shell exports override `env.sh`. Restart both
hosts after any refuse. Do not run skipped queries on the same cluster as a passing list.
