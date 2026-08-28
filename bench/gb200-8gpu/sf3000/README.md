# 8-CN SF3000 on gcn-09 + gcn-18. Source of truth: `env.sh`.

Occupancy 156.8 / 184 GiB. Dataset `/scratch/sirius/datasets/tpch_sf3000` is 1.2 T on
GPFS, larger than one box of LPDDR. 4-CN SF3000 used this same 120/36/16 split;
8 CNs should have more arena headroom per node, not less. If a query dies with
`exchange staging arena exhausted`, next arm is `GPU_MEM=112GiB STAGING=44GiB`
(the 4-CN SF10000 arena, occupancy still 156.8).

`pipeline_dop=18` is the 36-core CN formula. Do not use 36.

| Knob | Value | Why |
|---|---|---|
| GPU_MEM | 120GiB | 4-CN SF3000 pool |
| STAGING | 36GiB | 4-CN SF3000 arena; SF1000 8-CN needed 32, not 16 |
| HOST_MEM | 16GiB | 1.2 T > 957 GiB LPDDR; maximize page cache |
| pipeline_dop | 18 | same as SF1000 8-CN |
| datasource | uring (`true`) | GPFS; kvikio mixed the SF1000 A/B |
| arena | fabric | required cross-host |
| UCX_TLS | cuda_copy,cuda_ipc,tcp,self | no `rc_mlx5` |
| watchdog / RPC | 600 / 900 | 4-CN SF3000 preset |
| warmup | 600 s | larger RMM + 7 peers |
| QUERY_TIMEOUT / COLD | 5400 / 18000 | `1.8×3000` / `6×3000` |
| FE query_timeout | 18000 | must be ≥ cold timeout |
| q11 FRACTION | 0.000000033333 | 0.0001/3000 |
| TPCH_DATA | `/scratch/sirius/datasets/tpch_sf3000` | Scale, same path both hosts |

```bash
SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh   # on gcn-18
./bench/gb200-8gpu/sf3000/sweep.sh                   # on gcn-18 after 8 Alive=true
```

A full 22-query close at SF3000 is not bought by these knobs on 4 CN. 8 CN is the
retry, not a guarantee. Restart both hosts after any arena refuse.
