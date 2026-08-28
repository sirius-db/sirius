# 8-CN SF3000 on gcn-09 + gcn-18

Live knobs: [`env.sh`](env.sh). What we measured and threw away:
[`TUNING-DISCOVERY.md`](TUNING-DISCOVERY.md).

Dataset `/scratch/sirius/datasets/tpch_sf3000` is 1.2 T on GPFS. Occupancy
112+64+0.76 = 176.8 / 185 GiB ≈ 96 %. `pipeline_dop=18` filled 64 GiB of arena
on q08 (~280 leases) and then pool-OOM'd. Common queries close at dop=12.
q08/q18 close at dop=9. q09/q21 still refuse at dop=9.

| Knob | Common | Heavy (q08 q09 q18 q21) |
|---|---|---|
| GPU_MEM | 112GiB | 112GiB |
| STAGING | 64GiB | 64GiB |
| HOST_MEM | 16GiB | 16GiB |
| pipeline_dop | 12 | 9 |
| datasource | uring (`true`) | same |
| arena | fabric | same |
| UCX_TLS | cuda_copy,cuda_ipc,tcp,self | same |

```bash
SCALE_FACTOR=3000 ./configs/gb200-8gpu/relaunch.sh   # on gcn-18
./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-common.txt)
# if that arm left the cluster clean:
mysql -h127.0.0.1 -P9030 -uroot -e "SET GLOBAL pipeline_dop=9;"
PIPELINE_DOP=9 ./bench/gb200-8gpu/sweep.sh 3000 $(cat bench/gb200-8gpu/sf3000/queries-heavy.txt)
```

Restart both hosts after any arena refuse. q08 and q18 closed at dop=9. q09 and
q21 fill every arena; that is PLAN-01, not another split. This fleet is two
4-GPU nodes. Do not start SF10000.
