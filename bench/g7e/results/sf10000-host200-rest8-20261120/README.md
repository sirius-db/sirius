# G7e SF10000 custom host200 / REST8 — November 20, 2026

Measured on Amazon Linux 2023, g7e.8xlarge in us-east-1, using engine
`26b5caba3e2c96290fdc63d4cf40ac42572eaab8` and shared runner `2c48eed1dcbc58a6775b5a51904524f7ab91b5ae`.
This snapshot is separate from the November 11 baseline and saved G7e presets.

Cold uses one fresh process and cache clears per query measurement. Hot uses
one fresh process per query, with two consecutive iterations and LRU eviction.
There is no extra warmup. Hot iteration 1 is the first query execution in its
new process. The only resource-config difference is idle versus LRU eviction.
Both use host200Gi, 8 REST reactors, RA16, pipeline4 and compressed local NVMe spill.
The exact source YAMLs are under `configs/`; worker copies add only Quent.

All 22 queries completed for each column below, with CPU fallback disabled
and GPU-plan evidence. No independent CPU full-suite output comparison was run.

| Query | Cold (s) | Hot 1 (s) | Hot 2 (s) | Hot best (s) |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 56.189 | 57.555 | 55.130 | 55.130 |
| Q2 | 8.916 | 10.716 | 3.366 | 3.366 |
| Q3 | 52.043 | 50.526 | 46.939 | 46.939 |
| Q4 | 29.732 | 31.175 | 26.895 | 26.895 |
| Q5 | 61.329 | 63.106 | 59.111 | 59.111 |
| Q6 | 41.020 | 41.591 | 39.674 | 39.674 |
| Q7 | 66.759 | 66.156 | 65.081 | 65.081 |
| Q8 | 84.679 | 83.724 | 77.754 | 77.754 |
| Q9 | 355.130 | 494.885 | 581.308 | 494.885 |
| Q10 | 61.289 | 61.632 | 55.863 | 55.863 |
| Q11 | 12.987 | 14.509 | 13.674 | 13.674 |
| Q12 | 47.085 | 46.542 | 44.061 | 44.061 |
| Q13 | 260.331 | 350.105 | 327.790 | 327.790 |
| Q14 | 55.171 | 53.379 | 53.489 | 53.379 |
| Q15 | 50.783 | 50.721 | 50.783 | 50.721 |
| Q16 | 9.037 | 8.701 | 4.643 | 4.643 |
| Q17 | 46.062 | 48.044 | 44.688 | 44.688 |
| Q18 | 569.914 | 568.415 | 657.684 | 568.415 |
| Q19 | 70.357 | 70.736 | 70.471 | 70.471 |
| Q20 | 63.651 | 62.798 | 60.200 | 60.200 |
| Q21 | 196.954 | 1466.775 | 89.838 | 89.838 |
| Q22 | 16.838 | 13.271 | 5.621 | 5.621 |
| **SUM** | **2216.256** | **3715.064** | **2434.063** | **2258.198** |

The hot-best sum adds each query's minimum; it is not one sequential suite iteration.
Cold Q9 emitted a compression CUDA allocation error and then completed on GPU.
Hot Q21 took 1466.775056 / 89.838282 s: its first iteration spent a long time
in host-memory downgrade retries that freed zero bytes, then recovered without
interruption. No query was skipped or rerun, and no config was changed.
Full logs and raw Quent are saved locally and on EBS in a private archive.
`archive.json` records its SHA256 and payload inventory; all 1308 payload files
were verified after download.

## Reproduce

Use the exact runner commit and a separately built engine at the commit above.
The source engine directory must be clean and have its recursive submodules initialized.
The native release build used CUDA 13.5 and `120f-real`. The full build log and
locked environment are in the private archive. Run through the engine's locked
Pixi environment; replace the three directory paths below.

```bash
engine_root=/path/to/sirius-26b5caba
runner_root=/path/to/sirius-replay-2c48eed1
configs=/path/to/this-result-snapshot/configs
export AWS_REGION=us-east-1 AWS_DEFAULT_REGION=us-east-1
export SIRIUS_LOG_LEVEL=info
ulimit -Sn 65536
pixi run --manifest-path "$engine_root/pixi.toml" python "$runner_root/bench/common/replay.py" \
  --hardware g7e --source-dir "$engine_root" --sf 10000 --profile cold \
  --config "$configs/cold.yaml" --iterations 1 --worker-timeout 1800 \
  --input s3://sae-dataproc-aws-dev-sirius-scratch-us-east-1/datasets/tpch_sf10000_regenerated/ --output /path/to/new-cold-output
pixi run --manifest-path "$engine_root/pixi.toml" python "$runner_root/bench/common/replay.py" \
  --hardware g7e --source-dir "$engine_root" --sf 10000 --profile hot \
  --config "$configs/hot.yaml" --iterations 2 --worker-timeout 1800 \
  --input s3://sae-dataproc-aws-dev-sirius-scratch-us-east-1/datasets/tpch_sf10000_regenerated/ --output /path/to/new-hot-output
```

The private S3 prefix requires suitable AWS read credentials and an instance in
us-east-1. Verify its generation manifest (868 Parquet files) and region before
running. `/opt/dlami/nvme/spill` must resolve to mounted local EC2 instance NVMe
with at least 1 TiB available. The runner checks region and storage and refuses
silent EBS spill. It does not generate data, format storage, or launch instances.

The reproduction commands do not copy this historical source engine's
performance_test.py: the shared runner supplies the harness and selects the
pinned engine binary via `--source-dir`. See `manifest.json` for exact hashes.
