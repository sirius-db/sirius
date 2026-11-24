# G7e.8xlarge benchmarks

G7e publishes one configuration family: the **guide-derived presets** used by
our selected `26b5caba` London results. The reproducer selects the appropriate
guide preset by scale; there is no custom-config choice or `--config` override.

| Scale | Guide preset | Session threads |
| --- | --- | ---: |
| SF100 / SF300 | CPU-budget | Engine default |
| SF1000 | CPU-budget | 24 |
| SF3000 / SF10000 | SF10000 balanced-large-scans | 24 |

Cold preserves the measured guide YAML. Hot uses its corresponding LRU copy;
for SF3000/SF10000, eviction is the only YAML difference. Host-pinned uses the
hot YAML with explicit per-query host pinning. Exact files and detailed
instructions are in [guide-26b5caba](guide-26b5caba/README.md).

## Reproduce

Use a separate engine checkout built at
`26b5caba3e2c96290fdc63d4cf40ac42572eaab8` and same-region S3 data:

```bash
bash bench/g7e/replay.sh 3000 cold \
  --source-dir /path/to/built/sirius-26b5caba \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --iterations 1 --dry-run

bash bench/g7e/replay.sh 3000 hot \
  --source-dir /path/to/built/sirius-26b5caba \
  --input s3://YOUR_SAME_REGION_BUCKET/datasets/tpch_sf3000_regenerated/ \
  --iterations 2 --dry-run
```

Remove `--dry-run` to execute. Both this entry point and `bench/replay.sh g7e`
invoke the same reproducer. The wrapper's profile chooses cold/hot/host-pinned
behavior; it passes `--mode` to `performance_test.py`, not the harness's
`--profile` option. Source YAML is preserved; effective copies add only Quent.

Cold uses a fresh engine and cleared OS/Sirius caches for every measurement.
Hot and pinned use a fresh engine per query, retaining it and its cache between
the two measurements. There is no extra warmup. The reproducer verifies the
actual EC2 and bucket regions, dataset manifest, kTLS and local NVMe spill
storage. It never provisions, stops or terminates instances. Pinned additionally
requires an existing byte-verified local mirror; see the detailed instructions.

## Results

The [guide result snapshot](results/guide-ktls-20261123/README.md) contains the
best complete CPU-validated sums from SF100 through SF10000, both contributing
iterations, exact config hashes and retained archive checksums. SF3000 cold is
362.445844 s; hot sums are 371.765623 / 273.535501 s.

The older custom presets and historical custom-config snapshots are retained
locally outside this guide-only publication. The selected table and reproducer
refer to the same guide-derived configuration family and process protocol.
