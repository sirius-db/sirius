# G7e SF1000 host-pin cumulative ablation

The November 23 [BUILD_PROBE-first results](results/20261123/RESULTS.md) and
[type-narrowing-first results](results/20261123-type-first/RESULTS.md) preserve
complete stages, partial sums and allocation failures separately.
The [all-on except compressed pinning case](results/20261123-all-on-uncompressed/RESULTS.md)
keeps the other four switches enabled and makes every pin explicitly uncompressed.
The [compressed-pinning-only case](results/20261123-compressed-pin-only/RESULTS.md)
enables compressed host pins while explicitly disabling the other four switches.

This supports the corrected cumulative orders from the
[GB300 study](../../gb300/ablation/RESULTS.md#corrected-build_probe-first-comparison)
on the existing G7e guide SF1000 host-pinned benchmark. The engine stays at main
`26b5caba`; the [saved pinned preset](../guide-26b5caba/configs/cpu-budget-pinned.yaml)
and `SET threads=24` remain fixed. GB300 and G7e have different GPUs, host memory,
and engine revisions; their absolute timings are not a controlled hardware-only
comparison.

The six stages start with all five features off, then cumulatively enable
BUILD_PROBE admission, dynamic filtering, type narrowing, compressed host
pinning, and fused filtering during decode. BUILD_PROBE is disabled with both
build/probe byte limits set to zero; enabling it restores the pinned engine's
defaults. Every other setting stays fixed.

Pass `--order type-first` for the corrected alternative GB300 order: all off,
type narrowing, BUILD_PROBE admission, dynamic filtering, compressed pinning,
then fused filtering. The default is `--order build-first`. All other settings,
pin controls and process boundaries are identical. A new output directory
preserves the earlier order's measurements.

Pass `--all-on-uncompressed` to run a single additional case: type narrowing,
BUILD_PROBE, dynamic filtering and fused scan/filter enabled, with every host
pin explicitly using `compression=false` (GB300 stage 7). Its effective YAML
matches the all-on compressed stage; only the per-call pin compression policy
differs. The fused switch remains enabled, but there is no compressed pinned
input for fused decode to operate on. Use a new output directory. Compare only
matching completed queries if raw pins exceed host capacity.

Pass `--compressed-pin-only` for a single stage with only compressed host
pinning enabled (GB300 stage 6). Type narrowing, BUILD_PROBE, dynamic filtering
and fused scan/filter are explicitly disabled. This uses the all-off stage's
YAML, with only `lineitem` and `orders` pinned using `compression=true` and the
same checked Simpatico plans; every other table pin is explicitly uncompressed.
The two standalone case flags are mutually exclusive.

As on GB300, each stage starts one fresh engine, clears caches at startup, and
runs Q1–Q22 once. Each query pins only its referenced host columns, executes,
then unpins. Pin/unpin time is excluded; no warmup runs. There are no resets
between queries within a stage. The adapter uses the harness's inert default
grouped ordering without `--profile` or `--mode` overrides, so YAML is not
rewritten. This stage-level lifecycle is specific to the ablation; the regular
G7e hot/pinned reproducer still starts a fresh engine for each query pair.

Every pin call explicitly states `compression=false` in stages 0–3. In stages
4–5, only `lineitem` and `orders` use `compression=true` and the same checked
Simpatico plans as GB300. All other tables remain uncompressed. SQL and native
pin evidence are saved. The global pin-compression setting stays false to
prevent accidental automatic compression.

The native engine cannot pin S3 inputs at this revision, so pin setup uses
byte-identical SF1000 files previously staged from the same-region London S3
bucket onto local NVMe. Before timing, the runner verifies the live bucket
region, manifest/inventory, and every staged file SHA256. Host-pinned queries
then read resident data. Any enabled spill path must resolve to local NVMe;
the exact guide preset leaves disk spilling disabled. No EBS fallback is added.

```bash
pixi run --locked python bench/g7e/ablation/run.py \
  --source-dir /path/to/built-sirius-26b5caba \
  --input s3://same-region-bucket/datasets/tpch_sf1000_regenerated/ \
  --host-input /mnt/datasets/pinned-inputs/sf1000 \
  --references /path/to/independent-cpu-answers \
  --output /path/to/new-ablation-output
```

`--dry-run` prints the selected protocol without AWS or GPU access. Real runs
belong in detached tmux. Each stage has the GB300 study's 600-second wall limit,
including pinning. Errors and partial timings are retained, and later stages
continue. A failed or incomplete stage never receives a full-suite sum.
Independent CPU checks follow all timed stages; Q3/Q18 permit legal LIMIT ties.
The output includes exact source/effective YAMLs, session SQL, pin audits,
events, answers, full native logs, Quent profiles, per-query CSV and stage sums.

After an interrupted collection, `--resume` reuses every ended worker, including
failed ones, rather than repeating timings. It checks the original verified
dataset manifest and inventory, rechecks local data hashes, and verifies retained
stage configs first. An
existing live or unfinished worker is not silently restarted. Separate untimed
path checks use `diagnose.py --source-dir <engine> --run <output>
--host-input <local-data>`; their timings never enter the stage sums.
