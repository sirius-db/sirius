# G7e SF1000 compressed-first cumulative ablation - November 23, 2026

London g7e.8xlarge, Sirius main `26b5caba`, identical native binary, guide host-pinned preset and `SET threads=24`. Begin with compressed host pinning alone, then add type narrowing, BUILD_PROBE, dynamic filtering and fused scan/filter. Only lineitem/orders are pinned compressed with the same checked Simpatico plans in every stage. Every other table pin is explicitly uncompressed. The baseline is measured again here; no failed all-off stage is rerun.

One fresh engine per stage, Q1-Q22 once, pin/unpin per query and cache clear at stage startup only. No warmup or harness profile/mode override; YAML is preserved. Timings exclude pin/unpin and independent CPU validation. Other engine settings remain fixed.

| Stage | Optimization added | Completed | Full Q1-Q22 sum (s) | Change from prior | Partial sum (s) |
|---:|---|---:|---:|---:|---:|
| 0 | Compressed pinning only | 22/22 | 53.356745 | - | - |
| 1 | + Type narrowing | 22/22 | 46.469744 | -12.91% | - |
| 2 | + BUILD_PROBE | 22/22 | 41.308729 | -11.11% | - |
| 3 | + Dynamic filters | 22/22 | 30.712089 | -25.65% | - |
| 4 | + Fused filtering | 22/22 | 29.408593 | -4.24% | - |

All 110 completed timed answers passed independent CPU, GPU, pin and config validation. Quent retains 110 timed query executions in five stage contexts. There are no extra diagnostic runs. Failed or incomplete stages have no full-suite total.

These are conditional cumulative increments from one pass each. They do not measure independent main effects or establish that enabling features in this order makes an identical final configuration faster. The earlier all-off case failed while pinning Q1, so no full-suite speedup over all-off can be calculated.

## Identical configurations across runs

| Configuration | Previous full-suite sum (s) | New full-suite sum (s) |
|---|---:|---:|
| compressed_pin_only | 51.898669 | 53.356745 |
| all_except_fused | 30.713166 | 30.712089 |
| all_on | 29.425129 | 29.408593 |

The corresponding saved YAMLs match byte for byte, with identical per-call pin policies. Single-pass differences include run-to-run variation.

Every staged local NVMe input hash matched the same SF1000 generation manifest. The live S3 bucket and EC2 regions both matched London; inventory and manifest were unchanged afterward. Host-pin setup uses this verified local mirror because the pinned engine cannot pin S3 directly. The engine was not rebuilt. The unchanged guide preset leaves disk spilling disabled; no EBS fallback was added.

[Stage configs](configs/), [session SQL](session.sql), [feature plan](stage-plan.json), [pin audit](pin-audit.json), [native compression evidence](compressed-chunks.json), [per-query timings](per-query.csv), [sums](summary.csv), [full proofs](summary.json), [validation](validation.json) and [provenance](manifest.json) are saved here. All native logs, CPU/GPU answers and Quent traces remain in the locally retained archive identified by size and SHA256 in the manifest. Every payload hash was verified after download. Earlier results remain unchanged. The instance remains running.

## Reproduction command

The [study wrapper](reproduce.sh) invokes the measured [ablation runner](../../run.py)
with `--order compressed-first`, the recorded CUDA environment and London S3
provenance. It uses the instance's attached role. Run it in detached tmux on a
London g7e.8xlarge after building the pinned engine and preparing the verified
local dataset and independent CPU references:

```bash
bash bench/g7e/ablation/results/20261123-compressed-first/reproduce.sh \
  /home/ec2-user/sirius-26b5caba \
  /mnt/datasets/pinned-inputs/sf1000 \
  /path/to/independent-cpu-references \
  /path/to/new-output
```

The wrapper was added after measurement for convenience. The exact original
launch script, executed runner sources, CPU references, logs and Quent traces
remain in the verified archive. The wrapper runs the same five stages without
changing the saved configs, stopping the instance or adding diagnostic queries.
