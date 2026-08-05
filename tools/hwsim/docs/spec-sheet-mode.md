# Spec-sheet target mode (WS19) — predicting hardware you do not have

**Status: implemented 2026-08-05** (`sim/hwsim/{descriptor,target,target_cli}.py`,
descriptor library `tools/hwsim/hw-descriptors/`, 25 new unit tests). Motivated by the
[RTX PRO 6000 external validation](external-validation-rtx-pro-6000.md): the physics split
transfers across GPU arch / CPU arch / link class with zero recalibration, so the remaining
obstacle to predicting an *unseen* machine is turning its **advertised spec sheet** into a
knob vector. This mode does that derivation automatically:

```bash
python -m hwsim simulate <trace> --query-label L \
    --physics p.json \
    --target tools/hwsim/hw-descriptors/rtx-pro-6000-blackwell.yaml \
    [--source tools/hwsim/hw-descriptors/gb300.yaml]
python -m hwsim sweep <trace> --query-label L --target t.yaml --sweep gpu_mem_capacity=0.25,0.5,1
```

Output = a **target-mode header** (derived knob table with per-knob provenance and
confidence tier), the **standard report** at the *derated-nominal* vector, and a
**PREDICTION BAND** footer with the walls at both the derated-nominal and the
*advertised-optimistic* vectors. Explicit `--knob` values override the derived value in
both vectors; `--sweep` values override per point (target-grids). Without `--target`,
simulate/sweep are byte-identical to before.

## 1. Descriptor schema

One YAML file per machine, ADVERTISED values plus an optional `measured:` block for boxes
we have benchmarked with the throttle kit. Parsed by a strict stdlib YAML-subset parser
(nested maps, scalars, `#` comments — no lists/anchors; violations raise with a line
number). Unknown keys are rejected (typo safety).

```yaml
name: my-box
gpu:
  name: "..."                    # informational
  sm_count: 188                  # int
  boost_clock_mhz: 2617
  fp32_tflops_peak: 126          # advertised dense FP32; default = SMs x 128 x 2 x clk
  mem_class: gddr7               # hbm3e|gddr7|hbm3|hbm2e|gddr6|... (selects the derate)
  mem_bandwidth_gbs_peak: 1792
  vram_gb: 96
  l2_mb: 128                     # informational (not used in derivation yet)
link:                            # the HOST<->GPU channel Sirius H2D uses
  type: pcie                     # pcie | c2c | nvlink
  gen: 5                         # pcie only; per-dir peak = {3:1, 4:2, 5:4, 6:8} GB/s x lanes
  lanes: 16
  # gbs_peak_per_dir: 450        # explicit override (required for c2c/nvlink)
  # grace_colimit: false         # optional platform-law override (default: type == c2c)
cpu:
  arch: x86_64
  cores: 48
  base_clock_ghz: 3.1            # boost preferred when both present
  boost_clock_ghz: 3.4
  dram_gbs_peak: 512             # or dram_channels + dram_speed_mts (x8 B)
storage:
  seq_read_gbs: 14.5             # advertised sequential read
measured:                        # optional; throttle-kit conventions ONLY:
  fp32_tflops: 87.2              #   fma victim TFLOP/s
  gpu_mem_gbs: 1471              #   CE D2D copy, 2x-bytes traffic accounting
  link_h2d_gbs: 57.7             #   pinned H2D loop, per direction
  link_d2h_gbs: 57.3
  cpu_dram_gbs: 77.8             #   8-thread memcpy, 2x bytes
  storage_seq_read_gbs: 14.6     #   8-stream O_DIRECT sequential
  vram_usable_gb: 97.9           #   driver-reported total
```

**Library** (`tools/hwsim/hw-descriptors/`): `gb300.yaml` and
`rtx-pro-6000-blackwell.yaml` carry `measured:` blocks from our two benchmarked
platforms; `h100-sxm5.yaml`, `a100-sxm4-80g.yaml`, `l40s.yaml` are advertised-only from
NVIDIA datasheets/whitepapers (sources in file comments; cpu/storage are box-dependent —
copy and fill in for a concrete target).

## 2. Source characterization (where the "1.0 side" comes from)

Priority per resource: **measured-in-trace > `--source` descriptor `measured:` >
derived-from-advertised**.

- **Trace G6** ([ws9-new-fields.md §4](ws9-new-fields.md)): fresh traces carry the source
  GPU's `sm_count`, `sm_clock_khz`, `mem_clock_khz`, `mem_bus_width_bits`, host cores —
  read directly from `<session>/engine/*.ndjson` (spec-class values; the CUDA-derived
  membw peak `2 x mem_clk x bus_width` can sit below marketing: GB300 7.16 vs 8.0 TB/s).
  Old traces lack these → pass `--source <descriptor>`. If both are present and disagree
  >2% on SMs x clock, the trace wins and a warning asks whether the descriptor is for the
  right box.
- **Physics profile** (`--physics`): the nsys wire-side H2D peak
  (`diagnostics.channel_peak_gbps`) beats any spec for the source link, gated at the
  1 TB/s physical ceiling so coherent-C2C artifact rates are never used.
- The `--source` descriptor's `measured:` block supplies the throttle-kit anchors
  (fma, CE membw, DRAM, storage) that no trace carries.

## 3. Derating table — advertised → achievable, with sources

Anchored on the only two platforms we have measured; **not tunable**. Where the two
anchors disagree for a class, unanchored classes get the range (note emitted per knob).

| resource class | derate (advertised → achievable) | anchor & source |
|---|---|---|
| GPU membw, `hbm3e` | **0.702** | GB300: 5619 GB/s CE-measured / 8000 advertised ([membw-throttle.md](membw-throttle.md); NVIDIA Blackwell Ultra 8 TB/s/GPU) |
| GPU membw, `gddr7` | **0.821** | RTX PRO 6000: 1471 / 1792 ([external report](external-validation-rtx-pro-6000.md); NVIDIA datasheet) |
| GPU membw, other (`hbm3`, `hbm2e`, `gddr6`, …) | 0.76, range **[0.70, 0.82]** | no anchor — spans the two above |
| FP32 fma | 0.67, range **[0.648, 0.692]** | GB300: 52.16 / 80.6 theoretical ([compute-throttle.md](compute-throttle.md)); RTX: 87.2 / 126 advertised. The ~7% spread IS the cross-arch error floor of clock x SM scaling |
| link `pcie` | **0.902** (d2h 0.895) | RTX box: 57.7 / 64.0 (gen5 x16) |
| link `c2c` | **0.851** (d2h 0.829) | GB300: 383 / 450 per dir (NVIDIA NVLink-C2C 900 GB/s bidir) |
| link `nvlink` (as host link) | 0.851, range [0.83, 0.90] | **no anchor** — borrowed; order-of-magnitude |
| host DRAM | 0.38, range **[0.30, 0.55]** | Grace only: 196 (8-thr memcpy, 2x bytes) / 512 advertised LPDDR5X. Thread/NUMA-bound measure, not STREAM — wide band; the RTX box's DIMM config is unknown so x86 cannot anchor |
| NVMe seq read | 0.91, range [0.85, 1.0] | GB300: 6.525 8-stream / 7.167 QD32 ceiling ([io-throttle.md](io-throttle.md)); OEM drive has no public spec |
| VRAM capacity | 1.0 (driver-usable preferred) | GB300 advertises 288 GB, driver reports 269.2; RTX advertises 96, reports 97.9 |

Measured-value conventions (a `measured:` block MUST use the same kit):
fma = register-only FMA victim; membw = CE D2D copy counting read+write traffic
(the right basis against advertised bandwidth); link = pinned-copy payload per
direction; DRAM = 8-thread memcpy at 2x bytes; storage = 8-stream O_DIRECT.

## 4. Knob derivation rules

Every knob is `target_achievable / source_achievable`, resolved per §2/§3:

| knob | nominal | advertised-optimistic | notes |
|---|---|---|---|
| `gpu_compute` | measured fma ratio; else advertised(/theoretical) FP32 x 0.67 vs source achievable | target advertised FP32 x **0.692** (best anchor) / source achievable | full advertised FP32 is a theoretical peak no silicon sustains — the optimistic edge uses the best measured anchor fraction instead. **Cross-check**: SMs x boost_clock ratio vs FP32 ratio; >15% apart ⇒ loud arch-IPC warning (A100's 64 FP32 lanes/SM disagrees ~2x; the FP32 ratio wins) |
| `gpu_mem_bandwidth` | derated/measured BW ratio | target advertised / source achievable | ratio exactly 1.0 keeps the v0 `None` default (numerically identical; required for the consistency guarantees) |
| `gpu_mem_capacity` | usable-VRAM ratio | advertised-VRAM ratio | **pool-fraction convention**: assumes the same `usage_limit` fraction on both boxes. Different convention (e.g. the RTX box's absolute `usage_limit_bytes: 80GB`) ⇒ override `--knob gpu_mem_capacity=<target_pool/source_pool>` |
| `c2c_bandwidth` | derated/measured per-dir link ratio (source side: nsys wire peak > measured > derated) | target advertised link / source achievable | cross-link-class note when source/target classes differ |
| `io_bandwidth` | storage seq-read ratio | advertised / achievable | unresolved (missing storage section) ⇒ 1.0 + warning |
| `cpu_mem_bandwidth` | derated/measured DRAM ratio | advertised / achievable | co-limit input only |
| `cpu_compute` | cores x clock ratio (cores-only + warning when clocks unknown) | same | **UNVALIDATED, loud** — no physical validation on any platform |

**Platform law (descriptor-driven)**: the Grace C2C↔DRAM co-limit
(`min(c2c_bandwidth, cpu_mem_bandwidth)`, [laws.py](../sim/hwsim/physics/laws.py) law 2)
applies only when the **target** `link.type` is `c2c` (override with
`link.grace_colimit`). A PCIe target gets `c2c_bandwidth` alone — matching the external
report's measurement ("No Grace co-limit" on the RTX box). The mild PCIe⇄DRAM coupling
seen there (dram eater drags h2d 57.7→33.9) is NOT modeled.

**Band semantics**: *nominal* = best-available on both sides (measured > derated);
*optimistic* = the target hits its spec sheet while the source stays at its achieved
values. The optimistic bound therefore does **not** collapse to 1.0 at target==source —
your own box does not achieve its spec sheet either. The two consistency guarantees hold
for the nominal vector (unit-tested byte-compares, `tests/test_target.py`):
`--target X --source X` ⇒ all knobs 1.0 ⇒ report identical to plain simulate; gb300 with
SMs+FP32 halved ⇒ identical to `--knob gpu_compute=0.5`.

## 5. Confidence tiers (printed per knob in the header)

Propagated from [validation-results.md](validation-results.md) /
[suite-whatif-sf1000.md](suite-whatif-sf1000.md):

- `gpu_compute` ≤1: **validated** with `--physics` (G4b suite median +6.8%, 17/22 within
  ±15%); >1: **order-of-magnitude** (speedup cells need the [G4b, v0] band, §9.5).
- `gpu_mem_bandwidth`: **validated** with `--physics` on host-dominated lanes (E4 rescore
  −2.8/−5.3% medians); v0 = pessimistic roofline.
- `gpu_mem_capacity`: **validated** above the spill knee (≤0.5%); **order-of-magnitude
  (±40%)** below it.
- `c2c_bandwidth`: **validated** on link-bound lanes with `--physics` + wire cap
  (−9.1/−1.5%); **INERT** on coherent-C2C traces (the sanity warning fires).
- `io_bandwidth`: **validated ±5%** (≤1) on scan-bound cold lanes; decode-heavy ≤+20%
  pessimistic; INERT on GPU-pinned lanes; >1 = optimistic bound.
- `cpu_mem_bandwidth`: co-limit input only; **UNVALIDATED** standalone.
- `cpu_compute`: **UNVALIDATED** anywhere — the loudest warning in the mode; consider
  `--knob cpu_compute=1` to freeze host time.

## 6. Worked example — GB300 SF1000 trace → RTX PRO 6000 prediction

Derived vector (S-NSYS trace + per-query physics profiles, both library descriptors;
run 2026-08-05 on pmgb300ws-0163):

| knob | nominal | optimistic | provenance |
|---|---|---|---|
| gpu_compute | 1.672 | 1.672 | measured fma both sides (87.2/52.16); SMs x clock alt 1.564 |
| gpu_mem_bandwidth | 0.2618 | 0.3189 | measured CE both sides (1471/5619; adv 1792/5619) |
| gpu_mem_capacity | 0.3637 | 0.3566 | usable 97.9/269.2 |
| c2c_bandwidth | 0.163 | 0.181 | measured H2D 57.7 / **nsys wire peak 354** (beats the 383 spec-side anchor) |
| io_bandwidth | 2.238 | 2.238 | measured 14.6/6.525 |
| cpu_mem_bandwidth | 0.397 | 0.397 | measured 77.8/196 |
| cpu_compute | 0.667 | 0.667 | cores-only 48/72, clocks unknown (UNVALIDATED) |

platform law: co-limit **OFF** (PCIe target).

q9 (`--physics`, join 137/137 tasks, 100% busy time): source baseline 855.3 ms →
**nominal 4861.1 ms (+468%), optimistic 4360.5 ms (+410%)**, binding `gpu_device`.
On the v0 path (S-BASE trace) the same query lands 4696.7/3851.5 ms with binding
`gpu_memory` (the 86 GiB pool at 99% + 2 downgrades) — the two paths agree the RTX card
is ~5-6x slower on SF1000 q9 but disagree on the mechanism; trust the physics path.
Full 22-query suite (`experiments/bin/predict_cross_machine.py`): source baseline 6.34 s
→ **nominal 24.7 s (3.9x), optimistic 20.9 s (3.3x)**. Untested prediction — grading it
is exactly the [cross-machine experiment](cross-machine-experiment.md).

## 7. Limits (honest)

1. **Cross-arch kernel behavior beyond clock x SM ratios is not modeled**: occupancy
   cliffs, L2-size effects (the RTX's 128 MB L2 makes some GB300-membw-bound kernels
   L2-resident — the membw kit itself measured 3 PB/s on an L2-sized buffer there),
   different kernel selection by cudf on sm_120 vs GB300. The fma-anchor spread (~7%)
   is the *floor*, not the ceiling, of compute-transfer error.
2. **`cpu_compute` is unvalidated** and host time dominates NVMe-lane spans (85-93%,
   §7.1) — on host-dominated lanes the cross-machine wall is largely scaled by the
   weakest knob. Run predictions both with the derived value and with
   `--knob cpu_compute=1` to see the exposure.
3. **Scheduler/config differences are not descriptor-driven**: executor threads, pool
   fraction (`usage_limit`), scan-manager settings, engine version replay from the
   SOURCE trace. Capture the source trace with a config matching the target box's
   (or override the capacity knob) — see the cross-machine experiment doc.
4. **Spill-regime predictions inherit the G5 tier** (order-of-magnitude): a target with
   1/3 the VRAM at SF1000 (the worked example) sits partly below the spill knee.
5. Derates for unanchored classes are two-point interpolations; a third measured
   platform (the cross-machine experiment's byproduct) would tighten every range.
6. The optimistic edge is not a statistical upper bound — it is "target achieves spec";
   real machines occasionally beat one component's nominal while missing another's.
