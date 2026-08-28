# Working notes

Dated planning docs, audits, and design diaries. **Not** the source of truth for building,
bringing up a cluster, or running a benchmark.

| Need | Go here |
|---|---|
| Current unresolved work | [`OPEN.md`](OPEN.md) |
| Build the CN stack | `bench/rtxpro6000-2gpu/BUILD-SIRIUS-STARROCKS.md` and the `build-sirius-starrocks` skill |
| Run a TPC-H sweep | `bench/rtxpro6000-2gpu/TPCH-SWEEP-RUNBOOK.md` and the `tpch-cn-sweep` skill |
| Size `GPU_MEM` / `STAGING` / operator budgets | `bench/gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md` (this 4× GB200 box) or `bench/rtxpro6000-2gpu/SIRIUS-TUNING-RUNBOOK.md` (method) and the `cn-tuning` skill |
| Single-host 4× GB200 ops | `.claude/skills/tpch-bench/SKILL.md`, `bench/gb200-4gpu/HARDWARE.md`, `bench/gb200-4gpu/SIRIUS-TUNING-RUNBOOK.md` |
| Two physical GB200 hosts | `experimental/starrocks/benchmarks/2NODE-REPLICATE.md` and the `tpch-2host` skill |
| 8× GB200 (2 hosts × 4). Whole fleet. | [`bench/gb200-8gpu/`](../bench/gb200-8gpu/) (`sf1000/`, `sf3000/`; `sf10000/` is closed, not live) |
| SF3000 8-CN knob log | [`bench/gb200-8gpu/sf3000/TUNING-DISCOVERY.md`](../bench/gb200-8gpu/sf3000/TUNING-DISCOVERY.md) |
| SF3000 session handoff (2026-08-28) | [`2026-08-28-8gpu-handoff.md`](2026-08-28-8gpu-handoff.md) |
| Super Sirius architecture | `docs/super-sirius/` |

Do **not** quote timings, pass/fail tables, or commands from the SF1 / single-GPU L4 notes
(`2026-08-07-sf1-l4/`, `2026-08-05-multi-cn-nixl/`) as current. Those boxes and scales are not
the live campaign.

Folders are named `YYYY-MM-DD-<topic>` after the date the note was written, not the date it was
filed here.
