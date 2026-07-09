## Track A / #1014 — findings summary

**Verdict: PASS. Ship `dynamic_filter_build_priority=off` as the default.**

The build-task priority pass has no measurable upside on this workload and a clear
downside. Disabling it is faster on all six queries, far steadier run-to-run, and
leaves peak memory and result correctness unchanged.

All numbers below are relative percentages. Absolute seconds/bytes, machine details,
and artifact paths are in the companion report (`issue-1124-comment.md`).

### Decision: `off` vs `legacy`

| Query | Wall-time Δ (off vs legacy) | Legacy run-to-run spread | Off run-to-run spread | Peak-mem Δ (off vs legacy) |
|---|---:|---:|---:|---:|
| Q5 | −25.3% | 18.2% | 3.1% | <0.01% |
| Q7 | −19.2% | 11.2% | 6.8% | <0.01% |
| Q8 | −10.6% | 24.5% | 3.7% | <0.01% |
| Q9 | −12.0% | 20.4% | 4.1% | <0.01% |
| Q21 | −14.8% | 29.1% | 3.5% | <0.01% |
| many_join | −9.2% | 19.8% | 3.3% | <0.01% |

Spread is (max−min) as a percentage of the query's median over the retained runs.
Disabling the pass cuts wall time by 9–25% and collapses variance from up to ~29%
of median down to 3–7%.

### The pass was a net regression, not a neutral overhead

Compared against running with **no dynamic filters at all**, the priority pass
(`legacy`) actually made the filters a net loss on four of six queries — it cost more
than the filters saved. Turning it `off` restores the filters to a win (or neutral)
everywhere.

| Query | Filters + legacy, vs no filters | Filters + off, vs no filters |
|---|---:|---:|
| Q5 | +18.2% (slower) | −11.7% |
| Q7 | +26.5% (slower) | +2.1% |
| Q8 | −7.7% | −17.6% |
| Q9 | +2.0% (slower) | −10.2% |
| Q21 | +11.9% (slower) | −4.6% |
| many_join | −18.1% | −25.6% |

### Why there was no upside to lose

- **Coverage was identical (`legacy` == `off`).** On every query, zero rows entered a
  filter channel *before* publication — even under `legacy`. Publication already won
  every race, so prioritizing build tasks had no coverage benefit to deliver.
- **Peak memory was unchanged.** The feared front-loading cost did not appear: peak
  resident memory differs by <0.01% between `legacy` and `off`, well inside legacy's
  own spread. Feeder gauges balanced (`running_end=0`) on every query.
- **The switch does what it claims.** Prioritized feeder dispatches were nonzero under
  `legacy` and exactly zero under `off`.

### Correctness

One unique canonical result SHA-256 per query across all three configurations and all
iterations — result bags are bit-identical. DEBUG analyzer reported zero format
warnings in every process.

### Caveats

Measured on a single serialized GB200 at SF=300 with warm parquet on local disk. The
pre-publication race the pass was designed to win never occurred in this setup; cold
I/O or multi-GPU scheduling could behave differently. Rollback stays available for one
release via `SET dynamic_filter_build_priority='legacy'` (or YAML), and the recovery
path is tracked separately.
