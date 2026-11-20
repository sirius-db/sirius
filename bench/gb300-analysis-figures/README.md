# GB300 concurrency and ablation figures

The [concurrency figure](gb300-concurrency.pdf) shows completed queries per
second and the gain relative to one stream for SF100, SF300, and SF1000, at
1, 2, 4, 8, and 10 concurrent streams. The source is the saved
[concurrency summary](../gb300-results/concurrency/summary.csv). Each stream
ran all 22 queries with randomized query order and predicates. The union of
input columns was pinned on the host before timing, so pin time is excluded.
These are single-pass measurements without error bars; the workload omits a
refresh stream and is not an official TPC-H throughput score. Throughput
reaches about 25.4, 8.8, and 2.5 queries/s at eight streams for SF100, SF300,
and SF1000 respectively, then changes little at ten streams.

The [ablation figure](gb300-cumulative-ablation.pdf) compares the two
**corrected** SF1000 cumulative orders: [type narrowing first](../gb300-ablation/results/narrow-first-summary.csv)
and [BUILD_PROBE first](../gb300-ablation/results/join-first-corrected-summary.csv).
Both start with all five optimizations off, then add one optimization per step.
The first four stages pin all tables uncompressed; the last two pin `lineitem`
and `orders` with Simpatico compression. Each stage runs all 22 queries once
in a fresh process, and times exclude pinning and unpinning. The all-on totals
are 7.005 and 7.033 seconds respectively, versus about 22.15 seconds with all
five off. The incremental reductions depend on the order and enabled context;
the 0.028-second difference between all-on totals is too small to interpret
from one pass per order. Earlier ablation runs used a compression allowlist
that kept some pins compressed even in nominally uncompressed stages, so they
are not plotted.

Both figures have vector [SVG concurrency](gb300-concurrency.svg) and
[SVG ablation](gb300-cumulative-ablation.svg) exports, plus 600-dpi
[PNG concurrency](gb300-concurrency.png) and
[PNG ablation](gb300-cumulative-ablation.png) exports. Regenerate both from
this checkout with:

```bash
pixi exec --spec matplotlib python bench/gb300-analysis-figures/plot_analysis.py
```
