# GB300 concurrency and ablation figures

The [concurrency figure](gb300-concurrency.pdf) shows completed queries per
second for SF100, SF300, and SF1000, at 1, 2, 4, 8, and 10 concurrent streams.
The source is the saved
[concurrency summary](../results/concurrency/summary.csv). Each stream
ran all 22 queries with randomized query order and predicates. The union of
input columns was pinned on the host before timing, so pin time is excluded.
These are single-pass measurements without error bars; the workload omits a
refresh stream and is not an official TPC-H throughput score. Throughput
reaches about 25.4, 8.8, and 2.5 queries/s at eight streams for SF100, SF300,
and SF1000 respectively, then changes little at ten streams.

The [ablation figure](gb300-cumulative-ablation.pdf) shows the **corrected**
[compressed-pinning-first SF1000 run](../ablation/results/compressed-first-figure-summary.csv).
For paper labeling, its BUILD_PROBE admission step is called "Adaptive Join,"
its compressed-pinning step "Compression," and its fused filtering during
decode step "Operating on Compressed Data." The stages start with all five
optimizations off, then add one at a time. The baseline pins all tables raw;
the next five stages pin `lineitem` and `orders` with Simpatico compression.
Each stage runs all 22 queries once in a fresh process, and times exclude
pinning and unpinning. The all-on total is 7.016 seconds, versus 22.121 seconds
in a repeat of the all-off baseline. The first completed baseline measured
37.216 seconds because Q9 took 20.843 seconds, compared with 6.736 seconds in
the repeat. The figure uses the repeat baseline, and both measurements are
saved in the [ablation results](../ablation/RESULTS.md). The incremental
reductions depend on the order and enabled context.

Both figures have vector [SVG concurrency](gb300-concurrency.svg) and
[SVG ablation](gb300-cumulative-ablation.svg) exports, plus 600-dpi
[PNG concurrency](gb300-concurrency.png) and
[PNG ablation](gb300-cumulative-ablation.png) exports. Regenerate both from
this checkout with:

```bash
pixi exec --spec matplotlib python bench/gb300/analysis-figures/plot_analysis.py
```

Add `--ablation-only` to regenerate only the ablation figure.
