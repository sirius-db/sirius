# TPC-H query deviations from the stock text

Every deviation from the stock TPC-H query text, why it exists, and what it costs.

**These notes live here and NOT as `--` comments inside the `.sql` files.** `bench.sh` runs each
query as `mysql -e "$Q"`, which collapses newlines — a leading `--` comment therefore swallows the
entire statement and the query fails with a syntax error. Verified the hard way.

---

## q08, q09 — `FROM` clause reordered (2026-08-19)

```diff
     FROM
         part,
-        supplier,
-        lineitem,
+        lineitem,
+        supplier,
         orders,          -- q08
         partsupp,        -- q09
```

Semantically identical: these are inner joins, which commute. **But it is not the stock text, so
both engines in an A/B comparison must use THIS text or the comparison is invalid.**

### Why

The FE has **no statistics for `FILES()` external scans** — every node in the plan reports
`cardinality: 1` (see `plans/q08.verbose.txt`). With no cost signal the CBO joins `part` and
`supplier` first: they are adjacent in the stock `FROM` and share **no predicate** (both keys
route through `lineitem`). The result is `4:NESTLOOP JOIN / join op: CROSS JOIN`.

That build side is **real, not a bad estimate**:

| Query | HASH_JOIN requested | Decomposition |
|---|---|---|
| q08 | 537,032,000,000 B | ÷4 = 134,258 × 10⁶ = filtered_part × supplier |
| q09 | 2,694,604,000,000 B | ÷4 = 673,651 × 10⁶ = filtered_part × supplier |

The `× 10⁶` is simply SF100's supplier row count. The join OOMs after 100 retries, and
`engine.rs`'s blanket `parked.clear()` then wipes every parked sender output — so the FE reported
the collateral `no parked sender output to export for SenderSlot` error and the real cause was
discarded. (That masking is fixed separately; the wipe now records and reports the true cause.)

Reordering so every adjacent pair shares a predicate removes the cross join entirely: 0 NESTLOOP,
7 HASH JOIN for q08. **No session variables are required** — with `cardinality: 1` everywhere the
CBO has nothing to reorder with, so it follows the written order.

### Measured (2× RTX PRO 6000, 2 CNs, SF100)

| Query | Before | After | vs DuckDB oracle |
|---|---|---|---|
| q08 | never completed | **1897 ms** | MATCH, max rel diff 3.7e-05 |
| q09 | never completed | **2115 ms** | 175/175 rows, all LOW by ~0.147 % (the known decimal-lowering defect) |

Full sweep went from 19/22 to **20/22**, no timing regression (−0.4 % total).

### The principled fix

Give the CBO real cardinalities — then the stock text plans correctly and this deviation can be
reverted. Options not yet evaluated: `ANALYZE` on the external scans, an external catalog with
statistics, or injected stats. All may require a load step, which would break the benchmark's
"read parquet directly, no load" property. Until then this reorder is the documented workaround.
