# Which plan fragments and query plans the streaming primitives serve

A map from **exchange shape** → **primitive configuration** → **status**. Written to answer: given a
StarRocks query plan, which fragments get streaming operators, how are they parameterised, and which
shapes work today.

---

## 1. Three cardinalities, not one

Most confusion here comes from collapsing three independent numbers. They live in different places and mean
different things.

| Symbol | Meaning | Where it lives | Set by |
|---|---|---|---|
| **M** | senders feeding **one** input stream | `stream_input_spec::expected_senders` (a `std::set`) | number of *instances of one upstream fragment* |
| **K** | input streams **per fragment** | `fragment_spec::inputs` (a `std::map`) | number of `EXCHANGE_NODE`s in the fragment |
| **N** | destinations **per sink** | `_output_repositories.size()` | number of downstream receivers |

- **M > 1 is fan-in.** M instances of the *same* upstream fragment, one stream, **one** repository, **one**
  lifecycle holding M sender identities. EOS only when all M have closed.
- **K > 1 is a multi-input fragment.** K *different* upstream fragments — the join case. K separate
  `STREAMING_SOURCE` operators, each with its **own** id, repository and lifecycle.
- **N > 1 is fan-out.** One sink, N repositories, **one** shared lifecycle (the pipeline is the single
  sender), N independently-drainable outputs.

A join receiver reading two shuffled inputs has **K = 2**, and *each* of those streams may independently
have **M > 1**.

```
        upstream fragment A            upstream fragment B
        (M=3 instances)                (M=2 instances)
         │   │   │                       │   │
         └───┼───┘   fan-in per stream   └───┘
             ▼                             ▼
        stream id 2                    stream id 3        ← K = 2 input streams
      expected {0,1,2}                expected {0,1}
             │                             │
        STREAMING_SOURCE              STREAMING_SOURCE
             └──────────► HASH JOIN ◄──────┘
                             │
                        STREAMING_SINK (N destinations)
```

---

## 2. Three fragment roles

The role is decided by one predicate — whether the fragment declares an output stream
(`Fragment::declare_output`, `is_result()` in `sirius_ffi.cpp`).

| Role | Plan shape | Streaming operators | Declares output? |
|---|---|---|---|
| **Leaf / sender** | `SCAN → ops → STREAMING_SINK` | sink only | yes → parks native batches |
| **Intermediate** | `STREAMING_SOURCE ×K → ops → STREAMING_SINK` | both | yes → parks native batches |
| **Root / result** | `STREAMING_SOURCE ×K → ops → RESULT_COLLECTOR` | source only | **no** → produces Arrow |

`declare_output` is the whole path selector: with an output stream the fragment is rooted in a streaming
sink and its results park as native batches that outlive its own query; without one it takes the ordinary
single-shot path and produces Arrow.

---

## 3. The shape taxonomy

Every exchange is a point in (M, K, N). This table is the core of the document.

| Exchange shape | M | K | N | Sink constructor | Source config | Status |
|---|---|---|---|---|---|---|
| **Gather, single split** | 1 | 1 | 1 | 3-arg | `{0}` | ✅ **works** — this is TPC-H Q6 |
| **Gather, multi-split** | M | 1 | 1 | 3-arg | `{0..M-1}` | ⚠️ wired end to end, **never run** |
| **Multi-input (join) receiver** | any | **K ≥ 2** | — | — | K sources, K lifecycles | ⚠️ machinery exists, **never built in any test**; see §6 risk |
| **Shuffle / hash exchange** | M | 1–2 | N | **4-arg + `partition_spec`** | `{0..M-1}` | ❌ blocked on both halves |
| **Bucket shuffle** | M | 1–2 | N | 4-arg + spec | `{0..M-1}` | ❌ additionally needs CRC32 hash parity |
| **Broadcast** | 1..M | 1–2 | N (same rows to all) | *none* | — | ❌ **structurally impossible** (§5) |
| **Random / round-robin** | M | 1 | N | *none* | — | ❌ `partition_spec` is hash-only |
| **Merging (`ORDER BY` / top-N)** | M | 1 | 1 | 3-arg, but order is lost | `{0..M-1}` | ❌ unsupported |
| **Result fragment** | M | K | **0** | *none* (`RESULT_COLLECTOR`) | `{0..M-1}` | ✅ works for M = 1, K = 1; M > 1 shares the gather-multi-split ⚠, K > 1 the join ⚠ |

---

## 4. Query-plan patterns → configuration

### 4.1 Scalar aggregate — gather, M=1 ✅

```sql
SELECT sum(l_extendedprice * l_discount) FROM lineitem WHERE …;   -- new_planner_agg_stage = 1
```

```
Fragment 1 (sender)                       Fragment 2 (result)
  DATA_STREAM_SINK{dest_node_id: 2}         RESULT_SINK
   └ Project → Filter → FILE_SCAN   ══►      └ Aggregate sum
                                               └ EXCHANGE_NODE{node_id: 2}
```

Sink: 3-arg, one repository. Source: `expected_senders = {0}`. **This is the only shape demonstrated
end to end**, because the demo query names a single parquet file.

### 4.2 The same query over a sharded table — gather, M>1 ⚠️

Change one thing:

```sql
FILES("path"="file:///…/lineitem/*.parquet", …)   -- was part.0.parquet
```

The FE now plans **M scan instances**, each with a `DATA_STREAM_SINK` to the same `dest_node_id: 2`, each
carrying its own `sender_id`. `per_exch_num_senders[2] = M`, and the receiver's stream 2 gets
`expected_senders = {0..M-1}`. It will not finish until **every** one closes.

**This is the cheapest way to exercise fan-in — no code change, one glob.** The full chain is already wired:

| Step | Where | With M senders |
|---|---|---|
| FE declares the count | `per_exch_num_senders[node_id]` | `= M` |
| CN reads it | `compute_node_service.rs:246` | `register_receiver(finst, [(node_id, M)])` |
| Rendezvous | `push_sender` × M, deduped by `sender_id` | releases at M/M |
| CN declares to the engine | `engine.rs::run_fragment` | `declare_input_sender(…)` **per slot** |
| Engine | `stream_input_spec::expected_senders` | `stream_lifecycle({0..M-1})` |
| Relay | `relay_from(…)` **per slot** | M relays, M distinct `close_input`s |
| EOS | `mark_sender_done` × M | terminal only after all M |

No step assumes M = 1, and nothing refuses M > 1.

### 4.3 Two-phase GROUP BY — shuffle, M×N ❌

```sql
SET new_planner_agg_stage = 2;
SELECT l_returnflag, sum(l_quantity) FROM lineitem GROUP BY l_returnflag;
```

```
Fragment 1  ×M instances                    Fragment 2  ×N instances
  DATA_STREAM_SINK{HASH_PARTITIONED, N}       RESULT_SINK
   └ HASH_GROUP_BY (partial)      ══shuffle══► └ MERGE_AGGREGATE (final)
     └ FILE_SCAN                    by key      └ EXCHANGE_NODE
```

Every receiver has **M** senders; every sender has **N** destinations. This is Sirius's local
`HASH_GROUP_BY → PARTITION → MERGE_AGGREGATE` with **`PARTITION` replaced by the exchange** — same GPU
kernel, different routing.

Needs *both* halves: fan-in (#1297) and fan-out (#1299). Blocked three ways — see §6.

### 4.4 Shuffle join — shuffle on the join key, K=2 ❌

Both sides hash-partitioned on the join key so matching rows meet on the same node. The receiver has
**K = 2** input streams, each with its own M. Needs everything §4.3 needs, plus the multi-input-stream path
in §6.

### 4.5 Broadcast join ❌ — see §5

---

## 5. What is structurally impossible, not merely unimplemented

Worth separating from the roadmap, because these look like missing features and are actually shape
mismatches.

**Broadcast.** `hash_partition` puts each row in exactly **one** slice, and slice *i* goes to repository
*i*. A broadcast needs **every** row in **every** destination. No `partition_spec` value expresses that. It
needs a different code path — push the same batch handle N times, which is nearly free since batches are
shared handles, but it is not this operator's shape.

**Random / round-robin.** No key to hash, and `key_columns` must be non-empty when N > 1.

**Range partitioning.** No representation — the struct carries only hash keys and casts.

> **Naming.** `partition_spec` is really `hash_partition_spec`. Since #838's own message already defers
> "coalescing and range-partitioning", the name will mislead as soon as a second strategy lands. Worth
> renaming now, or turning it into a variant.

---

## 6. Status, gaps and risks

### Fan-in (M > 1) — wired, untested

Covered by unit tests (`SRC-24`, `stream_lifecycle` cases). **Not** covered at fragment level: no test sets
`expected_senders` with more than one id, and `FRAG-5` uses two senders but a single sender *id* and a
single `close_input`. Nothing refuses it. §4.2 is the cheapest way to find out if it works.

### Multi-input streams (K ≥ 2) — machinery exists, never built, **and one concrete risk**

`fragment_spec::inputs` is a map, `streaming_fragment::build()` loops over it and verifies each declared
stream was actually read, and the CN loops over `request.stream_inputs`. So K > 1 is expressed everywhere.

But **no test builds a fragment with two input streams** — every fragment test uses exactly one
(`test_streaming_fragment.cpp` lines 244, 331, 432, 511, all a single `spec.inputs[…]`).

> ⚠️ **Risk to verify before relying on K ≥ 2.** `task_scheduler::start_query()` schedules only the
> **first** source:
>
> ```cpp
> // src/pipeline/task_scheduler.cpp:217
> _task_creator->schedule(scans.front());
> ```
>
> With two `STREAMING_SOURCE` leaves under a join, only one is nominated. The second's sole re-nomination
> path is its own waker — but the waker is armed only by a `get_next_task_hint()` that returned WAITING,
> which requires it to have been scheduled at least once. If nothing nominates it, no push can wake it.
>
> The likely outcome is a hang or a silent empty result on any join-shaped receiver. This needs tracing
> (something else may nominate sibling sources) before the shuffle-join shape is attempted — it is cheap to
> check and expensive to discover later.
>
> **Update — half of this risk is already fixed by design in #1320 (`stream/01-source`).** That stack
> replaces the one-shot `arm_waker` with an `on_data` hook wired unconditionally in `set_pipeline()`,
> firing on **every** successful push — a source no longer needs to have been scheduled once before it can
> be woken. A push into either source under a join re-nominates it regardless of `scans.front()`, and an
> empty stream that closes without pushing is handled by `on_end_of_stream → update_pipeline_status`. On
> the new stack the residual K ≥ 2 gap is **coverage** (no test builds two input streams), not wake-up
> mechanics. The trace above still applies to *this* demo branch's `arm_waker` design.

### Fan-out (N > 1) — explicitly blocked, three independent ways

1. **The FFI cannot express it.** `sirius_ffi.cpp` sets `plan_source`, `inputs`, `outputs` — never
   `partitioning`, and `Fragment` has no method to supply one. A Rust caller who calls `declare_output`
   twice gets `outputs.size() == 2` with `partitioning == nullopt`, and `streaming_fragment`'s constructor
   **throws**. Fan-out over the FFI is not unimplemented — it is guaranteed to fail.
2. **The compute node refuses it.** `compute_node_service.rs:381` errors on `destinations.len() > 1`,
   deliberately: silently broadcasting would give every receiver but one no rows.
3. **The translator rejects two-phase aggregation**, so the FE never emits a shuffle to begin with.

### Hash parity — not started

The sink uses any consistent hash. Sufficient for an all-Sirius `HASH_PARTITIONED` shuffle (correctness
needs only that every sender agree); **wrong** for bucket-shuffle, which is anchored to the table's on-disk
layout computed by StarRocks' CRC32 at ingest, and wrong for any mixed Sirius / native-BE exchange. The
three regimes (fnv/xxh3 by `exchange_hash_function_version`, CRC32, bucket-id mapping) are untouched work.

---

## 7. Why sender identity, not a count

The one place this is load-bearing rather than pedantic. With M senders relayed in a loop, a wrapper bug
that hands the same `sender_id` twice would — under a counter — reach M closes while only M−1 distinct
senders had actually finished. The stream ends early and **the query silently returns short**.

As a `std::set`, the duplicate is a no-op and the stream correctly stays open (`SRC-24`). An id outside the
expected set **throws** rather than being counted, so a mis-wired sender is a loud error rather than a quiet
truncation.

This is also why the engine counts *identities* rather than connections: the abstraction survives the move
from in-process relay to a real network without change.

---

## 8. Summary — what to build next, in order

| Priority | Work | Why |
|---|---|---|
| 1 | Run §4.2 (the glob query) | Zero code, exercises fan-in through the whole chain |
| 2 | Rebase onto #1320's source redesign (or trace `scans.front()` here, §6) | The always-armed `on_data` waker removes the join-receiver wake-up risk by design; tracing is only needed if this branch's `arm_waker` design must ship first |
| 3 | Live-producer test | No waker (`arm_waker` here, `on_data` on #1320) has ever fired under a running engine with a concurrent producer |
| 4 | A K = 2 fragment test | The only shape gap that survives the #1320 rebase; every fragment test today builds exactly one input stream |
| 5 | `declare_partitioning` on `ffi::Fragment` | Turns a guaranteed throw into a working feature; small and well-scoped |
| 6 | Two-phase agg in the translator | Unblocks the shuffle shape end to end |
| 7 | Hash parity | Required before any mixed or bucketed deployment |
