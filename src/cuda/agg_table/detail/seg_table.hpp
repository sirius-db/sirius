// Growable, budgeted aggregation table: the prototype's round-3 `seg_table`
// (experimental/agg_bench/src/seg_table.{hpp,cu}), moved here with these changes:
//
//  * Memory: every allocation comes from the memory resource given at construction (the prototype
//    used the current device resource), and nothing is shared between tables: the shared-memory
//    budget of the block-local kernels is read from the table's device at construction, the
//    reserved-key flag is the owner's (options::seen).
//  * Input: a fold reads raw 8-byte columns handed over as pointers (`raw_batch`) plus the same
//    rows as a generic source; the benchmark's `batch` is gone.
//  * Growth cannot throw an allocation failure out of a fold: a step whose allocation fails
//    leaves the table as it was and reports "cannot grow" (the rows then come back as not
//    applied). See grow_to for the one case that cannot be undone.
//  * Left out: the sealed mode (seal, absorb_tag, fingerprints, per-segment flush), evict /
//    restore, the partitioned pre-aggregation (off by default in the prototype) and the
//    AGG_SEG_TRACE diagnostics.
//
// The slots live in `2^sbits` SEGMENTS, one device allocation each, selected by the top hash
// bits. All segments have the same capacity. The table GROWS: every segment is rehashed into a
// larger one, so the transient during growth is a few segments, not a second table. A byte budget
// bounds everything the table allocates; when it cannot grow any further, rows with new keys are
// reported back ("full") and the caller diverts them. Two slot layouts: ROW `[key, state...]`
// per slot with linear probing, and BUCKET, 8 slots per bucket stored as `[8 keys][8 state
// rows]`, used for large tables.
//
// Transient GPU memory beyond `bytes()` (W = row_words(), S = segments, seg = bytes()/S):
//   fold            the caller's `scratch`: 1 bit per batch row, plus the partial rows of a
//                   pre-aggregating launch (reserve_scratch). Nothing else.
//   growth          new segments next to old ones: with a budget B the table keeps
//                   `(S + 1) x seg <= B`, so growth and emit stay inside B.
//   emit            the dense chunk of the segments being emitted.
#pragma once

#include "common.hpp"

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

namespace sirius::agg_table::detail {

namespace segdev {
struct seg_layout;
}

/// The rows of one fold as the canonical kernels read them: 8-byte words on the device.
struct raw_batch {
  uint64_t rows       = 0;
  uint64_t const* key = nullptr;
  uint64_t const* col[NUM_VALUE_COLUMNS]{};
};

class seg_table {
 public:
  struct options {
    uint32_t sbits     = 6;    // 2^sbits segments (at most 2^8)
    uint32_t skip_bits = 0;    // hash bits shared by every key (table for one outer partition)
    int layout         = -1;   // -1: ROW up to small_bytes, then BUCKET; 0 ROW; 1 BUCKET
    double load        = 0.0;  // max load factor; 0: by state width (ROW at most 0.85)
    /// Tables up to `small_bytes` use the ROW layout and run sparse, in three size tiers: load
    /// 0.25 up to small_bytes / 10, 0.4 up to 0.6 x small_bytes, 0.6 up to small_bytes (see
    /// load_for_cap in seg_table.cu for the measurements). 0: no tiers.
    uint64_t small_bytes    = 320ull << 20;
    double growth           = 2.0;  // capacity factor of one growth step (> 1)
    uint64_t initial_groups = 0;    // groups the first allocation must hold (0: the minimum table)
    /// Expected final groups (0: unknown). Growth steps stop at the capacity for this many groups
    /// until the count proves the hint wrong.
    uint64_t hint_groups  = 0;
    uint64_t budget_bytes = 0;  // 0: unlimited
    /// A table that reaches its budget while it still runs sparse (the size tiers of
    /// `small_bytes`: ROW layout at load 0.25 to 0.6) is rehashed once more, into the BUCKET
    /// layout at the load for its state width, before it reports rows back as "full": the tiers
    /// are for tables that fit, and a table that does not should hold all its budget can. Off by
    /// default (a caller that knows the group count picks the layout up front).
    bool dense_at_budget = false;
    /// BUCKET layout: free slots hold no state (see BUSY_KEY in seg_table.cu): creating and
    /// growing a table writes the key lines only, and a new group's first state is written with
    /// plain stores instead of identities + atomics.
    /// -1: for 4 and more state words (measured: +7..+34% at 2^29..2^30 groups x 8 words, but
    /// -15% with 1-2 words, where the fence costs more than the atomics it replaces); 0 off; 1 on.
    int lazy_init = -1;
    /// Up to 4 groups: each thread aggregates its rows privately before touching shared state.
    bool private_slots = true;
    /// Block-local pre-aggregation: counts and integer sums of the block's table as native 32-bit
    /// shared-memory atomics (1; 0: 64-bit compare-and-swap loops, as rounds 1 to 3). -1: by state
    /// width, on with 1 and 2 state words (see flat_table::pre2_default for the measurement).
    int pre_v2 = -1;
    /// Experiments: rows per thread, threads per block and local slots of the pre-aggregating
    /// launch (0: the plan's own).
    uint32_t pre_rows  = 0;
    uint32_t pre_block = 0;
    uint32_t pre_slots = 0;
    /// Direct fold: rows with the same key that follow each other are combined, up to this many.
    uint32_t run_cap = 8;
    /// Partitioned pre-aggregation for mid cardinalities (see seg_table.cu): 0 off; n > 0: always
    /// partition a batch into 2^n hash ranges when the direct kernel would be used; -1: by plan.
    /// Growth without an estimate: grow for what the rest of the batch is expected to bring, and
    /// size a step by the final group count when the observed counts determine it (see
    /// seg_table::estimate_groups; that can only make a step smaller than `growth`, unless
    /// `growth_trust` allows more).
    bool grow_ahead = true;
    /// Largest capacity factor of a step sized by the estimated final count. At `growth` (the
    /// default when 0) a wrong estimate costs no more memory than plain stepping. Larger values
    /// save rehash passes when the estimate is right (uniform keys: x25 takes a table to its final
    /// size in one step) and over-allocate by up to that factor when it is not: keys that keep
    /// arriving at a steady rate next to repeated ones look like a larger uniform key set.
    double growth_trust = 0.0;
    int part_bits       = 0;
    /// By plan: partition only when a batch has at least this many rows per group.
    uint32_t part_min_rows = 16;
    /// Direct fold: rows per thread. 0 or 1: one row per thread, runs combined by the thread of
    /// their first row (rounds 1 and 2). 2 and more: every thread walks a strip of this many rows
    /// run by run. -1: by table (see seg_table::strip_for).
    int strip = -1;
    /// Direct fold, 4 and more state words: order of a row's state updates. 0: as the aggregates
    /// are listed; 1: the min / max words first (each begins with a load the thread waits for, so
    /// the state line is fetched once before the sums are sent as atomics); 2: the sums first.
    /// -1: by table: 1 with 4 and 8 state words on input without runs of equal keys, else 0 (GB10,
    /// 2^27 rows, fold phase: 1 against 0 is +18..+30% with 8 words at 187,500 to 2^26 uniform
    /// groups, within noise with 4 words, and -22% with 8 words on clustered keys. GB200, job
    /// 23550, 2^31 rows, Mrows/s 0 -> 1: 187,500 groups 16,939 -> 23,110 with 8 words and 30,037
    /// -> 36,787 with 4; n/32 with 4 words 9,429 -> 9,860; neutral elsewhere, clustered included).
    int order = -1;
    /// ---- growth without a group count (the blind mode of `adaptive` sets these; the defaults are
    /// the behaviour before them) ----
    /// estimate_groups accepts the uniform-keys model on REPEATED KEYS instead of on the share of
    /// distinct rows: once the rows folded so far hold at least this many repeats (rows minus
    /// groups; the relative error of the estimate is about 1 / sqrt(repeats)) and the model also
    /// reproduces the repeats of an earlier point. 0: the rule of round 3 (at least 2% repeated
    /// rows, the earlier group count within 3%), which needs R >= 0.04 G rows before it answers;
    /// with 16,384 repeats it answers after sqrt(32,768 G) rows (2^30 keys: 6 M rows, not 43 M).
    uint64_t model_min_repeats = 0;
    /// A hint is a target, not a size: the model above still sizes the steps when it has an
    /// answer (the hint may be wrong either way), and a step without one goes straight to the
    /// hinted size instead of one `growth` factor at a time. For a table that starts below its
    /// hint (`initial_groups` < `hint_groups`).
    bool hint_soft = false;
    /// With `hint_soft`: a step without an answer from the model first goes to this share of the
    /// hinted size (when the table is below it), then to the hinted size.
    double hint_share = 1.0;
    /// Steps of the fixed factor follow a ladder of GROUP counts, `initial_groups` x 1.05 x
    /// growth^j (the smallest table's groups without `initial_groups`), instead of multiplying
    /// the slots: the sizes then do not depend on where the sparse size tiers round a small table
    /// up, and a table whose first size was the groups of one batch reaches the size for 2^j
    /// batches with 15% to spare instead of just missing it.
    bool growth_by_groups = false;
    /// With `growth_by_groups`: the ladder stays anchored at the first size whatever sized the
    /// steps in between (a hint, the model): the next plain step goes to the first anchored rung
    /// above the table, not to `growth` times what a hint left it at.
    bool ladder_anchor = false;
    /// With `growth_by_groups` and `growth` > 2: a plain step may only take the table up to this
    /// many bytes with the full factor; beyond, the factor is 2. 0: no bound.
    uint64_t big_step_bytes = 0;
    /// A step the model sizes is refused (the table reports itself full, as at the budget) when
    /// the largest dense table of the budget would hold less than this share of the model's key
    /// count: the caller is going to flush such a table, so it should not first be grown to the
    /// budget. 0: never.
    double give_up_share = 0.0;
    /// Groups that arrive through `upsert` (partial rows: the groups of front tables that
    /// overflowed) do not switch the growth model off; the caller reports the input rows behind
    /// them with `add_observed`, so the model's (rows, groups) history stays that of the stream.
    bool keep_model_on_upsert = false;
    /// Growth of a table whose segments are separate allocations: this many old segments are
    /// rehashed between two stream synchronisations (and freed together), never more than fit the
    /// budget or than 2 new segments' bytes. 1: one at a time (transient: one segment). More trades
    /// synchronisations for transient bytes: with g segments the peak during a step is the new
    /// table plus g old segments, (g - 1) / (64 x factor) of the new table more than with 1.
    uint32_t grow_segs = 1;
    /// With `grow_segs` > 1: a table of at most this many bytes is rehashed with ONE
    /// synchronisation per step - all its old segments stay until the new ones are complete, as
    /// the whole old table of a packed one does, when the budget has room for them (0: off).
    /// A factor-2 step otherwise synchronises 16 times (4 old segments at a time); a table that
    /// starts small and doubles many times pays that at every step.
    uint64_t grow_small_bytes = 0;
    /// Long runs of equal keys, 4 and more state words (1; 0: off). When 1,022 of 1,024 evenly
    /// spaced rows of a batch are followed by a row with the same key (runs of about 500 rows and
    /// more), the batch goes through the pre-aggregating launch with private slots - every thread
    /// folds its rows in registers, a block adds its few groups to the table once - whatever the
    /// group count. The direct kernels send every row (the strip kernel: every 4 rows) to the
    /// table, and with runs that long all threads of the launch update the same slot: the f64 sum
    /// and the min / max words are compare-and-swap loops, which then retry against each other
    /// (GB200 job 24080, 3,000 keys in runs of 716,000 rows: 3,975 / 2,654 M rows/s with 4 / 8
    /// aggregates, against 36,564 / 23,088 for 187,500 uniform keys). With 1 and 2 state words
    /// the updates are native atomic adds and do not collapse.
    int run_preagg = 1;
    /// estimate_groups (with `model_min_repeats`): how far the repeats the model predicts for the
    /// earlier point may be from the observed ones, as a share of them (plus four standard
    /// deviations). 0.15 is the rule before; uniform keys are within 0.01 once a few ten thousand
    /// repeats are in, mildly skewed ones (Zipf 0.5: 0.08 to 0.25) are not, and their answer is
    /// wrong by a factor that grows with the rows.
    double model_tol = 0.15;
    /// A step the model sizes holds at most this many times the groups the table has now (0: no
    /// bound): what a wrong answer can cost is bounded by what was actually inserted.
    double jump_cap = 0.0;
    /// A fold that left rows out at the limit grows for what those rows are expected to bring
    /// (new groups per applied row of that launch x the rows left out), in one step, instead of
    /// one plain step per retry.
    bool retry_ahead = false;
    /// Growth by the measured curve (0: off, the ladder above). A step without an answer from the
    /// uniform-keys model is sized for the groups the table is expected to hold `curve_x` times
    /// further on - at the next rung of a ladder of ROWS, first batch x curve_x^j (the row prior of
    /// a caller that has no row count) - read off the (rows, groups) points the folds left behind:
    ///   * the groups at R, R/2, R/4 and R/8 rows (interpolated between the points kept);
    ///   * "known keys plus a uniform tail", D = G (1 - exp(-a R)), fitted through R/2 and R and
    ///     accepted when it reproduces R/4 (as `model_tol`): exact for uniform keys and for a few
    ///     heavy keys in front of uniform ones, whose steps then stop at D(rung) - or at G when
    ///     that is at most 1.125 times as much, or when nine rows in ten were new keys at the
    ///     start (uniform keys alone: what the uniform-keys model would have answered);
    ///   * else the increments per octave continued, with the ratio of two consecutive ones
    ///     falling short of 2 by a deficit that itself keeps its ratio from octave to octave
    ///     (constant ratio: a power law, Zipf; halving deficit: keys at a steady rate behind a
    ///     start-up; doubling deficit: saturation), times `curve_margin`;
    ///   * never more than the latest octave's new groups per row would bring, never less than
    ///     the need, and at most `jump_cap` times the groups held.
    /// Without four such points a step is a plain one. The first batch is folded in pieces that
    /// end at 1/8, 1/4 and 1/2 of it, so that its end has them (three more launches, once).
    double curve_x      = 0.0;
    double curve_margin = 1.125;
    /// With `curve_x`: a row count the caller was told (0: none; it may be wrong). A rung of the
    /// row ladder beyond it is replaced by it while the rows folded are below it, so the last
    /// step is sized for the hinted end instead of for up to `curve_x` times as much.
    double curve_rows = 0.0;
    /// With `curve_x` and no `curve_rows`: a curve-sized step holds at most this many times the
    /// groups held (0: no bound beyond `jump_cap`). The rung is then only a prior on the input's
    /// end.
    double curve_cap = 0.0;
    /// With `curve_x`: also once two folds were in flight at the same time (false: from then on
    /// the steps are plain ones, as without `curve_x`). The points of several streams are noisier
    /// (their first batches arrive out of order), and so is the uniform-keys model's own test:
    /// when it misses its first answer on uniform keys the curve takes a step for a rung that the
    /// model's answer then replaces - one more rehash of what the table holds by then.
    bool curve_shared = false;
    /// Hot keys (1; 0: off, the behaviour before). A few keys that each take a real share of the
    /// rows of a batch with many keys: the atomic updates of such a key's slot, issued next to
    /// updates of other slots, are what the direct kernels then spend their time on (seg_keys.cu
    /// has the measurements; GB200 job 24150: Zipf keys with exponent 1.5, 2.3 M keys, 4,495 M
    /// rows/s with 1 state word and 986 with 8, where 187,500 uniform keys run at 66,000 and
    /// 23,000). Every launch of a direct fold of `hot_min_rows` rows or more is preceded by a
    /// sample of 4,096 of its rows, taken and kept on the device (one block; nothing is read back
    /// before the fold): the keys sampled `hot_thr` times or more are the hot set, provided they
    /// cover `hot_cover` of the sample. The fold of a scratch whose PREVIOUS sample had a hot set
    /// uses the kernel with block-local accumulators for the hot keys of ITS OWN sample
    /// (seg_keys.cu), any other the direct kernels as before - so input without hot keys runs the
    /// kernels it always ran, plus the sample. A scratch's first launch reads its sample back.
    /// The sealed pass (absorb_tag with a scratch) does the same.
    int hot_keys          = 1;
    uint32_t hot_thr      = 16;    // of 4,096 sampled rows (0.4% of the rows)
    double hot_cover      = 0.03;  // share of the sampled rows the hot set must hold
    uint32_t hot_rows     = 8;     // rows per thread of the hot-keys kernels
    uint32_t hot_block    = 256;   // threads per block
    uint64_t hot_min_rows = 1ull << 16;
    /// A thread keeps its rows of the first hot key it meets in registers and adds them to the
    /// block's accumulator once (1; 0: every hot row is a shared-memory atomic).
    int hot_own = 1;
    /// A thread's rows: consecutive ones (1) or every `hot_block`-th (0: a warp reads 32 rows
    /// that follow each other; GB10: consecutive rows are 0 to 40% slower at every width).
    /// -1: as the direct fold of the table (strip_for).
    int hot_strip = 0;
    /// A scratch without a hot set samples only every this many launches (1: every launch).
    uint32_t hot_every = 1;
    /// Generic kernel (aggregate sets without a specialised kernel, the nullable layout): keys
    /// that repeat (1; 0: off, every row goes to the table through seg_upsert_kernel, the behaviour
    /// before). Every raw batch of 65,536 rows or more is sampled - the 1,024 rows of `run_preagg`
    /// and the 4,096 of `hot_keys`, read back with one synchronisation - and its first pass runs
    /// seg_fold_gen_kernel (seg_gen.cu) when
    ///   * 1,022 of the 1,024 sampled rows are followed by a row with the same key (the rule of
    ///     `run_preagg`: runs of about 500 rows and more): a thread accumulates the rows that
    ///     have the key of its previous row in registers and sends one state update per piece of
    ///     a run to the table - `gen_run_rows` rows per thread when no sampled row ended a run,
    ///     else `gen_run_rows_min`; or
    ///   * the sample has hot keys (`hot_thr`, `hot_cover`; also when the batch has few keys at
    ///     all): blocks of 256 threads x `hot_rows` rows with one accumulator per hot key in
    ///     shared memory, as the hot-keys kernels of the canonical sets.
    /// Any other batch runs seg_upsert_kernel as before, and so do the retries after growth.
    /// Without it the generic kernel sends every row's updates to the table, and on such input
    /// they meet on the same few slots, where the f64 sum and the min / max words are
    /// compare-and-swap loops that retry against each other. GB200 job 24654, 2^31 rows, 3,000
    /// keys in runs of 716,000 rows, 8 aggregates that are not a canonical set: 5.995 s, against
    /// 0.958 s on the cuDF path and 0.30 s for 187,500 uniform keys. GB10, one batch of 2^24 rows
    /// folded repeatedly, the same 8 aggregates, M rows/s without -> with (the canonical 8-word
    /// set): runs of 350,000 rows 330 -> 2,660 (3,400); one key 320 -> 2,860 (3,300); 2 keys on
    /// half of the rows of a 4 M key input 180 -> 860 (810); 16 keys on 90% of them 690 -> 1,480
    /// (1,800); 4 / 16 / 40 uniform keys 420 / 1,320 / 1,230 -> 2,520 / 2,230 / 2,200 (3,100 /
    /// 2,940 / 3,140). Input without runs or hot keys is unchanged: it runs the same kernel,
    /// after the two samples.
    /// Strided rows (`gen_run_strided`, as the block-local kernels read them) against consecutive
    /// ones, GB10, runs of 500 to 350,000 rows: 2,400 to 2,900 against 1,900 to 2,100.
    int gen_combine           = 1;
    uint32_t gen_run_rows     = 256;  // RUNS: rows per thread when no sampled row ended a run
    uint32_t gen_run_rows_min = 32;   // ... when one or two did
    uint32_t gen_run_strided  = 1;    // RUNS: a thread takes every 256th row of its block
    /// Measurement: time the initialisation and the rehash of growth separately (stats_t::
    /// init_seconds / rehash_seconds). Adds one stream synchronisation per initialised segment.
    bool profile = false;
    /// Two device words of the table's owner (seg_layout::seen): the reserved-key flag the fold
    /// kernels raise, and a scratch word. Must outlive the table.
    uint32_t* seen = nullptr;
  };

  /// Per-fold scratch. Not shared between threads.
  struct scratch {
    scratch(::cuda::stream_ref stream, rmm::device_async_resource_ref mr);
    rmm::device_async_resource_ref mr;
    /// Bit r set: row r of the last folded batch was NOT applied (the table was full).
    rmm::device_uvector<uint32_t> bitmap;
    /// Row-layout partial states not applied by the last fold (pre-aggregated block groups):
    /// the rows of `partials` whose bit is set in `partial_bitmap`, among `partial_count` rows.
    rmm::device_uvector<uint64_t> partials;
    rmm::device_uvector<uint32_t> partial_bitmap;
    uint64_t partial_count = 0;
    rmm::device_uvector<uint64_t> counters;
    uint64_t folds = 0;  // batches folded with this scratch
    /// Set by `reserve_scratch`: `fold` allocates nothing in this scratch. A chunk whose
    /// pre-aggregating launch needs more partial rows than `partials` holds is folded directly
    /// instead, and the size it wanted is kept in `want_partials` for the next reservation.
    bool fixed             = false;
    uint64_t want_partials = 0;
    /// options::hot_keys: the hot set of the latest sampled launch (device; see seg_dev.cuh), and
    /// what the host knows of it: -1 nothing yet, 0 no hot keys, 1 hot keys.
    rmm::device_uvector<uint64_t> hot;
    int hot_state     = -1;
    uint32_t hot_skip = 0;  // launches since the last sample
  };
  /// Allocates, before a fold, everything `fold` may need in `sc` for a batch of up to `n` rows:
  /// the bitmap (zeroed), the partial rows of a pre-aggregating launch and their bitmap. After
  /// this call a fold with `sc` allocates no scratch, so an allocation failure cannot interrupt a
  /// batch that is half applied.
  void reserve_scratch(scratch& sc, uint64_t n, ::cuda::stream_ref stream) const;

  /// What a fold could not apply because the table is full and may not grow.
  struct fold_result {
    uint64_t rows     = 0;  // input rows, marked in scratch::bitmap
    uint64_t partials = 0;  // partial states, marked in scratch::partial_bitmap
    /// New groups this fold inserted (counted by its own launches).
    uint64_t inserted = 0;
    bool full() const { return rows != 0 || partials != 0; }
  };

  struct stats_t {
    uint64_t growths         = 0;  // growth steps (each rehashes every segment)
    double grow_seconds      = 0;
    uint64_t rehashed_bytes  = 0;  // bytes of old segments read by growth
    uint64_t retries         = 0;  // folds that hit the limit and had rows re-applied after growth
    uint64_t rehashed_groups = 0;  // groups moved by growth, over all steps
    uint64_t grow_launches   = 0;  // kernel launches of growth (initialise + rehash)
    uint64_t init_bytes      = 0;  // bytes of new segments initialised (first allocation included)
    /// Seconds allocating + initialising segments (first allocation and growth) and rehashing.
    /// Only measured with `options::profile` (each needs a stream synchronisation of its own);
    /// `grow_seconds` is always measured and includes both.
    double init_seconds   = 0;
    double rehash_seconds = 0;
    uint64_t chunks =
      0;  // fold launches (a batch is folded in chunks while the table follows the count)
    uint64_t retry_rows  = 0;  // rows a fold left out at the limit (looked up again after growth)
    uint64_t jumps       = 0;  // growth steps sized by the estimated final count (beyond `growth`)
    uint64_t first_slots = 0;  // slots of the first allocation
    uint64_t gave_up     = 0;  // steps refused by options::give_up_share
    uint64_t curve_steps = 0;  // growth steps sized by options::curve_x
    uint64_t curve_fits  = 0;  // ... of them by the fitted "known keys plus uniform tail" model
    uint64_t run_chunks  = 0;  // fold launches that took options::run_preagg
    uint64_t hot_samples = 0;  // launches whose rows were sampled for hot keys (options::hot_keys)
    uint64_t hot_chunks  = 0;  // fold launches / sealed passes through the hot-keys kernels
    uint64_t hot_found   = 0;  // hot keys of the samples read back, summed
  };

  seg_table(state_spec const& spec,
            options const& opt,
            ::cuda::stream_ref stream,
            rmm::device_async_resource_ref mr);

  uint32_t row_words() const { return _W; }
  uint64_t num_segments() const { return 1ull << _opt.sbits; }
  uint64_t segment_cap() const { return _cap; }  // slots per segment (now)
  uint64_t bytes() const;                        // device bytes held by the table now (thread-safe)
  uint64_t group_limit() const;                  // groups it holds before growing / full
  double load() const { return _load; }
  bool bucket_layout() const { return _bk; }
  bool canonical() const { return _canon; }
  uint64_t groups() const { return _total.load(); }  // as of the last completed launch
  stats_t stats() const;
  /// Largest table (in groups) the budget allows; ~0 without a budget.
  uint64_t max_groups(bool dense = false) const;

  flat_table::source row_source(uint64_t const* rows) const;

  /// Folds a raw batch (`b` for the canonical kernels, `rsrc` the same rows for the generic one;
  /// a table whose state words are not a canonical set only reads `rsrc`). Grows the table as
  /// needed. Thread-safe: any number of threads may fold at once, each with its own stream and
  /// scratch (growth is exclusive and blocks them). Returns what was not applied; a row whose key
  /// is already in the table is always applied. `sc.bitmap` must be zero on entry.
  /// `allow_preagg` false: no block-local pre-aggregation (then no partial state is ever left
  /// over).
  fold_result fold(raw_batch const& b,
                   flat_table::source const& rsrc,
                   scratch& sc,
                   ::cuda::stream_ref stream,
                   bool allow_preagg = true);

  /// Generic fold of `n` rows described by `src` (raw rows or partial states). `bitmap` holds one
  /// bit per row. `only_marked`: only rows whose bit is set are tried, and a row that is applied
  /// has its bit cleared. Otherwise every row is tried and the bits of rows that could not be
  /// applied are set (pass a zeroed bitmap). Grows like `fold`. Returns the rows not applied.
  /// `counters`: at least 4 words of the caller's scratch.
  /// `gp` (raw rows, `only_marked` false): the kernel of the first pass, see gen_plan.
  struct gen_plan {
    uint32_t rpt        = 0;        // rows per thread; 0: seg_upsert_kernel (one row per thread)
    uint32_t strided    = 0;        // RUNS: a thread takes every 256th row of its block
    uint64_t const* hot = nullptr;  // the hot set of the batch's sample (HOT), or null (RUNS)
  };
  uint64_t upsert(flat_table::source const& src,
                  uint64_t n,
                  uint32_t* bitmap,
                  bool only_marked,
                  rmm::device_uvector<uint64_t>& counters,
                  ::cuda::stream_ref stream,
                  bool allow_insert  = true,
                  gen_plan const* gp = nullptr);

  /// Dense row-layout groups (unordered; nearly in hash order).
  struct chunk {
    rmm::device_uvector<uint64_t> rows;
    uint64_t groups;
  };
  /// Segments [p0, p0 + count) to ONE dense chunk in one pass, allocated from `out_mr`; the
  /// segments are freed. A packed table is emitted whole by the call that covers segment 0. Every
  /// segment is emitted exactly once; after the first call the table accepts nothing else. If the
  /// chunk cannot be allocated the call throws and may be repeated.
  chunk emit_segments(uint64_t p0,
                      uint64_t count,
                      ::cuda::stream_ref stream,
                      rmm::device_async_resource_ref out_mr);
  /// How many segments `emit_segments` may take at a time for its chunk to stay near `max_bytes`.
  uint64_t segments_per_chunk(uint64_t max_bytes) const;
  bool packed() const { return _packed; }

  /// What a table built with `opt` takes (nothing is allocated).
  struct sizing {
    uint64_t need_bytes;  // to hold opt.initial_groups without growing, were there no budget
    uint64_t min_bytes;   // smallest budget a table with these options can be built with
    uint64_t slots;       // slots of the first allocation (clamped to the budget)
    uint64_t limit;       // groups the first allocation holds
    uint64_t max_limit;   // groups the largest table the budget allows holds (~0 without a budget)
    bool bucket;          // layout of the first allocation
  };
  static sizing presize(state_spec const& spec, options const& opt);

  /// Changes the byte budget. A budget below what the table holds now only stops its growth.
  void set_budget(uint64_t bytes);
  /// Bytes the budget must cover for the table as it is now: S + 1 segments.
  uint64_t budget_need() const { return (num_segments() + 1) * _cap * _W * sizeof(uint64_t); }

  /// Words of a bitmap for `n` rows.
  static uint64_t bitmap_words(uint64_t n) { return (n + 31) / 32; }

  /// Input rows that were aggregated elsewhere before their groups were upserted here (see
  /// options::keep_model_on_upsert).
  void add_observed(double rows);
  /// The final group count as the growth model sees it now, else its last answer; 0: never had one.
  uint64_t model_groups() const;

 private:
  struct shape_only {};  // a table object that only answers sizing questions (no device memory)
  seg_table(state_spec const& spec, options const& opt, shape_only);
  void init_shape();
  uint64_t bytes_unlocked() const;
  uint64_t min_cap() const;
  bool pre_v2_on() const
  {
    return _opt.pre_v2 < 0 ? flat_table::pre2_default(_NS) : _opt.pre_v2 != 0;
  }
  uint32_t hot_flags() const  // see keys_fold
  {
    bool strips = _opt.hot_strip < 0 ? strip_for() > 1 : _opt.hot_strip != 0;
    return (pre_v2_on() ? 1u : 0u) | (_opt.hot_own != 0 ? 2u : 0u) | (strips ? 4u : 0u);
  }
  uint64_t cap_for_groups(uint64_t groups) const;
  uint64_t max_cap() const;
  double load_for_cap(uint64_t cap) const;
  uint64_t tier_bytes(int t) const;
  static constexpr int TIERS               = 3;
  static constexpr double TIER_LOAD[TIERS] = {0.25, 0.4, 0.6};
  uint64_t cap_for(double per_seg) const;
  uint64_t limit_for_cap(uint64_t cap) const;
  uint64_t need_after(uint64_t fails, bool rows_are_groups) const;
  bool grow_for(uint64_t need_groups,
                uint64_t seen_cap,
                ::cuda::stream_ref stream,
                uint64_t model_groups = 0,
                bool by_curve         = false,
                bool* declined        = nullptr);
  double curve_at(double r) const;
  double curve_groups(double R, double D, double H, bool* fitted) const;
  bool try_grow_to(uint64_t new_cap, ::cuda::stream_ref stream);
  void grow_to(uint64_t new_cap, ::cuda::stream_ref stream);
  bool densify(::cuda::stream_ref stream);
  void upload_pointers(::cuda::stream_ref stream);
  void fill_layout(void* out) const;
  bool lazy() const { return _bk && (_opt.lazy_init < 0 ? _NS >= 4 : _opt.lazy_init != 0); }
  bool can_pack(uint64_t cap) const;
  bool layout_for(uint64_t cap) const;
  uint64_t estimate_groups() const;
  uint32_t strip_for() const;
  void begin_emit(::cuda::stream_ref stream);
  void allocate(::cuda::stream_ref stream);
  uint64_t upsert_once(flat_table::source const& src,
                       uint64_t n,
                       uint32_t* bitmap,
                       bool only_marked,
                       bool allow_insert,
                       rmm::device_uvector<uint64_t>& counters,
                       uint64_t& seen_cap,
                       ::cuda::stream_ref stream,
                       uint64_t* inserted = nullptr,
                       gen_plan const* gp = nullptr);
  void note_count(uint64_t count);

  state_spec _spec;
  options _opt;
  rmm::device_async_resource_ref _mr;
  uint32_t _W;
  uint32_t _NS;
  bool _bk               = false;
  bool _canon            = false;
  double _load           = 0;
  uint64_t _cap          = 0;
  uint64_t _local_budget = 0;      // shared memory a block-local table may use on this device
  bool _packed           = false;  // small table: all segments in _segs[0]
  std::vector<rmm::device_uvector<uint64_t>> _segs;
  std::vector<bool> _emitted;
  rmm::device_uvector<uint64_t*> _ptrs;
  rmm::device_uvector<uint64_t> _count;  // [0] groups in the table, [1] growth's counter
  rmm::device_uvector<uint32_t> _full;   // per segment: no free slot within reach
  mutable std::shared_mutex _rw;         // shared: fold launches; exclusive: growth, emit
  mutable std::mutex _m;                 // host-side statistics below
  std::atomic<uint64_t> _total{0};
  std::atomic<int> _active{0};
  std::atomic<bool> _at_max{false};
  mutable uint64_t _model_last = 0;  // the model's last answer (under `_m`)
  bool _gave_up = false;  // options::give_up_share refused a step: the table grows no more
  bool _broken  = false;  // a growth step failed half way and could not be undone
  double _ladder =
    0;  // options::growth_by_groups: the current rung (groups); under the exclusive lock
  double _rate = -1.0;  // new groups per input row in the latest fold (-1: unknown)
  // Raw rows folded so far and two earlier (rows, groups) points, for estimate_groups.
  double _obs_rows = 0, _obs_mid_rows = 0, _obs_mid_groups = 0, _obs_old_rows = 0,
         _obs_old_groups = 0;
  bool _model_off        = false;
  // options::curve_x: (rows, groups) after folds that applied all their rows, about every 1.19
  // times the rows; the rows of the first batch folded (the first rung of the horizon ladder).
  std::vector<std::pair<double, double>> _hist;
  double _batch0    = 0;
  double _rows_done = 0;
  double _ins_done  = 0;
  std::atomic<bool> _curve_off{false};  // two folds were in flight at once (options::curve_shared)
  std::atomic<float> _runs{0.0f};       // sampled share of rows followed by a row with the same key
  bool _emitting       = false;
  uint64_t _emit_total = 0;  // groups in the table when the emit started
  stats_t _stats;
};

}  // namespace sirius::agg_table::detail
