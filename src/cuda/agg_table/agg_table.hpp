// Interface between the GPU aggregation table (this directory) and the engine operator
// (src/op/aggregate/table_aggregate*). Keep it free of DuckDB and Sirius-operator types:
// cuDF / RMM / CUDA only, so the library builds and is tested without the engine
// (experimental/agg_table_test).
//
// Model. One `table` per (operator instance, GPU). It lives on the device and in the memory
// resource it was created with and never moves. Several tasks may call fold() on the same table
// concurrently, each with its own CUDA stream ("streams"). A table never spans GPUs: with several
// GPUs admitted the operator keeps one table per GPU, and their contents meet only as partial rows
// (export_partial -> the engine's existing partition + merge path, or fold_partial into another
// table).
//
// v1 scope (plan milestones M0 + M2a): the table holds what fits in its budget. When a fold cannot
// place rows, it returns them (as a row selection of the input batch) and the caller sends those
// rows down the existing cuDF path; `flush()` turns the whole table into partial rows at any batch
// boundary (budget revoked, end of input, giving up). No parked/sealed mode, no spill, here.
//
// Implementation: agg_table.cu (types, NULLs, SQL-correctness layer, tickets, partial rows) on top
// of detail/seg_table.* (the prototype's segmented table and its kernels). The header comment of
// agg_table.cu has the state layout, the partial-row mapping and the locking design.
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sirius::agg_table {

/// Aggregate over one input column (or none, for COUNT(*)).
enum class agg_kind : uint8_t { count_star, count, sum, min, max };

struct agg_spec {
  agg_kind kind;
  int value_column;       ///< index into the fold's `values` view; -1 for count_star
  cudf::data_type input;  ///< type of that column
};

/// Bound on a value column of one batch, when the scan knows it (row-group statistics).
struct column_bound {
  bool known  = false;
  int64_t min = 0;
  int64_t max = 0;
};

struct config {
  cudf::data_type key_type;  ///< one fixed-width key of at most 64 bits (v1)
  std::vector<agg_spec> aggs;
  std::size_t budget_bytes = 0;         ///< ceiling for the table's device memory
  std::optional<uint64_t> groups_hint;  ///< trusted group count; absent => grow from measurements
  std::optional<uint64_t> rows_hint;    ///< best estimate of the input rows
};

/// What can be decided before any data: whether these keys and aggregates can use the table at
/// all. Everything this rejects stays on the cuDF path.
bool eligible(const config& cfg, std::string* why_not = nullptr);

/// Schema of partial rows (flush / fold_partial / the rows the engine merges): the key column
/// followed by ONE state column per aggregate, in `cfg.aggs` order - what the engine's local
/// grouped aggregation feeds to its merge stage (`grouped_aggregate_layout`: keys, then one cuDF
/// partial per expanded aggregate), so partial rows of the table and of the cuDF path merge
/// together:
///   count_star, count      INT64, never null
///   sum  INT8..INT64       INT64 (nullable) - see the note on wide sums below
///   sum  DECIMAL32(s)      DECIMAL64(s);   sum DECIMAL64(s)   DECIMAL128(s)
///   sum  FLOAT32 / 64      the input type (accumulated in float64)
///   min, max               the input type
/// SUM / MIN / MAX are NULL for a group without a non-null input.
/// Wide sums: a SUM over INT8..INT64 whose int64 state could have wrapped (the table then keeps
/// 128-bit sums) leaves a flush as DECIMAL128 scale 0 instead of INT64, for that flush only - the
/// engine's merge accepts either carrier per batch for such a SUM
/// (`wide_integer_sum_partial_types`) and so does fold_partial. This function returns the INT64
/// form.
std::vector<cudf::data_type> partial_schema(const config& cfg);

/// Per-batch progress, owned by the caller and kept with the task across an out-of-memory replay
/// of the same batch, so that a replay never applies a row twice.
/// `progress` is the table's record of the batch (its scratch, the bitmap of the rows it could
/// not take): device memory of the table's resource, so a ticket must be destroyed before the
/// table's memory resource is, and should be dropped as soon as the batch is finished with. A
/// ticket belongs to one batch and one table; the replay must pass the same rows.
///   not_started  nothing of the batch is in the table (a call that threw left it so)
///   mutated      the table holds the batch's rows; what was left to do after that (side groups,
///                wide sums, building the result) is resumed by the next call
///   done         the call returned its result; a further call returns the same result again
///                without touching the table
struct fold_ticket {
  enum class state : uint8_t { not_started, mutated, done };
  state st          = state::not_started;
  uint64_t batch_id = 0;
  std::shared_ptr<void> progress;
};

struct fold_result {
  /// Input rows consumed: all rows but those listed in `unapplied`.
  uint64_t rows_applied = 0;
  /// Rows of the input the table could not take (null when all were applied): a gather map
  /// (INT32 row indices, ascending) into the input batch. The caller aggregates these on the cuDF
  /// path.
  std::unique_ptr<cudf::column> unapplied;
  /// Almost always null. Partial rows (partial_schema) of input rows that the table consumed and
  /// pre-aggregated but then found no room for: the caller merges them like the rows of a flush.
  /// (Only a fold() whose table is at its budget and whose batch has few groups per thread block
  /// produces them; they are counted in `rows_applied`.)
  std::unique_ptr<cudf::table> unapplied_partial;
};

class table {
 public:
  /// Allocates nothing large; the first fold sizes the table. `mr` and the current CUDA device
  /// at construction are the table's for life.
  static std::unique_ptr<table> create(const config& cfg, rmm::device_async_resource_ref mr);
  ~table();

  /// Folds one batch (`keys.size()` rows; `values` may have no columns). Keys may be NULL and may
  /// take any value of their type: NULL is a group of its own (as cudf::null_policy::INCLUDE).
  /// Thread-safe against other fold() / fold_partial() calls with distinct streams, and against
  /// device_bytes() and shrink_budget(). All scratch is allocated before the first kernel that
  /// changes the table; if an allocation throws after that, the ticket says how far the batch got
  /// and a second call with the same ticket resumes. `bounds[c]` (may be shorter than `values`)
  /// bounds values.column(c).
  fold_result fold(const cudf::column_view& keys,
                   const cudf::table_view& values,
                   const std::vector<column_bound>& bounds,
                   fold_ticket& ticket,
                   ::cuda::stream_ref stream);

  /// Folds partial rows (partial_schema; an integer SUM may also come as DECIMAL128 scale 0)
  /// produced by another table or by the cuDF path. Same threading and ticket rules as fold().
  fold_result fold_partial(const cudf::table_view& partial,
                           fold_ticket& ticket,
                           ::cuda::stream_ref stream);

  /// Not concurrent with fold(). Empties the table into partial rows (partial_schema), in chunks
  /// of at most `max_rows` rows (0: no limit). Every key appears in exactly one row. After it the
  /// table is empty and may be folded into again. If it throws (allocation), nothing is lost: the
  /// next flush() / emit() continues and returns all the rows.
  std::vector<std::unique_ptr<cudf::table>> flush(std::size_t max_rows,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref out_mr);

  /// Not concurrent with fold(). Final rows: key, then one result column per aggregate (the table
  /// is left empty). The partial state of every supported aggregate is its result, so these are
  /// the rows of flush(); they are final only if no fold since the last flush returned
  /// `unapplied` / `unapplied_partial` rows.
  std::vector<std::unique_ptr<cudf::table>> emit(std::size_t max_rows,
                                                 ::cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref out_mr);

  /// One chunk of a flush per call (at most `max_rows` rows; 0: as the table's segments come),
  /// null when the table is empty: flush() is this in a loop. Lower peak memory than flush():
  /// the caller can pass a chunk on before the next one exists. Same rules as flush(); the calls
  /// of one flush must not be interleaved with folds.
  std::unique_ptr<cudf::table> flush_next(std::size_t max_rows,
                                          ::cuda::stream_ref stream,
                                          rmm::device_async_resource_ref out_mr);

  /// Thread-safe, also against fold(); no CUDA call (any thread, whatever device is current).
  std::size_t device_bytes() const;  ///< memory the table holds now (not the tickets' scratch)
  uint64_t groups() const;           ///< groups resident now (as of the folds that returned)
  /// Thread-safe, also against fold(): the table stops growing at the new budget (it keeps what
  /// it holds; flush() gives the memory back).
  void shrink_budget(std::size_t budget_bytes);

  /// One line for logs: the state layout, the kernel family, growth steps, launches by kind.
  std::string describe() const;

  /// Tests only: the next `times` fold() / fold_partial() calls that apply rows throw
  /// rmm::out_of_memory after the table was changed (ticket `mutated`) and before the result is
  /// built; the replay with the same ticket must finish the batch without applying it again.
  void test_fail_after_mutation(int times);

 private:
  table();
  struct impl;
  std::unique_ptr<impl> _impl;
};

}  // namespace sirius::agg_table
