/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_partition_consumer_operator.hpp"
#include "telemetry/data_batch_probe.hpp"
#include "vss/vector_join.hpp"
#include "vss/vector_join_materialized_side.hpp"

#include <cudf/column/column_view.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/device_buffer.hpp>

#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::scan_manager {
class sirius_scan_manager;
struct pinned_entry;
}  // namespace sirius::scan_manager

namespace cucascade {
class data_batch;
namespace memory {
class reservation;
}  // namespace memory
}  // namespace cucascade

namespace sirius::vss {
struct cluster_lists;
}  // namespace sirius::vss

namespace sirius::op {

/**
 * @brief One corpus chunk made device-resident for the duration of a single fold step.
 *
 * @c owner is null when the chunk was already device-resident (GPU-tier pin); otherwise it
 * holds the staged copy alive and releases it when the fold step drops it, which is what
 * bounds device memory to the chunks in flight rather than the whole corpus.
 */
struct staged_vector_chunk {
  cudf::column_view view;
  std::shared_ptr<::cucascade::data_batch> owner;
  /// Draws the staged copy from the task's device budget instead of committing fresh
  /// capacity; must outlive @c owner, so it is released alongside it.
  std::shared_ptr<::cucascade::memory::reservation> reservation;
  /// Held only when @c view points into a batch this chunk does not own -- a build-side batch
  /// that was already device-resident. The shared lock is what stops the downgrade executor
  /// spilling that batch out from under the fold; dropping it ends the borrow.
  std::optional<::cucascade::read_only_data_batch> reader;
  /// A staged copy this chunk made itself rather than through a data batch. Allocated from
  /// @c reservation, so it is declared after it and freed first.
  std::unique_ptr<rmm::device_buffer> buffer;
};

/**
 * @brief Where the streamed (corpus) side's chunks come from.
 *
 * The fold loop only needs "hand me chunk i, device-resident, and take it back when I am
 * done". Isolating that behind this interface is what lets the corpus live on the GPU, in
 * host memory, or -- later -- behind an ordinary child scan, without the fold changing.
 */
class vector_chunk_source {
 public:
  vector_chunk_source()                                      = default;
  vector_chunk_source(const vector_chunk_source&)            = delete;
  vector_chunk_source& operator=(const vector_chunk_source&) = delete;
  virtual ~vector_chunk_source()                             = default;

  [[nodiscard]] virtual std::size_t num_chunks() const = 0;

  /// Make chunk @p i device-resident in @p space. Called once per chunk per left batch.
  virtual staged_vector_chunk stage(std::size_t i,
                                    ::cucascade::memory::memory_space& space,
                                    rmm::cuda_stream_view stream) = 0;

  /// True when staging performs a host-to-device copy, i.e. the data is not resident.
  [[nodiscard]] virtual bool is_streaming() const = 0;

  /// Rows in chunk @p i, and its device footprint, both without staging it. The memory
  /// estimate has to size a task before any copy happens.
  [[nodiscard]] virtual std::size_t chunk_rows(std::size_t i) const  = 0;
  [[nodiscard]] virtual std::size_t chunk_bytes(std::size_t i) const = 0;
};

/// GPU-tier pin: chunks are already device-resident, so staging is a no-op view.
std::unique_ptr<vector_chunk_source> make_gpu_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  ::cucascade::memory::memory_space& space);

/// HOST-tier pin: each chunk is copied device-side on demand and released after the fold
/// step, which is what makes the corpus side out-of-core.
std::unique_ptr<vector_chunk_source> make_host_pinned_chunk_source(
  const sirius::scan_manager::pinned_entry& pin,
  const std::string& column,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info);

/// Cluster lists: chunks are fixed-size spans of the cluster-ordered copy, viewed in place on the
/// GPU tier and copied block by block from pinned host memory on the HOST tier.
std::unique_ptr<vector_chunk_source> make_cluster_lists_chunk_source(
  const sirius::vss::cluster_lists& lists);

/// Build phase: chunks are the batches a child scan deposited in the join's build port, taken
/// in the buffer's snapshot order. A batch the downgrade executor has spilled is brought back
/// device-side for the fold step and released after it, exactly as a HOST-tier pin chunk is;
/// one still resident is viewed in place.
std::unique_ptr<vector_chunk_source> make_materialized_chunk_source(
  sirius::vss::materialized_side_buffer& buffer,
  std::size_t column_index,
  std::int64_t dim,
  const telemetry::batch_telemetry_info& telemetry_info,
  duckdb::SiriusContext* ctx = nullptr);

/**
 * @brief The input handed to one sirius_physical_vector_join_stream::execute() call.
 *
 * One task per left (query) batch, not per (left, right) pair: the task streams every
 * right batch through a running top-k fold, so the pair grid is walked inside execute()
 * instead of being handed out as separate tasks.
 */
/// Task input: which left (probe) batch this task searches. The probe batch itself stays idle in
/// the "default" port until execute() stages it, so the engine can spill it meanwhile.
class vector_join_stream_input : public operator_data {
 public:
  vector_join_stream_input(std::size_t left_idx, std::size_t estimated_bytes)
    : _left_idx(left_idx), _estimated_bytes(estimated_bytes)
  {
  }

  [[nodiscard]] operator_data_type get_type() const override { return operator_data_type::BASE; }

  void prepare_for_processing(const ::cucascade::memory::memory_space* requested_memory_space,
                              ::cuda::stream_ref /*stream*/) override
  {
    _gpu_memory_space = const_cast<::cucascade::memory::memory_space*>(requested_memory_space);
  }

  [[nodiscard]] std::size_t get_estimated_size_in_bytes() const override
  {
    return _estimated_bytes;
  }

  [[nodiscard]] ::cucascade::memory::memory_space* get_gpu_memory_space() const
  {
    return _gpu_memory_space;
  }

  /// Index of this task's left batch; also the output partition, so the materialize
  /// stage can gather that batch's left columns.
  [[nodiscard]] std::size_t left_idx() const { return _left_idx; }

 private:
  std::size_t _left_idx;
  ::cucascade::memory::memory_space* _gpu_memory_space = nullptr;
  std::size_t _estimated_bytes;
};

/**
 * @brief Streaming exact k-nearest-neighbor vector join: search and merge fused.
 *
 * Replaces the VECTOR_JOIN_SELECT -> VECTOR_JOIN_REDUCE_LOCAL pair. Each task takes
 * one left (query) batch, keeps an `[n_left x k]` top-k accumulator, and folds every
 * right batch into it as it is searched, releasing each right batch's partial
 * immediately. The fold reuses cuVS `knn_merge_parts` with `n_parts = 2` (accumulator
 * + the batch just searched), so no new kernel is needed.
 *
 * Why this shape: the split design emitted one `[n_left x k]` partial per
 * (left batch, right batch) pair through the batch repo, so intermediate data was
 * `n_right_batches` times the final answer and the merge stage had to hold all of a
 * partition's partials at once. Folding in place makes device memory independent of
 * the number of right batches, which is what makes the operator out-of-core; see
 * `docs` note in vector_join.hpp and the measured residency/throughput tradeoff.
 *
 * Parallelism is across left batches: each task owns its own accumulator, so there is
 * no shared state and no lock on the hot path.
 *
 * Output matches what the merge stage used to emit -- `[neighbor_id INT64,
 * distance FLOAT32]` flattened `[n_left * k]`, partitioned by left batch index -- so
 * the materialize stage is unchanged.
 */
class sirius_physical_vector_join_stream : public sirius_physical_partition_consumer_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::VECTOR_JOIN_STREAM;

  /// @param build_side  When non-null the corpus comes from this operator's build port -- a
  ///                    child scan materialized by the build phase -- instead of a pinned
  ///                    catalog table, and the buffer is the row order both this operator and
  ///                    materialize resolve neighbour ids against.
  /// @param sirius_ctx  Where the clustered path reports what it pruned. Only the planner can
  ///                    resolve it, as with @p centroids; null on the exhaustive path, which
  ///                    prunes nothing.
  sirius_physical_vector_join_stream(
    duckdb::vector<sirius::logical_type> types,
    duckdb::idx_t estimated_cardinality,
    sirius::vss::vector_join_request request,
    sirius::scan_manager::sirius_scan_manager* scan_manager,
    std::shared_ptr<sirius::vss::materialized_side_buffer> build_side = nullptr,
    std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side = nullptr,
    const cudf::column* centroids                                     = nullptr,
    duckdb::SiriusContext* sirius_ctx                                 = nullptr);

  [[nodiscard]] const sirius::vss::vector_join_request& request() const { return _request; }

  /// True when the corpus is fed by a child scan rather than resolved from a pin.
  [[nodiscard]] bool has_build_phase() const { return _build_side != nullptr; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  // -----------------------------
  // Source interface
  // -----------------------------
  bool is_source() const override { return true; }

  //! The build-side CONCAT feeds the corpus into the "build" port; everything else is the probe.
  [[nodiscard]] std::string_view input_port_for(
    sirius_physical_operator const& producer) const override;

  std::optional<task_creation_hint> get_next_task_hint() override;
  [[nodiscard]] bool all_ports_empty() override;
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  // -----------------------------
  // Execution
  // -----------------------------
  /// Streams every right batch through this left batch's running top-k fold and
  /// returns the finished `[n_left * k]` result.
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         ::cuda::stream_ref stream) override;

  /// Routes the finished result to the materialize stage under its left batch index.
  void sink(const operator_data& output_data, ::cuda::stream_ref stream) override;

  [[nodiscard]] std::size_t no_history_peak_memory_estimate(
    const input_stats& stats) const override;

  std::string params_to_string() const override;

 private:
  /// Resolve both pinned tables and snapshot per-batch views plus right-batch row
  /// offsets. Idempotent; caller holds _op_mutex. On the build path this is a no-op until
  /// @ref build_side_ready_locked, so callers must check that first.
  void ensure_initialized_locked();

  /// Build @c _chunk_cluster_runs and @c _cluster_rows on first use. Deferred out of init
  /// because reading the cluster column needs a memory space, which only a running task
  /// supplies.
  void ensure_cluster_index(::cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);

  /// Whether the build port is wired and its producing pipeline has finished. Always true on
  /// the pinned path. Caller holds _op_mutex.
  bool build_side_ready_locked();

  /// Peak bytes for one left batch's task: the accumulator, the partial being folded
  /// in, and the stacked pair the merge reads. Independent of the right batch count,
  /// which is the point of the fold.
  [[nodiscard]] std::size_t per_left_batch_estimate(std::size_t left_idx,
                                                    bool with_routing = true) const;

  /// The corpus row order, shared with materialize. Null on the pinned path, where the
  /// pinned_entry plays the same role.
  std::shared_ptr<sirius::vss::materialized_side_buffer> _build_side;
  /// The probe row order, shared with materialize the same way. A probe chunk is read once, so
  /// this side needs no re-reads -- but its batch order still names the output partitions that
  /// materialize gathers left columns by, so the two must agree on it just the same.
  std::shared_ptr<sirius::vss::materialized_side_buffer> _probe_side;

  sirius::vss::vector_join_request _request;
  sirius::scan_manager::sirius_scan_manager* _scan_manager;
  /// Centroids of @c _request.clustering, owned by the session's index cache and valid until
  /// that entry is dropped. Null for an exhaustive join, which is what selects the fold's
  /// visit-everything path.
  const cudf::column* _centroids{nullptr};
  /// The corpus in cluster order, when the clustering was built into lists rather than carried
  /// as a column of the table. Owned by the index cache, like the centroids.
  const sirius::vss::cluster_lists* _lists{nullptr};
  /// HOST-tier lists registered in this query's repository manager under port "lists", which
  /// is what lets the downgrade executor spill them when the host fills; emptied when the
  /// operator finishes so the manager does not report them as leaked.
  ::cucascade::shared_data_repository* _lists_repo{nullptr};
  void on_finalize_operator() override;
  /// The pinned corpus, when there is one: where pushed-down corpus predicates read their
  /// columns. Null on the build path.
  std::shared_ptr<const sirius::scan_manager::pinned_entry> _right_pin;
  /// Session state: the clustered path reports its prune statistics to it, and a streamed build
  /// side asks its downgrade executor for room. Outlives the query.
  duckdb::SiriusContext* _sirius_ctx{nullptr};
  /// One contiguous run of a single cluster inside one corpus chunk. Rows are local to the
  /// chunk, so a slice can be searched the moment that chunk is staged and needs nothing from
  /// any other; @c _chunk_row_base turns its neighbour ids back into corpus row ids.
  struct chunk_cluster_run {
    std::int32_t cluster;
    std::int64_t begin;
    std::int64_t end;
  };
  /// Per corpus chunk, the cluster runs it holds, in row order. This is why pruning works at
  /// cluster granularity rather than chunk granularity: a cluster is a slice of a chunk, so
  /// visiting one costs a slice rather than a whole chunk. Keyed by chunk rather than by
  /// cluster because staging is per chunk -- the search walks chunks, and a chunk no probe run
  /// wants is never staged at all.
  std::vector<std::vector<chunk_cluster_run>> _chunk_cluster_runs;
  /// Per corpus chunk, its first row in corpus row space. A slice's neighbour ids come back
  /// local to the slice; this plus the slice's own begin is the base that makes them corpus
  /// row ids, exactly as the chunk offset does in the exhaustive fold.
  std::vector<std::int64_t> _chunk_row_base;
  /// Per cluster id, how many corpus rows it holds across every chunk. Lets a probe row's
  /// candidate count be known before any search is issued.
  std::vector<std::int64_t> _cluster_rows;
  std::int64_t _n_clusters{0};
  bool _cluster_index_built{false};
  std::mutex _op_mutex;
  bool _initialized{false};
  bool _hint_returned{false};
  std::unique_ptr<vector_chunk_source> _probe;
  //! Streamed side, behind the tier-agnostic seam.
  std::unique_ptr<vector_chunk_source> _corpus;
  //! Total right-table rows, used to clamp k the way the plan already does.
  std::int64_t _right_total_rows{0};
  //! Largest corpus chunk in bytes; reserved per task when the corpus is streamed.
  std::size_t _max_chunk_bytes{0};
  std::size_t _max_probe_chunk_bytes{0};
  std::size_t _num_left{0};
  /// Probe rows in total, counted only when the probe stands in for a scalar subquery.
  std::size_t _scalar_probe_rows{0};
  std::size_t _next_left{0};
};

}  // namespace sirius::op
