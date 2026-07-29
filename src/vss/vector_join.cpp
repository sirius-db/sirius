/*
 * Copyright 2025, Sirius Contributors.
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

#include "vss/vector_join.hpp"

#include "data/sirius_converter_registry.hpp"
#include "duckdb/common/exception.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/brute_force_search.hpp"
#include "vss/cudf_raft_interop.hpp"
#include "vss/distance_metric.hpp"
#include "vss/pinned_column.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <raft/core/device_resources.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <log/logging.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace sirius::vss {

namespace {

namespace ccm = cucascade::memory;

/// Resolved handles for one join run.
struct join_context {
  ccm::memory_space& space;
  const ccm::memory_space& host_space;
  rmm::device_async_resource_ref mr;
  rmm::cuda_stream_view stream;
};

/// Move a GPU result table to a host_data_representation the table function can stream out.
std::unique_ptr<cucascade::host_data_representation> join_result_d2h(
  const join_context& c, std::unique_ptr<cudf::table> table)
{
  cucascade::gpu_table_representation gpu_repr(std::move(table), c.space, c.stream);
  auto host_repr = converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, &c.host_space, c.stream);
  c.stream.synchronize();
  return host_repr;
}

struct side_layout {
  std::vector<std::string> staged;
  std::vector<int> output_slots;
};

// This avoids materializing vector column twice on device and tracks output column position.
side_layout plan_side_layout(const std::string& vec_col, const std::vector<std::string>& out_cols)
{
  side_layout layout;
  layout.staged.push_back(vec_col);
  layout.output_slots.reserve(out_cols.size());
  for (auto const& col : out_cols) {
    auto const it = std::find(layout.staged.begin(), layout.staged.end(), col);
    if (it == layout.staged.end()) {
      layout.output_slots.push_back(static_cast<int>(layout.staged.size()));
      layout.staged.push_back(col);
    } else {
      layout.output_slots.push_back(static_cast<int>(std::distance(layout.staged.begin(), it)));
    }
  }
  return layout;
}

/// View of @p staged reordered/duplicated into the requested output column order.
/// A slot referenced twice yields two views of the same column; the copy only
/// happens downstream in gather/repeat, where the output schema requires it.
cudf::table_view staged_as_output(const cudf::table_view& staged, const side_layout& layout)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(layout.output_slots.size());
  for (auto const slot : layout.output_slots) {
    cols.push_back(staged.column(slot));
  }
  return cudf::table_view{cols};
}

/// Return the positions of @p columns within the pinned column layout on host.
std::vector<std::size_t> get_pinned_column_indices(const scan_manager::pinned_entry& pin,
                                                   const std::vector<std::string>& columns)
{
  auto const& names = pin.cache_info.column_names();
  auto index_of     = [&](const std::string& name) -> std::size_t {
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == name) { return i; }
    }
    throw duckdb::InvalidInputException("sirius_knn_join: column '" + name +
                                        "' not found in pinned table");
  };
  std::vector<std::size_t> idx;
  idx.reserve(columns.size());
  for (auto const& col : columns) {
    idx.push_back(index_of(col));
  }
  return idx;
}

/// Upload the selected columns of one host chunk to the GPU as an owning table.
std::unique_ptr<cudf::table> upload_chunk_h2d(const cucascade::host_data_representation& chunk,
                                              const std::vector<std::size_t>& col_indices,
                                              ccm::memory_space& gpu_space,
                                              rmm::cuda_stream_view stream)
{
  auto sliced   = chunk.slice(col_indices);
  auto gpu_repr = converter_registry::get().convert<cucascade::gpu_table_representation>(
    *sliced, &gpu_space, stream);
  return gpu_repr->release_table(stream);
}

/// Estimate GPU footprint of the selected columns across all host chunks.
std::size_t host_columns_bytes(const scan_manager::pinned_entry& pin,
                               const std::vector<std::size_t>& col_indices)
{
  std::size_t total = 0;
  for (auto const& chunk : pin.host_chunks) {
    if (!chunk) { continue; }
    for (auto idx : col_indices) {
      total += chunk->column_size(idx);
    }
  }
  return total;
}

/// Borrowed views of a GPU-pinned table's staged columns and return chunk-major
std::vector<std::vector<cudf::column_view>> pinned_staged_chunk_columns(
  const scan_manager::pinned_entry& pin, const side_layout& layout, ccm::memory_space& space)
{
  std::vector<std::vector<cudf::column_view>> by_column;
  by_column.reserve(layout.staged.size());
  std::size_t n_chunks = 0;
  for (auto const& name : layout.staged) {
    auto views = pinned_column_chunk_views(pin, name, space);
    if (by_column.empty()) {
      n_chunks = views.size();
    } else if (views.size() != n_chunks) {
      throw duckdb::InvalidInputException(
        "sirius_knn_join: pinned table columns have inconsistent chunk counts");
    }
    by_column.push_back(std::move(views));
  }

  std::vector<std::vector<cudf::column_view>> by_chunk(n_chunks);
  for (std::size_t ci = 0; ci < n_chunks; ++ci) {
    by_chunk[ci].reserve(by_column.size());
    for (auto const& col : by_column) {
      by_chunk[ci].push_back(col[ci]);
    }
  }
  return by_chunk;
}

// TODO: cudf::concatenate copies even for a single input, so a one-chunk GPU corpus
// (the common case at the default 512 MiB scan batch) is duplicated on device for
// nothing. Borrow the chunk in place when it is unsliced and null-free.
std::unique_ptr<cudf::table> make_corpus_contiguous_on_device(const scan_manager::pinned_entry& pin,
                                                              const side_layout& layout,
                                                              ccm::memory_space& space,
                                                              const join_context& c)
{
  auto const by_chunk = pinned_staged_chunk_columns(pin, layout, space);
  std::vector<cudf::table_view> chunk_tables;
  chunk_tables.reserve(by_chunk.size());
  for (auto const& cols : by_chunk) {
    chunk_tables.emplace_back(cols);
  }
  return cudf::concatenate(chunk_tables, c.stream, c.mr);
}

/// Upload every chunk of HOST-pinned corpus to the GPU and concatenate into a resident.
std::unique_ptr<cudf::table> assemble_corpus_h2d(const scan_manager::pinned_entry& pin,
                                                 const std::vector<std::size_t>& col_indices,
                                                 ccm::memory_space& gpu_space,
                                                 const join_context& c)
{
  std::vector<std::unique_ptr<cudf::table>> uploaded;
  uploaded.reserve(pin.host_chunks.size());
  for (auto const& chunk : pin.host_chunks) {
    if (!chunk) { continue; }
    uploaded.push_back(upload_chunk_h2d(*chunk, col_indices, gpu_space, c.stream));
  }
  if (uploaded.empty()) {
    throw duckdb::InvalidInputException("sirius_knn_join: corpus has no host chunks to upload");
  }
  if (uploaded.size() == 1) { return std::move(uploaded.front()); }
  std::vector<cudf::table_view> views;
  views.reserve(uploaded.size());
  for (auto const& t : uploaded) {
    views.push_back(t->view());
  }
  return cudf::concatenate(views, c.stream, c.mr);
}

/// One searchable slice of the corpus: a staged table `[vector, distinct outputs...]`
/// whose vector column satisfies @ref list_column_as_dataset_view's preconditions.
/// @c owned backs the tile when it had to be materialized (uploaded or compacted);
/// it is null when the tile borrows GPU-resident pinned chunks.
struct corpus_tile {
  std::unique_ptr<cudf::table> owned;
  cudf::table_view staged;
  cudf::table_view pass;
  dataset_matrix_view dataset;
};

/// Hand @p staged to @p fn as a tile, compacting first if the vector column is sliced
/// or carries nulls (cuVS reads it as a raw blob, so it must be unsliced and gap-free).
/// Empty tiles are dropped rather than searched.
template <typename Fn>
void emit_corpus_tile(cudf::table_view staged,
                      std::unique_ptr<cudf::table> owned,
                      const side_layout& layout,
                      std::int64_t dim,
                      const join_context& c,
                      Fn&& fn)
{
  if (auto const& vec = staged.column(0); vec.offset() != 0 || vec.null_count() != 0) {
    auto valid = cudf::is_valid(vec, c.stream, c.mr);
    owned      = cudf::apply_boolean_mask(staged, valid->view(), c.stream, c.mr);
    staged     = owned->view();
  }
  if (staged.num_rows() == 0) { return; }
  fn(corpus_tile{std::move(owned),
                 staged,
                 staged_as_output(staged, layout),
                 list_column_as_dataset_view(staged.column(0), dim)});
}

/// Reduce the per-corpus-tile candidates for one probe chunk to the @p k nearest per
/// probe row. Each candidate table is `[probe_row_id, corpus outputs..., distance]` and
/// holds exactly @p m_per_row rows for every probe row once concatenated, so after
/// sorting by (probe row, distance) each probe row owns one fixed-size run and the
/// survivors are the first @p k of every run.
std::unique_ptr<cudf::table> merge_corpus_tile_candidates(
  std::vector<std::unique_ptr<cudf::table>> const& candidates,
  std::int64_t m_per_row,
  std::int64_t k,
  const join_context& c)
{
  std::vector<cudf::table_view> views;
  views.reserve(candidates.size());
  for (auto const& t : candidates) {
    views.push_back(t->view());
  }
  auto all = cudf::concatenate(views, c.stream, c.mr);

  auto const n_cols   = all->num_columns();
  auto const by_probe = all->view().select({0, n_cols - 1});
  auto order          = cudf::sorted_order(by_probe,
                                           {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                           {cudf::null_order::AFTER, cudf::null_order::AFTER},
                                  c.stream,
                                  c.mr);
  // TODO: this gather plus the apply_boolean_mask below materialize the full-width
  // table (including string outputs) twice per tile. Filter `order` down to the
  // survivors first, then gather once.
  auto sorted = cudf::gather(
    all->view(), order->view(), cudf::out_of_bounds_policy::DONT_CHECK, c.stream, c.mr);
  if (k >= m_per_row) { return sorted; }

  // Row i now sits at rank i % m_per_row within its probe row's run.
  // TODO: these counters are INT32, so the merge breaks past 2^31 candidate rows
  // (n_queries * m_per_row). Widen to INT64 or bound the probe chunk.
  cudf::numeric_scalar<std::int32_t> const zero{0, true, c.stream};
  cudf::numeric_scalar<std::int32_t> const one{1, true, c.stream};
  cudf::numeric_scalar<std::int32_t> const run{
    static_cast<std::int32_t>(m_per_row), true, c.stream};
  cudf::numeric_scalar<std::int32_t> const keep{static_cast<std::int32_t>(k), true, c.stream};
  auto positions = cudf::sequence(sorted->num_rows(), zero, one, c.stream, c.mr);
  auto rank      = cudf::binary_operation(positions->view(),
                                     run,
                                     cudf::binary_operator::MOD,
                                     cudf::data_type{cudf::type_id::INT32},
                                     c.stream,
                                     c.mr);
  auto mask      = cudf::binary_operation(rank->view(),
                                     keep,
                                     cudf::binary_operator::LESS,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     c.stream,
                                     c.mr);
  return cudf::apply_boolean_mask(sorted->view(), mask->view(), c.stream, c.mr);
}

/// Keep only rows whose trailing distance column is <= @p threshold.
std::unique_ptr<cudf::table> filter_by_threshold(std::unique_ptr<cudf::table> table,
                                                 float threshold,
                                                 const join_context& c)
{
  auto const dist = table->view().column(table->num_columns() - 1);
  cudf::numeric_scalar<float> eps{threshold, true, c.stream};
  auto mask = cudf::binary_operation(dist,
                                     eps,
                                     cudf::binary_operator::LESS_EQUAL,
                                     cudf::data_type{cudf::type_id::BOOL8},
                                     c.stream,
                                     c.mr);
  return cudf::apply_boolean_mask(table->view(), mask->view(), c.stream, c.mr);
}

/// Reduce per-row candidates to the @p k pairs with globally smallest distance.
/// Uses a partial top-k selection (O(N), not a full O(N log N) sort), then orders
/// just those k rows by distance so the result is ranked nearest-first.
std::unique_ptr<cudf::table> take_global_top_k(std::unique_ptr<cudf::table> table,
                                               std::int64_t k,
                                               const join_context& c)
{
  auto const view = table->view();
  auto const n    = static_cast<cudf::size_type>(std::min<std::int64_t>(k, view.num_rows()));
  if (n <= 0) { return table; }

  // Gather the k smallest distances without sorting the whole table.
  auto top_idx = cudf::top_k_order(
    view.column(view.num_columns() - 1), n, cudf::order::ASCENDING, c.stream, c.mr);
  auto top =
    cudf::gather(view, top_idx->view(), cudf::out_of_bounds_policy::DONT_CHECK, c.stream, c.mr);

  // Order those k rows by distance (top_k_order does not guarantee sorted output).
  auto order = cudf::sorted_order(cudf::table_view{{top->view().column(top->num_columns() - 1)}},
                                  {cudf::order::ASCENDING},
                                  {cudf::null_order::AFTER},
                                  c.stream,
                                  c.mr);
  return cudf::gather(
    top->view(), order->view(), cudf::out_of_bounds_policy::DONT_CHECK, c.stream, c.mr);
}

}  // namespace

std::vector<std::unique_ptr<cucascade::host_data_representation>> run_vector_join(
  duckdb::SiriusContext& ctx, const vector_join_request& req)
{
  auto& memory_manager = ctx.get_memory_manager();
  auto gpu_spaces      = memory_manager.get_memory_spaces_for_tier(ccm::Tier::GPU);
  if (gpu_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_knn_join: no GPU memory space available");
  }
  auto* space          = const_cast<ccm::memory_space*>(gpu_spaces.front());
  int const target_gpu = space->get_device_id();
  rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{target_gpu}};
  auto mr = space->get_default_allocator();
  // The GPU->host converter's cudaMemcpyBatchAsync rejects the default stream.
  // Synchronized before we return, so the owned stream is safe to destroy on exit.
  rmm::cuda_stream stream_owner;
  auto stream = stream_owner.view();

  auto host_spaces = memory_manager.get_memory_spaces_for_tier(ccm::Tier::HOST);
  if (host_spaces.empty()) {
    throw duckdb::InvalidInputException("sirius_knn_join: no HOST memory space available");
  }
  join_context join_cxt{*space, *host_spaces.front(), mr, stream};

  // Both tables must be pinned on either tier (host or device)
  auto const* probe_pin = ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    req.probe_catalog, req.probe_schema, req.probe_table);
  auto const* corpus_pin = ctx.get_scan_manager().find_pinned_entry_for_duckdb_table(
    req.corpus_catalog, req.corpus_schema, req.corpus_table);
  if (probe_pin == nullptr) {
    throw duckdb::InvalidInputException("sirius_knn_join: probe table '" + req.probe_table +
                                        "' must be pinned (pin_table on the GPU or HOST tier)");
  }
  if (corpus_pin == nullptr) {
    throw duckdb::InvalidInputException("sirius_knn_join: corpus table '" + req.corpus_table +
                                        "' must be pinned (pin_table on the GPU or HOST tier)");
  }
  if (probe_pin->num_rows == 0 || corpus_pin->num_rows == 0) { return {}; }

  auto const corpus_layout = plan_side_layout(req.corpus_vector_column, req.corpus_output_columns);
  auto const probe_layout  = plan_side_layout(req.probe_vector_column, req.probe_output_columns);

  // If the corpus fits in free GPU memory, copy it over in one piece and search it
  // in one call per probe chunk. If it doesn't fit, split it and search each piece,
  // then merge each probe row's top-k across pieces. A corpus already on the GPU
  // splits with no copying; a host corpus is re-uploaded once per probe chunk.
  if (corpus_pin->tier != ccm::Tier::GPU && corpus_pin->tier != ccm::Tier::HOST) {
    throw duckdb::InvalidInputException("sirius_knn_join: corpus '" + req.corpus_table +
                                        "' is on an unsupported memory tier");
  }
  auto const corpus_on_gpu  = corpus_pin->tier == ccm::Tier::GPU;
  auto const corpus_col_idx = corpus_on_gpu
                                ? std::vector<std::size_t>{}
                                : get_pinned_column_indices(*corpus_pin, corpus_layout.staged);
  auto const corpus_avail   = req.max_stage_bytes.value_or(space->get_available_memory());
  // TODO: the estimate is wrong in both branches. The GPU branch counts only the vector
  // column, but make_corpus_contiguous_on_device concatenates every staged column; the
  // HOST branch peaks at ~2x because assemble_corpus_h2d holds all uploaded chunks and
  // then concatenates. Both can OOM despite passing this check.
  auto const corpus_stage_est =
    corpus_on_gpu ? corpus_pin->num_rows * static_cast<std::size_t>(req.dim) * sizeof(float)
                  : host_columns_bytes(*corpus_pin, corpus_col_idx);
  // TODO: this is advisory only -- nothing reserves the memory, so another pipeline can
  // take it between here and the staging copy. Use space->make_reservation_or_null() and
  // fall back to the tiled path when it returns null (see cuvs_index_cache).
  auto const tile_corpus = corpus_stage_est > corpus_avail;

  SIRIUS_LOG_DEBUG(
    "sirius_knn_join: corpus '{}' on {} ({} rows); staging needs ~{} Bytes, GPU available {} "
    "Bytes -> {}",
    req.corpus_table,
    corpus_on_gpu ? "GPU" : "HOST",
    corpus_pin->num_rows,
    corpus_stage_est,
    corpus_avail,
    tile_corpus ? "tiled" : "contiguous");

  // Set when the corpus is staged as a whole, null if tiled
  std::unique_ptr<cudf::table> corpus_resident;
  // Set when tiling a GPU-resident corpus
  std::vector<std::vector<cudf::column_view>> corpus_chunk;
  std::size_t n_corpus_tiles = 1;

  // Case 1: whole corpus (the contiguous copy) fits on GPU memory
  if (!tile_corpus) {
    corpus_resident =
      corpus_on_gpu ? make_corpus_contiguous_on_device(*corpus_pin, corpus_layout, *space, join_cxt)
                    : assemble_corpus_h2d(*corpus_pin, corpus_col_idx, *space, join_cxt);
    // Drop null rows
    if (corpus_resident->view().column(0).null_count() != 0) {
      auto valid_mask = cudf::is_valid(corpus_resident->view().column(0), stream, mr);
      corpus_resident =
        cudf::apply_boolean_mask(corpus_resident->view(), valid_mask->view(), stream, mr);
    }
    if (corpus_resident->num_rows() == 0) { return {}; }
  }

  // Case 2: corpus chunks are pinned on device, but the contiguous copy needs to be tiled
  else if (corpus_on_gpu) {
    corpus_chunk   = pinned_staged_chunk_columns(*corpus_pin, corpus_layout, *space);
    n_corpus_tiles = corpus_chunk.size();
    // Borrowed tiles are searched in place, a sliced or nullable vector column has to be compacted
    auto const needs_compaction =
      std::any_of(corpus_chunk.begin(), corpus_chunk.end(), [](auto const& cols) {
        return cols[0].offset() != 0 || cols[0].null_count() != 0;
      });
    if (needs_compaction) {
      SIRIUS_LOG_WARN(
        "sirius_knn_join: corpus '{}' is tiled but its vector column is sliced or nullable; each "
        "tile is compacted once per probe chunk",
        req.corpus_table);
    }
  }

  // Case 3: corpus can only be pinned on the host (not enough space on device)
  // Note: the corpus is re-uploaded for every probe chunk because it's on the inner loop
  else {
    n_corpus_tiles = 0;
    for (auto const& chunk : corpus_pin->host_chunks) {
      if (chunk) { ++n_corpus_tiles; }
    }
    SIRIUS_LOG_WARN(
      "sirius_knn_join: corpus '{}' does not fit on the GPU; streaming {} tiles per probe chunk "
      "(the corpus is re-uploaded for every probe chunk)",
      req.corpus_table,
      n_corpus_tiles);
  }
  if (n_corpus_tiles == 0) { return {}; }

  // Run fn over every corpus tile. Resident tiles are handed out as borrowed views;
  auto for_each_corpus_tile = [&](auto&& fn) {
    // Case 1
    if (corpus_resident) {
      emit_corpus_tile(corpus_resident->view(), nullptr, corpus_layout, req.dim, join_cxt, fn);
    }
    // Case 2
    else if (corpus_on_gpu) {
      for (auto const& cols : corpus_chunk) {
        emit_corpus_tile(cudf::table_view{cols}, nullptr, corpus_layout, req.dim, join_cxt, fn);
      }
    }
    // Case 3 (HOST tiles are uploaded on demand and freed as soon as fn returns)
    else {
      for (auto const& chunk : corpus_pin->host_chunks) {
        if (!chunk) { continue; }
        auto uploaded     = upload_chunk_h2d(*chunk, corpus_col_idx, *space, stream);
        auto const staged = uploaded->view();
        emit_corpus_tile(staged, std::move(uploaded), corpus_layout, req.dim, join_cxt, fn);
      }
    }
  };

  // Candidates are merged per probe row only when more than one tile contributes;
  // a single tile is already query-major and distance-sorted.
  auto const tiled = n_corpus_tiles > 1;

  // One RAFT handle reused across probe chunks (workspace setup paid once). Each
  // per-chunk search runs async on c.stream; res must outlive the join sync.
  raft::device_resources res{stream};

  // Search one probe chunk against every corpus tile and emit its pairs. cuVS returns
  // query-major [q * k_t] neighbors per tile, so gather(tile, neighbors) lines up
  // row-for-row with that tile's distances, and repeat(probe, kept) lines up with the
  // merged result once every probe row holds the same number of survivors.
  //
  // In per-row mode, a chunk's pairs are final once searched, so they go straight to the
  // host and the device memory is reclaimed before the next chunk. Global mode instead
  // folds each chunk into a running top-k, so at most k + one chunk of pairs is live.
  std::vector<std::unique_ptr<cucascade::host_data_representation>> outputs;
  std::unique_ptr<cudf::table> global_best;
  auto process_probe_chunk = [&](cudf::table_view probe_chunk) {
    std::unique_ptr<cudf::table> compacted;
    if (auto const& vec = probe_chunk.column(0); vec.offset() != 0 || vec.null_count() != 0) {
      auto valid_mask = cudf::is_valid(vec, stream, mr);
      compacted       = cudf::apply_boolean_mask(probe_chunk, valid_mask->view(), stream, mr);
      probe_chunk     = compacted->view();
    }
    if (probe_chunk.num_rows() == 0) { return; }

    auto const n_queries = probe_chunk.num_rows();
    auto const queries   = list_column_as_dataset_view(probe_chunk.column(0), req.dim);

    // Fold each tile into a running best-so-far rather than collecting every tile's
    // candidates first: only the survivors plus one tile's worth are ever live.
    // Layout is [probe_row_id (tiled only), corpus outputs..., distance], with exactly
    // `kept` rows for every probe row.
    std::unique_ptr<cudf::table> corpus_side;
    std::int64_t kept = 0;
    for_each_corpus_tile([&](const corpus_tile& tile) {
      auto const k_t = std::min<std::int64_t>(tile.staged.num_rows(), req.k);
      if (k_t <= 0) { return; }
      auto knn = brute_force_knn(
        res, tile.dataset, queries, k_t, enn_distance_type_from_metric(req.metric), mr);
      auto matched = cudf::gather(
        tile.pass, knn.neighbors->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

      auto matched_cols = matched->release();
      std::vector<std::unique_ptr<cudf::column>> cols;
      cols.reserve(matched_cols.size() + 2);
      if (tiled) {
        // Query q owns rows [q * k_t, (q + 1) * k_t), so repeating an iota by k_t
        // labels each candidate with the probe row it belongs to.
        cudf::numeric_scalar<std::int32_t> const zero{0, true, stream};
        cudf::numeric_scalar<std::int32_t> const one{1, true, stream};
        auto ids = cudf::sequence(n_queries, zero, one, stream, mr);
        auto id_cols =
          cudf::repeat(
            cudf::table_view{{ids->view()}}, static_cast<cudf::size_type>(k_t), stream, mr)
            ->release();
        cols.push_back(std::move(id_cols.front()));
      }
      for (auto& col : matched_cols) {
        cols.push_back(std::move(col));
      }
      cols.push_back(std::move(knn.distances));
      auto cand = std::make_unique<cudf::table>(std::move(cols));

      if (!corpus_side) {
        corpus_side = std::move(cand);
        kept        = k_t;
        return;
      }
      std::vector<std::unique_ptr<cudf::table>> pair;
      pair.push_back(std::move(corpus_side));
      pair.push_back(std::move(cand));
      auto const m = kept + k_t;
      corpus_side  = merge_corpus_tile_candidates(pair, m, req.k, join_cxt);
      kept         = std::min<std::int64_t>(m, req.k);
    });
    if (!corpus_side) { return; }

    auto corpus_cols = corpus_side->release();
    // Drop the probe_row_id label added for the merge; it is not part of the schema.
    if (tiled) { corpus_cols.erase(corpus_cols.begin()); }

    auto probe_repeated = cudf::repeat(
      staged_as_output(probe_chunk, probe_layout), static_cast<cudf::size_type>(kept), stream, mr);

    auto probe_cols = probe_repeated->release();
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(probe_cols.size() + corpus_cols.size());
    for (auto& col : probe_cols) {
      out_cols.push_back(std::move(col));
    }
    for (auto& col : corpus_cols) {
      out_cols.push_back(std::move(col));
    }
    auto chunk_out = std::make_unique<cudf::table>(std::move(out_cols));
    if (req.threshold) {
      chunk_out = filter_by_threshold(std::move(chunk_out), *req.threshold, join_cxt);
    }
    if (chunk_out->num_rows() == 0) { return; }

    if (req.global) {
      chunk_out = take_global_top_k(std::move(chunk_out), req.k, join_cxt);
      if (global_best) {
        std::vector<cudf::table_view> const both{global_best->view(), chunk_out->view()};
        chunk_out = take_global_top_k(cudf::concatenate(both, stream, mr), req.k, join_cxt);
      }
      global_best = std::move(chunk_out);
      return;
    }
    outputs.push_back(join_result_d2h(join_cxt, std::move(chunk_out)));
  };

  // --- Probe: iterate resident chunks, or stream host chunks up per batch ---
  if (probe_pin->tier == ccm::Tier::GPU) {
    SIRIUS_LOG_DEBUG(
      "sirius_knn_join: probe '{}' GPU-resident ({} rows)", req.probe_table, probe_pin->num_rows);
    std::vector<std::vector<cudf::column_view>> probe_staged_chunks;
    probe_staged_chunks.reserve(probe_layout.staged.size());
    std::size_t n_probe_chunks = 0;
    for (auto const& name : probe_layout.staged) {
      auto views = pinned_column_chunk_views(*probe_pin, name, *space);
      if (probe_staged_chunks.empty()) {
        n_probe_chunks = views.size();
      } else if (views.size() != n_probe_chunks) {
        throw duckdb::InvalidInputException(
          "sirius_knn_join: probe columns have inconsistent chunk counts");
      }
      probe_staged_chunks.push_back(std::move(views));
    }
    for (std::size_t ci = 0; ci < n_probe_chunks; ++ci) {
      std::vector<cudf::column_view> pcols;
      pcols.reserve(probe_staged_chunks.size());
      for (auto const& sc : probe_staged_chunks) {
        pcols.push_back(sc[ci]);
      }
      process_probe_chunk(cudf::table_view{pcols});
    }
  } else if (probe_pin->tier == ccm::Tier::HOST) {
    auto const col_idx  = get_pinned_column_indices(*probe_pin, probe_layout.staged);
    auto const n_chunks = probe_pin->host_chunks.size();
    SIRIUS_LOG_DEBUG("sirius_knn_join: streaming probe '{}' from HOST ({} chunks, {} rows)",
                     req.probe_table,
                     n_chunks,
                     probe_pin->num_rows);
    std::size_t ci = 0;
    for (auto const& chunk : probe_pin->host_chunks) {
      ++ci;
      if (!chunk) { continue; }
      // Upload this batch, search it, and free it immediately.
      auto uploaded = upload_chunk_h2d(*chunk, col_idx, *space, stream);
      SIRIUS_LOG_DEBUG("sirius_knn_join: probe chunk {}/{} uploaded ({} rows)",
                       ci,
                       n_chunks,
                       uploaded->num_rows());
      process_probe_chunk(uploaded->view());
    }
  } else {
    throw duckdb::InvalidInputException("sirius_knn_join: probe '" + req.probe_table +
                                        "' is on an unsupported memory tier");
  }

  if (global_best) { outputs.push_back(join_result_d2h(join_cxt, std::move(global_best))); }
  return outputs;
}

}  // namespace sirius::vss
