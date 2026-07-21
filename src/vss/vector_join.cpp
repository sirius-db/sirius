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
#include <cudf/column/column_factories.hpp>
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

/// Positions of [vec_col, out_cols...] within the pinned column layout on host.
std::vector<std::size_t> get_pinned_column_indices(const scan_manager::pinned_entry& pin,
                                                   const std::string& vec_col,
                                                   const std::vector<std::string>& out_cols)
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
  idx.reserve(out_cols.size() + 1);
  idx.push_back(index_of(vec_col));
  for (auto const& col : out_cols) {
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

/// An empty column matching @p name's type in @p pin.
std::unique_ptr<cudf::column> empty_like_pinned_column(const scan_manager::pinned_entry& pin,
                                                       const std::string& name,
                                                       const join_context& c)
{
  // Device path
  if (pin.tier == ccm::Tier::GPU) {
    auto it = pin.data_batches_by_column.find(name);
    if (it == pin.data_batches_by_column.end() || it->second.empty()) {
      throw duckdb::InvalidInputException("sirius_knn_join: pinned table missing output column '" +
                                          name + "'");
    }
    return cudf::empty_like(it->second.front()->view());
  }
  if (pin.host_chunks.empty() || !pin.host_chunks.front()) {
    throw duckdb::InvalidInputException("sirius_knn_join: cannot resolve type for column '" + name +
                                        "' (no host chunks)");
  }

  // Host path
  auto const& names = pin.cache_info.column_names();
  std::size_t idx   = names.size();
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) {
      idx = i;
      break;
    }
  }
  if (idx == names.size()) {
    throw duckdb::InvalidInputException("sirius_knn_join: column '" + name +
                                        "' not found in pinned table");
  }
  auto uploaded =
    upload_chunk_h2d(*pin.host_chunks.front(), std::vector<std::size_t>{idx}, c.space, c.stream);
  return cudf::empty_like(uploaded->view().column(0));
}

/// Empty result with the join's schema ([probe cols..., corpus cols..., distance]).
std::unique_ptr<cudf::table> make_empty_join_output(const scan_manager::pinned_entry& probe_pin,
                                                    const scan_manager::pinned_entry& corpus_pin,
                                                    const vector_join_request& req,
                                                    const join_context& c)
{
  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(req.probe_output_columns.size() + req.corpus_output_columns.size() + 1);
  for (auto const& name : req.probe_output_columns) {
    cols.push_back(empty_like_pinned_column(probe_pin, name, c));
  }
  for (auto const& name : req.corpus_output_columns) {
    cols.push_back(empty_like_pinned_column(corpus_pin, name, c));
  }
  cols.push_back(cudf::make_empty_column(cudf::data_type{cudf::type_id::FLOAT32}));
  return std::make_unique<cudf::table>(std::move(cols));
}

/// Estimated GPU footprint of the selected columns across all host chunks.
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

std::unique_ptr<cudf::table> make_corpus_contiguous_on_device(const scan_manager::pinned_entry& pin,
                                                              const vector_join_request& req,
                                                              ccm::memory_space& space,
                                                              const join_context& c)
{
  // Fetch chunks from GPU, in column major
  auto const vec_chunks = pinned_column_chunk_views(pin, req.corpus_vector_column, space);
  auto const n_chunks   = vec_chunks.size();
  std::vector<std::vector<cudf::column_view>> out_chunks;
  out_chunks.reserve(req.corpus_output_columns.size());
  for (auto const& name : req.corpus_output_columns) {
    auto views = pinned_column_chunk_views(pin, name, space);
    if (views.size() != n_chunks) {
      throw duckdb::InvalidInputException(
        "sirius_knn_join: corpus columns have inconsistent chunk counts");
    }
    out_chunks.push_back(std::move(views));
  }

  // Make chunk major
  std::vector<std::vector<cudf::column_view>> chunk_cols(n_chunks);
  std::vector<cudf::table_view> chunk_tables;
  chunk_tables.reserve(n_chunks);
  for (std::size_t ci = 0; ci < n_chunks; ++ci) {
    auto& cols = chunk_cols[ci];
    cols.reserve(out_chunks.size() + 1);
    cols.push_back(vec_chunks[ci]);
    for (auto const& oc : out_chunks) {
      cols.push_back(oc[ci]);
    }
    chunk_tables.emplace_back(cols);
  }

  // Concat chunks into contiguous space
  return cudf::concatenate(chunk_tables, c.stream, c.mr);
}

/// Upload every chunk of HOST-pinned corpus to the GPU and concatenate into a resident.
std::unique_ptr<cudf::table> assemble_corpus_h2d(const scan_manager::pinned_entry& pin,
                                                 const std::vector<std::size_t>& col_indices,
                                                 ccm::memory_space& gpu_space,
                                                 const join_context& c)
{
  // TODO: skip the intermediate and compute the total row count upfront, allocate one contiguous
  // target table and upload each chunk directly into it
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
std::unique_ptr<cudf::table> take_global_top_k(std::unique_ptr<cudf::table> table,
                                               std::int64_t k,
                                               const join_context& c)
{
  auto const view      = table->view();
  auto const& dist_col = view.column(view.num_columns() - 1);
  auto order           = cudf::sorted_order(cudf::table_view{{dist_col}},
                                            {cudf::order::ASCENDING},
                                            {cudf::null_order::AFTER},
                                  c.stream,
                                  c.mr);
  auto sorted =
    cudf::gather(view, order->view(), cudf::out_of_bounds_policy::DONT_CHECK, c.stream, c.mr);
  auto const n   = static_cast<cudf::size_type>(std::min<std::int64_t>(k, sorted->num_rows()));
  auto const top = cudf::slice(sorted->view(), {0, n}, c.stream).front();
  return std::make_unique<cudf::table>(top, c.stream, c.mr);
}

}  // namespace

std::unique_ptr<cucascade::host_data_representation> run_vector_join(duckdb::SiriusContext& ctx,
                                                                     const vector_join_request& req)
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
  join_context c{*space, *host_spaces.front(), mr, stream};

  // The join reads from pinned tables: the corpus must be GPU-resident; the probe is streamed.
  // GPU-resident: iterated in place
  // HOST-pinned: uploaded chunk by chunk
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
  if (probe_pin->num_rows == 0 || corpus_pin->num_rows == 0) {
    return join_result_d2h(c, make_empty_join_output(*probe_pin, *corpus_pin, req, c));
  }

  // --- Corpus residency (size-driven; the 3-case cascade) ---
  std::unique_ptr<cudf::table> corpus;
  if (corpus_pin->tier == ccm::Tier::GPU) {
    SIRIUS_LOG_DEBUG("sirius_knn_join: corpus '{}' is GPU-resident ({} rows)",
                     req.corpus_table,
                     corpus_pin->num_rows);
    corpus = make_corpus_contiguous_on_device(*corpus_pin, req, *space, c);
  } else if (corpus_pin->tier == ccm::Tier::HOST) {
    auto const col_idx =
      get_pinned_column_indices(*corpus_pin, req.corpus_vector_column, req.corpus_output_columns);
    auto const bytes = host_columns_bytes(*corpus_pin, col_idx);
    auto const avail = space->get_available_memory();
    SIRIUS_LOG_DEBUG("sirius_knn_join: corpus '{}' on HOST; need {} Bytes, GPU available {} Bytes",
                     req.corpus_table,
                     bytes,
                     avail);
    // The upload allocates on demand through the space's reservation-aware allocator.
    if (bytes > avail) {
      throw duckdb::InvalidInputException(
        "sirius_knn_join: corpus '" + req.corpus_table + "' (" + std::to_string(bytes) +
        " Bytes) does not fit in available GPU memory (" + std::to_string(avail) + " Bytes).");
    }
    SIRIUS_LOG_DEBUG("sirius_knn_join: uploading corpus '{}' to GPU", req.corpus_table);
    corpus = assemble_corpus_h2d(*corpus_pin, col_idx, *space, c);
  } else {
    throw duckdb::InvalidInputException("sirius_knn_join: corpus '" + req.corpus_table +
                                        "' is on an unsupported memory tier");
  }

  // Drop null-vector corpus rows (vector column is always at index 0).
  if (corpus->view().column(0).null_count() != 0) {
    auto valid_mask = cudf::is_valid(corpus->view().column(0), stream, mr);
    corpus          = cudf::apply_boolean_mask(corpus->view(), valid_mask->view(), stream, mr);
  }
  auto const corpus_view = corpus->view();
  if (corpus_view.num_rows() == 0) {
    return join_result_d2h(c, make_empty_join_output(*probe_pin, *corpus_pin, req, c));
  }
  auto const k_eff = std::min<int64_t>(corpus_view.num_rows(), req.k);

  auto const corpus_dataset = list_column_as_dataset_view(corpus_view.column(0), req.dim);
  std::vector<cudf::column_view> corpus_passthroughs;
  corpus_passthroughs.reserve(corpus_view.num_columns() - 1);
  for (int i = 1; i < corpus_view.num_columns(); ++i) {
    corpus_passthroughs.push_back(corpus_view.column(i));
  }
  cudf::table_view const corpus_pass{corpus_passthroughs};

  // One RAFT handle reused across probe chunks (workspace setup paid once). Each
  // per-chunk search runs async on c.stream; res must outlive the join sync.
  raft::device_resources res{stream};

  // Search one probe chunk against the corpus and append their columns to results. cuVS
  // returns query-major [q * k_eff] neighbors, so repeat(probe, k_eff) and
  // gather(corpus, neighbors) line up row-for-row with distances.
  std::vector<std::unique_ptr<cudf::table>> results;
  auto process_probe_chunk = [&](cudf::table_view probe_chunk) {
    std::unique_ptr<cudf::table> compacted;
    if (auto const& vec = probe_chunk.column(0); vec.offset() != 0 || vec.null_count() != 0) {
      auto valid_mask = cudf::is_valid(vec, stream, mr);
      compacted       = cudf::apply_boolean_mask(probe_chunk, valid_mask->view(), stream, mr);
      probe_chunk     = compacted->view();
    }
    if (probe_chunk.num_rows() == 0) { return; }

    auto const queries = list_column_as_dataset_view(probe_chunk.column(0), req.dim);
    auto knn           = brute_force_knn(
      res, corpus_dataset, queries, k_eff, enn_distance_type_from_metric(req.metric), mr);

    auto corpus_matched = cudf::gather(
      corpus_pass, knn.neighbors->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);

    std::vector<cudf::column_view> probe_passthroughs;
    probe_passthroughs.reserve(probe_chunk.num_columns() - 1);
    for (int i = 1; i < probe_chunk.num_columns(); ++i) {
      probe_passthroughs.push_back(probe_chunk.column(i));
    }
    auto probe_repeated = cudf::repeat(
      cudf::table_view{probe_passthroughs}, static_cast<cudf::size_type>(k_eff), stream, mr);

    auto probe_cols  = probe_repeated->release();
    auto corpus_cols = corpus_matched->release();
    std::vector<std::unique_ptr<cudf::column>> out_cols;
    out_cols.reserve(probe_cols.size() + corpus_cols.size() + 1);
    for (auto& col : probe_cols) {
      out_cols.push_back(std::move(col));
    }
    for (auto& col : corpus_cols) {
      out_cols.push_back(std::move(col));
    }
    out_cols.push_back(std::move(knn.distances));
    auto chunk_out = std::make_unique<cudf::table>(std::move(out_cols));
    if (req.threshold) { chunk_out = filter_by_threshold(std::move(chunk_out), *req.threshold, c); }
    if (chunk_out->num_rows() == 0) { return; }
    results.push_back(std::move(chunk_out));
  };

  // --- Probe: iterate resident chunks, or stream host chunks up per batch ---
  if (probe_pin->tier == ccm::Tier::GPU) {
    SIRIUS_LOG_DEBUG(
      "sirius_knn_join: probe '{}' GPU-resident ({} rows)", req.probe_table, probe_pin->num_rows);
    auto const probe_vec_chunks =
      pinned_column_chunk_views(*probe_pin, req.probe_vector_column, *space);
    auto const n_probe_chunks = probe_vec_chunks.size();
    std::vector<std::vector<cudf::column_view>> probe_out_chunks;
    probe_out_chunks.reserve(req.probe_output_columns.size());
    for (auto const& name : req.probe_output_columns) {
      auto views = pinned_column_chunk_views(*probe_pin, name, *space);
      if (views.size() != n_probe_chunks) {
        throw duckdb::InvalidInputException(
          "sirius_knn_join: probe columns have inconsistent chunk counts");
      }
      probe_out_chunks.push_back(std::move(views));
    }
    for (std::size_t ci = 0; ci < n_probe_chunks; ++ci) {
      std::vector<cudf::column_view> pcols;
      pcols.reserve(probe_out_chunks.size() + 1);
      pcols.push_back(probe_vec_chunks[ci]);
      for (auto const& oc : probe_out_chunks) {
        pcols.push_back(oc[ci]);
      }
      process_probe_chunk(cudf::table_view{pcols});
    }
  } else if (probe_pin->tier == ccm::Tier::HOST) {
    auto const col_idx =
      get_pinned_column_indices(*probe_pin, req.probe_vector_column, req.probe_output_columns);
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

  if (results.empty()) {
    return join_result_d2h(c, make_empty_join_output(*probe_pin, *corpus_pin, req, c));
  }

  std::unique_ptr<cudf::table> joined;
  if (results.size() == 1) {
    joined = std::move(results.front());
  } else {
    std::vector<cudf::table_view> views;
    views.reserve(results.size());
    for (auto const& t : results) {
      views.push_back(t->view());
    }
    joined = cudf::concatenate(views, stream, mr);
  }

  if (req.global) { joined = take_global_top_k(std::move(joined), req.k, c); }
  return join_result_d2h(c, std::move(joined));
}

}  // namespace sirius::vss
