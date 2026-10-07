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

#include "vss/sirius_physical_vector_join_materialize.hpp"

#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "pipeline/batch_lock_utils.hpp"
#include "scan_manager/sirius_scan_manager.hpp"
#include "sirius_context.hpp"
#include "vss/pinned_column.hpp"
#include "vss/staging_shortfall.hpp"
#include "vss/vector_search_internal.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/error.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op {

namespace {
/// A side batch or answer piece the downgrade executor took to disk has to be moved before it can
/// be read: release the caller's read lock, bring the batch back to the host tier through the
/// engine (in place) and return a fresh read lock on it. Batches on GPU or HOST pass through.
cucascade::read_only_data_batch host_or_device_resident(
  const std::shared_ptr<cucascade::data_batch>& batch,
  cucascade::read_only_data_batch ro,
  duckdb::SiriusContext* ctx,
  ::cuda::stream_ref stream,
  char const* what)
{
  if (ro.get_current_tier() != cucascade::memory::Tier::DISK) { return ro; }
  const cucascade::memory::memory_space* host_space = nullptr;
  if (ctx != nullptr) {
    auto spaces =
      ctx->get_memory_manager().get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (!spaces.empty()) { host_space = spaces.front(); }
  }
  if (host_space == nullptr) {
    throw std::runtime_error(std::string("[sirius_physical_vector_join_materialize] ") + what +
                             " is on disk and no HOST memory space is available to restore it");
  }
  std::ignore   = cucascade::data_batch::to_idle(std::move(ro));
  auto prepared = sirius::pipeline::lock_and_prepare_batch(batch, host_space, stream);
  if (!prepared) {
    throw std::runtime_error(
      std::string("[sirius_physical_vector_join_materialize] could not restore ") + what +
      " from disk");
  }
  return std::visit([](auto& r) { return std::move(r.ro_lock); }, *prepared);
}
}  // namespace

sirius_physical_vector_join_materialize::sirius_physical_vector_join_materialize(
  duckdb::vector<sirius::logical_type> types,
  duckdb::idx_t estimated_cardinality,
  sirius::vss::vector_join_request request,
  sirius::scan_manager::sirius_scan_manager* scan_manager,
  std::shared_ptr<sirius::vss::materialized_side_buffer> build_side,
  std::shared_ptr<sirius::vss::materialized_side_buffer> probe_side,
  duckdb::SiriusContext* sirius_ctx)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::VECTOR_JOIN_MATERIALIZE, std::move(types), estimated_cardinality),
    _request(std::move(request)),
    _scan_manager(scan_manager),
    _build_side(std::move(build_side)),
    _probe_side(std::move(probe_side)),
    _sirius_ctx(sirius_ctx)
{
}

std::vector<std::unique_ptr<cudf::column>>
sirius_physical_vector_join_materialize::build_side_output_columns(
  std::size_t num_output_columns,
  rmm::cuda_stream_view stream,
  ::cucascade::memory::memory_space& space)
{
  // The fold numbered neighbour ids by walking this snapshot, so concatenating in the same
  // order is what makes id i address row i. Re-deriving the order here -- from the pin, or by
  // asking the repository again -- is the bug this shares a handle to avoid: the batches
  // arrive in scan-completion order, which is not the table's row order.
  auto const ids = _build_side->batch_ids();
  auto* repo     = _build_side->repo();
  if (repo == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] build side has no snapshot; the join stage "
      "should have taken it before any row reached this operator");
  }

  // Output columns sit after the vector column, which the corpus scan projects first.
  std::vector<std::size_t> out_cols(num_output_columns);
  for (std::size_t c = 0; c < num_output_columns; ++c) {
    out_cols[c] = c + 1;
  }

  // Copied out a batch at a time, and each batch's lock released as soon as its copy has
  // landed. Holding every batch until one final concatenate kept the whole corpus -- vectors
  // and all -- unspillable exactly when this needs room, so a corpus larger than the pool
  // failed here after the join itself had streamed it.
  auto const mr = space.get_default_allocator();
  std::vector<std::vector<std::unique_ptr<cudf::column>>> parts(num_output_columns);
  for (auto& p : parts) {
    p.reserve(ids.size());
  }
  for (auto const id : ids) {
    auto batch = repo->get_data_batch_by_id(id, /*partition_idx=*/0);
    if (!batch) {
      throw std::runtime_error("[sirius_physical_vector_join_materialize] build-side batch " +
                               std::to_string(id) + " is no longer in the repository");
    }
    auto ro = host_or_device_resident(batch,
                                      batch->to_read_only(),
                                      _sirius_ctx,
                                      ::cuda::stream_ref{stream.value()},
                                      "a build-side batch");
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      auto const table = sirius::get_cudf_table_view(ro);
      for (std::size_t c = 0; c < num_output_columns; ++c) {
        parts[c].push_back(std::make_unique<cudf::column>(
          table.column(static_cast<cudf::size_type>(c + 1)), stream, mr));
      }
      stream.synchronize();
      continue;
    }

    // Spilled: bring back the output columns alone. The vector column is the bulk of the
    // batch and nothing here reads it.
    const auto* data = ro.get_data();
    if (data == nullptr) {
      throw std::runtime_error(
        "[sirius_physical_vector_join_materialize] build-side batch has no data representation");
    }
    auto data_rep    = data->cast<cucascade::host_data_representation>().slice(out_cols);
    auto const bytes = data_rep->get_size_in_bytes();
    std::shared_ptr<cucascade::memory::reservation> reservation{
      space.make_reservation_or_null(bytes)};
    if (!reservation) {
      // An OOM rather than a plain error: the task is retried after a downgrade, and batches
      // this loop has already released are what that downgrade can spill.
      throw rmm::out_of_memory(
        "[sirius_physical_vector_join_materialize] build-side output columns need " +
        std::to_string(bytes) + " bytes device-side");
    }
    auto const batch_id = sirius::get_next_batch_id();
    auto staged         = cucascade::data_batch::make(
      batch_id,
      std::move(data_rep),
      telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
    {
      auto mut = staged->to_mutable();
      mut.convert_to<cucascade::gpu_table_representation>(
        sirius::converter_registry::get(), *reservation, stream);
    }
    auto const table = sirius::get_cudf_table_view(*staged);
    for (std::size_t c = 0; c < num_output_columns; ++c) {
      parts[c].push_back(
        std::make_unique<cudf::column>(table.column(static_cast<cudf::size_type>(c)), stream, mr));
    }
    stream.synchronize();
  }

  std::vector<std::unique_ptr<cudf::column>> cols;
  cols.reserve(num_output_columns);
  for (auto& p : parts) {
    if (p.size() == 1) {
      cols.push_back(std::move(p.front()));
      continue;
    }
    std::vector<cudf::column_view> views;
    views.reserve(p.size());
    for (auto const& part : p) {
      views.push_back(part->view());
    }
    cols.push_back(cudf::concatenate(views, stream, mr));
    p.clear();
  }
  return cols;
}

std::vector<std::vector<cudf::column_view>>
sirius_physical_vector_join_materialize::probe_side_output_views(
  std::size_t num_output_columns,
  rmm::cuda_stream_view stream,
  ::cucascade::memory::memory_space& space)
{
  auto const ids = _probe_side->batch_ids();
  auto* repo     = _probe_side->repo();
  if (repo == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] probe side has no snapshot; the join stage "
      "should have taken it before any row reached this operator");
  }

  // Output columns sit after the vector column, which the probe scan projects first.
  std::vector<std::size_t> out_cols(num_output_columns);
  for (std::size_t c = 0; c < num_output_columns; ++c) {
    out_cols[c] = c + 1;
  }

  std::vector<std::vector<cudf::column_view>> per_column(num_output_columns);
  for (auto const id : ids) {
    auto batch = repo->get_data_batch_by_id(id, /*partition_idx=*/0);
    if (!batch) {
      throw std::runtime_error("[sirius_physical_vector_join_materialize] probe-side batch " +
                               std::to_string(id) + " is no longer in the repository");
    }
    auto ro = host_or_device_resident(batch,
                                      batch->to_read_only(),
                                      _sirius_ctx,
                                      ::cuda::stream_ref{stream.value()},
                                      "a probe-side batch");
    cudf::table_view table;
    cudf::size_type first = 0;
    if (ro.get_current_tier() == cucascade::memory::Tier::GPU) {
      table = sirius::get_cudf_table_view(ro);
      first = 1;
    } else {
      const auto* data = ro.get_data();
      if (data == nullptr) {
        throw std::runtime_error(
          "[sirius_physical_vector_join_materialize] probe-side batch has no data representation");
      }
      auto data_rep    = data->cast<cucascade::host_data_representation>().slice(out_cols);
      auto const bytes = data_rep->get_size_in_bytes();
      std::shared_ptr<cucascade::memory::reservation> reservation{
        space.make_reservation_or_null(bytes)};
      if (!reservation) {
        vss::throw_staging_shortfall(
          space, bytes, "[sirius_physical_vector_join_materialize] probe-side output columns");
      }
      auto const batch_id = sirius::get_next_batch_id();
      auto staged         = cucascade::data_batch::make(
        batch_id,
        std::move(data_rep),
        telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id));
      {
        auto mut = staged->to_mutable();
        mut.convert_to<cucascade::gpu_table_representation>(
          sirius::converter_registry::get(), *reservation, stream);
      }
      table = sirius::get_cudf_table_view(*staged);
      _probe_restaged.push_back(std::move(staged));
      _probe_reservations.push_back(std::move(reservation));
    }
    for (std::size_t c = 0; c < num_output_columns; ++c) {
      per_column[c].push_back(table.column(first + static_cast<cudf::size_type>(c)));
    }
    _probe_readers.push_back(std::move(ro));
  }
  return per_column;
}

void sirius_physical_vector_join_materialize::ensure_initialized(
  rmm::cuda_stream_view stream, ::cucascade::memory::memory_space& space)
{
  std::lock_guard<std::mutex> lg(_init_mutex);
  if (_initialized) { return; }
  if (_scan_manager == nullptr) {
    throw std::runtime_error("[sirius_physical_vector_join_materialize] no scan manager set");
  }

  auto const& left  = _request.left;
  auto const& right = _request.right;

  auto left_pin  = _probe_side ? nullptr
                               : _scan_manager->find_pinned_entry_for_duckdb_table(
                                  left.catalog, left.schema, left.table);
  auto right_pin = _build_side ? nullptr
                               : _scan_manager->find_pinned_entry_for_duckdb_table(
                                   right.catalog, right.schema, right.table);
  if ((!_probe_side && left_pin == nullptr) || (!_build_side && right_pin == nullptr)) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] left/right table is no longer pinned");
  }
  // Staged rather than aliased so a HOST-tier pin works: output columns are the small
  // non-vector columns, and the right side is concatenated on device below regardless, so
  // staging does not change the memory profile for a GPU-tier pin (where it is zero-copy).
  if (_probe_side) {
    _left_output_cols = probe_side_output_views(left.output_columns.size(), stream, space);
  } else {
    _left_output_cols.resize(left.output_columns.size());
    for (std::size_t c = 0; c < left.output_columns.size(); ++c) {
      auto staged = vss::stage_pinned_column(
        *left_pin, left.output_columns[c], space, stream, batch_telemetry());
      _left_output_cols[c] = staged.views;
      _staged_left.push_back(std::move(staged));
    }
  }

  // Build path: the corpus output columns concatenated once, so a global right id gathers
  // straight into row i. Pinned corpus: only where its chunks end is recorded here, and each
  // partition stages just the chunks its neighbours fall in -- a 100M-row corpus would otherwise
  // copy its whole id column in for a 10-row answer.
  if (_build_side) {
    _right_output_concat = std::make_unique<cudf::table>(
      build_side_output_columns(right.output_columns.size(), stream, space));
  } else if (!right.output_columns.empty()) {
    _right_pin         = right_pin;
    auto const& column = right.output_columns.front();
    std::int64_t end   = 0;
    if (right_pin->tier == cucascade::memory::Tier::GPU) {
      for (auto const& v : vss::pinned_column_chunk_views(*right_pin, column, space)) {
        end += v.size();
        _right_chunk_ends.push_back(end);
      }
    } else {
      for (auto const& chunk : right_pin->host_chunks) {
        auto const* host = dynamic_cast<cucascade::host_data_representation const*>(chunk.get());
        if (host == nullptr) {
          throw std::runtime_error("[sirius_physical_vector_join_materialize] column '" + column +
                                   "' is pinned with host compression, which staging cannot "
                                   "slice");
        }
        end += host->get_host_table()->columns.front().num_rows;
        _right_chunk_ends.push_back(end);
      }
    }
  }

  _initialized = true;
}

std::unique_ptr<cudf::table> sirius_physical_vector_join_materialize::gather_right_on_host(
  cudf::column_view neighbors, rmm::cuda_stream_view stream, cucascade::memory::memory_space& space)
{
  // Random reads of pinned host memory, one per value: cheap for thousands, slower than copying
  // chunks in for millions.
  constexpr std::size_t kMaxValues = std::size_t{1} << 18;
  auto const n                     = static_cast<std::size_t>(neighbors.size());
  auto const& names                = _right_pin->cache_info.column_names();
  if (n * _request.right.output_columns.size() > kMaxValues) { return nullptr; }

  // Each output column's metadata in every chunk, if all are fixed-width without nulls.
  struct chunk_column {
    cucascade::memory::host_table_allocation const* table;
    cucascade::memory::column_metadata const* meta;
  };
  std::vector<std::vector<chunk_column>> columns;
  std::vector<cudf::data_type> types;
  for (auto const& name : _request.right.output_columns) {
    auto const it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) { return nullptr; }
    auto const col = static_cast<std::size_t>(std::distance(names.begin(), it));
    std::vector<chunk_column> per_chunk;
    std::optional<cudf::data_type> type;
    for (auto const& chunk : _right_pin->host_chunks) {
      auto const* host = dynamic_cast<cucascade::host_data_representation const*>(chunk.get());
      if (host == nullptr) { return nullptr; }
      auto const& table = host->get_host_table();
      if (!table || col >= table->columns.size()) { return nullptr; }
      auto const& meta = table->columns[col];
      cudf::data_type const t{static_cast<cudf::type_id>(meta.type_id)};
      if (!cudf::is_fixed_width(t) || cudf::is_fixed_point(t) || !meta.has_data ||
          !meta.children.empty() || meta.null_count != 0 || (type && *type != t)) {
        return nullptr;
      }
      type = t;
      per_chunk.push_back({table.get(), &meta});
    }
    if (!type) { return nullptr; }
    columns.push_back(std::move(per_chunk));
    types.push_back(*type);
  }

  std::vector<std::int64_t> rows(n);
  if (n > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(rows.data(),
                                  neighbors.data<std::int64_t>(),
                                  n * sizeof(std::int64_t),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
  }
  // Each row's chunk and its row within it.
  std::vector<std::pair<std::size_t, std::int64_t>> where(n);
  for (std::size_t i = 0; i < n; ++i) {
    auto const c = static_cast<std::size_t>(
      std::upper_bound(_right_chunk_ends.begin(), _right_chunk_ends.end(), rows[i]) -
      _right_chunk_ends.begin());
    if (rows[i] < 0 || c >= _right_chunk_ends.size()) { return nullptr; }
    where[i] = {c, rows[i] - (c == 0 ? 0 : _right_chunk_ends[c - 1])};
  }

  auto const mr = space.get_default_allocator();
  std::vector<std::unique_ptr<cudf::column>> out;
  std::vector<std::byte> values;
  for (std::size_t k = 0; k < columns.size(); ++k) {
    auto const width = static_cast<std::size_t>(cudf::size_of(types[k]));
    values.resize(n * width);
    for (std::size_t i = 0; i < n; ++i) {
      auto const& [c, local] = where[i];
      auto const& cc         = columns[k][c];
      auto const& blocks     = *cc.table->allocation;
      // Buffers start 8-byte aligned in blocks that are a multiple of 8, so a value never
      // straddles two blocks.
      auto const byte = cc.meta->data_offset + static_cast<std::size_t>(local) * width;
      std::memcpy(values.data() + i * width,
                  blocks.at(byte / blocks.block_size()).data() + byte % blocks.block_size(),
                  width);
    }
    auto column = cudf::make_fixed_width_column(
      types[k], static_cast<cudf::size_type>(n), cudf::mask_state::UNALLOCATED, stream, mr);
    if (n > 0) {
      CUDF_CUDA_TRY(cudaMemcpyAsync(column->mutable_view().head(),
                                    values.data(),
                                    values.size(),
                                    cudaMemcpyHostToDevice,
                                    stream.value()));
      // The next column reuses the host buffer.
      stream.synchronize();
    }
    out.push_back(std::move(column));
  }
  return std::make_unique<cudf::table>(std::move(out));
}

std::unique_ptr<cudf::table> sirius_physical_vector_join_materialize::gather_right_from_pin(
  cudf::column_view neighbors, rmm::cuda_stream_view stream, cucascade::memory::memory_space& space)
{
  if (_right_pin->tier != cucascade::memory::Tier::GPU) {
    if (auto on_host = gather_right_on_host(neighbors, stream, space)) { return on_host; }
  }
  auto const mr       = space.get_default_allocator();
  auto const n_chunks = static_cast<cudf::size_type>(_right_chunk_ends.size());

  // Each neighbour's chunk is the first whose end lies past it.
  auto ends = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n_chunks, cudf::mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(ends->mutable_view().data<std::int64_t>(),
                                _right_chunk_ends.data(),
                                _right_chunk_ends.size() * sizeof(std::int64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  auto const chunk_of   = cudf::upper_bound(cudf::table_view{{ends->view()}},
                                          cudf::table_view{{neighbors}},
                                            {cudf::order::ASCENDING},
                                            {cudf::null_order::BEFORE},
                                          stream,
                                          mr);
  auto const needed_col = cudf::distinct(cudf::table_view{{chunk_of->view()}},
                                         {0},
                                         cudf::duplicate_keep_option::KEEP_ANY,
                                         cudf::null_equality::EQUAL,
                                         cudf::nan_equality::ALL_EQUAL,
                                         stream,
                                         mr);
  auto const n_needed   = needed_col->num_rows();
  std::vector<cudf::size_type> needed(static_cast<std::size_t>(n_needed));
  if (n_needed > 0) {
    CUDF_CUDA_TRY(cudaMemcpyAsync(needed.data(),
                                  needed_col->view().column(0).data<cudf::size_type>(),
                                  needed.size() * sizeof(cudf::size_type),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    stream.synchronize();
  }
  std::sort(needed.begin(), needed.end());
  // No neighbours still needs typed columns to gather nothing from.
  if (needed.empty() && n_chunks > 0) { needed.push_back(0); }

  // A neighbour's row in the concatenation of the needed chunks is its pin row plus its chunk's
  // shift: the chunk's place in the concatenation less its place in the pin.
  std::vector<std::int64_t> shift(static_cast<std::size_t>(n_chunks), 0);
  std::int64_t placed = 0;
  for (auto const c : needed) {
    auto const begin = c == 0 ? 0 : _right_chunk_ends[static_cast<std::size_t>(c) - 1];
    shift[static_cast<std::size_t>(c)] = placed - begin;
    placed += _right_chunk_ends[static_cast<std::size_t>(c)] - begin;
  }
  auto shifts = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT64}, n_chunks, cudf::mask_state::UNALLOCATED, stream, mr);
  CUDF_CUDA_TRY(cudaMemcpyAsync(shifts->mutable_view().data<std::int64_t>(),
                                shift.data(),
                                shift.size() * sizeof(std::int64_t),
                                cudaMemcpyHostToDevice,
                                stream.value()));
  auto const shift_of = cudf::gather(cudf::table_view{{shifts->view()}},
                                     chunk_of->view(),
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     stream,
                                     mr);
  auto const rows     = cudf::binary_operation(neighbors,
                                           shift_of->view().column(0),
                                           cudf::binary_operator::ADD,
                                           cudf::data_type{cudf::type_id::INT64},
                                           stream,
                                           mr);

  std::vector<std::unique_ptr<cudf::column>> columns;
  for (auto const& name : _request.right.output_columns) {
    std::vector<vss::staged_pinned_chunk> staged;
    std::vector<cudf::column_view> views;
    for (auto const c : needed) {
      staged.push_back(vss::stage_pinned_column_chunk(
        *_right_pin, name, static_cast<std::size_t>(c), space, stream, batch_telemetry()));
      views.push_back(staged.back().view);
    }
    auto concat   = views.size() == 1 ? std::make_unique<cudf::column>(views.front(), stream, mr)
                                      : cudf::concatenate(views, stream, mr);
    auto gathered = cudf::gather(cudf::table_view{{concat->view()}},
                                 rows->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
    columns.push_back(std::move(gathered->release().front()));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::get_next_task_input_data()
{
  // One task per partition (= one left batch): drain its merge outputs.
  std::lock_guard<std::mutex> lg(_drain_mutex);

  auto* repo = ports.begin()->second->repo;
  if (_current_partition_index >= repo->num_partitions()) { return nullptr; }

  std::vector<std::shared_ptr<cucascade::data_batch>> all_batches;
  while (true) {
    auto batch = repo->pop_next_data_batch(_current_partition_index);
    if (!batch) { break; }
    all_batches.push_back(std::move(batch));
  }
  auto const partition_idx = _current_partition_index++;
  if (all_batches.empty()) { return nullptr; }
  return std::make_unique<partitioned_operator_data>(std::move(all_batches), partition_idx);
}

std::unique_ptr<operator_data> sirius_physical_vector_join_materialize::execute(
  const operator_data& input_data, ::cuda::stream_ref stream_ref)
{
  rmm::cuda_stream_view stream{stream_ref};
  nvtx3::scoped_range nvtx_range{"sirius_physical_vector_join_materialize::execute"};

  auto const& input        = dynamic_cast<const partitioned_operator_data&>(input_data);
  auto const partition_idx = input.get_partition_idx().value_or(0);  // = left batch index
  // The executor read-locked the pieces where they are. A piece the downgrade executor took to
  // disk has to be moved (exclusive lock) before it can be read, so this stage drops those locks
  // and takes its own per piece, restoring a disk-resident one to the host tier through the engine.
  auto const piece_ptrs = input.get_data_batches();
  const_cast<partitioned_operator_data&>(input).remove_read_only_lock();
  std::vector<cucascade::read_only_data_batch> input_batches;
  input_batches.reserve(piece_ptrs.size());
  for (auto const& ptr : piece_ptrs) {
    input_batches.push_back(host_or_device_resident(
      ptr, ptr->to_read_only(), _sirius_ctx, stream_ref, "a join output piece"));
  }

  if (input_batches.empty()) {
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<cucascade::data_batch>>{});
  }
  // The stream stage always ends a left batch with a device-resident piece.
  cucascade::memory::memory_space* space = nullptr;
  for (auto const& batch : input_batches) {
    if (batch.get_current_tier() == cucascade::memory::Tier::GPU) {
      space = batch.get_memory_space();
      break;
    }
  }
  if (space == nullptr) {
    throw std::runtime_error(
      "[sirius_physical_vector_join_materialize] no join output piece is on the device");
  }

  ensure_initialized(stream, *space);

  // A threshold whose answer outgrew the device arrives in several pieces, all but the last moved
  // to host memory by the stream stage. This stage runs in the same task, so nothing restages them
  // for it: each is brought back, materialized and its output moved out again, so only one piece
  // is on the device at a time.
  auto& registry = sirius::converter_registry::get();
  std::vector<std::shared_ptr<cucascade::data_batch>> batches;
  for (auto const& piece : input_batches) {
    cucascade::memory::memory_space* host = nullptr;
    std::shared_ptr<cucascade::data_batch> staged;
    if (piece.get_current_tier() != cucascade::memory::Tier::GPU) {
      host   = piece.get_memory_space();
      staged = piece.clone_to<cucascade::gpu_table_representation>(
        registry, sirius::get_next_batch_id(), space, stream);
      // Copied on a converter stream of its own; freed on this one instead.
      staged->to_mutable().rebind_stream(stream);
    }
    auto const tv =
      staged ? sirius::get_cudf_table_view(*staged) : sirius::get_cudf_table_view(piece);
    auto out_table = materialize_piece(partition_idx, tv, *space, stream);
    staged.reset();
    auto batch = sirius::make_data_batch(std::move(out_table), *space, stream, batch_telemetry());
    if (host != nullptr) {
      batch->to_mutable().convert_to<cucascade::host_data_representation>(registry, host, stream);
    }
    batches.push_back(std::move(batch));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<cudf::table> sirius_physical_vector_join_materialize::materialize_piece(
  std::size_t partition_idx,
  cudf::table_view pairs,
  cucascade::memory::memory_space& space,
  rmm::cuda_stream_view stream)
{
  auto const mr                         = space.get_default_allocator();
  cudf::column_view const left_row_view = pairs.column(0);  // INT32 row index into the left batch
  cudf::column_view const neighbor_view = pairs.column(1);  // INT64 global right id
  cudf::column_view const distance_view = pairs.column(2);  // FLOAT32 distance

  // Left columns gathered by the left row each pair belongs to. This used to repeat every
  // left row k times, which assumed a fixed k per row; threshold and global top-k are ragged
  // by construction, so the join stage now names the left row for each pair instead.
  // Either side can contribute no columns once projection pushdown has narrowed the output --
  // `SELECT count(*)`, or a query reading only the score -- and gathering a table of no columns
  // is not something cudf defines, so the gather is skipped rather than fed an empty table.
  std::vector<cudf::column_view> left_batch_cols;
  left_batch_cols.reserve(_left_output_cols.size());
  for (auto const& per_batch : _left_output_cols) {
    left_batch_cols.push_back(per_batch[partition_idx]);
  }
  std::unique_ptr<cudf::table> left_repeated;
  if (!left_batch_cols.empty()) {
    left_repeated = cudf::gather(cudf::table_view(left_batch_cols),
                                 left_row_view,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
  }

  // Right columns gathered by the global neighbor id.
  std::unique_ptr<cudf::table> right_gathered;
  if (_right_pin != nullptr) {
    right_gathered = gather_right_from_pin(neighbor_view, stream, space);
  } else if (_right_output_concat && _right_output_concat->num_columns() > 0) {
    right_gathered = cudf::gather(_right_output_concat->view(),
                                  neighbor_view,
                                  cudf::out_of_bounds_policy::DONT_CHECK,
                                  stream,
                                  mr);
  }

  // Score: distance, or cosine similarity = max(0, 1 - distance).
  std::unique_ptr<cudf::column> score;
  if (_request.metric == "cosine") {
    cudf::numeric_scalar<float> const lo(0.0F, true, stream);
    cudf::numeric_scalar<float> const hi(2.0F, true, stream);
    auto distance = cudf::clamp(distance_view, lo, hi, stream, mr);
    if (_request.output_type == sirius::vss::vector_join_output_type::similarity) {
      cudf::numeric_scalar<float> const one(1.0F, true, stream);
      score = cudf::binary_operation(one,
                                     distance->view(),
                                     cudf::binary_operator::SUB,
                                     cudf::data_type{cudf::type_id::FLOAT32},
                                     stream,
                                     mr);
    } else {
      score = std::move(distance);
    }
  } else {
    score = std::make_unique<cudf::column>(distance_view, stream, mr);
  }

  // Assemble [left cols..., right cols..., score] — the TVF schema.
  std::vector<std::unique_ptr<cudf::column>> out_cols;
  if (left_repeated) {
    for (auto& c : left_repeated->release()) {
      out_cols.push_back(std::move(c));
    }
  }
  if (right_gathered) {
    for (auto& c : right_gathered->release()) {
      out_cols.push_back(std::move(c));
    }
  }
  out_cols.push_back(std::move(score));
  // Output columns gathered from a pin keep the pin's storage type, which compressed
  // materialization may have narrowed; the operator declares the native types, so the carriers
  // are widened back before anything downstream reduces over them.
  sirius::vss::restore_native_carriers(
    out_cols, std::vector<sirius::logical_type>(types.begin(), types.end()), stream, mr);
  return std::make_unique<cudf::table>(std::move(out_cols));
}

std::size_t sirius_physical_vector_join_materialize::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  return std::max<std::size_t>(stats.bytes, std::size_t{1} << 20);
}

std::string sirius_physical_vector_join_materialize::params_to_string() const
{
  return _request.left.table + " x " + _request.right.table + " k=" + std::to_string(_request.k);
}

}  // namespace sirius::op
