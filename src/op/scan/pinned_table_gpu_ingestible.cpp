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

// sirius
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <expression/ast/from_duckdb.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <log/logging.hpp>
#include <op/scan/pinned_table_gpu_ingestible.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>

// cudf
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime.h>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/gpu_data_representation.hpp>
#include <cucascade/data/representation_converter.hpp>

// standard library
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace sirius::op::scan {

pinned_table_gpu_ingestible::pinned_table_gpu_ingestible(
  std::unique_ptr<io::ingestible_table_info> table_info,
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
  std::vector<::cucascade::memory::memory_space*> chunk_memory_spaces,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<scan_plan const> plan)
  : io::gpu_ingestible(std::move(table_info)),
    _chunk_memory_spaces(std::move(chunk_memory_spaces)),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
  std::size_t const num_batches =
    columns_per_request.empty() ? 0 : columns_per_request.front().size();
  for (auto const& col_chunks : columns_per_request) {
    if (col_chunks.size() != num_batches) {
      throw std::runtime_error(
        "[pinned_table_gpu_ingestible] mismatched chunk count across requested columns");
    }
  }

  _batches.reserve(num_batches);
  for (std::size_t b = 0; b < num_batches; ++b) {
    gpu_chunk chunk_cols;
    chunk_cols.reserve(columns_per_request.size());
    for (auto& col_chunks : columns_per_request) {
      chunk_cols.push_back(std::move(col_chunks[b]));
    }
    _batches.emplace_back(std::move(chunk_cols));
  }
}

pinned_table_gpu_ingestible::pinned_table_gpu_ingestible(
  std::unique_ptr<io::ingestible_table_info> table_info,
  std::vector<std::shared_ptr<::cucascade::host_data_representation>> host_chunks,
  std::vector<std::size_t> column_indices,
  ::cucascade::memory::memory_space& memory_space,
  std::unordered_map<int, ::cucascade::memory::memory_space*> gpu_memory_spaces,
  std::shared_ptr<duckdb::Expression> filter_expression,
  std::shared_ptr<scan_plan const> plan)
  : io::gpu_ingestible(std::move(table_info)),
    _column_indices(std::move(column_indices)),
    _memory_space(&memory_space),
    _gpu_memory_spaces(std::move(gpu_memory_spaces)),
    _filter_expression(std::move(filter_expression)),
    _plan(std::move(plan))
{
  if (_gpu_memory_spaces.empty()) {
    throw std::runtime_error(
      "[pinned_table_gpu_ingestible] HOST-tier constructor requires non-empty "
      "gpu_memory_spaces so produce_batch can convert host chunks to the executing "
      "GPU's space");
  }
  _batches.reserve(host_chunks.size());
  for (auto& chunk : host_chunks) {
    if (!chunk) {
      throw std::runtime_error(
        "[pinned_table_gpu_ingestible] null host_data_representation in pinned chunks");
    }
    _batches.emplace_back(std::move(chunk));
  }
}

pinned_table_gpu_ingestible::~pinned_table_gpu_ingestible() = default;

bool pinned_table_post_filter_and_projection_info::has_work() const
{
  // Gates only the cached batch's static filter and output assembly; membership filtering happens
  // in the downstream dynamic-filter operator.
  return apply_filter || apply_assembly;
}

bool pinned_table_gpu_ingestible::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
pinned_table_gpu_ingestible::next_split_provider()
{
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _batches.size()) { return nullptr; }

  bool const apply_filter   = static_cast<bool>(_filter_expression);
  bool const apply_assembly = needs_output_assembly(*_plan);
  // A split is post-processed only for a static filter or output assembly; with neither, the
  // cached batch forwards unchanged. Membership filtering happens in the downstream operator.
  bool const need_post_processing = apply_filter || apply_assembly;

  return [this, batch_idx, apply_filter, apply_assembly, need_post_processing]()
           -> std::vector<std::unique_ptr<op::operator_data>> {
    auto batch = produce_batch(batch_idx);

    std::unique_ptr<io::post_filter_and_projection_info> filter_info;
    if (need_post_processing) {
      auto pf            = std::make_unique<pinned_table_post_filter_and_projection_info>();
      pf->apply_filter   = apply_filter;
      pf->apply_assembly = apply_assembly;
      filter_info        = std::move(pf);
    }

    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::make_unique<scan_operator_with_pinned_table_input>(std::move(batch),
                                                                          std::move(filter_info)));
    return out;
  };
}

std::shared_ptr<::cucascade::data_batch> pinned_table_gpu_ingestible::produce_batch(
  std::size_t batch_idx)
{
  return std::visit(
    [this, batch_idx](auto& chunk) -> std::shared_ptr<::cucascade::data_batch> {
      using T = std::decay_t<decltype(chunk)>;
      if constexpr (std::is_same_v<T, gpu_chunk>) {
        std::vector<cudf::column_view> col_views;
        col_views.reserve(chunk.size());
        std::size_t alloc_size = 0;
        for (auto const& col_ptr : chunk) {
          col_views.emplace_back(col_ptr->view());
          alloc_size += col_ptr->alloc_size();
        }
        cudf::table_view view(col_views);
        auto* chunk_space =
          !_chunk_memory_spaces.empty() ? _chunk_memory_spaces.at(batch_idx) : _memory_space;
        auto gpu_repr = std::make_unique<::cucascade::gpu_table_representation>(
          view, std::move(chunk), alloc_size, *chunk_space, rmm::cuda_stream_view{});
        return std::make_shared<::cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                         std::move(gpu_repr));
      } else if constexpr (std::is_same_v<T, host_chunk>) {
        // Emit a host-resident batch; the host->GPU upload is deferred to
        // scan_operator_with_pinned_table_input::prepare_for_processing,
        // which has the scheduler-provided target memory_space.
        auto sliced = chunk->slice(std::span<const std::size_t>(_column_indices));
        return std::make_shared<::cucascade::data_batch>(::sirius::get_next_batch_id(),
                                                         std::move(sliced));
      }
    },
    _batches[batch_idx]);
}

io::filtered_table pinned_table_gpu_ingestible::materialize_table(
  io::scan_info const& /*info*/,
  ::cucascade::memory::memory_space const& /*mem_space*/,
  rmm::cuda_stream_view /*stream*/)
{
  // The cached path never goes through materialize_table — its emitted input
  // type is scan_operator_with_pinned_table_input, which dispatches directly
  // into post_filter_and_project. Throwing here makes any mis-routing loud.
  throw std::runtime_error(
    "[pinned_table_gpu_ingestible::materialize_table] not reachable; cached path "
    "uses scan_operator_with_pinned_table_input + post_filter_and_project only.");
}

std::unique_ptr<cudf::table> pinned_table_gpu_ingestible::post_filter_and_project(
  std::unique_ptr<cudf::table> input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  // The scan operator enters the cached path through the view overload below; this owning
  // entry point remains for interface completeness. `input` backs the view through the call.
  return post_filter_and_project(input->view(), info, mem_space, stream);
}

namespace {

/// Zero-copy output-layout permutation of @p input: a table_view selecting the plan's DATA
/// columns in output order. nullopt when the layout has any non-DATA entry (hive partitions)
/// or an out-of-range index — callers fall back to the owning @ref assemble_scan_output,
/// which reports those exactly as the fresh-read path would.
std::optional<cudf::table_view> assembled_data_view(scan_plan const& plan,
                                                    cudf::table_view const& input)
{
  std::vector<cudf::column_view> cols;
  cols.reserve(plan.output_layout.size());
  for (auto const& entry : plan.output_layout) {
    if (entry.source != scan_plan::output_entry::DATA ||
        entry.idx >= static_cast<std::size_t>(input.num_columns())) {
      return std::nullopt;
    }
    cols.push_back(input.column(static_cast<cudf::size_type>(entry.idx)));
  }
  return cudf::table_view(cols);
}

}  // namespace

std::unique_ptr<cudf::table> pinned_table_gpu_ingestible::post_filter_and_project(
  cudf::table_view const& view_input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& /*mem_space*/,
  rmm::cuda_stream_view stream)
{
  auto const& pf = static_cast<pinned_table_post_filter_and_projection_info const&>(info);
  auto const mr  = cudf::get_current_device_resource_ref();

  // Every stage reads a view and produces a fresh table, so the cached batch is never copied
  // wholesale: `owned` is the most recent stage's product, `current` the view to read next.
  // Whatever is returned was materialized by a stage (or the final fallback) and never aliases
  // the cache.
  std::unique_ptr<cudf::table> owned;
  cudf::table_view current = view_input;

  if (pf.apply_filter && _filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_filter_expression);
    sirius::gpu_expression_executor exec(sirius_filter_ast.get(), mr, stream);
    owned   = exec.select(current);
    current = owned->view();
    SIRIUS_LOG_DEBUG(
      "[pinned_table_gpu_ingestible::post_filter_and_project] Applied duckdb filter "
      "expression on cached batch.");
  }

  if (pf.apply_assembly) {
    if (owned) {
      // The filter stage's product is ours to reshape — moving columns costs nothing.
      owned   = assemble_scan_output(*_plan, std::move(owned), /*partition_values=*/{}, stream);
      current = owned->view();
    } else if (auto assembled = assembled_data_view(*_plan, current)) {
      // Pure permutation of the cache view; a later stage (or the final fallback)
      // materializes only the output columns.
      current = *assembled;
    } else {
      // Non-DATA layout entries — gated upstream off the cached path; materialize and
      // delegate so the failure mode matches assemble_scan_output's.
      owned   = assemble_scan_output(*_plan,
                                   std::make_unique<cudf::table>(current, stream, mr),
                                   /*partition_values=*/{},
                                   stream);
      current = owned->view();
    }
    SIRIUS_LOG_DEBUG(
      "[pinned_table_gpu_ingestible::post_filter_and_project] Assembled cached batch "
      "to plan layout.");
  }

  // No stage materialized (assembly was a pure permutation, or nothing applied): realize
  // `current` so the emitted batch owns its memory.
  return owned ? std::move(owned) : std::make_unique<cudf::table>(current, stream, mr);
}

}  // namespace sirius::op::scan
