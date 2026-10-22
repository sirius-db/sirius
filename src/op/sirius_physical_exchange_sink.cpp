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

#include "op/sirius_physical_exchange_sink.hpp"

#include "data/data_batch_utils.hpp"
#include "exchange/exchange_service.hpp"
#include "log/logging.hpp"
#include "op/partition/gpu_partition_impl.hpp"
#include "sirius/exception.hpp"

#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/search.hpp>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <utility>

namespace sirius::op {

sirius_physical_exchange_sink::sirius_physical_exchange_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<exchange::exchange_service> service,
  exchange::stream_id stream,
  exchange_spec spec)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::EXCHANGE_SINK, std::move(types), estimated_cardinality),
    _service(std::move(service)),
    _stream(stream),
    _spec(std::move(spec))
{
  using mode = exchange_spec::mode;
  if (!_service) { throw sirius::invalid_input_exception("exchange sink: no exchange service"); }
  bool const keyed = _spec.routing == mode::hash || _spec.routing == mode::range;
  if (keyed == _spec.key_columns.empty()) {
    throw sirius::invalid_input_exception(
      "exchange sink: hash and range routing need key columns; broadcast and gather take none");
  }
  for (auto const key : _spec.key_columns) {
    if (key < 0 || static_cast<std::size_t>(key) >= this->types.size()) {
      throw sirius::invalid_input_exception(
        "exchange sink: key column {} of a {}-column input", key, this->types.size());
    }
  }
  if (!_spec.key_cast_types.empty() && _spec.key_cast_types.size() != _spec.key_columns.size()) {
    throw sirius::invalid_input_exception("exchange sink: {} cast types for {} key columns",
                                          _spec.key_cast_types.size(),
                                          _spec.key_columns.size());
  }
  if (_spec.routing == mode::range && (_spec.column_order.size() != _spec.key_columns.size() ||
                                       _spec.null_precedence.size() != _spec.key_columns.size())) {
    throw sirius::invalid_input_exception("exchange sink: range routing needs an order per key");
  }
  if (_spec.routing == mode::gather && _spec.target >= _service->world_size()) {
    throw sirius::invalid_input_exception(
      "exchange sink: gather to rank {} of {}", _spec.target, _service->world_size());
  }
}

void sirius_physical_exchange_sink::set_range_boundaries(
  std::shared_ptr<cudf::table const> boundaries)
{
  auto const ranks = static_cast<cudf::size_type>(_service->world_size());
  if (!boundaries || boundaries->num_rows() != ranks - 1 ||
      boundaries->num_columns() != static_cast<cudf::size_type>(_spec.key_columns.size())) {
    throw sirius::invalid_input_exception(
      "exchange sink: range boundaries must be {} rows of the {} key columns",
      ranks - 1,
      _spec.key_columns.size());
  }
  std::lock_guard lock{_boundaries_mutex};
  _boundaries = std::move(boundaries);
}

std::unique_ptr<operator_data> sirius_physical_exchange_sink::execute(
  const operator_data& input_data, ::cuda::stream_ref /*stream*/)
{
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_data_batches());
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_exchange_sink::route(
  cucascade::read_only_data_batch const& batch,
  std::shared_ptr<cucascade::data_batch> const& handle,
  ::cuda::stream_ref stream)
{
  using mode       = exchange_spec::mode;
  auto const ranks = _service->world_size();
  std::vector<std::shared_ptr<cucascade::data_batch>> slices(ranks);
  if (_spec.routing == mode::gather) {
    slices[_spec.target] = handle;
    return slices;
  }
  if (_spec.routing == mode::broadcast || ranks == 1) {
    // Each rank's transport copies what it sends; this rank keeps the batch itself.
    std::fill(slices.begin(), slices.end(), handle);
    return slices;
  }
  auto* space = batch.get_memory_space();
  if (space == nullptr) {
    throw sirius::internal_exception("exchange sink: the input batch is not resident");
  }
  if (_spec.routing == mode::hash) {
    auto parts = gpu_partition_impl::hash_partition(batch,
                                                    _spec.key_columns,
                                                    _spec.key_cast_types,
                                                    static_cast<int>(ranks),
                                                    stream,
                                                    *space,
                                                    batch_telemetry(),
                                                    _spec.hash_seed);
    for (std::size_t r = 0; r < parts.size(); ++r) {
      slices[r] = std::move(parts[r]);
    }
    return slices;
  }

  // Range: rank of a row = how many boundaries are at or below it, which puts a row equal to
  // boundary i on rank i + 1, as SORT_PARTITION's lower_bound split does.
  std::shared_ptr<cudf::table const> boundaries;
  {
    std::lock_guard lock{_boundaries_mutex};
    boundaries = _boundaries;
  }
  if (!boundaries) {
    throw sirius::internal_exception("exchange sink: range routing before its boundaries are set");
  }
  auto const table    = get_cudf_table_view(batch);
  auto const keys     = table.select(_spec.key_columns);
  auto const mr       = space->get_default_allocator();
  auto const ranks_of = cudf::upper_bound(
    boundaries->view(), keys, _spec.column_order, _spec.null_precedence, stream, mr);
  auto [grouped, offsets] =
    cudf::partition(table, ranks_of->view(), static_cast<cudf::size_type>(ranks), stream, mr);
  // offsets has ranks + 1 entries: rank r's rows are [offsets[r], offsets[r + 1]).
  std::shared_ptr<cudf::table> owner = std::move(grouped);
  std::vector<cudf::size_type> bounds;
  for (std::size_t r = 0; r < ranks; ++r) {
    bounds.push_back(offsets[r]);
    bounds.push_back(offsets[r + 1]);
  }
  auto const pieces = cudf::slice(owner->view(), bounds, stream);
  auto const total  = std::max<std::size_t>(1, static_cast<std::size_t>(owner->num_rows()));
  for (std::size_t r = 0; r < ranks; ++r) {
    if (pieces[r].num_rows() == 0) { continue; }
    // Each slice is charged its share of the partitioned table.
    auto const bytes = owner->alloc_size() * static_cast<std::size_t>(pieces[r].num_rows()) / total;
    slices[r] =
      make_data_batch_from_view(pieces[r], owner, bytes, *space, stream, batch_telemetry());
  }
  return slices;
}

void sirius_physical_exchange_sink::sink(const operator_data& input_data, ::cuda::stream_ref stream)
{
  auto const& input   = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto const& handles = input.get_data_batches();
  auto const batches  = input.get_read_only_batches();
  for (std::size_t b = 0; b < handles.size(); ++b) {
    auto slices = route(batches[b], handles[b], stream);
    for (exchange::rank_t r = 0; r < slices.size(); ++r) {
      if (!slices[r] || get_cudf_table_view(*slices[r]).num_rows() == 0) { continue; }
      _service->send(_stream, r, std::move(slices[r]));
    }
  }
}

std::size_t sirius_physical_exchange_sink::no_history_peak_memory_estimate(
  const input_stats& stats) const
{
  // Hash and range hold a partitioned copy beside their input; broadcast and gather send the
  // input itself.
  using mode        = exchange_spec::mode;
  bool const splits = _spec.routing == mode::hash || _spec.routing == mode::range;
  return splits ? stats.bytes * 2 : 0;
}

void sirius_physical_exchange_sink::on_finalize_operator()
{
  // A failed exchange throws here; its receivers already learn of it through on_error.
  try {
    if (_spec.routing == exchange_spec::mode::gather) {
      _service->close(_stream, _spec.target);
      return;
    }
    for (exchange::rank_t r = 0; r < _service->world_size(); ++r) {
      _service->close(_stream, r);
    }
  } catch (std::exception const& e) {
    SIRIUS_LOG_WARN("exchange sink: stream {} not closed: {}", _stream, e.what());
  }
}

}  // namespace sirius::op
