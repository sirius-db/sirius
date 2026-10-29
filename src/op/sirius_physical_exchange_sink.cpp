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

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/search.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <optional>
#include <utility>

namespace sirius::op {
namespace {

struct null_key_split {
  std::shared_ptr<cucascade::data_batch> null_keys;  ///< Rows with a NULL in a key column.
  std::shared_ptr<cucascade::data_batch> keyed;      ///< The other rows.
};

/// Splits @p batch by whether a row has a NULL in any of @p key_columns; nullopt when none does.
std::optional<null_key_split> split_null_keys(cucascade::read_only_data_batch const& batch,
                                              std::vector<int> const& key_columns,
                                              ::cuda::stream_ref stream,
                                              telemetry::batch_telemetry_info const& telemetry)
{
  auto const table = get_cudf_table_view(batch);
  auto const keys  = table.select(key_columns);
  if (!cudf::has_nulls(keys)) { return std::nullopt; }
  auto* space = batch.get_memory_space();
  if (space == nullptr) {
    throw sirius::internal_exception("exchange sink: the input batch is not resident");
  }
  auto const mr = space->get_default_allocator();
  auto keyed    = cudf::is_valid(keys.column(0), stream, mr);
  for (cudf::size_type k = 1; k < keys.num_columns(); ++k) {
    auto const valid = cudf::is_valid(keys.column(k), stream, mr);
    keyed            = cudf::binary_operation(keyed->view(),
                                   valid->view(),
                                   cudf::binary_operator::LOGICAL_AND,
                                   cudf::data_type{cudf::type_id::BOOL8},
                                   stream,
                                   mr);
  }
  auto const null_keys =
    cudf::unary_operation(keyed->view(), cudf::unary_operator::NOT, stream, mr);
  return null_key_split{
    make_data_batch(
      cudf::apply_boolean_mask(table, null_keys->view(), stream, mr), *space, stream, telemetry),
    make_data_batch(
      cudf::apply_boolean_mask(table, keyed->view(), stream, mr), *space, stream, telemetry)};
}

}  // namespace

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
  if (_spec.null_keys_to_every_rank && _spec.routing != mode::hash) {
    throw sirius::invalid_input_exception(
      "exchange sink: null keys go to every rank only in hash routing");
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
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  auto const& input   = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto const& handles = input.get_data_batches();
  auto const batches  = input.get_read_only_batches();
  std::vector<std::shared_ptr<cucascade::data_batch>> slices;
  std::vector<std::size_t> ranks;
  if (!batches.empty()) {
    std::lock_guard lock{_input_schema_mutex};
    if (!_input_schema) { _input_schema = cudf::empty_like(get_cudf_table_view(batches.front())); }
  }
  for (std::size_t b = 0; b < handles.size(); ++b) {
    // Nothing to send, and a range exchange's boundaries may not exist when no rank has rows.
    if (get_cudf_table_view(batches[b]).num_rows() == 0) { continue; }
    std::vector<std::shared_ptr<cucascade::data_batch>> routed;
    auto split = _spec.null_keys_to_every_rank
                   ? split_null_keys(batches[b], _spec.key_columns, stream, batch_telemetry())
                   : std::nullopt;
    if (!split) {
      routed = route(batches[b], handles[b], stream);
    } else {
      for (exchange::rank_t r = 0; r < _service->world_size(); ++r) {
        slices.push_back(split->null_keys);
        ranks.push_back(r);
      }
      auto const keyed = split->keyed->to_read_only();
      if (get_cudf_table_view(keyed).num_rows() > 0) {
        routed = route(keyed, split->keyed, stream);
      }
    }
    for (exchange::rank_t r = 0; r < routed.size(); ++r) {
      if (!routed[r] || get_cudf_table_view(*routed[r]).num_rows() == 0) { continue; }
      slices.push_back(std::move(routed[r]));
      ranks.push_back(r);
    }
  }
  return std::make_unique<routed_operator_data>(std::move(slices), std::move(ranks));
}

std::unique_ptr<cudf::table> sirius_physical_exchange_sink::empty_like_input() const
{
  std::lock_guard lock{_input_schema_mutex};
  if (!_input_schema) { return nullptr; }
  return cudf::empty_like(_input_schema->view());
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

  // Range: the batch is sorted on the keys, so each rank's rows are one contiguous run. Rank i
  // starts at the first row not below boundary i - 1: a row equal to a boundary goes to the
  // higher rank, as SORT_PARTITION splits, and each run stays sorted.
  std::shared_ptr<cudf::table const> boundaries;
  {
    std::lock_guard lock{_boundaries_mutex};
    boundaries = _boundaries;
  }
  if (!boundaries) {
    throw sirius::internal_exception("exchange sink: range routing before its boundaries are set");
  }
  auto const table  = get_cudf_table_view(batch);
  auto const starts = cudf::lower_bound(table.select(_spec.key_columns),
                                        boundaries->view(),
                                        _spec.column_order,
                                        _spec.null_precedence,
                                        stream,
                                        space->get_default_allocator());
  std::vector<cudf::size_type> split(ranks - 1);
  CUDF_CUDA_TRY(cudaMemcpyAsync(split.data(),
                                starts->view().data<cudf::size_type>(),
                                split.size() * sizeof(cudf::size_type),
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  std::vector<cudf::size_type> bounds{0};
  for (auto const at : split) {
    bounds.push_back(at);
    bounds.push_back(at);
  }
  bounds.push_back(table.num_rows());
  auto const pieces = cudf::slice(table, bounds, stream);
  for (std::size_t r = 0; r < ranks; ++r) {
    if (pieces[r].num_rows() == 0) { continue; }
    auto owned = std::make_unique<cudf::table>(pieces[r], stream, space->get_default_allocator());
    slices[r]  = make_data_batch(std::move(owned), *space, stream, batch_telemetry());
  }
  return slices;
}

void sirius_physical_exchange_sink::sink(const operator_data& input_data,
                                         ::cuda::stream_ref /*stream*/)
{
  auto const& routed  = dynamic_cast<const routed_operator_data&>(input_data);
  auto const& batches = routed.get_data_batches();
  for (std::size_t i = 0; i < batches.size(); ++i) {
    _service->send(_stream, static_cast<exchange::rank_t>(routed.destinations()[i]), batches[i]);
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
