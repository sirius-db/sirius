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

// sirius
#include <data/data_batch_utils.hpp>
#include <data/spill_chunked_converters.hpp>
#include <helper/type_conversions.hpp>
#include <memory/host_reservation.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>
#include <telemetry/nvtx.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// sirius exceptions
#include "sirius/exception.hpp"

// duckdb
#include <duckdb/main/prepared_statement_data.hpp>
#include <duckdb/main/query_result.hpp>

// standard library
#include <algorithm>
#include <cassert>
#include <utility>

namespace sirius {
namespace op {

sirius_physical_result_collector::sirius_physical_result_collector(
  ::sirius::sirius_prepared_statement_data& data)
  : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR,
                             {sirius::logical_type::make(sirius::type_id::BOOLEAN)},
                             0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = sirius::from_duckdb_vec(data.prepared->types);
  // Full DuckDB types incl. nested children — sirius::logical_type cannot
  // represent them.
  this->result_column_types = data.prepared->types;
}

std::unique_ptr<operator_data> sirius_physical_result_collector::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_result_collector::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_data_batches(),
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_position());
}

duckdb::vector<duckdb::const_reference<sirius_physical_operator>>
sirius_physical_result_collector::get_children() const
{
  return {plan};
}

void sirius_physical_result_collector::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // operator is a sink, build a pipeline
  D_ASSERT(children.empty());

  // RESULT_COLLECTOR is both the root of `current` (appended here, operators[0]
  // post-reverse) and the sink of its own child_meta (pre-populated by
  // create_child_meta_pipeline).
  auto& state = meta_pipeline.get_state();
  state.add_pipeline_operator(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  ::sirius::sirius_prepared_statement_data& data, duckdb::ClientContext& client_ctx)
  : sirius_physical_result_collector(data),
    result_collection(
      duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, result_column_types)),
    _client_ctx(client_ctx)
{
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result()
{
  auto props = _client_ctx.GetClientProperties();

  std::lock_guard<std::mutex> guard(lock);
  // Return an empty result collection if the result_collection is null (from a move)
  if (!result_collection) {
    result_collection =
      duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, result_column_types);
  }

  // All producer tasks have finished before get_result(). Sparse positions are intentional.
  for (auto& [position, collection] : _positioned_collections) {
    result_collection->Combine(*collection);
  }
  _positioned_collections.clear();

  return duckdb::make_uniq<duckdb::QueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(const operator_data& input_data,
                                                  ::cuda::stream_ref stream)
{
  std::unique_ptr<cucascade::memory::reservation> reservation;
  sink_with_reservation(input_data, stream, reservation);
}

void sirius_physical_materialized_collector::sink_with_reservation(
  const operator_data& input_data,
  ::cuda::stream_ref stream,
  std::unique_ptr<cucascade::memory::reservation>& reservation)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_materialized_collector::sink"};
  using cucascade::memory::Tier;

  auto const& input   = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto const& batches = input.get_data_batches();
  if (batches.empty()) { return; }

  auto position = input.get_position();
  if (position && batches.size() != 1) {
    throw internal_exception("Positioned result requires one output batch");
  }

  // Retain the read locks from sizing through the copies so that the staging requirement is
  // computed from the exact representations to be copied.
  auto inputs = input.get_read_only_batches();
  if (inputs.size() != batches.size()) {
    throw invalid_input_exception("Materialized result contains a null batch");
  }
  std::optional<std::size_t> staging_bytes;
  for (const auto& ro : inputs) {
    auto const* data = ro.get_data();
    if (!data) { throw invalid_input_exception("Materialized result batch has no data"); }
    if (data->get_current_tier() == Tier::GPU && data->get_size_in_bytes() != 0) {
      staging_bytes = std::max(staging_bytes.value_or(0),
                               spill::gpu_to_host_staging_bytes(
                                 data->cast<cucascade::gpu_table_representation>(), stream));
    }
  }

  std::size_t copy_chunk_bytes = 0;
  if (staging_bytes) {
    const auto ctx   = _client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
    copy_chunk_bytes = ctx->get_config().get_downgrade_executor_config().copy_chunk_bytes;
    if (reservation) {
      auto required =
        memory::host_reservation_size(reservation->get_memory_space(), *staging_bytes);
      if (!required || reservation->size() < *required) { reservation.reset(); }
    }
    if (!reservation) {
      auto host_spaces = cucascade::memory::any_memory_space_in_tier{Tier::HOST}.get_candidates(
        ctx->get_memory_manager());
      reservation = memory::try_reserve_host(host_spaces, *staging_bytes);
      if (!reservation) { throw memory::host_reservation_required(*staging_bytes); }
    }
  }

  auto local_collection =
    duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, result_column_types);
  duckdb::ColumnDataAppendState append_state;
  local_collection->InitializeAppend(append_state);

  for (const auto& ro : inputs) {
    auto const* data = ro.get_data();
    if (data->get_size_in_bytes() == 0) { continue; }

    std::optional<cucascade::read_only_data_batch> result_ro;
    if (data->get_current_tier() == Tier::GPU) {
      auto next_batch_id = sirius::get_next_batch_id();
      auto probe = telemetry::quent_data_batch_probe::create(batch_telemetry(), next_batch_id);
      auto host  = spill::copy_gpu_to_host(
        data->cast<cucascade::gpu_table_representation>(), *reservation, stream, copy_chunk_bytes);
      auto result_batch =
        cucascade::data_batch::make(next_batch_id, std::move(host), std::move(probe));
      result_ro = result_batch->to_read_only();
      data      = result_ro->get_data();
    } else if (data->get_current_tier() != cucascade::memory::Tier::HOST) {
      // Data must be in HOST tier (i.e., cannot currently reside in DISK tier)
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] Expected host_data_representation in HOST tier");
    }
    // Data already in HOST tier -- read directly through the read_only accessor
    assert(dynamic_cast<const cucascade::host_data_representation*>(data) != nullptr);

    using host_table_chunk_reader = ::sirius::op::result::host_table_chunk_reader;
    auto const& host_table        = data->cast<cucascade::host_data_representation>();
    // host_table_chunk_reader expects get_host_table() and ->allocation to be non-null;
    // otherwise it will dereference a null unique_ptr (e.g. in column_reader::initialize).
    auto const* ht = host_table.get_host_table().get();
    if (!ht) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] host_data_representation has null "
        "get_host_table()");
    }
    if (!ht->allocation) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] host_table allocation is null (cannot read chunks)");
    }

    host_table_chunk_reader chunk_reader(_client_ctx, host_table, result_column_types);
    while (true) {
      duckdb::DataChunk chunk;
      if (!chunk_reader.get_next_chunk(chunk)) { break; }
      local_collection->Append(append_state, chunk);
    }
  }

  // Every batch has been staged and its HOST blocks returned. Publish the whole task at once.
  reservation.reset();
  std::lock_guard<std::mutex> guard(lock);
  if (_positioned && *_positioned != position.has_value()) {
    throw internal_exception("Cannot mix positioned and unpositioned query results");
  }
  if (position) {
    if (!_positioned_collections.try_emplace(*position, std::move(local_collection)).second) {
      throw internal_exception("Duplicate query result position");
    }
  } else {
    if (!result_collection) {
      result_collection =
        duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, result_column_types);
    }
    result_collection->Combine(*local_collection);
  }
  _positioned = position.has_value();
}

}  // namespace op
}  // namespace sirius
