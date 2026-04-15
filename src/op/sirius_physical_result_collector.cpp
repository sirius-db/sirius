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

#include <nvtx3/nvtx3.hpp>

#include <data/sirius_converter_registry.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/data/cpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// duckdb
#include <duckdb/common/exception.hpp>
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <cassert>

namespace sirius {
namespace op {

sirius_physical_result_collector::sirius_physical_result_collector(
  ::sirius::sirius_prepared_statement_data& data)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::RESULT_COLLECTOR, {duckdb::LogicalType::BOOLEAN}, 0),
    statement_type(data.prepared->statement_type),
    properties(data.prepared->properties),
    plan(*data.sirius_physical_plan),
    names(data.prepared->names)
{
  this->types = data.prepared->types;
}

std::unique_ptr<operator_data> sirius_physical_result_collector::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_result_collector::execute"};
  return std::make_unique<operator_data>(input_data);
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
  sink_state.reset();

  D_ASSERT(children.empty());

  // single operator: the operator becomes the data source of the current pipeline
  auto& state = meta_pipeline.get_state();
  state.set_pipeline_source(current, *this);

  // we create a new pipeline starting from the child
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(plan);
}

sirius_physical_materialized_collector::sirius_physical_materialized_collector(
  ::sirius::sirius_prepared_statement_data& data, duckdb::ClientContext& client_ctx)
  : sirius_physical_result_collector(data),
    _client_ctx(client_ctx),
    result_collection(duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, types))
{
}

duckdb::unique_ptr<duckdb::QueryResult> sirius_physical_materialized_collector::get_result(
  duckdb::GlobalSinkState& state)
{
  (void)state;  // Silence unused parameter warning

  auto props = _client_ctx.GetClientProperties();

  std::lock_guard<std::mutex> guard(lock);
  // Return an empty result collection if the result_collection is null (from a move)
  if (!result_collection) {
    result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, types);
  }

  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(const operator_data& input_data,
                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_materialized_collector::sink"};
  const auto& input_batches     = input_data.get_data_batches();
  using host_table_chunk_reader = ::sirius::op::result::host_table_chunk_reader;

  if (input_batches.empty()) {
    return;  // todo(kevin) we should handle this case properly
    throw duckdb::InvalidInputException("[GPUPhysicalMaterializedCollector] input_batches is null");
  }

  auto sink_single_batch = [this,
                            stream](std::shared_ptr<cucascade::data_batch> const& input_batch) {
    auto* data = input_batch->get_data();
    std::shared_ptr<cucascade::data_batch> clone_batch;
    if (!data) {
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] data_batch has no data representation");
    }
    if (data->get_size_in_bytes() == 0) { return; }

    // If data is in GPU tier, convert to HOST tier first
    if (data->get_current_tier() == cucascade::memory::Tier::GPU) {
      // Make the HOST memory reservation
      auto sirius_ctx  = _client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      auto& memory_mgr = sirius_ctx->get_memory_manager();
      /// TODO: Find the closest memory space, not just any memory space, in HOST tier
      auto reservation = memory_mgr.request_reservation(
        cucascade::memory::any_memory_space_in_tier{cucascade::memory::Tier::HOST},
        data->get_size_in_bytes());
      if (!reservation) {
        throw duckdb::InternalException(
          "[GPUPhysicalMaterializedCollector] Failed to reserve host memory for result collection");
      }

      // Convert to host representation
      auto& registry      = sirius::converter_registry::get();
      auto& mem_space     = reservation->get_memory_space();
      auto& data_repo_mgr = sirius_ctx->get_data_repository_manager();
      auto next_batch_id  = data_repo_mgr.get_next_data_batch_id();
      clone_batch         = input_batch->clone(next_batch_id, stream);
      // todo (bobbi) pass stream to sink
      clone_batch->convert_to<cucascade::host_data_representation>(registry, &mem_space, stream);
      data = clone_batch->get_data();
    } else if (data->get_current_tier() != cucascade::memory::Tier::HOST) {
      // Data must be in HOST tier (i.e., cannot currently reside in DISK tier)
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] Expected host_data_representation in HOST tier");
    }

    // Only accepting host_data_representation for now
    assert(dynamic_cast<cucascade::host_data_representation*>(data) != nullptr);

    // Push chunks to result collection
    auto const& host_table = data->cast<cucascade::host_data_representation>();
    // host_table_chunk_reader expects get_host_table() and ->allocation to be non-null;
    // otherwise it will dereference a null unique_ptr (e.g. in column_reader::initialize).
    auto const* ht = host_table.get_host_table().get();
    if (!ht) {
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] host_data_representation has null "
        "get_host_table()");
    }
    if (!ht->allocation) {
      throw duckdb::InvalidInputException(
        "[GPUPhysicalMaterializedCollector] host_table allocation is null (cannot read chunks)");
    }
    host_table_chunk_reader chunk_reader(_client_ctx, host_table, types);

    // Push chunks to result collection
    while (true) {
      // TODO(amin): it is fishy that append take a mutable reference to the chunk reader and we are
      // passing local variable chunk reader by reference. We should investigate if this can cause
      // any issues (e.g., if duckdb does not consume all data from the chunk reader in append and
      // we move to the next chunk reader, then the previous chunk reader's state will be lost).
      duckdb::DataChunk chunk;
      if (!chunk_reader.get_next_chunk(chunk)) { break; }

      std::lock_guard<std::mutex> guard(lock);
      // Initialize result collection if it is null (from a move)
      if (!result_collection) {
        result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, types);
      }
      result_collection->Append(chunk);
    }
  };

  std::for_each(input_batches.begin(), input_batches.end(), sink_single_batch);
}

}  // namespace op
}  // namespace sirius
