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
#include <helper/type_conversions.hpp>
#include <op/result/host_table_chunk_reader.hpp>
#include <op/sirius_physical_result_collector.hpp>
#include <pipeline/sirius_meta_pipeline.hpp>
#include <pipeline/sirius_pipeline.hpp>
#include <sirius_interface.hpp>

// cucascade
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

// sirius exceptions
#include "sirius/exception.hpp"

// duckdb
#include <duckdb/main/materialized_query_result.hpp>
#include <duckdb/main/prepared_statement_data.hpp>

// standard library
#include <algorithm>
#include <cassert>

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
}

std::unique_ptr<operator_data> sirius_physical_result_collector::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_result_collector::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches());
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
    result_collection(
      duckdb::make_uniq<duckdb::ColumnDataCollection>(client_ctx, sirius::to_duckdb_vec(types))),
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
      duckdb::make_uniq<duckdb::ColumnDataCollection>(_client_ctx, sirius::to_duckdb_vec(types));
  }

  return duckdb::make_uniq<duckdb::MaterializedQueryResult>(
    statement_type, properties, names, std::move(result_collection), props);
}

void sirius_physical_materialized_collector::sink(const operator_data& input_data,
                                                  rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_materialized_collector::sink"};
  auto& pipelineable_input  = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = pipelineable_input.get_data_batches();

  if (input_batches.empty()) {
    return;  // todo(kevin) we should handle this case properly
  }

  auto sink_single_batch = [this,
                            stream](std::shared_ptr<cucascade::data_batch> const& input_batch) {
    // Acquire read-only access to inspect the batch
    auto ro    = input_batch->to_read_only();
    auto* data = ro.get_data();

    if (!data) {
      throw invalid_input_exception(
        "[GPUPhysicalMaterializedCollector] data_batch has no data representation");
    }
    if (data->get_size_in_bytes() == 0) { return; }

    std::optional<cucascade::read_only_data_batch> result_ro_opt;

    if (data->get_current_tier() == cucascade::memory::Tier::GPU) {
      // Use clone_to to clone directly into HOST representation
      auto sirius_ctx  = _client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
      auto& memory_mgr = sirius_ctx->get_memory_manager();

      auto host_spaces = memory_mgr.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
      if (host_spaces.empty()) {
        throw internal_exception(
          "[GPUPhysicalMaterializedCollector] No HOST memory space available for result "
          "collection");
      }
      // Pick the host space with the most available memory.
      /// TODO: prefer the NUMA-closest host space to the source GPU for locality.
      auto const* mem_space =
        *std::max_element(host_spaces.begin(), host_spaces.end(), [](auto const* a, auto const* b) {
          return a->get_available_memory() < b->get_available_memory();
        });

      auto& registry      = sirius::converter_registry::get();
      auto& data_repo_mgr = sirius_ctx->get_data_repository_manager();
      auto next_batch_id  = data_repo_mgr.get_next_data_batch_id();

      // clone_to: creates new batch with data converted to host_data_representation
      auto result_batch = ro.clone_to<cucascade::host_data_representation>(
        registry, next_batch_id, mem_space, stream);

      // Access the result batch's data. Declared outside the if-block so result_ro outlives
      // the branch — data points into it and must not dangle when we reach the assert below.
      result_ro_opt = result_batch->to_read_only();
      data          = result_ro_opt->get_data();

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

    host_table_chunk_reader chunk_reader(_client_ctx, host_table, types);

    // Push chunks to result collection
    while (true) {
      // TODO(amin): it is fishy that append take a mutable reference to the chunk reader and we
      // are passing local variable chunk reader by reference. We should investigate if this can
      // cause any issues (e.g., if duckdb does not consume all data from the chunk reader in
      // append and we move to the next chunk reader, then the previous chunk reader's state will
      // be lost).
      duckdb::DataChunk chunk;
      if (!chunk_reader.get_next_chunk(chunk)) { break; }

      std::lock_guard<std::mutex> guard(lock);
      // Initialize result collection if it is null (from a move)
      if (!result_collection) {
        result_collection = duckdb::make_uniq<duckdb::ColumnDataCollection>(
          _client_ctx, sirius::to_duckdb_vec(types));
      }
      result_collection->Append(chunk);
    }
  };

  std::for_each(input_batches.begin(), input_batches.end(), sink_single_batch);
}

}  // namespace op
}  // namespace sirius
