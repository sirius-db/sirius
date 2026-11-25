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

#include "helper/type_conversions.hpp"
#include "operator_test_utils.hpp"

#include <cudf/utilities/default_stream.hpp>

#include <cucascade/data/data_repository.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_merge_sort.hpp>
#include <op/sirius_physical_sort_key_sample.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <pipeline/sirius_pipeline.hpp>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace sirius::test::sort_range {

using key_cell = std::variant<std::monostate, std::int64_t, std::string>;
using key_row  = std::vector<key_cell>;

struct layout {
  std::uint64_t input_batches = 0;
  op::sirius_physical_sort_sample::range_sample_stats sample;
  std::vector<op::sirius_physical_sort_partition::range_size> counted;
  std::vector<op::sirius_physical_sort_partition::range_size> published;
  std::vector<std::vector<key_row>> keys;
};

class mock_pipeline : public pipeline::sirius_pipeline {
 public:
  explicit mock_pipeline(bool finished)
    : sirius_pipeline(pipeline::pipeline_build_context{nullptr}), _finished(finished)
  {
  }
  void set_finished(bool finished) { _finished = finished; }
  bool is_pipeline_finished() const override { return _finished; }

 private:
  bool _finished;
};

inline duckdb::LogicalType duckdb_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT32: return duckdb::LogicalType::INTEGER;
    case cudf::type_id::INT64: return duckdb::LogicalType::BIGINT;
    case cudf::type_id::STRING: return duckdb::LogicalType::VARCHAR;
    default: throw std::runtime_error("Unsupported sort-range test key type");
  }
}

inline std::vector<key_cell> read_column(cudf::column_view const& col)
{
  auto valid = operator_utils::copy_validity_to_host(col);
  std::vector<key_cell> result(col.size());
  switch (col.type().id()) {
    case cudf::type_id::INT32: {
      auto values = operator_utils::copy_column_to_host<std::int32_t>(col);
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = valid[i] ? key_cell{std::int64_t{values[i]}} : key_cell{std::monostate{}};
      }
      break;
    }
    case cudf::type_id::INT64: {
      auto values = operator_utils::copy_column_to_host<std::int64_t>(col);
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = valid[i] ? key_cell{values[i]} : key_cell{std::monostate{}};
      }
      break;
    }
    case cudf::type_id::STRING: {
      auto values = operator_utils::copy_column_to_host<std::string>(col);
      for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = valid[i] ? key_cell{values[i]} : key_cell{std::monostate{}};
      }
      break;
    }
    default: throw std::runtime_error("Unsupported sort-range test key type");
  }
  return result;
}

inline std::vector<key_row> read_keys(cucascade::data_batch& batch,
                                      std::vector<std::size_t> const& key_columns)
{
  auto table = sirius::get_cudf_table_view(batch);
  std::vector<key_row> rows(table.num_rows());
  for (auto const col_index : key_columns) {
    auto values = read_column(table.column(col_index));
    for (std::size_t row = 0; row < rows.size(); ++row) {
      rows[row].push_back(std::move(values[row]));
    }
  }
  return rows;
}

//! Keeps the key rows of @p batches for @p sample, as the SORT_KEY_SAMPLE that ends the producing
//! pipeline does before the batches reach SORT_SAMPLE's input port.
inline void keep_key_rows(op::sirius_physical_sort_sample& sample,
                          std::vector<std::shared_ptr<cucascade::data_batch>> const& batches,
                          ::cuda::stream_ref stream = cudf::get_default_stream())
{
  op::sirius_physical_sort_key_sample key_sample(sample.types,
                                                 op::copy_orders(sample.orders),
                                                 sample.estimated_cardinality,
                                                 sample.key_sample());
  (void)key_sample.execute(op::pipelineable_operator_data(batches), stream);
}

//! Runs @p batches through SORT_KEY_SAMPLE, SORT_SAMPLE, SORT_PARTITION and MERGE_SORT as the
//! pipelines do: each batch gives its key rows and is pushed in turn, and every task that is ready
//! runs before the next batch arrives. The input ends, if @p upstream_finished, only after the
//! last batch. @p max_sample_key_rows of 0 keeps the operator's cap.
inline layout run_layout(std::vector<std::shared_ptr<cucascade::data_batch>> const& batches,
                         duckdb::vector<duckdb::BoundOrderByNode> const& orders,
                         std::uint64_t sort_sample_bytes,
                         std::uint64_t max_partition_bytes,
                         std::size_t estimated_cardinality,
                         bool upstream_finished,
                         std::size_t max_sample_key_rows = 0)
{
  if (batches.empty()) { throw std::runtime_error("Sort-range test needs input"); }
  auto table = sirius::get_cudf_table_view(*batches.front());
  duckdb::vector<duckdb::LogicalType> duck_types;
  for (auto const& col : table) {
    duck_types.push_back(duckdb_type(col.type().id()));
  }
  auto types = sirius::from_duckdb_vec(duck_types);
  op::sirius_physical_sort_sample sample(
    types, op::copy_orders(orders), estimated_cardinality, sort_sample_bytes, max_partition_bytes);
  if (max_sample_key_rows != 0) { sample.set_max_sample_key_rows(max_sample_key_rows); }
  op::sirius_physical_sort_key_sample key_sample(
    types, op::copy_orders(orders), estimated_cardinality, sample.key_sample());
  op::sirius_physical_sort_partition partition(
    types, op::copy_orders(orders), {}, estimated_cardinality);
  partition.set_sample_op(&sample);
  op::sirius_physical_merge_sort merge(types, op::copy_orders(orders), {}, estimated_cardinality);
  cucascade::shared_data_repository input_repo;
  cucascade::data_repository output_repo;
  auto source        = std::make_shared<mock_pipeline>(false);
  auto port          = std::make_unique<op::sirius_physical_operator::port>();
  port->type         = op::MemoryBarrierType::PIPELINE;
  port->repo         = &input_repo;
  port->src_pipeline = source;
  sample.add_port("input", std::move(port));
  merge.add_port("default",
                 std::make_unique<op::sirius_physical_operator::port>(
                   op::MemoryBarrierType::FULL, &output_repo, nullptr, nullptr));
  partition.add_next_port_after_sink({&merge, "default"});
  auto stream                = cudf::get_default_stream();
  auto const run_ready_tasks = [&] {
    while (auto next = sample.get_next_task_input_data()) {
      auto sampled = sample.execute(*next, stream);
      auto cut     = partition.execute(*sampled, stream);
      partition.sink(*cut, stream);
    }
  };
  for (auto const& batch : batches) {
    std::vector<std::shared_ptr<cucascade::data_batch>> const one{batch};
    (void)key_sample.execute(op::pipelineable_operator_data(one), stream);
    input_repo.add_data_batch(batch, 0);
    run_ready_tasks();
  }
  source->set_finished(upstream_finished);
  run_ready_tasks();
  layout result;
  result.input_batches = sample.get_input_batch_count();
  result.sample        = sample.get_range_sample_stats();
  result.counted       = partition.get_range_sizes();
  result.published.resize(output_repo.num_partitions());
  result.keys.resize(output_repo.num_partitions());
  std::vector<std::size_t> key_columns;
  for (auto const& order : orders) {
    key_columns.push_back(order.expression->Cast<duckdb::BoundReferenceExpression>().Index());
  }
  for (std::size_t range = 0; range < output_repo.num_partitions(); ++range) {
    std::vector<std::shared_ptr<cucascade::data_batch>> range_batches;
    for (auto id : output_repo.get_batch_ids(range)) {
      auto batch = output_repo.get_data_batch_by_id(id, range);
      range_batches.push_back(batch);
      auto& actual = result.published[range];
      actual.rows += sirius::get_cudf_table_view(*batch).num_rows();
      actual.bytes += batch->to_read_only().get_data()->get_size_in_bytes();
      ++actual.slices;
    }
    if (!range_batches.empty()) {
      auto merged = merge.execute(op::pipelineable_operator_data(range_batches), stream);
      for (auto const& batch :
           dynamic_cast<op::pipelineable_operator_data const&>(*merged).get_data_batches()) {
        auto values = read_keys(*batch, key_columns);
        result.keys[range].insert(result.keys[range].end(), values.begin(), values.end());
      }
    }
  }
  return result;
}

}  // namespace sirius::test::sort_range
