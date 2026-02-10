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

#include "op/sirius_physical_concat.hpp"

#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "expression_executor/gpu_expression_executor.hpp"
#include "log/logging.hpp"
#include "op/merge/gpu_merge_impl.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_order.hpp"
#include "op/sirius_physical_partition.hpp"
#include "op/sirius_physical_top_n.hpp"

namespace sirius {
namespace op {

sirius_physical_concat::sirius_physical_concat(duckdb::vector<duckdb::LogicalType> types,
                                               duckdb::idx_t estimated_cardinality,
                                               sirius_physical_operator* parent_op,
                                               bool is_build)
  : sirius_physical_partition_consumer_operator(
      SiriusPhysicalOperatorType::CONCAT, std::move(types), estimated_cardinality)
{
  _num_partitions = (estimated_cardinality + PARTITION_SIZE - 1) / PARTITION_SIZE;
  _parent_op      = parent_op;
  _is_build       = is_build;
  // check if parent_op is a hash join
  if (parent_op->type == SiriusPhysicalOperatorType::HASH_JOIN) {
    auto hash_join = dynamic_cast<sirius_physical_hash_join*>(parent_op);
    if (hash_join->join_type == duckdb::JoinType::LEFT ||
        hash_join->join_type == duckdb::JoinType::ANTI) {
      // if the join type is left or anti, then we need to concat all the batches into one batch for
      // the build side
      _concat_all = is_build;
    } else if (hash_join->join_type == duckdb::JoinType::RIGHT ||
               hash_join->join_type == duckdb::JoinType::RIGHT_ANTI) {
      // if the join type is right or right anti, then we need to concat all the batches into one
      // batch for the probe side
      _concat_all = !is_build;
    } else if (hash_join->join_type == duckdb::JoinType::INNER ||
               hash_join->join_type == duckdb::JoinType::SEMI ||
               hash_join->join_type == duckdb::JoinType::RIGHT_SEMI ||
               hash_join->join_type == duckdb::JoinType::MARK) {
      _concat_all = false;
    } else {
      throw std::runtime_error("sirius_physical_concat: unsupported join type");
    }
  } else {
    throw std::runtime_error("sirius_physical_concat: parent_op is not a hash join");
  }
}

std::optional<std::vector<std::shared_ptr<::cucascade::data_batch>>>
sirius_physical_concat::get_next_task_input_batch()
{
  // iterate through all the partition and pull the
  std::lock_guard<std::mutex> lg(lock);

  // assert that there is only one port
  if (ports.size() != 1) {
    throw std::runtime_error("sirius_physical_concat: there should be only one port");
  }

  auto port_ptr = ports.begin()->second.get();
  for (size_t i = 0; i < port_ptr->repo->num_partitions(); i++) {
    std::vector<std::shared_ptr<::cucascade::data_batch>> input_batch;
    // get all the batch ids from the partition
    auto batch_ids          = port_ptr->repo->get_batch_ids(i);
    size_t total_batch_size = 0;
    for (auto& batch_id : batch_ids) {
      auto batch =
        port_ptr->repo->get_data_batch_by_id(batch_id, ::cucascade::batch_state::task_created, i);
      auto batch_size = batch->get_data()->get_size_in_bytes();
      total_batch_size += batch_size;
      // Check if the batch size is already exceed the threshold
      if (!_concat_all && total_batch_size > duckdb::Config::DEFAULT_SCAN_TASK_BATCH_SIZE) {
        // if the batch size is already exceed the threshold, then we need to return the batch right
        // away
        if (input_batch.size() == 0) {
          // this mean that there is a batch that is bigger than the threshold, then we just output
          // that batch right away
          auto popped_batch =
            port_ptr->repo->pop_data_batch(::cucascade::batch_state::task_created, i);
          input_batch.push_back(std::move(popped_batch));
        }
        break;
      } else {
        // if the batch size does not exceed the threshold, then we need to add the batch to the
        // input batch
        auto popped_batch =
          port_ptr->repo->pop_data_batch(::cucascade::batch_state::task_created, i);
        input_batch.push_back(std::move(popped_batch));
      }
    }
    if (input_batch.size() != 0) { return std::move(input_batch); }
  }
  return std::nullopt;
}

std::vector<std::shared_ptr<cucascade::data_batch>> sirius_physical_concat::execute(
  const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
  rmm::cuda_stream_view stream)
{
  std::vector<std::shared_ptr<cucascade::data_batch>> valid_batches;
  valid_batches.reserve(input_batches.size());
  for (auto const& batch : input_batches) {
    if (batch) { valid_batches.push_back(batch); }
  }
  if (valid_batches.empty()) { return {}; }

  cucascade::memory::memory_space* space = valid_batches[0]->get_memory_space();
  if (space == nullptr) { throw std::runtime_error("sirius_physical_concat: space is nullptr"); }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(1);
  if (valid_batches.size() == 1) {
    output_batches.push_back(valid_batches[0]);
  } else {
    auto merged_batch = gpu_merge_impl::concat(valid_batches, stream, *space);
    output_batches.push_back(std::move(merged_batch));
  }
  return output_batches;
}

std::string sirius_physical_concat::get_name() const { return "CONCAT"; }

bool sirius_physical_concat::is_source() const { return true; }

bool sirius_physical_concat::is_sink() const { return true; }

bool sirius_physical_concat::is_build_concat() { return _is_build; }

}  // namespace op
}  // namespace sirius
