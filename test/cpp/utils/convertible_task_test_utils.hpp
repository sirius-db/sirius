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

#pragma once

#include "exec/multi_index_priority_queue.hpp"
#include "op/sirius_physical_operator.hpp"
#include "parallel/task.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "utils/telemetry_utils.hpp"

#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Fixtures shared by the tests that feed gpu_pipeline_tasks to the downgrade task-queue tier.
namespace sirius::test {

/// Tier of a data_batch, read under a temporary shared lock.
inline cucascade::memory::Tier get_batch_tier(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_memory_space()->get_tier();
}

/// Size in bytes of a data_batch's data, read under a temporary shared lock.
inline std::size_t get_batch_size(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return ro.get_data()->get_size_in_bytes();
}

/// Key extractor that puts every task in one priority level (plain FIFO), for tests that
/// exercise extraction and RAII return rather than queue ordering.
inline sirius::exec::multi_index_priority_queue<sirius::parallel::itask>::key_extractor
flat_key_extractor()
{
  return [](const sirius::parallel::itask&) {
    return sirius::exec::index_keys{
      0, sirius::op::SiriusPhysicalOperatorType::INVALID, 0, sirius::exec::no_preferred_device};
  };
}

/// A gpu_pipeline_task whose input is the given batches, as the task-queue tier sees it.
/// The batches stay idle.
inline std::unique_ptr<sirius::pipeline::gpu_pipeline_task> make_test_gpu_task(
  uint64_t task_id, std::vector<std::shared_ptr<cucascade::data_batch>> batches)
{
  auto op_data = std::make_unique<sirius::op::pipelineable_operator_data>(std::move(batches));
  auto local =
    std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(op_data));
  auto global = std::make_shared<sirius::pipeline::gpu_pipeline_task_global_state>(
    nullptr, make_test_telemetry_context());
  return std::make_unique<sirius::pipeline::gpu_pipeline_task>(
    task_id,
    std::vector<cucascade::shared_data_repository*>{},
    std::move(local),
    std::move(global));
}

}  // namespace sirius::test
