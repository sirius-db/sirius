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

// Fixtures for readahead ticket tests: a real on-disk data file, a one-GPU topology,
// and a one-scan query that parks the readahead worker.

#include "memory/topology_index.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/pipeline_build_context.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "planner/query.hpp"
#include "query_id.hpp"
#include "utils/telemetry_utils.hpp"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

namespace sirius::test {

/// A one-GPU topology (device 0, NUMA node 0) for a scan manager under test.
inline std::shared_ptr<const sirius::memory::topology_index> single_gpu_index_for_prefetch()
{
  cucascade::memory::system_topology_info topology;
  topology.num_gpus = 1;
  cucascade::memory::gpu_topology_info gpu;
  gpu.id        = 0;
  gpu.numa_node = 0;
  topology.gpus.push_back(std::move(gpu));
  return std::make_shared<sirius::memory::topology_index>(topology, std::vector<int>{0});
}

/// A real on-disk file, so the uring backend opens it and the cache sizes its
/// slot table from a genuine object size.
struct temp_data_file {
  std::filesystem::path path;

  explicit temp_data_file(std::size_t bytes)
  {
    path = std::filesystem::temp_directory_path() /
           ("sirius_scan_input_prefetch_" + std::to_string(::getpid()) + "_" +
            std::to_string(reinterpret_cast<std::uintptr_t>(this)));
    std::ofstream out(path, std::ios::binary);
    std::vector<char> data(bytes, 'z');
    out.write(data.data(), static_cast<std::streamsize>(bytes));
  }

  ~temp_data_file()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  temp_data_file(temp_data_file const&)            = delete;
  temp_data_file& operator=(temp_data_file const&) = delete;
};

/// Minimal concrete scan carrying only the operator id the readahead keys its
/// work queue on.
struct test_scan : sirius::op::sirius_physical_operator {
  test_scan()
    : sirius::op::sirius_physical_operator(sirius::op::SiriusPhysicalOperatorType::GPU_SCAN, {}, 0)
  {
    operator_id = 1;
  }
};

/// A query with one GPU_SCAN operator, which is what it takes to park the
/// worker: with no work queues at all it runs out of order and exits on its
/// own, and a worker that is already gone hides everything a second start does.
class single_scan_query {
 public:
  single_scan_query()
  {
    auto pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(_ctx);
    _build.set_pipeline_source(*pipeline, *_scan);
    auto const id = sirius::make_query_id(1);
    _query        = std::make_unique<sirius::planner::query>(
      std::vector<std::shared_ptr<sirius::pipeline::sirius_pipeline>>{pipeline},
      _telemetry->context(),
      id,
      sirius::telemetry::query_telemetry_info{quent::now_v7(), _telemetry->worker_id().raw(), id});
  }

  [[nodiscard]] const sirius::planner::query& get() const { return *_query; }

 private:
  std::shared_ptr<const sirius::telemetry::telemetry_context> _telemetry =
    sirius::test::make_test_telemetry_context();
  sirius::pipeline::pipeline_build_context _ctx{nullptr, true};
  sirius::pipeline::sirius_pipeline_build_state _build;
  std::unique_ptr<test_scan> _scan = std::make_unique<test_scan>();
  std::unique_ptr<sirius::planner::query> _query;
};

}  // namespace sirius::test
