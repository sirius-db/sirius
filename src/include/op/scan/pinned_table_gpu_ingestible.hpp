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

// sirius
#include <compression/compressed_representation.hpp>
#include <io/gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>

// cudf
#include <cudf/column/column.hpp>

// cucascade
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/planner/expression.hpp>

// standard library
#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

namespace sirius::op::scan {

class pinned_table_post_filter_and_projection_info : public io::post_filter_and_projection_info {
 public:
  bool apply_filter   = false;
  bool apply_assembly = false;
};

class pinned_table_gpu_ingestible : public io::gpu_ingestible {
 public:
  using gpu_chunk             = std::vector<std::shared_ptr<cudf::column>>;
  using host_chunk            = std::shared_ptr<::cucascade::host_data_representation>;
  using compressed_host_chunk = std::shared_ptr<sirius::compressed_host_representation>;
  using chunk_variant         = std::variant<gpu_chunk, host_chunk, compressed_host_chunk>;

  /// GPU-tier constructor.
  pinned_table_gpu_ingestible(
    std::unique_ptr<io::ingestible_table_info> table_info,
    std::vector<std::vector<std::shared_ptr<cudf::column>>> columns_per_request,
    std::vector<::cucascade::memory::memory_space*> chunk_memory_spaces,
    std::shared_ptr<duckdb::Expression> filter_expression,
    std::shared_ptr<scan_plan const> plan);

  /// HOST-tier constructor (uncompressed).
  pinned_table_gpu_ingestible(
    std::unique_ptr<io::ingestible_table_info> table_info,
    std::vector<std::shared_ptr<::cucascade::host_data_representation>> host_chunks,
    std::vector<std::size_t> column_indices,
    ::cucascade::memory::memory_space& memory_space,
    std::unordered_map<int, ::cucascade::memory::memory_space*> gpu_memory_spaces,
    std::shared_ptr<duckdb::Expression> filter_expression,
    std::shared_ptr<scan_plan const> plan);

  /// HOST-tier constructor (Simpatico-compressed chunks).
  pinned_table_gpu_ingestible(
    std::unique_ptr<io::ingestible_table_info> table_info,
    std::vector<std::shared_ptr<sirius::compressed_host_representation>> compressed_chunks,
    std::vector<std::size_t> column_indices,
    ::cucascade::memory::memory_space& memory_space,
    std::unordered_map<int, ::cucascade::memory::memory_space*> gpu_memory_spaces,
    std::shared_ptr<duckdb::Expression> filter_expression,
    std::shared_ptr<scan_plan const> plan);

  ~pinned_table_gpu_ingestible() override;

  [[nodiscard]] bool has_more_splits() const override;
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

  io::filtered_table materialize_table(io::scan_info const& info,
                                       ::cucascade::memory::memory_space const& mem_space,
                                       rmm::cuda_stream_view stream) override;

  std::unique_ptr<cudf::table> post_filter_and_project(
    std::unique_ptr<cudf::table> input,
    io::post_filter_and_projection_info const& info,
    ::cucascade::memory::memory_space const& mem_space,
    rmm::cuda_stream_view stream) override;

 private:
  std::shared_ptr<::cucascade::data_batch> produce_batch(std::size_t batch_idx);

  std::vector<chunk_variant> _batches;
  std::vector<::cucascade::memory::memory_space*> _chunk_memory_spaces;  // GPU mode
  std::vector<std::size_t> _column_indices;                              // HOST mode
  ::cucascade::memory::memory_space* _memory_space = nullptr;            // HOST mode anchor
  std::unordered_map<int, ::cucascade::memory::memory_space*> _gpu_memory_spaces;  // HOST mode
  std::shared_ptr<duckdb::Expression> _filter_expression;
  std::shared_ptr<scan_plan const> _plan;
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::op::scan
