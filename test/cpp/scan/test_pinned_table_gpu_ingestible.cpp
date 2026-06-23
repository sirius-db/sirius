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

// Unit tests for pinned_table_gpu_ingestible constructor-argument
// invariants. These guard the contract that the ingestible enforces in
// each ctor body (replacing the equivalent checks in the deleted
// cached_split_provider). End-to-end pinned-table behavior is covered by
// the [scan_manager] tests (test_pin_table_multi_gpu.cpp) and TPC-H
// integration with CALL pin_table().

#include "catch.hpp"

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/pinned_table_gpu_ingestible.hpp>
#include <op/scan/scan_plan.hpp>

#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

namespace sscan = sirius::op::scan;

// Empty plan suffices for the ctor-argument validation tests — no
// chunks/columns means assemble_scan_output is never invoked.
std::shared_ptr<sscan::scan_plan const> empty_plan()
{
  return std::make_shared<sscan::scan_plan const>(sscan::scan_plan{});
}

std::unique_ptr<sscan::parquet_ingestible_table_info> empty_parquet_table_info()
{
  return std::make_unique<sscan::parquet_ingestible_table_info>();
}

}  // namespace

TEST_CASE("pinned_table_gpu_ingestible - GPU ctor rejects mismatched chunk counts",
          "[scan][pinned_table_gpu_ingestible]")
{
  // columns_per_request[0] has 2 chunks, columns_per_request[1] has 1 —
  // the ctor must throw because the cross-column chunk count is
  // inconsistent (every column must contribute the same number of chunks).
  std::vector<std::vector<std::shared_ptr<cudf::column>>> columns;
  columns.emplace_back();
  columns.back().push_back(nullptr);
  columns.back().push_back(nullptr);
  columns.emplace_back();
  columns.back().push_back(nullptr);

  std::vector<::cucascade::memory::memory_space*> chunk_memory_spaces(2, nullptr);

  REQUIRE_THROWS_AS(sscan::pinned_table_gpu_ingestible(empty_parquet_table_info(),
                                                       std::move(columns),
                                                       std::move(chunk_memory_spaces),
                                                       /*filter_expression=*/nullptr,
                                                       empty_plan()),
                    std::runtime_error);
}

TEST_CASE("pinned_table_gpu_ingestible - GPU ctor with no chunks has no work",
          "[scan][pinned_table_gpu_ingestible]")
{
  // Empty columns_per_request → no batches → has_more_splits returns false
  // immediately. The split_provider::run loop would no-op on this and
  // close the connector; downstream consumers see "drained, no error."
  sscan::pinned_table_gpu_ingestible ingestible(empty_parquet_table_info(),
                                                /*columns_per_request=*/{},
                                                /*chunk_memory_spaces=*/{},
                                                /*filter_expression=*/nullptr,
                                                empty_plan());

  REQUIRE_FALSE(ingestible.has_more_splits());
}
