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

#include "op/scan/duckdb_native_metadata.hpp"
#include "op/scan/duckdb_native_scan_info.hpp"
#include "op/sirius_physical_operator.hpp"
#include "scan_manager/split_provider.hpp"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sirius::scan_manager {

class duckdb_native_split_provider : public split_provider {
 public:
  struct split_payload : public op::operator_data {
    std::vector<op::scan::duckdb_row_group_metadata> row_groups;
    std::shared_ptr<op::scan::duckdb_native_scan_info const> scan_info;
  };

  explicit duckdb_native_split_provider(op::scan::duckdb_native_scan_info info);

  ~duckdb_native_split_provider() override;

  duckdb_native_split_provider(const duckdb_native_split_provider&)            = delete;
  duckdb_native_split_provider& operator=(const duckdb_native_split_provider&) = delete;
  duckdb_native_split_provider(duckdb_native_split_provider&&)                 = delete;
  duckdb_native_split_provider& operator=(duckdb_native_split_provider&&)      = delete;

  [[nodiscard]] bool has_more_splits() const override;

  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

 private:
  struct row_group_batch {
    std::size_t first_idx;
    std::size_t count;
  };

  std::shared_ptr<op::scan::duckdb_native_scan_info const> _scan_info;
  op::scan::duckdb_native_metadata _metadata;
  std::vector<row_group_batch> _batches;
  std::atomic<std::size_t> _next_batch_idx{0};
};

}  // namespace sirius::scan_manager
