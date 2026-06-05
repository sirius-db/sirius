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
#include <vector>

namespace sirius::io {
class sirius_ioctx;
class sirius_io_object;
}  // namespace sirius::io

namespace sirius::scan_manager {

class duckdb_native_split_provider : public split_provider {
 public:
  struct split_payload : public op::operator_data {
    std::vector<op::scan::duckdb_row_group_metadata> row_groups;
    std::shared_ptr<op::scan::duckdb_native_scan_info const> scan_info;
    /// sirius_io substrate handles for reading .db blocks via
    /// sirius_ioctx. Always non-null (the provider requires a
    /// non-null io_ctx); the scan task threads them onto every
    // decode.
    std::shared_ptr<sirius::io::sirius_ioctx> io_ctx;
    std::shared_ptr<sirius::io::sirius_io_object> db_io_object;
  };

  explicit duckdb_native_split_provider(op::scan::duckdb_native_scan_info info,
                                        std::shared_ptr<sirius::io::sirius_ioctx> io_ctx);

  ~duckdb_native_split_provider() override;

  duckdb_native_split_provider(const duckdb_native_split_provider&)            = delete;
  duckdb_native_split_provider& operator=(const duckdb_native_split_provider&) = delete;
  duckdb_native_split_provider(duckdb_native_split_provider&&)                 = delete;
  duckdb_native_split_provider& operator=(duckdb_native_split_provider&&)      = delete;

  [[nodiscard]] bool has_more_splits() const override;

  // Each claimed split is a *row-group range*, not a finished batch. The
  // thunk emits one raw split_payload carrying the range's parsed row groups;
  // the scan operator's batch_coalescer packs those into cap-sized batches
  // (with a single tail batch). next_split_provider() itself only does the
  // cheap atomic claim.
  std::function<std::vector<std::unique_ptr<op::operator_data>>()> next_split_provider() override;

 private:
  std::shared_ptr<op::scan::duckdb_native_scan_info const> _scan_info;
  op::scan::duckdb_native_walk_plan _plan;
  std::size_t _chunk_row_groups = 1;  ///< The number of row groups claimed per range (tunable via
                                      ///< config SIRIUS_METADATA_PARSE_CHUNK)
  std::size_t _num_ranges = 0;
  std::atomic<std::size_t> _next_range_idx{0};
  /// sirius_io handles threaded onto every split so the scan task reads .db
  /// blocks via sirius_ioctx. Both non-null (validated in the ctor).
  std::shared_ptr<sirius::io::sirius_ioctx> _io_ctx;
  std::shared_ptr<sirius::io::sirius_io_object> _db_io_object;
};

}  // namespace sirius::scan_manager
