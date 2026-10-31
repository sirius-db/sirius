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
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/table_scan/split_validation.hpp"

namespace sirius::op::scan {
struct composite_scan_child {
  std::shared_ptr<gpu_ingestible> source;
  std::shared_ptr<split_validator const> validator;
  later_check_set checks;
};
// Drives lazily discovered children on query-owner and metadata-worker lanes.
// Children produce the same output schema and suppress their empty-source carriers.
// Discovery runs on the query owner; nullopt ends the whole source.
// Claim/coalescing has one coordinator. Each child lane uses bounded advance
// and returns at most one tail batch. Native-reader lanes are not supported.
class composite_scan_driver {
 public:
  using discover_child = std::function<std::optional<composite_scan_child>()>;
  composite_scan_driver(scan_contract_id,
                        uint64_t generation,
                        later_check_set supported_checks,
                        size_t target_batch_bytes,
                        discover_child);
  ~composite_scan_driver();
  composite_scan_driver(composite_scan_driver const&)            = delete;
  composite_scan_driver& operator=(composite_scan_driver const&) = delete;
  std::vector<gpu_ingestible::preparation_source> make_preparation_sources(
    io::ioctx_resolver,
    std::shared_ptr<scan_manager::preparation_ledger>,
    std::shared_ptr<pipeline::completion_handler>,
    std::function<void(scan_manager::preparation_wakeup)> bind_wakeup = {});
  std::unique_ptr<batch_coalescer> create_batch_coalescer() const;
  std::shared_ptr<split_validator const> validator() const;
  bool exhausted() const;
  // Called after source jobs and close have drained. Splits keep their child alive.
  void stop_preparation() noexcept;
  void finish_preparation() noexcept;
  // Validate with the bound validator before materializing. nullopt denotes the
  // whole-source empty carrier; the caller supplies its schema.
  std::optional<filtered_table> materialize(scan_info const&,
                                            cucascade::memory::memory_space const&,
                                            ::cuda::stream_ref,
                                            bool,
                                            std::shared_ptr<sirius::like_multiliteral_cache const>,
                                            expression_query_policy);

 private:
  struct impl;
  std::shared_ptr<impl> impl_;
};
}  // namespace sirius::op::scan
