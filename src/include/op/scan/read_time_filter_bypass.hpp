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

#include <atomic>

namespace sirius::op::scan {

/// @brief One-way, per-execution latch: this scan's batches will not pass through a read-time
/// dynamic-filter phase in this execution.
///
/// A pinned-cache hit serves every split of a parquet scan from resident chunks, so the reader AST
/// that normally consumes AST-capable dynamic filters never runs; the scan's post-decode wrapper
/// reads this latch to promote its plan-time `membership_masks_only` mode to
/// `include_ast_row_masks` (main doc, "Pinned-cache-served scans").
///
/// Protocol: created by the plan generator when it wraps a scan with a dynamic-filter operator;
/// co-owned by `sirius_gpu_scan_operator` and `sirius_physical_dynamic_filter`. Single writer:
/// `sirius_scan_manager::prepare_for_query` marks it (through
/// `sirius_gpu_scan_operator::mark_served_from_pinned_cache`) strictly before pipeline execution
/// starts (`SiriusContext::create_query` sequencing), so wrapper tasks never observe a mid-query
/// change and the keep-ratio gate trains under one mode. Physical plans are per-execution
/// ("Execution-scoped state" in the main doc), so the latch needs no reset; prepared-plan reuse
/// would require one and is forbidden until then.
class read_time_filter_bypass {
 public:
  void mark_bypassed() noexcept { _bypassed.store(true, std::memory_order_relaxed); }

  [[nodiscard]] bool bypassed() const noexcept { return _bypassed.load(std::memory_order_relaxed); }

 private:
  std::atomic<bool> _bypassed{false};
};

}  // namespace sirius::op::scan
