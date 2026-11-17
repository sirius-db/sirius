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

#include "op/partition/partition_spec.hpp"
#include "op/sirius_physical_operator.hpp"

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op {

class sirius_physical_partition;

struct partition_input_stats {
  uint64_t bytes = 0;
  uint64_t rows  = 0;
  /// Exact metadata rows and decoded bytes, available only for unprojected measurements.
  std::optional<uint64_t> measured_rows  = 0;
  std::optional<uint64_t> resident_bytes = 0;
  /// The bytes are a deferred scan's known total, not what this input received.
  bool from_deferred_scan = false;
};

/// A join input that is a plain scan with no static filter. Its size is known before the query
/// runs and nothing feeds it, so it can start whenever the layout exists.
struct deferred_scan {
  std::size_t input;  ///< Its index among the exchange's inputs; never the sizing input.
  uint64_t bytes;     ///< What this process will receive from it.
};

/// One plan-bound decision, shared by its input routers. No semantic consumer roles live here.
/// The physical plan owns the routers and the consumer captured by choose_layouts.
class partition_exchange {
 public:
  using choose_layouts = std::function<std::vector<partition_layout>(
    std::vector<partition_input_stats> const&, bool /* completed empty sizing input */)>;

  /// Bind before conversion. Every router is configured before the first layout is published.
  static void bind(std::vector<sirius_physical_partition*> inputs,
                   std::size_t sizing_input,
                   bool requires_all_inputs,
                   choose_layouts choose);

  /// Declares @p input a deferred scan of @p bytes. Before conversion. The sizing input is
  /// collected whole and the other input waits for it, so a sizing input larger than the other
  /// is held for nothing: once it has received @p bytes, the layout is sized from the scan
  /// instead, the scan starts, and the sizing input is partitioned as it arrives.
  void defer_scan(sirius_physical_partition const& input, uint64_t bytes);

  /// Whether the layout was sized from the deferred scan while the sizing input was incomplete.
  [[nodiscard]] bool scan_holds() const noexcept
  {
    return _scan_holds.load(std::memory_order_acquire);
  }

  /// Whether `input` is the sizing input, the one whose measurement resolves the shared layout.
  [[nodiscard]] bool is_sizing_input(sirius_physical_partition const& input) const
  {
    return _inputs[_sizing_input] == &input;
  }

  /// nullopt means resolved. Otherwise schedule the indicated producer before trying again.
  std::optional<task_creation_hint> prepare();

  /// Nonempty unresolved inputs nominate the sizing input without running consumer policy.
  /// This preserves build-first task creation and filter-publication dependencies.
  std::optional<task_creation_hint> hint();

 private:
  std::optional<task_creation_hint> prepare_impl(bool resolve);
  partition_exchange(std::vector<sirius_physical_partition*> inputs,
                     std::size_t sizing_input,
                     bool requires_all_inputs,
                     choose_layouts choose);
  std::vector<sirius_physical_partition*> _inputs;
  std::size_t _sizing_input;
  bool _requires_all_inputs;
  choose_layouts _choose;
  std::optional<deferred_scan> _deferred_scan;
  std::atomic<bool> _scan_holds{false};
  std::mutex _mutex;
  bool _resolved = false;
};

}  // namespace sirius::op
