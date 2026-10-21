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

#include <functional>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op {

class sirius_physical_partition;

struct partition_input_stats {
  uint64_t bytes = 0;
  uint64_t rows  = 0;
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
  std::mutex _mutex;
  bool _resolved = false;
};

}  // namespace sirius::op
