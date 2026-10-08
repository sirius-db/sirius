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
#include "op/scan/table_scan/scan_contract.hpp"
#include "transparent/replay_admission.hpp"

#include <bitset>
#include <compare>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace sirius::op::scan {
struct iceberg_delete_set;
}
namespace sirius::scan_manager {
enum class required_input : uint8_t { footer, segments, delete_set, checkpoint_iteration };
using required_input_set = std::bitset<4>;
enum class unit_state : uint8_t { pending, ready, failed, cancelled };
struct unit_key {
  op::scan::scan_contract_id contract;
  uint64_t unit_id;
  auto operator<=>(unit_key const&) const = default;
};
struct preparation_failure {
  transparent::late_failure_cause cause;
  std::optional<op::scan::verdict_reason> reason;
  std::string detail;
  std::exception_ptr original;
  void validate() const;
};
struct unit_record {
  unit_state state = unit_state::pending;
  required_input_set completed;
  std::shared_ptr<op::scan::split_dependencies const> deps;
  std::optional<preparation_failure> failure;
};
struct footer_input {
  std::shared_ptr<cudf::io::parquet::FileMetaData const> footer;
  std::shared_ptr<op::scan::parquet_input_approval const> approval;
  std::shared_ptr<io::sirius_datasource> datasource;
};
struct segments_input {
  std::shared_ptr<op::scan::physical_profile_table> profiles;
};
struct delete_set_input {
  std::shared_ptr<op::scan::iceberg_delete_set const> value;
};
struct checkpoint_input {
  uint64_t iteration;
};

struct preparation_gate {
  std::mutex mutex;
  std::condition_variable cv;
  // Bound before arm; callers hold mutex while checking admission or publication.
  std::function<bool()> interrupted;
  void check_interrupted();
  bool closed      = false;
  size_t callbacks = 0;
  std::function<void(std::exception_ptr)> report_error;
  std::function<void(preparation_failure const&)> report_failure;
};
// The attempt owns registration; workers receive only this unit and independent inputs.
// Readiness is not publication: the attempt's publication gate is checked separately.
class preparation_unit {
 public:
  preparation_unit(unit_key key,
                   required_input_set required,
                   std::shared_ptr<preparation_gate> gate = {});
  unit_key key() const noexcept { return key_; }
  required_input_set required() const noexcept { return required_; }
  bool complete_input(footer_input);
  bool complete_input(segments_input);
  bool complete_input(delete_set_input);
  bool complete_input(checkpoint_input);
  bool fail(preparation_failure);
  bool cancel();
  unit_record record() const;

 private:
  friend class preparation_coordinator;
  bool cancel_under_gate();
  bool accept(required_input);
  void finish_input(required_input);
  std::shared_ptr<preparation_gate> gate_;
  unit_key key_;
  required_input_set required_;
  mutable std::mutex mutex_;
  unit_record record_;
  op::scan::split_dependencies deps_;
};
// Map only actual dependencies; a connector explicitly contributes delete membership closure.
required_input_set required_inputs(op::scan::later_check_set const&, bool needs_delete_set);
transparent::failure_cause classify_failure(preparation_failure const&);
}  // namespace sirius::scan_manager
