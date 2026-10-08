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

#include "scan_manager/preparation.hpp"

#include "duckdb/common/exception.hpp"
#include "op/scan/iceberg_delete_set.hpp"

namespace sirius::scan_manager {
void preparation_gate::check_interrupted()
{
  if (interrupted && interrupted()) {
    closed = true;
    cv.notify_all();
    throw duckdb::InterruptException();
  }
}
void preparation_failure::validate() const
{
  if (reason && cause != transparent::late_failure_cause::physical_input)
    throw std::invalid_argument("preparation verdict reason requires physical_input cause");
}
transparent::failure_cause classify_failure(preparation_failure const& failure)
{
  failure.validate();
  return {failure.cause, failure.detail};
}
preparation_unit::preparation_unit(unit_key key,
                                   required_input_set required,
                                   std::shared_ptr<preparation_gate> gate)
  : gate_(std::move(gate)), key_(key), required_(required)
{
  if (required_.none()) {
    record_.state = unit_state::ready;
    record_.deps  = std::make_shared<op::scan::split_dependencies const>(deps_);
  }
}
bool preparation_unit::accept(required_input input)
{
  auto bit = static_cast<size_t>(input);
  return record_.state == unit_state::pending && required_[bit] && !record_.completed[bit];
}
void preparation_unit::finish_input(required_input input)
{
  // Construct the immutable snapshot before committing the last bit/terminal state.
  auto completed = record_.completed;
  completed.set(static_cast<size_t>(input));
  if ((completed & required_) == required_) {
    record_.deps  = std::make_shared<op::scan::split_dependencies const>(deps_);
    record_.state = unit_state::ready;
  }
  record_.completed = completed;
  if (gate_) gate_->cv.notify_all();
}
bool preparation_unit::complete_input(footer_input input)
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) {
    gate_lock = std::unique_lock(gate_->mutex);
    if (gate_->closed) return false;
  }
  std::lock_guard lock(mutex_);
  if (!accept(required_input::footer)) return false;
  if (!input.footer || !input.approval) throw std::invalid_argument("footer input lacks approval");
  deps_.footer           = std::move(input.footer);
  deps_.parquet_approval = std::move(input.approval);
  deps_.datasource       = std::move(input.datasource);
  finish_input(required_input::footer);
  return true;
}
bool preparation_unit::complete_input(segments_input input)
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) {
    gate_lock = std::unique_lock(gate_->mutex);
    if (gate_->closed) return false;
  }
  std::lock_guard lock(mutex_);
  if (!accept(required_input::segments)) return false;
  if (!input.profiles) throw std::invalid_argument("segments input is absent");
  deps_.profiles = std::move(input.profiles);
  finish_input(required_input::segments);
  return true;
}
bool preparation_unit::complete_input(delete_set_input input)
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) {
    gate_lock = std::unique_lock(gate_->mutex);
    if (gate_->closed) return false;
  }
  std::lock_guard lock(mutex_);
  if (!accept(required_input::delete_set)) return false;
  if (!input.value) throw std::invalid_argument("pending delete set is not an empty result");
  deps_.delete_set = std::move(input.value);
  finish_input(required_input::delete_set);
  return true;
}
bool preparation_unit::complete_input(checkpoint_input input)
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) {
    gate_lock = std::unique_lock(gate_->mutex);
    if (gate_->closed) return false;
  }
  std::lock_guard lock(mutex_);
  if (!accept(required_input::checkpoint_iteration)) return false;
  deps_.checkpoint_iteration = input.iteration;
  finish_input(required_input::checkpoint_iteration);
  return true;
}
bool preparation_unit::fail(preparation_failure failure)
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) {
    gate_lock = std::unique_lock(gate_->mutex);
    if (gate_->closed) return false;
  }
  {
    std::lock_guard lock(mutex_);
    if (record_.state != unit_state::pending) return false;
    failure.validate();
    record_.failure = failure;
    record_.state   = unit_state::failed;
    deps_           = {};
  }
  if (gate_) {
    // arm binds once; this callback stays stable until accepted users have drained.
    auto* report = gate_->report_failure ? &gate_->report_failure : nullptr;
    if (report) ++gate_->callbacks;
    gate_->cv.notify_all();
    gate_lock.unlock();
    if (report) {
      struct callback_use {
        std::shared_ptr<preparation_gate> gate;
        ~callback_use()
        {
          std::lock_guard lock(gate->mutex);
          --gate->callbacks;
          gate->cv.notify_all();
        }
      } use{gate_};
      (*report)(failure);
    }
  }

  return true;
}
bool preparation_unit::cancel()
{
  std::unique_lock<std::mutex> gate_lock;
  if (gate_) gate_lock = std::unique_lock(gate_->mutex);
  return cancel_under_gate();
}
bool preparation_unit::cancel_under_gate()
{
  std::lock_guard lock(mutex_);
  if (record_.state != unit_state::pending) return false;
  record_.state = unit_state::cancelled;
  deps_         = {};
  if (gate_) gate_->cv.notify_all();
  return true;
}
unit_record preparation_unit::record() const
{
  std::lock_guard lock(mutex_);
  return record_;
}
required_input_set required_inputs(op::scan::later_check_set const& checks, bool needs_delete_set)
{
  using op::scan::later_check;
  auto has = [&](later_check check) { return checks[static_cast<size_t>(check)]; };
  required_input_set result;
  result.set(static_cast<size_t>(required_input::footer),
             has(later_check::footer_per_file) || has(later_check::profile_per_file) ||
               has(later_check::schema_per_file));
  result.set(static_cast<size_t>(required_input::segments),
             has(later_check::segments_per_range) || has(later_check::matrix_per_range));
  result.set(static_cast<size_t>(required_input::delete_set), needs_delete_set);
  result.set(static_cast<size_t>(required_input::checkpoint_iteration), has(later_check::key_held));
  return result;
}
}  // namespace sirius::scan_manager
