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

// sirius
#include <op/scan/coalescing_carrier.hpp>
#include <op/scan/coalescing_gpu_ingestible.hpp>
#include <op/sirius_physical_operator.hpp>  // op::operator_data device-id accessors
#include <scan_manager/balancing_strategy.hpp>
#include <scan_manager/split_connector.hpp>

// standard library
#include <stdexcept>
#include <utility>

namespace sirius::op::scan {

std::unique_ptr<op::operator_data> coalescing_gpu_ingestible::consume_next_input(
  scan_manager::split_connector& connector)
{
  // Coalesce parsed ranges into cap-sized batches as they arrive, so early batches decode while
  // later ranges are still being walked. get_next_split() blocks until a carrier is ready.
  for (;;) {
    if (_coalescer->has_ready()) { return emit_batch(_coalescer->pop_ready()); }

    auto next = connector.get_next_split();
    if (!next.has_value()) {
      // Connector closed and drained: emit the single tail batch, if any.
      auto tail = _coalescer->flush();
      if (!tail.empty()) { return emit_batch(std::move(tail)); }
      return nullptr;
    }

    auto* carrier = dynamic_cast<coalescing_carrier*>(next->get());
    if (carrier == nullptr) {
      throw std::runtime_error(
        "[coalescing_gpu_ingestible::consume_next_input] unexpected operator_data type from "
        "split connector; expected coalescing_carrier.");
    }
    for (auto& unit : carrier->units) {
      _coalescer->push(std::move(unit));
    }
  }
}

bool coalescing_gpu_ingestible::consumer_drained() const
{
  // The scan operator conjoins this with split_connector::is_closed(). Gating on an empty coalescer
  // keeps a single-split scan (small table, or chunk >= row-group count) from dropping its
  // queued/tail batches when the connector closes after the first pull.
  return _coalescer == nullptr || _coalescer->empty();
}

std::unique_ptr<op::operator_data> coalescing_gpu_ingestible::emit_batch(
  std::vector<coalescing_unit> units)
{
  auto batch = make_batch(std::move(units));

  // Assign the batch's GPU per the balancing strategy, at the one site a real decode batch exists.
  // The task creator reads this back to route the batch's task. Skip resident or already-placed
  // batches (matching split_provider::apply_balancing's guards); a coalesced batch is neither.
  if (batch != nullptr && _balancing_strategy != nullptr && !batch->is_resident() &&
      !batch->get_preferred_device_id().has_value()) {
    auto const device_id = _balancing_strategy->get_next_gpu(_pipeline_id, batch.get());
    if (device_id >= 0) { batch->set_preferred_device_id(device_id); }
  }
  return batch;
}

}  // namespace sirius::op::scan
