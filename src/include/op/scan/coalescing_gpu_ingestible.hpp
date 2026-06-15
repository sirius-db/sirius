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

// sirius
#include <io/gpu_ingestible.hpp>
#include <op/scan/batch_coalescer.hpp>
#include <op/scan/coalescing_unit.hpp>

// standard library
#include <cstddef>
#include <memory>
#include <vector>

namespace sirius::scan_manager {
class balancing_strategy;
}  // namespace sirius::scan_manager

namespace sirius::op::scan {

/**
 * @brief Base for scans that build decode batches on the consumer side.
 *
 * Producers run in parallel and emit @c coalescing_carrier objects — each a bag of
 * format-blind @c coalescing_unit work items. The single consumer pulls carriers from the
 * connector, feeds their units to a @c batch_coalescer that packs them into cap-sized batches
 * (never crossing a @c group_key boundary), and calls @c make_batch to turn each packed set into
 * a real decode batch — assigning its GPU there, the one place a decode batch exists. A concrete
 * format supplies the producer and @c make_batch; the coalescing, batching, and GPU assignment
 * are shared here.
 */
class coalescing_gpu_ingestible : public io::gpu_ingestible {
 public:
  explicit coalescing_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info)
    : io::gpu_ingestible(std::move(info))
  {
  }

  /**
   * @brief Pull coalescing_carriers, pack their units, and serve cap-sized batches (one tail).
   *
   * Returns nullptr when the connector is closed and drained.
   */
  std::unique_ptr<op::operator_data> consume_next_input(
    scan_manager::split_connector& connector) final;

  /**
   * @brief False while the coalescer still holds unserved units.
   */
  [[nodiscard]] bool consumer_drained() const final;

  /**
   * @brief Install the GPU balancing strategy so each emitted batch is round-robined across GPUs.
   *
   * Called once at wiring time for coalescing scans. Because each coalesced batch is stamped here
   * (on the consumer side), the producer must NOT also balance the intermediate carriers, which are
   * discarded — otherwise the round-robin cursor advances twice per batch.
   */
  void install_balancing(std::shared_ptr<scan_manager::balancing_strategy> strategy,
                         std::size_t pipeline_id)
  {
    _balancing_strategy = std::move(strategy);
    _pipeline_id        = pipeline_id;
  }

 protected:
  /**
   * @brief Build one decode batch from a vector of coalescing units.
   *
   * The format owns the units' opaque payloads end-to-end, so it may safely downcast them here.
   */
  virtual std::unique_ptr<op::operator_data> make_batch(std::vector<coalescing_unit> units) = 0;

  // The coalescer that packs parsed ranges into cap-sized batches (constructed by the concrete
  // format's constructor)
  std::unique_ptr<batch_coalescer> _coalescer;

 private:
  /// @ref make_batch plus the per-batch device-id stamp. This is the one site a real decode batch
  /// exists on the consumer side, so it is the only correct place to assign its GPU.
  std::unique_ptr<op::operator_data> emit_batch(std::vector<coalescing_unit> units);

  std::shared_ptr<scan_manager::balancing_strategy> _balancing_strategy;
  std::size_t _pipeline_id = 0;
};

}  // namespace sirius::op::scan
