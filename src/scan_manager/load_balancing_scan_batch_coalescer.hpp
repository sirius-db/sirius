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

#include "cucascade/data/data_batch.hpp"
#include "exec/try.hpp"
#include "late_mat/column_origin.hpp"
#include "op/scan/batch_coalescer.hpp"
#include "op/scan/dynamic_filter_merge.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "scan_manager/balancing_strategy.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"
#include "scan_manager/preparation_coordinator.hpp"
#include "scan_manager/readahead_scan_manager.hpp"
#include "scan_manager/split_connector.hpp"
#include "scan_manager/split_provider.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <cstddef>
#include <memory>
#include <stop_token>

namespace sirius::scan_manager {

struct databatch_provider {
  op::scan::pin_validation validation;
  op::scan::scan_contract_id contract_id = 0;
  /// One cached chunk or one insert-delta split, plus its (optional)
  /// per-query MVCC keep-mask. A batch carries either @ref data (resident
  /// cached chunk) or @ref scan_info (delta split, yielded after the
  /// resident chunks); both null means end-of-stream.
  struct batch {
    std::shared_ptr<cucascade::data_batch> data;
    mvcc_chunk_mask mvcc_keep_mask;  ///< default = all rows visible
    /// True when scan normalization will cast at least one selected column of this cached chunk:
    /// its recorded carrier differs from the carrier the scan plans for that column.
    bool needs_carrier_conversion{false};
    /// Exact bytes of the destination columns (data plus validity mask) those casts allocate.
    /// Zero when nothing converts or when a converting chunk's row count is unreadable; the scan
    /// memory estimate then falls back to its conservative maximum-expansion bound.
    std::size_t conversion_destination_bytes{0};
    std::unique_ptr<op::scan::scan_info> scan_info;
    int preferred_device{-1};  ///< placement for scan_info splits (-1 = none)
    /// Where this batch's rows came from, for late materialization: the pinned
    /// entry and the contiguous span of pin-order rows this chunk covers. Null
    /// unless the late-mat gate is on and the scan is one whose batches are a
    /// chunk each — the addressing means nothing otherwise.
    std::shared_ptr<late_mat::scan_batch_origin const> origin;
    /// The identities of dynamic filters attached to this batch's decoder configuration.
    /// Transferred by the coalescer into the scan_operator_input.
    std::vector<op::scan::applied_entry> decode_attached;
  };

  virtual ~databatch_provider()  = default;
  virtual batch get_next_batch() = 0;
};

// Registers scan outputs with shared placement and readahead state. The coordinator
// prepares live inputs; cached providers drain on workers with the same publication gate.
class load_balancing_scan_batch_coalescer {
 public:
  /// Per-scan publication and cached-input state.
  struct metadata_processing_state {
    explicit metadata_processing_state(std::size_t op_id,
                                       std::size_t pipeline_id,
                                       std::shared_ptr<op::scan::batch_coalescer> coalescer,
                                       std::shared_ptr<split_connector> connector,
                                       std::shared_ptr<balancing_strategy> balancer,
                                       std::shared_ptr<readahead_scan_manager> readahead = nullptr)
      : op_id(op_id),
        pipeline_id(pipeline_id),
        coalescer(std::move(coalescer)),
        balancer(std::move(balancer)),
        connector(std::move(connector)),
        readahead(std::move(readahead))
    {
      assert(this->coalescer);
      assert(this->connector);
      assert(this->balancer);
    }

    void attach_batch_provider(std::unique_ptr<databatch_provider> provider)
    {
      batch_provider = std::move(provider);
    }

    std::shared_ptr<preparation_gate> publication;
    std::size_t op_id{0};
    std::size_t pipeline_id{0};
    std::shared_ptr<op::scan::batch_coalescer> coalescer;
    std::shared_ptr<balancing_strategy> balancer;
    std::shared_ptr<split_connector> connector;
    std::unique_ptr<databatch_provider> batch_provider;
    /// Per-query readahead bookkeeping this slot's splits report into; null
    /// when the caller does not track readahead (e.g. unit tests).
    std::shared_ptr<readahead_scan_manager> readahead;
    /// Whether the op's ingestible applies a row-filter expression to cached
    /// batches; stamped onto each drained split so its working-set estimate
    /// covers the filter-by-copy peak.
    bool row_filter_pending{false};
    op::scan::sirius_gpu_scan_operator* scan_op = nullptr;
  };

  load_balancing_scan_batch_coalescer()                                           = default;
  load_balancing_scan_batch_coalescer(load_balancing_scan_batch_coalescer const&) = delete;
  load_balancing_scan_batch_coalescer& operator=(load_balancing_scan_batch_coalescer const&) =
    delete;

  /// Register per-scan publication state. The returned slot remains valid
  /// until this coalescer is destroyed.
  metadata_processing_state* register_pipeline(
    op::scan::sirius_gpu_scan_operator* scan_op,
    std::shared_ptr<balancing_strategy> balancer,
    std::shared_ptr<readahead_scan_manager> readahead = nullptr);

  void register_coordinator_source(op::scan::sirius_gpu_scan_operator*,
                                   split_provider&,
                                   preparation_coordinator&);
  // Legacy cached providers may block: keep their executor and count their owner window.
  std::vector<exec::invocable<void(std::stop_token)>> prepare_cached_workers(
    preparation_coordinator& coordinator)
  {
    std::vector<exec::invocable<void(std::stop_token)>> work;
    for (auto id : _pipeline_order) {
      auto state = _slots.at(id);
      if (!state->batch_provider) continue;
      if (state->publication) throw std::logic_error("cached slot already has a coordinator owner");
      state->publication = coordinator.publication_gate();
      auto use           = coordinator.external_use();
      if (!use) continue;
      work.emplace_back(
        [this, id, use = std::move(use)](std::stop_token stop) { slot_loop(id, stop); });
    }
    return work;
  }
  void close_all_slots() noexcept
  {
    for (auto const& [id, state] : _slots) {
      try {
        if (state->readahead) state->readahead->mark_operator_closed(state->op_id);
      } catch (...) {
      }
      state->connector->close();
    }
  }
  void use_cached_entries_for_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                       std::unique_ptr<databatch_provider> provider);

  /// Drain @p provider into @p connector: pull batches until the provider is
  /// exhausted (or @p stop fires), wrapping each as a resident
  /// scan_operator_input with @p row_filter_pending stamped on. Always closes
  /// the connector on exit — with the pending exception when the provider
  /// throws, so the consumer's get_next_split() rethrows a clean query error
  /// instead of blocking forever (the dispatcher swallows task exceptions, so
  /// an unclosed connector is a silent query hang). Static and public so the
  /// drain behavior is unit-testable against a fake provider; production use
  /// is the sequencer's cached-slot path.
  static void drain_cached_provider(databatch_provider& provider,
                                    split_connector& connector,
                                    std::stop_token const& stop,
                                    bool row_filter_pending);
  static void drain_cached_provider(databatch_provider& provider,
                                    split_connector& connector,
                                    std::stop_token const& stop,
                                    bool row_filter_pending,
                                    std::shared_ptr<readahead_scan_manager> readahead,
                                    std::size_t operator_id = 0);
  static void drain_cached_provider(databatch_provider& provider,
                                    split_connector& connector,
                                    std::stop_token const& stop,
                                    bool row_filter_pending,
                                    op::scan::scan_contract_id expected,
                                    uint64_t query_token,
                                    duckdb::SiriusContext* observer                   = nullptr,
                                    bool invalidate_witness                           = false,
                                    bool native_pin                                   = false,
                                    std::shared_ptr<readahead_scan_manager> readahead = nullptr,
                                    std::size_t operator_id                           = 0,
                                    std::shared_ptr<preparation_gate> publication     = {});

 private:
  /// Drain one slot to completion.  Runs concurrently with the other slots'
  /// loops, so everything it touches is per-slot state.
  void slot_loop(std::size_t pipeline_id, std::stop_token const& stop);

  void process_cached_entries(metadata_processing_state& state, std::stop_token const& stop);

  /// Publication callbacks retain their slot through source drain.
  std::vector<std::size_t> _pipeline_order;
  std::unordered_map<std::size_t, std::shared_ptr<metadata_processing_state>> _slots;
};

}  // namespace sirius::scan_manager
