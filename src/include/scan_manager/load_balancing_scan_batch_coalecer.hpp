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

#include "blockingconcurrentqueue.h"
#include "cucascade/data/data_batch.hpp"
#include "exec/try.hpp"
#include "op/scan/batch_coalecer.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "scan_manager/balancing_strategy.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <concurrentqueue.h>

#include <chrono>
#include <cstddef>
#include <memory>
#include <stop_token>

namespace sirius::scan_manager {

struct databatch_provider {
  virtual ~databatch_provider()                                   = default;
  virtual std::shared_ptr<cucascade::data_batch> get_next_batch() = 0;
};

/**
 * @brief Pipeline-ordered sequencer for @c fadvise(opportunistic) calls.
 *
 * Per-pipeline slots collect (datasource, ranges) pairs emitted by each
 * pipeline's split provider during metadata scan.  A single sequencer
 * task drains the slots in the order they were registered, calling
 * @c fadvise(opportunistic, ranges) on each entry's datasource, and
 * advances to the next slot only after the current slot signals closure
 * (a null-datasource sentinel).  This serialises the opportunistic
 * fadvise tier across pipelines so the prefetching cache receives ranges
 * in execution order rather than in metadata-scan-completion order —
 * giving the cache its longest possible lead time for the head-of-line
 * pipeline before later pipelines start competing for the buffer pool.
 *
 * Usage:
 *   - scan_manager calls @c add_pipeline_slot(pipeline_id) once per
 *     parquet pipeline that needs opportunistic prefetching.  The
 *     returned pointer is handed to that pipeline's split provider.
 *   - The provider pushes one @c fadvise_entry per file via the slot's
 *     queue + semaphore; it pushes a sentinel (null datasource) when
 *     all batches of its metadata scan are complete.
 *   - scan_manager calls @c register_ranges(stop, dispatcher) once, which
 *     enqueues the sequencer task on the dispatcher.  The task processes
 *     slots in insertion order until either all slots are drained or the
 *     stop_token fires.
 */
class load_balancing_scan_batch_coalecer {
 public:
  /// Per-pipeline mailbox.  The provider produces, the sequencer task
  /// consumes.  Holds its own semaphore so the sequencer can block on
  /// an empty slot without spinning.
  struct metadata_processing_state {
    explicit metadata_processing_state(std::size_t op_id,
                                       std::size_t pipeline_id,
                                       std::shared_ptr<op::scan::batch_coalecer> coalecer,
                                       std::shared_ptr<split_connector> connector,
                                       std::shared_ptr<balancing_strategy> balancer)
      : op_id(op_id),
        pipeline_id(pipeline_id),
        coalecer(std::move(coalecer)),
        connector(std::move(connector)),
        balancer(std::move(balancer))
    {
      assert(this->coalecer);
      assert(this->connector);
      assert(this->balancer);
    }

    void attach_batch_provider(std::unique_ptr<databatch_provider> provider)
    {
      batch_provider = std::move(provider);
    }

    std::size_t op_id{0};
    std::size_t pipeline_id{0};
    using provider_value_t = exec::try_t<std::unique_ptr<op::scan::scan_info>>;
    duckdb_moodycamel::BlockingConcurrentQueue<provider_value_t> queue;
    std::shared_ptr<op::scan::batch_coalecer> coalecer;
    std::shared_ptr<balancing_strategy> balancer;
    std::shared_ptr<split_connector> connector;
    std::unique_ptr<databatch_provider> batch_provider;
  };

  load_balancing_scan_batch_coalecer()                                          = default;
  load_balancing_scan_batch_coalecer(load_balancing_scan_batch_coalecer const&) = delete;
  load_balancing_scan_batch_coalecer& operator=(load_balancing_scan_batch_coalecer const&) = delete;

  /// Register a slot for @p pipeline_id.  Slots are processed by the
  /// sequencer task in the order they were added — typically scan_manager
  /// adds them in pipeline-id order so the head-of-line pipeline drains
  /// first.  The returned pointer is valid for the manager's lifetime.
  metadata_processing_state* register_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                               std::shared_ptr<balancing_strategy> balancer);

  void use_cached_entries_for_pipeline(op::scan::sirius_gpu_scan_operator* scan_op,
                                       std::unique_ptr<databatch_provider> provider);

  std::function<void(exec::try_t<std::unique_ptr<op::scan::scan_info>>&&)>
  get_split_provider_bridge(op::scan::sirius_gpu_scan_operator* scan_op);

  /// Spawn the sequencer task on @p dispatcher.  The dispatcher must
  /// expose @c enqueue(callable) and inject a @c std::stop_token when
  /// the callable asks for one (e.g. @c scoped_dispatcher).  Call once
  /// after all slots have been added — the task captures the slot list
  /// by reference, so adding slots after this call has undefined
  /// ordering with respect to the sequencer walk.
  template <class Dispatcher>
  void spawn_workers(Dispatcher& dispatcher)
  {
    dispatcher.enqueue([this](std::stop_token const& stop) { worker_loop(stop); });
  }

 private:
  /// The sequencer body.  Walks _slots in order; for each, drains
  /// entries (semaphore-blocked with a poll timeout so stop_token is
  /// observed promptly) until it hits a closure sentinel (null
  /// datasource) or stop is requested.
  void worker_loop(std::stop_token const& stop);

  void process_provider_inputs(metadata_processing_state& state, std::stop_token const& stop);

  void process_cached_entries(metadata_processing_state& state, std::stop_token const& stop);

  static constexpr auto SEQUENCER_POLL_INTERVAL = std::chrono::milliseconds(50);

  /// unique_ptr storage: the slot contains a semaphore and a moodycamel
  /// queue, both of which are non-movable, so we need stable addresses
  /// in the vector.
  std::vector<std::size_t> _pipeline_order;
  std::unordered_map<std::size_t, std::shared_ptr<metadata_processing_state>> _slots;
};

}  // namespace sirius::scan_manager
