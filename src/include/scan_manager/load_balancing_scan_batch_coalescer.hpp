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
#include "op/scan/batch_coalescer.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/sirius_gpu_scan_operator.hpp"
#include "scan_manager/balancing_strategy.hpp"
#include "scan_manager/mvcc_chunk_mask.hpp"
#include "scan_manager/split_connector.hpp"

#include <cudf/io/text/byte_range_info.hpp>

#include <concurrentqueue.h>

#include <cstddef>
#include <memory>
#include <stop_token>

namespace sirius::scan_manager {

struct databatch_provider {
  /// One cached chunk plus its (optional) per-query MVCC keep-mask.
  /// A default-constructed batch (null @ref data) means end-of-stream.
  struct batch {
    std::shared_ptr<cucascade::data_batch> data;
    mvcc_chunk_mask mvcc_keep_mask;  ///< default = all rows visible
  };

  virtual ~databatch_provider()  = default;
  virtual batch get_next_batch() = 0;
};

/**
 * @brief Pipeline-ordered sequencer for @c fadvise(opportunistic) calls.
 *
 * Each pipeline registers a per-pipeline slot (a @c metadata_processing_state)
 * holding a blocking queue its split provider feeds and the coalescer,
 * balancer and connector used to place and forward the resulting splits.  A
 * single sequencer task drains the slots in the order they were registered,
 * coalescing each pipeline's splits, choosing a device, issuing the
 * @c fadvise(opportunistic) / prefetch hints and pushing onto the connector,
 * advancing to the next slot only after the current one signals closure.  This
 * serialises the opportunistic fadvise tier across pipelines so the prefetching
 * cache receives ranges in execution order rather than in
 * metadata-scan-completion order — giving the cache its longest possible lead
 * time for the head-of-line pipeline before later pipelines start competing for
 * the buffer pool.
 *
 * Usage:
 *   - scan_manager calls @c register_pipeline(scan_op, balancer) once per
 *     pipeline that needs opportunistic prefetching; the returned slot pointer
 *     drives that pipeline's splits.  Its split provider is wired up via
 *     @c get_split_provider_bridge (live metadata scan) or
 *     @c use_cached_entries_for_pipeline (cached-batch replay).
 *   - scan_manager calls @c spawn_workers(dispatcher) once, after all slots
 *     have been registered, to launch the sequencer task.  The task processes
 *     slots in registration order until either all slots are drained or the
 *     stop_token fires.
 */
class load_balancing_scan_batch_coalescer {
 public:
  /// Per-pipeline mailbox.  The provider produces, the sequencer task
  /// consumes.  Holds its own semaphore so the sequencer can block on
  /// an empty slot without spinning.
  struct metadata_processing_state {
    explicit metadata_processing_state(std::size_t op_id,
                                       std::size_t pipeline_id,
                                       std::shared_ptr<op::scan::batch_coalescer> coalescer,
                                       std::shared_ptr<split_connector> connector,
                                       std::shared_ptr<balancing_strategy> balancer)
      : op_id(op_id),
        pipeline_id(pipeline_id),
        coalescer(std::move(coalescer)),
        balancer(std::move(balancer)),
        connector(std::move(connector))
    {
      assert(this->coalescer);
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
    std::shared_ptr<op::scan::batch_coalescer> coalescer;
    std::shared_ptr<balancing_strategy> balancer;
    std::shared_ptr<split_connector> connector;
    std::unique_ptr<databatch_provider> batch_provider;
  };

  load_balancing_scan_batch_coalescer()                                           = default;
  load_balancing_scan_batch_coalescer(load_balancing_scan_batch_coalescer const&) = delete;
  load_balancing_scan_batch_coalescer& operator=(load_balancing_scan_batch_coalescer const&) =
    delete;

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

  /// Drain @p provider into @p connector: pull batches until the provider is
  /// exhausted (or @p stop fires), wrapping each as a resident
  /// scan_operator_input. Always closes the connector on exit — with the
  /// pending exception when the provider throws, so the consumer's
  /// get_next_split() rethrows a clean query error instead of blocking
  /// forever (the dispatcher swallows task exceptions, so an unclosed
  /// connector is a silent query hang). Static and public so the drain
  /// behavior is unit-testable against a fake provider; production use is
  /// the sequencer's cached-slot path.
  static void drain_cached_provider(databatch_provider& provider,
                                    split_connector& connector,
                                    std::stop_token const& stop);

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
  void worker_loop(std::stop_token const& stop);

  void process_provider_inputs(metadata_processing_state& state, std::stop_token const& stop);

  void process_cached_entries(metadata_processing_state& state, std::stop_token const& stop);

  /// shared_ptr storage: the slot contains a semaphore and a moodycamel
  /// queue, both of which are non-movable, so slots need stable addresses —
  /// and the split-provider bridge lambda shares ownership of its slot.
  std::vector<std::size_t> _pipeline_order;
  std::unordered_map<std::size_t, std::shared_ptr<metadata_processing_state>> _slots;
};

}  // namespace sirius::scan_manager
