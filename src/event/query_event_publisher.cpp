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

#include "event/query_event_publisher.hpp"

#include <algorithm>
#include <ranges>
#include <tuple>
#include <utility>

namespace sirius::event {

// ---------------------------------------------------------------------------
// registration (private; only the subscriber base ever calls these)
// ---------------------------------------------------------------------------

subscriber_registration query_event_publisher::register_subscriber(
  std::span<event_type const> events)
{
  auto queue = std::make_shared<event_queue>();
  {
    std::unique_lock g{_queues_mtx};
    if (!_stopped) {
      _queues.push_back(queue);
      for (auto e : events) {
        auto& bucket = _by_event[static_cast<std::size_t>(e)];
        // A repeated event would double-deliver, and a caller assembling the
        // list from overlapping sets should not have to care.
        if (std::ranges::find(bucket, queue.get()) == bucket.end()) {
          bucket.push_back(queue.get());
        }
      }
    }
  }
  return subscriber_registration{std::move(queue), _stop_source.get_token()};
}

void query_event_publisher::unregister_subscriber(
  std::shared_ptr<event_queue> const& queue) noexcept
{
  std::unique_lock g{_queues_mtx};
  std::erase_if(_queues, [&queue](auto const& q) { return q == queue; });
  // The buckets hold raw pointers into what _queues owns, so they have to be
  // cleared in the same critical section: a pointer left behind would outlive
  // the queue it names.
  auto* raw = queue.get();
  for (auto& bucket : _by_event) {
    std::erase(bucket, raw);
  }
}

void query_event_publisher::stop() noexcept
{
  _stop_source.request_stop();
  std::vector<std::shared_ptr<event_queue>> queues;
  {
    std::unique_lock g{_queues_mtx};
    _stopped = true;
    queues.swap(_queues);
    for (auto& bucket : _by_event) {
      bucket.clear();
    }
  }
  for (auto const& q : queues) {
    // try_ so this stays genuinely noexcept: the allocating enqueue could
    // throw, and a subscriber that misses the sentinel still comes out on the
    // stop token within a poll interval.  The sentinel only buys promptness.
    std::ignore = q->try_enqueue(nullptr);
  }
}

// ---------------------------------------------------------------------------
// reporting
// ---------------------------------------------------------------------------

void query_event_publisher::publish_task_created(query_id_t query_id,
                                                 std::size_t operator_id,
                                                 op::SiriusPhysicalOperatorType operator_type,
                                                 exec::queue_priority priority) noexcept
{
  publish<task_created_event>(query_id, operator_id, operator_type, priority);
}

void query_event_publisher::publish_task_deployed(query_id_t query_id,
                                                  std::size_t operator_id,
                                                  op::SiriusPhysicalOperatorType operator_type,
                                                  int gpu_id) noexcept
{
  publish<task_deployed_event>(query_id, operator_id, operator_type, gpu_id);
}

void query_event_publisher::publish_failed_to_create_task(query_id_t query_id,
                                                          std::size_t source_operator_id,
                                                          std::size_t failed_operator_id) noexcept
{
  publish<failed_to_create_task_event>(query_id, source_operator_id, failed_operator_id);
}

void query_event_publisher::publish_task_queue_empty() noexcept
{
  publish<task_queue_empty_event>();
}

void query_event_publisher::publish_pipeline_closed(query_id_t query_id,
                                                    std::size_t pipeline_id,
                                                    std::size_t source_operator_id) noexcept
{
  publish<pipeline_closed_event>(query_id, pipeline_id, source_operator_id);
}

void query_event_publisher::publish_executor_awaiting_task(int gpu_id) noexcept
{
  publish<executor_awaiting_task_event>(gpu_id);
}

void query_event_publisher::publish_memory_downgrade_for_task(query_id_t query_id,
                                                              std::size_t operator_id,
                                                              int gpu_id,
                                                              std::size_t shortfall_bytes) noexcept
{
  publish<memory_downgrade_for_task_event>(query_id, operator_id, gpu_id, shortfall_bytes);
}

void query_event_publisher::publish_wait_for_memory_for_task(query_id_t query_id,
                                                             std::size_t operator_id,
                                                             int gpu_id,
                                                             std::size_t bytes_needed) noexcept
{
  publish<wait_for_memory_for_task_event>(query_id, operator_id, gpu_id, bytes_needed);
}

}  // namespace sirius::event
