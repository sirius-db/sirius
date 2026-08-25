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

#include "exec/queue_priority.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "query_id.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <variant>

namespace sirius::event {

/// Discriminator for @ref event.  One entry per @c publish_* entry
/// point on @ref query_event_publisher; the enumerator is what the dispatch in
/// @ref query_event_subscriber switches on, so the two must stay in step.
enum class event_type : std::uint8_t {
  task_created,
  task_deployed,
  failed_to_create_task,
  task_queue_empty,
  pipeline_closed,
  executor_awaiting_task,
  memory_downgrade_for_task,
  wait_for_memory_for_task
};

/// One published event: the tag that says which it is, and the arguments the
/// reporter passed, kept together so the event can be queued and replayed on
/// another thread exactly as it was raised.
///
/// The tag is a template parameter rather than a member so every event is a
/// distinct type, which is what lets @ref query_events be a @c std::variant
/// and the dispatch be a compile-time @c if @c constexpr chain rather than a
/// switch that could silently fall through.
using event_id_t  = std::size_t;
using timestamp_t = std::chrono::system_clock::time_point;

template <event_type EventType, typename... Types>
struct event {
  using param_type = std::tuple<Types...>;

  static constexpr event_type type = EventType;

  event_id_t event_id{};
  timestamp_t timestamp{};
  param_type data;
};

using task_created_event = event<event_type::task_created,
                                 query_id_t,
                                 std::size_t,
                                 op::SiriusPhysicalOperatorType,
                                 exec::queue_priority>;

using task_deployed_event =
  event<event_type::task_deployed, query_id_t, std::size_t, op::SiriusPhysicalOperatorType, int>;

using failed_to_create_task_event =
  event<event_type::failed_to_create_task, query_id_t, std::size_t, std::size_t>;

using task_queue_empty_event = event<event_type::task_queue_empty>;

using pipeline_closed_event =
  event<event_type::pipeline_closed, query_id_t, std::size_t, std::size_t>;

using executor_awaiting_task_event = event<event_type::executor_awaiting_task, int>;

using memory_downgrade_for_task_event =
  event<event_type::memory_downgrade_for_task, query_id_t, std::size_t, int, std::size_t>;

using wait_for_memory_for_task_event =
  event<event_type::wait_for_memory_for_task, query_id_t, std::size_t, int, std::size_t>;

/// Every event a @ref query_event_publisher can publish.  Subscribers receive this
/// and nothing else, so adding an event is: an enumerator, an alias, an arm
/// here, and an arm in the dispatch.
using query_events = std::variant<task_created_event,
                                  task_deployed_event,
                                  failed_to_create_task_event,
                                  task_queue_empty_event,
                                  pipeline_closed_event,
                                  executor_awaiting_task_event,
                                  memory_downgrade_for_task_event,
                                  wait_for_memory_for_task_event>;

/// Number of distinct events, and so the width of the publisher's routing table.
inline constexpr std::size_t n_query_events = std::variant_size_v<query_events>;

/// An event's slot in that table. The variant arms are declared in enumerator
/// order, which is what lets the tag double as the index -- reorder one list
/// without the other and every event routes to the wrong subscribers.
template <typename Event>
inline constexpr std::size_t event_index_v = static_cast<std::size_t>(Event::type);

static_assert(
  std::is_same_v<std::variant_alternative_t<event_index_v<task_created_event>, query_events>,
                 task_created_event>);
static_assert(
  std::is_same_v<
    std::variant_alternative_t<event_index_v<wait_for_memory_for_task_event>, query_events>,
    wait_for_memory_for_task_event>);

/// Every event, for the @ref query_event_publisher::register_subscriber overload
/// that takes no subscription list.
inline constexpr std::array<event_type, n_query_events> all_query_events{
  event_type::task_created,
  event_type::task_deployed,
  event_type::failed_to_create_task,
  event_type::task_queue_empty,
  event_type::pipeline_closed,
  event_type::executor_awaiting_task,
  event_type::memory_downgrade_for_task,
  event_type::wait_for_memory_for_task};

}  // namespace sirius::event
