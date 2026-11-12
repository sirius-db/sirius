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

#include "op/dynamic_filter/complete_build_inventory.hpp"

#include <cuda/stream>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>

namespace cucascade {
class data_batch;
class read_only_data_batch;
}  // namespace cucascade

namespace sirius::op {

class dynamic_filter_publication_session;
class pipelineable_operator_data;
class sirius_physical_operator;

/**
 * @brief Runs one dynamic-filter construction site's side of its join's
 * `dynamic_filter_publication_session`, so that the build PARTITION (through
 * `hash_join_partition_observer`) and `sirius_physical_build_filter` follow one sequence.
 *
 * The site forwards three events. Each batch pushed into its FULL `default` port is recorded in a
 * `build_arrival_ledger` (`on_arrival`). Before each claim of a task input, `decide` reads the
 * site's ports (`read_build_port_facts`), applies `certify_decision`, and certifies the ledger; it
 * then either declines the session (`dynamic_filter_publication_session::decline`) or begins it
 * with the `complete_build_inventory` (`dynamic_filter_publication_session::begin`). The decision
 * is made once and is final. Each task input then contributes its one certified batch
 * (`contribute`), and the site publishes after building its output
 * (`dynamic_filter_publication_session::publish_if_final`).
 *
 * Nothing happens while the session's plan is disabled. `decide` runs on the task creator's thread
 * under an internal mutex; `on_arrival` and `contribute` may run concurrently with it and with each
 * other.
 */
class build_input_certifier final {
 public:
  /**
   * @param site The construction site whose input ports are read; it outlives this certifier
   * @param session Its join's session; it outlives this certifier
   */
  build_input_certifier(sirius_physical_operator const& site,
                        dynamic_filter_publication_session& session) noexcept;

  build_input_certifier(build_input_certifier const&)            = delete;
  build_input_certifier& operator=(build_input_certifier const&) = delete;

  /**
   * @brief Records @p batch, pushed into @p port_id, when that port is the site's FULL `default`
   * port.
   */
  void on_arrival(std::string_view port_id, cucascade::data_batch& batch);

  /**
   * @brief Decides, once, whether the site constructs its join's dynamic filters.
   *
   * Logs the reason for a decline. Only the ledger of a session that began keeps its batches.
   *
   * @return false while the site's FULL input is still produced, so the site must not claim a task
   * input yet; true once decided
   */
  [[nodiscard]] bool decide();

  /**
   * @brief Contributes @p batch, the batch of task input @p input, to the attempt `decide` began.
   *
   * A task input that is not exactly one original batch, or a site with a late-materialization
   * directive, ends the attempt instead (`dynamic_filter_publication_session::decline_task_input`).
   *
   * @return The original ID for which the task calls
   * `dynamic_filter_publication_session::publish_if_final` after building its output; empty when
   * nothing was contributed
   */
  [[nodiscard]] std::optional<std::uint64_t> contribute(
    pipelineable_operator_data const& input,
    cucascade::read_only_data_batch const& batch,
    ::cuda::stream_ref stream);

 private:
  sirius_physical_operator const& _site;
  dynamic_filter_publication_session& _session;
  build_arrival_ledger _arrival_ledger;
  std::mutex _decision_mutex;
  bool _decided = false;
  std::atomic<bool> _began{false};  ///< Whether `decide` began the session's attempt
};

}  // namespace sirius::op
