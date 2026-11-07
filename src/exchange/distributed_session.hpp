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

#include "exchange/host_staging.hpp"

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius {
struct operator_params;
}

namespace sirius::exchange {

class exchange_service;

/**
 * @brief One process's part in distributed execution: its exchange, the numbering of its
 * distributed queries, and which execution window runs which of them.
 *
 * SiriusContext creates it at the first distributed use and keeps it until it is destroyed; a
 * process that never runs a distributed query has none, so its queries pay nothing for it.
 * Thread-safe. No call holds the internal lock while it waits on the exchange.
 */
class distributed_session {
 public:
  /// The settings the exchange connected with; they cannot change afterwards. A setting added
  /// here and in from() is one that service() refuses a change to.
  struct settings {
    std::vector<std::string> peers;
    std::uint32_t rank = 0;
    exchange::host_staging host_staging{};
    std::string mnnvl;
    std::uint64_t frame_bytes              = 0;
    std::uint32_t send_frames              = 0;
    std::uint32_t recv_frames_per_peer     = 0;
    bool operator==(settings const&) const = default;

    /// The exchange settings of @p params.
    [[nodiscard]] static settings from(operator_params const& params);
  };

  distributed_session()                                      = default;
  distributed_session(distributed_session const&)            = delete;
  distributed_session& operator=(distributed_session const&) = delete;

  /**
   * @brief The exchange for @p wanted, connected by @p connect on first use.
   *
   * Serialized, so concurrent first uses connect once. Does not hold the lock the per-query
   * calls take while connecting, which can wait for every peer.
   * @throws sirius::invalid_input_exception when the settings changed after it connected.
   */
  [[nodiscard]] std::shared_ptr<exchange_service> service(
    settings const& wanted, std::function<std::shared_ptr<exchange_service>()> const& connect);

  /// The connected exchange, or null.
  [[nodiscard]] std::shared_ptr<exchange_service> service() const;

  /// Test hook: @p service becomes the exchange, as if connected with no settings; null drops it.
  void set_service_for_testing(std::shared_ptr<exchange_service> service);

  /**
   * @brief Numbers a distributed query of connection @p connection, from 1. Every rank runs the
   * same statements, so the same query gets the same number, and its exchanges the same stream
   * ids. That holds for one statement stream only, so one connection per process runs them.
   * @param assigned Nonzero: the number to use, chosen by whoever sends every rank its
   * statements. The numbers skipped are retired at once, or when the exchange connects.
   * @throws sirius::invalid_input_exception when another connection already runs them, or when
   * @p assigned does not exceed the last number used.
   */
  [[nodiscard]] std::uint64_t next_query(std::uint64_t connection, std::uint64_t assigned = 0);

  /// Connection @p connection is closing: the next to number a query takes over, and the
  /// numbering continues.
  void release_connection(std::uint64_t connection);

  /// Execution window @p window runs distributed query @p query.
  void tie(std::uint64_t window, std::uint64_t query);

  /// The distributed query window @p window runs, or 0.
  [[nodiscard]] std::uint64_t query_of(std::uint64_t window) const;

  /// Window @p window has ended.
  void forget_window(std::uint64_t window);

  /// Window @p window's query failed with @p error: every peer's run of it fails too.
  void abort(std::uint64_t window, std::exception_ptr const& error) noexcept;

  /// Window @p window failed to begin with @p error: aborts its query on every rank, retires it
  /// here and unties the window. The destructor of a scope that failed to begin does not run, so
  /// nothing else would.
  void fail_window_begin(std::uint64_t window, std::exception_ptr const& error) noexcept;

  /// Window @p window's query retires: waits (up to a minute) until the exchange holds none of
  /// the batches it sent, then forgets the query's exchange state.
  void retire(std::uint64_t window);

  /// Stops the exchange; later queries find none. For SiriusContext::terminate().
  void shutdown();

 private:
  std::mutex _connect_mutex;
  mutable std::mutex _mutex;
  std::shared_ptr<exchange_service> _service;
  std::optional<settings> _settings;
  std::uint64_t _queries = 0;
  /// Numbers skipped before the exchange connected (first, last), retired once it does.
  std::vector<std::pair<std::uint64_t, std::uint64_t>> _skipped_before_connect;
  /// The connection that runs this process's distributed queries.
  std::optional<std::uint64_t> _connection;
  /// Each open execution window's distributed query. A window may retire while another
  /// connection's window runs, so it names its own.
  std::unordered_map<std::uint64_t, std::uint64_t> _by_window;
};

}  // namespace sirius::exchange
