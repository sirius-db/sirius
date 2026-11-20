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

#include "exchange/distributed_session.hpp"

#include "exchange/exchange_service.hpp"
#include "log/logging.hpp"
#include "sirius/exception.hpp"
#include "sirius_config.hpp"

#include <chrono>
#include <utility>

namespace sirius::exchange {

distributed_session::settings distributed_session::settings::from(operator_params const& params)
{
  return {.peers                = params.exchange_peers,
          .rank                 = params.exchange_rank,
          .host_staging         = params.exchange_host_staging,
          .mnnvl                = params.exchange_mnnvl,
          .frame_bytes          = params.exchange_frame_bytes,
          .send_frames          = params.exchange_send_frames,
          .recv_frames_per_peer = params.exchange_recv_frames_per_peer,
          .tcp_lanes            = params.exchange_tcp_lanes};
}

std::shared_ptr<exchange_service> distributed_session::service(
  settings const& wanted, std::function<std::shared_ptr<exchange_service>()> const& connect)
{
  std::lock_guard connecting{_connect_mutex};
  {
    std::lock_guard lock{_mutex};
    if (_service) {
      if (_settings && *_settings != wanted) {
        throw sirius::invalid_input_exception("the exchange settings cannot change once connected");
      }
      return _service;
    }
  }
  auto connected = connect();
  std::vector<std::pair<std::uint64_t, std::uint64_t>> skipped;
  {
    std::lock_guard lock{_mutex};
    _service  = connected;
    _settings = wanted;
    skipped   = std::exchange(_skipped_before_connect, {});
  }
  for (auto const& [first, last] : skipped) {
    connected->forget_range(first, last);
  }
  return connected;
}

std::shared_ptr<exchange_service> distributed_session::service() const
{
  std::lock_guard lock{_mutex};
  return _service;
}

void distributed_session::set_service_for_testing(std::shared_ptr<exchange_service> service)
{
  std::vector<std::pair<std::uint64_t, std::uint64_t>> skipped;
  std::shared_ptr<exchange_service> connected;
  {
    std::lock_guard lock{_mutex};
    _service = std::move(service);
    _settings.reset();
    connected = _service;
    if (connected) { skipped = std::exchange(_skipped_before_connect, {}); }
  }
  for (auto const& [first, last] : skipped) {
    connected->forget_range(first, last);
  }
}

std::uint64_t distributed_session::next_query(std::uint64_t connection, std::uint64_t assigned)
{
  std::unique_lock lock{_mutex};
  if (!_connection) { _connection = connection; }
  if (*_connection != connection) {
    throw sirius::invalid_input_exception(
      "distributed execution runs on one connection per process: connection {} runs it, so "
      "connection {} cannot",
      *_connection,
      connection);
  }
  if (assigned != 0 && assigned == _queries) {
    throw sirius::invalid_input_exception(
      "distributed query {} was already planned on this rank, and its plan must be rebuilt; "
      "retry the statement, which numbers it again",
      assigned);
  }
  if (assigned != 0 && assigned < _queries) {
    throw sirius::invalid_input_exception(
      "sirius_distributed_query_id {} must exceed the last distributed query, {}",
      assigned,
      _queries);
  }
  auto const query = assigned != 0 ? assigned : _queries + 1;
  if (query >= distributed_query_limit) {
    throw sirius::invalid_input_exception(
      "distributed query {} is not below {}", query, distributed_query_limit);
  }
  // This rank never hands out the numbers between the last one and this one.
  auto const skipped_first = _queries + 1;
  std::shared_ptr<exchange_service> service;
  if (query > skipped_first) {
    if (_service) {
      service = _service;
    } else {
      _skipped_before_connect.emplace_back(skipped_first, query - 1);
    }
  }
  _queries = query;
  lock.unlock();
  if (service) { service->forget_range(skipped_first, query - 1); }
  return query;
}

void distributed_session::release_connection(std::uint64_t connection)
{
  std::lock_guard lock{_mutex};
  if (_connection == connection) { _connection.reset(); }
}

void distributed_session::tie(std::uint64_t window, std::uint64_t query)
{
  std::lock_guard lock{_mutex};
  _by_window[window] = query;
}

std::uint64_t distributed_session::query_of(std::uint64_t window) const
{
  std::lock_guard lock{_mutex};
  auto const it = _by_window.find(window);
  return it == _by_window.end() ? 0 : it->second;
}

void distributed_session::forget_window(std::uint64_t window)
{
  std::lock_guard lock{_mutex};
  _by_window.erase(window);
}

void distributed_session::abort(std::uint64_t window, std::exception_ptr const& error) noexcept
{
  try {
    auto const query = query_of(window);
    if (query == 0) { return; }
    auto const service = this->service();
    if (!service) { return; }
    std::string what = "unknown error";
    try {
      std::rethrow_exception(error);
    } catch (std::exception const& e) {
      what = e.what();
    } catch (...) {
    }
    SIRIUS_LOG_WARN("Distributed query {} failed; aborting it on every rank: {}", query, what);
    service->abort(query, what);
  } catch (std::exception const& e) {  // best effort: the peers' own timeouts remain
    try {
      SIRIUS_LOG_WARN(
        "Aborting distributed window {} on the other ranks failed: {}", window, e.what());
    } catch (...) {
    }
  } catch (...) {
  }
}

void distributed_session::fail_window_begin(std::uint64_t window,
                                            std::exception_ptr const& error) noexcept
{
  abort(window, error);
  try {
    std::uint64_t query = 0;
    std::shared_ptr<exchange_service> service;
    {
      std::lock_guard lock{_mutex};
      if (auto const it = _by_window.find(window); it != _by_window.end()) {
        query = it->second;
        _by_window.erase(it);
      }
      service = _service;
    }
    // Unlike retire(), no wait for sent batches: the window ran nothing.
    if (query != 0 && service) { service->forget(query); }
  } catch (...) {  // best effort, like abort()
  }
}

void distributed_session::retire(std::uint64_t window)
{
  // The exchange may still hold batches this rank sent, until their last frame is staged; they
  // must be freed before the query's memory is released.
  auto const query = query_of(window);
  if (query == 0) { return; }
  auto const service = this->service();
  if (!service) { return; }
  if (!service->wait_sent(query, std::chrono::minutes{1})) {
    SIRIUS_LOG_WARN("Distributed query {}: sent batches still held by the exchange after 1 min",
                    query);
  }
  service->forget(query);
}

void distributed_session::shutdown()
{
  // Plans can hold the service past this (a prepared statement), so dropping the reference
  // alone would not stop it.
  std::shared_ptr<exchange_service> service;
  {
    std::lock_guard lock{_mutex};
    service = std::move(_service);
  }
  if (service) { service->shutdown(); }
}

}  // namespace sirius::exchange
