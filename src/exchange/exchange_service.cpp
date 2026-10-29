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

#include "exchange/exchange_service.hpp"

#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace sirius::exchange {
namespace {

std::string what_of(std::exception_ptr const& error)
{
  try {
    std::rethrow_exception(error);
  } catch (std::exception const& e) {
    return e.what();
  } catch (...) {
    return "unknown error";
  }
}

}  // namespace

exchange_service::exchange_service(fabric& fabric,
                                   frame_ring& send,
                                   frame_ring& recv,
                                   cucascade::memory::memory_space& gpu,
                                   cucascade::memory::memory_space* host,
                                   transport::options opts)
{
  transport::callbacks cb{[this](stream_id stream,
                                 rank_t from,
                                 std::uint64_t batch,
                                 std::shared_ptr<cucascade::data_batch> data) {
                            if (auto r = receiver_of(stream, {from, batch, data})) {
                              r->on_batch(from, batch, std::move(data));
                            }
                          },
                          [this](stream_id stream, rank_t from) {
                            if (auto r = receiver_of(stream, {from, 0, nullptr})) {
                              r->on_end(from);
                            }
                          },
                          [this](std::exception_ptr error) {
                            std::map<stream_id, std::shared_ptr<inbound>> receivers;
                            {
                              std::lock_guard lock{_mutex};
                              // Fail-stop: nothing reconnects, so later queries say how to recover.
                              _error    = std::make_exception_ptr(sirius::internal_exception(
                                "exchange: the transport to the other ranks failed ({}); restart "
                                   "every rank to run distributed queries again",
                                what_of(error)));
                              receivers = _receivers;
                            }
                            for (auto& [stream, r] : receivers) {
                              r->on_error(error);
                            }
                          },
                          [this](std::uint64_t query, rank_t from, std::string what) {
                            end_query(
                              query,
                              std::make_exception_ptr(sirius::internal_exception(
                                "exchange: query {} failed on rank {}: {}", query, from, what)));
                          }};
  _transport = std::make_unique<transport>(fabric, send, recv, gpu, host, std::move(cb), opts);
}

std::shared_ptr<inbound> exchange_service::receiver_of(stream_id stream, held event)
{
  std::lock_guard lock{_mutex};
  if (_aborted.contains(query_of(stream))) { return nullptr; }
  if (auto it = _receivers.find(stream); it != _receivers.end()) { return it->second; }
  if (!_detached.contains(stream)) { _held[stream].push_back(std::move(event)); }
  return nullptr;
}

bool exchange_service::end_query(std::uint64_t query, std::exception_ptr error)
{
  std::vector<std::shared_ptr<inbound>> receivers;
  {
    std::lock_guard lock{_mutex};
    if (!_aborted.emplace(query, error).second) { return false; }
    for (auto const& [stream, r] : _receivers) {
      if (query_of(stream) == query) { receivers.push_back(r); }
    }
    std::erase_if(_held, [query](auto const& h) { return query_of(h.first) == query; });
  }
  for (auto const& r : receivers) {
    r->on_error(error);
  }
  return true;
}

void exchange_service::abort(std::uint64_t query, std::string const& what)
{
  auto error = std::make_exception_ptr(
    sirius::internal_exception("exchange: query {} failed on rank {}: {}", query, rank(), what));
  if (end_query(query, std::move(error))) { _transport->abort(query, what); }
}

void exchange_service::attach(stream_id stream, std::shared_ptr<inbound> receiver)
{
  // Replayed under the lock, so nothing arriving meanwhile overtakes what was held.
  std::lock_guard lock{_mutex};
  if (!_receivers.emplace(stream, receiver).second) {
    throw sirius::invalid_input_exception("exchange: stream {} already has a receiver", stream);
  }
  if (auto it = _held.find(stream); it != _held.end()) {
    for (auto& e : it->second) {
      if (e.data) {
        receiver->on_batch(e.from, e.batch, std::move(e.data));
      } else {
        receiver->on_end(e.from);
      }
    }
    _held.erase(it);
  }
  if (_error) { receiver->on_error(_error); }
  if (auto it = _aborted.find(query_of(stream)); it != _aborted.end()) {
    receiver->on_error(it->second);
  }
}

void exchange_service::send(stream_id stream,
                            rank_t to,
                            std::shared_ptr<cucascade::data_batch> batch)
{
  throw_if_failed();
  _transport->send(stream, to, std::move(batch));
}

void exchange_service::close(stream_id stream, rank_t to)
{
  throw_if_failed();
  _transport->close(stream, to);
}

void exchange_service::throw_if_failed()
{
  std::lock_guard lock{_mutex};
  if (_error) { std::rethrow_exception(_error); }
}

void exchange_service::detach(stream_id stream)
{
  std::lock_guard lock{_mutex};
  _receivers.erase(stream);
  _held.erase(stream);
  _detached[stream] = true;
}

}  // namespace sirius::exchange
