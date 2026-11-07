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

#include "log/logging.hpp"
#include "sirius/exception.hpp"

#include <cucascade/data/data_batch.hpp>

#include <algorithm>
#include <exception>
#include <iterator>
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

std::uint64_t bytes_of(cucascade::data_batch& batch)
{
  return batch.to_read_only().get_data()->get_size_in_bytes();
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
                            if (from != _rank && data) {
                              auto const bytes = bytes_of(*data);
                              std::lock_guard lock{_mutex};
                              auto& t = _traffic[query_of(stream)];
                              t.received_bytes += bytes;
                              ++t.received_batches;
                            }
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
                          [this](distributed_query_id query, rank_t from, std::string what) {
                            end_query(
                              query,
                              std::make_exception_ptr(sirius::internal_exception(
                                "exchange: query {} failed on rank {}: {}", query, from, what)));
                          }};
  _transport  = std::make_unique<transport>(fabric, send, recv, gpu, host, std::move(cb), opts);
  _rank       = _transport->rank();
  _world_size = _transport->world_size();
}

std::shared_ptr<inbound> exchange_service::receiver_of(stream_id stream, held event)
{
  std::lock_guard lock{_mutex};
  if (_aborted.contains(query_of(stream))) { return nullptr; }
  if (auto it = _receivers.find(stream); it != _receivers.end()) { return it->second; }
  if (!retired(query_of(stream)) && !_detached.contains(stream)) {
    _held[stream].push_back(std::move(event));
  }
  return nullptr;
}

bool exchange_service::end_query(distributed_query_id query, std::exception_ptr error)
{
  std::vector<std::shared_ptr<inbound>> receivers;
  {
    std::lock_guard lock{_mutex};
    if (retired(query) || !_aborted.emplace(query, error).second) { return false; }
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

void exchange_service::abort(distributed_query_id query, std::string const& what)
{
  auto error = std::make_exception_ptr(
    sirius::internal_exception("exchange: query {} failed on rank {}: {}", query, rank(), what));
  if (!end_query(query, std::move(error))) { return; }
  std::lock_guard lock{_mutex};
  if (_transport) { _transport->abort(query, what); }
}

bool exchange_service::wait_sent(distributed_query_id query, std::chrono::milliseconds timeout)
{
  {
    std::lock_guard lock{_mutex};
    if (!_transport) { return true; }
  }
  return _transport->wait_sent(query, timeout);
}

void exchange_service::shutdown()
{
  std::unique_ptr<transport> stopped;
  {
    std::lock_guard lock{_mutex};
    if (!_error) {
      _error = std::make_exception_ptr(
        sirius::internal_exception("exchange: shut down with the Sirius context"));
    }
    stopped = std::move(_transport);
    _held.clear();
    _receivers.clear();
  }
  // Outside the lock: the engine thread calls back into this service until it is joined.
  stopped.reset();
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
  auto const bytes = to != _rank && batch ? bytes_of(*batch) : 0;
  {
    std::lock_guard lock{_mutex};
    if (bytes > 0) {
      auto& t = _traffic[query_of(stream)];
      t.sent_bytes += bytes;
      ++t.sent_batches;
    }
    if (_error) { std::rethrow_exception(_error); }
    // A peer failed this query: stop producing for it. close() still goes through, since sinks
    // and all-gathers close their streams on the way out.
    if (auto it = _aborted.find(query_of(stream)); it != _aborted.end()) {
      std::rethrow_exception(it->second);
    }
  }
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
  // Sources go with their plan, which can outlive the query's retirement.
  if (!retired(query_of(stream))) { _detached.insert(stream); }
}

bool exchange_service::retired(distributed_query_id query) const
{
  // Queries are numbered from 1; streams of query 0 belong to no query and never retire.
  if (query == 0) { return false; }
  if (query <= _retired) { return true; }
  auto const after = _retired_above.upper_bound(query);
  return after != _retired_above.begin() && query <= std::prev(after)->second;
}

void exchange_service::forget(distributed_query_id query) { forget_range(query, query); }

void exchange_service::forget_range(distributed_query_id first, distributed_query_id last)
{
  first = std::max<distributed_query_id>(first, 1);
  if (first > last) { return; }
  std::lock_guard lock{_mutex};
  // Merge [first, last] with the ranges it overlaps or touches, then advance the watermark.
  // Numbers stay below distributed_query_limit, so hi + 1 cannot wrap.
  auto lo = first;
  auto hi = last;
  auto it = _retired_above.upper_bound(lo);
  if (it != _retired_above.begin()) {
    if (auto const before = std::prev(it); before->second + 1 >= lo) {
      lo = before->first;
      hi = std::max(hi, before->second);
      it = _retired_above.erase(before);
    }
  }
  while (it != _retired_above.end() && it->first <= hi + 1) {
    hi = std::max(hi, it->second);
    it = _retired_above.erase(it);
  }
  if (lo <= _retired + 1) {
    _retired = std::max(_retired, hi);
  } else {
    _retired_above.emplace(lo, hi);
  }
  auto const in_range = [first, last](distributed_query_id query) {
    return query >= first && query <= last;
  };
  std::erase_if(_detached, [&](stream_id stream) { return in_range(query_of(stream)); });
  std::erase_if(_held, [&](auto const& h) { return in_range(query_of(h.first)); });
  _aborted.erase(_aborted.lower_bound(first), _aborted.upper_bound(last));
  for (auto t = _traffic.lower_bound(first); t != _traffic.end() && t->first <= last;) {
    SIRIUS_LOG_INFO(
      "exchange: query {} on rank {}: sent {:.3f} GB in {} batches, received {:.3f} GB in {} "
      "batches",
      t->first,
      _rank,
      static_cast<double>(t->second.sent_bytes) / 1e9,
      t->second.sent_batches,
      static_cast<double>(t->second.received_bytes) / 1e9,
      t->second.received_batches);
    t = _traffic.erase(t);
  }
}

}  // namespace sirius::exchange
