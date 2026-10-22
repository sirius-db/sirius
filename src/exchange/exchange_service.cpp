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

#include <utility>
#include <vector>

namespace sirius::exchange {

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
                              _error    = error;
                              receivers = _receivers;
                            }
                            for (auto& [stream, r] : receivers) {
                              r->on_error(error);
                            }
                          }};
  _transport = std::make_unique<transport>(fabric, send, recv, gpu, host, std::move(cb), opts);
}

std::shared_ptr<inbound> exchange_service::receiver_of(stream_id stream, held event)
{
  std::lock_guard lock{_mutex};
  if (auto it = _receivers.find(stream); it != _receivers.end()) { return it->second; }
  if (!_detached.contains(stream)) { _held[stream].push_back(std::move(event)); }
  return nullptr;
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
}

void exchange_service::detach(stream_id stream)
{
  std::lock_guard lock{_mutex};
  _receivers.erase(stream);
  _held.erase(stream);
  _detached[stream] = true;
}

}  // namespace sirius::exchange
