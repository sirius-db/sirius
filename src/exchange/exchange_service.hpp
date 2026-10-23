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

#include "exchange/transport.hpp"

#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace sirius::exchange {

/// Where an exchange stream's batches arrive: its EXCHANGE_SOURCE. Called from any thread.
class inbound {
 public:
  virtual ~inbound() = default;
  /// Batch @p batch (numbered per sender, from 0) of the stream, from rank @p from.
  virtual void on_batch(rank_t from,
                        std::uint64_t batch,
                        std::shared_ptr<cucascade::data_batch> data) = 0;
  /// Rank @p from will send no more.
  virtual void on_end(rank_t from) = 0;
  /// The exchange failed: no more batches will come.
  virtual void on_error(std::exception_ptr error) = 0;
};

/**
 * @brief One process's end of every exchange: a transport, and the streams' receivers.
 *
 * EXCHANGE_SINKs send through it; each EXCHANGE_SOURCE attaches to its stream id. A peer may send
 * before this process has planned the query, so what arrives for a stream not yet attached is held
 * and replayed on attach. A transport error reaches every attached receiver, and any attached
 * later. An aborted query's error reaches its streams' receivers, attached now or later, on
 * every rank, and its later arrivals are dropped.
 */
class exchange_service {
 public:
  exchange_service(fabric& fabric,
                   frame_ring& send,
                   frame_ring& recv,
                   cucascade::memory::memory_space& gpu,
                   cucascade::memory::memory_space* host,
                   transport::options opts);
  exchange_service(fabric& fabric,
                   frame_ring& send,
                   frame_ring& recv,
                   cucascade::memory::memory_space& gpu,
                   cucascade::memory::memory_space* host)
    : exchange_service(fabric, send, recv, gpu, host, transport::options{})
  {
  }

  [[nodiscard]] rank_t rank() const noexcept { return _transport->rank(); }
  [[nodiscard]] rank_t world_size() const noexcept { return _transport->world_size(); }

  /// @throws the exchange's error once it has failed.
  void send(stream_id stream, rank_t to, std::shared_ptr<cucascade::data_batch> batch)
  {
    _transport->send(stream, to, std::move(batch));
  }
  /// @throws the exchange's error once it has failed.
  void close(stream_id stream, rank_t to) { _transport->close(stream, to); }

  /// See transport::wait_sent.
  [[nodiscard]] bool wait_sent(std::uint64_t query, std::chrono::milliseconds timeout)
  {
    return _transport->wait_sent(query, timeout);
  }

  /// Routes @p stream to @p receiver, replaying what arrived for it before.
  /// @throws sirius::invalid_input_exception when the stream already has a receiver.
  void attach(stream_id stream, std::shared_ptr<inbound> receiver);
  /// Stops routing @p stream; later arrivals for it are dropped.
  void detach(stream_id stream);

  /// This rank's run of @p query failed with @p what: fails its streams here and on every peer,
  /// so no rank waits for the others. Once per query; a query a peer aborted is not re-sent.
  void abort(std::uint64_t query, std::string const& what);

 private:
  struct held {
    rank_t from;
    std::uint64_t batch;
    std::shared_ptr<cucascade::data_batch> data;  // null: the end of `from`
  };

  [[nodiscard]] std::shared_ptr<inbound> receiver_of(stream_id stream, held event);
  /// Fails @p query's receivers with @p error. @return false if it had already ended.
  bool end_query(std::uint64_t query, std::exception_ptr error);

  std::mutex _mutex;
  std::map<stream_id, std::shared_ptr<inbound>> _receivers;
  std::map<stream_id, std::vector<held>> _held;
  std::map<stream_id, bool> _detached;
  std::map<std::uint64_t, std::exception_ptr> _aborted;
  std::exception_ptr _error;
  // Last: its engine thread calls back into the members above until it is destroyed.
  std::unique_ptr<transport> _transport;
};

}  // namespace sirius::exchange
