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
#include <set>
#include <string>
#include <vector>

namespace sirius::exchange {

/// Where an exchange stream's batches arrive: its EXCHANGE_SOURCE. Called from any thread, possibly
/// under the service's lock (on attach() replay), so a callback must not call into
/// exchange_service.
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

  [[nodiscard]] rank_t rank() const noexcept { return _rank; }
  [[nodiscard]] rank_t world_size() const noexcept { return _world_size; }

  /// @throws the exchange's error once it has failed.
  void send(stream_id stream, rank_t to, std::shared_ptr<cucascade::data_batch> batch);
  /// @throws the exchange's error once it has failed.
  void close(stream_id stream, rank_t to);

  /// See transport::wait_sent. True once shut down: the transport freed what it held.
  [[nodiscard]] bool wait_sent(distributed_query_id query, std::chrono::milliseconds timeout);

  /// Routes @p stream to @p receiver, replaying what arrived for it before.
  /// @throws sirius::invalid_input_exception when the stream already has a receiver.
  void attach(stream_id stream, std::shared_ptr<inbound> receiver);
  /// Stops routing @p stream; later arrivals for it are dropped.
  void detach(stream_id stream);

  /// This rank's run of @p query failed with @p what: fails its streams here and on every peer,
  /// so no rank waits for the others. Once per query; a query a peer aborted is not re-sent.
  void abort(distributed_query_id query, std::string const& what);

  /// Stops the transport, joining its engine thread and freeing the batches it and this service
  /// hold, while the memory spaces they live in still exist. Plans may keep the service alive
  /// past that point; their later send() and close() throw.
  void shutdown();

  /// This rank has retired @p query: drops what is kept for it, and what arrives for it from now
  /// on. Queries are numbered when planned, so one run later (a prepared statement) can retire
  /// after higher-numbered ones.
  void forget(distributed_query_id query);

 private:
  struct held {
    rank_t from;
    std::uint64_t batch;
    std::shared_ptr<cucascade::data_batch> data;  // null: the end of `from`
  };

  [[nodiscard]] std::shared_ptr<inbound> receiver_of(stream_id stream, held event);
  void throw_if_failed();
  /// Fails @p query's receivers with @p error. @return false if it had already ended.
  bool end_query(distributed_query_id query, std::exception_ptr error);
  [[nodiscard]] bool retired(distributed_query_id query) const;

  rank_t _rank;
  rank_t _world_size;
  std::mutex _mutex;
  std::map<stream_id, std::shared_ptr<inbound>> _receivers;
  /// TODO(#2376): held batches sit outside any repository, so they cannot spill.
  std::map<stream_id, std::vector<held>> _held;
  std::set<stream_id> _detached;
  std::map<distributed_query_id, std::exception_ptr> _aborted;
  /// Every query from 1 to this one has retired here, and so has each in _retired_above.
  distributed_query_id _retired{0};
  std::set<distributed_query_id> _retired_above;
  /// The transport's failure, worded for the queries after it.
  std::exception_ptr _error;
  // Last: its engine thread calls back into the members above until it is destroyed. Null once
  // shut down.
  std::unique_ptr<transport> _transport;
};

}  // namespace sirius::exchange
