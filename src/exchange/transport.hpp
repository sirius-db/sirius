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

#include "exchange/fabric.hpp"
#include "exchange/messages.hpp"
#include "sirius/exception.hpp"

#include <rmm/cuda_stream.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exchange {

/// The exchange's id for one logical flow of batches, the same on every rank.
using stream_id = std::uint64_t;

/// A distributed query's number, the same on every rank, from 1; not a sirius::query_id_t window.
using distributed_query_id = std::uint64_t;

/// A stream id's bits above these number its query, the same on every rank.
constexpr unsigned query_stream_bits = 20;

[[nodiscard]] constexpr distributed_query_id query_of(stream_id stream) noexcept
{
  return stream >> query_stream_bits;
}

/// Stream @p n of query @p query.
/// @throws sirius::invalid_input_exception when @p n does not fit below query_stream_bits.
[[nodiscard]] inline stream_id make_stream_id(distributed_query_id query, std::uint64_t n)
{
  if (n >= (std::uint64_t{1} << query_stream_bits)) {
    throw sirius::invalid_input_exception("exchange: query {} needs more than {} streams",
                                          query,
                                          std::uint64_t{1} << query_stream_bits);
  }
  return (query << query_stream_bits) | n;
}

/**
 * @brief Moves batches between the ranks of an exchange, through a fabric.
 *
 * Senders call send() and close() from any thread. A batch for this rank is handed straight to
 * on_batch(); one for a peer is staged frame by frame into the send ring and written into the
 * peer's receive ring.
 *
 * **Credits.** The receive ring is split evenly among the peers: each owns
 * receive frames / (world size - 1) of them, fixed at start. A sender writes only into frames it
 * holds as credits; the receiver hands a frame back once it has copied it out.
 * TODO(#2371): grant credits on demand, so ring memory does not grow with the world size.
 *
 * **Landing.** A received batch lands on the GPU under a reservation if one is free now, else in
 * host memory. With no room in either, its frames stay in the receive ring, holding their
 * credits, which stops that peer's writes until room appears.
 *
 * **End of stream.** close() sends the number of batches sent; the receiver calls on_end() once
 * it has delivered that many, whatever order the notifications arrived in.
 *
 * **Failure.** Fail-stop (TODO(#2373): survive a lost peer): a failed write, a malformed
 * notification, a peer's error or a stall longer than options::stall_timeout calls on_error() once,
 * tells every peer, and ends the transport; send() and close() then rethrow. A failed query is
 * narrower: abort() tells every peer, whose on_abort() ends that query's streams, and the transport
 * carries on. So does an on_batch() or on_end() that throws on the engine thread: it aborts its
 * stream's query here and on every peer.
 *
 * One engine thread drives the fabric and calls on_batch(), on_end() and on_error() for remote
 * senders; for this rank's own batches they run on the caller's thread.
 */
class transport {
 public:
  /// Called from the engine thread and from send()/close() callers at the same time, so they must
  /// be thread-safe, and must not call send() or close(): the engine thread would wait on itself.
  struct callbacks {
    /// @p batch numbers the batches of one stream from one sender, from 0, in send() order.
    std::function<void(
      stream_id, rank_t from, std::uint64_t batch, std::shared_ptr<cucascade::data_batch>)>
      on_batch;
    std::function<void(stream_id, rank_t from)> on_end;
    std::function<void(std::exception_ptr)> on_error;
    /// Rank @p from's run of query @p query failed with @p what.
    std::function<void(distributed_query_id query, rank_t from, std::string what)> on_abort;
  };

  struct options {
    /// send() blocks while its batch would put more than this in flight to peers; a larger batch
    /// still goes once nothing else is queued. Bounds how long senders' batches stay pinned in
    /// the queue (they are already in GPU memory, under their producers' reservations).
    std::size_t max_queued_bytes = std::size_t{1} << 30;
    /// How long the engine may hold work without progress before it fails: a lost peer is
    /// fail-stop (#2373), so this bounds how long one hangs a query.
    std::chrono::milliseconds stall_timeout{std::chrono::minutes{10}};
    /// Called on the engine thread, once per batch, when a batch of this many bytes finds no room
    /// on either tier. Must not block. Unset: the batch waits for room to appear.
    std::function<void(std::size_t bytes)> request_room;
  };

  /**
   * @param fabric Its rank and world size are the transport's; its writes go between @p send and
   *        @p recv, which it registered. All referenced objects must outlive the transport.
   *        Construct on the device of @p gpu: the engine thread it starts runs on the current one.
   * @param gpu Where received batches land first, and @p host (may be null) where they land when
   *        the GPU has no room.
   * @throws sirius::invalid_input_exception when the receive ring has fewer frames than peers, or
   *         its frames differ in size from the send ring's.
   */
  transport(fabric& fabric,
            frame_ring& send,
            frame_ring& recv,
            cucascade::memory::memory_space& gpu,
            cucascade::memory::memory_space* host,
            callbacks callbacks,
            options opts);
  transport(fabric& fabric,
            frame_ring& send,
            frame_ring& recv,
            cucascade::memory::memory_space& gpu,
            cucascade::memory::memory_space* host,
            callbacks callbacks)
    : transport(fabric, send, recv, gpu, host, std::move(callbacks), options{})
  {
  }

  /// Stops the engine; work still in flight is dropped.
  ~transport();

  transport(transport const&)            = delete;
  transport& operator=(transport const&) = delete;

  [[nodiscard]] rank_t rank() const noexcept;
  [[nodiscard]] rank_t world_size() const noexcept;

  /// Sends @p batch, on the GPU, to rank @p to on @p stream. Batches of one stream to one rank
  /// may be delivered in any order. A batch with no rows is dropped for a peer and not counted.
  /// @throws the transport's error once it has failed.
  void send(stream_id stream, rank_t to, std::shared_ptr<cucascade::data_batch> batch);

  /// No more batches of @p stream to rank @p to. Call once per stream and rank, after its last
  /// send(): a second close, or a send after it, fails the transport on the peer. The stream's
  /// count of batches sent to @p to goes with it.
  /// @throws the transport's error once it has failed.
  void close(stream_id stream, rank_t to);

  /// Tells every peer that this rank's run of @p query failed with @p what. Does nothing once the
  /// transport has failed: its peers have heard of that.
  void abort(distributed_query_id query, std::string what);

  /// Waits until this rank holds no batch it sent on a stream of @p query: the transport keeps a
  /// sent batch until its last frame is staged, or until it fails, and its memory may belong to
  /// the query.
  /// @return false after @p timeout.
  [[nodiscard]] bool wait_sent(distributed_query_id query, std::chrono::milliseconds timeout);

 private:
  struct engine;

  void run();
  /// A batch of @p query sent to a peer was freed.
  void released(distributed_query_id query);

  std::unique_ptr<engine> _engine;
  std::mutex _mutex;
  std::condition_variable _room;
  /// Sent batches not yet freed, by query; _released is notified as they are.
  std::map<distributed_query_id, std::size_t> _held;
  std::condition_variable _released;
  std::exception_ptr _error;
  /// abort() notices for the engine thread to send to every peer.
  std::vector<abort_message> _aborts;
  bool _stop{false};
  std::thread _thread;
};

}  // namespace sirius::exchange
