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

#include <rmm/cuda_stream.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exchange {

/// The exchange's id for one logical flow of batches, the same on every rank.
using stream_id = std::uint64_t;

/**
 * @brief Moves batches between the ranks of an exchange, through a fabric.
 *
 * Senders call send() and close() from any thread. A batch for this rank is handed straight to
 * on_batch(); one for a peer is staged frame by frame into the send ring and written into the
 * peer's receive ring.
 *
 * **Credits.** The receive ring is split evenly among the peers: each owns
 * receive frames / (world size - 1) of them, fixed at start. A sender writes only into frames it
 * holds as credits; the receiver hands a frame back once it has copied it out. There is no lane
 * set-up and no grant.
 *
 * **Landing.** A received batch lands on the GPU under a reservation if one is free now, else in
 * host memory. With no room in either, its frames stay in the receive ring, holding their
 * credits, which stops that peer's writes until room appears.
 *
 * **End of stream.** close() sends the number of batches sent; the receiver calls on_end() once
 * it has delivered that many, whatever order the notifications arrived in.
 *
 * **Failure.** Fail-stop: a failed write, a malformed notification, a peer's error or a stall
 * longer than options::stall_timeout calls on_error() once, tells every peer, and ends the
 * transport; send() and close() then rethrow.
 *
 * One engine thread drives the fabric and calls on_batch(), on_end() and on_error() for remote
 * senders; for this rank's own batches they run on the caller's thread.
 */
class transport {
 public:
  struct callbacks {
    std::function<void(stream_id, rank_t from, std::shared_ptr<cucascade::data_batch>)> on_batch;
    std::function<void(stream_id, rank_t from)> on_end;
    std::function<void(std::exception_ptr)> on_error;
  };

  struct options {
    /// send() blocks while batches queued for peers hold more than this.
    std::size_t max_queued_bytes = std::size_t{1} << 30;
    std::chrono::milliseconds stall_timeout{std::chrono::minutes{10}};
  };

  /**
   * @param fabric Its rank and world size are the transport's; its writes go between @p send and
   *        @p recv, which it registered.
   * @param gpu Where received batches land first, and @p host (may be null) where they land when
   *        the GPU has no room.
   * @throws sirius::invalid_input_exception when the receive ring has fewer frames than peers.
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
  /// may be delivered in any order.
  /// @throws the transport's error once it has failed.
  void send(stream_id stream, rank_t to, std::shared_ptr<cucascade::data_batch> batch);

  /// No more batches of @p stream to rank @p to.
  /// @throws the transport's error once it has failed.
  void close(stream_id stream, rank_t to);

 private:
  struct engine;

  void run();

  std::unique_ptr<engine> _engine;
  std::mutex _mutex;
  std::condition_variable _room;
  std::exception_ptr _error;
  bool _stop{false};
  std::thread _thread;
};

}  // namespace sirius::exchange
