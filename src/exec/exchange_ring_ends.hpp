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

#include "exec/exchange_ring.hpp"
#include "exec/exchange_ring_wire.hpp"

#include <rmm/cuda_stream.hpp>

#include <cucascade/memory/common.hpp>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exec {

class exchange_inbox;

/// No tier had room for a landing batch: the ring exchange's only failure (invariant I5).
struct ring_no_room : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief The engine's two ends of the ring exchange on one CN.
 *
 * Owns the send and receive rings (frame_ring) and the copies in and out of them; the transport
 * (the CN's Rust side) owns everything between: which frame goes where, NIXL WRITEs, credits.
 *
 * Sender: export_batch() takes a batch out of its fragment's output and describes it as a stream
 * (exchange_ring_wire.hpp): `gpu_columns` from the GPU, `host_columns` from a batch spilled to
 * host, `compressed` from a compressed host batch. Any other batch (on disk, say) is brought back
 * to the GPU first. stage() copies one frame's worth of the stream into a send frame.
 *
 * Receiver: begin() decodes a header and picks where the batch lands, without waiting: the GPU
 * within its landing budget, else host within its share, else disk. copy_out() copies one received
 * frame to the landing; once done, the frame can go back to its sender (I2). seal() hands the
 * whole batch to the exchange inbox, which holds it under a token like any received batch.
 *
 * Every copy is asynchronous and returns a ticket that done() polls; none waits on GPU memory,
 * another transfer or an operator (I1). A disk landing's copy-out waits on its write, on a thread
 * of its own.
 */
class ring_exchange {
 public:
  struct options {
    std::size_t frame_bytes = std::size_t{64} << 20;
    std::size_t send_bytes  = std::size_t{1} << 30;
    std::size_t recv_bytes  = std::size_t{3} << 29;
  };

  /// Allocates both rings on @p device. Call before the engine's pool is built, so the rings are
  /// not carved out of its budget.
  ring_exchange(int device, options const& opts);
  ~ring_exchange();

  ring_exchange(ring_exchange const&)            = delete;
  ring_exchange& operator=(ring_exchange const&) = delete;

  /// Where landing batches may go, and how much each tier may hold of batches landed and not yet
  /// taken. A null space is a tier that is not used.
  struct landing_tiers {
    cucascade::memory::memory_space* host{nullptr};
    cucascade::memory::memory_space* disk{nullptr};
    std::size_t gpu_bytes{0};
    std::size_t host_bytes{0};
  };

  /// Wires the ends to the engine once it is up: batches land in @p exchange's GPU space and are
  /// handed to it. Until then export_batch() and begin() throw.
  void attach(std::shared_ptr<exchange_inbox> exchange, landing_tiers tiers);

  [[nodiscard]] frame_ring& send_ring() noexcept { return _send; }
  [[nodiscard]] frame_ring& recv_ring() noexcept { return _recv; }
  [[nodiscard]] int device() const noexcept { return _device; }
  [[nodiscard]] std::size_t frame_bytes() const noexcept { return _send.frame_bytes(); }

  // ---- sender ------------------------------------------------------------------------------

  struct send_job {
    std::uint64_t token;
    ring::batch_kind kind;
    std::uint64_t rows;
    std::uint64_t stream_bytes;
    std::vector<std::uint8_t> header;
  };

  /// Holds @p batch until release_job(), and describes it. Brings a batch that is neither on the
  /// GPU, raw on host, nor compressed on host back to the GPU first, which may wait for memory.
  /// @return nullopt for a batch with no rows.
  /// @throws sirius::invalid_input_exception on a column neither fixed-width nor STRING.
  [[nodiscard]] std::optional<send_job> export_batch(std::shared_ptr<cucascade::data_batch> batch);

  /// Copies the stream's frame @p k of job @p token into send frame @p frame.
  /// @return a ticket for done().
  [[nodiscard]] std::uint64_t stage(std::uint64_t token, std::uint32_t k, std::uint32_t frame);

  /// Frees what job @p token holds; an unknown token is ignored.
  void release_job(std::uint64_t token);

  // ---- receiver ----------------------------------------------------------------------------

  struct landing {
    std::uint64_t id;
    cucascade::memory::Tier tier;
    std::uint64_t rows;
    std::uint64_t stream_bytes;
    /// Frames the stream takes: ceil(stream_bytes / frame_bytes).
    std::uint32_t frames;
  };

  /// Decodes @p header and reserves room for the batch, without waiting.
  /// @throws sirius::invalid_input_exception on a malformed header.
  /// @throws ring_no_room when no tier has room, with each tier's numbers.
  [[nodiscard]] landing begin(std::span<std::uint8_t const> header);

  /// Copies @p len bytes of receive frame @p frame, the stream's frame @p k, to landing
  /// @p landing. @return a ticket for done().
  /// @throws sirius::invalid_input_exception on a frame out of the stream or copied twice.
  [[nodiscard]] std::uint64_t copy_out(std::uint64_t landing,
                                       std::uint32_t k,
                                       std::uint32_t frame,
                                       std::uint64_t len);

  /// Hands a landing whose every frame was copied and done to the exchange inbox.
  /// @return the exchange inbox's token for it.
  /// @throws sirius::invalid_input_exception when a frame is missing or a copy not yet done.
  [[nodiscard]] std::uint64_t seal(std::uint64_t landing);

  /// Frees what a landing holds; an unknown landing is ignored. Its copies still in flight finish
  /// first.
  void abort(std::uint64_t landing);

  /// Frees every job and landing while the memory spaces they came from still exist; later
  /// calls throw. The rings stay allocated until the last reference goes, so a transport that
  /// registered them can deregister them first.
  void close();

  // ---- both --------------------------------------------------------------------------------

  /// Whether ticket @p ticket's copies finished, without waiting; true consumes the ticket.
  /// @throws std::runtime_error when they failed (a disk write, say), consuming it.
  [[nodiscard]] bool done(std::uint64_t ticket);

  /// Waits for @p ticket, then consumes it as done() does.
  void wait(std::uint64_t ticket);

  struct counters {
    std::uint64_t sent_batches[4]{};    ///< by batch_kind
    std::uint64_t sent_bytes[4]{};      ///< stream bytes, by batch_kind
    std::uint64_t landed_batches[3]{};  ///< by tier: GPU, host, disk
    std::uint64_t landed_bytes[3]{};
    std::uint64_t refused{0};    ///< ring_no_room
    std::uint64_t gpu_held{0};   ///< landed on GPU, not yet taken
    std::uint64_t host_held{0};  ///< landed on host, not yet taken
  };
  [[nodiscard]] counters stats() const;
  /// stats() as one log line.
  [[nodiscard]] std::string stats_line() const;

 private:
  struct job;
  struct landing_state;
  struct ticket;
  struct disk_task;
  struct held;

  [[nodiscard]] std::uint64_t new_ticket(std::shared_ptr<ticket> t);
  [[nodiscard]] std::shared_ptr<ticket> record(rmm::cuda_stream_view stream);
  void run_disk_writes();

  int _device;
  frame_ring _send;
  frame_ring _recv;
  rmm::cuda_stream _send_stream;
  rmm::cuda_stream _recv_stream;

  std::shared_ptr<exchange_inbox> _exchange;
  landing_tiers _tiers;
  bool _closed{false};
  /// Bytes landed per tier and not yet taken; shared with the exchange inbox's on_gone callbacks.
  std::shared_ptr<held> _held;

  mutable std::mutex _mutex;
  std::uint64_t _next{1};
  std::map<std::uint64_t, std::shared_ptr<job>> _jobs;
  std::map<std::uint64_t, std::shared_ptr<landing_state>> _landings;
  std::map<std::uint64_t, std::shared_ptr<ticket>> _tickets;
  counters _counters;

  // Disk landings' copy-outs, written one at a time through a pinned bounce frame.
  std::mutex _disk_mutex;
  std::condition_variable _disk_cv;
  std::deque<std::shared_ptr<disk_task>> _disk_queue;
  bool _disk_stop{false};
  void* _bounce{nullptr};
  std::thread _disk_thread;
};

}  // namespace sirius::exec
