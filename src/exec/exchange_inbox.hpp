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

#include <rmm/cuda_stream.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace cucascade {
class data_batch;
class data_repository;
namespace memory {
class memory_space;
class reservation;
}  // namespace memory
}  // namespace cucascade

namespace sirius::exec {

class exchange_staging;

/**
 * @brief Batches a transport received from another process, held by token until their receiver
 *        takes them.
 *
 * The transport lands a batch in this process's memory and adopts it here; the receiver pushes it
 * into a fragment input by its token (Fragment::push_received). Tokens start at 1 and are never
 * reused. Every method is thread-safe.
 */
class exchange_inbox {
 public:
  explicit exchange_inbox(cucascade::memory::memory_space& gpu);

  /// Frees at least `bytes` of GPU memory if it can, e.g. by spilling to host; blocks until done.
  /// Returns a reservation of `bytes` it made as soon as there was room, so that no other reserver
  /// takes the freed bytes first, or null to have the caller try once more itself.
  using make_room_fn =
    std::function<std::unique_ptr<cucascade::memory::reservation>(std::size_t bytes)>;

  /// Lets held batches spill: they become visible to @p staging, so the downgrade executor may
  /// move them to host while they wait for their receivers, and bring_to_gpu() calls
  /// @p make_room before giving up on GPU memory. Without this, a held batch stays on the GPU
  /// until it is taken.
  void enable_spill(exchange_staging& staging, make_room_fn make_room = {});

  /// Moves a spilled @p batch back to the GPU, for a sender reading it out; a batch already there
  /// is left as it is. The copy runs on the exchange's stream.
  /// @throws rmm::out_of_memory when it cannot be reserved, even after making room.
  void bring_to_gpu(cucascade::data_batch& batch);

  /// Holds @p batch, already received whole, until its receiver takes it by the returned token.
  /// @p on_gone runs once when the token is taken, released or closed, outside the exchange's
  /// lock, so a transport can account for what its landings still hold. It must not throw: close()
  /// runs it from the Context's destructor.
  /// @throws sirius::invalid_input_exception on a null batch.
  [[nodiscard]] std::uint64_t adopt(std::shared_ptr<cucascade::data_batch> batch,
                                    std::function<void()> on_gone = {});

  /// Consumes the token of a received batch, possibly spilled to host.
  /// @throws sirius::invalid_input_exception on a token that holds no received batch.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> take_batch(std::uint64_t token);

  /// The batch @p token holds, left in place for its receiver.
  /// @throws sirius::invalid_input_exception on a token that holds no received batch.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> peek_batch(std::uint64_t token) const;

  /// Frees what @p token holds; an unknown or consumed token is ignored.
  void release(std::uint64_t token);

  /// Tokens neither released nor consumed.
  [[nodiscard]] std::size_t outstanding() const;

  /// Frees every token. Later calls throw, except close() and the accessors.
  void close();

  [[nodiscard]] int device() const noexcept { return _device; }
  [[nodiscard]] cucascade::memory::memory_space& space() const noexcept { return _gpu; }

 private:
  void require_open() const;
  /// A reservation of @p bytes, making room first if there is too little, or null.
  [[nodiscard]] std::unique_ptr<cucascade::memory::reservation> reserve(std::size_t bytes);
  /// Runs and forgets @p token's on_gone, if any; call without the lock.
  void gone(std::uint64_t token);

  cucascade::memory::memory_space& _gpu;
  int _device;
  /// Where bring_to_gpu() copies, so it never waits for an operator's kernels.
  rmm::cuda_stream _stream;
  mutable std::mutex _mutex;
  make_room_fn _make_room;
  std::uint64_t _next{1};
  bool _closed{false};
  /// Held batches, by batch id; tracked by the exchange staging once spill is on.
  std::shared_ptr<cucascade::data_repository> _held;
  /// Token -> batch id in _held.
  std::map<std::uint64_t, std::uint64_t> _held_ids;
  /// Callbacks of adopted tokens, run when they go.
  std::map<std::uint64_t, std::function<void()>> _on_gone;
};

}  // namespace sirius::exec
