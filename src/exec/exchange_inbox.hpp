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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>

namespace cucascade {
class data_batch;
namespace memory {
class memory_space;
}
}  // namespace cucascade

namespace sirius::exec {

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

  /// Holds @p batch, already received whole, until its receiver takes it by the returned token.
  /// @p on_gone runs once when the token is taken, released or closed, outside the exchange's
  /// lock, so a transport can account for what its landings still hold. It must not throw: close()
  /// runs it from the Context's destructor.
  /// @throws sirius::invalid_input_exception on a null batch.
  [[nodiscard]] std::uint64_t adopt(std::shared_ptr<cucascade::data_batch> batch,
                                    std::function<void()> on_gone = {});

  /// Consumes the token of a received batch.
  /// @throws sirius::invalid_input_exception on a token that holds no received batch.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> take_batch(std::uint64_t token);

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
  /// Runs and forgets @p token's on_gone, if any; call without the lock.
  void gone(std::uint64_t token);

  cucascade::memory::memory_space& _gpu;
  int _device;
  mutable std::mutex _mutex;
  std::uint64_t _next{1};
  bool _closed{false};
  std::map<std::uint64_t, std::shared_ptr<cucascade::data_batch>> _held;
  /// Callbacks of adopted tokens, run when they go.
  std::map<std::uint64_t, std::function<void()>> _on_gone;
};

}  // namespace sirius::exec
