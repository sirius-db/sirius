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

#include "memory/slab_memory_resource.hpp"

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <variant>
#include <vector>

namespace cucascade {
class data_batch;
class data_repository;
namespace memory {
class reservation;
}
}  // namespace cucascade

namespace sirius::exec {
class exchange_staging;
}

namespace sirius::exec {

/// One column of a batch sent by direct exchange; its buffers travel separately.
struct direct_column {
  cudf::data_type type;
  cudf::size_type null_count;
  bool has_mask;
  cudf::type_id offsets{cudf::type_id::EMPTY};  ///< STRING only: INT32 or INT64.
  std::uint64_t chars{0};                       ///< STRING only: bytes of character data.

  bool operator==(direct_column const&) const = default;
};

/**
 * @brief The shape of a batch sent by direct exchange.
 *
 * Its buffers are walked per column in this order: the null mask if any, the data (the chars
 * for STRING), then the STRING offsets.
 */
struct direct_layout {
  cudf::size_type rows;
  std::vector<direct_column> columns;

  bool operator==(direct_layout const&) const = default;
};

/// One buffer of a direct_layout: `wire` bytes are sent into an allocation of `alloc` bytes.
struct direct_buffer {
  std::size_t column;
  std::size_t wire;
  std::size_t alloc;

  bool operator==(direct_buffer const&) const = default;
};

[[nodiscard]] std::vector<std::uint8_t> encode_layout(direct_layout const& layout);

/// @throws sirius::invalid_input_exception unless @p bytes is a well-formed layout of at least
///         one row and one fixed-width or STRING column.
[[nodiscard]] direct_layout decode_layout(std::span<std::uint8_t const> bytes);

/// @p layout's buffers in walk order, each allocation 256-byte aligned within @p limit bytes.
/// @throws sirius::invalid_input_exception when the aligned allocations exceed @p limit.
[[nodiscard]] std::vector<direct_buffer> plan_buffers(direct_layout const& layout,
                                                      std::size_t limit);

/// A table's layout and its buffer addresses, in plan_buffers order.
struct direct_export {
  direct_layout layout;
  std::vector<void const*> buffers;
};

/// Describes @p table, which must have rows and no sliced (non-zero offset) column. STRING chars
/// sizes are read on @p stream.
/// @throws sirius::invalid_input_exception on a column that is neither fixed-width nor STRING.
[[nodiscard]] direct_export describe_table(cudf::table_view const& table,
                                           rmm::cuda_stream_view stream);

/**
 * @brief Batches sent and received by direct exchange, held by token until the transfer ends.
 *
 * A sender exports a batch and keeps it alive under its token while the transport writes its
 * buffers into the receiver's; a receiver allocates those buffers from its slab under another
 * token, then takes them as a table. Tokens start at 1 and are never reused. Every method is
 * thread-safe and runs on the slab's device.
 */
class direct_exchange {
 public:
  struct exported {
    std::uint64_t token;
    std::uint64_t rows;
    std::vector<std::uint8_t> layout;
    std::vector<std::uint64_t> src;  ///< [address, length] per non-empty buffer.
  };

  direct_exchange(cucascade::memory::memory_space& gpu, memory::slab_region region);

  /// Frees at least `bytes` of GPU memory if it can, e.g. by spilling to host; blocks until done.
  using make_room_fn = std::function<void(std::size_t bytes)>;

  /// Lets received batches spill: sealed batches are visible to @p staging, and allocate() and
  /// export_batch() call @p make_room once before giving up on GPU memory, again if it throws
  /// (the downgrade executor cancels queued requests whenever a query window closes). Without
  /// this, a full pool fails the transfer at once.
  void enable_spill(exchange_staging& staging, make_room_fn make_room);

  /// Holds @p batch, or a copy of it if it is sliced or not in the slab, until released.
  /// Its buffers are ready to read when this returns.
  /// @return nullopt for a batch with no rows.
  /// @throws sirius::invalid_input_exception on a batch that is not on the GPU or has a column
  ///         that is neither fixed-width nor STRING.
  [[nodiscard]] std::optional<exported> export_batch(std::shared_ptr<cucascade::data_batch> batch);

  /// Allocates the buffers of the encoded @p layout. When they cannot be reserved and spilling is
  /// enabled, it makes room once and retries; it never waits on other transfers.
  /// @return the token and the [address, length] of each non-empty buffer.
  /// @throws sirius::invalid_input_exception on a malformed layout.
  /// @throws rmm::out_of_memory when the buffers cannot be reserved.
  [[nodiscard]] std::pair<std::uint64_t, std::vector<std::uint64_t>> allocate(
    std::span<std::uint8_t const> layout);

  /// Consumes the token of a received batch, wrapping its buffers without a copy.
  /// @throws sirius::invalid_input_exception on a token that holds no unsealed received batch.
  [[nodiscard]] std::unique_ptr<cudf::table> take(std::uint64_t token);

  /// Wraps a fully received batch as a data batch the downgrade executor may spill while it
  /// waits for its receiver. A sealed, consumed or unknown token is left as it is.
  void seal(std::uint64_t token);

  /// Consumes the token of a received batch, sealed (possibly spilled to host) or not.
  /// @throws sirius::invalid_input_exception on a token that holds no received batch.
  [[nodiscard]] std::shared_ptr<cucascade::data_batch> take_batch(std::uint64_t token);

  /// Frees what @p token holds; an unknown or consumed token is ignored.
  void release(std::uint64_t token);

  [[nodiscard]] std::size_t outstanding() const;

  /// Frees every token. Later calls throw, except close() and the accessors.
  void close();

  [[nodiscard]] memory::slab_region const& region() const noexcept { return _region; }
  [[nodiscard]] cucascade::memory::memory_space& space() const noexcept { return _gpu; }

 private:
  struct received {
    direct_layout layout;
    std::vector<rmm::device_buffer> buffers;
  };

  void require_open() const;
  /// Wraps @p entry's buffers as a table; @p entry is consumed.
  [[nodiscard]] static std::unique_ptr<cudf::table> to_table(received&& entry);
  /// Moves a spilled @p batch back to the GPU before it is sent.
  void bring_to_gpu(cucascade::data_batch& batch);

  cucascade::memory::memory_space& _gpu;
  memory::slab_region _region;
  mutable std::mutex _mutex;
  std::uint64_t _next{1};
  bool _closed{false};
  /// A sent batch's keepalive, or a received batch's buffers.
  std::map<std::uint64_t, std::variant<std::shared_ptr<void const>, received>> _entries;
  make_room_fn _make_room;

  /// A reservation of @p bytes, making room first if there is too little.
  std::unique_ptr<cucascade::memory::reservation> reserve(std::size_t bytes);
  /// Sealed received batches, by batch id; tracked by the exchange staging once spill is on.
  std::shared_ptr<cucascade::data_repository> _sealed;
  /// Token -> batch id in _sealed.
  std::map<std::uint64_t, std::uint64_t> _sealed_ids;
};

}  // namespace sirius::exec
