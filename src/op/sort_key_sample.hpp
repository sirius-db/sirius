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

#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/stream>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cucascade::memory {
class memory_space;
}  // namespace cucascade::memory

namespace sirius::op {

/**
 * @brief Evenly spaced key rows of the sorted batches of one sort, kept in host memory.
 *
 * Every batch gives the rows at one common stride, so each kept row stands for the same number
 * of input rows. The stride is a power of two. It doubles until at most max_key_rows() rows are
 * kept, and the rows taken at a smaller stride are then thinned to it. The rows kept at a stride
 * are among those kept at every smaller stride, so the kept rows depend only on the batches, not
 * on the order they are added in.
 *
 * Thread safe.
 */
class sort_key_sample {
 public:
  static constexpr std::size_t default_max_key_rows = 1048576;

  //! @throws std::invalid_argument if @p max_key_rows is zero.
  explicit sort_key_sample(std::size_t max_key_rows = default_max_key_rows);

  //! Position of the first row kept at @p stride, which must be a power of two.
  [[nodiscard]] static constexpr std::size_t first_row(std::size_t stride) noexcept
  {
    // Bits 0101...: below 2/3 of the stride, and first_row(2s) is first_row(s) or s more.
    return std::size_t{0x5555555555555555ULL} & (stride - 1);
  }

  //! Rows kept at @p stride from a batch of @p rows rows.
  [[nodiscard]] static constexpr std::size_t key_row_count(std::size_t rows,
                                                           std::size_t stride) noexcept
  {
    auto const first = first_row(stride);
    return rows > first ? 1 + (rows - first - 1) / stride : 0;
  }

  //! Input the kept rows were taken from.
  struct totals {
    std::size_t batches = 0;
    std::size_t rows    = 0;
    std::size_t bytes   = 0;
  };

  /**
   * @brief Keep the key rows of one sorted batch.
   *
   * Adding a batch id again replaces its earlier rows, so a retried task counts once.
   *
   * @param batch_id Names the batch.
   * @param sorted_keys Key columns of the sorted batch, on @p space's GPU.
   * @param batch_bytes Size of the whole batch.
   * @param space Memory space of the batch. Rows of earlier batches are thinned on its GPU when
   * the stride doubles.
   * @param stream Stream of the calling task. Synchronized before this returns.
   */
  void add(std::uint64_t batch_id,
           cudf::table_view sorted_keys,
           std::size_t batch_bytes,
           cucascade::memory::memory_space& space,
           ::cuda::stream_ref stream);

  [[nodiscard]] totals seen() const;
  [[nodiscard]] std::size_t stride() const;
  [[nodiscard]] std::size_t max_key_rows() const noexcept { return _max_key_rows; }

  //! Key rows held in host memory: at most max_key_rows() while no add() is running.
  [[nodiscard]] std::size_t host_key_rows() const;

  //! Later add() calls keep nothing.
  void close() noexcept { _closed.store(true, std::memory_order_release); }

  //! Drop the kept rows.
  void clear();

 private:
  struct piece;

 public:
  //! The kept rows as sorted tables on one GPU, all at one stride.
  struct device_keys {
    //! One table per batch that has key rows.
    std::vector<cudf::table_view> tables;
    std::size_t stride   = 1;
    std::size_t key_rows = 0;
    totals input;

    //! What @ref tables point into.
    std::vector<std::shared_ptr<piece const>> pieces;
    std::vector<std::unique_ptr<rmm::device_buffer>> buffers;
    std::vector<std::unique_ptr<cudf::table>> thinned;
  };

  //! The kept rows, copied to @p space's GPU.
  [[nodiscard]] device_keys on_device(cucascade::memory::memory_space& space,
                                      ::cuda::stream_ref stream) const;

 private:
  //! Rows kept at @p stride from every batch except @p skip_batch_id. Caller holds _mutex.
  [[nodiscard]] std::size_t kept_at(std::size_t stride, std::uint64_t const* skip_batch_id) const;

  //! Key rows held in host memory. Caller holds _mutex.
  [[nodiscard]] std::size_t held_key_rows() const;

  //! Thin the rows taken at a smaller stride than the current one until the held rows fit the cap.
  //! One caller thins at a time; the others return and leave the rows they added to it.
  void thin_held_rows(cucascade::memory::memory_space& space, ::cuda::stream_ref stream);

  std::size_t _max_key_rows;
  mutable std::mutex _mutex;
  std::unordered_map<std::uint64_t, std::shared_ptr<piece const>> _pieces;
  totals _seen;
  std::size_t _stride = 1;
  bool _thinning      = false;  //!< A caller is in thin_held_rows(). Guarded by _mutex.
  std::atomic<bool> _closed{false};
};

}  // namespace sirius::op
