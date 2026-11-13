/*
 * Copyright 2025, Sirius Contributors.
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
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>
#include <span>

namespace sirius::op::merge::detail {

/// Device view of a kway_key_comparator; defined in detail/kway_comparator.cuh (CUDA sources only).
struct kway_key_device_view;

/**
 * @brief Host owner of the device state needed to compare rows across k key windows (pimpl).
 *
 * Move-only. create() does not wait on the stream: its device state is filled by a copy queued on
 * that stream, which reads host memory the comparator owns. The comparator must therefore stay
 * alive until the stream has run that copy and every kernel that uses its device view (in
 * practice: until the next synchronize of the stream), and its device view must only be used on
 * the stream it was created with.
 */
class kway_key_comparator {
 public:
  /**
   * @brief Build the comparator for k key windows.
   *
   * @param key_windows key_windows[i] holds only the key columns of window i, in key order.
   * @param column_order One entry per key column.
   * @param null_precedence One entry per key column.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used for the comparator's device state.
   */
  static kway_key_comparator create(std::span<const cudf::table_view> key_windows,
                                    std::span<const cudf::order> column_order,
                                    std::span<const cudf::null_order> null_precedence,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

  kway_key_comparator(kway_key_comparator&&) noexcept;
  kway_key_comparator& operator=(kway_key_comparator&&) noexcept;
  ~kway_key_comparator();

  /// Number of key windows (k) the comparator was created with.
  [[nodiscard]] int32_t num_inputs() const noexcept;

  /// Device state; defined by the comparator's CUDA source.
  struct impl;

 private:
  explicit kway_key_comparator(std::unique_ptr<impl>);
  std::unique_ptr<impl> _impl;
  friend struct kway_key_device_view device_view_of(const kway_key_comparator&);
};

/**
 * @brief Multi-sequence selection ("co-rank") of the k windows at each requested rank.
 *
 * out[r * k + i] = c_i(ranks[r]): per-input counts with Σ_i c_i = ranks[r] such that every
 * selected row orders before (by the comparator, incl. tie-break) every unselected row.
 *
 * @param cmp Comparator over the k key windows.
 * @param lengths Row count of each window (k entries).
 * @param ranks The ranks to split at; each in [0, Σ lengths].
 * @param out Receives ranks.size() * k counts.
 * @param stream CUDA stream used for kernel launches.
 */
void corank_split(const kway_key_comparator& cmp,
                  cudf::device_span<const cudf::size_type> lengths,
                  cudf::device_span<const cudf::size_type> ranks,
                  cudf::device_span<cudf::size_type> out,
                  rmm::cuda_stream_view stream);

/**
 * @brief target = min(max_rows, r_safe, Σ lengths) as defined on bounded_kway_merge.
 *
 * Returns it on the host (syncs). A truncated window with no rows makes it 0: the caller must
 * refill such a window before the round.
 *
 * @param cmp Comparator over the k key windows.
 * @param lengths Row count of each window (k entries).
 * @param truncated One flag per window: non-zero when its source run has unread rows beyond it.
 * @param max_rows Upper bound on the rows emitted.
 * @param stream CUDA stream used for kernel launches.
 */
cudf::size_type safe_target(const kway_key_comparator& cmp,
                            cudf::device_span<const cudf::size_type> lengths,
                            cudf::device_span<const std::uint8_t> truncated,
                            cudf::size_type max_rows,
                            rmm::cuda_stream_view stream);

/// Default output rows per tile of tile_merge.
inline constexpr cudf::size_type kDefaultTileRows = 2048;

/**
 * @brief Merge each output tile independently, producing an (input, row) gather map.
 *
 * For tile t in [0, tiles), tile_coranks[t*k .. t*k+k) and [(t+1)*k ..) bound its sub-ranges;
 * writes the merged order of output rows [t*tile_rows, min((t+1)*tile_rows, target)) as
 * (input, row) pairs. tiles = ceil(target / tile_rows).
 *
 * @param cmp Comparator over the k key windows.
 * @param lengths Row count of each window (k entries).
 * @param tile_coranks (tiles + 1) * k co-ranks: the corank_split of ranks 0, tile_rows, ...,
 *        target.
 * @param target Total output rows.
 * @param tile_rows Output rows per tile (the last tile may be shorter).
 * @param out_input Receives the source window of each output row (target entries).
 * @param out_row Receives the row within that window of each output row (target entries).
 * @param stream CUDA stream used for kernel launches.
 */
void tile_merge(const kway_key_comparator& cmp,
                cudf::device_span<const cudf::size_type> lengths,
                cudf::device_span<const cudf::size_type> tile_coranks,
                cudf::size_type target,
                cudf::size_type tile_rows,
                cudf::device_span<std::int32_t> out_input,
                cudf::device_span<cudf::size_type> out_row,
                rmm::cuda_stream_view stream);

/// How tile_merge launches its kernel for one round.
struct tile_merge_shape {
  std::int32_t sub_tiles;      ///< threads per tile, each merging sub_rows output rows
  cudf::size_type sub_rows;    ///< output rows per sub-tile
  std::int32_t block_tiles;    ///< tiles per block
  std::int32_t block_threads;  ///< block_tiles * sub_tiles
  std::uint32_t blocks;        ///< ceil(tiles / block_tiles)
  std::size_t shared_bytes;    ///< shared memory per block
};

/**
 * @brief The launch shape of tile_merge: a host computation, exposed for tests.
 *
 * A block's shared memory is at most 40 KiB, and at most 32 KiB when there are more blocks than
 * SMs (a single thread at a fan-in above 2048 needs more than that and keeps it).
 *
 * @param tiles Tiles of the round, at least 1.
 * @param tile_rows Output rows per tile, at least 1.
 * @param num_inputs Fan-in k, in [1, kMaxKwayMergeInputs].
 * @param sm_count SMs of the device.
 * @param requested_sub_tiles Sub-tiles per tile instead of the count chosen from the device
 *        (0: choose); capped like the chosen count.
 */
tile_merge_shape tile_merge_launch_shape(cudf::size_type tiles,
                                         cudf::size_type tile_rows,
                                         std::int32_t num_inputs,
                                         int sm_count,
                                         std::int32_t requested_sub_tiles = 0);

/// A small device array that gather_kway copies to the host in the same wait on the stream as
/// its own counts. `source` and `destination` have the same number of entries.
struct gather_read_back {
  cudf::device_span<const cudf::size_type> source;
  std::span<cudf::size_type> destination;
};

/**
 * @brief Gather rows from k windows into one table.
 *
 * output row j = windows[src_input[j]].row(src_row[j]); all columns incl. null masks and STRING.
 * The null counts of the output are exact; those of the windows are not read (a row is null when
 * its bit in the window's null mask is clear).
 *
 * Waits on the stream once to read back the null counts and STRING chars sizes of all columns,
 * and once more when there is a STRING column, whatever the number of columns and windows, plus
 * once per STRING column whose chars reach the large-strings threshold (its offsets are rebuilt).
 * Everything queued on @p stream has run when it returns.
 *
 * @param windows The k windows; all share one schema.
 * @param src_input Source window of each output row.
 * @param src_row Row within the source window of each output row.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the output table.
 * @param read_back Optional device array to copy to the host in the first wait.
 */
std::unique_ptr<cudf::table> gather_kway(std::span<const cudf::table_view> windows,
                                         cudf::device_span<const std::int32_t> src_input,
                                         cudf::device_span<const cudf::size_type> src_row,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr,
                                         gather_read_back read_back = {});

/// Test-only: out[i] = less(ia[i], ra[i], ib[i], rb[i]) of the comparator's device view. All
/// spans have the same number of entries. Does not wait on the stream.
void compare_pairs_for_test(const kway_key_comparator& cmp,
                            cudf::device_span<const int32_t> ia,
                            cudf::device_span<const cudf::size_type> ra,
                            cudf::device_span<const int32_t> ib,
                            cudf::device_span<const cudf::size_type> rb,
                            cudf::device_span<uint8_t> out,
                            rmm::cuda_stream_view stream);

/// Test-only: tile_merge with @p sub_tiles sub-tiles per tile instead of the count chosen from
/// the device (still capped as in tile_merge).
void tile_merge_with_sub_tiles_for_test(const kway_key_comparator& cmp,
                                        cudf::device_span<const cudf::size_type> lengths,
                                        cudf::device_span<const cudf::size_type> tile_coranks,
                                        cudf::size_type target,
                                        cudf::size_type tile_rows,
                                        cudf::device_span<std::int32_t> out_input,
                                        cudf::device_span<cudf::size_type> out_row,
                                        std::int32_t sub_tiles,
                                        rmm::cuda_stream_view stream);

}  // namespace sirius::op::merge::detail
