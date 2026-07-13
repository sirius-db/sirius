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

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace cudf {
class column;
}  // namespace cudf

namespace cucascade::memory {
class memory_reservation_manager;
}  // namespace cucascade::memory

namespace sirius::vss {

/// Session cache of *contiguous* GPU columns derived from a pinned table's
/// chunked storage. A pinned column is stored as many GPU chunks (one per scan
/// batch), but cuVS search and cudf::gather need one contiguous column. This
/// caches that coalesce, keyed by (table, column), so repeated searches over the
/// same pinned table reuse it.
///
/// Lifetime: entries hold GPU memory until @ref erase_table / @ref clear or
/// session teardown; they must be dropped when a table is unpinned/re-pinned so a
/// stale coalesce can't outlive the chunks it was built from.
///
/// Each cached column is built through a GPU reservation from the reservation
/// manager, so its footprint is accounted for in Sirius's memory budget; the
/// reservation is held alive by the returned column's deleter and released only
/// once the last holder drops it.
///
/// NOTE: the cache still copies the pinned chunks into a contiguous column
/// (roughly doubling the footprint of accessed columns).
///
/// Thread-safety: guarded by an internal mutex.
class pinned_column_cache {
 public:
  explicit pinned_column_cache(cucascade::memory::memory_reservation_manager& reservation_manager);
  ~pinned_column_cache();

  pinned_column_cache(const pinned_column_cache&)            = delete;
  pinned_column_cache& operator=(const pinned_column_cache&) = delete;
  pinned_column_cache(pinned_column_cache&&)                 = delete;
  pinned_column_cache& operator=(pinned_column_cache&&)      = delete;

  /// Return the coalesced column for (@p table, @p column), building it on a miss
  /// and caching the result. On a miss, @p estimated_bytes of GPU memory is
  /// reserved (on @p preferred_gpu, or any GPU if < 0) and @p build is invoked
  /// with the reservation's memory resource and a cache-owned stream so the
  /// column allocates against them. The stream outlives the cache's columns, so
  /// their async frees at erase/teardown run on a live stream. The returned
  /// column is shared with the cache; callers read it via .view() and must not
  /// mutate it.
  [[nodiscard]] std::shared_ptr<cudf::column> get_or_build(
    const std::string& table,
    const std::string& column,
    std::size_t estimated_bytes,
    int preferred_gpu,
    const std::function<std::unique_ptr<cudf::column>(rmm::device_async_resource_ref,
                                                      rmm::cuda_stream_view)>& build);

  /// Drop every cached column belonging to @p table (call on unpin/re-pin).
  void erase_table(std::string_view table);

  /// Drop all cached columns.
  void clear();

 private:
  // Lazily create (or fetch) the durable build/free stream for @p device_id.
  // Caller must hold mutex_.
  rmm::cuda_stream_view stream_for_device(int device_id);

  cucascade::memory::memory_reservation_manager& reservation_manager_;
  mutable std::mutex mutex_;
  // Per-device streams cached columns are allocated on and freed on. Declared
  // before columns_ so it is destroyed *after* them: the columns' async frees
  // must run on a live stream.
  std::unordered_map<int, rmm::cuda_stream> build_streams_;
  // Key is table + '\0' + column so table names and column names can't collide.
  // Each column's deleter keeps its GPU reservation alive.
  std::unordered_map<std::string, std::shared_ptr<cudf::column>> columns_;
};

}  // namespace sirius::vss
