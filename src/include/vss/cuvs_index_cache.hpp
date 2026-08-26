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

#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace cucascade::memory {
class memory_reservation_manager;
class reservation;
}  // namespace cucascade::memory

namespace sirius::vss {

/// Which cuVS index algorithm a pinned entry holds. Drives how a search
/// operator down-casts the type-erased index back to its concrete cuVS type.
/// Only @c ivf_flat is built today; the rest are placeholders.
enum class index_kind : std::uint8_t {
  ivf_flat,
  // ivf_pq,  // not supported yet
  // cagra,  // not supported yet
};

/// Small, copyable description of a pinned cuVS index. Kept separate from the
/// (type-erased) index payload so the cache can be inspected, logged, and
/// matched without instantiating any cuVS index type.
///
/// The (@c catalog_name, @c schema_name, @c table_name, @c column_name, @c metric) tuple is
/// the index's auto-routing identity.
struct index_metadata {
  index_kind kind{index_kind::ivf_flat};
  std::string catalog_name;  ///< Resolved catalog the table lives in
  std::string schema_name;   ///< Resolved schema the table lives in
  std::string table_name;    ///< Base table the index was built on
  std::string column_name;   ///< Vector column the index was built on
  std::int64_t dim{0};       ///< Vector dimensionality
  std::int64_t num_rows{0};  ///< Number of indexed vectors
  std::int64_t n_lists{0};   ///< IVF-Flat inverted-list count (0 if not applicable).
  cuvs::distance::DistanceType metric{cuvs::distance::DistanceType::L2Expanded};
  std::size_t resident_bytes{
    0};  ///< Resident GPU bytes the finished index occupies (capacity-accounted)
};

/// Type-erased owner of a cuVS index. The cache stores indexes through
/// this base so it never has to template over the concrete cuVS index type
/// (@c cuvs::neighbors::ivf_flat::index, ...). The concrete index lives in
/// @ref cuvs_index_holder; a search operator recovers it with
/// @ref pinned_index_entry::index_as.
struct any_cuvs_index {
  any_cuvs_index()                                 = default;
  any_cuvs_index(const any_cuvs_index&)            = delete;  // copy not allowed
  any_cuvs_index& operator=(const any_cuvs_index&) = delete;  // assign not allowed
  virtual ~any_cuvs_index()                        = default;
};

/// Concrete holder owning one cuVS index of type @c Index.
template <class Index>
struct cuvs_index_holder final : any_cuvs_index {
  explicit cuvs_index_holder(Index&& idx) : index(std::move(idx)) {}
  Index index;
};

/// Wrap a freshly-built cuVS index in a type-erased holder ready for the cache.
template <class Index>
[[nodiscard]] std::unique_ptr<any_cuvs_index> make_cuvs_index(Index&& idx)
{
  return std::make_unique<cuvs_index_holder<std::decay_t<Index>>>(std::forward<Index>(idx));
}

/// One pinned cuVS index: its metadata, the type-erased index payload, and the
/// stream the index was built on.
///
/// The index's device memory is ordinary capacity-accounted GPU memory: it was
/// charged to a reservation only during the build, which was released once the
/// build finished. The entry keeps the build stream so the index can be freed on
/// it when the entry is dropped (unpin or session end). Move-only.
///
/// The cache stores entries as @c shared_ptr, and lookups hand back a shared
/// handle, so a caller that is mid-search keeps the whole entry alive even if it
/// is dropped or replaced concurrently.
struct pinned_index_entry {
  index_metadata meta;
  // The index's device buffers were built on this stream and are freed on it
  // when the index is destroyed. It is declared before the index because members
  // are destroyed in reverse order, and the index needs to be freed first.
  rmm::cuda_stream build_stream;
  std::unique_ptr<any_cuvs_index> index;

  /// Recover the concrete cuVS index, or nullptr if the held index is not of
  /// type @c Index. The returned pointer is owned by this entry.
  template <class Index>
  [[nodiscard]] Index* index_as() const noexcept
  {
    auto* holder = dynamic_cast<cuvs_index_holder<Index>*>(index.get());
    return holder ? &holder->index : nullptr;
  }
};

/// Session-lifetime registry of GPU-resident cuVS indexes, keyed by name.
///
/// This is the ANN analogue of the scan manager's pin-table cache: a named entry
/// pins a built cuVS index on the GPU so later searches reuse it instead of
/// rebuilding. It is deliberately a standalone cache (not an extension of the
/// pin-table cache) so each future index type can add its own search operator
/// without touching shared scan code.
///
/// Memory ownership: @ref reserve_index_memory reserves the build's footprint so
/// the caller can admit it against the GPU budget. The caller attaches that
/// reservation to the build stream for the build only, so cuVS's allocations are
/// charged and bounded by it, then releases the reservation. The finished index's
/// device memory stays as ordinary capacity-accounted GPU memory, and the entry
/// keeps only the build stream so the index can be freed on it later. Nothing here
/// ever allocates outside the GPU memory space's allocator.
///
/// Lifetime: entries live until @ref erase / @ref clear or session teardown.
/// There is no eviction or spilling yet (future work). Lookups return a shared
/// handle to the entry, so the entry (and its index) stays alive for as long as
/// any caller holds the handle, even if the named entry is erased or replaced in
/// the meantime.
///
/// Thread-safety: all members are guarded by an internal mutex. The mutex only
/// guards the map itself; the shared handle returned by a lookup is what keeps
/// the entry alive after the lock is released.
class cuvs_index_cache {
 public:
  explicit cuvs_index_cache(cucascade::memory::memory_reservation_manager& reservation_manager);
  ~cuvs_index_cache();

  cuvs_index_cache(const cuvs_index_cache&)            = delete;  // copy ctor not allowed
  cuvs_index_cache& operator=(const cuvs_index_cache&) = delete;  // copy assignment not allowed
  cuvs_index_cache(cuvs_index_cache&&)                 = delete;  // move ctor not allowed
  cuvs_index_cache& operator=(cuvs_index_cache&&)      = delete;  // move assignment not allowed

  /// Reserve @p bytes of GPU memory for building an index. The caller attaches
  /// this reservation to the build stream so cuVS's allocations are charged and
  /// bounded by it, then releases it once the index is built.
  ///
  /// Non-blocking, and pinned to @p preferred_gpu: the index must be reserved on
  /// the same GPU that holds the table's pinned data. Returns null instead of
  /// waiting, so the caller can fail cleanly rather than blocking indefinitely
  /// when the device cannot fit the index.
  ///
  /// \param bytes         Estimated GPU footprint of the index to build
  /// \param preferred_gpu Device id that holds the table's data (>= 0); null is
  ///                      returned for a negative id or an unknown device
  /// \returns The GPU reservation on @p preferred_gpu, or null if it cannot satisfy @p bytes
  [[nodiscard]] std::unique_ptr<cucascade::memory::reservation> reserve_index_memory(
    std::size_t bytes, int preferred_gpu = -1);

  /// Pin a built index under @p name, replacing any existing entry with that
  /// name. The old entry is unlinked here; its index is freed once no outstanding
  /// lookup handle still refers to it. Takes ownership of the index payload and
  /// the stream it was built on (the index is freed on that stream).
  void insert(std::string name,
              index_metadata meta,
              std::unique_ptr<any_cuvs_index> index,
              rmm::cuda_stream build_stream);

  /// Look up a pinned index by its management name, or nullptr if absent. The
  /// returned handle keeps the entry alive for as long as it is held, even if the
  /// entry is erased or replaced afterward (no eviction).
  [[nodiscard]] std::shared_ptr<const pinned_index_entry> find(std::string_view name) const;

  /// Find a pinned index by its auto-routing identity, i.e., the first entry whose
  /// metadata matches (@p catalog, @p schema, @p table, @p column, @p metric).
  /// This is the lookup a search recognizer uses to decide whether a query can use
  /// ANN. Returns nullptr if no pinned index covers that column under that metric.
  /// Metrics are compared up to canonicalization: L2SqrtExpanded/L2SqrtUnexpanded fold together,
  /// so a query's unexpanded metric still matches an index built expanded.
  ///
  /// Like @ref find, the returned handle keeps the entry alive while held.
  [[nodiscard]] std::shared_ptr<const pinned_index_entry> find_by_column(
    std::string_view catalog,
    std::string_view schema,
    std::string_view table,
    std::string_view column,
    cuvs::distance::DistanceType metric) const;

  /// Remove every entry whose auto-routing identity matches
  /// (@p catalog, @p schema, @p table, @p column, @p metric). Metrics are compared
  /// up to canonicalization. Returns the number of entries removed.
  std::size_t erase_by_column(std::string_view catalog,
                              std::string_view schema,
                              std::string_view table,
                              std::string_view column,
                              cuvs::distance::DistanceType metric);

  [[nodiscard]] bool contains(std::string_view name) const;

  /// Remove the entry for @p name. Its index and reservation are freed once no
  /// outstanding lookup handle still refers to it. Returns true iff an entry was
  /// removed.
  bool erase(std::string_view name);

  /// Drop all pinned indexes.
  void clear();

  [[nodiscard]] std::size_t size() const;

 private:
  cucascade::memory::memory_reservation_manager& _reservation_manager;
  mutable std::mutex _mutex;
  std::unordered_map<std::string, std::shared_ptr<pinned_index_entry>> _entries;
};

}  // namespace sirius::vss
