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

#include "vss/cuvs_index_cache.hpp"

#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <utility>

namespace sirius::vss {

namespace {

/// Fold metrics that produce the same reported distance in an IVF-Flat search
/// into one canonical value, so the auto-route match is on distance semantics
/// rather than the exact enum. The recognizer derives L2SqrtUnexpanded for
/// array_distance (avoids catastrophic cancellation on the brute-force path),
/// while an index is built L2SqrtExpanded for correct coarse cluster selection.
/// IVF-Flat's fine scan is unexpanded for both, so they return identical
/// distances and must match here.
cuvs::distance::DistanceType canonical_metric(cuvs::distance::DistanceType metric)
{
  switch (metric) {
    case cuvs::distance::DistanceType::L2SqrtExpanded:
    case cuvs::distance::DistanceType::L2SqrtUnexpanded:
      return cuvs::distance::DistanceType::L2SqrtExpanded;
    default: return metric;
  }
}

}  // namespace

cuvs_index_cache::cuvs_index_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager)
  : _reservation_manager(reservation_manager)
{
}

// Out-of-line so pinned_index_entry's reservation (a forward-declared type in
// the header) is destroyed here, where cucascade::memory::reservation is complete.
cuvs_index_cache::~cuvs_index_cache() = default;

std::unique_ptr<cucascade::memory::reservation> cuvs_index_cache::reserve_index_memory(
  std::size_t bytes, int preferred_gpu)
{
  using namespace cucascade::memory;
  if (preferred_gpu >= 0) {
    if (auto* space = _reservation_manager.get_memory_space(Tier::GPU, preferred_gpu)) {
      return space->make_reservation_or_null(bytes);
    }
  }
  return nullptr;
}

void cuvs_index_cache::insert(std::string name,
                              index_metadata meta,
                              std::unique_ptr<any_cuvs_index> index,
                              std::unique_ptr<cucascade::memory::reservation> reservation)
{
  auto entry         = std::make_shared<pinned_index_entry>();
  entry->meta        = std::move(meta);
  entry->index       = std::move(index);
  entry->reservation = std::move(reservation);

  std::lock_guard<std::mutex> lock(_mutex);
  _entries[std::move(name)] = std::move(entry);
}

std::shared_ptr<const pinned_index_entry> cuvs_index_cache::find(std::string_view name) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto it = _entries.find(std::string(name));
  return it == _entries.end() ? nullptr : it->second;
}

std::shared_ptr<const pinned_index_entry> cuvs_index_cache::find_by_column(
  std::string_view catalog,
  std::string_view schema,
  std::string_view table,
  std::string_view column,
  cuvs::distance::DistanceType metric) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto const wanted = canonical_metric(metric);
  for (auto const& kv : _entries) {
    auto const& entry = kv.second;
    if (entry->meta.catalog_name == catalog && entry->meta.schema_name == schema &&
        entry->meta.table_name == table && entry->meta.column_name == column &&
        canonical_metric(entry->meta.metric) == wanted) {
      return entry;
    }
  }
  return nullptr;
}

bool cuvs_index_cache::contains(std::string_view name) const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _entries.contains(std::string(name));
}

bool cuvs_index_cache::erase(std::string_view name)
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _entries.erase(std::string(name)) > 0;
}

std::size_t cuvs_index_cache::erase_by_column(std::string_view catalog,
                                              std::string_view schema,
                                              std::string_view table,
                                              std::string_view column,
                                              cuvs::distance::DistanceType metric)
{
  std::lock_guard<std::mutex> lock(_mutex);
  auto const wanted   = canonical_metric(metric);
  std::size_t removed = 0;
  for (auto it = _entries.begin(); it != _entries.end();) {
    auto const& meta = it->second->meta;
    if (meta.catalog_name == catalog && meta.schema_name == schema && meta.table_name == table &&
        meta.column_name == column && canonical_metric(meta.metric) == wanted) {
      it = _entries.erase(it);
      ++removed;
    } else {
      ++it;
    }
  }
  return removed;
}

void cuvs_index_cache::clear()
{
  std::lock_guard<std::mutex> lock(_mutex);
  _entries.clear();
}

std::size_t cuvs_index_cache::size() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _entries.size();
}

}  // namespace sirius::vss
