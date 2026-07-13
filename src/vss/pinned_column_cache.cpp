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

#include "vss/pinned_column_cache.hpp"

#include <cudf/column/column.hpp>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>

#include <rmm/cuda_device.hpp>

#include <utility>

namespace sirius::vss {

namespace {
std::string make_key(std::string_view table, std::string_view column)
{
  std::string key;
  key.reserve(table.size() + column.size() + 1);
  key.append(table);
  key.push_back('\0');
  key.append(column);
  return key;
}
}  // namespace

pinned_column_cache::pinned_column_cache(
  cucascade::memory::memory_reservation_manager& reservation_manager)
  : reservation_manager_(reservation_manager)
{
}

// Out-of-line so the cached columns (whose deleters hold cucascade::memory::
// reservation, forward-declared in the header) are destroyed here.
pinned_column_cache::~pinned_column_cache() = default;

rmm::cuda_stream_view pinned_column_cache::stream_for_device(int device_id)
{
  auto it = build_streams_.find(device_id);
  if (it == build_streams_.end()) {
    rmm::cuda_set_device_raii guard{rmm::cuda_device_id{device_id}};
    it = build_streams_.try_emplace(device_id).first;
  }
  return it->second.view();
}

std::shared_ptr<cudf::column> pinned_column_cache::get_or_build(
  const std::string& table,
  const std::string& column,
  std::size_t estimated_bytes,
  int preferred_gpu,
  const std::function<std::unique_ptr<cudf::column>(rmm::device_async_resource_ref,
                                                    rmm::cuda_stream_view)>& build)
{
  auto const key = make_key(table, column);
  {
    // Fast path, cache hit
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = columns_.find(key);
    if (it != columns_.end()) { return it->second; }
  }

  // Reserve the column's GPU footprint so it participates in Sirius's memory
  // budget, then build straight into the reservation's resource.
  using namespace cucascade::memory;
  std::shared_ptr<reservation> resv   = preferred_gpu >= 0
                                          ? reservation_manager_.request_reservation(
                                            any_memory_space_in_tier_with_preference(
                                              Tier::GPU, static_cast<std::size_t>(preferred_gpu)),
                                            estimated_bytes)
                                          : reservation_manager_.request_reservation(
                                            any_memory_space_in_tier(Tier::GPU), estimated_bytes);

  // Build on a cache-owned stream (per device) so the column's buffers free
  // themselves on a stream that lives as long as the cache, not the caller.
  int const build_device =
    preferred_gpu >= 0 ? preferred_gpu : rmm::get_current_cuda_device().value();
  rmm::cuda_stream_view build_stream;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    build_stream = stream_for_device(build_device);
  }

  std::unique_ptr<cudf::column> built = build(resv->get_memory_resource(), build_stream);
  // estimated_bytes over-estimates (concat merges offsets/masks), so return the
  // slack to the budget now that the real footprint is allocated.
  resv->shrink_to_fit();
  // The column will be shared and read on other (caller) streams, so make sure
  // the coalesce is complete before we hand it out.
  build_stream.synchronize();

  // Hand back a shared column whose deleter keeps the reservation alive: the device buffers
  // are freed first (deleter runs), then the captured reservation is released.
  std::shared_ptr<cudf::column> col(built.release(), [resv](cudf::column* p) mutable {
    delete p;
    resv.reset();
  });

  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = columns_.try_emplace(key, std::move(col));
  return it->second;
}

void pinned_column_cache::erase_table(std::string_view table)
{
  auto const prefix = make_key(table, "");  // "table\0"
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto it = columns_.begin(); it != columns_.end();) {
    if (it->first.starts_with(prefix)) {
      it = columns_.erase(it);
    } else {
      ++it;
    }
  }
}

void pinned_column_cache::clear()
{
  std::lock_guard<std::mutex> lock(mutex_);
  columns_.clear();
}

}  // namespace sirius::vss
