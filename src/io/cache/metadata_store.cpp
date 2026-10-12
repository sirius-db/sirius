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

#include "io/cache/metadata_store.hpp"

#include <utility>

namespace sirius::io::cache {

void metadata_store::register_metadata(io_object const& obj,
                                       std::shared_ptr<io_object_metadata> metadata,
                                       std::string const& profile)
{
  if (!metadata) return;
  auto legacy_metadata = metadata;
  _cache.put(
    _scope, obj.object_path(), profile, obj.identity(), obj.open_generation(), std::move(metadata));
  if (profile != "parquet-v1") return;

  auto const& path = obj.object_path();
  auto const& key  = obj.raw_file_cache_id();
  std::lock_guard lock{_legacy_mutex};
  if (auto path_it = _legacy_key_by_path.find(path);
      path_it != _legacy_key_by_path.end() && path_it->second != key) {
    _legacy_by_key.erase(path_it->second);
    path_it->second = key;
  } else if (path_it == _legacy_key_by_path.end()) {
    _legacy_key_by_path.emplace(path, key);
  }
  auto const current = _legacy_by_key.find(key);
  if (current == _legacy_by_key.end() || obj.open_generation() >= current->second.generation) {
    _legacy_by_key.insert_or_assign(
      key, legacy_entry{obj.open_generation(), std::move(legacy_metadata)});
  }
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj,
                                                                 std::string const& profile) const
{
  return _cache.get(_scope, obj.object_path(), profile, obj.identity());
}
std::unique_lock<std::mutex> metadata_store::lock_metadata_miss(io_object const& obj,
                                                                std::string const& profile) const
{
  return _cache.lock_miss(_scope, obj.object_path(), profile);
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(std::string_view cache_key) const
{
  std::lock_guard lock{_legacy_mutex};
  auto const it = _legacy_by_key.find(std::string(cache_key));
  return it == _legacy_by_key.end() ? nullptr : it->second.value;
}
bool metadata_store::has_candidate(std::string_view path) const
{
  return _cache.contains(_scope, std::string(path));
}
}  // namespace sirius::io::cache
