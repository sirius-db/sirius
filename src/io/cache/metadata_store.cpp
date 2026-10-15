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
                                       std::shared_ptr<io_object_metadata> metadata)
{
  if (!metadata) return;
  auto const& key  = obj.raw_file_cache_id();
  auto const& path = obj.object_path();
  std::unique_lock lk(_mtx);
  if (auto it = _key_by_path.find(path); it != _key_by_path.end() && it->second != key) {
    _by_key.erase(it->second);
    it->second = key;
  } else if (it == _key_by_path.end()) {
    _key_by_path.emplace(path, key);
  }
  _by_key[key] = std::move(metadata);
}

bool metadata_store::has_path(std::string_view object_path) const noexcept
{
  std::shared_lock lk(_mtx);
  return _key_by_path.contains(object_path);
}

std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj) const
{
  return get_metadata(obj.raw_file_cache_id());
}

std::shared_ptr<io_object_metadata> metadata_store::get_metadata(std::string_view cache_key) const
{
  std::shared_lock lk(_mtx);
  auto it = _by_key.find(cache_key);
  if (it == _by_key.end()) return nullptr;
  return it->second;
}

}  // namespace sirius::io::cache
