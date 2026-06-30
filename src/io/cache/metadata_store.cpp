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

void metadata_store::register_metadata(sirius_io_object const& obj,
                                       std::shared_ptr<sirius_io_object_metadata> metadata)
{
  if (!metadata) return;
  auto const& key = obj.raw_file_cache_id();
  std::unique_lock lk(_mtx);
  _by_key[key] = std::move(metadata);
}

std::shared_ptr<sirius_io_object_metadata> metadata_store::get_metadata(
  sirius_io_object const& obj) const
{
  auto const& key = obj.raw_file_cache_id();
  std::shared_lock lk(_mtx);
  auto it = _by_key.find(key);
  if (it == _by_key.end()) return nullptr;
  return it->second;
}

}  // namespace sirius::io::cache
