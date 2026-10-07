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
  _cache->put(
    _scope, obj.object_path(), profile, obj.identity(), obj.open_generation(), std::move(metadata));
}
std::shared_ptr<io_object_metadata> metadata_store::get_metadata(io_object const& obj,
                                                                 std::string const& profile) const
{
  return _cache->get(_scope, obj.object_path(), profile, obj.identity());
}
bool metadata_store::has_candidate(std::string_view path) const
{
  return _cache->contains(_scope, std::string(path));
}
}  // namespace sirius::io::cache
