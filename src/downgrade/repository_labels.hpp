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

#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>

namespace sirius::downgrade {

namespace detail {
struct repository_labels {
  std::mutex mutex;
  std::unordered_map<void const*, std::string> names;
};
inline repository_labels& labels()
{
  static repository_labels instance;
  return instance;
}
}  // namespace detail

/// Names a data repository by the operator and port that read it, for the spill log. Entries are
/// never erased: a repository allocated at a reused address is labelled again when it is wired.
inline void label_repository(void const* repository, std::string name)
{
  auto& l = detail::labels();
  std::lock_guard<std::mutex> guard(l.mutex);
  l.names[repository] = std::move(name);
}

inline std::string repository_label(void const* repository)
{
  auto& l = detail::labels();
  std::lock_guard<std::mutex> guard(l.mutex);
  auto const it = l.names.find(repository);
  return it == l.names.end() ? std::string{"unlabelled"} : it->second;
}

}  // namespace sirius::downgrade
