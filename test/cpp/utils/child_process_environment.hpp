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

#if defined(__APPLE__)
#include <crt_externs.h>
#else
extern char** environ;
#endif

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::test {

/** An inherited environment with child-only overrides, suitable for execve/posix_spawn. */
class child_process_environment {
 public:
  using override = std::pair<std::string, std::string>;

  explicit child_process_environment(std::vector<override> overrides,
                                     std::vector<std::string> removals = {})
  {
#if defined(__APPLE__)
    auto const current_environment = *_NSGetEnviron();
#else
    auto const current_environment = environ;
#endif

    auto const is_replaced_or_removed = [&](std::string_view name) {
      auto const is_override = std::any_of(
        overrides.begin(), overrides.end(), [&](auto const& item) { return item.first == name; });
      auto const is_removal = std::any_of(
        removals.begin(), removals.end(), [&](auto const& item) { return item == name; });
      return is_override || is_removal;
    };

    for (auto entry = current_environment; entry != nullptr && *entry != nullptr; ++entry) {
      std::string_view const value{*entry};
      auto const separator = value.find('=');
      auto const name      = value.substr(0, separator);
      if (!is_replaced_or_removed(name)) { entries_.emplace_back(value); }
    }

    for (auto& [name, value] : overrides) {
      entries_.push_back(std::move(name) + "=" + std::move(value));
    }

    envp_.reserve(entries_.size() + 1);
    for (auto& entry : entries_) {
      envp_.push_back(entry.data());
    }
    envp_.push_back(nullptr);
  }

  child_process_environment(child_process_environment const&)            = delete;
  child_process_environment& operator=(child_process_environment const&) = delete;
  child_process_environment(child_process_environment&&)                 = delete;
  child_process_environment& operator=(child_process_environment&&)      = delete;

  [[nodiscard]] char* const* data() noexcept { return envp_.data(); }

 private:
  std::vector<std::string> entries_;
  std::vector<char*> envp_;
};

}  // namespace sirius::test
