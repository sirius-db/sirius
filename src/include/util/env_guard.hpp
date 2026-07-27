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

#include <optional>
#include <string>

namespace sirius {
namespace util {

/**
 * @brief RAII guard for a process environment variable.
 *
 * On construction the variable named @p name is set to @p value (overwriting any
 * existing value). On destruction the variable is restored to whatever state it
 * had before this guard was constructed: its previous value if it was set, or
 * unset if it was not previously present.
 *
 * Move-only. A moved-from guard is inert and restores nothing.
 *
 * @note This wraps ::setenv / ::unsetenv, which are not thread-safe with respect
 *       to concurrent getenv/setenv calls. Construct/destroy env_guards from a
 *       single thread (e.g. during setup/teardown), not while other threads may
 *       be reading the environment.
 */
class env_guard {
 public:
  /// \brief Set \p name to \p value, remembering the prior state for restoration.
  env_guard(std::string name, const std::string& value);

  /// \brief Restore \p name to its prior state (previous value, or unset).
  ~env_guard();

  env_guard(env_guard&& other) noexcept;
  env_guard& operator=(env_guard&& other) noexcept;

  env_guard(const env_guard&)            = delete;
  env_guard& operator=(const env_guard&) = delete;

 private:
  void restore() noexcept;

  std::string name_;
  std::optional<std::string> previous_value_;  ///< Prior value; nullopt if it was unset.
  bool active_ = false;                        ///< False once moved-from; suppresses restore.
};

}  // namespace util
}  // namespace sirius
