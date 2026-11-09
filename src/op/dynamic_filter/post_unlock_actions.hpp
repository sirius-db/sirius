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

#pragma once

#include "op/dynamic_filter/dynamic_filter_key_outcome.hpp"

#include <vector>

namespace sirius::op {

/**
 * @brief Work that `dynamic_filter_publication_session` decides while holding its mutex but must
 * run only after releasing it
 *
 * A session produces one value from every path that can make it terminal, while it holds its mutex;
 * producing or moving a value never allocates. The public session entry that holds the value calls
 * `dispatch` once the mutex is released, so no logging runs under the session lock. Today a value
 * carries only the session's pending `[dynamic_filter_key]` lines.
 *
 * The value is move-only and cannot be move-assigned, so pending work is merged with `append`
 * rather than overwritten. Destroying a value that still holds work is a bug: debug builds assert,
 * because every key must produce exactly one line.
 */
class [[nodiscard]] post_unlock_actions final {
 public:
  post_unlock_actions() noexcept = default;

  /**
   * @brief Takes ownership of @p key_lines, to be logged by `dispatch`
   */
  explicit post_unlock_actions(std::vector<dynamic_filter_key_line> key_lines) noexcept;

  post_unlock_actions(post_unlock_actions&& other) noexcept  = default;
  post_unlock_actions& operator=(post_unlock_actions&&)      = delete;
  post_unlock_actions(post_unlock_actions const&)            = delete;
  post_unlock_actions& operator=(post_unlock_actions const&) = delete;

  ~post_unlock_actions() noexcept;

  /**
   * @brief Moves @p other's work after this value's work
   *
   * Allocates only when both values hold work.
   */
  void append(post_unlock_actions&& other);

  /**
   * @brief Runs and clears every action: logs each pending key line at DEBUG level, in order
   *
   * Call without holding the session mutex.
   */
  void dispatch() && noexcept;

  [[nodiscard]] bool empty() const noexcept { return _key_lines.empty(); }

 private:
  std::vector<dynamic_filter_key_line> _key_lines;
};

}  // namespace sirius::op
