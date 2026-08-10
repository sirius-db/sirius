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

#include <cstdlib>
#include <string_view>

namespace sirius::test {

inline constexpr char child_runner_env[]       = "SIRIUS_INTERNAL_CHILD_RUNNER";
inline constexpr char child_runner_env_value[] = "1";

inline bool is_child_runner()
{
  auto const* value = std::getenv(child_runner_env);
  return value != nullptr && std::string_view{value} == child_runner_env_value;
}

inline bool mark_child_runner()
{
  return ::setenv(child_runner_env, child_runner_env_value, 1) == 0;
}

}  // namespace sirius::test
