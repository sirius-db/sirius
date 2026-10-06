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

#include <rmm/error.hpp>

#include <cucascade/memory/memory_space.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace sirius::vss {

/// For a staging reservation that came back null. rmm::out_of_memory when @p bytes could fit
/// @p space at all: the task is then retried after a downgrade frees memory. A plain error when
/// it never could, so a request no retry can satisfy fails at once rather than after every retry.
[[noreturn]] inline void throw_staging_shortfall(cucascade::memory::memory_space const& space,
                                                 std::size_t bytes,
                                                 std::string const& what)
{
  auto const limit = space.get_max_memory();
  auto const msg   = what + " needs " + std::to_string(bytes) + " bytes device-side";
  if (bytes <= limit) { throw rmm::out_of_memory(msg); }
  throw std::runtime_error(msg + ", more than the device memory space's " + std::to_string(limit));
}

}  // namespace sirius::vss
