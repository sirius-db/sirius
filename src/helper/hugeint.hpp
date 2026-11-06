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

#include <duckdb/common/hugeint.hpp>

#include <bit>
#include <type_traits>

namespace sirius {

// duckdb::hugeint_t is {uint64_t lower; int64_t upper;}, the little-endian layout of __int128_t.
// The GPU's HUGEINT carrier and the result reader's direct copy rely on this as well.
static_assert(sizeof(duckdb::hugeint_t) == sizeof(__int128_t) &&
              std::is_trivially_copyable_v<duckdb::hugeint_t>);

constexpr __int128_t to_int128(duckdb::hugeint_t value) { return std::bit_cast<__int128_t>(value); }

constexpr duckdb::hugeint_t to_hugeint(__int128_t value)
{
  return std::bit_cast<duckdb::hugeint_t>(value);
}

}  // namespace sirius
