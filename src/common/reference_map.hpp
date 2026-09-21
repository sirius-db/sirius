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

#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace sirius {

//! Hash by object identity (address of the referenced object).
template <class T>
struct ReferenceHashFunction {
  uint64_t operator()(const std::reference_wrapper<T>& ref) const
  {
    return std::hash<void*>()((void*)&ref.get());
  }
};

//! Equality by object identity.
template <class T>
struct ReferenceEquality {
  bool operator()(const std::reference_wrapper<T>& a, const std::reference_wrapper<T>& b) const
  {
    return &a.get() == &b.get();
  }
};

//! Unordered map keyed by reference identity. Drop-in replacement for duckdb::reference_map_t.
template <typename K, typename V>
using reference_map_t =
  std::unordered_map<std::reference_wrapper<K>, V, ReferenceHashFunction<K>, ReferenceEquality<K>>;

//! Unordered set keyed by reference identity. Drop-in replacement for duckdb::reference_set_t.
template <typename K>
using reference_set_t =
  std::unordered_set<std::reference_wrapper<K>, ReferenceHashFunction<K>, ReferenceEquality<K>>;

}  // namespace sirius
