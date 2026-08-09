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

#include <cudf/types.hpp>

#include <cstdint>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace sirius::op {

/**
 * @brief Exact host-side scalar for the Top-N type allowlist
 *
 * Carries the value and its cuDF storage type together so comparison and device-scalar
 * construction cannot disagree about representation. The variant covers the admitted types'
 * *physical representations*, which is not the same as the allowlist: a fixed-point key is held as
 * its scaled integer in the matching integer alternative, and `storage_type` is what distinguishes
 * `DECIMAL64` from `INT64` and carries the scale. Widening the allowlist therefore widens the
 * variant only when the new type introduces a representation none of the alternatives already
 * holds -- `DECIMAL128` is the next such case, and it needs
 * `detail::boundary_filter_params::component::value` and the kernel's width switch widened in the
 * same change, not just this variant. Immutable after construction; freely copyable; no device
 * state.
 */
class exact_host_scalar final {
 public:
  using value_type = std::variant<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;

  exact_host_scalar(value_type value, cudf::data_type storage_type) noexcept
    : _value(value), _storage_type(storage_type)
  {
  }

  [[nodiscard]] value_type const& value() const noexcept { return _value; }
  [[nodiscard]] cudf::data_type storage_type() const noexcept { return _storage_type; }

  /**
   * @brief Three-way comparison in the given SQL order
   *
   * @pre Both operands share one storage type; the coordinator guarantees this by construction.
   * @return Negative/zero/positive when `*this` orders before/equal-to/after @p other under
   * @p order. Null ordering is not this class's concern -- a null boundary never constructs one.
   */
  [[nodiscard]] int compare(exact_host_scalar const& other, cudf::order order) const noexcept;

 private:
  value_type _value;
  cudf::data_type _storage_type;
};

/**
 * @brief One ORDER BY key's frozen semantics
 *
 * `null_order` is the exact value fed to cuDF sorting (the output of `to_cudf_null_order`), so it
 * carries cuDF's contract: null precedence applies in the ascending frame and the whole per-key
 * verdict is reversed for a DESCENDING key.
 */
struct top_n_key_semantics {
  cudf::data_type storage_type;
  cudf::order order;
  cudf::null_order null_order;
};

/**
 * @brief The exact host boundary tuple: one optional component per ORDER BY key
 *
 * A disengaged component records that the boundary row's key is null there -- legal in tail
 * positions, and the reason the type is not `std::vector<exact_host_scalar>`. Immutable and
 * copyable, like its component type.
 */
class exact_host_key_tuple final {
 public:
  explicit exact_host_key_tuple(std::vector<std::optional<exact_host_scalar>> components)
    : _components(std::move(components))
  {
  }

  [[nodiscard]] std::size_t size() const noexcept { return _components.size(); }
  [[nodiscard]] std::optional<exact_host_scalar> const& component(std::size_t i) const
  {
    return _components.at(i);
  }

  /**
   * @brief Lexicographic three-way comparison under per-key semantics
   *
   * Honors each key's direction and null placement; null components order per `null_order`,
   * mirroring cuDF's row comparator (ascending-frame null precedence, verdict reversed for
   * DESCENDING).
   *
   * @pre `semantics.size() == size()` and both tuples share component storage types.
   */
  [[nodiscard]] int lex_compare(exact_host_key_tuple const& other,
                                std::span<top_n_key_semantics const> semantics) const noexcept;

 private:
  std::vector<std::optional<exact_host_scalar>> _components;
};

}  // namespace sirius::op
