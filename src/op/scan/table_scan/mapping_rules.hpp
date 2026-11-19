/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include <cstdint>
#include <stdexcept>

namespace sirius::op::scan {
enum class rule_set_id : uint8_t { parquet_name_v1, iceberg_field_id_v1, iceberg_flat_v1 };
enum class refusal_rule : uint8_t {
  not_flat,
  untagged_field,
  invalid_or_duplicate_id,
  ambiguous_physical_name,
  unqualified_conversion,
  missing_typed_default,
  wrong_default_type,
  no_qualified_carrier,
  program_target_mismatch,
  unbound_generation,
  partition_constant_unproved
};
// Compatible with existing callers catching invalid_argument; the reason is
// produced at the actual check, never reconstructed from diagnostic text.
class mapping_error : public std::invalid_argument {
 public:
  mapping_error(refusal_rule rule, char const* text) : std::invalid_argument(text), rule_(rule) {}
  [[nodiscard]] refusal_rule rule() const noexcept { return rule_; }

 private:
  refusal_rule rule_;
};
}  // namespace sirius::op::scan
