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
#include "op/scan/table_scan/scan_contract.hpp"

namespace sirius::op::scan {
struct split_validation_context {
  scan_contract_id contract;
  later_check_set required;
  std::optional<key_held_witness> const& key;
};
struct split_validation_evidence {
  // One requirement set per certificate. Format-specific applicability (native
  // insert deltas, for example) is determined by the trusted validator.
  std::vector<later_check_set> coverage;
  // Legacy formats use the unique scan contract as their generation witness.
  // Owned materialized inputs additionally carry an explicit program generation.
  std::optional<uint64_t> generation;
};
class split_validator {
 public:
  virtual ~split_validator()                                        = default;
  virtual bool accepts(scan_info const&) const noexcept             = 0;
  virtual split_validation_evidence inspect(split_validation_context const&,
                                            scan_info const&) const = 0;
};
std::shared_ptr<split_validator const> parquet_split_validator();
std::shared_ptr<split_validator const> native_split_validator();
std::shared_ptr<split_validator const> combine_split_validators(
  std::vector<std::shared_ptr<split_validator const>>);
std::shared_ptr<split_validator const> default_split_validator(materializer_kind);
// Keep contract mismatches observable even when registry lookup is unavailable.
void validate_split_contract(scan_contract_id, scan_info const&);
// Common checks cannot be replaced by a provider's validator. The caller supplies
// the policy, generation and profile owner from its immutable bound registry entry.
void validate_split_for_gpu(scan_contract_id,
                            later_check_set const&,
                            std::optional<key_held_witness> const&,
                            scan_info const&,
                            split_validator const&,
                            uint64_t generation,
                            std::shared_ptr<physical_profile_table> const& profiles);
}  // namespace sirius::op::scan
