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

#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace sirius::transparent {

class read_view_registry;

enum class candidate_origin : uint8_t { copy, replan };

struct original_binding {
  duckdb::idx_t table_index;
  std::shared_ptr<op::scan::bound_read_view const> view;
};

struct candidate_binding {
  duckdb::idx_t table_index;
  std::shared_ptr<op::scan::bound_read_view const> view;
};

struct read_view_comparison {
  bool equal = false;
  std::string correspondence;
  std::string reason;
  std::vector<std::string> only_original;
  std::vector<std::string> only_candidate;
  std::size_t original_count            = 0;
  std::size_t candidate_count           = 0;
  std::size_t different_original_count  = 0;
  std::size_t different_candidate_count = 0;
  std::optional<uint64_t> original_hash;
  std::optional<uint64_t> candidate_hash;
  std::optional<op::scan::evidence_depth> original_depth;
  std::optional<op::scan::evidence_depth> candidate_depth;
};

read_view_comparison compare_read_views(
  candidate_origin,
  std::span<original_binding const> logical_original,
  std::span<op::scan::read_view_fingerprint const> physical_original,
  std::span<candidate_binding const> candidate);
read_view_comparison compare_read_views(
  candidate_origin,
  op::scan::logical_bound_read_view_capture const* logical_original,
  std::span<op::scan::bound_read_view const> physical_original,
  read_view_registry const& candidate);
void share_equal_read_view_identities(op::scan::logical_bound_read_view_capture* logical_original,
                                      std::span<op::scan::bound_read_view> physical_original,
                                      read_view_registry const& candidate);
std::string describe_read_view_mismatch(read_view_comparison const&);

struct read_view_registry_entry {
  op::scan::bound_table_scan contract;
  op::scan::eligibility_certificate eligibility;
  std::shared_ptr<op::scan::file_evidence_arrays const> physical_evidence;
  std::optional<uint64_t> window_id;
  uint64_t finalize_generation = 0;
};

// Planning and finalize, including execution rebuilds, complete mutations before dispatch.
// Dispatcher threads only read published entries; mutation must not overlap those reads.
class read_view_registry {
 public:
  [[nodiscard]] read_view_registry_entry const& entry(op::scan::scan_contract_id id) const;
  [[nodiscard]] read_view_registry_entry const& entry_for_scan_node(uint64_t scan_node_id) const;
  [[nodiscard]] std::vector<read_view_registry_entry> const& entries() const noexcept
  {
    return entries_;
  }
  [[nodiscard]] std::vector<candidate_binding> candidate_bindings() const;
  void publish_supported(op::scan::certificate_evidence_scope scope,
                         std::string correspondence,
                         std::span<op::scan::bound_read_view const> physical_original);
  void inject_mismatch_for_testing(bool swap);

 private:
  friend op::scan::scan_contract_id op::scan::allocate_scan_contract(
    read_view_registry&,
    std::optional<uint64_t>,
    uint64_t,
    uint64_t,
    std::shared_ptr<op::scan::bound_read_view const>,
    op::scan::column_requirements,
    op::scan::predicate_contract,
    op::scan::materializer_contract_identity,
    duckdb::vector<duckdb::LogicalType>,
    duckdb::idx_t);

  std::vector<read_view_registry_entry> entries_;
  std::unordered_map<op::scan::scan_contract_id, std::size_t> by_contract_id_;
  std::unordered_map<uint64_t, std::size_t> by_scan_node_id_;
};

}  // namespace sirius::transparent
