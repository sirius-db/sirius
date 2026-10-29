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
#include "ducklake/scan_options.hpp"
#include "ducklake/unit_descriptor.hpp"
#include "op/scan/table_scan/bound_read_view.hpp"

#include <duckdb/function/table_function.hpp>

namespace sirius::op::scan {
class arrow_import_ingestible;
}
namespace sirius::scan_manager {
class preparation_admission;
class charging_allocator;
class preparation_stop_token;
}  // namespace sirius::scan_manager
namespace sirius::ducklake {
class parquet_unit_ingestible;
class inline_reader;
struct dv_visibility;
enum class dv_snapshot_policy;
// Only the implementation may include provider-private headers. Availability
// permits source verification; each scan still requires contract certification.
bool bridge_available() noexcept;
duckdb::vector<duckdb::TableFunction> trusted_scan_functions(
  duckdb::DatabaseInstance&, duckdb::TableFunction const& trusted_parquet);
bool binding_matches(duckdb::BaseTableFunction const&,
                     duckdb::FunctionData const*,
                     duckdb::ClientContext&);
class binding_handle : public std::enable_shared_from_this<binding_handle> {
 public:
  static std::shared_ptr<binding_handle> capture(duckdb::BoundTableFunction const&,
                                                 duckdb::FunctionData const*,
                                                 duckdb::ClientContext&);
  ~binding_handle();
  // Internal per-unit capability; callers must account for every original unit.
  // Whole-query lowering must use the complete bound scan.
  // count_only means no user fields; it cannot be combined with required_columns.
  std::shared_ptr<parquet_unit_ingestible> make_parquet_unit(
    uint64_t original_file_index,
    uint64_t contract,
    uint64_t generation,
    std::shared_ptr<scan_manager::preparation_admission>,
    std::vector<size_t> required_columns = {},
    bool count_only                      = false,
    unit_scan_options options            = {});

  std::unique_ptr<inline_reader> open_inline_unit(uint64_t original_file_index,
                                                  std::vector<size_t> const& required_columns);
  std::shared_ptr<op::scan::arrow_import_ingestible> make_inline_unit(
    uint64_t original_file_index,
    uint64_t contract,
    uint64_t generation,
    std::shared_ptr<scan_manager::preparation_admission>,
    bool count_only                      = false,
    std::vector<size_t> required_columns = {},
    unit_scan_options options            = {});

  // Independent native delete reader; called only on the captured owner lane.
  std::shared_ptr<dv_visibility const> read_parquet_deletes(uint64_t original_file_index,
                                                            scan_manager::charging_allocator&,
                                                            scan_manager::preparation_stop_token);

  // Test-only snapshot visibility; ordinary callers use the snapshot-free overload.
  std::shared_ptr<dv_visibility const> read_parquet_deletes(uint64_t original_file_index,
                                                            scan_manager::charging_allocator&,
                                                            scan_manager::preparation_stop_token,
                                                            dv_snapshot_policy);

  op::scan::bound_read_view const& view() const noexcept;
  void validate_statement() const;
  // Planning may finish on a different thread. Before any provider discovery,
  // the statement preparation boundary transfers this private reader factory to Q.
  void bind_execution_owner();
  // Apply provider filtering to this execution's private list before discovery.
  void configure_scan(std::vector<size_t> const&, duckdb::TableFilterSet*);
  bool can_prune_unit(uint64_t file_index) const;
  // Query/provider owner lane only. Lazily expand the original provider list;
  // return at most one descriptor, preserving its original index. No CPU scan.
  std::optional<unit_descriptor> unit_at(uint64_t original_file_index) const;
  // Copy positions strictly after `after` into caller-owned bounded storage.
  position_page inline_delete_positions(uint64_t original_file_index,
                                        std::span<uint64_t> destination,
                                        std::optional<uint64_t> after = {}) const;

 private:
  struct impl;
  explicit binding_handle(std::unique_ptr<impl>);
  std::unique_ptr<impl> impl_;
};
op::scan::bound_read_view capture_read_view(duckdb::BoundTableFunction const&,
                                            duckdb::FunctionData const*,
                                            duckdb::ClientContext&);
}  // namespace sirius::ducklake
