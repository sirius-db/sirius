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
#include "op/scan/field_id_mapping.hpp"
namespace sirius::ducklake {
// DuckLake policy, separate from the shared mapping mechanism. The caller must
// supply original bound IDs and initial_default, never the current write default.
op::scan::field_requirement bind_field(int32_t id,
                                       duckdb::LogicalType const& type,
                                       duckdb::Value const& initial_default);
// Fixed scalar intersection of provider, mapper and owned Arrow importer.
// nullptr denotes a type outside the current inline/default profile.
char const* fixed_scalar_arrow_format(duckdb::LogicalType const&);
// Empty means all bound columns; repeated/out-of-range source indices are invalid.
std::vector<size_t> required_field_indices(size_t bound_count, std::vector<size_t> selected);
// Stable-ID top-level scalar selections; name maps and visibility are gated by the unit.
op::scan::field_id_mapping compile_field_mapping(std::shared_ptr<op::scan::parquet_metadata const>,
                                                 std::string const& identity,
                                                 op::scan::materialization_program,
                                                 std::vector<op::scan::field_requirement>);
}  // namespace sirius::ducklake
