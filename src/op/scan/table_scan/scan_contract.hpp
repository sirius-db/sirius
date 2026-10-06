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

#include "op/scan/table_scan/bound_read_view.hpp"
#include "op/scan/table_scan/materializer_capabilities.hpp"

#include <duckdb/common/column_index.hpp>

namespace cudf::io::parquet {
struct FileMetaData;
}
namespace sirius::io {
class sirius_datasource;
}

namespace sirius::op::scan {
class scan_info;
}
namespace sirius::transparent {
class read_view_registry;
}
namespace sirius::op::scan {
using scan_contract_id = uint64_t;

struct column_requirements {
  duckdb::vector<duckdb::ColumnIndex> column_ids;
  duckdb::vector<duckdb::idx_t> projection_ids;
  bool requires_row_id = false;
  duckdb::vector<duckdb::column_t> virtual_columns;
};
struct predicate_contract {
  std::string static_filter_fingerprint;
  std::string pushdown_mode;
};
struct bound_table_scan {
  uint64_t scan_node_id     = 0;
  duckdb::idx_t table_index = duckdb::DConstants::INVALID_INDEX;
  std::shared_ptr<bound_read_view const> view;
  duckdb::vector<duckdb::LogicalType> output_types;
  column_requirements columns;
  predicate_contract predicates;
  materializer_contract_identity materializer;
  scan_contract_id contract_id = 0;
};
struct split_materializer_certificate {
  scan_contract_id contract_id = 0;
  uint64_t split_id            = 0;
  std::string input_identity;
  std::string profile;
  std::string validation;
};
struct split_dependencies {
  std::shared_ptr<cudf::io::parquet::FileMetaData const> footer;
  std::shared_ptr<io::sirius_datasource> datasource;
  std::optional<uint64_t> checkpoint_iteration;
};
enum class eligibility_verdict : uint8_t { not_evaluated, supported, unsupported, incomplete };
enum class certificate_evidence_scope : uint8_t { none, binding_correspondence };
struct eligibility_certificate {
  scan_contract_id contract_id              = 0;
  eligibility_verdict verdict               = eligibility_verdict::not_evaluated;
  certificate_evidence_scope evidence_scope = certificate_evidence_scope::none;
  std::string cpu_gpu_view_identity;
  std::string correspondence;
  evidence_depth depth = evidence_depth::path;
  materializer_contract_identity materializer;
  std::vector<std::string> later_checks;
};

scan_contract_id allocate_scan_contract(
  transparent::read_view_registry&,
  std::optional<uint64_t> window_id,
  uint64_t finalize_generation,
  uint64_t scan_node_id,
  std::shared_ptr<bound_read_view const>,
  column_requirements,
  predicate_contract,
  materializer_contract_identity,
  duckdb::vector<duckdb::LogicalType> output_types = {},
  duckdb::idx_t table_index                        = duckdb::DConstants::INVALID_INDEX);
bound_table_scan const& contract_of(transparent::read_view_registry const&, scan_contract_id);
void validate_split_for_gpu(scan_contract_id expected, scan_info const& split);
}  // namespace sirius::op::scan
