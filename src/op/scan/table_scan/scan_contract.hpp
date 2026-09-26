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

#include <algorithm>
#include <array>
#include <atomic>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudf::io::parquet {
struct FileMetaData;
enum class Type : int8_t;
struct LogicalType;
}  // namespace cudf::io::parquet
namespace duckdb {
class AttachedDatabase;
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
enum class eligibility_verdict : uint8_t { not_evaluated, supported, unsupported, incomplete };

enum class later_check : uint8_t {
  footer_per_file,
  profile_per_file,
  schema_per_file,
  segments_per_range,
  matrix_per_range,
  host_staged,
  key_held
};
using later_check_set = std::bitset<8>;
using validation_set  = std::bitset<8>;
using leaf_set        = std::vector<bool>;
using profile_id      = uint32_t;
struct physical_check_counters;
struct physical_column_profile {
  uint32_t type                  = 0;
  uint64_t data_codecs           = 0;
  uint64_t validity_or_encodings = 0;
  bool type_mismatch             = false;
  uint32_t logical_annotation    = 0;
  uint32_t converted_annotation  = 0;
  int32_t scale                  = 0;
  int32_t precision              = 0;
};
struct physical_profile {
  uint64_t storage_version = 0;
  std::vector<physical_column_profile> columns;
};
// Owned by the query registry (or a direct pin's ingestible); workers append
// under a lock. Certificates carry only fixed-size ids and shared dependencies.
class physical_profile_table {
 public:
  std::shared_ptr<physical_check_counters> counters;
  profile_id add(physical_profile profile)
  {
    std::lock_guard lock(mutex_);
    if (profiles_.size() >= std::numeric_limits<profile_id>::max())
      throw std::overflow_error("physical profile id space exhausted");
    profiles_.push_back(std::move(profile));
    return static_cast<profile_id>(profiles_.size());
  }
  physical_profile get(profile_id id) const
  {
    std::lock_guard lock(mutex_);
    if (!id) throw std::out_of_range("unevaluated physical profile");
    return profiles_.at(id - 1);
  }

 private:
  mutable std::mutex mutex_;
  std::vector<physical_profile> profiles_;
};
using parquet_physical_type = cudf::io::parquet::Type;
using parquet_logical_type  = cudf::io::parquet::LogicalType;

[[nodiscard]] inline later_check_set check_bit(later_check check) noexcept
{
  later_check_set bits;
  bits.set(static_cast<std::size_t>(check));
  return bits;
}

enum class type_conversion : uint8_t { none };
struct effective_reader_projection {
  std::vector<uint32_t> projected;
  std::vector<uint32_t> filter;
  std::vector<uint32_t> carrier;
  bool natural_read = false;
  // Reader data-column order (D-space), including filter-only inputs.
  std::vector<std::string> names;
  std::vector<duckdb::LogicalType> bound_types;
};

enum class verdict_reason : uint16_t {
  none,
  iceberg_no_table_path,
  iceberg_table_path_not_string,
  iceberg_table_path_empty,
  iceberg_selector_not_snapshot_id,
  iceberg_moved_paths,
  iceberg_no_snapshot_id,
  iceberg_field_id_gap,
  iceberg_snapshot_id_not_integer,
  iceberg_equality_deletes,
  iceberg_schema_footers_unreadable,
  iceberg_schema_no_rows,
  iceberg_schema_no_field_ids,
  iceberg_schema_missing_field,
  iceberg_schema_promoted_type,
  iceberg_schema_field_count,
  iceberg_schema_physical_order,
  native_type_128bit,
  native_type_nested,
  native_type_decimal128,
  native_array_element,
  native_array_child,
  native_type_sentinel,
  native_type_unenumerated,
  native_varchar_overflow,
  native_block_manager,
  native_encrypted,
  native_storage_version_unqualified,
  native_segment_codec,
  native_validity_codec,
  carrier_missing,
  parquet_encrypted,
  parquet_codec_unsupported,
  parquet_type_unqualified,
  parquet_encryption_evidence_missing,
  budget_time,
  budget_bytes,
  evidence_missing,
  interface_unavailable
};

struct physical_check_counters {
  // Installed before a test query, cleared only after its workers have joined.
  // true = before footer processing, false = after successful cuDF decode.
  std::function<void(std::string const&, bool)> parquet_phase_for_testing;
  void parquet_phase(std::string const& file, bool footer) const
  {
    if (track_units && parquet_phase_for_testing) parquet_phase_for_testing(file, footer);
  }

  std::atomic<bool> track_units{false};  // latched when the test process creates its planner
  mutable std::mutex units_mutex;
  std::map<std::string, uint64_t> parquet_reader_calls;
  std::map<std::string, uint64_t> native_decoder_calls;
  void reader_call(std::string const& file)
  {
    if (!track_units) return;
    std::lock_guard lock(units_mutex);
    ++parquet_reader_calls[file];
  }
  void decoder_call(std::string const& group)
  {
    if (!track_units) return;
    std::lock_guard lock(units_mutex);
    ++native_decoder_calls[group];
  }
  std::atomic<uint64_t> checks{0};
  std::array<std::atomic<uint64_t>,
             static_cast<std::size_t>(verdict_reason::interface_unavailable) + 1>
    rejections{};
  std::atomic<uint64_t> type_mismatches{0};
  std::atomic<uint64_t> type_refusals{0};
  void record(verdict_reason reason, uint64_t mismatches = 0)
  {
    checks.fetch_add(1, std::memory_order_relaxed);
    type_mismatches.fetch_add(mismatches, std::memory_order_relaxed);
    if (reason != verdict_reason::none)
      rejections[static_cast<std::size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
    if (reason == verdict_reason::parquet_type_unqualified)
      type_refusals.fetch_add(1, std::memory_order_relaxed);
  }
};

struct certification_cost {
  uint64_t added_time_us              = 0;
  std::size_t added_bytes             = 0;
  std::size_t inherited_capture_bytes = 0;
  uint64_t borrowed_files             = 0;
  uint64_t delete_preparation_time_us = 0;
};

class certification_budget {
 public:
  certification_budget(std::chrono::milliseconds time = std::chrono::milliseconds{50},
                       std::size_t bytes              = 8u << 20,
                       bool test_declines             = false)
    : time_allowance_(time), byte_allowance_(bytes), test_declines_(test_declines)
  {
  }

  void charge(std::chrono::microseconds added, std::size_t added_bytes) noexcept
  {
    auto const time_us      = static_cast<uint64_t>(std::max<int64_t>(0, added.count()));
    consumed_.added_time_us = saturating_add(consumed_.added_time_us, time_us);
    consumed_.added_bytes   = saturating_add(consumed_.added_bytes, added_bytes);
  }

  [[nodiscard]] bool exceeded() const noexcept { return time_exceeded() || bytes_exceeded(); }
  [[nodiscard]] bool time_exceeded() const noexcept
  {
    return consumed_.added_time_us >
           static_cast<uint64_t>(
             std::chrono::duration_cast<std::chrono::microseconds>(time_allowance_).count());
  }
  [[nodiscard]] bool bytes_exceeded() const noexcept
  {
    return consumed_.added_bytes > byte_allowance_;
  }
  [[nodiscard]] bool declines() const noexcept { return test_declines_ && exceeded(); }
  [[nodiscard]] certification_cost consumed() const noexcept { return consumed_; }

 private:
  template <typename T>
  static constexpr T saturating_add(T lhs, T rhs) noexcept
  {
    return rhs > std::numeric_limits<T>::max() - lhs ? std::numeric_limits<T>::max() : lhs + rhs;
  }
  std::chrono::milliseconds time_allowance_;
  std::size_t byte_allowance_;
  bool test_declines_;
  certification_cost consumed_;
};

struct certification_result {
  eligibility_verdict verdict = eligibility_verdict::not_evaluated;
  verdict_reason reason       = verdict_reason::none;
  std::string reason_text;
  later_check_set later_checks;
  certification_cost cost;
  std::optional<uint64_t> storage_version;
  leaf_set semantic_columns;
};

struct pin_validation {
  struct check {
    bool applies = false;
    bool passed  = false;
  };
  check identity, layout, iteration, visibility, structure;
  uint64_t query_token = 0;
};

struct key_held_witness {
  duckdb::AttachedDatabase const* database = nullptr;
  std::string db_path;
  uint64_t query_token = 0;
};

struct scan_verdict_declined : std::runtime_error {
  scan_verdict_declined(std::string text,
                        std::vector<std::pair<scan_contract_id, verdict_reason>> reasons)
    : std::runtime_error(std::move(text)), declined(std::move(reasons))
  {
  }
  std::vector<std::pair<scan_contract_id, verdict_reason>> declined;
};

struct unsupported_physical_input : std::runtime_error {
  unsupported_physical_input(scan_contract_id id,
                             std::string identity,
                             verdict_reason why,
                             std::string text)
    : std::runtime_error(std::move(text)),
      contract(id),
      input_identity(std::move(identity)),
      reason(why)
  {
  }
  scan_contract_id contract;
  std::string input_identity;
  verdict_reason reason;
};

struct certificate_incomplete : std::runtime_error {
  certificate_incomplete(scan_contract_id id, later_check_set absent, std::string text)
    : std::runtime_error(std::move(text)), contract(id), missing(absent)
  {
  }
  scan_contract_id contract;
  later_check_set missing;
};

struct pre_decline {
  eligibility_verdict verdict;
  verdict_reason reason;
  std::string text;
};

// The live metadata rows stay owned by one scan node until Iceberg lowering.
// S3 populates this from the single inventory query.
struct iceberg_delete_inventory {
  struct entry {
    std::string content;
    std::string file_path;
    int64_t manifest_sequence_number = 0;
    std::string file_format;
    std::string manifest_path;
  };
  std::vector<entry> entries;
};

struct pre_capture_result {
  std::optional<pre_decline> decline;
  std::optional<iceberg_delete_inventory> inventory;
  certification_cost cost;
};

// Read once per planning attempt or per query preparation, never from a metadata/GPU task.
struct test_injections {
  uint64_t certification_delay_ms = 0;
  uint64_t certification_bytes    = 0;
  bool budget_declines            = false;
  bool strip_encryption_evidence  = false;
  bool hold_published_batch       = false;
  std::string scan_verdict;
  std::string iceberg_discovery;
  uint64_t pause_after_certify_ms = 0;
  bool lineage_unmodelled         = false;
  bool invalidate_pin_witness     = false;
  uint64_t hold_footer_index      = 0;
  std::string synthetic_native_segment;
  std::string synthetic_parquet_codec;
  bool fail_on_host_staging_refusal  = false;
  uint64_t gpu_task_oom              = 0;
  uint64_t gpu_task_launch_error     = 0;
  uint64_t gpu_task_retry_limit      = 0;
  uint64_t gpu_task_retry_backoff_ms = 0;
  bool override_read_only            = false;
  bool transaction_mismatch          = false;
  bool interrupt_before_replay       = false;
  bool non_rollbackable_state        = false;
};

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
  std::shared_ptr<physical_profile_table> profiles;
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
  profile_id profile = 0;
  validation_set validation;
  std::optional<key_held_witness> key_held;
};
struct split_dependencies {
  std::shared_ptr<cudf::io::parquet::FileMetaData const> footer;
  std::shared_ptr<io::sirius_datasource> datasource;
  std::optional<uint64_t> checkpoint_iteration;
  std::shared_ptr<physical_profile_table> profiles;
};
enum class certificate_evidence_scope : uint8_t { none, binding_correspondence };
struct eligibility_certificate {
  scan_contract_id contract_id              = 0;
  eligibility_verdict verdict               = eligibility_verdict::not_evaluated;
  certificate_evidence_scope evidence_scope = certificate_evidence_scope::none;
  std::string cpu_gpu_view_identity;
  std::string correspondence;
  evidence_depth depth = evidence_depth::path;
  materializer_contract_identity materializer;
  verdict_reason reason = verdict_reason::none;
  std::string reason_text;
  std::optional<uint64_t> storage_version;
  certification_cost cost;
  later_check_set later_checks;
  leaf_set semantic_columns;
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
