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

#include "op/scan/table_scan/split_validation.hpp"

#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "transparent/replay_admission.hpp"

#include <duckdb/storage/single_file_block_manager.hpp>

namespace sirius::op::scan {
namespace {
[[noreturn]] void fail(scan_contract_id id, std::string text, later_check_set missing = {})
{
  throw certificate_incomplete(id, missing, "scan certificate incomplete: " + text);
}
bool same_source(auto const& a, auto const& b)
{
  return a && b && &a->get_io_object() == &b->get_io_object();
}
class parquet_validator final : public split_validator {
 public:
  bool accepts(scan_info const& split) const noexcept override
  {
    return dynamic_cast<parquet_split_info const*>(&split) != nullptr;
  }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const& split) const override
  {
    auto* parquet = dynamic_cast<parquet_split_info const*>(&split);
    auto fail     = [&](std::string text, later_check_set missing = {}) {
      ::sirius::op::scan::fail(context.contract, std::move(text), missing);
    };
    if (!parquet) fail("validator/input mismatch");
    auto certificates = split.certificates();
    auto dependencies = split.dependencies();
    auto required     = context.required;
    split_validation_evidence evidence;
    bool empty = std::all_of(parquet->rg_slices.begin(),
                             parquet->rg_slices.end(),
                             [](auto const& s) { return s.row_group_indices.empty(); });
    if (certificates.empty() && empty) return evidence;
    if (certificates.size() != parquet->rg_slices.size()) fail("Parquet slice count");
    for (std::size_t i = 0; i < certificates.size(); ++i) {
      auto const& slice      = parquet->rg_slices[i];
      auto const& dependency = dependencies[i];
      if (!dependency.parquet_approval || !dependency.parquet_approval->footer_bytes)
        fail("Parquet file approval missing");
      auto const& approval = *dependency.parquet_approval;
      if (approval.object_identity.empty() || !approval.metadata_owner || !dependency.datasource ||
          approval.object_identity != dependency.datasource->get_io_object().identity_cache_key() ||
          dependency.footer.owner_before(approval.metadata_owner) ||
          approval.metadata_owner.owner_before(dependency.footer))
        fail("Parquet evidence belongs to another object identity");
      auto const prefix =
        slice.file_path + "|footer=" + std::to_string(dependency.parquet_approval->footer_bytes);
      auto const& identity = certificates[i].input_identity;
      if (!slice.file_metadata || dependency.footer != slice.file_metadata ||
          !same_source(dependency.datasource, slice.datasource) ||
          (identity != prefix && !identity.starts_with(prefix + "|")))
        fail("Parquet slice identity or dependencies");
      for (auto rg : slice.row_group_indices)
        if (rg < 0 || static_cast<std::size_t>(rg) >= slice.file_metadata->row_groups.size())
          fail("Parquet row group outside footer");
        else if (!std::binary_search(dependency.parquet_approval->row_groups.begin(),
                                     dependency.parquet_approval->row_groups.end(),
                                     rg))
          fail("Parquet row group was not physically checked");
      evidence.coverage.push_back(required);
    }
    return evidence;
  }
};
class native_validator final : public split_validator {
 public:
  bool accepts(scan_info const& split) const noexcept override
  {
    return dynamic_cast<duckdb_native_scan_info const*>(&split) != nullptr;
  }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const& split) const override
  {
    auto* native = dynamic_cast<duckdb_native_scan_info const*>(&split);
    auto fail    = [&](std::string text, later_check_set missing = {}) {
      ::sirius::op::scan::fail(context.contract, std::move(text), missing);
    };
    if (!native) fail("validator/input mismatch");
    auto certificates        = split.certificates();
    auto dependencies        = split.dependencies();
    auto required            = context.required;
    auto const& expected_key = context.key;
    split_validation_evidence evidence;
    if (certificates.empty() && native->row_groups.empty()) return evidence;
    if (certificates.size() != native->row_groups.size()) fail("native row-group count");
    for (std::size_t i = 0; i < certificates.size(); ++i) {
      auto const& group       = native->row_groups[i];
      auto const& certificate = certificates[i];
      auto const& dependency  = dependencies[i];
      auto suffix             = "|row_group=" + std::to_string(group.row_group_index);
      if (certificate.split_id != group.row_group_index ||
          !certificate.input_identity.ends_with(suffix))
        fail("native row-group identity");
      if (dependency.footer) fail("native unit carries a file footer");
      later_check_set checks = required;
      if (native->is_insert_delta) {
        checks         = check_bit(later_check::key_held);
        bool file_read = false;
        auto segments  = [&](auto const& lane) {
          for (auto const& segment : lane) {
            checks |= segment.is_transient ? check_bit(later_check::host_staged)
                                            : check_bit(later_check::segments_per_range) |
                                               check_bit(later_check::matrix_per_range);
            if (segment.is_transient && !segment.all_null && !segment.host_ptr)
              fail("transient segment has no staged bytes");
            file_read |= !segment.is_transient && segment.block_id >= 0;
          }
        };
        for (auto const& column : group.columns) {
          segments(column.data_segments);
          segments(column.validity_segments);
          segments(column.array_child_data_segments);
          segments(column.array_child_validity_segments);
        }
        if (!expected_key || !expected_key->database || !certificate.key_held ||
            certificate.key_held->database != expected_key->database ||
            certificate.key_held->db_path != expected_key->db_path ||
            certificate.key_held->query_token != expected_key->query_token)
          fail("delta checkpoint key does not belong to this query/database",
               check_bit(later_check::key_held));
        if (dependency.checkpoint_iteration) fail("delta carries a fresh-native iteration");
        if (file_read && !same_source(dependency.datasource, native->datasource))
          fail("delta file dependency");
        if (dependency.datasource && !same_source(dependency.datasource, native->datasource))
          fail("delta datasource mismatch");
      } else {
        if (!dependency.checkpoint_iteration ||
            !same_source(dependency.datasource, native->datasource))
          fail("native datasource or iteration missing");
        auto const identity = native->datasource->get_io_object().object_path() +
                              "|checkpoint=" + std::to_string(*dependency.checkpoint_iteration) +
                              suffix;
        if (certificate.input_identity != identity) fail("native identity/iteration mismatch");
        if (!native->block_manager ||
            native->block_manager->GetCheckpointIteration() != *dependency.checkpoint_iteration)
          fail("native checkpoint iteration changed");
      }
      evidence.coverage.push_back(checks);
    }
    return evidence;
  }
};

class combined_validator final : public split_validator {
  std::vector<std::shared_ptr<split_validator const>> validators;

 public:
  explicit combined_validator(std::vector<std::shared_ptr<split_validator const>> v)
    : validators(std::move(v))
  {
    if (validators.empty() ||
        std::any_of(validators.begin(), validators.end(), [](auto const& p) { return !p; }))
      throw std::invalid_argument("missing split validator");
  }
  bool accepts(scan_info const& split) const noexcept override
  {
    return std::any_of(
      validators.begin(), validators.end(), [&](auto const& v) { return v->accepts(split); });
  }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const& split) const override
  {
    split_validator const* selected = nullptr;
    for (auto const& v : validators)
      if (v->accepts(split)) {
        if (selected) fail(context.contract, "ambiguous split validator");
        selected = v.get();
      }
    if (!selected) fail(context.contract, "unknown input class");
    return selected->inspect(context, split);
  }
};
}  // namespace
std::shared_ptr<split_validator const> parquet_split_validator()
{
  static auto v = std::make_shared<parquet_validator>();
  return v;
}
std::shared_ptr<split_validator const> native_split_validator()
{
  static auto v = std::make_shared<native_validator>();
  return v;
}
std::shared_ptr<split_validator const> combine_split_validators(
  std::vector<std::shared_ptr<split_validator const>> v)
{
  return std::make_shared<combined_validator>(std::move(v));
}
std::shared_ptr<split_validator const> default_split_validator(materializer_kind kind)
{
  switch (kind) {
    case materializer_kind::parquet:
    case materializer_kind::iceberg: return parquet_split_validator();
    case materializer_kind::duckdb_native: return native_split_validator();
    case materializer_kind::stream: return nullptr;  // streaming does not use scan metadata
  }
  throw std::invalid_argument("unknown materializer kind");
}
void validate_split_contract(scan_contract_id expected, scan_info const& split)
{
  if (split.contract_id() != expected)
    throw transparent::classified_execution_error(transparent::late_failure_cause::certificate,
                                                  "scan split contract mismatch: expected " +
                                                    std::to_string(expected) + ", got " +
                                                    std::to_string(split.contract_id()));
  auto certificates = split.certificates();
  for (auto const& certificate : certificates)
    if (certificate.contract_id != expected)
      throw transparent::classified_execution_error(
        transparent::late_failure_cause::certificate,
        "scan split certificate contract mismatch: expected " + std::to_string(expected) +
          ", got " + std::to_string(certificate.contract_id));
}
void validate_split_for_gpu(scan_contract_id expected,
                            later_check_set const& required,
                            std::optional<key_held_witness> const& key,
                            scan_info const& split,
                            split_validator const& validator,
                            uint64_t generation,
                            std::shared_ptr<physical_profile_table> const& profiles)
{
  validate_split_contract(expected, split);
  auto certificates = split.certificates();
  auto dependencies = split.dependencies();
  if (certificates.size() != dependencies.size()) fail(expected, "non-parallel dependencies");
  for (auto const& dependency : dependencies)
    if (profiles && dependency.profiles && dependency.profiles != profiles)
      fail(expected, "physical profile belongs to another scan registry");
  if (!validator.accepts(split)) fail(expected, "validator/input mismatch");
  auto evidence = validator.inspect({expected, required, key}, split);
  if (evidence.generation && *evidence.generation != generation)
    fail(expected, "split generation mismatch");
  if (evidence.coverage.size() != certificates.size())
    fail(expected, "validator certificate coverage mismatch");
  for (std::size_t i = 0; i < certificates.size(); ++i) {
    auto missing = evidence.coverage[i] & ~certificates[i].validation;
    if (missing.any()) fail(expected, "missing physical checks", missing);
    if (!dependencies[i].profiles || !dependencies[i].profiles->contains(certificates[i].profile))
      fail(expected, "missing physical profile");
  }
}
// Compatibility for standalone legacy-format qualification tests. Execution uses
// the overload with the policy and owner already selected by trusted planning.
void validate_split_for_gpu(scan_contract_id id,
                            later_check_set const& required,
                            std::optional<key_held_witness> const& key,
                            scan_info const& split)
{
  static auto legacy =
    combine_split_validators({parquet_split_validator(), native_split_validator()});
  validate_split_for_gpu(id, required, key, split, *legacy, 0, {});
}
}  // namespace sirius::op::scan
