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

#include "transparent/read_view_registry.hpp"

#include "io/sirius_datasource.hpp"
#include "op/scan/duckdb_native_gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/parquet_gpu_ingestible.hpp"
#include "op/sirius_physical_table_scan.hpp"
#include "planner/connector_registry.hpp"

#include <duckdb/storage/single_file_block_manager.hpp>

#include <algorithm>
#include <atomic>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace sirius::transparent {
namespace {
constexpr std::size_t mismatch_summary_limit = 3;

std::string fingerprint_summary(op::scan::read_view_fingerprint const& fingerprint)
{
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << fingerprint.hash;
  return out.str();
}

bool same_fingerprint(op::scan::read_view_fingerprint const& left,
                      op::scan::read_view_fingerprint const& right)
{
  return &left == &right || left.canonical == right.canonical;
}

struct fingerprint_count {
  op::scan::read_view_fingerprint const* value = nullptr;
  std::size_t original                         = 0;
  std::size_t candidate                        = 0;
};

using fingerprint_buckets = std::unordered_map<uint64_t, std::vector<std::size_t>>;

void count_fingerprint(std::vector<fingerprint_count>& counts,
                       fingerprint_buckets& buckets,
                       op::scan::read_view_fingerprint const& value,
                       bool original)
{
  auto& bucket = buckets[value.hash];
  auto found   = std::find_if(bucket.begin(), bucket.end(), [&](auto index) {
    // The precomputed hash chooses only the bucket. Canonical equality still decides whether
    // two entries are the same fingerprint, so a hash collision cannot admit a different view.
    return same_fingerprint(*counts[index].value, value);
  });
  if (found == bucket.end()) {
    bucket.push_back(counts.size());
    counts.push_back({&value});
    found = std::prev(bucket.end());
  }
  auto& count = counts[*found];
  ++(original ? count.original : count.candidate);
}

void append_difference(std::vector<std::string>& output,
                       op::scan::read_view_fingerprint const& value,
                       std::size_t count)
{
  while (count-- > 0 && output.size() < mismatch_summary_limit) {
    output.push_back(fingerprint_summary(value));
  }
}

bool compare_multiset(std::span<op::scan::read_view_fingerprint const* const> original,
                      std::span<candidate_binding const> candidate,
                      read_view_comparison& result)
{
  if (original.size() == 1 && candidate.size() == 1) {
    auto const* original_value = original.front();
    auto const& binding        = candidate.front();
    if (!original_value || !binding.view || !binding.view->identity) {
      result.reason = "missing_identity";
      return false;
    }
    auto const& candidate_value = binding.view->identity->fingerprint;
    if (same_fingerprint(*original_value, candidate_value)) return true;
    result.different_original_count  = 1;
    result.different_candidate_count = 1;
    append_difference(result.only_original, *original_value, 1);
    append_difference(result.only_candidate, candidate_value, 1);
    result.reason = "fingerprint_mismatch";
    return false;
  }

  std::vector<fingerprint_count> counts;
  counts.reserve(original.size() + candidate.size());
  fingerprint_buckets buckets;
  buckets.reserve(original.size() + candidate.size());
  for (auto const* value : original) {
    if (!value) {
      result.reason = "missing_identity";
      return false;
    }
    count_fingerprint(counts, buckets, *value, true);
  }
  for (auto const& binding : candidate) {
    if (!binding.view || !binding.view->identity) {
      result.reason = "missing_identity";
      return false;
    }
    count_fingerprint(counts, buckets, binding.view->identity->fingerprint, false);
  }

  for (auto const& count : counts) {
    if (count.original > count.candidate) {
      auto const difference = count.original - count.candidate;
      result.different_original_count += difference;
      append_difference(result.only_original, *count.value, difference);
    }
  }
  for (auto const& count : counts) {
    if (count.candidate > count.original) {
      auto const difference = count.candidate - count.original;
      result.different_candidate_count += difference;
      append_difference(result.only_candidate, *count.value, difference);
    }
  }
  if (!result.only_original.empty() || !result.only_candidate.empty() ||
      original.size() != candidate.size()) {
    result.reason = "fingerprint_mismatch";
    return false;
  }
  return true;
}

bool same_identity(std::shared_ptr<op::scan::bound_read_view const> const& left,
                   std::shared_ptr<op::scan::bound_read_view const> const& right)
{
  return left && right && left->identity && right->identity &&
         (left->identity == right->identity ||
          same_fingerprint(left->identity->fingerprint, right->identity->fingerprint));
}

bool same_identity(op::scan::bound_read_view const* left,
                   std::shared_ptr<op::scan::bound_read_view const> const& right)
{
  return left && right && left->identity && right->identity &&
         (left->identity == right->identity ||
          same_fingerprint(left->identity->fingerprint, right->identity->fingerprint));
}

bool selector_proven(op::scan::bound_read_view const& candidate,
                     op::scan::bound_read_view const* original)
{
  if (!candidate.selector_evidence_required) return true;
  return candidate.logical_selector_evidence && original && original->selector_evidence_required &&
         original->logical_selector_evidence == candidate.logical_selector_evidence;
}

struct original_binding_ref {
  duckdb::idx_t table_index;
  op::scan::bound_read_view const* view;
};

read_view_comparison compare_read_views_impl(
  candidate_origin origin,
  std::span<original_binding_ref const> logical_original,
  std::span<op::scan::read_view_fingerprint const* const> physical_original,
  std::span<candidate_binding const> candidate)
{
  read_view_comparison result;
  result.original_count  = physical_original.size();
  result.candidate_count = candidate.size();
  if (!physical_original.empty() && physical_original.front()) {
    result.original_hash = physical_original.front()->hash;
  }
  if (!candidate.empty() && candidate.front().view && candidate.front().view->identity) {
    result.candidate_hash = candidate.front().view->identity->fingerprint.hash;
  }
  if (origin == candidate_origin::replan) {
    result.correspondence = candidate.empty() ? "none" : "single";
    if (candidate.size() > 1) {
      result.correspondence = "none";
      result.reason         = "no_correspondence";
      return result;
    }
  } else {
    result.correspondence = "table_index";
  }
  if (!compare_multiset(physical_original, candidate, result)) return result;

  if (origin == candidate_origin::replan) {
    if (candidate.empty()) {
      result.equal = true;
      return result;
    }
    op::scan::bound_read_view const* selector_original = nullptr;
    if (logical_original.size() == 1 &&
        same_identity(logical_original.front().view, candidate.front().view)) {
      selector_original = logical_original.front().view;
    }
    if (!selector_proven(*candidate.front().view, selector_original)) {
      result.reason = "selector_unproven";
      return result;
    }
    result.equal = true;
    return result;
  }

  if (logical_original.size() != candidate.size()) {
    result.reason = "binding_mismatch";
    return result;
  }
  std::unordered_set<duckdb::idx_t> matched_table_indexes;
  matched_table_indexes.reserve(candidate.size());
  for (auto const& candidate_binding : candidate) {
    if (!matched_table_indexes.insert(candidate_binding.table_index).second) {
      result.reason = "binding_mismatch";
      return result;
    }
    auto const found =
      std::find_if(logical_original.begin(), logical_original.end(), [&](auto const& original) {
        return original.table_index == candidate_binding.table_index;
      });
    if (found == logical_original.end() || !same_identity(found->view, candidate_binding.view)) {
      result.reason = "binding_mismatch";
      return result;
    }
    if (!selector_proven(*candidate_binding.view, found->view)) {
      result.reason = "selector_unproven";
      return result;
    }
  }
  result.equal = true;
  return result;
}
}  // namespace

read_view_comparison compare_read_views(
  candidate_origin origin,
  std::span<original_binding const> logical_original,
  std::span<op::scan::read_view_fingerprint const> physical_original,
  std::span<candidate_binding const> candidate)
{
  std::vector<original_binding_ref> logical;
  logical.reserve(logical_original.size());
  for (auto const& binding : logical_original)
    logical.push_back({binding.table_index, binding.view.get()});
  std::vector<op::scan::read_view_fingerprint const*> physical;
  physical.reserve(physical_original.size());
  for (auto const& fingerprint : physical_original)
    physical.push_back(&fingerprint);
  return compare_read_views_impl(origin, logical, physical, candidate);
}

read_view_comparison compare_read_views(
  candidate_origin origin,
  op::scan::logical_bound_read_view_capture const* logical_original,
  std::span<op::scan::bound_read_view const> physical_original,
  read_view_registry const& candidate)
{
  std::vector<original_binding_ref> logical;
  if (logical_original) {
    logical.reserve(logical_original->views.size());
    for (auto const& binding : logical_original->views) {
      logical.push_back({binding.table_index, &binding.view});
    }
  }
  std::vector<op::scan::read_view_fingerprint const*> physical;
  physical.reserve(physical_original.size());
  for (auto const& view : physical_original) {
    if (!view.identity) {
      read_view_comparison result;
      result.reason = "missing_identity";
      return result;
    }
    physical.push_back(&view.identity->fingerprint);
  }
  auto bindings = candidate.candidate_bindings();
  auto result   = compare_read_views_impl(origin, logical, physical, bindings);
  if (!physical_original.empty()) result.original_depth = physical_original.front().depth;
  if (!bindings.empty() && bindings.front().view) {
    result.candidate_depth = bindings.front().view->depth;
  }
  return result;
}

void share_equal_read_view_identities(op::scan::logical_bound_read_view_capture* logical_original,
                                      std::span<op::scan::bound_read_view> physical_original,
                                      read_view_registry const& candidate)
{
  auto const bindings = candidate.candidate_bindings();
  auto share          = [&](op::scan::bound_read_view& original, bool required) {
    if (!original.identity) {
      throw std::logic_error("cannot share a missing original read-view identity");
    }
    auto const found = std::find_if(bindings.begin(), bindings.end(), [&](auto const& binding) {
      return binding.view && binding.view->identity &&
             (binding.view->identity == original.identity ||
              same_fingerprint(binding.view->identity->fingerprint,
                               original.identity->fingerprint));
    });
    if (found == bindings.end()) {
      if (required) throw std::logic_error("cannot share unequal read-view identities");
      return;
    }
    original.identity = found->view->identity;
  };

  if (logical_original) {
    for (auto& binding : logical_original->views)
      share(binding.view, false);
  }
  if (physical_original.size() == 1 && bindings.size() == 1 && bindings.front().view &&
      bindings.front().view->identity) {
    // The caller invokes sharing only after the single-entry physical multiset comparison
    // passed, so this is the already-proved match and needs no second canonical scan.
    physical_original.front().identity = bindings.front().view->identity;
  } else {
    for (auto& view : physical_original)
      share(view, true);
  }
}

std::string describe_read_view_mismatch(read_view_comparison const& comparison)
{
  auto append_summaries =
    [](std::ostringstream& out, std::string_view label, std::vector<std::string> const& values) {
      out << ", " << label << "=[";
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) out << ',';
        out << values[index];
      }
      out << ']';
    };
  std::ostringstream out;
  out << "read-view mismatch: reason=" << comparison.reason
      << ", correspondence=" << comparison.correspondence
      << ", original_count=" << comparison.original_count
      << ", candidate_count=" << comparison.candidate_count << ", different_total="
      << comparison.different_original_count + comparison.different_candidate_count
      << ", different_original=" << comparison.different_original_count
      << ", different_candidate=" << comparison.different_candidate_count << ", original_hash=";
  if (comparison.original_hash) {
    out << std::hex << std::setw(16) << std::setfill('0') << *comparison.original_hash << std::dec;
  } else {
    out << "none";
  }
  out << ", candidate_hash=";
  if (comparison.candidate_hash) {
    out << std::hex << std::setw(16) << std::setfill('0') << *comparison.candidate_hash << std::dec;
  } else {
    out << "none";
  }
  auto append_depth = [&](std::string_view label,
                          std::optional<op::scan::evidence_depth> const& depth) {
    out << ", " << label << '=';
    if (!depth) {
      out << "none";
      return;
    }
    switch (*depth) {
      case op::scan::evidence_depth::path: out << "path"; break;
      case op::scan::evidence_depth::path_and_size: out << "path_and_size"; break;
      case op::scan::evidence_depth::path_size_and_tag: out << "path_size_and_tag"; break;
    }
  };
  append_depth("original_depth", comparison.original_depth);
  append_depth("candidate_depth", comparison.candidate_depth);
  append_summaries(out, "only_original", comparison.only_original);
  append_summaries(out, "only_candidate", comparison.only_candidate);
  return out.str();
}

read_view_registry_entry const& read_view_registry::entry(op::scan::scan_contract_id id) const
{
  auto const found = by_contract_id_.find(id);
  if (found == by_contract_id_.end()) {
    throw std::runtime_error("unknown scan contract id " + std::to_string(id));
  }
  return entries_.at(found->second);
}

read_view_registry_entry const& read_view_registry::entry_for_scan_node(uint64_t scan_node_id) const
{
  auto const found = by_scan_node_id_.find(scan_node_id);
  if (found == by_scan_node_id_.end()) {
    throw std::runtime_error("unknown scan node id " + std::to_string(scan_node_id));
  }
  return entries_.at(found->second);
}

std::vector<candidate_binding> read_view_registry::candidate_bindings() const
{
  std::vector<candidate_binding> result;
  result.reserve(entries_.size());
  for (auto const& entry : entries_) {
    result.push_back({entry.contract.table_index, entry.contract.view});
  }
  return result;
}

void read_view_registry::record_delete_preparation(op::scan::scan_contract_id id,
                                                   uint64_t elapsed_us)
{
  entries_.at(by_contract_id_.at(id)).eligibility.cost.delete_preparation_time_us += elapsed_us;
}

void read_view_registry::record_verdict(op::scan::scan_contract_id id,
                                        op::scan::certification_result const& result)
{
  auto const found = by_contract_id_.find(id);
  if (found == by_contract_id_.end()) {
    throw std::runtime_error("unknown scan contract id " + std::to_string(id));
  }
  auto& certificate = entries_[found->second].eligibility;
  if (certificate.verdict != op::scan::eligibility_verdict::not_evaluated) {
    throw std::logic_error("scan verdict already recorded for contract " + std::to_string(id));
  }
  if (result.verdict == op::scan::eligibility_verdict::not_evaluated) {
    throw std::logic_error("cannot record an unevaluated scan verdict");
  }
  if ((result.verdict == op::scan::eligibility_verdict::supported &&
       result.reason != op::scan::verdict_reason::none) ||
      (result.verdict != op::scan::eligibility_verdict::supported &&
       result.reason == op::scan::verdict_reason::none)) {
    throw std::logic_error("scan verdict and reason disagree");
  }
  if (result.verdict == op::scan::eligibility_verdict::incomplete &&
      result.reason != op::scan::verdict_reason::budget_time &&
      result.reason != op::scan::verdict_reason::budget_bytes &&
      result.reason != op::scan::verdict_reason::evidence_missing &&
      result.reason != op::scan::verdict_reason::interface_unavailable) {
    throw std::logic_error("incomplete scan verdict has a non-incomplete reason");
  }
  certificate.verdict          = result.verdict;
  certificate.reason           = result.reason;
  certificate.reason_text      = result.reason_text;
  certificate.later_checks     = result.later_checks;
  certificate.cost             = result.cost;
  certificate.storage_version  = result.storage_version;
  certificate.semantic_columns = result.semantic_columns;
}

op::scan::scan_contract_id read_view_registry::allocate_declined_scan(
  op::sirius_physical_table_scan const& scan, planner::connector const& connector)
{
  if (!scan.read_views || scan.read_views.get() != this) {
    throw std::logic_error("declined scan is not owned by this read-view registry");
  }
  op::scan::column_requirements columns;
  columns.column_ids     = scan.column_ids;
  columns.projection_ids = scan.projection_ids;
  for (auto const& column : scan.column_ids) {
    if (column.IsRowIdColumn()) columns.requires_row_id = true;
    if (column.IsVirtualColumn() && column.HasPrimaryIndex()) {
      columns.virtual_columns.push_back(column.GetPrimaryIndex());
    }
  }
  auto kind = op::scan::materializer_kind::parquet;
  if (connector.kind == op::scan::source_kind::duckdb_native) {
    kind = op::scan::materializer_kind::duckdb_native;
  } else if (connector.kind == op::scan::source_kind::stream_source) {
    kind = op::scan::materializer_kind::stream;
  } else if (connector.function_name == "iceberg_scan") {
    kind = op::scan::materializer_kind::iceberg;
  }
  return op::scan::allocate_scan_contract(*this,
                                          scan.contract_window_id,
                                          scan.contract_finalize_generation,
                                          scan.scan_node_id,
                                          nullptr,
                                          std::move(columns),
                                          {},
                                          {kind, connector.registry_profile},
                                          scan.duckdb_types,
                                          scan.table_index);
}

void read_view_registry::publish_correspondence(
  op::scan::certificate_evidence_scope scope,
  std::string correspondence,
  std::span<op::scan::bound_read_view const> physical_original)
{
  std::vector<bool> consumed(physical_original.size(), false);
  for (auto& entry : entries_) {
    if (entry.eligibility.verdict == op::scan::eligibility_verdict::not_evaluated) {
      entry.eligibility.verdict = op::scan::eligibility_verdict::supported;
    }
    entry.eligibility.evidence_scope = scope;
    entry.eligibility.correspondence = correspondence;
    if (entry.contract.view && entry.contract.view->identity) {
      entry.eligibility.cpu_gpu_view_identity =
        fingerprint_summary(entry.contract.view->identity->fingerprint);
      auto found = physical_original.size();
      for (std::size_t index = 0; index < physical_original.size(); ++index) {
        auto const& original = physical_original[index];
        if (!consumed[index] && original.identity &&
            (original.identity == entry.contract.view->identity ||
             same_fingerprint(original.identity->fingerprint,
                              entry.contract.view->identity->fingerprint))) {
          found = index;
          break;
        }
      }
      if (found == physical_original.size()) {
        throw std::logic_error("cannot publish eligibility without a physical original");
      }
      consumed[found]         = true;
      entry.physical_evidence = physical_original[found].evidence;
      entry.eligibility.depth = physical_original[found].depth;
    }
  }
}

void read_view_registry::inject_mismatch_for_testing(bool swap)
{
  if (entries_.empty()) return;
  if (swap && entries_.size() > 1) {
    std::swap(entries_[0].contract.view, entries_[1].contract.view);
    return;
  }
  auto const& current = entries_.front().contract.view;
  if (!current || !current->identity) return;
  auto identity = *current->identity;
  identity.fingerprint.canonical += "|test-mismatch";
  identity.fingerprint.hash ^= 0x9e3779b97f4a7c15ULL;
  auto changed      = std::make_shared<op::scan::bound_read_view>(*current);
  changed->identity = std::make_shared<op::scan::bound_read_identity const>(std::move(identity));
  entries_.front().contract.view = std::move(changed);
}

}  // namespace sirius::transparent

namespace sirius::op::scan {
namespace {
std::atomic<scan_contract_id> next_contract_id{1};

scan_contract_id reserve_contract_id()
{
  auto current = next_contract_id.load(std::memory_order_relaxed);
  for (;;) {
    if (current == 0) { throw std::overflow_error("scan contract id space exhausted"); }
    auto const next = current == std::numeric_limits<scan_contract_id>::max() ? 0 : current + 1;
    if (next_contract_id.compare_exchange_weak(
          current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
      return current;
    }
  }
}
}  // namespace

scan_contract_id allocate_scan_contract(transparent::read_view_registry& registry,
                                        std::optional<uint64_t> window_id,
                                        uint64_t finalize_generation,
                                        uint64_t scan_node_id,
                                        std::shared_ptr<bound_read_view const> view,
                                        column_requirements columns,
                                        predicate_contract predicates,
                                        materializer_contract_identity materializer,
                                        duckdb::vector<duckdb::LogicalType> output_types,
                                        duckdb::idx_t table_index)
{
  if (registry.by_scan_node_id_.contains(scan_node_id)) {
    throw std::runtime_error("duplicate scan node id " + std::to_string(scan_node_id));
  }
  auto const id = reserve_contract_id();

  bound_table_scan contract;
  contract.scan_node_id = scan_node_id;
  contract.table_index  = table_index;
  contract.view         = std::move(view);
  contract.output_types = std::move(output_types);
  contract.columns      = std::move(columns);
  contract.predicates   = std::move(predicates);
  contract.materializer = std::move(materializer);
  contract.contract_id  = id;

  eligibility_certificate eligibility;
  eligibility.contract_id  = id;
  eligibility.depth        = contract.view ? contract.view->depth : evidence_depth::path;
  eligibility.materializer = contract.materializer;
  switch (contract.materializer.kind) {
    case materializer_kind::duckdb_native:
      eligibility.later_checks = check_bit(later_check::segments_per_range);
      break;
    case materializer_kind::parquet:
    case materializer_kind::iceberg:
      eligibility.later_checks = check_bit(later_check::footer_per_file);
      break;
    case materializer_kind::stream: eligibility.later_checks = {}; break;
  }

  auto const index = registry.entries_.size();
  registry.entries_.push_back(
    {std::move(contract), std::move(eligibility), nullptr, window_id, finalize_generation});
  registry.by_contract_id_.emplace(id, index);
  registry.by_scan_node_id_.emplace(scan_node_id, index);
  return id;
}

bound_table_scan const& contract_of(transparent::read_view_registry const& registry,
                                    scan_contract_id contract_id)
{
  return registry.entry(contract_id).contract;
}

void validate_split_for_gpu(scan_contract_id expected,
                            later_check_set const& required,
                            std::optional<key_held_witness> const& expected_key,
                            scan_info const& split)
{
  // Preserve R1's handle mismatch and its precedence.
  if (split.contract_id() != expected) {
    throw std::runtime_error("scan split contract mismatch: expected " + std::to_string(expected) +
                             ", got " + std::to_string(split.contract_id()));
  }
  for (auto const& certificate : split.certificates()) {
    if (certificate.contract_id != expected) {
      throw std::runtime_error("scan split certificate contract mismatch: expected " +
                               std::to_string(expected) + ", got " +
                               std::to_string(certificate.contract_id));
    }
  }
  auto fail = [&](std::string text, later_check_set missing = {}) {
    throw certificate_incomplete(expected, missing, "scan certificate incomplete: " + text);
  };
  auto certificates = split.certificates();
  auto dependencies = split.dependencies();
  if (certificates.size() != dependencies.size()) fail("non-parallel dependencies");
  auto same_source = [](auto const& a, auto const& b) {
    return a && b && &a->get_io_object() == &b->get_io_object();
  };
  auto coverage = [&](std::size_t i, later_check_set checks) {
    auto missing = checks & ~certificates[i].validation;
    if (missing.any()) fail("missing physical checks", missing);
    if (!dependencies[i].profiles || !dependencies[i].profiles->contains(certificates[i].profile))
      fail("missing physical profile");
  };
  if (auto const* parquet = dynamic_cast<parquet_split_info const*>(&split)) {
    bool empty = std::all_of(parquet->rg_slices.begin(),
                             parquet->rg_slices.end(),
                             [](auto const& s) { return s.row_group_indices.empty(); });
    if (certificates.empty() && empty) return;
    if (certificates.size() != parquet->rg_slices.size()) fail("Parquet slice count");
    for (std::size_t i = 0; i < certificates.size(); ++i) {
      auto const& slice      = parquet->rg_slices[i];
      auto const& dependency = dependencies[i];
      if (!dependency.parquet_approval || !dependency.parquet_approval->footer_bytes)
        fail("Parquet file approval missing");
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
      coverage(i, required);
    }
    return;
  }
  if (auto const* native = dynamic_cast<duckdb_native_scan_info const*>(&split)) {
    if (certificates.empty() && native->row_groups.empty()) return;
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
      coverage(i, checks);
    }
    return;
  }
  fail("unknown input class");
}

void admit_resident_batch(scan_contract_id expected,
                          pin_validation const& validation,
                          uint64_t query_token)
{
  auto fail = [&](std::string const& check) {
    throw certificate_incomplete(expected, {}, "resident certificate incomplete: " + check);
  };
  if (validation.query_token != query_token) fail("query token mismatch");
  auto check = [&](char const* name, pin_validation::check value, bool mandatory) {
    if ((mandatory && !value.applies) || (value.applies && !value.passed)) fail(name);
  };
  check("identity", validation.identity, true);
  check("layout", validation.layout, true);
  check("structure", validation.structure, true);
  if (validation.iteration.applies != validation.visibility.applies) fail("native applicability");
  check("iteration", validation.iteration, false);
  check("visibility", validation.visibility, false);
}

}  // namespace sirius::op::scan
