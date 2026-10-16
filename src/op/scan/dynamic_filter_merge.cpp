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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <data/data_batch_utils.hpp>
#include <log/logging.hpp>
#include <op/dynamic_filter/detail/accumulated_bloom_builder.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/dynamic_filter/dynamic_filter_mask_ops.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/**
 * @brief The strategy selection logic.
 *
 * If not every keep ratio is known, and
 *  - the row width is >= k_deferred_minimum_row_width, use DEFERRED_KEYS strategy;
 *  - otherwise, use CASCADE strategy.
 * If the row width is < k_deferred_minimum_row_width, and
 *  - every keep ratio >= k_gather_once_minimum_narrow_keep_ratio,
 *    use GATHER_ONCE strategy;
 *  - otherwise, use CASCADE strategy.
 * If the row width is >= k_deferred_minimum_row_width, and
 *  - any keep ratio <= k_deferred_selective_keep_ratio,
 *    use DEFERRED_KEYS strategy;
 *  - otherwise, use GATHER_ONCE strategy.
 *
 * Measured filters keeping more than dynamic_filter_gate::filter_skippable() allows are skipped
 * before selection, except that rows at least k_deferred_minimum_row_width wide keep filters up to
 * k_wide_row_skip_keep_ratio: gather once applies such weak filters for roughly the cost per
 * removed row that cascade pays at the gate's threshold.
 *
 * See evaluate_compaction_policy() for the implementation.
 * @note These are empirically chosen parameters from experiments on a GB300 machine. They may not
 *       extrapolate perfectly to all architectures.
 */
constexpr std::size_t k_deferred_minimum_row_width       = 64;
constexpr double k_deferred_selective_keep_ratio         = 0.35;
constexpr double k_gather_once_minimum_narrow_keep_ratio = 0.40;
constexpr double k_wide_row_skip_keep_ratio              = 0.60;

}  // namespace

namespace detail {

/// @brief One selected filter application.
struct selected_step {
  applied_entry identity;
  cudf::size_type column_index;
  std::vector<applied_entry> represented;
  sirius::op::sirius_mask_applicable const* mask_source = nullptr;
  sirius::op::sirius_ast_lowerable const* ast_source    = nullptr;
  std::optional<double> expected_keep;
  bool sample                = false;
  std::size_t snapshot_index = 0;
};

/// @brief The state of a program of (possibly) multiple selected filter applications.
struct application_program_state {
  sirius::op::dynamic_filter_snapshot snapshot;
  batch_receipt receipt;
  std::vector<selected_step> membership;
  std::vector<selected_step> ast;
  compaction_strategy strategy = compaction_strategy::CASCADE;
  cudf::size_type rows         = 0;
  std::vector<cudf::data_type> types;
  int device_id        = -1;
  bool sample_combined = false;
  bool want_counts     = false;
  bool validate        = false;
};

/// @brief A key column compacted alongside the survivor row IDs.
struct aligned_key_column {
  std::size_t index;                     ///< The key column index of the batch.
  std::unique_ptr<cudf::column> column;  ///< The compacted key (same length as the row IDs).
};

/// @brief State for DEFERRED_KEYS compaction strategy.
struct deferred_selection_state {
  std::unique_ptr<cudf::column> row_ids;          ///< The survivor row IDs.
  std::optional<aligned_key_column> aligned_key;  ///< The last compacted key column, if kept.
};

/// @brief Pinned RAII owner for memory for storing counts of type `cudf::size_type`.
class pinned_counts {
 public:
  pinned_counts(std::size_t size, ::cuda::stream_ref stream)
    : _mr(cudf::get_pinned_memory_resource()),
      _stream(stream),
      _bytes(size * sizeof(cudf::size_type)),
      _data(static_cast<cudf::size_type*>(_mr.allocate(stream, _bytes, alignof(cudf::size_type))))
  {
  }
  ~pinned_counts() { _mr.deallocate(_stream, _data, _bytes, alignof(cudf::size_type)); }
  pinned_counts(pinned_counts const&)            = delete;
  pinned_counts& operator=(pinned_counts const&) = delete;
  [[nodiscard]] cudf::size_type* data() const noexcept { return _data; }
  [[nodiscard]] std::size_t size() const noexcept { return _bytes / sizeof(cudf::size_type); }

 private:
  rmm::host_device_async_resource_ref _mr;
  ::cuda::stream_ref _stream;
  std::size_t _bytes;
  cudf::size_type* _data;
};

/// @brief A single observation of a pending filter application, including the key and row counts
/// before and after.
struct pending_observation {
  bound_filter_key key;
  cudf::size_type before;
  cudf::size_type after;
};

/// @brief The result of a single filter application, including all relevant state and outputs.
struct application_result_state {
  std::unique_ptr<application_program_state> program;
  std::optional<cucascade::read_only_data_batch> source;
  std::shared_ptr<cudf::table const> test_source;
  std::unique_ptr<cudf::ast::tree> tree;
  std::unique_ptr<cudf::column> ast_mask;
  std::unique_ptr<cudf::column> mask;
  std::unique_ptr<cudf::column> conjunction;
  std::unique_ptr<cudf::numeric_scalar<cudf::size_type>> zero;
  std::unique_ptr<cudf::table> output;
  std::unique_ptr<cudf::table> scratch_table;
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::vector<std::unique_ptr<cudf::column>> output_columns;
  std::vector<cudf::size_type> validation_ids;
  deferred_selection_state deferred;
  std::optional<rmm::device_uvector<cudf::size_type>> counters;
  std::unique_ptr<pinned_counts> readback;
  std::vector<cudf::size_type> counts;
  std::vector<bool> contributed;
  std::vector<pending_observation> observations;
  cudf::size_type final_rows = 0;
  bool ast_contributed       = false;
  bool counts_read_back      = false;
  bool completed             = false;
};
}  // namespace detail

namespace {
struct compaction_policy_decision {
  detail::compaction_strategy strategy;
  char const* reason;
};

[[nodiscard]] bool is_known_keep_ratio(std::optional<double> estimate) noexcept
{
  return estimate && std::isfinite(*estimate) && *estimate >= 0.0 && *estimate <= 1.0;
}

/// @brief Average bytes per row, or empty when the byte accounting cannot size the batch.
[[nodiscard]] std::optional<std::size_t> average_row_width(
  std::size_t rows, std::optional<std::size_t> input_bytes) noexcept
{
  if (rows == 0 || !input_bytes || *input_bytes == 0 ||
      *input_bytes == std::numeric_limits<std::size_t>::max() || *input_bytes < rows) {
    return std::nullopt;
  }
  return *input_bytes / rows;
}

/// @brief Whether a measured membership filter is too weak to apply at this row width.
[[nodiscard]] bool skip_measured_filter(double kept, std::optional<std::size_t> row_width) noexcept
{
  if (row_width && *row_width >= k_deferred_minimum_row_width) {
    return kept > k_wide_row_skip_keep_ratio;
  }
  return dynamic_filter_gate::filter_skippable(kept);
}

/// @brief Choose the optimal compaction strategy.
[[nodiscard]] compaction_policy_decision evaluate_compaction_policy(
  detail::compaction_policy_input const& input) noexcept
{
  using detail::compaction_strategy;
  if (input.membership.size() + static_cast<std::size_t>(input.has_ast_mask) < 2) {
    return {compaction_strategy::CASCADE, "fewer_than_two_candidates"};
  }
  auto const width = average_row_width(input.rows, input.input_bytes);
  if (!width) { return {compaction_strategy::CASCADE, "invalid_width"}; }

  auto const row_width = *width;
  auto const all_known = std::ranges::all_of(
    input.membership, [](auto const& keep) { return is_known_keep_ratio(keep); });
  if (!all_known) {
    return row_width >= k_deferred_minimum_row_width
             ? compaction_policy_decision{compaction_strategy::DEFERRED_KEYS,
                                          "unknown_membership_wide"}
             : compaction_policy_decision{compaction_strategy::CASCADE,
                                          "unknown_membership_narrow"};
  }

  if (row_width >= k_deferred_minimum_row_width) {
    auto const has_selective = std::ranges::any_of(
      input.membership, [](auto const& keep) { return *keep <= k_deferred_selective_keep_ratio; });
    return has_selective ? compaction_policy_decision{compaction_strategy::DEFERRED_KEYS,
                                                      "selective_membership_wide"}
                         : compaction_policy_decision{compaction_strategy::GATHER_ONCE,
                                                      "weak_memberships_wide"};
  }

  auto const all_weak = std::ranges::all_of(input.membership, [](auto const& keep) {
    return *keep >= k_gather_once_minimum_narrow_keep_ratio;
  });
  return all_weak
           ? compaction_policy_decision{compaction_strategy::GATHER_ONCE, "weak_memberships_narrow"}
           : compaction_policy_decision{compaction_strategy::CASCADE,
                                        "selective_membership_narrow"};
}

/// @brief ANDs @p fragment onto @p root, or starts the conjunction when @p root is null.
[[nodiscard]] cudf::ast::expression const* and_into(cudf::ast::tree& tree,
                                                    cudf::ast::expression const* root,
                                                    cudf::ast::expression const& fragment)
{
  return root ? &tree.emplace<cudf::ast::operation>(
                  cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
              : &fragment;
}

void validate_selection_state(detail::deferred_selection_state const& state,
                              cudf::size_type original_rows,
                              ::cuda::stream_ref stream,
                              std::vector<cudf::size_type>& host_ids)
{
  if (!state.row_ids || state.row_ids->type().id() != cudf::type_id::INT32 ||
      (state.aligned_key && state.aligned_key->column->size() != state.row_ids->size())) {
    throw std::logic_error("deferred dynamic-filter selection state is misaligned");
  }

  host_ids.resize(static_cast<std::size_t>(state.row_ids->size()));
  if (!host_ids.empty()) {
    auto const status = cudaMemcpyAsync(host_ids.data(),
                                        state.row_ids->view().data<cudf::size_type>(),
                                        host_ids.size() * sizeof(cudf::size_type),
                                        cudaMemcpyDeviceToHost,
                                        stream.get());
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
    stream.sync();
  }
  for (std::size_t index = 0; index < host_ids.size(); ++index) {
    if (host_ids[index] < 0 || host_ids[index] >= original_rows ||
        (index != 0 && host_ids[index - 1] >= host_ids[index])) {
      throw std::logic_error("deferred dynamic-filter row IDs violate original-row ordering");
    }
  }
}

}  // namespace

detail::compaction_strategy detail::choose_compaction_strategy(
  compaction_policy_input const& input) noexcept
{
  return evaluate_compaction_policy(input).strategy;
}

cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::dynamic_filter_snapshot const& filters,
  scan_plan const& plan,
  int device_id)
{
  device_id        = sirius::op::detail::resolve_dynamic_filter_device_id(device_id);
  auto const* root = existing_root;
  for (auto const& [col_idx, filter] : filters.entries()) {
    if (col_idx >= plan.output_layout.size()) { continue; }
    auto const& entry = plan.output_layout[col_idx];
    if (entry.source != scan_plan::output_entry::DATA) { continue; }  // hive — skip
    auto const& parquet_col_name = plan.data_columns[entry.idx].name;

    if (!filter->is_available_on_device(device_id)) { continue; }
    auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(filter.get());
    if (!lowerable) { continue; }
    auto const& col_ref = tree.emplace<cudf::ast::column_name_reference>(parquet_col_name);
    root                = and_into(tree, root, lowerable->to_ast(tree, col_ref, device_id));
  }
  return root;
}

namespace {
using detail::application_program_state;
using detail::application_result_state;
using detail::selected_step;

bool contains(std::vector<applied_entry> const& entries, applied_entry const& identity)
{
  return std::ranges::find(entries, identity) != entries.end();
}

template <typename Filter>
  requires requires(Filter const& filter) { filter.domain(); }
bool accepts_carrier(sirius::op::sirius_dynamic_filter const& filter, cudf::data_type type)
{
  auto const* concrete = dynamic_cast<Filter const*>(&filter);
  return !concrete || sirius::op::membership_probe_compatible(concrete->domain(), type);
}

bool accepts_carrier(sirius::op::sirius_dynamic_filter const& filter, cudf::data_type type)
{
  if (auto const* zone = dynamic_cast<sirius::op::sirius_dynamic_zone_map_filter const*>(&filter)) {
    return !zone->zones().empty() && zone->zones().front().min->type() == type;
  }
  return accepts_carrier<sirius::op::sirius_dynamic_small_in_list_filter>(filter, type) &&
         accepts_carrier<sirius::op::sirius_dynamic_in_list_filter>(filter, type) &&
         accepts_carrier<sirius::op::sirius_dynamic_bloom_filter>(filter, type);
}

/**
 * @brief Expected strength of a membership filter before any measurement, strongest first.
 *
 * Follows the `sirius::membership_probe` ordering signal: exact set forms keep fewer rows than a
 * Bloom filter over the same keys, and fewer build keys keep fewer rows.
 */
struct static_selectivity {
  /// 0 for a small IN-list, 1 for a hash IN-list, 2 for Bloom; other kinds sort last.
  std::uint8_t kind_rank = std::numeric_limits<std::uint8_t>::max();
  /// Build-side key count; an unknown count sorts last.
  std::uint64_t num_keys = std::numeric_limits<std::uint64_t>::max();

  [[nodiscard]] auto operator<=>(static_selectivity const&) const = default;
};

[[nodiscard]] static_selectivity static_selectivity_of(
  sirius::op::sirius_dynamic_filter const& filter)
{
  if (auto const* small =
        dynamic_cast<sirius::op::sirius_dynamic_small_in_list_filter const*>(&filter)) {
    return {0, small->size()};
  }
  if (auto const* set = dynamic_cast<sirius::op::sirius_dynamic_in_list_filter const*>(&filter)) {
    return {1, set->size()};
  }
  if (filter.kind() == sirius::op::sirius_dynamic_filter_kind::BLOOM) { return {.kind_rank = 2}; }
  return {};
}

struct candidate {
  applied_entry identity;
  cudf::size_type column_index;
  std::optional<double> expected_keep;
  std::size_t snapshot_index;
  static_selectivity prior = {};  ///< Filled only for decode selection.
};

std::vector<candidate> capture_candidates(sirius::op::dynamic_filter_snapshot const& snapshot,
                                          dynamic_filter_gate const& gate,
                                          std::span<binding const> bindings,
                                          gate_selection& selection)
{
  if (snapshot.empty()) {
    selection = {};
    return {};
  }
  std::vector<candidate> candidates;
  std::vector<bound_filter_key> keys;
  std::size_t snapshot_index = 0;
  for (auto const& [ordinal, filter] : snapshot.entries()) {
    auto const index  = snapshot_index++;
    auto const mapped = std::ranges::find(bindings, ordinal, &binding::output_ordinal);
    if (mapped == bindings.end() || mapped->column_index < 0) { continue; }
    candidates.push_back({{filter, ordinal}, mapped->column_index, std::nullopt, index});
    keys.push_back({filter.get(), ordinal});
  }
  std::vector<std::optional<double>> estimates(keys.size());
  selection = gate.capture(snapshot, keys, estimates);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (is_known_keep_ratio(estimates[i])) { candidates[i].expected_keep = estimates[i]; }
  }
  return candidates;
}

template <typename Step>
bool in_normal_order(Step const& a, Step const& b)
{
  if (a.expected_keep.has_value() != b.expected_keep.has_value()) {
    return a.expected_keep.has_value();
  }
  if (a.expected_keep && *a.expected_keep != *b.expected_keep) {
    return *a.expected_keep < *b.expected_keep;
  }
  if (a.snapshot_index != b.snapshot_index) { return a.snapshot_index < b.snapshot_index; }
  return a.identity.output_ordinal < b.identity.output_ordinal;
}

/// @brief Decode order: measured keep ratios first, then the static signal for ties and unmeasured
/// filters.
bool in_decode_order(candidate const& a, candidate const& b)
{
  if (a.expected_keep != b.expected_keep || a.prior == b.prior) { return in_normal_order(a, b); }
  return a.prior < b.prior;
}

bool same_value(candidate const& a, candidate const& b)
{
  return a.identity.filter == b.identity.filter && a.column_index == b.column_index;
}

/// @brief Groups candidates that probe the same filter on the same input column, in order of first
/// appearance; each group keeps its members in input order.
std::vector<std::vector<candidate const*>> group_aliases(std::span<candidate const> candidates)
{
  std::vector<std::vector<candidate const*>> groups;
  for (auto const& current : candidates) {
    auto group = std::ranges::find_if(
      groups, [&](auto const& aliases) { return same_value(*aliases.front(), current); });
    if (group == groups.end()) {
      groups.push_back({&current});
    } else {
      group->push_back(&current);
    }
  }
  return groups;
}

/**
 * @brief Selects and orders this batch's filter applications.
 *
 * @p history_known is false when the input may carry decode history it does not report: the steps
 * still execute, but none of them samples, and the batch makes no combined observation.
 */
std::unique_ptr<application_program_state> prepare_state(
  sirius::op::dynamic_filter_snapshot snapshot,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set const> endpoint,
  dynamic_filter_gate const& gate,
  consumer_config const& config,
  application_input const& input,
  bool history_known)
{
  if (input.source.num_rows() == 0 || input.source.num_columns() == 0) { return nullptr; }
  gate_selection selection{};
  auto candidates = capture_candidates(snapshot, gate, input.bindings, selection);
  if (!selection.applicable) { return nullptr; }
  auto program              = std::make_unique<application_program_state>();
  program->snapshot         = std::move(snapshot);
  program->receipt.endpoint = endpoint;
  if (input.prior && input.prior->endpoint == endpoint) { program->receipt = *input.prior; }
  program->receipt.applied.reserve(program->receipt.applied.size() + candidates.size());
  program->rows      = input.source.num_rows();
  program->device_id = sirius::op::detail::resolve_dynamic_filter_device_id(input.device_id);
  program->sample_combined =
    history_known && selection.needs_combined_sample && program->receipt.decode_attached.empty();
  program->validate = config.validate_deferred_selection;
  for (auto const& column : input.source) {
    program->types.push_back(column.type());
  }
  auto const width = average_row_width(static_cast<std::size_t>(program->rows), input.input_bytes);
  for (auto const& aliases : group_aliases(candidates)) {
    auto const& current = *aliases.front();
    if (current.column_index >= input.source.num_columns()) { continue; }
    auto const covered_by = [&](std::vector<applied_entry> const& history) {
      return std::ranges::any_of(
        aliases, [&](candidate const* alias) { return contains(history, alias->identity); });
    };
    if (covered_by(program->receipt.applied) ||
        !current.identity.filter->is_available_on_device(program->device_id) ||
        !accepts_carrier(*current.identity.filter, program->types[current.column_index])) {
      continue;
    }
    auto const* mask =
      dynamic_cast<sirius::op::sirius_mask_applicable const*>(current.identity.filter.get());
    auto const* ast =
      config.mode == dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS
        ? dynamic_cast<sirius::op::sirius_ast_lowerable const*>(current.identity.filter.get())
        : nullptr;
    if (!mask && !ast) { continue; }
    std::vector<applied_entry> represented;
    candidate const* representative = nullptr;
    for (auto const* alias : aliases) {
      if (mask && alias->expected_keep && skip_measured_filter(*alias->expected_keep, width)) {
        continue;
      }
      if (!representative || in_normal_order(*alias, *representative)) { representative = alias; }
      represented.push_back(alias->identity);
    }
    if (!representative) { continue; }
    selected_step step{.identity      = representative->identity,
                       .column_index  = current.column_index,
                       .represented   = std::move(represented),
                       .mask_source   = mask,
                       .ast_source    = ast,
                       .expected_keep = representative->expected_keep,
                       .sample        = history_known && mask &&
                                 !covered_by(program->receipt.decode_attached) &&
                                 !representative->expected_keep,
                       .snapshot_index = representative->snapshot_index};
    if (mask) {
      program->membership.push_back(std::move(step));
    } else {
      program->ast.push_back(std::move(step));
    }
  }
  std::stable_sort(
    program->membership.begin(), program->membership.end(), in_normal_order<selected_step>);
  if (program->membership.empty() && program->ast.empty()) { return nullptr; }
  std::vector<std::optional<double>> estimates;
  for (auto const& step : program->membership) {
    estimates.push_back(step.expected_keep);
  }
  program->strategy = config.strategy_override.value_or(
    evaluate_compaction_policy({static_cast<std::size_t>(program->rows),
                                input.input_bytes,
                                !program->ast.empty(),
                                estimates})
      .strategy);
  // The combined observation reads only the input and output row counts, which the host knows.
  program->want_counts =
    config.force_counts || std::ranges::any_of(program->membership, &selected_step::sample);
  return program;
}

/// @brief Whether the AST mask or any membership mask selected rows in this application.
bool any_contributed(application_result_state const& result)
{
  return result.ast_contributed || std::ranges::any_of(result.contributed, std::identity{});
}

/// @brief Records the marginal of every sampled membership step that executed on a nonempty input.
void complete_observations(application_result_state& result)
{
  if (result.counts.empty()) { return; }
  auto const& program = *result.program;
  for (std::size_t i = 0; i < program.membership.size(); ++i) {
    auto const& step  = program.membership[i];
    auto const before = result.counts[i];
    if (!result.contributed[i] || !step.sample || before == 0) { continue; }
    result.observations.push_back(
      {{step.identity.filter.get(), step.identity.output_ordinal}, before, result.counts[i + 1]});
  }
}

/**
 * @brief Adds the bindings this application proved it ran to the program's receipt.
 *
 * A membership step runs when rows enter it. Without prefix counts only a nonempty output proves
 * that, so an empty output claims no membership coverage. Entries fill the capacity `prepare_state`
 * reserved; an entry that would not fit is dropped, which only loses optional coverage.
 */
void add_receipt_coverage(application_result_state& result) noexcept
{
  auto const& program = *result.program;
  auto& applied       = result.program->receipt.applied;
  auto const add      = [&](selected_step const& step) noexcept {
    for (auto const& identity : step.represented) {
      if (!contains(applied, identity) && applied.size() < applied.capacity()) {
        applied.push_back(identity);
      }
    }
  };
  if (result.ast_contributed) { std::ranges::for_each(program.ast, add); }
  for (std::size_t i = 0; i < program.membership.size(); ++i) {
    auto const ran = result.counts.empty() ? result.final_rows > 0 : result.counts[i] > 0;
    if (result.contributed[i] && ran) { add(program.membership[i]); }
  }
}

/// @brief Releases the source and every scratch owner, keeping what `commit` and `finish` read.
void release_execution_storage(application_result_state& result) noexcept
{
  result.source.reset();
  result.test_source.reset();
  result.tree.reset();
  result.ast_mask.reset();
  result.mask.reset();
  result.conjunction.reset();
  result.zero.reset();
  result.scratch_table.reset();
  result.columns.clear();
  result.output_columns.clear();
  result.validation_ids.clear();
  result.deferred = {};
  result.counters.reset();
  result.readback.reset();
}

//===----------CASCADE compaction strategy----------===//
/// @brief Apply the CASCADE compaction strategy.
void apply_cascade(application_result_state& result,
                   cudf::table_view input,
                   ::cuda::stream_ref stream,
                   rmm::device_async_resource_ref mr)
{
  auto current = input;
  if (result.ast_mask) {
    result.output = sirius::ApplyRetentionMask(current, result.ast_mask->view(), stream, mr);
    current       = result.output->view();
  }
  result.counts[0] = current.num_rows();
  for (std::size_t i = 0; i < result.program->membership.size(); ++i) {
    auto const& step = result.program->membership[i];
    if (current.num_rows() != 0) {
      result.mask = step.mask_source->compute_mask(
        current.column(step.column_index), result.program->device_id, stream, mr);
      if (result.mask) {
        result.output = sirius::ApplyRetentionMask(current, result.mask->view(), stream, mr);
        current       = result.output->view();
        result.contributed[i] = true;
      }
    }
    result.counts[i + 1] = current.num_rows();
  }
}

//===----------DEFERRED_KEYS compaction strategy----------===//
void compact_row_ids(application_result_state& result,
                     cudf::column_view mask,
                     ::cuda::stream_ref stream,
                     rmm::device_async_resource_ref mr)
{
  result.scratch_table = sirius::ApplyRetentionMask(
    cudf::table_view({result.deferred.row_ids->view()}), mask, stream, mr);
  result.columns          = result.scratch_table->release();
  result.deferred.row_ids = std::move(result.columns.front());
}

/// @brief Apply the DEFERRED_KEYS compaction strategy.
///
/// A step's compacted key is kept only when the next step probes the same column or no step
/// follows (materialization then reuses it); otherwise only the row IDs are compacted.
void apply_deferred_keys(application_result_state& result,
                         cudf::table_view input,
                         ::cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
{
  auto& state = result.deferred;
  // The testing entry point enables invariant checks after each deferred-selection transition.
  // Production disables these GPU-to-host copies and stream synchronizations.
  auto validate = [&] {
    if (result.program->validate) {
      validate_selection_state(state, input.num_rows(), stream, result.validation_ids);
    }
  };
  result.zero   = std::make_unique<cudf::numeric_scalar<cudf::size_type>>(0, true, stream, mr);
  state.row_ids = cudf::sequence(input.num_rows(), *result.zero, stream, mr);
  validate();
  if (result.ast_mask) {
    compact_row_ids(result, result.ast_mask->view(), stream, mr);
    validate();
  }
  result.counts[0]  = state.row_ids->size();
  auto const& steps = result.program->membership;
  for (std::size_t i = 0; i < steps.size(); ++i) {
    auto const& step = steps[i];
    if (state.row_ids->size() != 0) {
      auto const key            = input.column(step.column_index);
      auto const identity_space = !state.aligned_key && state.row_ids->size() == input.num_rows();
      if (!identity_space &&
          (!state.aligned_key ||
           state.aligned_key->index != static_cast<std::size_t>(step.column_index))) {
        result.scratch_table = cudf::gather(cudf::table_view({key}),
                                            state.row_ids->view(),
                                            cudf::out_of_bounds_policy::DONT_CHECK,
                                            stream,
                                            mr);
        result.columns       = result.scratch_table->release();
        state.aligned_key = detail::aligned_key_column{static_cast<std::size_t>(step.column_index),
                                                       std::move(result.columns.front())};
        validate();
      }
      auto const probe = state.aligned_key ? state.aligned_key->column->view() : key;
      result.mask = step.mask_source->compute_mask(probe, result.program->device_id, stream, mr);
      if (result.mask) {
        auto const keep_key =
          i + 1 == steps.size() || steps[i + 1].column_index == step.column_index;
        if (keep_key) {
          result.scratch_table = sirius::ApplyRetentionMask(
            cudf::table_view({probe, state.row_ids->view()}), result.mask->view(), stream, mr);
          result.columns    = result.scratch_table->release();
          state.aligned_key = detail::aligned_key_column{
            static_cast<std::size_t>(step.column_index), std::move(result.columns[0])};
          state.row_ids = std::move(result.columns[1]);
        } else {
          compact_row_ids(result, result.mask->view(), stream, mr);
          state.aligned_key.reset();
        }
        result.contributed[i] = true;
        validate();
      }
    }
    result.counts[i + 1] = state.row_ids->size();
  }
  if (!any_contributed(result)) { return; }
  if (!state.aligned_key) {
    result.output = cudf::gather(
      input, state.row_ids->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
    return;
  }
  auto const key_index = state.aligned_key->index;
  std::vector<cudf::size_type> payload_indices;
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    if (std::cmp_not_equal(i, key_index)) { payload_indices.push_back(i); }
  }
  result.output_columns.resize(input.num_columns());
  if (!payload_indices.empty()) {
    result.scratch_table = cudf::gather(input.select(payload_indices),
                                        state.row_ids->view(),
                                        cudf::out_of_bounds_policy::DONT_CHECK,
                                        stream,
                                        mr);
    result.columns       = result.scratch_table->release();
    for (std::size_t i = 0; i < payload_indices.size(); ++i) {
      result.output_columns[payload_indices[i]] = std::move(result.columns[i]);
    }
  }
  result.output_columns[key_index] = std::move(state.aligned_key->column);
  result.output = std::make_unique<cudf::table>(std::move(result.output_columns));
}

//===----------GATHER_ONCE compaction strategy----------===//
/// @brief Apply the GATHER_ONCE compaction strategy.
void apply_gather_once(application_result_state& result,
                       cudf::table_view input,
                       ::cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr)
{
  auto const& program = *result.program;
  // One payload selection already supplies the only needed marginal without a count copy.
  if (!result.ast_mask && program.membership.size() == 1) {
    result.counts.resize(2);
    apply_cascade(result, input, stream, mr);
    return;
  }
  auto const size = program.membership.size() + 1;
  // Without an AST mask, every input row enters the first membership step: the host already knows
  // that count, and the first membership fold initializes the conjunction.
  auto const device_counts = result.ast_mask ? size : size - 1;
  if (program.want_counts) {
    result.counters.emplace(device_counts, stream, mr);
    RMM_CUDA_TRY(cudaMemsetAsync(
      result.counters->data(), 0, device_counts * sizeof(cudf::size_type), stream.get()));
  }
  auto* const counts = result.counters ? result.counters->data() : nullptr;
  result.conjunction = cudf::make_numeric_column(cudf::data_type{cudf::type_id::BOOL8},
                                                 input.num_rows(),
                                                 cudf::mask_state::UNALLOCATED,
                                                 stream,
                                                 mr);
  if (result.ast_mask) {
    sirius::op::fold_dynamic_filter_mask(
      result.conjunction->mutable_view(), result.ast_mask->view(), counts, true, stream);
  }
  auto* const membership_counts = counts && result.ast_mask ? counts + 1 : counts;
  for (std::size_t i = 0; i < program.membership.size(); ++i) {
    auto const& step = program.membership[i];
    result.mask      = step.mask_source->compute_mask(
      input.column(step.column_index), program.device_id, stream, mr);
    result.contributed[i] = result.mask != nullptr;
    sirius::op::fold_dynamic_filter_mask(
      result.conjunction->mutable_view(),
      result.mask ? std::optional{result.mask->view()} : std::nullopt,
      membership_counts ? membership_counts + i : nullptr,
      i == 0 && !result.ast_mask,
      stream);
  }
  if (program.want_counts) {
    result.readback = std::make_unique<detail::pinned_counts>(device_counts, stream);
    result.counts.resize(size);
    if (!result.ast_mask) { result.counts[0] = input.num_rows(); }
    RMM_CUDA_TRY(cudaMemcpyAsync(result.readback->data(),
                                 result.counters->data(),
                                 device_counts * sizeof(cudf::size_type),
                                 cudaMemcpyDeviceToHost,
                                 stream.get()));
  }
  if (any_contributed(result)) {
    result.output = sirius::ApplyRetentionMask(input, result.conjunction->view(), stream, mr);
  }
}

/// @brief Production execution: every fault hook compiles to nothing.
struct no_execution_faults {
  static constexpr bool forces_join() noexcept { return false; }
  static constexpr void after_submission(application_result_state const&,
                                         ::cuda::stream_ref) noexcept
  {
  }
  static constexpr bool fails_join(bool) noexcept { return false; }
  static constexpr void retain(std::unique_ptr<application_result_state>&) noexcept {}
};

/**
 * @brief Waits for the work submitted on @p stream.
 *
 * If the wait fails, GPU work may still read the storage @p result owns, so that storage is leaked
 * (or, after a simulated failure that did join, handed to @p faults) and `unjoined_gpu_work` is
 * thrown. During @p cleanup it nests the exception being handled.
 */
template <typename Faults>
void join_or_leak(std::unique_ptr<application_result_state>& result,
                  ::cuda::stream_ref stream,
                  Faults& faults,
                  bool cleanup)
{
  auto const status = cudaStreamSynchronize(stream.get());
  if (status == cudaSuccess && !faults.fails_join(cleanup)) { return; }
  if (status == cudaSuccess) { faults.retain(result); }
  (void)result.release();
  if (cleanup) { std::throw_with_nested(sirius::op::detail::unjoined_gpu_work{}); }
  throw sirius::op::detail::unjoined_gpu_work{};
}

void validate_program_input(application_program_state const& program, cudf::table_view input)
{
  if (input.num_rows() != program.rows ||
      static_cast<std::size_t>(input.num_columns()) != program.types.size()) {
    throw std::invalid_argument("dynamic-filter program/source shape mismatch");
  }
  for (cudf::size_type i = 0; i < input.num_columns(); ++i) {
    if (input.column(i).type() != program.types[i]) {
      throw std::invalid_argument("dynamic-filter program/source type mismatch");
    }
  }
}

/**
 * @brief The device other than the current one that holds @p result's source or that its program
 * was prepared for, if any
 *
 * A test source held as a bare table has no memory space and lives on the current device.
 */
[[nodiscard]] std::optional<int> other_device_of(application_result_state const& result) noexcept
{
  auto const current = sirius::op::detail::resolve_dynamic_filter_device_id(-1);
  if (result.source) {
    if (auto const* space = result.source->get_memory_space();
        space && space->get_device_id() != current) {
      return space->get_device_id();
    }
  }
  if (result.program->device_id != current) { return result.program->device_id; }
  return std::nullopt;
}

template <typename Faults>
std::unique_ptr<application_result_state> execute_program(
  std::unique_ptr<application_result_state> result,
  cudf::table_view input,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  Faults& faults)
{
  nvtx_scoped_range nvtx_range{"dynfilter::apply_output"};
  auto const& program = *result->program;
  result->contributed.resize(program.membership.size(), false);
  result->observations.reserve(program.membership.size());
  if (program.strategy != detail::compaction_strategy::GATHER_ONCE) {
    result->counts.resize(program.membership.size() + 1);
  }
  result->tree = std::make_unique<cudf::ast::tree>();
  try {
    cudf::ast::expression const* root = nullptr;
    for (auto const& step : program.ast) {
      auto const& reference = result->tree->emplace<cudf::ast::column_reference>(step.column_index);
      root                  = and_into(
        *result->tree, root, step.ast_source->to_ast(*result->tree, reference, program.device_id));
    }
    if (root) {
      // Cross-column AST masks update only the scan-level gate.
      result->ast_mask        = cudf::compute_column(input, *root, stream, mr);
      result->ast_contributed = result->ast_mask != nullptr;
    }
    switch (program.strategy) {
      case detail::compaction_strategy::CASCADE: apply_cascade(*result, input, stream, mr); break;
      case detail::compaction_strategy::DEFERRED_KEYS:
        apply_deferred_keys(*result, input, stream, mr);
        break;
      case detail::compaction_strategy::GATHER_ONCE:
        apply_gather_once(*result, input, stream, mr);
        break;
    }
    faults.after_submission(*result, stream);
    // Without a count readback the host does not wait for these reads, so the source's next mutable
    // accessor or its destruction must.
    if (!result->readback && result->source) { result->source->record_reader_event(stream); }
  } catch (...) {
    join_or_leak(result, stream, faults, true);
    throw;
  }
  // Only the count readback needs the host to wait. Scratch is freed in stream order.
  if (result->readback || faults.forces_join()) {
    join_or_leak(result, stream, faults, false);
    if (result->readback) {
      // The device counted the trailing prefix counts; apply_gather_once filled any leading one.
      auto const copied = result->readback->size();
      std::copy_n(result->readback->data(),
                  copied,
                  result->counts.end() - static_cast<std::ptrdiff_t>(copied));
      result->counts_read_back = true;
    }
  }
  result->completed  = true;
  result->final_rows = result->output ? result->output->num_rows() : input.num_rows();
  complete_observations(*result);
  release_execution_storage(*result);
  return result;
}

void commit_observations(application_result_state const& result, dynamic_filter_gate& gate) noexcept
{
  if (!result.completed) { return; }
  auto const& program = *result.program;
  for (auto const& observation : result.observations) {
    gate.record_filter_keep_ratio(observation.key,
                                  static_cast<double>(observation.after) / observation.before,
                                  program.snapshot.generation());
  }
  if (program.sample_combined && any_contributed(result)) {
    gate.record_keep_ratio(program.rows, result.final_rows, program.snapshot.generation());
  }
}
}  // namespace

//===----------------------------------------------------------------------===//
// application_program
//===----------------------------------------------------------------------===//
application_program::application_program(
  std::unique_ptr<detail::application_program_state> state) noexcept
  : _state(std::move(state))
{
}
application_program::application_program(application_program&&) noexcept            = default;
application_program& application_program::operator=(application_program&&) noexcept = default;
application_program::~application_program()                                         = default;
application_result::application_result(
  std::unique_ptr<detail::application_result_state> state) noexcept
  : _state(std::move(state))
{
}

//===----------------------------------------------------------------------===//
// application_result
//===----------------------------------------------------------------------===//
application_result::application_result(application_result&&) noexcept            = default;
application_result& application_result::operator=(application_result&&) noexcept = default;
application_result::~application_result()                                        = default;
std::unique_ptr<cudf::table> application_result::take_output() noexcept
{
  return _state ? std::move(_state->output) : nullptr;
}
std::span<cudf::size_type const> application_result::prefix_counts() const noexcept
{
  return _state ? std::span<cudf::size_type const>{_state->counts}
                : std::span<cudf::size_type const>{};
}
bool application_result::counts_read_back() const noexcept
{
  return _state && _state->counts_read_back;
}
std::optional<detail::compaction_strategy> application_result::strategy() const noexcept
{
  return _state ? std::optional{_state->program->strategy} : std::nullopt;
}

//===----------------------------------------------------------------------===//
// dynamic_filter_consumer
//===----------------------------------------------------------------------===//
dynamic_filter_consumer::dynamic_filter_consumer(
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel,
  consumer_config config,
  std::vector<binding> decode_bindings)
  : _channel(std::move(channel)),
    _config(config),
    _gate(config.keep_threshold),
    _decode_bindings(std::move(decode_bindings))
{
  if (!_channel) { throw std::invalid_argument("dynamic-filter consumer requires a channel"); }
}
std::shared_ptr<sirius::op::sirius_dynamic_filter_set> const& dynamic_filter_consumer::channel()
  const noexcept
{
  return _channel;
}

std::span<binding const> dynamic_filter_consumer::decode_bindings() const noexcept
{
  return _decode_bindings;
}

decode_selection dynamic_filter_consumer::select_for_decode(std::span<binding const> bindings) const
{
  auto snapshot = _channel->snapshot();
  gate_selection selection{};
  auto candidates = capture_candidates(snapshot, _gate, bindings, selection);
  decode_selection result{_channel, snapshot.generation(), {}};
  if (!selection.applicable || selection.needs_combined_sample) { return result; }
  for (auto& current : candidates) {
    current.prior = static_selectivity_of(*current.identity.filter);
  }
  std::stable_sort(candidates.begin(), candidates.end(), in_decode_order);
  std::erase_if(candidates, [](candidate const& current) {
    return !dynamic_cast<sirius::op::sirius_mask_applicable const*>(
             current.identity.filter.get()) ||
           (current.expected_keep && skip_measured_filter(*current.expected_keep, std::nullopt));
  });
  // Sorting first makes each group's first member its preferred representative.
  for (auto const& aliases : group_aliases(candidates)) {
    decode_step step{aliases.front()->identity, aliases.front()->column_index, {}};
    step.represented.reserve(aliases.size());
    for (auto const* alias : aliases) {
      step.represented.push_back(alias->identity);
    }
    result.steps.push_back(std::move(step));
  }
  return result;
}

std::optional<application_program> dynamic_filter_consumer::prepare(
  application_input const& input) const
{
  // A scan endpoint whose decode may attach probes reports that history in the scan's receipt, so
  // an input without one has an unknown history, not an empty one.
  auto const history_known =
    (input.prior && input.prior->endpoint == _channel) || decode_bindings().empty();
  auto state = prepare_state(_channel->snapshot(), _channel, _gate, _config, input, history_known);
  if (!state) { return std::nullopt; }
  return application_program(std::move(state));
}

template <typename Faults>
application_result dynamic_filter_consumer::run(application_program&& program,
                                                std::unique_ptr<application_result_state> state,
                                                std::optional<cudf::table_view> input_view,
                                                ::cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr,
                                                Faults& faults) const
{
  if (!program._state || program._state->receipt.endpoint != _channel) {
    throw std::invalid_argument("dynamic-filter program belongs to another consumer");
  }
  state->program   = std::move(program._state);
  auto const input = input_view           ? *input_view
                     : state->test_source ? state->test_source->view()
                                          : sirius::get_cudf_table_view(*state->source);
  validate_program_input(*state->program, input);
  // The stream and memory resource belong to the current device. Filtering is optional, so a batch
  // held on another device passes through unfiltered rather than failing the query or reading that
  // device's memory; an incomplete result records no observation and claims no coverage.
  if (auto const other_device = other_device_of(*state)) {
    if (_config.stats) {
      _config.stats->applications_skipped_other_device.fetch_add(1, std::memory_order_relaxed);
    }
    if (!_reported_other_device.exchange(true, std::memory_order_relaxed)) {
      SIRIUS_LOG_WARN(
        "[dynamic_filter_consumer] a batch on device {} passed through unfiltered because the "
        "current device is {}; this consumer logs no further such batches",
        *other_device,
        sirius::op::detail::resolve_dynamic_filter_device_id(-1));
    }
    release_execution_storage(*state);
    return application_result(std::move(state));
  }
  return application_result(execute_program(std::move(state), input, stream, mr, faults));
}

application_result dynamic_filter_consumer::apply(application_program&& program,
                                                  cucascade::read_only_data_batch source,
                                                  ::cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr) const
{
  auto state = std::make_unique<application_result_state>();
  state->source.emplace(std::move(source));
  no_execution_faults faults;
  return run(std::move(program), std::move(state), std::nullopt, stream, mr, faults);
}

void dynamic_filter_consumer::commit(application_result&& result) noexcept
{
  auto const state = std::move(result._state);
  if (state && state->program->receipt.endpoint == _channel) { commit_observations(*state, _gate); }
}

batch_receipt dynamic_filter_consumer::finish(application_result&& result,
                                              std::uint64_t original_batch_id) noexcept
{
  auto state = std::move(result._state);
  if (!state || !state->completed || state->program->receipt.endpoint != _channel) { return {}; }
  commit_observations(*state, _gate);
  add_receipt_coverage(*state);
  state->program->receipt.original_batch_id = original_batch_id;
  return std::move(state->program->receipt);
}

dynamic_filter_gate& detail::consumer_test_access::gate(dynamic_filter_consumer& consumer) noexcept
{
  return consumer._gate;
}
detail::retained_storage detail::consumer_test_access::storage(
  application_result const& result) noexcept
{
  if (!result._state) { return {}; }
  auto const& state = *result._state;
  return {state.source.has_value() || state.test_source != nullptr,
          state.program ? state.program->snapshot.entries().size() : 0,
          state.tree != nullptr && state.ast_mask != nullptr,
          state.mask != nullptr,
          state.output != nullptr,
          state.deferred.row_ids != nullptr,
          state.counters.has_value(),
          state.completed,
          state.readback
            ? std::span<cudf::size_type const>{state.readback->data(), state.readback->size()}
            : std::span<cudf::size_type const>{}};
}

namespace {
/// @brief Whether streams held by `execution_fault::STALLED_COMPLETION` may continue.
std::atomic<bool> stalled_streams_released{true};
/// @brief Whether a stalled stream continued because its deadline passed.
std::atomic<bool> stalled_stream_deadline_passed{false};

/// @brief Longest a stalled stream waits for `release_stalled_streams`.
constexpr std::chrono::seconds stalled_stream_deadline{5};

/**
 * @brief Host function that holds its stream until `release_stalled_streams` is called or
 * `stalled_stream_deadline` passes
 *
 * The deadline turns a wait for the stalled stream inside the code under test into a test failure
 * instead of a hang.
 */
void CUDART_CB wait_for_stalled_stream_release(void*)
{
  auto const deadline = std::chrono::steady_clock::now() + stalled_stream_deadline;
  while (!stalled_streams_released.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      stalled_stream_deadline_passed.store(true, std::memory_order_release);
      return;
    }
    std::this_thread::yield();
  }
}

/// @brief Test execution: injects one `detail::execution_fault` and keeps storage handed back after
/// a simulated failure.
class injected_execution_faults {
 public:
  explicit injected_execution_faults(detail::execution_fault fault) noexcept : _fault(fault) {}

  [[nodiscard]] bool forces_join() const noexcept
  {
    return _fault == detail::execution_fault::FAILED_COMPLETION;
  }

  void after_submission(application_result_state const& result, ::cuda::stream_ref stream)
  {
    // A readback would wait for the stalled stream, so only a program without one stalls.
    if (_fault == detail::execution_fault::STALLED_COMPLETION && !result.readback) {
      stalled_stream_deadline_passed.store(false, std::memory_order_release);
      stalled_streams_released.store(false, std::memory_order_release);
      if (cudaLaunchHostFunc(stream.get(), wait_for_stalled_stream_release, nullptr) !=
          cudaSuccess) {
        stalled_streams_released.store(true, std::memory_order_release);
        throw std::runtime_error("could not stall the dynamic-filter stream");
      }
    }
    if (_fault == detail::execution_fault::AFTER_COUNTS_COPY && result.readback) {
      throw std::runtime_error("injected dynamic-filter failure after count copy");
    }
    if (_fault == detail::execution_fault::FAILED_CLEANUP) {
      _failed_before_cleanup = true;
      throw rmm::out_of_memory{"injected dynamic-filter failure before cleanup"};
    }
  }

  [[nodiscard]] bool fails_join(bool cleanup) const noexcept
  {
    return cleanup ? _failed_before_cleanup : _fault == detail::execution_fault::FAILED_COMPLETION;
  }

  void retain(std::unique_ptr<application_result_state>& result) noexcept
  {
    _retained = std::move(result);
  }

  [[nodiscard]] std::unique_ptr<application_result_state> take_retained() noexcept
  {
    return std::move(_retained);
  }

 private:
  detail::execution_fault _fault;
  bool _failed_before_cleanup = false;
  std::unique_ptr<application_result_state> _retained;
};
}  // namespace

void detail::consumer_test_access::release_stalled_streams() noexcept
{
  stalled_streams_released.store(true, std::memory_order_release);
}

bool detail::consumer_test_access::stalled_stream_timed_out() noexcept
{
  return stalled_stream_deadline_passed.load(std::memory_order_acquire);
}

application_result detail::consumer_test_access::run(
  dynamic_filter_consumer const& consumer,
  application_program&& program,
  std::unique_ptr<application_result_state> state,
  std::optional<cudf::table_view> input_view,
  ::cuda::stream_ref stream,
  rmm::device_async_resource_ref mr,
  execution_fault fault,
  std::optional<application_result>* retained)
{
  injected_execution_faults faults{fault};
  try {
    return consumer.run(std::move(program), std::move(state), input_view, stream, mr, faults);
  } catch (...) {
    if (auto handed_back = faults.take_retained(); handed_back && retained) {
      retained->emplace(application_result(std::move(handed_back)));
    }
    throw;
  }
}

application_result detail::consumer_test_access::apply(dynamic_filter_consumer const& consumer,
                                                       application_program&& program,
                                                       cucascade::read_only_data_batch source,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr,
                                                       execution_fault fault,
                                                       std::optional<application_result>* retained)
{
  auto state = std::make_unique<application_result_state>();
  state->source.emplace(std::move(source));
  return run(
    consumer, std::move(program), std::move(state), std::nullopt, stream, mr, fault, retained);
}

application_result detail::consumer_test_access::apply(dynamic_filter_consumer const& consumer,
                                                       application_program&& program,
                                                       std::shared_ptr<cudf::table const> source,
                                                       ::cuda::stream_ref stream,
                                                       rmm::device_async_resource_ref mr,
                                                       execution_fault fault,
                                                       std::optional<application_result>* retained,
                                                       std::optional<cudf::table_view> input_view)
{
  if (!source) { throw std::invalid_argument("invalid test application input"); }
  auto state         = std::make_unique<application_result_state>();
  state->test_source = std::move(source);
  return run(
    consumer, std::move(program), std::move(state), input_view, stream, mr, fault, retained);
}

std::shared_ptr<dynamic_filter_consumer> make_dynamic_filter_consumer(
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> channel,
  dynamic_filter_apply_mode mode,
  double keep_threshold,
  std::vector<binding> decode_bindings,
  sirius::op::dynamic_filter_stats* stats)
{
  return std::make_shared<dynamic_filter_consumer>(
    std::move(channel),
    consumer_config{.mode = mode, .keep_threshold = keep_threshold, .stats = stats},
    std::move(decode_bindings));
}

std::vector<binding> identity_bindings(std::size_t count)
{
  std::vector<binding> bindings;
  bindings.reserve(count);
  for (std::size_t ordinal = 0; ordinal < count; ++ordinal) {
    bindings.push_back({ordinal, static_cast<cudf::size_type>(ordinal)});
  }
  return bindings;
}

//===----------------------------------------------------------------------===//
// dynamic_filter_gate
//===----------------------------------------------------------------------===//
std::optional<double> dynamic_filter_gate::current_keep_ratio(
  bound_filter_key filter, std::size_t observed_filter_count) const
{
  auto const it = _filter_keep_ratios.find(filter);
  if (it == _filter_keep_ratios.end()) { return std::nullopt; }
  // Skipping an optional filter cannot affect correctness, so its verdict is permanent.
  if (filter_skippable(it->second.kept)) { return it->second.kept; }
  // New filters can change this filter's marginal selectivity.
  if (it->second.observed_filter_count < observed_filter_count) { return std::nullopt; }
  return it->second.kept;
}

void dynamic_filter_gate::record_filter_keep_ratio(bound_filter_key filter,
                                                   double kept,
                                                   std::size_t observed_filter_count) noexcept
{
  if (!is_known_keep_ratio(kept)) { return; }
  try {
    std::scoped_lock lock(_filter_ratios_mu);
    auto const it = _filter_keep_ratios.find(filter);
    if (it != _filter_keep_ratios.end() &&
        it->second.observed_filter_count >= observed_filter_count) {
      return;
    }
    _filter_keep_ratios.insert_or_assign(filter, filter_measurement{kept, observed_filter_count});
  } catch (...) {
    // Optional hints cannot invalidate an already constructed output.
  }
}

gate_selection dynamic_filter_gate::capture(sirius::op::dynamic_filter_snapshot const& filters,
                                            std::span<bound_filter_key const> keys,
                                            std::span<std::optional<double>> estimates) const
{
  if (keys.size() != estimates.size()) {
    throw std::invalid_argument("dynamic-filter gate capture size mismatch");
  }
  if (filters.empty()) {
    std::ranges::fill(estimates, std::nullopt);
    return {};
  }
  std::scoped_lock lock(_decision_mu, _filter_ratios_mu);
  auto const current = _state.load(std::memory_order_relaxed);
  auto const applicable =
    current != state::DISABLED ||
    filters.generation() > _decided_filter_count.load(std::memory_order_relaxed);
  for (std::size_t i = 0; i < keys.size(); ++i) {
    estimates[i] = current_keep_ratio(keys[i], filters.generation());
  }
  return {applicable, applicable && current != state::ACTIVE};
}

void dynamic_filter_gate::record_keep_ratio(std::size_t rows_before,
                                            std::size_t rows_after,
                                            std::size_t observed_filter_count) noexcept
{
  if (rows_before == 0 || rows_after > rows_before) { return; }
  try {
    std::scoped_lock decision_lock(_decision_mu);
    auto const current = _state.load(std::memory_order_relaxed);
    if (current == state::ACTIVE) { return; }
    if (current == state::DISABLED &&
        observed_filter_count <= _decided_filter_count.load(std::memory_order_relaxed)) {
      return;  // same filters the disabling batch saw -- no new information
    }
    auto const kept = static_cast<double>(rows_after) / static_cast<double>(rows_before);
    _decided_filter_count.store(observed_filter_count, std::memory_order_relaxed);
    _state.store(kept > _keep_threshold ? state::DISABLED : state::ACTIVE,
                 std::memory_order_relaxed);
  } catch (...) {
    // Failure to acquire optional decision state leaves its previous verdict intact.
  }
}

bool detail::gate_test_access::applicable(dynamic_filter_gate const& gate,
                                          sirius::op::dynamic_filter_snapshot const& filters)
{
  return gate.capture(filters, {}, {}).applicable;
}

bool detail::gate_test_access::needs_combined_sample(
  dynamic_filter_gate const& gate, sirius::op::dynamic_filter_snapshot const& filters)
{
  return gate.capture(filters, {}, {}).needs_combined_sample;
}

std::optional<double> detail::gate_test_access::filter_keep_ratio(dynamic_filter_gate const& gate,
                                                                  bound_filter_key filter,
                                                                  std::size_t observed_filter_count)
{
  std::scoped_lock lock(gate._filter_ratios_mu);
  return gate.current_keep_ratio(filter, observed_filter_count);
}

}  // namespace sirius::op::scan
