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

#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <log/logging.hpp>
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <telemetry/nvtx.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/**
 * @brief The strategy selection logic.
 *
 * If not every keep ratio is known, and
 *  - the row width is >= k_deferred_minimum_row_width, use deferred_keys strategy;
 *  - otherwise, use cascade strategy.
 * If the row width is < k_deferred_minimum_row_width, and
 *  - every keep ratio >= k_gather_once_minimum_narrow_keep_ratio,
 *    use gather_once strategy;
 *  - otherwise, use cascade strategy.
 * If the row width is >= k_deferred_minimum_row_width, and
 *  - any keep ratio <= k_deferred_selective_keep_ratio,
 *    use deferred_keys strategy;
 *  - otherwise, use gather_once strategy.
 *
 * See evaluate_compaction_policy() for the implementation.
 * @note These are empirically chosen parameters from experiments on a GB300 machine. They may not
 *       extrapolate perfectly to all architectures.
 */
constexpr std::size_t k_deferred_minimum_row_width       = 64;
constexpr double k_deferred_selective_keep_ratio         = 0.35;
constexpr double k_gather_once_minimum_narrow_keep_ratio = 0.40;

struct membership_step {
  std::size_t column_index;                               ///< The key column index of the batch.
  sirius::op::sirius_mask_applicable const* mask_source;  ///< The compute mask capability.
  sirius::op::sirius_dynamic_filter const* identity;      ///< The filter object pointer.
  std::optional<double> expected_keep;  ///< The last observed keep ratio (empty if unknown).
};

struct filter_application_result {
  std::unique_ptr<cudf::table> table;
  std::size_t masks_applied = 0;
};

class exceptional_stream_retirement {
 public:
  explicit exceptional_stream_retirement(rmm::cuda_stream_view stream)
    : _stream(stream), _uncaught_on_entry(std::uncaught_exceptions())
  {
  }

  ~exceptional_stream_retirement() noexcept
  {
    if (_submitted && std::uncaught_exceptions() > _uncaught_on_entry) {
      _stream.synchronize_no_throw();
    }
  }

  exceptional_stream_retirement(exceptional_stream_retirement const&)            = delete;
  exceptional_stream_retirement& operator=(exceptional_stream_retirement const&) = delete;

  void mark_submitted() noexcept { _submitted = true; }

 private:
  rmm::cuda_stream_view _stream;
  int _uncaught_on_entry;
  bool _submitted = false;
};

/// @brief State for deferred_keys compaction strategy.
struct deferred_selection_state {
  std::unique_ptr<cudf::column> row_ids;  ///< The survivor row IDs.
  std::unique_ptr<cudf::column>
    aligned_key;                       ///< The last compacted key column (same length as row_ids).
  std::unique_ptr<cudf::column> mask;  ///< The mask just computed for the next compaction.
  std::optional<std::size_t> aligned_key_index;  ///< The index of the last compacted key column.
  std::unique_ptr<cudf::table> transition;       ///< The transition table (key + row IDs).
  std::unique_ptr<cudf::table> payload;  ///< The payload table (all columns except the key).
  std::vector<std::unique_ptr<cudf::column>> output_columns;  ///< The output columns.
};

struct compaction_policy_decision {
  detail::compaction_strategy strategy;
  char const* reason;
};

[[nodiscard]] bool is_known_keep_ratio(std::optional<double> estimate) noexcept
{
  return estimate && std::isfinite(*estimate) && *estimate >= 0.0 && *estimate <= 1.0;
}

/// @brief Choose the optimal compaction strategy.
[[nodiscard]] compaction_policy_decision evaluate_compaction_policy(
  detail::compaction_policy_input const& input) noexcept
{
  using detail::compaction_strategy;
  if (input.candidate_step_count < 2) {
    return {compaction_strategy::cascade, "fewer_than_two_candidates"};
  }
  if (input.rows == 0 || !input.input_bytes || *input.input_bytes == 0 ||
      *input.input_bytes == std::numeric_limits<std::size_t>::max() ||
      *input.input_bytes < input.rows) {
    return {compaction_strategy::cascade, "invalid_width"};
  }

  auto const row_width = *input.input_bytes / input.rows;
  auto const all_known = std::ranges::all_of(
    input.membership, [](auto const& keep) { return is_known_keep_ratio(keep); });
  if (!all_known) {
    return row_width >= k_deferred_minimum_row_width
             ? compaction_policy_decision{compaction_strategy::deferred_keys,
                                          "unknown_membership_wide"}
             : compaction_policy_decision{compaction_strategy::cascade,
                                          "unknown_membership_narrow"};
  }

  if (row_width >= k_deferred_minimum_row_width) {
    auto const has_selective = std::ranges::any_of(
      input.membership, [](auto const& keep) { return *keep <= k_deferred_selective_keep_ratio; });
    return has_selective ? compaction_policy_decision{compaction_strategy::deferred_keys,
                                                      "selective_membership_wide"}
                         : compaction_policy_decision{compaction_strategy::gather_once,
                                                      "weak_memberships_wide"};
  }

  auto const all_weak = std::ranges::all_of(input.membership, [](auto const& keep) {
    return *keep >= k_gather_once_minimum_narrow_keep_ratio;
  });
  return all_weak
           ? compaction_policy_decision{compaction_strategy::gather_once, "weak_memberships_narrow"}
           : compaction_policy_decision{compaction_strategy::cascade,
                                        "selective_membership_narrow"};
}

void validate_selection_state(deferred_selection_state const& state,
                              cudf::size_type original_rows,
                              rmm::cuda_stream_view stream,
                              bool enabled)
{
  if (!enabled) { return; }
  if (!state.row_ids || state.row_ids->type().id() != cudf::type_id::INT32 ||
      (state.aligned_key && state.aligned_key->size() != state.row_ids->size())) {
    throw std::logic_error("deferred dynamic-filter selection state is misaligned");
  }

  std::vector<cudf::size_type> host_ids(static_cast<std::size_t>(state.row_ids->size()));
  if (!host_ids.empty()) {
    auto const status = cudaMemcpyAsync(host_ids.data(),
                                        state.row_ids->view().data<cudf::size_type>(),
                                        host_ids.size() * sizeof(cudf::size_type),
                                        cudaMemcpyDeviceToHost,
                                        stream.value());
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
    stream.synchronize();
  }
  for (std::size_t index = 0; index < host_ids.size(); ++index) {
    if (host_ids[index] < 0 || host_ids[index] >= original_rows ||
        (index != 0 && host_ids[index - 1] >= host_ids[index])) {
      throw std::logic_error("deferred dynamic-filter row IDs violate original-row ordering");
    }
  }
}

void record_marginal_keep(dynamic_filter_gate* gate,
                          membership_step const& step,
                          cudf::size_type rows_before,
                          cudf::size_type rows_after,
                          std::size_t observed_generation)
{
  if (!gate || step.expected_keep || rows_before == 0) { return; }
  auto const kept = static_cast<double>(rows_after) / static_cast<double>(rows_before);
  gate->record_filter_keep_ratio(step.identity, kept, observed_generation);
}

//===----------cascade compaction strategy----------===//
/// @brief Apply the cascade compaction strategy.
filter_application_result apply_cascade(cudf::table_view const& input,
                                        std::unique_ptr<cudf::column> ast_mask,
                                        std::span<membership_step const> steps,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr,
                                        dynamic_filter_gate* gate,
                                        std::size_t observed_generation,
                                        int device_id)
{
  filter_application_result result;
  std::unique_ptr<cudf::table> table;
  std::unique_ptr<cudf::column> mask;
  exceptional_stream_retirement retirement{stream};
  cudf::table_view current = input;

  auto compact = [&] {
    if (!mask) { return false; }
    retirement.mark_submitted();
    table   = sirius::ApplyRetentionMask(current, mask->view(), stream, mr);
    current = table->view();
    ++result.masks_applied;
    return true;
  };

  mask = std::move(ast_mask);
  (void)compact();
  for (auto const& step : steps) {
    if (current.num_rows() == 0) { break; }
    auto const rows_before = current.num_rows();
    retirement.mark_submitted();
    mask = step.mask_source->compute_mask(current.column(step.column_index), device_id, stream, mr);
    if (!compact()) { continue; }
    record_marginal_keep(gate, step, rows_before, current.num_rows(), observed_generation);
  }
  result.table = std::move(table);
  return result;
}

//===----------deferred_keys compaction strategy----------===//
std::unique_ptr<cudf::column> make_identity_row_ids(cudf::size_type rows,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  cudf::numeric_scalar<cudf::size_type> zero{0, true, stream, mr};
  return cudf::sequence(rows, zero, stream, mr);
}

std::unique_ptr<cudf::column> gather_one(cudf::column_view const& source,
                                         cudf::column_view const& row_ids,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto gathered = cudf::gather(
    cudf::table_view({source}), row_ids, cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  auto columns = gathered->release();
  assert(columns.size() == 1);
  return std::move(columns.front());
}

std::unique_ptr<cudf::table> materialize_deferred_result(cudf::table_view const& input,
                                                         deferred_selection_state& state,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
  if (!state.aligned_key) {
    return cudf::gather(
      input, state.row_ids->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  }

  assert(state.aligned_key_index);
  assert(*state.aligned_key_index < static_cast<std::size_t>(input.num_columns()));
  std::vector<cudf::size_type> payload_indices;
  payload_indices.reserve(static_cast<std::size_t>(input.num_columns()) - 1);
  for (cudf::size_type index = 0; index < input.num_columns(); ++index) {
    if (static_cast<std::size_t>(index) != *state.aligned_key_index) {
      payload_indices.push_back(index);
    }
  }

  state.output_columns.resize(static_cast<std::size_t>(input.num_columns()));
  if (!payload_indices.empty()) {
    state.payload = cudf::gather(input.select(payload_indices),
                                 state.row_ids->view(),
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 stream,
                                 mr);
    auto columns  = state.payload->release();
    assert(columns.size() == payload_indices.size());
    for (std::size_t index = 0; index < columns.size(); ++index) {
      state.output_columns[static_cast<std::size_t>(payload_indices[index])] =
        std::move(columns[index]);
    }
  }
  state.output_columns[*state.aligned_key_index] = std::move(state.aligned_key);
  return std::make_unique<cudf::table>(std::move(state.output_columns));
}

/// @brief Apply the deferred_keys compaction strategy.
filter_application_result apply_deferred_keys(cudf::table_view const& input,
                                              std::unique_ptr<cudf::column> ast_mask,
                                              std::span<membership_step const> steps,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr,
                                              dynamic_filter_gate* gate,
                                              std::size_t observed_generation,
                                              int device_id,
                                              bool validate_indices)
{
  filter_application_result result;
  deferred_selection_state state;
  exceptional_stream_retirement retirement{stream};
  retirement.mark_submitted();
  state.row_ids = make_identity_row_ids(input.num_rows(), stream, mr);
  validate_selection_state(state, input.num_rows(), stream, validate_indices);

  if (ast_mask) {
    state.mask = std::move(ast_mask);
    retirement.mark_submitted();
    state.transition = sirius::ApplyRetentionMask(
      cudf::table_view({state.row_ids->view()}), state.mask->view(), stream, mr);
    auto columns = state.transition->release();
    assert(columns.size() == 1);
    state.row_ids = std::move(columns.front());
    ++result.masks_applied;
    validate_selection_state(state, input.num_rows(), stream, validate_indices);
  }

  for (auto const& step : steps) {
    if (state.row_ids->size() == 0) { break; }
    auto const key            = input.column(static_cast<cudf::size_type>(step.column_index));
    auto const identity_space = !state.aligned_key && state.row_ids->size() == input.num_rows();
    if (state.aligned_key_index != step.column_index) {
      if (!identity_space) {
        retirement.mark_submitted();
        state.aligned_key       = gather_one(key, state.row_ids->view(), stream, mr);
        state.aligned_key_index = step.column_index;
        validate_selection_state(state, input.num_rows(), stream, validate_indices);
      } else {
        state.aligned_key.reset();
        state.aligned_key_index.reset();
      }
    }

    auto const rows_before = state.row_ids->size();
    retirement.mark_submitted();
    auto const probe = state.aligned_key ? state.aligned_key->view() : key;
    state.mask       = step.mask_source->compute_mask(probe, device_id, stream, mr);
    if (!state.mask) { continue; }

    retirement.mark_submitted();
    auto const compact_view =
      state.aligned_key ? cudf::table_view({state.aligned_key->view(), state.row_ids->view()})
                        : cudf::table_view({key, state.row_ids->view()});
    state.transition = sirius::ApplyRetentionMask(compact_view, state.mask->view(), stream, mr);
    auto columns     = state.transition->release();
    assert(columns.size() == 2);
    state.aligned_key       = std::move(columns[0]);
    state.row_ids           = std::move(columns[1]);
    state.aligned_key_index = step.column_index;
    ++result.masks_applied;
    validate_selection_state(state, input.num_rows(), stream, validate_indices);
    record_marginal_keep(gate, step, rows_before, state.row_ids->size(), observed_generation);
  }

  if (result.masks_applied == 0) { return result; }
  retirement.mark_submitted();
  result.table = materialize_deferred_result(input, state, stream, mr);
  return result;
}

//===----------gather_once compaction strategy----------===//
/// @brief Apply the gather_once compaction strategy.
filter_application_result apply_gather_once(cudf::table_view const& input,
                                            std::unique_ptr<cudf::column> ast_mask,
                                            std::span<membership_step const> steps,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr,
                                            int device_id)
{
  filter_application_result result;
  std::unique_ptr<cudf::column> accumulated_mask;
  std::unique_ptr<cudf::column> next_mask;
  exceptional_stream_retirement retirement{stream};

  auto fold_mask = [&] {
    if (!next_mask) { return; }
    ++result.masks_applied;
    if (!accumulated_mask) {
      accumulated_mask = std::move(next_mask);
      return;
    }
    retirement.mark_submitted();
    accumulated_mask = cudf::binary_operation(accumulated_mask->view(),
                                              next_mask->view(),
                                              cudf::binary_operator::LOGICAL_AND,
                                              cudf::data_type{cudf::type_id::BOOL8},
                                              stream,
                                              mr);
    next_mask.reset();
  };

  next_mask = std::move(ast_mask);
  fold_mask();
  for (auto const& step : steps) {
    retirement.mark_submitted();
    next_mask = step.mask_source->compute_mask(
      input.column(static_cast<cudf::size_type>(step.column_index)), device_id, stream, mr);
    fold_mask();
  }

  if (!accumulated_mask) { return result; }
  retirement.mark_submitted();
  result.table = sirius::ApplyRetentionMask(input, accumulated_mask->view(), stream, mr);
  return result;
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
    auto const& col_ref  = tree.emplace<cudf::ast::column_name_reference>(parquet_col_name);
    auto const& fragment = lowerable->to_ast(tree, col_ref, device_id);
    root                 = root ? &tree.emplace<cudf::ast::operation>(
                    cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
                                : &fragment;
  }
  return root;
}

namespace {

std::unique_ptr<cudf::table> apply_dynamic_filters_to_view_impl(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  dynamic_filter_gate* gate,
  int device_id,
  std::optional<std::size_t> input_bytes,
  std::optional<detail::compaction_strategy> strategy_override,
  bool validate_indices)
{
  nvtx_scoped_range nvtx_range{"dynfilter::apply_output"};
  if (input.num_rows() == 0 || input.num_columns() == 0) { return nullptr; }

  device_id           = sirius::op::detail::resolve_dynamic_filter_device_id(device_id);
  auto const num_cols = static_cast<std::size_t>(input.num_columns());
  auto const mr       = cudf::get_current_device_resource_ref();

  auto const include_ast_masks = mode == dynamic_filter_apply_mode::include_ast_row_masks;

  std::unique_ptr<cudf::column> ast_mask;
  exceptional_stream_retirement retirement{stream};
  if (include_ast_masks) {
    cudf::ast::tree tree;
    cudf::ast::expression const* root = nullptr;
    for (auto const& [col_idx, filter] : filters.entries()) {
      if (col_idx >= num_cols) { continue; }
      if (!filter->is_available_on_device(device_id)) { continue; }
      auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(filter.get());
      if (!lowerable) { continue; }
      auto const& col_ref =
        tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
      auto const& fragment = lowerable->to_ast(tree, col_ref, device_id);
      root                 = root ? &tree.emplace<cudf::ast::operation>(
                      cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
                                  : &fragment;
    }
    if (root) {
      // Cross-column AST masks update only the scan-level gate.
      retirement.mark_submitted();
      ast_mask = cudf::compute_column(input, *root, stream, mr);
    }
  }

  auto const observed_filter_count = filters.generation();
  std::vector<membership_step> entries;
  for (auto const& [col_idx, filter] : filters.entries()) {
    if (col_idx >= num_cols) { continue; }
    if (!filter->is_available_on_device(device_id)) { continue; }
    auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(filter.get());
    if (!applicable) { continue; }
    auto recorded =
      gate ? gate->filter_keep_ratio(filter.get(), observed_filter_count) : std::nullopt;
    if (recorded && dynamic_filter_gate::filter_skippable(*recorded)) { continue; }
    entries.push_back({col_idx, applicable, filter.get(), recorded});
  }
  std::stable_sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
    return a.expected_keep.value_or(1.0) < b.expected_keep.value_or(1.0);
  });

  std::vector<std::optional<double>> estimates;
  estimates.reserve(entries.size());
  for (auto const& entry : entries) {
    estimates.push_back(entry.expected_keep);
  }

  detail::compaction_policy_input const policy_input{
    .rows                 = static_cast<std::size_t>(input.num_rows()),
    .input_bytes          = input_bytes,
    .candidate_step_count = entries.size() + static_cast<std::size_t>(ast_mask != nullptr),
    .membership           = std::span<std::optional<double> const>{estimates}};
  auto const decision = evaluate_compaction_policy(policy_input);
  auto const strategy = strategy_override.value_or(decision.strategy);
  filter_application_result result;
  switch (strategy) {
    case detail::compaction_strategy::cascade:
      result = apply_cascade(
        input, std::move(ast_mask), entries, stream, mr, gate, observed_filter_count, device_id);
      break;
    case detail::compaction_strategy::deferred_keys:
      result = apply_deferred_keys(input,
                                   std::move(ast_mask),
                                   entries,
                                   stream,
                                   mr,
                                   gate,
                                   observed_filter_count,
                                   device_id,
                                   validate_indices);
      break;
    case detail::compaction_strategy::gather_once:
      result = apply_gather_once(input, std::move(ast_mask), entries, stream, mr, device_id);
      break;
  }

  if (!result.table) { return nullptr; }
  auto const* strategy_name = strategy == detail::compaction_strategy::cascade ? "cascade"
                              : strategy == detail::compaction_strategy::deferred_keys
                                ? "deferred_keys"
                                : "gather_once";
  auto const* mode_name     = mode == dynamic_filter_apply_mode::include_ast_row_masks
                                ? "include_ast_row_masks"
                                : "membership_masks_only";
  auto const row_width      = policy_input.rows != 0 && policy_input.input_bytes
                                ? *policy_input.input_bytes / policy_input.rows
                                : 0;
  std::optional<double> strongest_keep;
  for (auto const& estimate : estimates) {
    if (is_known_keep_ratio(estimate)) {
      strongest_keep = strongest_keep ? std::min(*strongest_keep, *estimate) : estimate;
    }
  }
  SIRIUS_LOG_DEBUG(
    "[apply_dynamic_filters] device={} mode={} strategy={} reason={} bytes={} row_width={} "
    "candidates={} filters={} strongest_keep={} apply: {} -> {} rows.",
    device_id,
    mode_name,
    strategy_name,
    strategy_override ? "forced" : decision.reason,
    input_bytes.value_or(0),
    row_width,
    policy_input.candidate_step_count,
    entries.size(),
    strongest_keep.value_or(-1.0),
    input.num_rows(),
    result.table->num_rows());
  return std::move(result.table);
}

}  // namespace

std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  dynamic_filter_gate* gate,
  int device_id,
  std::optional<std::size_t> input_bytes)
{
  return apply_dynamic_filters_to_view_impl(
    input, filters, stream, mode, gate, device_id, input_bytes, std::nullopt, false);
}

std::unique_ptr<cudf::table> detail::apply_dynamic_filters_to_view_for_testing(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  rmm::cuda_stream_view stream,
  compaction_strategy strategy,
  dynamic_filter_apply_mode mode,
  dynamic_filter_gate* gate,
  int device_id)
{
  return apply_dynamic_filters_to_view_impl(
    input, filters, stream, mode, gate, device_id, std::nullopt, strategy, true);
}

std::optional<double> dynamic_filter_gate::filter_keep_ratio(
  sirius::op::sirius_dynamic_filter const* filter, std::size_t observed_filter_count) const
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto it = _filter_keep_ratios.find(filter);
  if (it == _filter_keep_ratios.end()) { return std::nullopt; }
  // Skipping an optional filter cannot affect correctness, so its verdict is permanent.
  if (filter_skippable(it->second.kept)) { return it->second.kept; }
  // New filters can change this filter's marginal selectivity.
  if (it->second.observed_filter_count < observed_filter_count) { return std::nullopt; }
  return it->second.kept;
}

void dynamic_filter_gate::record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter,
                                                   double kept,
                                                   std::size_t observed_filter_count)
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto const it = _filter_keep_ratios.find(filter);
  if (it != _filter_keep_ratios.end() &&
      it->second.observed_filter_count >= observed_filter_count) {
    return;
  }
  _filter_keep_ratios.insert_or_assign(
    filter, filter_measurement{.kept = kept, .observed_filter_count = observed_filter_count});
  if (filter_skippable(kept)) {
    SIRIUS_LOG_DEBUG(
      "[apply_dynamic_filters] per-filter gate: marginal kept {:.3f} against {} filters -> SKIP "
      "filter permanently.",
      kept,
      observed_filter_count);
  }
}

bool dynamic_filter_gate::applicable(sirius::op::dynamic_filter_snapshot const& filters) const
{
  if (filters.empty()) { return false; }
  if (_state.load(std::memory_order_relaxed) != state::disabled) { return true; }
  // Renewed filters may change the gate's verdict, so we check if the snapshot is newer than the
  // last one that disabled the gate.
  return filters.generation() > _decided_filter_count.load(std::memory_order_relaxed);
}

void dynamic_filter_gate::record_keep_ratio(std::size_t rows_before,
                                            std::size_t rows_after,
                                            std::size_t observed_filter_count)
{
  if (rows_before == 0) { return; }

  std::scoped_lock decision_lock(_decision_mu);
  auto const current = _state.load(std::memory_order_relaxed);
  if (current == state::active) { return; }
  if (current == state::disabled &&
      observed_filter_count <= _decided_filter_count.load(std::memory_order_relaxed)) {
    return;  // same filters the disabling batch saw — no new information
  }
  auto const kept = static_cast<double>(rows_after) / static_cast<double>(rows_before);
  _decided_filter_count.store(observed_filter_count, std::memory_order_relaxed);
  _state.store(kept > _keep_threshold ? state::disabled : state::active, std::memory_order_relaxed);
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] selectivity gate: kept {:.3f} ({} filters) -> {}.",
                   kept,
                   observed_filter_count,
                   kept > _keep_threshold ? "DISABLED" : "ACTIVE");
}

std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& snapshot,
  dynamic_filter_gate& gate,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  int device_id,
  std::optional<std::size_t> input_bytes)
{
  if (!gate.applicable(snapshot)) { return nullptr; }
  auto const observed_filters = snapshot.generation();
  auto const rows_before      = input.num_rows();
  auto filtered =
    apply_dynamic_filters_to_view(input, snapshot, stream, mode, &gate, device_id, input_bytes);
  if (!filtered) { return nullptr; }
  gate.record_keep_ratio(
    rows_before, static_cast<std::size_t>(filtered->num_rows()), observed_filters);
  return filtered;
}

}  // namespace sirius::op::scan
