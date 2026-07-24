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

#include "scan_manager/delta_promotion.hpp"

#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "op/scan/gpu_ingestible_types.hpp"

#include <cudf/column/column.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <exception>
#include <utility>

namespace sirius::scan_manager {

bool promotion_sink::try_begin_capture(std::string const& entry_name, duckdb::idx_t first_row_group)
{
  std::lock_guard<std::mutex> guard(mutex_);
  return seen_.emplace(entry_name, first_row_group).second;
}

void promotion_sink::add(std::string const& entry_name, promotion_captured_slice slice)
{
  std::lock_guard<std::mutex> guard(mutex_);
  captures_[entry_name].slices.push_back(std::move(slice));
}

void promotion_sink::record_skip(std::string const& entry_name, std::string reason)
{
  std::lock_guard<std::mutex> guard(mutex_);
  captures_[entry_name].last_skip_reason = std::move(reason);
}

bool promotion_sink::empty() const
{
  std::lock_guard<std::mutex> guard(mutex_);
  return captures_.empty();
}

std::unordered_map<std::string, promotion_sink::entry_capture> promotion_sink::take_all()
{
  std::lock_guard<std::mutex> guard(mutex_);
  auto out = std::move(captures_);
  captures_.clear();
  seen_.clear();
  return out;
}

namespace {

constexpr char const* kTag = "[delta_promotion]";

/// GPU tier: photocopy the kept decoded columns (D2D) into entry order.
/// Reservation-first on the split's own space; the copies allocate from it.
bool capture_gpu_slice(promotion_capture const& ticket,
                       cudf::table_view const& view,
                       cucascade::memory::memory_space& space,
                       rmm::cuda_stream_view stream,
                       promotion_captured_slice& slice,
                       std::string& skip_reason)
{
  auto const& plan = *ticket.plan;
  auto reservation = space.make_reservation_or_null(ticket.reserve_bytes);
  if (!reservation) {
    skip_reason = "reservation-failed";
    return false;
  }
  auto mr = space.get_default_allocator();
  std::vector<std::shared_ptr<cudf::column>> columns(plan.column_names.size());
  for (std::size_t ci = 0; ci < plan.entry_pos_by_decoded_pos.size(); ++ci) {
    auto const pos = plan.entry_pos_by_decoded_pos[ci];
    if (pos == kDropDecodedColumn) { continue; }
    columns[pos] =
      std::make_shared<cudf::column>(view.column(static_cast<cudf::size_type>(ci)), stream, mr);
  }
  for (auto const& column : columns) {
    if (!column) {
      skip_reason = "decoded-columns-incomplete";
      return false;
    }
  }
  stream.synchronize();
  slice.column_names = plan.column_names;
  slice.columns      = std::move(columns);
  slice.space        = &space;
  slice.reservation  = std::shared_ptr<void>(std::move(reservation));
  return true;
}

/// HOST tier: photocopy via one transient D2D of the kept columns, then the
/// registry's D2H conversion into NUMA-local pinned host memory (the
/// materialize_pin_to_host pattern, but skip-on-null reservation).
bool capture_host_slice(promotion_capture const& ticket,
                        cudf::table_view const& view,
                        cucascade::memory::memory_space& space,
                        rmm::cuda_stream_view stream,
                        promotion_captured_slice& slice,
                        std::string& skip_reason)
{
  auto const& plan     = *ticket.plan;
  auto const host_slot = plan.host_space_by_gpu.find(space.get_device_id());
  if (host_slot == plan.host_space_by_gpu.end() || host_slot->second == nullptr) {
    skip_reason = "no-host-space-for-gpu";
    return false;
  }
  auto& host_space = *host_slot->second;

  std::vector<cudf::column_view> ordered(plan.column_names.size());
  std::vector<bool> filled(plan.column_names.size(), false);
  for (std::size_t ci = 0; ci < plan.entry_pos_by_decoded_pos.size(); ++ci) {
    auto const pos = plan.entry_pos_by_decoded_pos[ci];
    if (pos == kDropDecodedColumn) { continue; }
    ordered[pos] = view.column(static_cast<cudf::size_type>(ci));
    filled[pos]  = true;
  }
  if (std::find(filled.begin(), filled.end(), false) != filled.end()) {
    skip_reason = "decoded-columns-incomplete";
    return false;
  }

  auto gpu_copy =
    std::make_unique<cudf::table>(cudf::table_view{ordered}, stream, space.get_default_allocator());
  cucascade::gpu_table_representation gpu_repr(std::move(gpu_copy), space, stream);
  auto host_reservation = host_space.make_reservation_or_null(gpu_repr.get_size_in_bytes());
  if (!host_reservation) {
    skip_reason = "reservation-failed";
    return false;
  }
  auto host_repr = converter_registry::get().convert<cucascade::host_data_representation>(
    gpu_repr, *host_reservation, stream);
  stream.synchronize();
  slice.host_chunk  = std::move(host_repr);
  slice.reservation = std::shared_ptr<void>(std::move(host_reservation));
  return true;
}

}  // namespace

void capture_promoted_slice(promotion_capture const& ticket,
                            op::scan::filtered_table const& materialized,
                            cucascade::memory::memory_space& space,
                            rmm::cuda_stream_view stream) noexcept
{
  if (!ticket.plan || !ticket.plan->sink || ticket.row_group_indices.empty()) { return; }
  auto const& plan = *ticket.plan;
  auto& sink       = *plan.sink;
  try {
    auto const view = materialized.table.view();
    if (view.num_rows() != static_cast<cudf::size_type>(ticket.row_count) ||
        static_cast<std::size_t>(view.num_columns()) != plan.entry_pos_by_decoded_pos.size()) {
      sink.record_skip(plan.entry_name, "decoded-shape-mismatch");
      SIRIUS_LOG_WARN(
        "{} pinned entry '{}': promotable split decoded {} rows x {} cols, ticket expected {} "
        "rows x {} mapped cols; skipping promotion",
        kTag,
        plan.entry_name,
        view.num_rows(),
        view.num_columns(),
        ticket.row_count,
        plan.entry_pos_by_decoded_pos.size());
      return;
    }
    // First-op-wins: a self-join decodes the same bundle once per operator.
    if (!sink.try_begin_capture(plan.entry_name, ticket.row_group_indices.front())) { return; }

    promotion_captured_slice slice;
    slice.first_rowid       = ticket.first_rowid;
    slice.row_count         = ticket.row_count;
    slice.row_group_indices = ticket.row_group_indices;

    std::string skip_reason;
    bool const captured = plan.tier == cucascade::memory::Tier::GPU
                            ? capture_gpu_slice(ticket, view, space, stream, slice, skip_reason)
                            : capture_host_slice(ticket, view, space, stream, slice, skip_reason);
    if (!captured) {
      sink.record_skip(plan.entry_name, skip_reason);
      SIRIUS_LOG_INFO(
        "{} pinned entry '{}': promotion capture skipped ({}); the delta stays on "
        "the per-query path",
        kTag,
        plan.entry_name,
        skip_reason);
      return;
    }
    sink.add(plan.entry_name, std::move(slice));
  } catch (std::exception const& e) {
    try {
      sink.record_skip(plan.entry_name, std::string("capture-failed: ") + e.what());
    } catch (...) {
    }  // NOLINT(bugprone-empty-catch)
    SIRIUS_LOG_WARN("{} pinned entry '{}': promotion capture failed ({}); the query is unaffected",
                    kTag,
                    plan.entry_name,
                    e.what());
  } catch (...) {
    try {
      sink.record_skip(plan.entry_name, "capture-failed: unknown");
    } catch (...) {
    }  // NOLINT(bugprone-empty-catch)
    SIRIUS_LOG_WARN(
      "{} pinned entry '{}': promotion capture failed (unknown); the query is "
      "unaffected",
      kTag,
      plan.entry_name);
  }
}

std::vector<promotion_captured_slice> select_promotion_prefix(
  std::vector<promotion_captured_slice> slices,
  std::size_t n_cache,
  std::vector<promotion_captured_slice>& dropped)
{
  std::sort(slices.begin(), slices.end(), [](auto const& a, auto const& b) {
    return a.first_rowid < b.first_rowid;
  });

  std::vector<promotion_captured_slice> selected;
  std::size_t expected = n_cache;
  for (auto& slice : slices) {
    // Advance the base prefix only by a slice that begins exactly where it
    // currently ends. Once a gap appears every later slice starts even higher
    // (sorted), so nothing else can match — the base stays one unbroken run.
    if (slice.first_rowid == expected && slice.row_count != 0) {
      expected += slice.row_count;
      selected.push_back(std::move(slice));
    } else {
      dropped.push_back(std::move(slice));
    }
  }
  return selected;
}

}  // namespace sirius::scan_manager
