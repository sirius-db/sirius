/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/aggregate/table_aggregate.hpp"

#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/input_free_operator_data.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/error.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <limits>
#include <shared_mutex>
#include <stdexcept>
#include <utility>

namespace sirius {
namespace op {

namespace {

/// A table smaller than this cannot pay for itself; the GPU then keeps the cuDF path.
constexpr std::size_t min_table_budget_bytes = std::size_t{64} << 20;

/// A table that the memory space made flush this many times is not rebuilt on that GPU.
constexpr int max_pressure_flushes = 2;

/// Folds over which a table's absorbed share is measured before the operator may give it up.
constexpr std::uint64_t give_up_window_folds = 8;

/// Absorbed share of the rows offered to a table below which the table is given up: it then
/// costs a probe per row and returns almost every row to the cuDF path anyway.
constexpr double give_up_absorbed_share = 0.10;

std::optional<agg_table::agg_kind> to_table_kind(cudf::aggregation::Kind kind)
{
  switch (kind) {
    case cudf::aggregation::Kind::SUM: return agg_table::agg_kind::sum;
    case cudf::aggregation::Kind::MIN: return agg_table::agg_kind::min;
    case cudf::aggregation::Kind::MAX: return agg_table::agg_kind::max;
    case cudf::aggregation::Kind::COUNT_VALID: return agg_table::agg_kind::count;
    case cudf::aggregation::Kind::COUNT_ALL: return agg_table::agg_kind::count_star;
    default: return std::nullopt;
  }
}

bool is_signed_integer(cudf::data_type type)
{
  switch (type.id()) {
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64: return true;
    default: return false;
  }
}

bool is_narrow_decimal(cudf::data_type type)
{
  return type.id() == cudf::type_id::DECIMAL32 || type.id() == cudf::type_id::DECIMAL64;
}

/// Key types the engine offers to the table: fixed-width, at most 64 bits, with bitwise equality
/// equal to SQL equality. Floating-point keys need -0.0 and NaN normalised first and are left out.
bool key_type_supported(cudf::data_type type)
{
  if (!cudf::is_fixed_width(type) || cudf::size_of(type) > 8) { return false; }
  if (cudf::is_floating_point(type)) { return false; }
  return cudf::is_integral(type) || cudf::is_timestamp(type) || cudf::is_duration(type) ||
         is_narrow_decimal(type);
}

/// Input types the engine offers to the table for an aggregate of kind @p kind. The table's own
/// `eligible` narrows this further.
bool value_type_supported(cudf::aggregation::Kind kind, cudf::data_type type)
{
  switch (kind) {
    case cudf::aggregation::Kind::COUNT_ALL:
    case cudf::aggregation::Kind::COUNT_VALID: return true;
    case cudf::aggregation::Kind::SUM:
      // cuDF sums FLOAT32 in FLOAT32 and unsigned integers in UINT64; neither is offered.
      return is_signed_integer(type) || is_narrow_decimal(type) ||
             type.id() == cudf::type_id::FLOAT64;
    case cudf::aggregation::Kind::MIN:
    case cudf::aggregation::Kind::MAX:
      if (!cudf::is_fixed_width(type) || cudf::size_of(type) > 8) { return false; }
      // Both float widths: the table orders NaN above every number, as SQL does.
      return (cudf::is_integral(type) && type.id() != cudf::type_id::BOOL8) ||
             cudf::is_timestamp(type) || cudf::is_duration(type) || is_narrow_decimal(type) ||
             cudf::is_floating_point(type);
    default: return false;
  }
}

/// Whether a column of the table's partial type @p from may be cast to the local aggregate's
/// partial type @p to without changing a value: equal types, a wider integer, or a wider decimal
/// of the same scale.
bool is_value_preserving_widening(cudf::data_type from, cudf::data_type to)
{
  if (from == to) { return true; }
  if (cudf::is_fixed_point(from) && cudf::is_fixed_point(to)) {
    return from.scale() == to.scale() && cudf::size_of(from) <= cudf::size_of(to);
  }
  if (is_signed_integer(from) && is_signed_integer(to)) {
    return cudf::size_of(from) <= cudf::size_of(to);
  }
  // An integer SUM that the local aggregate carries in 128 bits (wide_integer_sum_type()).
  if (is_signed_integer(from) && to.id() == cudf::type_id::DECIMAL128) { return to.scale() == 0; }
  return false;
}

std::size_t fixed_width_bytes(cudf::data_type type)
{
  return cudf::is_fixed_width(type) ? cudf::size_of(type) : 16;
}

/// Process-wide counters behind `table_aggregate_process_stats`.
struct process_counters {
  std::atomic<std::uint64_t> folded_batches{0};
  std::atomic<std::uint64_t> folded_rows{0};
  std::atomic<std::uint64_t> returned_rows{0};
  std::atomic<std::uint64_t> returned_partial_rows{0};
  std::atomic<std::uint64_t> legacy_batches{0};
  std::atomic<std::uint64_t> replayed_folds{0};
  std::atomic<std::uint64_t> boundary_flushes{0};
  std::atomic<std::uint64_t> end_of_input_flushes{0};
  std::atomic<std::uint64_t> flushed_rows{0};
  std::array<std::atomic<std::uint64_t>, table_aggregate_stats::max_gpus> folded_batches_by_gpu{};
};

process_counters& counters() noexcept
{
  static process_counters instance;
  return instance;
}

void count(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) noexcept
{
  counter.fetch_add(amount, std::memory_order_relaxed);
}

}  // namespace

std::size_t table_aggregate_stats::gpus_folded() const noexcept
{
  return static_cast<std::size_t>(std::count_if(folded_batches_by_gpu.begin(),
                                                folded_batches_by_gpu.end(),
                                                [](auto batches) { return batches > 0; }));
}

table_aggregate_stats table_aggregate_stats::since(
  table_aggregate_stats const& before) const noexcept
{
  auto delta = *this;
  delta.folded_batches -= before.folded_batches;
  delta.folded_rows -= before.folded_rows;
  delta.returned_rows -= before.returned_rows;
  delta.returned_partial_rows -= before.returned_partial_rows;
  delta.legacy_batches -= before.legacy_batches;
  delta.replayed_folds -= before.replayed_folds;
  delta.boundary_flushes -= before.boundary_flushes;
  delta.end_of_input_flushes -= before.end_of_input_flushes;
  delta.flushed_rows -= before.flushed_rows;
  for (std::size_t gpu = 0; gpu < max_gpus; ++gpu) {
    delta.folded_batches_by_gpu[gpu] -= before.folded_batches_by_gpu[gpu];
  }
  return delta;
}

table_aggregate_stats table_aggregate_process_stats() noexcept
{
  auto const& from = counters();
  table_aggregate_stats stats;
  stats.folded_batches        = from.folded_batches.load(std::memory_order_relaxed);
  stats.folded_rows           = from.folded_rows.load(std::memory_order_relaxed);
  stats.returned_rows         = from.returned_rows.load(std::memory_order_relaxed);
  stats.returned_partial_rows = from.returned_partial_rows.load(std::memory_order_relaxed);
  stats.legacy_batches        = from.legacy_batches.load(std::memory_order_relaxed);
  stats.replayed_folds        = from.replayed_folds.load(std::memory_order_relaxed);
  stats.boundary_flushes      = from.boundary_flushes.load(std::memory_order_relaxed);
  stats.end_of_input_flushes  = from.end_of_input_flushes.load(std::memory_order_relaxed);
  stats.flushed_rows          = from.flushed_rows.load(std::memory_order_relaxed);
  for (std::size_t gpu = 0; gpu < table_aggregate_stats::max_gpus; ++gpu) {
    stats.folded_batches_by_gpu[gpu] =
      from.folded_batches_by_gpu[gpu].load(std::memory_order_relaxed);
  }
  return stats;
}

/// One GPU's table and what belongs to it.
struct table_aggregate_state::device_state {
  device_state(int device_id_, cucascade::memory::memory_space& space_)
    : device_id(device_id_), space(&space_)
  {
  }

  int const device_id;
  cucascade::memory::memory_space* const space;

  /// Shared by folds; exclusive to create, flush and destroy the table.
  std::shared_mutex table_mutex;
  std::unique_ptr<agg_table::table> table;
  /// Partial rows that left the table and are not batches yet. A flush that runs out of memory
  /// while converting them leaves the rest here for the next flush of this GPU.
  std::vector<std::unique_ptr<cudf::table>> flushed_tables;
  /// Batches of the end-of-input flush, kept across a replay of its task.
  std::vector<std::shared_ptr<cucascade::data_batch>> flushed_batches;

  /// Guards the budget fields below. Taken after `table_mutex`, never before it.
  std::mutex ledger_mutex;
  /// Bytes the table may hold. Nothing of it is held as a reservation between folds: a
  /// reservation that outlives its task keeps the memory space's waiters parked (a task waiting
  /// for memory is only handed a partial grant once no reservation is alive), and the table's
  /// would live until the end of the input.
  std::size_t budget = 0;

  std::atomic<bool> has_rows{false};  ///< The table or `flushed_tables` holds rows
  std::atomic<bool> claimed{false};   ///< The end-of-input flush task was handed out
  std::atomic<bool> disabled{false};  ///< No table is built on this GPU any more
  std::atomic<bool> give_up{false};   ///< Flush and disable at the next batch boundary
  /// Batches whose fold changed the table and then failed; their replay must find the table.
  std::atomic<int> incomplete_folds{0};
  std::atomic<int> pressure_flushes{0};
  std::atomic<std::uint64_t> window_folds{0};
  std::atomic<std::uint64_t> window_rows{0};
  std::atomic<std::uint64_t> window_applied{0};
};

/// What a task did with one batch, kept until its `execute` returns.
struct table_aggregate_state::fold_entry {
  /// GPU of the table the batch was folded into (and of `unapplied`); -1 before the first fold.
  int device_id = -1;
  agg_table::fold_ticket ticket;
  bool folded     = false;  ///< `fold` returned; `unapplied` is final
  bool incomplete = false;  ///< Counted in the device's `incomplete_folds`
  /// Rows the table did not take, as a gather map into the batch; null when it took them all.
  std::unique_ptr<cudf::column> unapplied;
  /// Partial rows this task flushed out of a table. They are part of its output on every attempt.
  std::vector<std::shared_ptr<cucascade::data_batch>> flushed;
};

std::shared_ptr<table_aggregate_state> table_aggregate_state::try_create(
  table_aggregate_plan plan, table_aggregate_options options, std::string* why_not)
{
  auto const refuse = [&](std::string reason) -> std::shared_ptr<table_aggregate_state> {
    if (why_not != nullptr) { *why_not = std::move(reason); }
    return nullptr;
  };
  auto const count = plan.aggregates.size();
  if (plan.key_column < 0) { return refuse("no single key column"); }
  if (plan.aggregate_columns.size() != count || plan.aggregate_input_types.size() != count) {
    return refuse("aggregate description is incomplete");
  }
  if (!(options.memory_fraction > 0.0) || options.memory_fraction > 1.0) {
    return refuse("memory fraction is outside (0, 1]");
  }
  if (!key_type_supported(plan.key_type)) { return refuse("key type is not supported"); }

  agg_table::config config;
  config.key_type  = plan.key_type;
  config.rows_hint = plan.rows_hint;
  // The planner's group-count estimate is not trusted (integration plan, section 5): the table
  // grows from what it measures.
  config.groups_hint = std::nullopt;

  // The fold's `values` view holds each distinct input column once.
  std::vector<int> value_columns;
  std::vector<agg_table::column_bound> bounds;
  for (std::size_t i = 0; i < count; ++i) {
    auto const kind = to_table_kind(plan.aggregates[i]);
    if (!kind) { return refuse("aggregate is not COUNT, SUM, MIN or MAX"); }
    auto const input_type = plan.aggregate_input_types[i];
    if (!value_type_supported(plan.aggregates[i], input_type)) {
      return refuse("aggregate input type is not supported");
    }
    if (*kind == agg_table::agg_kind::count_star) {
      config.aggs.push_back({*kind, -1, input_type});
      continue;
    }
    auto const column = plan.aggregate_columns[i];
    auto position     = std::find(value_columns.begin(), value_columns.end(), column);
    if (position == value_columns.end()) {
      value_columns.push_back(column);
      bounds.emplace_back();
      position = value_columns.end() - 1;
    }
    auto const value_idx = static_cast<std::size_t>(position - value_columns.begin());
    config.aggs.push_back({*kind, static_cast<int>(value_idx), input_type});
    // A plan-time bound on a SUM input holds for every batch, so it is every batch's bound.
    auto const max_abs = i < plan.input_max_abs.size() ? plan.input_max_abs[i] : std::nullopt;
    if (*kind == agg_table::agg_kind::sum && max_abs &&
        *max_abs <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
      auto const bound  = static_cast<std::int64_t>(*max_abs);
      bounds[value_idx] = agg_table::column_bound{true, -bound, bound};
    }
  }

  std::string table_reason;
  if (!agg_table::eligible(config, &table_reason)) {
    return refuse("table: " + (table_reason.empty() ? std::string{"not eligible"} : table_reason));
  }
  auto schema = agg_table::partial_schema(config);
  if (schema.size() != 1 + count) {
    return refuse("table partial rows are not one state column per aggregate");
  }

  auto state                   = std::shared_ptr<table_aggregate_state>(new table_aggregate_state(
    std::move(plan), options, std::move(config), std::move(value_columns), std::move(bounds)));
  state->_table_partial_schema = std::move(schema);
  for (auto const& type : state->_table_partial_schema) {
    // A widened decimal SUM is the widest column the local schema can hold in place of this one.
    state->_partial_row_bytes += std::max<std::size_t>(fixed_width_bytes(type), 8);
  }
  auto const rows = std::max<std::size_t>(
    options.flush_chunk_bytes / std::max<std::size_t>(state->_partial_row_bytes, 1), 1);
  state->_flush_chunk_rows = std::min<std::size_t>(
    rows, static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()));
  return state;
}

table_aggregate_state::table_aggregate_state(table_aggregate_plan plan,
                                             table_aggregate_options options,
                                             agg_table::config config,
                                             std::vector<int> value_columns,
                                             std::vector<agg_table::column_bound> bounds)
  : _plan(std::move(plan)),
    _options(options),
    _config(std::move(config)),
    _value_columns(std::move(value_columns)),
    _bounds(std::move(bounds)),
    _inject_fold_oom(options.inject_fold_oom)
{
}

table_aggregate_state::~table_aggregate_state()
{
  // Runs on query teardown as well as after a normal finish, on a thread with any device current
  // (the finalize hook does not run when a query fails). Device memory is freed with its own
  // device current, the tables before the budget that covers them.
  for (auto& [batch_id, entry] : _entries) {
    if (entry->device_id >= 0) {
      // The gather map and the ticket's record of the batch are device memory.
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{entry->device_id}};
      entry->unapplied.reset();
      entry->ticket.progress.reset();
    }
    entry->flushed.clear();
  }
  for (auto& [device_id, dev] : _devices) {
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{device_id}};
    dev->flushed_batches.clear();
    dev->flushed_tables.clear();
    dev->table.reset();
  }
}

std::shared_ptr<table_aggregate_state::fold_entry> table_aggregate_state::find_or_create_entry(
  std::uint64_t batch_id)
{
  std::lock_guard guard(_mutex);
  auto& entry = _entries[batch_id];
  if (!entry) {
    entry                  = std::make_shared<fold_entry>();
    entry->ticket.batch_id = batch_id;
  }
  return entry;
}

table_aggregate_state::device_state& table_aggregate_state::device(
  cucascade::memory::memory_space& space)
{
  std::lock_guard guard(_mutex);
  auto& dev = _devices[space.get_device_id()];
  if (!dev) { dev = std::make_unique<device_state>(space.get_device_id(), space); }
  return *dev;
}

table_aggregate_state::device_state* table_aggregate_state::find_device(int device_id) const
{
  std::lock_guard guard(_mutex);
  auto const it = _devices.find(device_id);
  return it == _devices.end() ? nullptr : it->second.get();
}

void table_aggregate_state::disable(std::string_view reason) noexcept
{
  if (!_disabled.exchange(true, std::memory_order_acq_rel)) {
    log::log_noexcept([&] {
      SIRIUS_LOG_INFO("table aggregate: back on the cuDF path for new batches: {}", reason);
    });
  }
}

void table_aggregate_state::complete_batches(std::span<const std::uint64_t> batch_ids) noexcept
{
  std::lock_guard guard(_mutex);
  for (auto const batch_id : batch_ids) {
    // Entries of a normally returned execute hold no device memory worth a device switch: the
    // task's own device is current and is the one `unapplied` lives on.
    _entries.erase(batch_id);
  }
}

void table_aggregate_state::check_partial_schema(cudf::table_view const& input,
                                                 legacy_aggregate const& legacy,
                                                 ::cuda::stream_ref stream)
{
  std::lock_guard guard(_mutex);
  if (_schema_checked) { return; }
  // The local aggregate's partial schema is whatever the cuDF path produces for these input
  // types; ask it, on no rows, instead of restating its rules here.
  std::vector<cudf::size_type> const no_rows{0, 0};
  std::shared_ptr<cucascade::data_batch> probe_batch;
  try {
    probe_batch = legacy(cudf::slice(input, no_rows, stream).front());
  } catch (std::logic_error const& error) {
    // The cuDF path refused an empty input. Nothing is known about its schema then.
    _schema_checked = true;
    disable(std::string{"could not probe the local aggregate's schema: "} + error.what());
    return;
  }
  auto const probe   = get_cudf_table_view(*probe_batch);
  auto const columns = _table_partial_schema.size();
  if (static_cast<std::size_t>(probe.num_columns()) != columns) {
    _schema_checked = true;
    disable("the local aggregate's output is not the key and one column per aggregate");
    return;
  }
  std::vector<cudf::data_type> engine_schema;
  engine_schema.reserve(columns);
  for (std::size_t c = 0; c < columns; ++c) {
    auto const from = _table_partial_schema[c];
    auto const to   = probe.column(static_cast<cudf::size_type>(c)).type();
    engine_schema.push_back(to);
    // Equal, or a cast that cannot change a value. (A wide integer SUM that the table flushes
    // as DECIMAL128 with scale 0 is settled per flush, in conform_partial.)
    if (!is_value_preserving_widening(from, to)) {
      _schema_checked = true;
      disable("table partial column " + std::to_string(c) +
              " cannot be merged with the local aggregate's");
      return;
    }
  }
  _engine_partial_schema = std::move(engine_schema);
  _schema_checked        = true;
}

bool table_aggregate_state::batch_matches_plan(cudf::table_view const& input) const
{
  auto const columns = input.num_columns();
  if (_plan.key_column >= columns) { return false; }
  auto const key = input.column(_plan.key_column);
  // NULL keys are fine: the table keeps them as a group of their own, as the cuDF path does
  // (null_policy::INCLUDE), and the merge combines the two.
  if (key.type() != _plan.key_type) { return false; }
  for (std::size_t i = 0; i < _config.aggs.size(); ++i) {
    auto const& agg = _config.aggs[i];
    if (agg.value_column < 0) { continue; }
    auto const column = _value_columns[static_cast<std::size_t>(agg.value_column)];
    // A narrow carrier (compressed materialization) arrives in another type than planned.
    if (column >= columns || input.column(column).type() != agg.input) { return false; }
  }
  return true;
}

bool table_aggregate_state::ensure_table(device_state& dev)
{
  {
    std::shared_lock shared(dev.table_mutex);
    if (dev.table) { return true; }
  }
  std::unique_lock exclusive(dev.table_mutex);
  if (dev.table) { return true; }
  if (dev.disabled.load(std::memory_order_acquire)) { return false; }
  auto const capacity = dev.space->get_max_memory();
  auto budget = static_cast<std::size_t>(static_cast<double>(capacity) * _options.memory_fraction);
  if (budget < min_table_budget_bytes) {
    dev.disabled.store(true, std::memory_order_release);
    return false;
  }
  // An explicit cap is taken as given, however small: rows that do not fit keep the cuDF path.
  if (_options.max_budget_bytes > 0) { budget = std::min(budget, _options.max_budget_bytes); }
  auto config         = _config;
  config.budget_bytes = budget;
  // A budget below the smallest table the library builds for these aggregates (a small explicit
  // cap) holds no group at all: this GPU keeps the cuDF path. `create` throws for such a budget.
  if (std::string why_not; !agg_table::eligible(config, &why_not)) {
    dev.disabled.store(true, std::memory_order_release);
    SIRIUS_LOG_DEBUG("table aggregate: no table on GPU {} with a budget of {} bytes: {}",
                     dev.device_id,
                     budget,
                     why_not);
    return false;
  }
  // A space that cannot grant the budget now keeps this batch on the cuDF path and is asked again
  // for the next one. The grant is given back at once (see `budget`).
  if (!dev.space->make_reservation_or_null(budget)) { return false; }
  auto table = agg_table::table::create(config, dev.space->get_default_allocator());
  {
    std::lock_guard ledger(dev.ledger_mutex);
    dev.budget = budget;
  }
  dev.table = std::move(table);
  dev.window_folds.store(0, std::memory_order_relaxed);
  dev.window_rows.store(0, std::memory_order_relaxed);
  dev.window_applied.store(0, std::memory_order_relaxed);
  SIRIUS_LOG_DEBUG(
    "table aggregate: table on GPU {} with a budget of {} bytes", dev.device_id, budget);
  return true;
}

void table_aggregate_state::after_fold_failure(device_state& dev,
                                               fold_entry& entry,
                                               bool out_of_memory) noexcept
{
  // The caller holds `table_mutex` (shared).
  if (entry.ticket.st != agg_table::fold_ticket::state::not_started) {
    dev.has_rows.store(true, std::memory_order_release);
    if (!entry.incomplete) {
      entry.incomplete = true;
      dev.incomplete_folds.fetch_add(1, std::memory_order_acq_rel);
    }
    count(counters().replayed_folds);
  } else {
    // Nothing of the batch is in the table, and the replay starts over: give the batch's scratch
    // back now, on this GPU, instead of holding it while the task waits for memory.
    entry.ticket.progress.reset();
  }
  if (!out_of_memory) { return; }
  try {
    std::lock_guard ledger(dev.ledger_mutex);
    if (dev.table) {
      // The memory is not there. Stop growing, so that the replay returns the rows it cannot
      // place instead of failing again.
      dev.budget = dev.table->device_bytes();
      dev.table->shrink_budget(dev.budget);
    }
  } catch (...) {
    // Accounting only; the task is replayed either way.
  }
}

std::uint64_t table_aggregate_state::take_injected_fold_oom() noexcept
{
  auto left = _inject_fold_oom.load(std::memory_order_relaxed);
  while (left > 0) {
    if (_inject_fold_oom.compare_exchange_weak(left, left - 1, std::memory_order_relaxed)) {
      return left;
    }
  }
  return 0;
}

bool table_aggregate_state::fold(cudf::table_view const& input,
                                 device_state& dev,
                                 fold_entry& entry,
                                 ::cuda::stream_ref stream,
                                 const telemetry::batch_telemetry_info& telemetry_info)
{
  std::shared_lock shared(dev.table_mutex);
  if (!dev.table) {
    if (entry.ticket.st != agg_table::fold_ticket::state::not_started) {
      // Flushes wait for `incomplete_folds` to drain, so this is a broken invariant, not a race.
      throw std::runtime_error("table aggregate: the table of GPU " +
                               std::to_string(dev.device_id) +
                               " was flushed while a batch was partly folded into it");
    }
    return false;  // flushed between admission and here; nothing was applied
  }
  entry.device_id = dev.device_id;

  std::vector<cudf::column_view> values;
  values.reserve(_value_columns.size());
  for (auto const column : _value_columns) {
    values.push_back(input.column(column));
  }
  // Test hook. Odd injections fail inside the table, after it took the batch's rows and before it
  // built the result; even ones fail here, after the table returned. A fold that is being
  // resumed applies no rows, so it only takes the second kind.
  bool inject_after_fold = false;
  if (auto const injection = take_injected_fold_oom(); injection > 0) {
    if (injection % 2 == 1 && entry.ticket.st == agg_table::fold_ticket::state::not_started) {
      dev.table->test_fail_after_mutation(1);
    } else {
      inject_after_fold = true;
    }
  }
  agg_table::fold_result result;
  std::uint64_t returned_partial_rows = 0;
  try {
    // A replay with a ticket that is `mutated` resumes the batch; with one that is `done` it gets
    // the first call's result again and the table is not touched.
    result = dev.table->fold(
      input.column(_plan.key_column), cudf::table_view{values}, _bounds, entry.ticket, stream);
    if (result.rows_applied > 0) { dev.has_rows.store(true, std::memory_order_release); }
    if (inject_after_fold) { throw rmm::out_of_memory("injected table aggregate fold OOM"); }
    // Rows the table consumed and pre-aggregated but had no room for come back as partial rows.
    // They are this task's output on every attempt, like the rows of a flush.
    if (result.unapplied_partial && result.unapplied_partial->num_rows() > 0) {
      returned_partial_rows = static_cast<std::uint64_t>(result.unapplied_partial->num_rows());
      auto partial          = std::move(result.unapplied_partial);
      if (auto conformed = conform_partial(*partial, stream, dev.space->get_default_allocator())) {
        partial = std::move(conformed);
      }
      entry.flushed.reserve(entry.flushed.size() + 1);
      entry.flushed.push_back(
        make_data_batch(std::move(partial), *dev.space, stream, telemetry_info));
    }
  } catch (rmm::out_of_memory const&) {
    after_fold_failure(dev, entry, true);
    throw;
  } catch (...) {
    after_fold_failure(dev, entry, false);
    throw;
  }

  // Only now is the batch on record. Until here a failed attempt counts as incomplete, which
  // keeps the table from being flushed before the replay has asked it for the result again.
  entry.folded    = true;
  entry.unapplied = std::move(result.unapplied);
  // The table's record of the batch (scratch, in the table's memory resource) is no longer
  // needed: a replay of this task does not ask the table again. Dropping it here also means no
  // record outlives a flush of the table.
  entry.ticket.progress.reset();
  {
    auto const rows     = static_cast<std::uint64_t>(input.num_rows());
    auto const returned = entry.unapplied ? static_cast<std::uint64_t>(entry.unapplied->size()) : 0;
    auto& stats         = counters();
    count(stats.folded_batches);
    count(stats.folded_rows, rows - std::min(rows, returned));
    count(stats.returned_rows, returned);
    count(stats.returned_partial_rows, returned_partial_rows);
    auto const slot = std::min<std::size_t>(static_cast<std::size_t>(std::max(dev.device_id, 0)),
                                            table_aggregate_stats::max_gpus - 1);
    count(stats.folded_batches_by_gpu[slot]);
  }
  if (entry.incomplete) {
    entry.incomplete = false;
    dev.incomplete_folds.fetch_sub(1, std::memory_order_acq_rel);
  }

  // Give the table up when it has stopped absorbing rows: it then costs a probe per row and
  // returns nearly all of them.
  dev.window_rows.fetch_add(static_cast<std::uint64_t>(input.num_rows()),
                            std::memory_order_relaxed);
  // Rows that came back as partial rows were consumed but not absorbed: with distinct keys they
  // are one partial row per input row, which the merge then pays for like any unaggregated row.
  dev.window_applied.fetch_add(
    result.rows_applied - std::min(result.rows_applied, returned_partial_rows),
    std::memory_order_relaxed);
  if (dev.window_folds.fetch_add(1, std::memory_order_relaxed) + 1 >= give_up_window_folds) {
    auto const rows    = dev.window_rows.exchange(0, std::memory_order_relaxed);
    auto const applied = dev.window_applied.exchange(0, std::memory_order_relaxed);
    dev.window_folds.store(0, std::memory_order_relaxed);
    if (rows > 0 &&
        static_cast<double>(applied) < give_up_absorbed_share * static_cast<double>(rows)) {
      dev.give_up.store(true, std::memory_order_release);
    }
  }
  return true;
}

void table_aggregate_state::flush_locked(device_state& dev, ::cuda::stream_ref stream)
{
  if (!dev.table) { return; }
  if (dev.incomplete_folds.load(std::memory_order_acquire) != 0) {
    throw std::logic_error("table aggregate: flush while a batch is partly folded");
  }
  SIRIUS_LOG_DEBUG(
    "table aggregate: flushing the table of GPU {}: {}", dev.device_id, dev.table->describe());
  auto chunks = dev.table->flush(_flush_chunk_rows, stream, dev.space->get_default_allocator());
  for (auto const& chunk : chunks) {
    if (chunk) { count(counters().flushed_rows, static_cast<std::uint64_t>(chunk->num_rows())); }
  }
  // From here on the rows exist only in `chunks`: move them where a replay finds them before
  // anything else can throw.
  if (dev.flushed_tables.empty()) {
    dev.flushed_tables = std::move(chunks);
  } else {
    dev.flushed_tables.insert(dev.flushed_tables.end(),
                              std::make_move_iterator(chunks.begin()),
                              std::make_move_iterator(chunks.end()));
  }
  // The flush's kernels read the table on this stream; order them before the free.
  stream.sync();
  dev.table.reset();
  std::lock_guard ledger(dev.ledger_mutex);
  dev.budget = 0;
}

std::unique_ptr<cudf::table> table_aggregate_state::conform_partial(
  cudf::table& partial, ::cuda::stream_ref stream, rmm::device_async_resource_ref mr)
{
  auto const view = partial.view();
  // Cast first, into new columns: a cast that runs out of memory leaves `partial` whole.
  std::vector<std::unique_ptr<cudf::column>> cast(static_cast<std::size_t>(view.num_columns()));
  bool any = false;
  for (cudf::size_type c = 0; c < view.num_columns(); ++c) {
    auto const idx = static_cast<std::size_t>(c);
    if (idx >= _engine_partial_schema.size()) { break; }
    auto const actual   = view.column(c).type();
    auto const expected = _engine_partial_schema[idx];
    if (actual == expected) { continue; }
    // An integer SUM whose 64-bit state could have wrapped leaves the table as DECIMAL128 with
    // scale 0. For a wide integer SUM the merge takes either carrier, batch by batch.
    bool const wide_sum_carrier = idx >= 1 && idx - 1 < _plan.is_wide_integer_sum.size() &&
                                  _plan.is_wide_integer_sum[idx - 1] &&
                                  expected.id() == cudf::type_id::INT64 &&
                                  actual.id() == cudf::type_id::DECIMAL128 && actual.scale() == 0;
    if (wide_sum_carrier) { continue; }
    cast[idx] = cudf::cast(view.column(c), expected, stream, mr);
    any       = true;
  }
  if (!any) { return nullptr; }
  auto columns = partial.release();
  for (std::size_t idx = 0; idx < columns.size(); ++idx) {
    if (cast[idx]) { columns[idx] = std::move(cast[idx]); }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

void table_aggregate_state::drain_flushed_locked(
  device_state& dev,
  ::cuda::stream_ref stream,
  const telemetry::batch_telemetry_info& telemetry_info,
  std::vector<std::shared_ptr<cucascade::data_batch>>& out)
{
  auto const mr = dev.space->get_default_allocator();
  out.reserve(out.size() + dev.flushed_tables.size());
  for (auto& partial : dev.flushed_tables) {
    if (!partial) { continue; }
    if (partial->num_rows() == 0) {
      partial.reset();
      continue;
    }
    if (auto conformed = conform_partial(*partial, stream, mr)) { partial = std::move(conformed); }
    // `out` has room, so the batch is recorded as soon as it exists.
    out.push_back(make_data_batch(std::move(partial), *dev.space, stream, telemetry_info));
  }
  dev.flushed_tables.clear();
}

void table_aggregate_state::adopt_on_device(fold_entry& entry,
                                            int device_id,
                                            ::cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  if (!entry.folded) {
    if (entry.ticket.st == agg_table::fold_ticket::state::not_started) {
      // Nothing was applied anywhere; start over on this GPU with a new ticket.
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{entry.device_id}};
      auto const batch_id   = entry.ticket.batch_id;
      entry.ticket          = agg_table::fold_ticket{};
      entry.ticket.batch_id = batch_id;
      entry.device_id       = -1;
      return;
    }
    // The rows already applied are in another GPU's table and the rest of the fold can only run
    // there. The engine does not pin an unpinned task to the GPU of its first attempt, so this
    // can happen; failing the query is the only exact answer.
    throw std::runtime_error("table aggregate: batch " + std::to_string(entry.ticket.batch_id) +
                             " was partly folded on GPU " + std::to_string(entry.device_id) +
                             " and replayed on GPU " + std::to_string(device_id));
  }
  if (entry.unapplied) {
    // The gather map is plain fixed-width data: carry it over through host memory.
    auto const type  = entry.unapplied->type();
    auto const rows  = entry.unapplied->size();
    auto const bytes = static_cast<std::size_t>(rows) * cudf::size_of(type);
    std::vector<std::uint8_t> host(bytes);
    {
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{entry.device_id}};
      // The first attempt's stream was synchronized before the task was rescheduled.
      CUDF_CUDA_TRY(cudaMemcpy(
        host.data(), entry.unapplied->view().head<std::uint8_t>(), bytes, cudaMemcpyDeviceToHost));
      entry.unapplied.reset();
    }
    auto moved =
      cudf::make_fixed_width_column(type, rows, cudf::mask_state::UNALLOCATED, stream, mr);
    CUDF_CUDA_TRY(cudaMemcpyAsync(moved->mutable_view().head<std::uint8_t>(),
                                  host.data(),
                                  bytes,
                                  cudaMemcpyHostToDevice,
                                  stream.get()));
    stream.sync();  // `host` goes out of scope
    entry.unapplied = std::move(moved);
  }
  {
    // The ticket's record of the batch belongs to the other GPU's table; the batch is done with.
    rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{entry.device_id}};
    entry.ticket.progress.reset();
  }
  entry.device_id = device_id;
}

bool table_aggregate_state::admit(cudf::table_view const& input,
                                  device_state& dev,
                                  fold_entry& entry,
                                  ::cuda::stream_ref stream,
                                  const telemetry::batch_telemetry_info& telemetry_info)
{
  // Batch boundary: give the table back when the memory space wants memory, or when it no longer
  // earns its keep. The partial rows become part of this task's output.
  bool const pressure = dev.space->should_downgrade_memory();
  bool const give_up  = dev.give_up.load(std::memory_order_acquire);
  if ((pressure || give_up) && dev.has_rows.load(std::memory_order_acquire)) {
    std::unique_lock exclusive(dev.table_mutex);
    // Another task may have flushed while this one waited; a partly folded batch keeps the table.
    if (dev.has_rows.load(std::memory_order_acquire) &&
        dev.incomplete_folds.load(std::memory_order_acquire) == 0) {
      flush_locked(dev, stream);
      drain_flushed_locked(dev, stream, telemetry_info, entry.flushed);
      dev.has_rows.store(false, std::memory_order_release);
      count(counters().boundary_flushes);
      auto const flushes = pressure ? dev.pressure_flushes.fetch_add(1) + 1 : 0;
      if (give_up || flushes >= max_pressure_flushes) {
        dev.disabled.store(true, std::memory_order_release);
      }
      SIRIUS_LOG_DEBUG(
        "table aggregate: flushed the table of GPU {} at a batch boundary ({}); disabled={}",
        dev.device_id,
        give_up ? "not absorbing" : "memory pressure",
        dev.disabled.load());
    }
  }
  if (pressure || dev.disabled.load(std::memory_order_acquire) ||
      dev.give_up.load(std::memory_order_acquire)) {
    return false;
  }
  if (!enabled() || input.num_rows() == 0 || !batch_matches_plan(input)) { return false; }

  // The sorted-keys path keeps priority (integration plan, section 5): a batch whose key arrives
  // ascending is cheaper there than in any hash table. The check costs a pass, so it stops once
  // the input has shown no ascending batch.
  auto const checks = _sorted_checks.load(std::memory_order_relaxed);
  if (checks < sorted_check_patience || _sorted_hits.load(std::memory_order_relaxed) > 0) {
    _sorted_checks.fetch_add(1, std::memory_order_relaxed);
    if (keys_ascending(input.column(_plan.key_column), stream)) {
      _sorted_hits.fetch_add(1, std::memory_order_relaxed);
      return false;
    }
  }
  return ensure_table(dev);
}

void table_aggregate_state::aggregate_batch(
  cudf::table_view const& input,
  std::uint64_t batch_id,
  ::cuda::stream_ref stream,
  cucascade::memory::memory_space& space,
  const telemetry::batch_telemetry_info& telemetry_info,
  legacy_aggregate const& legacy,
  std::vector<std::shared_ptr<cucascade::data_batch>>& results)
{
  auto const entry = find_or_create_entry(batch_id);
  auto const mr    = space.get_default_allocator();
  if (entry->device_id >= 0 && entry->device_id != space.get_device_id()) {
    adopt_on_device(*entry, space.get_device_id(), stream, mr);
  }

  bool whole_batch_on_cudf = false;
  if (!entry->folded) {
    auto& dev = device(space);
    if (entry->ticket.st == agg_table::fold_ticket::state::not_started) {
      // A ticket belongs to one table, and the table of an earlier attempt may be gone.
      entry->ticket.progress.reset();
      if (enabled()) { check_partial_schema(input, legacy, stream); }
      // `admit` runs even when the table path is disabled: it is also the batch-boundary flush.
      whole_batch_on_cudf = !admit(input, dev, *entry, stream, telemetry_info);
    }
    if (!whole_batch_on_cudf) {
      whole_batch_on_cudf = !fold(input, dev, *entry, stream, telemetry_info);
    }
  }

  // Partial rows this task flushed, in this attempt or an earlier one.
  results.insert(results.end(), entry->flushed.begin(), entry->flushed.end());
  if (whole_batch_on_cudf) {
    count(counters().legacy_batches);
    results.push_back(legacy(input));
    return;
  }
  if (!entry->unapplied || entry->unapplied->size() == 0) { return; }
  if (entry->unapplied->size() == input.num_rows()) {
    // The gather map selects distinct rows, so all of them is the batch itself.
    results.push_back(legacy(input));
    return;
  }
  auto const rows = cudf::gather(
    input, entry->unapplied->view(), cudf::out_of_bounds_policy::DONT_CHECK, stream, mr);
  results.push_back(legacy(rows->view()));
}

void table_aggregate_state::release_idle_tables() noexcept
{
  try {
    std::vector<device_state*> devices;
    {
      std::lock_guard guard(_mutex);
      for (auto& [device_id, dev] : _devices) {
        devices.push_back(dev.get());
      }
    }
    for (auto* dev : devices) {
      std::unique_lock exclusive(dev->table_mutex);
      if (dev->has_rows.load(std::memory_order_acquire)) { continue; }
      rmm::cuda_set_device_raii device_guard{rmm::cuda_device_id{dev->device_id}};
      dev->table.reset();
      std::lock_guard ledger(dev->ledger_mutex);
      dev->budget = 0;
    }
  } catch (...) {
    // Cleanup that the destructor repeats.
  }
}

bool table_aggregate_state::has_unclaimed() const noexcept
{
  try {
    std::lock_guard guard(_mutex);
    return std::any_of(_devices.begin(), _devices.end(), [](auto const& device) {
      return device.second->has_rows.load(std::memory_order_acquire) &&
             !device.second->claimed.load(std::memory_order_acquire);
    });
  } catch (...) {
    return true;  // never report state as drained when it may not be
  }
}

pending_state_source::totals table_aggregate_state::unclaimed_totals() const
{
  std::lock_guard guard(_mutex);
  totals total;
  for (auto const& [device_id, dev] : _devices) {
    if (!dev->has_rows.load(std::memory_order_acquire) ||
        dev->claimed.load(std::memory_order_acquire)) {
      continue;
    }
    // The producer's pipeline has finished, so no fold or flush runs.
    std::shared_lock shared(dev->table_mutex);
    std::uint64_t rows = dev->table ? dev->table->groups() : 0;
    for (auto const& partial : dev->flushed_tables) {
      if (partial) { rows += static_cast<std::uint64_t>(partial->num_rows()); }
    }
    total.rows += rows;
    total.bytes = memory::saturating_add(
      total.bytes, memory::saturating_mul(static_cast<std::size_t>(rows), _partial_row_bytes));
  }
  return total;
}

std::unique_ptr<operator_data> table_aggregate_state::claim()
{
  std::lock_guard guard(_mutex);
  for (auto& [device_id, dev] : _devices) {
    if (!dev->has_rows.load(std::memory_order_acquire)) { continue; }
    if (dev->claimed.exchange(true, std::memory_order_acq_rel)) { continue; }
    std::uint64_t rows = 0;
    {
      std::shared_lock shared(dev->table_mutex);
      rows = dev->table ? dev->table->groups() : 0;
      for (auto const& partial : dev->flushed_tables) {
        if (partial) { rows += static_cast<std::uint64_t>(partial->num_rows()); }
      }
    }
    auto const bytes = std::max<std::size_t>(
      memory::saturating_mul(static_cast<std::size_t>(rows), _partial_row_bytes),
      std::size_t{1} << 20);
    auto unit = std::make_unique<input_free_operator_data>(bytes);
    // The table can only be read on its own GPU. The pin survives a reschedule of the task.
    unit->set_preferred_device_id(device_id);
    return unit;
  }
  return nullptr;
}

std::vector<std::shared_ptr<cucascade::data_batch>> table_aggregate_state::emit(
  const operator_data& unit,
  ::cuda::stream_ref stream,
  const telemetry::batch_telemetry_info& telemetry_info)
{
  auto const* input_free = dynamic_cast<input_free_operator_data const*>(&unit);
  if (input_free == nullptr) {
    throw std::invalid_argument("table aggregate: end-of-input flush needs its claimed unit");
  }
  auto const device_id = input_free->get_gpu_memory_space().get_device_id();
  auto* dev            = find_device(device_id);
  if (dev == nullptr) {
    throw std::runtime_error("table aggregate: end-of-input flush ran on GPU " +
                             std::to_string(device_id) + ", which holds no table");
  }
  std::unique_lock exclusive(dev->table_mutex);
  // A replay finds the table already emptied and continues from what the first attempt left.
  flush_locked(*dev, stream);
  drain_flushed_locked(*dev, stream, telemetry_info, dev->flushed_batches);
  dev->has_rows.store(false, std::memory_order_release);
  count(counters().end_of_input_flushes);
  return std::exchange(dev->flushed_batches, {});
}

}  // namespace op
}  // namespace sirius
