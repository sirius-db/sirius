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

#include "late_mat/pin_uniqueness.hpp"

#include "log/logging.hpp"
#include "sirius_config.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/unique_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>

namespace sirius::late_mat {

namespace {

/// Host value of an integer scalar, widened into the probe's common domain.
/// Returns nullopt for anything the probe must not interpret — an invalid
/// scalar (an all-null chunk reduces to one) or a non-integer type.
std::optional<__int128_t> integer_scalar_value(cudf::scalar const& s, ::cuda::stream_ref stream)
{
  if (!s.is_valid(stream)) { return std::nullopt; }
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<int8_t> const&>(s).value(stream));
    case cudf::type_id::INT16:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<int16_t> const&>(s).value(stream));
    case cudf::type_id::INT32:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<int32_t> const&>(s).value(stream));
    case cudf::type_id::INT64:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<int64_t> const&>(s).value(stream));
    case cudf::type_id::UINT8:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<uint8_t> const&>(s).value(stream));
    case cudf::type_id::UINT16:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<uint16_t> const&>(s).value(stream));
    case cudf::type_id::UINT32:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<uint32_t> const&>(s).value(stream));
    case cudf::type_id::UINT64:
      return static_cast<__int128_t>(
        static_cast<cudf::numeric_scalar<uint64_t> const&>(s).value(stream));
    default: return std::nullopt;
  }
}

}  // namespace

std::vector<std::size_t> exact_retention_priority(std::span<cudf::data_type const> types)
{
  std::vector<std::size_t> priority(types.size());
  std::iota(priority.begin(), priority.end(), std::size_t{0});
  std::stable_partition(priority.begin(), priority.end(), [&](std::size_t c) {
    return cudf::is_integral_not_bool(types[c]);
  });
  return priority;
}

std::size_t exact_uniqueness_byte_cap(std::uint64_t configured_bytes,
                                      std::uint64_t gpu_capacity_bytes)
{
  if (configured_bytes > 0) { return static_cast<std::size_t>(configured_bytes); }
  return static_cast<std::size_t>(static_cast<double>(gpu_capacity_bytes) *
                                  config::PIN_TABLE_UNIQUE_EXACT_MEMORY_FRACTION);
}

std::optional<bool> exact_distinct_over_chunks(std::span<cudf::column_view const> chunks,
                                               ::cuda::stream_ref stream)
{
  if (chunks.empty()) { return std::nullopt; }

  std::int64_t total_rows = 0;
  for (auto const& chunk : chunks) {
    if (chunk.null_count() > 0) { return std::nullopt; }
    if (cudf::is_nested(chunk.type()) || !cudf::is_relationally_comparable(chunk.type())) {
      return std::nullopt;
    }
    if (chunk.type() != chunks.front().type()) { return std::nullopt; }
    total_rows += chunk.size();
  }
  if (total_rows == 0) { return std::nullopt; }
  if (total_rows > static_cast<std::int64_t>(std::numeric_limits<cudf::size_type>::max())) {
    SIRIUS_LOG_DEBUG("[late-mat] exact uniqueness check skipped: {} rows exceed a cudf column",
                     total_rows);
    return std::nullopt;
  }

  std::vector<cudf::column_view> views{chunks.begin(), chunks.end()};
  auto assembled = cudf::concatenate(views, stream);
  // Sort first, then count consecutive runs: over sorted values that count IS
  // the exact distinct count, and it needs no hash table (which at these row
  // counts is the difference between a check that fits and one that does not).
  auto sorted =
    cudf::sort(cudf::table_view{{assembled->view()}}, {cudf::order::ASCENDING}, {}, stream);
  assembled.reset();
  auto const distinct = cudf::unique_count(
    sorted->view().column(0), cudf::null_policy::EXCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
  return static_cast<std::int64_t>(distinct) == total_rows;
}

unique_probe::unique_probe(std::vector<bool> selected)
{
  _columns.resize(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    _columns[i].observed  = selected[i];
    _columns[i].candidate = selected[i];
    if (selected[i]) { ++_live_candidates; }
  }
}

void unique_probe::drop(std::size_t column_pos, std::string_view why)
{
  if (!_columns[column_pos].candidate) { return; }
  // Every refusal is reported: a proof that silently did not happen looks
  // exactly like one that was never asked for, and the difference is the whole
  // question when the ride it would have unlocked does not appear.
  SIRIUS_LOG_DEBUG("[late-mat] uniqueness probe: column {} not provable ({})", column_pos, why);
  _columns[column_pos].candidate = false;
  _columns[column_pos].ranges.clear();
  --_live_candidates;
}

void unique_probe::exact_unavailable(std::size_t column_pos)
{
  if (column_pos < _columns.size()) { _columns[column_pos].exact_available = false; }
}

void unique_probe::observe(cudf::table_view const& chunk,
                           ::cuda::stream_ref stream,
                           std::function<bool(std::size_t bytes)> const& try_reserve_sort_memory)
{
  if (_live_candidates == 0) { return; }

  if (static_cast<std::size_t>(chunk.num_columns()) != _columns.size()) {
    // The selection is positional with the pinned columns; a chunk of a
    // different width means we cannot say WHICH column we would be observing.
    // Abandon the whole proof rather than record a fact against the wrong one.
    SIRIUS_LOG_WARN(
      "[late-mat] uniqueness probe abandoned: chunk has {} columns, selection covers {}",
      chunk.num_columns(),
      _columns.size());
    for (std::size_t i = 0; i < _columns.size(); ++i) {
      drop(i, "chunk width disagrees with the selection");
    }
    return;
  }

  // The chunk's value range, or nothing when it cannot be read (including for want of memory).
  auto const chunk_range = [&](cudf::column_view const& values) -> std::optional<range> {
    try {
      auto const [min_s, max_s] = cudf::minmax(values, stream);
      auto const lo             = integer_scalar_value(*min_s, stream);
      auto const hi             = integer_scalar_value(*max_s, stream);
      if (!lo || !hi) { return std::nullopt; }
      return range{*lo, *hi};
    } catch (std::exception const&) {
      return std::nullopt;
    }
  };

  for (std::size_t i = 0; i < _columns.size() && _live_candidates > 0; ++i) {
    if (!_columns[i].candidate) { continue; }
    auto const col = chunk.column(static_cast<cudf::size_type>(i));

    // An empty chunk carries no evidence either way: no values to duplicate and
    // no range to overlap. Skip it without touching the column's standing.
    if (col.size() == 0) { continue; }

    if (col.has_nulls()) {
      drop(i, "nullable");
      continue;
    }
    if (!cudf::is_integral_not_bool(col.type())) {
      // Out of the cheap stage's reach. Every column of a pin is observed, and the exact stage's
      // sort is not worth paying on a big table's non-integer columns, so such a column stays in
      // the running only while its table is small: a dimension's name column, say, which can prove
      // a group-by-rowid ride (q5, q9 and q10 group by nation.n_name). The cap is on rows seen so
      // far, so a big table's first chunk ends the column's chances with no sort and no retained
      // values. A nested column is never worth it: the exact stage cannot judge one.
      _columns[i].observed_rows += static_cast<std::size_t>(col.size());
      if (cudf::is_nested(col.type()) || _columns[i].observed_rows > small_table_max_rows) {
        drop(i, "not an integer, and nested or past the small-table row cap");
        continue;
      }
      _columns[i].cheap_undecidable = true;
      continue;
    }

    // Counted off SORTEDNESS, never off a hash set. cuco's open-addressing
    // extent is bounded, and a pinned chunk is sized in GIGABYTES: at SF1000 an
    // `orders` chunk overruns "requested extent divided by load factor exceeds
    // maximum representable value" and takes the whole pin down with it. Over
    // sorted values, counting consecutive runs IS the exact distinct count and
    // costs one pass.
    std::unique_ptr<cudf::table> sorted_copy;
    cudf::column_view sorted_col = col;
    cudf::size_type distinct     = 0;
    std::optional<range> early_range;
    try {
      // With the exact stage out of reach, a column can only be proven by chunks whose ranges
      // never overlap, so a chunk that overlaps an earlier one ends its chances and is not worth
      // sorting.
      if (!_columns[i].exact_available) {
        early_range = chunk_range(col);
        if (!early_range) {
          drop(i, "its range cannot be read and the exact stage cannot settle it");
          continue;
        }
        auto const overlaps = std::any_of(
          _columns[i].ranges.begin(), _columns[i].ranges.end(), [&](range const& earlier) {
            return earlier.min <= early_range->max && early_range->min <= earlier.max;
          });
        if (overlaps) {
          drop(i, "its chunks overlap in range and the exact stage cannot settle it");
          continue;
        }
      }
      if (!cudf::is_sorted(cudf::table_view{{col}}, {cudf::order::ASCENDING}, {}, stream)) {
        // Most columns of a fact table are unsorted and repeat values, so sort this one chunk:
        // a repeat refuses the column here, before the exact stage retains any of its values.
        // The sort is optional. A pin that cannot spare the memory for it loses this proof,
        // never the pin itself.
        if (try_reserve_sort_memory &&
            !try_reserve_sort_memory(static_cast<std::size_t>(col.size()) *
                                     cudf::size_of(col.type()) * sort_scratch_factor)) {
          drop(i, "no GPU memory reserved for sorting a chunk");
          continue;
        }
        sorted_copy = cudf::sort(cudf::table_view{{col}}, {cudf::order::ASCENDING}, {}, stream);
        sorted_col  = sorted_copy->view().column(0);
      }
      distinct = cudf::unique_count(
        sorted_col, cudf::null_policy::INCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
    } catch (std::exception const& e) {
      drop(i, std::string{"the check failed: "} + e.what());
      continue;
    }
    sorted_copy.reset();
    if (distinct != col.size()) {
      drop(i, "a chunk repeats a value");
      continue;
    }

    auto const observed_range = early_range ? early_range : chunk_range(col);
    if (!observed_range) {
      _columns[i].cheap_undecidable = true;
      continue;
    }
    _columns[i].ranges.push_back(*observed_range);
  }
}

std::vector<unique_verdict> unique_probe::verdicts() const
{
  std::vector<unique_verdict> out(_columns.size(), unique_verdict::not_observed);
  for (std::size_t i = 0; i < _columns.size(); ++i) {
    auto const& state = _columns[i];
    if (!state.observed) { continue; }
    if (!state.candidate) {
      out[i] = unique_verdict::refused;
      continue;
    }
    if (state.cheap_undecidable) {
      out[i] = unique_verdict::undecided;
      continue;
    }

    // Every chunk was internally distinct; the table is distinct iff no two
    // chunks can share a value, i.e. their ranges do not overlap. Sorting by
    // `min` makes that a single adjacent-pair scan.
    auto ranges = state.ranges;
    std::sort(
      ranges.begin(), ranges.end(), [](range const& a, range const& b) { return a.min < b.min; });
    bool disjoint = true;
    for (std::size_t r = 1; r < ranges.size(); ++r) {
      if (ranges[r].min <= ranges[r - 1].max) {
        disjoint = false;
        break;
      }
    }
    out[i] = disjoint ? unique_verdict::proven : unique_verdict::undecided;
  }
  return out;
}

std::vector<bool> unique_probe::proven() const
{
  auto const verdict = verdicts();
  std::vector<bool> out(verdict.size(), false);
  for (std::size_t i = 0; i < verdict.size(); ++i) {
    out[i] = verdict[i] == unique_verdict::proven;
  }
  return out;
}

std::vector<std::string> unique_probe::proven_names(std::span<std::string const> column_names) const
{
  std::vector<std::string> out;
  if (column_names.size() != _columns.size()) {
    // Same positional argument as in observe(): a width disagreement means we
    // cannot name what we proved, so we claim nothing.
    SIRIUS_LOG_WARN("[late-mat] uniqueness probe: {} names for {} observed columns; reporting none",
                    column_names.size(),
                    _columns.size());
    return out;
  }
  auto const flags = proven();
  for (std::size_t i = 0; i < flags.size(); ++i) {
    if (flags[i]) { out.push_back(column_names[i]); }
  }
  return out;
}

}  // namespace sirius::late_mat
