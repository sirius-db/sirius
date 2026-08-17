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

#include <cudf/concatenate.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/unique_count.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string_view>

namespace sirius::late_mat {

namespace {

std::string lowered(std::string_view s)
{
  std::string out{s};
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return out;
}

/// Host value of an integer scalar, widened into the probe's common domain.
/// Returns nullopt for anything the probe must not interpret — an invalid
/// scalar (an all-null chunk reduces to one) or a non-integer type.
std::optional<__int128> integer_scalar_value(cudf::scalar const& s, rmm::cuda_stream_view stream)
{
  if (!s.is_valid(stream)) { return std::nullopt; }
  switch (s.type().id()) {
    case cudf::type_id::INT8:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<int8_t> const&>(s).value(stream));
    case cudf::type_id::INT16:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<int16_t> const&>(s).value(stream));
    case cudf::type_id::INT32:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<int32_t> const&>(s).value(stream));
    case cudf::type_id::INT64:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<int64_t> const&>(s).value(stream));
    case cudf::type_id::UINT8:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<uint8_t> const&>(s).value(stream));
    case cudf::type_id::UINT16:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<uint16_t> const&>(s).value(stream));
    case cudf::type_id::UINT32:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<uint32_t> const&>(s).value(stream));
    case cudf::type_id::UINT64:
      return static_cast<__int128>(
        static_cast<cudf::numeric_scalar<uint64_t> const&>(s).value(stream));
    default: return std::nullopt;
  }
}

}  // namespace

std::vector<bool> pin_unique_probe_selection(std::span<std::string const> column_names)
{
  std::vector<bool> selected(column_names.size(), false);

  char const* raw = std::getenv("SIRIUS_LATE_MAT_PIN_UNIQUE_COLS");
  if (raw == nullptr || raw[0] == '\0') { return selected; }
  std::string const spec = lowered(raw);
  if (spec == "0" || spec == "none" || spec == "off") { return selected; }

  if (spec == "all") {
    selected.assign(column_names.size(), true);
    return selected;
  }

  // Comma-separated names. An entry that matches nothing is not an error — a
  // single setting is meant to cover a suite of pins over different tables.
  std::size_t pos = 0;
  while (pos <= spec.size()) {
    auto const comma = spec.find(',', pos);
    auto const end   = comma == std::string::npos ? spec.size() : comma;
    std::string_view wanted{spec.data() + pos, end - pos};
    while (!wanted.empty() && std::isspace(static_cast<unsigned char>(wanted.front()))) {
      wanted.remove_prefix(1);
    }
    while (!wanted.empty() && std::isspace(static_cast<unsigned char>(wanted.back()))) {
      wanted.remove_suffix(1);
    }
    if (!wanted.empty()) {
      for (std::size_t i = 0; i < column_names.size(); ++i) {
        if (lowered(column_names[i]) == wanted) { selected[i] = true; }
      }
    }
    if (comma == std::string::npos) { break; }
    pos = comma + 1;
  }
  return selected;
}

std::size_t exact_uniqueness_row_cap()
{
  static std::size_t const value = []() -> std::size_t {
    char const* v = std::getenv("SIRIUS_LATE_MAT_EXACT_MAX_ROWS");
    if (v == nullptr || v[0] == '\0') { return 300'000'000; }
    char* end         = nullptr;
    auto const parsed = std::strtoull(v, &end, 10);
    return (end == v || parsed == 0) ? 300'000'000 : static_cast<std::size_t>(parsed);
  }();
  return value;
}

std::optional<bool> exact_distinct_over_chunks(std::span<cudf::column_view const> chunks,
                                               rmm::cuda_stream_view stream)
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

void unique_probe::observe(cudf::table_view const& chunk, rmm::cuda_stream_view stream)
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
      // Out of the cheap stage's reach, not out of the running: the exact check
      // sorts whatever cuDF can sort.
      _columns[i].cheap_undecidable = true;
      continue;
    }

    // Counted off SORTEDNESS, never off a hash set. cuco's open-addressing
    // extent is bounded, and a pinned chunk is sized in GIGABYTES: at SF1000 an
    // `orders` chunk overruns "requested extent divided by load factor exceeds
    // maximum representable value" and takes the whole pin down with it. Over
    // sorted values, counting consecutive runs IS the exact distinct count and
    // costs one pass.
    if (!cudf::is_sorted(cudf::table_view{{col}}, {cudf::order::ASCENDING}, {}, stream)) {
      // Unsorted says nothing either way — leave it to the exact stage.
      _columns[i].cheap_undecidable = true;
      continue;
    }
    auto const distinct =
      cudf::unique_count(col, cudf::null_policy::INCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
    if (distinct != col.size()) {
      drop(i, "a chunk repeats a value");
      continue;
    }

    auto const [min_s, max_s] = cudf::minmax(col, stream);
    auto const lo             = integer_scalar_value(*min_s, stream);
    auto const hi             = integer_scalar_value(*max_s, stream);
    if (!lo || !hi) {
      _columns[i].cheap_undecidable = true;
      continue;
    }
    _columns[i].ranges.push_back(range{*lo, *hi});
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
