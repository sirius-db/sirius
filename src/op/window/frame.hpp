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

#include <cstddef>
#include <cstdint>
#include <variant>

namespace sirius::op {

/**
 * @brief Whether a window frame counts rows or order key values.
 */
enum class frame_unit : std::uint8_t {
  /// `ROWS`: offsets count rows. Peers are not special.
  rows,
  /// `RANGE`: offsets are distances in the order key's value, and `CURRENT ROW` includes the
  /// current row's peers.
  range,
};

/// The frame reaches the edge of the partition on this side: `UNBOUNDED PRECEDING` as a start,
/// `UNBOUNDED FOLLOWING` as an end.
struct frame_unbounded {
  bool operator==(frame_unbounded const&) const = default;
};

/// `CURRENT ROW`. In `ROWS` the current row, in `RANGE` the current row and its peers.
struct frame_current_row {
  bool operator==(frame_current_row const&) const = default;
};

/// `n PRECEDING` or `n FOLLOWING` in a `ROWS` frame.
struct frame_row_offset {
  /// The number of rows. Must not be negative.
  std::int64_t rows;
  /// true for `PRECEDING`, false for `FOLLOWING`.
  bool preceding;
  bool operator==(frame_row_offset const&) const = default;
};

/**
 * @brief `δ PRECEDING` or `δ FOLLOWING` in a `RANGE` frame with exactly one order key.
 *
 * `delta` is expressed in the unit of the order key's cuDF type: the integer itself for an integer
 * key, the unscaled value at the key's scale for a decimal key, and a count of the key's
 * resolution for a timestamp key (days for `TIMESTAMP_DAYS`, microseconds for
 * `TIMESTAMP_MICROSECONDS`, ...).
 */
struct frame_range_offset {
  /// The distance. Must not be negative and must fit the key's representation.
  __int128_t delta;
  /// true for `PRECEDING`, false for `FOLLOWING`.
  bool preceding;
  bool operator==(frame_range_offset const&) const = default;
};

/// One end of a window frame.
using frame_bound =
  std::variant<frame_unbounded, frame_current_row, frame_row_offset, frame_range_offset>;

/**
 * @brief The frame of a window aggregate: `unit BETWEEN start AND end`.
 *
 * The frame is host data only. A `RANGE` offset is turned into a device scalar by the evaluator,
 * so building a frame touches no device memory. A frame whose start lies after its end is valid
 * and empty.
 */
struct window_frame {
  frame_unit unit;
  /// The start. frame_unbounded here means `UNBOUNDED PRECEDING`.
  frame_bound start;
  /// The end. frame_unbounded here means `UNBOUNDED FOLLOWING`.
  frame_bound end;
  bool operator==(window_frame const&) const = default;
};

/// The SQL default frame with an ORDER BY: `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`.
inline window_frame default_window_frame()
{
  return {frame_unit::range, frame_unbounded{}, frame_current_row{}};
}

/**
 * @brief Check that @p frame is well formed for a window with @p order_key_count order keys.
 *
 * @throws std::invalid_argument unless every bound fits the unit (frame_row_offset only in `ROWS`,
 * frame_range_offset only in `RANGE`), offsets are not negative, and a frame with a
 * frame_range_offset has exactly one order key.
 */
void validate(window_frame const& frame, std::size_t order_key_count);

}  // namespace sirius::op
