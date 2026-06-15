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

// sirius
#include <helper/logical_type.hpp>
#include <op/scan/coalescing_unit.hpp>
#include <op/scan/duckdb_native_metadata.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <deque>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief A streaming batch coalescer.
 *
 * The coalescer packs coalescing units (the parse units of scan manager worker threads) into
 * batches under 2 caps:
 *  - total decoded bytes per batch <= approximate_batch_size
 *  - per-varchar-column chars per batch < kCudfInt32StringsThreshold
 * A batch closes and becomes available via has_read()/pop_read() once the next coalescing unit
 * would breach a cap. flush() returns the final accumulator remainder.
 */
class batch_coalescer {
 public:
  batch_coalescer(std::size_t approximate_batch_size,
                  std::vector<sirius::logical_type> const& projected_types)
    : _cap_bytes(approximate_batch_size)
  {
    _is_varchar.reserve(projected_types.size());
    for (auto const& t : projected_types) {
      auto const is_v = t.is_varchar();
      _is_varchar.push_back(is_v);
      _any_varchar = _any_varchar || is_v;
    }
    _current_varchar.assign(projected_types.size(), 0);
  }
  explicit batch_coalescer(std::size_t approximate_batch_size) : _cap_bytes(approximate_batch_size)
  {
  }

  /**
   * @brief Add one coalescing_unit to the current accumulating set.
   *
   * Empty units (row_count == 0) are dropped. A coalescing unit that makes the current set exceed
   * the cap becomes the sole member of a new set, and the accumulated set is closed into a batch.
   */
  void push(coalescing_unit unit)
  {
    if (unit.row_count == 0) { return; }

    auto const this_bytes = unit.decoded_bytes;
    if (!_current.empty() && would_close(this_bytes, unit)) { close_current(); }

    _current_bytes += this_bytes;
    if (_any_varchar) {
      for (std::size_t c = 0; c < _is_varchar.size(); ++c) {
        if (_is_varchar[c]) { _current_varchar[c] += unit.varchar_bytes_per_col[c]; }
      }
    }
    if (_current.empty()) { _current_group = unit.group_key; }
    _current.push_back(std::move(unit));
  }

  /**
   * @brief True iff a completed batch is queued for the consumer.
   */
  [[nodiscard]] bool has_ready() const { return !_ready.empty(); }

  /**
   * @brief True iff nothing remains: no completed batch queued and no coalescing unit accumulating.
   */
  [[nodiscard]] bool empty() const { return _ready.empty() && _current.empty(); }

  /**
   * @brief Pop the oldest completed batch.
   *
   * Precondition: has_ready().
   */
  std::vector<coalescing_unit> pop_ready()
  {
    assert(has_ready());
    auto batch = std::move(_ready.front());
    _ready.pop_front();
    return batch;
  }

  /**
   * @brief return the in-progress accumulating set as the final tail. Call after all coalescing
   * units have been pushed.
   */
  std::vector<coalescing_unit> flush()
  {
    auto tail = std::move(_current);
    reset_current();
    return tail;
  }

 private:
  /**
   * @brief True iff the given unit would cause the current set to exceed its caps.
   */
  [[nodiscard]] bool would_close(std::size_t this_bytes, coalescing_unit const& unit) const
  {
    if (_cap_bytes > 0 && _current_bytes + this_bytes > _cap_bytes) { return true; }
    if (_any_varchar) {
      for (std::size_t c = 0; c < _is_varchar.size(); ++c) {
        if (_is_varchar[c] &&
            _current_varchar[c] + unit.varchar_bytes_per_col[c] >= kCudfInt32StringsThreshold) {
          return true;
        }
      }
    }
    if (unit.group_key != _current_group) { return true; }  // never merge across groups
    return false;
  }

  /**
   * @brief Close the current set of coalescing units into a batch and restart accumulation with a
   * new set.
   */
  void close_current()
  {
    _ready.push_back(std::move(_current));
    reset_current();
  }

  /**
   * @brief Reset the current accumulating set of coalescing units.
   */
  void reset_current()
  {
    _current.clear();
    _current_bytes = 0;
    std::fill(_current_varchar.begin(), _current_varchar.end(), std::size_t{0});
    _current_group.clear();
  }

  //===----------Fields----------===//
  std::size_t _cap_bytes;
  std::vector<bool> _is_varchar;
  bool _any_varchar = false;

  std::vector<coalescing_unit> _current;
  std::size_t _current_bytes = 0;
  std::vector<std::size_t> _current_varchar;
  std::vector<std::string> _current_group;

  std::deque<std::vector<coalescing_unit>> _ready;
};

}  // namespace sirius::op::scan
