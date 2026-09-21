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

#include "helper/logical_type.hpp"
#include "op/scan/duckdb_native_metadata.hpp"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Streaming, order-independent row-group -> batch coalescer.
//
// The single consumer feeds each parsed row group in via push(), in any order;
// the coalescer packs them into batches under two caps:
//   - total decoded bytes per batch  <= approximate_batch_size (0 disables), and
//   - per-varchar-column chars per batch < kCudfInt32StringsThreshold.
// A batch closes and becomes available via has_ready()/pop_ready() once the next
// row group would breach a cap. flush() returns the final accumulator remainder.
// Rowids are absolute per row group (row_group_start), so packing order does not
// affect correctness.
//===----------------------------------------------------------------------===//
class batch_coalescer {
 public:
  batch_coalescer(std::size_t approximate_batch_size,
                  const std::vector<sirius::logical_type>& projected_types)
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

  /// Add one parsed row group. Empty row groups (row_count == 0) are dropped.
  /// An over-cap row group becomes the sole member of its own batch.
  void push(duckdb_row_group_metadata rg)
  {
    if (rg.row_count == 0) { return; }

    auto const this_bytes = rg.decoded_bytes_budget;
    if (!_current.empty() && would_close(this_bytes, rg)) { close_current(); }

    _current_bytes += this_bytes;
    if (_any_varchar) {
      for (std::size_t c = 0; c < _is_varchar.size(); ++c) {
        if (_is_varchar[c]) { _current_varchar[c] += rg.varchar_bytes_per_col[c]; }
      }
    }
    _current.push_back(std::move(rg));
  }

  /// True iff a completed (cap-closed) batch is queued for the consumer.
  [[nodiscard]] bool has_ready() const { return !_ready.empty(); }

  /// True iff nothing remains: no completed batch queued and no row group
  /// accumulating.
  [[nodiscard]] bool empty() const { return _ready.empty() && _current.empty(); }

  /// Pop the oldest completed batch. Precondition: has_ready().
  std::vector<duckdb_row_group_metadata> pop_ready()
  {
    auto batch = std::move(_ready.front());
    _ready.pop_front();
    return batch;
  }

  /// Return the in-progress accumulator as the final tail batch. Call after all
  /// row groups have been pushed. May be empty.
  std::vector<duckdb_row_group_metadata> flush()
  {
    auto tail = std::move(_current);
    reset_current();
    return tail;
  }

 private:
  [[nodiscard]] bool would_close(std::size_t this_bytes, const duckdb_row_group_metadata& rg) const
  {
    if (_cap_bytes > 0 && _current_bytes + this_bytes > _cap_bytes) { return true; }
    if (_any_varchar) {
      for (std::size_t c = 0; c < _is_varchar.size(); ++c) {
        if (_is_varchar[c] &&
            _current_varchar[c] + rg.varchar_bytes_per_col[c] >= kCudfInt32StringsThreshold) {
          return true;
        }
      }
    }
    return false;
  }

  void close_current()
  {
    _ready.push_back(std::move(_current));
    reset_current();
  }

  void reset_current()
  {
    _current.clear();
    _current_bytes = 0;
    std::fill(_current_varchar.begin(), _current_varchar.end(), std::size_t{0});
  }

  std::size_t _cap_bytes;
  std::vector<bool> _is_varchar;
  bool _any_varchar = false;

  std::vector<duckdb_row_group_metadata> _current;
  std::size_t _current_bytes = 0;
  std::vector<std::size_t> _current_varchar;
  std::deque<std::vector<duckdb_row_group_metadata>> _ready;
};

}  // namespace sirius::op::scan
