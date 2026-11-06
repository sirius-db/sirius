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

#include <compare>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace sirius::op::sampled_merge {

inline uint64_t checked_add(uint64_t a, uint64_t b)
{
  if (b > std::numeric_limits<uint64_t>::max() - a) {
    throw std::overflow_error("Sampled merge addition overflow");
  }
  return a + b;
}

inline uint64_t checked_mul(uint64_t a, uint64_t b)
{
  if (a && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error("Sampled merge multiplication overflow");
  }
  return a * b;
}

struct parameters {
  uint64_t stride;
  uint64_t sample_step;
  uint64_t uncertainty;
  uint64_t max_tile_rows;
};

// k counts frozen nonempty contributions. The caller caps R to cuDF's native
// limit; the arithmetic itself is independent of CUDA and valid over uint64_t.
inline parameters choose_parameters(uint64_t k, uint64_t row_limit)
{
  if (!k || !row_limit) {
    throw std::invalid_argument("Sampled merge fan-in and row limit must be positive");
  }
  auto stride      = row_limit / k / 4 + 1;
  auto uncertainty = checked_mul(k, stride - 1);
  auto sample_step = (row_limit - uncertainty) / stride;
  auto bound       = checked_add(checked_mul(stride, sample_step), uncertainty);
  return {stride, sample_step, uncertainty, bound};
}

inline uint64_t sample_count(uint64_t rows, uint64_t stride)
{
  if (!stride) { throw std::invalid_argument("Sample stride must be positive"); }
  return rows / stride;
}

// Zero-based sample index; no incomplete tail is sampled.
inline uint64_t sample_row(uint64_t index, uint64_t stride)
{
  if (!stride) { throw std::invalid_argument("Sample stride must be positive"); }
  return checked_add(checked_mul(index, stride), stride - 1);
}

struct coordinate {
  uint64_t input;
  uint64_t row;
  auto operator<=>(coordinate const&) const = default;
};

struct contribution_size {
  uint64_t rows;
  uint64_t bytes;
};

struct coalescing_group {
  std::size_t begin;
  std::size_t end;
  uint64_t rows;
  uint64_t bytes;
};

// One fixed pass over normalized nonempty inputs of one range. Large singletons
// pass through; the caller admits multi-input groups including output/workspace.
// Groups are never recombined after completion, even when outputs are small.
inline std::vector<coalescing_group> plan_coalescing(std::span<contribution_size const> inputs,
                                                     uint64_t chunk_bytes,
                                                     uint64_t native_rows)
{
  if (!chunk_bytes || !native_rows) {
    throw std::invalid_argument("Coalescing byte and row targets must be positive");
  }
  std::vector<coalescing_group> result;
  coalescing_group pending{0, 0, 0, 0};
  auto flush = [&] {
    if (pending.begin != pending.end) { result.push_back(pending); }
    pending = {pending.end, pending.end, 0, 0};
  };
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (!input.rows || input.rows > native_rows) {
      throw std::invalid_argument("Coalescing requires normalized nonempty contributions");
    }
    if (input.bytes >= chunk_bytes) {
      flush();
      result.push_back({i, i + 1, input.rows, input.bytes});
      pending = {i + 1, i + 1, 0, 0};
      continue;
    }
    if (input.bytes > chunk_bytes - pending.bytes || input.rows > native_rows - pending.rows) {
      flush();
    }
    pending.end   = i + 1;
    pending.rows  = checked_add(pending.rows, input.rows);
    pending.bytes = checked_add(pending.bytes, input.bytes);
  }
  flush();
  return result;
}

// SQL comparator equivalence is represented by [lower, upper). Coordinate
// order only resolves that interval, without comparing floating values for ==.
inline uint64_t conceptual_cut(uint64_t input, coordinate splitter, uint64_t lower, uint64_t upper)
{
  if (input < splitter.input) { return upper; }
  if (input > splitter.input) { return lower; }
  return checked_add(splitter.row, 1);
}

}  // namespace sirius::op::sampled_merge
