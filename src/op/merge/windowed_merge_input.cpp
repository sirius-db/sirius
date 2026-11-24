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

#include "op/merge/windowed_merge_input.hpp"

#include "memory/size_arithmetic.hpp"
#include "op/merge/merge_config.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace sirius::op::merge {

namespace {

constexpr std::size_t kShrinkOutputTargetBytesFloor = std::size_t{16} << 20;

/// Halve @p value down to @p floor; a value already at or below the floor is left alone.
bool halve_to_floor(std::size_t& value, std::size_t floor) noexcept
{
  if (value <= floor) { return false; }
  value = std::max(value / 2, floor);
  return true;
}

}  // namespace

windowed_merge_input::windowed_merge_input(std::size_t partition,
                                           std::vector<merge_input_cursor> cursors,
                                           bool final_pass,
                                           std::size_t first_chunk,
                                           window_config cfg,
                                           std::size_t row_width,
                                           std::size_t key_row_width,
                                           std::size_t gpu_runs,
                                           std::size_t fallback_bytes,
                                           std::size_t compressed_runs,
                                           std::size_t fetch_columns,
                                           std::size_t string_columns,
                                           std::size_t string_row_width,
                                           bool round_follows_output)
  : _partition(partition),
    _cursors(std::move(cursors)),
    _final_pass(final_pass),
    _first_chunk(first_chunk),
    _cfg(cfg),
    _row_width(row_width),
    _key_row_width(key_row_width),
    _gpu_runs(gpu_runs),
    _fallback_bytes(fallback_bytes),
    _compressed_runs(compressed_runs),
    _fetch_columns(fetch_columns),
    _string_columns(string_columns),
    _string_row_width(string_row_width),
    _round_follows_output(round_follows_output)
{
}

void windowed_merge_input::prepare_for_processing(
  const cucascade::memory::memory_space* requested_memory_space, ::cuda::stream_ref)
{
  _gpu_memory_space = const_cast<cucascade::memory::memory_space*>(requested_memory_space);
}

std::size_t windowed_merge_input::get_estimated_size_in_bytes() const
{
  // A run whose window buffers the kept merger holds allocates none, like a run sliced in place.
  // One of the runs sliced in place is charged all the same: the spare for a run that is
  // spilled between the planning of the task and its read lock.
  auto const in_place = _gpu_runs > 0 ? _gpu_runs - 1 : std::size_t{0};
  auto const task     = estimate_task_bytes(_cursors.size(),
                                        sirius::memory::saturating_add(in_place, _kept_buffer_runs),
                                        _cfg.window_rows,
                                        _row_width,
                                        _key_row_width,
                                        _cfg.output_target_bytes,
                                        _fallback_bytes,
                                        _fetch_columns,
                                        _string_columns,
                                        _cfg.round_rows,
                                        _string_row_width);
  auto const reader   = estimate_reader_state_bytes(
    _compressed_runs, _fetch_columns, _string_columns, _cfg.window_rows);
  return sirius::memory::saturating_add(task, reader);
}

std::size_t windowed_merge_input::get_estimated_working_set_size_in_bytes() const
{
  return get_estimated_size_in_bytes();
}

bool windowed_merge_input::shrink_for_retry()
{
  _kept_buffer_runs = 0;
  if (_gpu_runs > 0) {
    // The estimate left out the window buffers of the runs that were on the GPU when the task
    // was planned. If they were spilled since, that is what the task ran short of: the retry
    // keeps its sizes and reserves for every run.
    _gpu_runs = 0;
    if (_recharge_observer) { _recharge_observer(); }
    return true;
  }
  const bool window_shrunk = halve_to_floor(_cfg.window_rows, kShrinkWindowRowsFloor);
  const bool output_shrunk =
    halve_to_floor(_cfg.output_target_bytes, kShrinkOutputTargetBytesFloor);
  if (!window_shrunk && !output_shrunk) { return false; }
  if (_cfg.round_rows != 0) {
    _cfg.round_rows = _round_follows_output
                        ? round_rows_for(_cfg.window_rows, _cfg.output_target_bytes, _row_width)
                        : std::max(_cfg.round_rows / 2, _cfg.window_rows);
  }
  if (_shrink_observer) { _shrink_observer(); }
  return true;
}

void windowed_merge_input::set_recharge_observer(std::function<void()> observer)
{
  _recharge_observer = std::move(observer);
}

std::size_t windowed_merge_input::gpu_runs() const noexcept { return _gpu_runs; }

void windowed_merge_input::set_shrink_observer(std::function<void()> observer)
{
  _shrink_observer = std::move(observer);
}

void windowed_merge_input::set_kept_buffer_runs(std::size_t runs) noexcept
{
  _kept_buffer_runs = runs;
}

std::size_t windowed_merge_input::partition() const noexcept { return _partition; }

const std::vector<merge_input_cursor>& windowed_merge_input::cursors() const noexcept
{
  return _cursors;
}

bool windowed_merge_input::final_pass() const noexcept { return _final_pass; }

std::size_t windowed_merge_input::first_chunk() const noexcept { return _first_chunk; }

const window_config& windowed_merge_input::config() const noexcept { return _cfg; }

cucascade::memory::memory_space& windowed_merge_input::gpu_memory_space() const
{
  if (_gpu_memory_space == nullptr ||
      _gpu_memory_space->get_tier() != cucascade::memory::Tier::GPU) {
    throw std::runtime_error("Windowed merge task requires preparation in its reserved GPU space");
  }
  return *_gpu_memory_space;
}

}  // namespace sirius::op::merge
