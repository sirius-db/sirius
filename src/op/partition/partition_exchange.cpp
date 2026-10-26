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

#include "op/partition/partition_exchange.hpp"

#include "log/logging.hpp"
#include "op/sirius_physical_partition.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <algorithm>
#include <stdexcept>

namespace sirius::op {
namespace {

char const* mode_name(partition_mode mode)
{
  switch (mode) {
    case partition_mode::hash: return "hash";
    case partition_mode::broadcast: return "broadcast";
    case partition_mode::single: return "single";
    case partition_mode::local: return "local";
  }
  return "unknown";
}

}  // namespace

partition_exchange::partition_exchange(std::vector<sirius_physical_partition*> inputs,
                                       std::size_t sizing_input,
                                       bool requires_all_inputs,
                                       choose_layouts choose)
  : _inputs(std::move(inputs)),
    _sizing_input(sizing_input),
    _requires_all_inputs(requires_all_inputs),
    _choose(std::move(choose))
{
  if (_inputs.empty() || sizing_input >= _inputs.size() || !_choose) {
    throw std::invalid_argument("partition exchange requires inputs, a sizing input, and a policy");
  }
  for (std::size_t i = 0; i < _inputs.size(); ++i) {
    if (!_inputs[i] ||
        std::find(_inputs.begin(), _inputs.begin() + i, _inputs[i]) != _inputs.begin() + i) {
      throw std::invalid_argument("partition exchange inputs must be non-null and distinct");
    }
  }
}

void partition_exchange::bind(std::vector<sirius_physical_partition*> inputs,
                              std::size_t sizing_input,
                              bool requires_all_inputs,
                              choose_layouts choose)
{
  auto state = std::shared_ptr<partition_exchange>(new partition_exchange(
    std::move(inputs), sizing_input, requires_all_inputs, std::move(choose)));
  for (auto* input : state->_inputs) {
    input->set_exchange(state);
  }
}

std::optional<task_creation_hint> partition_exchange::prepare() { return prepare_impl(true); }
std::optional<task_creation_hint> partition_exchange::hint() { return prepare_impl(false); }

std::optional<task_creation_hint> partition_exchange::prepare_impl(bool resolve)
{
  std::lock_guard guard(_mutex);
  if (_resolved) { return std::nullopt; }
  if (_requires_all_inputs) {
    for (auto* input : _inputs) {
      if (!input->input_complete()) {
        for (auto const& port : input->get_ports()) {
          if (port->src_pipeline && !port->src_pipeline->is_pipeline_finished()) {
            return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA,
                                      &port->src_pipeline->get_operators()[0].get()};
          }
        }
      }
    }
  }
  auto& driver = *_inputs[_sizing_input];
  if (!driver.input_complete()) {
    auto hint = driver.sizing_hint();
    if (hint && hint->hint == TaskCreationHint::WAITING_FOR_INPUT_DATA) { return hint; }
    if (!hint && !driver.input_complete()) {
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, &driver};
    }
  }
  // Completion can change while obtaining the hint. Decide emptiness after that observation;
  // otherwise a producer finishing empty between the two polls could select a whole-build mode.
  auto const empty = driver.input_complete() && driver.input_empty();
  if (!resolve && !empty) { return task_creation_hint{TaskCreationHint::READY, &driver}; }
  std::vector<partition_input_stats> stats;
  for (auto* input : _inputs) {
    stats.push_back(input->input_statistics(_inputs.size() == 1));
  }
  auto layouts = _choose(stats, empty);
  if (layouts.size() != _inputs.size()) {
    throw std::logic_error("partition policy must resolve every bound input");
  }
  for (auto const& layout : layouts) {
    if (*layout.placement != *layouts.front().placement) {
      throw std::logic_error("partition exchange inputs must share a placement");
    }
  }
  SIRIUS_LOG_DEBUG(
    "partition_exchange resolved {} inputs on sizing input {}: {} partitions, mode {}, sized from "
    "{} bytes and {} rows, placement {}",
    _inputs.size(),
    _sizing_input,
    layouts.front().placement->num_partitions(),
    mode_name(layouts.front().mode),
    stats[_sizing_input].bytes,
    stats[_sizing_input].rows,
    layouts.front().placement->to_string());
  for (std::size_t i = 0; i < layouts.size(); ++i) {
    _inputs[i]->set_layout(std::move(layouts[i]));
  }
  _resolved = true;
  return std::nullopt;
}

}  // namespace sirius::op
