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

#include "op/order/sampled_merge.hpp"

#include <algorithm>
#include <functional>
#include <vector>

namespace sirius::test {

struct sampled_merge_reference_plan {
  std::vector<op::sampled_merge::coordinate> splitters;
  // Each row includes the 0 and input.rows endpoints. Empty inputs are allowed
  // in this reference, but excluded from the fan-in used to select the stride.
  std::vector<std::vector<uint64_t>> cuts;
  uint64_t samples{0};
};

template <typename T, typename Less = std::less<T>>
sampled_merge_reference_plan reference_sampled_cuts(std::vector<std::vector<T>> const& inputs,
                                                    uint64_t row_limit,
                                                    Less less = {})
{
  namespace sm = op::sampled_merge;
  if (!row_limit) { throw std::invalid_argument("Sampled merge row limit must be positive"); }
  uint64_t rows = 0, k = 0;
  for (auto const& input : inputs) {
    if (!std::is_sorted(input.begin(), input.end(), less)) {
      throw std::invalid_argument("Sampled merge reference requires sorted contributions");
    }
    rows = sm::checked_add(rows, input.size());
    k += !input.empty();
  }
  sampled_merge_reference_plan result;
  if (rows > row_limit) {
    auto p = sm::choose_parameters(k, row_limit);
    std::vector<sm::coordinate> samples;
    for (uint64_t i = 0; i < inputs.size(); ++i) {
      for (uint64_t j = 0; j < sm::sample_count(inputs[i].size(), p.stride); ++j) {
        samples.push_back({i, sm::sample_row(j, p.stride)});
      }
    }
    auto conceptual_less = [&](sm::coordinate a, sm::coordinate b) {
      auto const& x = inputs[a.input][a.row];
      auto const& y = inputs[b.input][b.row];
      if (less(x, y)) { return true; }
      if (less(y, x)) { return false; }
      return a < b;
    };
    std::sort(samples.begin(), samples.end(), conceptual_less);
    result.samples = samples.size();
    auto count     = samples.size() / p.sample_step;
    for (uint64_t t = 0; t < count; ++t) {
      result.splitters.push_back(samples[sm::sample_row(t, p.sample_step)]);
    }
  }
  for (uint64_t i = 0; i < inputs.size(); ++i) {
    auto const& input = inputs[i];
    std::vector<uint64_t> cuts{0};
    for (auto splitter : result.splitters) {
      auto const& key = inputs[splitter.input][splitter.row];
      auto lower      = std::lower_bound(input.begin(), input.end(), key, less) - input.begin();
      auto upper      = std::upper_bound(input.begin(), input.end(), key, less) - input.begin();
      cuts.push_back(sm::conceptual_cut(i, splitter, lower, upper));
    }
    cuts.push_back(input.size());
    result.cuts.push_back(std::move(cuts));
  }
  return result;
}

}  // namespace sirius::test
