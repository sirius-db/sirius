// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/validators.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace sirius::config {

/// `sirius.topology`: which GPUs the engine uses.
struct topology_options : block_base<topology_options> {
  /// How many GPUs (`all`, held as 0 = every visible one), or which device ids. Ignored when
  /// sirius.space lists configure memory spaces directly.
  config_option<std::size_t, std::vector<int>> gpus =
    option<std::size_t>("num_gpus")
      // `all` is spelled out: a count of 0 read as "every GPU" was easy to misread.
      .parser([](const node& n, std::size_t& out) -> parse_status {
        if (n.is_scalar() && n.text() == "all") {
          out = 0;
          return {};
        }
        std::size_t count = 0;
        if (auto status = read_value(n, count); !status) { return status; }
        if (count == 0) {
          return parse_status::failure(
            "must be all or a positive count; use num_gpus: all for every visible GPU");
        }
        out = count;
        return {};
      })
      .writer([](const std::size_t& count) {
        return count == 0 ? std::string("all") : std::to_string(count);
      })
      .or_type<std::vector<int>>("gpu_ids")
      .validate([](const std::vector<int>& ids) -> parse_status {
        if (ids.empty()) { return parse_status::failure("must contain at least one device id"); }
        if (std::ranges::any_of(ids, [](int id) { return id < 0; })) {
          return parse_status::failure("must contain only non-negative device ids");
        }
        auto sorted = ids;
        std::ranges::sort(sorted);
        if (std::ranges::adjacent_find(sorted) != sorted.end()) {
          return parse_status::failure("must not contain duplicate device ids");
        }
        return {};
      })
      .fallback(std::size_t{0})
      .describe("GPUs Sirius uses, by count or by device id")
      .build();

  /// GPUs admitted per query at first; 0 = every active GPU.
  config_option<int> gpus_per_query = option<int>("gpus_per_query")
                                        .validate(non_negative<int>{})
                                        .fallback(0)
                                        .describe("GPUs allocated to each query at admission")
                                        .build();

  auto fields() { return field_list(gpus, gpus_per_query); }
};

}  // namespace sirius::config
