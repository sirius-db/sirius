// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <compare>
#include <cstdint>
#include <optional>

namespace sirius {
// Identifies logical work, not physical storage or an individual retry attempt.
// Different tasks consuming the same shared batch have different identities.
struct logical_task_id {
  std::uint64_t query_id                         = 0;
  std::uint64_t original_task_id                 = 0;
  auto operator<=>(logical_task_id const&) const = default;
};

struct operator_execution_context {
  constexpr operator_execution_context(std::uint64_t query_id,
                                       std::uint64_t attempt,
                                       std::optional<std::uint64_t> original = std::nullopt)
    : logical_task{query_id, original.value_or(attempt)}, attempt_id(attempt)
  {
  }
  logical_task_id logical_task;
  std::uint64_t attempt_id;
};
}  // namespace sirius
