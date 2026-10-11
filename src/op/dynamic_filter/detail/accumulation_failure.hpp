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

#include "op/dynamic_filter/detail/accumulated_bloom_builder.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"

#include <cstdint>
#include <exception>
#include <new>
#include <utility>

namespace sirius::op::detail {

/**
 * @brief How a failed accumulation step ends its optional attempt.
 *
 * ADMISSION and TRANSIENT end the attempt locally; ERROR and LEAK propagate.
 */
enum class accumulation_failure : std::uint8_t {
  ADMISSION,  ///< Host allocation refused (`std::bad_alloc`)
  TRANSIENT,  ///< A kernel launch failed with a retryable code
  ERROR,      ///< Any other failure
  LEAK        ///< Failure cleanup could not join GPU work, so its storage is retained
};

/**
 * @brief Classifies the exception @p error holds.
 */
[[nodiscard]] inline accumulation_failure classify_failure(std::exception_ptr error) noexcept
{
  try {
    std::rethrow_exception(std::move(error));
  } catch (unjoined_gpu_work const&) {
    return accumulation_failure::LEAK;
  } catch (accumulation_cuda_error const& e) {
    return e.transient_launch_failure() ? accumulation_failure::TRANSIENT
                                        : accumulation_failure::ERROR;
  } catch (std::bad_alloc const&) {
    return accumulation_failure::ADMISSION;
  } catch (...) {
    return accumulation_failure::ERROR;
  }
}

/**
 * @brief Whether a failure of @p kind ends the optional attempt without propagating.
 */
[[nodiscard]] constexpr bool recoverable(accumulation_failure kind) noexcept
{
  return kind == accumulation_failure::ADMISSION || kind == accumulation_failure::TRANSIENT;
}

/**
 * @brief Counts a failure of @p kind in @p outcome. A LEAK is also counted as an error.
 */
constexpr void count_failure(dynamic_filter_stats_snapshot& outcome,
                             accumulation_failure kind) noexcept
{
  switch (kind) {
    case accumulation_failure::ADMISSION: outcome.accumulations_skipped_admission = 1; break;
    case accumulation_failure::TRANSIENT: outcome.accumulations_skipped_transient = 1; break;
    case accumulation_failure::LEAK: outcome.accumulation_storage_leaks = 1; [[fallthrough]];
    case accumulation_failure::ERROR: outcome.accumulations_skipped_error = 1; break;
  }
}

}  // namespace sirius::op::detail
