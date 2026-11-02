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

#pragma once

/**
 * @file
 * @brief Constants shared between the host code generator and the generated kernels.
 *
 * Fragment 1 of the NVRTC expression evaluator device runtime.
 */

namespace sirius::nvrtc::abi {

/** @brief Component selector of the `year(...)` .. `microsecond(...)` extraction functions. */
enum class extract_component : int {
  YEAR        = 0,
  MONTH       = 1,
  DAY         = 2,
  HOUR        = 3,
  MINUTE      = 4,
  SECOND      = 5,
  MILLISECOND = 6,
  MICROSECOND = 7,
};

/**
 * @brief Bit set in status word 0 by a kernel when a poisoned (overflowed) value reached a sink.
 */
inline constexpr unsigned status_overflow_bit = 1U;
/**
 * @brief Bit set in status word 0 when a `CASE ... THEN error(...)` arm was selected on some row.
 */
inline constexpr unsigned status_subquery_error_bit = 2U;

}  // namespace sirius::nvrtc::abi
