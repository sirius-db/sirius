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

#include <codegen/selection/decode_policy.hpp>
#include <log/logging.hpp>

namespace sirius {

//===----------------------------------------------------------------------===//
// Policy for filtering during decompression
//===----------------------------------------------------------------------===//
//
// The knobs themselves live with the decode that acts on them
// (codegen/selection/decode_policy.hpp) — one definition each, so the scan, the
// memory estimator and the decode cannot disagree about whether the feature is
// on or how selective a batch has to be. Re-exported here because this is the
// header the scan side reaches for.

using sirius::codegen::decode_filtering_enabled;
using sirius::codegen::decode_max_selectivity;

/// Promote the decision trace to INFO (SIRIUS_EXP_FUSED_SCAN_DIAG).
inline bool decode_filter_diag_enabled() { return sirius::codegen::decode_diag_enabled(); }

}  // namespace sirius

/// Routes one decision-trace line to INFO when the diag env is set, DEBUG
/// otherwise. A macro (not a function) so the level dispatch keeps the
/// call-site file/line and the lazy formatting of the underlying macros.
#define SIRIUS_DECODE_DIAG(...)                   \
  do {                                            \
    if (::sirius::decode_filter_diag_enabled()) { \
      SIRIUS_LOG_INFO(__VA_ARGS__);               \
    } else {                                      \
      SIRIUS_LOG_DEBUG(__VA_ARGS__);              \
    }                                             \
  } while (0)
