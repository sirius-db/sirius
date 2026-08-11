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

#include <log/logging.hpp>

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace sirius {

//===----------------------------------------------------------------------===//
// Policy for filtering during decompression
//===----------------------------------------------------------------------===//
//
// One reader per knob, shared by everything that has to agree on it: the scan
// (does it bother analysing its filter?), the memory estimator (will the batch
// come back smaller?) and the decoder itself. They used to be copied per
// translation unit and had already drifted — one accepted only "1" where the
// decoder accepts anything but "0" — so a value like "true" turned the feature
// on in one layer and off in another.
//
// The decode-side thresholds not read here (how selective the filter must be
// for each output shape, how many join filters one decode carries) live with
// the decode implementation, which is the only thing that can act on them.

/// Master gate for applying a scan's filter during decompression
/// (SIRIUS_EXP_FUSED_SCAN_FILTER). Set and not exactly "0" = on.
inline bool decode_filtering_enabled()
{
  static bool const enabled = [] {
    char const* value = std::getenv("SIRIUS_EXP_FUSED_SCAN_FILTER");
    return value != nullptr && std::string_view{value} != "0";
  }();
  return enabled;
}

/// Promote the decision trace to INFO (SIRIUS_EXP_FUSED_SCAN_DIAG), same "set
/// and not exactly 0" contract.
///
/// The trace is permanent tooling, not temporary instrumentation: it records
/// every accept/decline decision, and raising the level is the first move
/// whenever a batch quietly falls back to a plain decode. It stays at DEBUG
/// otherwise, which harness runs drop at the sink.
inline bool decode_filter_diag_enabled()
{
  static bool const enabled = [] {
    char const* value = std::getenv("SIRIUS_EXP_FUSED_SCAN_DIAG");
    return value != nullptr && std::string_view{value} != "0";
  }();
  return enabled;
}

/// Surviving-row fraction above which the decode gives up compaction and
/// produces ordinary full-width columns (SIRIUS_EXP_FUSED_SCAN_MAX_SEL,
/// default 0.35). Every batch that DOES come back compacted is bounded by it,
/// which is what makes a memory reservation off it sound.
///
/// Mirrors the decode implementation's own reader — keep the default in sync.
inline double decode_max_selectivity()
{
  static double const value = [] {
    char const* s = std::getenv("SIRIUS_EXP_FUSED_SCAN_MAX_SEL");
    if (s == nullptr || *s == '\0') { return 0.35; }
    char* end      = nullptr;
    double const d = std::strtod(s, &end);
    if (end == s || d <= 0.0) { return 0.35; }
    return std::min(d, 1.0);
  }();
  return value;
}

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
