// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nvtx3/nvtx3.hpp>

namespace simpatico {

/// NVTX domain carrying every range Sirius emits, simpatico's included. Keeping
/// that activity out of the default domain lets a profiler tell it apart from
/// the ranges emitted by the libraries underneath us (`libcudf`,
/// `libcucascade`, ...).
///
/// Sirius declares a second tag type for the same domain in
/// src/include/telemetry/nvtx.hpp, because neither include tree can reach the
/// other: simpatico also builds standalone, without src/include on the include
/// path, and the parquet_benchmark target is given src/include but not
/// simpatico's include root. The name cannot drift between the two, though — it
/// arrives from the single SIRIUS_NVTX_DOMAIN_NAME compile definition set in
/// this directory's CMakeLists.txt. NVTX keys domains by name, so both tag types
/// resolve to one domain in the profiler.
struct nvtx_domain {
  static constexpr char const* name{SIRIUS_NVTX_DOMAIN_NAME};
};

/// RAII range in the Sirius domain; the drop-in replacement for
/// `nvtx3::scoped_range`, which would publish into the default domain.
using nvtx_scoped_range = nvtx3::scoped_range_in<nvtx_domain>;

}  // namespace simpatico
