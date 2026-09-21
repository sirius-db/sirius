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

#include <nvtx3/nvtx3.hpp>

namespace sirius {

/// NVTX domain carrying every range Sirius emits. Keeping Sirius activity out of
/// the default domain lets a profiler tell it apart from the ranges emitted by
/// the libraries underneath us (`libcudf`, `libcucascade`, ...).
///
/// Simpatico declares a second tag type for the same domain in
/// codegen/util/nvtx.hpp, because neither include tree can reach the other:
/// simpatico also builds standalone, without src on the include path, and the
/// parquet_benchmark target is given src but not simpatico's
/// include root. The name cannot drift between the two, though — it arrives from
/// the single SIRIUS_NVTX_DOMAIN_NAME compile definition derived from the CMake
/// project name. NVTX keys domains by name, so both tag types resolve to one
/// domain in the profiler.
struct nvtx_domain {
  static constexpr char const* name{SIRIUS_NVTX_DOMAIN_NAME};
};

/// RAII range in the Sirius domain; the drop-in replacement for
/// `nvtx3::scoped_range`, which would publish into the default domain.
using nvtx_scoped_range = nvtx3::scoped_range_in<nvtx_domain>;

}  // namespace sirius
