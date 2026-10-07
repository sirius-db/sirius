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

namespace sirius::vss {

/// What the current GPU reaches on five microbenchmarks, one per resource a vector join's cost
/// depends on: FP32 FMAs on CUDA cores, FLOAT16 and INT8 wmma tiles (the instructions the bounded
/// kernel issues), pinned host-to-device copies, and device-to-device copies (read + write bytes).
struct device_rates {
  double fp32{0};  ///< flop/s
  double f16{0};   ///< flop/s
  double int8{0};  ///< op/s
  double pcie{0};  ///< B/s
  double hbm{0};   ///< B/s
};

/// Runs the microbenchmarks on the current device (~50 ms, 128 MiB of device and 64 MiB of pinned
/// host memory while it runs). False when any step fails; @p out is then unspecified.
bool measure_device_rates(device_rates& out);

/// The current device over the RTX A5000 the vector join's cost constants were fitted on, one
/// ratio per resource: a constant fitted there is scaled by the ratio of the resource it measures.
/// Measured the first time a GPU is used and kept on disk by its UUID (~/.cache/sirius/
/// device_rates); later processes read it. All 1 under SIRIUS_VSS_DEVICE_RATES=reference, or when
/// the measurement fails.
struct device_scale {
  double fp32{1};
  double f16{1};
  double int8{1};
  double pcie{1};
  double hbm{1};
};

device_scale const& current_device_scale();

}  // namespace sirius::vss
