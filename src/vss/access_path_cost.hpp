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

#include "vss/device_rates.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sirius::vss {

/// What the planner knows about one exact vector join over a pinned corpus when it picks how to
/// run it: by brute force over the pinned rows, or through cluster lists searched in full.
struct access_path_shape {
  double probe_rows{0};          ///< estimated rows on the probe side (M)
  double corpus_rows{0};         ///< N
  double dim{0};                 ///< d
  bool corpus_on_device{false};  ///< the pin is GPU-resident (otherwise streamed from the host)
  // The lists, when there are (or would be) any:
  double list_bytes_per_value{1};  ///< 1 for UINT8, 2 for FLOAT16, 4 for FLOAT32
  bool lists_on_device{true};
  double n_clusters{1024};
  bool inexact_unseeded{false};  ///< FLOAT16 rows in clusters too large to seed (a fixed sweep 0)
};

/// Seconds for an exact join, by an analytic model fitted on the A5000 grid (RTX A5000 24 GB,
/// measured 2026-10-01; within ~15% of the measured cells it was fitted on). On another GPU each
/// rate is scaled by how that device compares on the resource the rate measures (device_scale);
/// the fixed overheads (statement, slice, chunk) are launch and synchronisation costs and are kept.
/// Only the comparison between the paths matters, so constants shared by both are dropped.
struct access_path_cost {
  static constexpr double statement = 0.004;   ///< per-statement floor (s)
  static constexpr double slice     = 1e-4;    ///< per cluster slice launched alone (> 128 rows)
  static constexpr double chunk     = 3.5e-3;  ///< per pinned chunk of a small brute batch

  explicit access_path_cost(device_scale const& d)
    : fp32_gemm{8.5e12 * d.fp32},
      int8_gemm{77e12 * d.int8},
      f16_gemm{41e12 * d.f16},
      pcie{21e9 * d.pcie},
      hbm{500e9 * d.hbm},
      sweep0{0.25 / d.f16},
      fit_per_value{1e-9 / d.fp32},
      write_per_value_u8{1.0e-9 / d.hbm},
      write_per_value_f16{1.4e-9 / d.hbm}
  {
  }

  double fp32_gemm;  ///< brute-force FP32 GEMM, multiply-adds x 2 per s
  double int8_gemm;  ///< bounded int8 tile kernel on UINT8 rows
  double f16_gemm;   ///< bounded FLOAT16 kernel
  double pcie;       ///< host-pinned rows streamed per query (B/s)
  double hbm;        ///< device-resident lists read by a small batch
  double sweep0;     ///< un-seeded FLOAT16 first sweep, once M >= ~100
  double fit_per_value;
  double write_per_value_u8;
  double write_per_value_f16;

  [[nodiscard]] double brute(access_path_shape const& s) const
  {
    double const work = 2.0 * s.probe_rows * s.corpus_rows * s.dim;
    if (!s.corpus_on_device) {
      return statement + std::max(4.0 * s.corpus_rows * s.dim / pcie, work / fp32_gemm);
    }
    double const chunks = std::ceil(4.0 * s.corpus_rows * s.dim / 512e6);
    return statement + (s.probe_rows < 512 ? chunks * chunk : 0.0) + work / fp32_gemm;
  }

  [[nodiscard]] double lists(access_path_shape const& s) const
  {
    double const work  = 2.0 * s.probe_rows * s.corpus_rows * s.dim;
    double const rate  = s.list_bytes_per_value == 1   ? int8_gemm
                         : s.list_bytes_per_value == 2 ? f16_gemm
                                                       : fp32_gemm;
    double const bytes = s.corpus_rows * s.dim * s.list_bytes_per_value;
    return statement + std::max(bytes / (s.lists_on_device ? hbm : pcie), work / rate) +
           (s.probe_rows > 128 ? s.n_clusters * slice : 0.0) +
           (s.inexact_unseeded && s.probe_rows >= 100 ? sweep0 : 0.0);
  }

  /// Fitting the clustering (k-means on at most 2M sampled rows) and writing the lists. The fit
  /// varies with the data (0.65-3.3 s over the grid), so it is rounded up rather than modelled.
  [[nodiscard]] double build(access_path_shape const& s) const
  {
    double const train = std::min(s.corpus_rows, 2e6);
    return 1.5 + fit_per_value * train * s.dim +
           (s.list_bytes_per_value == 1 ? write_per_value_u8 : write_per_value_f16) *
             s.corpus_rows * s.dim;
  }
};

}  // namespace sirius::vss
