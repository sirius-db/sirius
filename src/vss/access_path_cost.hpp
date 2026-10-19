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
      assign_gemm{25e12 * d.f16},
      fit_per_value{3.7e-9 / d.fp32},
      write_per_value_u8{0.55e-9 / d.hbm},
      write_per_value_f16{1.55e-9 / d.hbm},
      codes_per_value_device{0.3e-9 / d.hbm},
      codes_per_value_host{0.62e-9 / d.pcie}
  {
  }

  double fp32_gemm;    ///< brute-force FP32 GEMM, multiply-adds x 2 per s
  double int8_gemm;    ///< bounded int8 tile kernel on UINT8 rows
  double f16_gemm;     ///< bounded FLOAT16 kernel
  double pcie;         ///< host-pinned rows streamed per query (B/s)
  double hbm;          ///< device-resident lists read by a small batch
  double sweep0;       ///< un-seeded FLOAT16 first sweep, once M >= ~100
  double assign_gemm;  ///< the list build's row-to-centroid assignment (seed pass and FP16 filter)
  double fit_per_value;
  double write_per_value_u8;
  double write_per_value_f16;
  double codes_per_value_device;
  double codes_per_value_host;

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

  /// An approximate join through the lists: each probe row's @p probes nearest clusters scored
  /// on the tensor cores, after its nearest one in full (sweep 0). Meant for comparing cluster
  /// counts on large probe sides; a few thousand probe rows are bound by launches it leaves out.
  [[nodiscard]] double probed(access_path_shape const& s, double probes) const
  {
    double const rate    = s.list_bytes_per_value == 1   ? int8_gemm
                           : s.list_bytes_per_value == 2 ? f16_gemm
                                                         : fp32_gemm;
    double const cluster = s.corpus_rows / s.n_clusters;
    double const touched = std::min(1.0, s.probe_rows * probes / s.n_clusters);
    double const bytes   = touched * s.corpus_rows * s.dim * s.list_bytes_per_value;
    double const work = 2.0 * s.probe_rows * std::min(probes + 1.0, s.n_clusters) * cluster * s.dim;
    return statement + std::max(bytes / (s.lists_on_device ? hbm : pcie), work / rate) +
           (s.probe_rows * probes / s.n_clusters > 128 ? s.n_clusters * slice : 0.0);
  }

  /// Fitting the clustering (k-means on 256 sampled rows a cluster), assigning every row to its
  /// nearest centroid, and writing the lists. Fitted 2026-10-13 on 100M-row builds of 1,024 to
  /// 32,768 clusters (within 10% for 96-d rows, 25% for the 128-d fit). The fit's fixed part varies
  /// with the data (0.65-3.3 s over the grid), so it is rounded up rather than modelled.
  [[nodiscard]] double build(access_path_shape const& s) const
  {
    double const values = s.corpus_rows * s.dim;
    double const train  = std::min(s.corpus_rows, 256.0 * s.n_clusters);
    double const fit = 1.5 + fit_per_value * train * s.dim * std::pow(s.n_clusters / 1024.0, 0.6);
    double const assign = 2.0 * values * s.n_clusters / assign_gemm;
    return fit + assign +
           (s.list_bytes_per_value == 1 ? write_per_value_u8 : write_per_value_f16) * values;
  }

  /// Writing the exact join's compact layout: one list of codes, with no clustering to fit and no
  /// rows to assign, so one pass over the rows. Fitted 2026-10-17 on 1M-100M-row single-list
  /// builds: ~0.3 ns a value into device lists, ~0.6 ns into host lists, plus the one-centroid fit.
  [[nodiscard]] double codes(access_path_shape const& s) const
  {
    double const values = s.corpus_rows * s.dim;
    return 0.06 + values * (s.lists_on_device ? codes_per_value_device : codes_per_value_host);
  }
};

/// The cluster count for lists a statement builds for itself, and the probes that go with it.
struct cluster_count_choice {
  std::int64_t n_clusters{0};
  std::int64_t probes{0};
};

/// The count vector_join_probes is stated against: the square root of the rows, at most 1,024.
[[nodiscard]] inline std::int64_t reference_cluster_count(double corpus_rows)
{
  return std::clamp<std::int64_t>(static_cast<std::int64_t>(std::sqrt(corpus_rows)), 1, 1024);
}

/// @p probes of the reference count, as probes of @p n_clusters. Recall follows probes over the
/// square root of the cluster count, so this holds it: on SIFT100M and Deep100M between 1,024
/// and 16,384 clusters, within 0.003 from recall 0.97 up and higher below it.
[[nodiscard]] inline std::int64_t scaled_probes(std::int64_t probes,
                                                std::int64_t n_clusters,
                                                std::int64_t reference)
{
  // Every reference cluster is every cluster: an exact answer stays one.
  if (probes >= reference) { return n_clusters; }
  if (n_clusters <= reference) { return std::min(probes, n_clusters); }
  auto const scaled =
    std::ceil(static_cast<double>(probes) * std::sqrt(static_cast<double>(n_clusters) / reference));
  return std::min(static_cast<std::int64_t>(scaled), n_clusters);
}

/// More clusters cost the build once (the fit and the assignment grow with the count) and save
/// every probe row search time (it scans probes x rows-per-cluster rows, which at held recall
/// falls with the square root of the count). Picks the count, among the reference and its
/// doublings up to 16 times it with at least 256 rows a cluster, with the least estimated build
/// plus search for @p shape's probe rows; @p probes is stated against the reference.
[[nodiscard]] inline cluster_count_choice choose_cluster_count(access_path_cost const& cost,
                                                               access_path_shape shape,
                                                               std::int64_t probes)
{
  auto const reference = reference_cluster_count(shape.corpus_rows);
  cluster_count_choice best{reference, std::min(probes, reference)};
  double best_cost = 0;
  for (auto n = reference; n <= 16 * reference; n *= 2) {
    if (n > reference && shape.corpus_rows / static_cast<double>(n) < 256.0) { break; }
    shape.n_clusters    = static_cast<double>(n);
    auto const p        = scaled_probes(probes, n, reference);
    double const cost_n = cost.build(shape) + cost.probed(shape, static_cast<double>(p));
    if (n == reference || cost_n < best_cost) {
      best      = cluster_count_choice{n, p};
      best_cost = cost_n;
    }
  }
  return best;
}

}  // namespace sirius::vss
