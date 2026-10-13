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

#include <catch.hpp>
#include <vss/access_path_cost.hpp>
#include <vss/device_rates.hpp>

#include <cmath>

using sirius::vss::access_path_cost;
using sirius::vss::device_rates;
using sirius::vss::device_scale;

TEST_CASE("measure_device_rates measures every resource of the current GPU", "[vss]")
{
  device_rates r;
  REQUIRE(sirius::vss::measure_device_rates(r));
  WARN("fp32 " << r.fp32 << " flop/s, f16 " << r.f16 << ", int8 " << r.int8 << " op/s, pcie "
               << r.pcie << " B/s, device " << r.hbm << " B/s");
  // Sanity bounds that any GPU this builds for clears and none exceeds by far.
  CHECK(r.fp32 > 1e12);
  CHECK(r.fp32 < 2e14);
  CHECK(r.f16 >= r.fp32 * 0.5);
  CHECK(r.int8 >= r.f16 * 0.5);
  CHECK(r.pcie > 1e9);
  CHECK(r.pcie < 1e12);
  CHECK(r.hbm > 5e10);
  CHECK(r.hbm < 2e13);
}

TEST_CASE("the access path cost model scales each rate by its own resource", "[vss]")
{
  access_path_cost const a5000{device_scale{}};
  CHECK(a5000.fp32_gemm == 8.5e12);
  CHECK(a5000.int8_gemm == 77e12);
  CHECK(a5000.pcie == 21e9);

  // Twice the tensor throughput, the same everything else: only the lists' compute side moves.
  access_path_cost const tensor2{device_scale{.f16 = 2, .int8 = 2}};
  CHECK(tensor2.int8_gemm == 2 * 77e12);
  CHECK(tensor2.f16_gemm == 2 * 41e12);
  CHECK(tensor2.fp32_gemm == 8.5e12);
  CHECK(tensor2.sweep0 == 0.125);

  sirius::vss::access_path_shape s;
  s.probe_rows           = 10000;
  s.corpus_rows          = 1e8;
  s.dim                  = 128;
  s.corpus_on_device     = false;
  s.list_bytes_per_value = 1;
  s.lists_on_device      = true;
  CHECK(tensor2.brute(s) == a5000.brute(s));
  CHECK(tensor2.lists(s) < a5000.lists(s));
}

TEST_CASE("lists a statement builds for itself get more clusters for a larger probe side", "[vss]")
{
  using sirius::vss::choose_cluster_count;
  using sirius::vss::scaled_probes;
  access_path_cost const cost{device_scale{}};
  sirius::vss::access_path_shape s;
  s.corpus_rows          = 1e8;
  s.dim                  = 128;
  s.list_bytes_per_value = 1;
  s.lists_on_device      = true;

  // A few thousand probe rows cannot repay a larger build: the reference count, probes as given.
  s.probe_rows     = 1e4;
  auto const small = choose_cluster_count(cost, s, 16);
  CHECK(small.n_clusters == 1024);
  CHECK(small.probes == 16);

  // A self-join can, and its probes follow the square root of the count.
  s.probe_rows    = 1e8;
  auto const self = choose_cluster_count(cost, s, 16);
  CHECK(self.n_clusters > 1024);
  CHECK(self.n_clusters <= 16 * 1024);
  CHECK(self.probes == scaled_probes(16, self.n_clusters, 1024));

  std::int64_t last = 0;
  for (double const m : {1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9}) {
    s.probe_rows = m;
    auto const c = choose_cluster_count(cost, s, 16).n_clusters;
    CHECK(c >= last);
    last = c;
  }

  // Probing every reference cluster asks for the exact answer, which more clusters only delay.
  s.probe_rows = 1e8;
  CHECK(choose_cluster_count(cost, s, 1024).n_clusters == 1024);

  // No count leaves a cluster fewer than 256 rows.
  s.corpus_rows = 1e5;
  CHECK(choose_cluster_count(cost, s, 4).n_clusters == 316);

  CHECK(scaled_probes(16, 1024, 1024) == 16);
  CHECK(scaled_probes(16, 4096, 1024) == 32);
  CHECK(scaled_probes(16, 16384, 1024) == 64);
  CHECK(scaled_probes(5, 2048, 1024) == 8);
  CHECK(scaled_probes(1024, 4096, 1024) == 4096);
  CHECK(scaled_probes(16, 512, 1024) == 16);
  CHECK(scaled_probes(900, 512, 1024) == 512);

  // The two sides of the trade: the build grows with the count, the search falls.
  s.corpus_rows       = 1e8;
  s.probe_rows        = 1e7;
  s.n_clusters        = 1024;
  auto const build_1k = cost.build(s), search_1k = cost.probed(s, 16);
  s.n_clusters = 16384;
  CHECK(cost.build(s) > build_1k);
  CHECK(cost.probed(s, 64) < search_1k);
}
