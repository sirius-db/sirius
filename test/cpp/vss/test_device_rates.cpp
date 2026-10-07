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
