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

#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <vss/cluster_lists.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

// rotate_rows promises each output is one fmaf chain over k = 0..d-1 in order: a row's codes
// never depend on the rows it was batched with, and rotation_slack bounds that chain's error.
// The CPU's fmaf is the same correctly rounded operation, so the two must agree bit for bit.
TEST_CASE("rotate_rows is an in-order fmaf chain per output whatever the batch", "[vss]")
{
  auto const stream = cudf::get_default_stream();
  std::mt19937 gen(3);
  std::normal_distribution<float> normal(0.f, 3.f);
  for (std::int64_t const d : {16, 48, 96, 128, 160, 256, 100}) {
    INFO("d " << d);
    std::int64_t const rows = 1000;  // not a multiple of any tile
    auto const r            = sirius::vss::random_rotation(d, 11);
    std::vector<float> x(static_cast<std::size_t>(rows * d));
    for (auto& v : x) {
      v = normal(gen);
    }
    rmm::device_uvector<float> dx(x.size(), stream), dr(r.size(), stream), doubt(x.size(), stream);
    REQUIRE(cudaMemcpy(dx.data(), x.data(), x.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(dr.data(), r.data(), r.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    // The whole batch, and the same rows as a 37-row batch from an odd offset.
    sirius::vss::rotate_rows(dx.data(), rows, d, dr.data(), doubt.data(), stream);
    std::vector<float> got(x.size());
    stream.synchronize();
    REQUIRE(cudaMemcpy(got.data(), doubt.data(), got.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    std::int64_t const off = 101, part = 37;
    sirius::vss::rotate_rows(dx.data() + off * d, part, d, dr.data(), doubt.data(), stream);
    std::vector<float> got_part(static_cast<std::size_t>(part * d));
    stream.synchronize();
    REQUIRE(cudaMemcpy(got_part.data(), doubt.data(), got_part.size() * 4, cudaMemcpyDeviceToHost) ==
            cudaSuccess);
    std::int64_t mismatches = 0;
    for (std::int64_t i = 0; i < rows; ++i) {
      for (std::int64_t j = 0; j < d; ++j) {
        float acc = 0.f;
        for (std::int64_t k = 0; k < d; ++k) {
          acc = std::fmaf(x[i * d + k], r[j * d + k], acc);
        }
        mismatches += std::memcmp(&acc, &got[i * d + j], sizeof(float)) != 0;
        if (i >= off && i < off + part) {
          mismatches += std::memcmp(&acc, &got_part[(i - off) * d + j], sizeof(float)) != 0;
        }
      }
    }
    REQUIRE(mismatches == 0);

    // The range taken while rotating, never writing the rows, is column_min_max's over them.
    rmm::device_uvector<unsigned int> lo(d, stream), hi(d, stream), lo2(d, stream), hi2(d, stream);
    for (auto* v : {&lo, &lo2}) {
      REQUIRE(cudaMemsetAsync(v->data(), 0xff, d * 4, stream.value()) == cudaSuccess);
    }
    for (auto* v : {&hi, &hi2}) {
      REQUIRE(cudaMemsetAsync(v->data(), 0, d * 4, stream.value()) == cudaSuccess);
    }
    sirius::vss::rotate_rows(dx.data(), rows, d, dr.data(), doubt.data(), stream);
    sirius::vss::column_min_max(doubt.data(), rows, d, lo.data(), hi.data(), stream);
    if (sirius::vss::rotate_rows_min_max(
          dx.data(), rows, d, dr.data(), lo2.data(), hi2.data(), stream)) {
      std::vector<unsigned int> a(d), b(d), c(d), e(d);
      stream.synchronize();
      REQUIRE(cudaMemcpy(a.data(), lo.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(b.data(), lo2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(c.data(), hi.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(cudaMemcpy(e.data(), hi2.data(), d * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
      REQUIRE(a == b);
      REQUIRE(c == e);
    } else {
      REQUIRE(d % 16 != 0);  // the fused path covers every width a multiple of 16
    }
  }
}
