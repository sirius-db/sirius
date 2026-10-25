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
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

#include <catch.hpp>
#include <cuda_fp16.h>
#include <vss/bound_gemm.hpp>
#include <vss/device_rates.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace {

using sirius::vss::bound_candidates;
using sirius::vss::bound_slice;

rmm::cuda_stream_view stream() { return cudf::get_default_stream(); }

template <class T>
rmm::device_uvector<T> to_device(std::vector<T> const& h)
{
  rmm::device_uvector<T> d(h.size(), stream());
  if (!h.empty()) {
    REQUIRE(cudaMemcpyAsync(
              d.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice, stream().value()) ==
            cudaSuccess);
  }
  return d;
}

template <class T>
std::vector<T> to_host(rmm::device_uvector<T> const& d, std::size_t n)
{
  std::vector<T> h(n);
  if (n > 0) {
    REQUIRE(cudaMemcpyAsync(
              h.data(), d.data(), n * sizeof(T), cudaMemcpyDeviceToHost, stream().value()) ==
            cudaSuccess);
  }
  stream().synchronize();
  return h;
}

using pair_key = std::pair<int32_t, int64_t>;

/// The appended triples, keyed by (probe row, id); a pair appended twice fails.
std::map<pair_key, float> read_out(bound_candidates const& out)
{
  auto const count = to_host(out.count, 1)[0];
  REQUIRE(count <= static_cast<unsigned long long>(out.capacity()));
  auto const rows = to_host(out.rows, count);
  auto const ids  = to_host(out.ids, count);
  auto const d    = to_host(out.distances, count);
  std::map<pair_key, float> got;
  for (std::size_t i = 0; i < count; ++i) {
    REQUIRE(got.emplace(pair_key{rows[i], ids[i]}, d[i]).second);
  }
  return got;
}

/// One side of a search: rows of shifted int8 or half, with the norms the kernel takes.
struct side {
  int64_t rows{0};
  std::vector<int8_t> i8;
  std::vector<uint16_t> f16;
  std::vector<double> value;  // the components as the kernel sees them
  std::vector<int32_t> sq_i8;
  std::vector<float> sq_f16;
};

side make_side(int64_t rows, int64_t d, bool f16, std::mt19937& gen)
{
  side s;
  s.rows = rows;
  s.value.resize(rows * d);
  std::uniform_int_distribution<int> byte(-128, 127);
  std::uniform_real_distribution<float> unit(-1.f, 1.f);
  for (int64_t i = 0; i < rows * d; ++i) {
    if (f16) {
      auto const h = __float2half(unit(gen));
      uint16_t bits;
      std::memcpy(&bits, &h, sizeof(bits));
      s.f16.push_back(bits);
      s.value[i] = __half2float(h);
    } else {
      s.i8.push_back(static_cast<int8_t>(byte(gen)));
      s.value[i] = s.i8.back();
    }
  }
  for (int64_t r = 0; r < rows; ++r) {
    double sq = 0;
    for (int64_t c = 0; c < d; ++c) {
      sq += s.value[r * d + c] * s.value[r * d + c];
    }
    if (f16) {
      s.sq_f16.push_back(static_cast<float>(sq));
    } else {
      s.sq_i8.push_back(static_cast<int32_t>(sq));
    }
  }
  return s;
}

double distance(side const& p, int64_t pr, side const& x, int64_t xr, int64_t d)
{
  double s = 0;
  for (int64_t c = 0; c < d; ++c) {
    auto const diff = p.value[pr * d + c] - x.value[xr * d + c];
    s += diff * diff;
  }
  return s;
}

/// Each probe row's bound: about @p share of its pairs with the corpus fall under it. Odd rows
/// get a fractional bound, which an integer distance must not round past.
std::vector<float> make_bounds(side const& p, side const& x, int64_t d, double share)
{
  std::vector<float> bound(p.rows);
  std::vector<double> ds(x.rows);
  for (int64_t r = 0; r < p.rows; ++r) {
    for (int64_t j = 0; j < x.rows; ++j) {
      ds[j] = distance(p, r, x, j, d);
    }
    auto const at = static_cast<std::size_t>(share * (x.rows - 1));
    std::nth_element(ds.begin(), ds.begin() + at, ds.end());
    bound[r] = static_cast<float>(ds[at]) + (r % 2 == 1 ? 0.5f : 0.f);
  }
  return bound;
}

/// What the search must return: every pair under its row's bound. For int8 the distances are
/// exact integers; for half they are FP32 sums, so pairs within @p tol of the bound may go
/// either way and distances may differ by @p tol.
void check_pairs(std::map<pair_key, float> const& got,
                 side const& p,
                 std::vector<int64_t> const& rows,
                 side const& x,
                 int64_t x_first,
                 int64_t n,
                 int64_t d,
                 std::vector<float> const& bound,
                 bool f16,
                 std::function<int64_t(int64_t)> const& id_of,
                 std::map<pair_key, float>& expected_seen)
{
  for (auto const r : rows) {
    for (int64_t j = 0; j < n; ++j) {
      auto const dist = distance(p, r, x, x_first + j, d);
      auto const tol  = f16 ? 1e-5 * (p.sq_f16[r] + x.sq_f16[x_first + j]) + 1e-3 : 0.0;
      pair_key const key{static_cast<int32_t>(r), id_of(j)};
      auto const it = got.find(key);
      if (dist <= bound[r] - tol) {
        INFO("row " << r << " id " << key.second << " distance " << dist << " bound " << bound[r]);
        REQUIRE(it != got.end());
      } else if (dist > bound[r] + tol) {
        INFO("row " << r << " id " << key.second << " distance " << dist << " bound " << bound[r]);
        REQUIRE(it == got.end());
      }
      if (it != got.end()) {
        REQUIRE(std::abs(it->second - dist) <= tol);
        expected_seen.emplace(key, it->second);
      }
    }
  }
}

void check_single(int64_t m, int64_t n, int64_t d, bool f16, bool with_map, uint32_t seed)
{
  INFO("m " << m << " n " << n << " d " << d << (f16 ? " half" : " int8")
            << (with_map ? " id_map" : ""));
  std::mt19937 gen(seed);
  // The probe side has more rows than the search reads, in a shuffled order.
  auto const p = make_side(m + 7, d, f16, gen);
  auto const x = make_side(n, d, f16, gen);
  std::vector<int64_t> all(p.rows);
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), gen);
  std::vector<int64_t> rows(all.begin(), all.begin() + m);
  auto const bound      = make_bounds(p, x, d, 0.02);
  int64_t const id_base = 1000;
  std::vector<int64_t> map(n);
  for (int64_t j = 0; j < n; ++j) {
    map[j] = 3 * j + 5;
  }
  auto const id_of = [&](int64_t j) { return with_map ? map[j] : id_base + j; };

  auto const mr = cudf::get_current_device_resource_ref();
  bound_candidates out(m * n / 8 + 1024, stream(), mr);
  REQUIRE(cudaMemsetAsync(out.count.data(), 0, sizeof(unsigned long long), stream().value()) ==
          cudaSuccess);
  auto const d_rows  = to_device(rows);
  auto const d_bound = to_device(bound);
  auto const d_map   = to_device(map);
  if (f16) {
    auto const dx = to_device(x.f16), dp = to_device(p.f16);
    auto const dxs = to_device(x.sq_f16), dps = to_device(p.sq_f16);
    sirius::vss::bound_filter_f16(dx.data(),
                                  dxs.data(),
                                  n,
                                  id_base,
                                  dp.data(),
                                  dps.data(),
                                  d_rows.data(),
                                  m,
                                  d,
                                  d_bound.data(),
                                  sirius::vss::float16_slack{},
                                  out,
                                  stream());
  } else {
    auto const dx = to_device(x.i8), dp = to_device(p.i8);
    auto const dxs = to_device(x.sq_i8), dps = to_device(p.sq_i8);
    sirius::vss::bound_filter_int8(dx.data(),
                                   dxs.data(),
                                   n,
                                   id_base,
                                   with_map ? d_map.data() : nullptr,
                                   dp.data(),
                                   dps.data(),
                                   d_rows.data(),
                                   m,
                                   d,
                                   d_bound.data(),
                                   out,
                                   stream());
  }
  auto const got = read_out(out);
  std::map<pair_key, float> seen;
  check_pairs(got, p, rows, x, 0, n, d, bound, f16, id_of, seen);
  REQUIRE(seen.size() == got.size());
}

}  // namespace

TEST_CASE("the bounded int8 search returns exactly the pairs under each row's bound", "[vss]")
{
  // Probe and corpus counts on and off the 128-row tile, widths through one to four 64-byte
  // staging steps and one ending half way through a 32-byte step.
  for (int64_t const d : {16, 48, 80, 96, 128, 256}) {
    check_single(300, 1500, d, false, false, 11 + static_cast<uint32_t>(d));
  }
  check_single(128, 1024, 96, false, true, 7);
  check_single(17, 129, 128, false, false, 8);
}

TEST_CASE("the bounded half search returns the pairs under each row's bound", "[vss]")
{
  for (int64_t const d : {16, 48, 96, 128, 256}) {
    check_single(300, 1500, d, true, false, 21 + static_cast<uint32_t>(d));
  }
  check_single(17, 129, 960, true, false, 9);
}

TEST_CASE("a grouped bounded search answers each slice as a search of its own", "[vss]")
{
  for (bool const f16 : {false, true}) {
    INFO((f16 ? "half" : "int8"));
    int64_t const d = 96;
    std::mt19937 gen(f16 ? 31 : 32);
    auto const p     = make_side(400, d, f16, gen);
    auto const x     = make_side(2000, d, f16, gen);
    auto const bound = make_bounds(p, x, d, 0.02);
    // Slices of the one corpus: (first row, rows, probe rows); one of each kernel shape.
    struct cut {
      int64_t first, n;
      std::vector<int64_t> rows;
    };
    std::vector<cut> cuts;
    std::vector<int64_t> all(p.rows);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), gen);
    cuts.push_back({0, 700, {all.begin(), all.begin() + 300}});
    cuts.push_back({700, 129, {all.begin() + 50, all.begin() + 53}});
    cuts.push_back({829, 1171, {all.begin() + 100, all.begin() + 230}});
    auto const mr      = cudf::get_current_device_resource_ref();
    auto const dx_i8   = to_device(x.i8);
    auto const dp_i8   = to_device(p.i8);
    auto const dx_f16  = to_device(x.f16);
    auto const dp_f16  = to_device(p.f16);
    auto const dxs_i8  = to_device(x.sq_i8);
    auto const dps_i8  = to_device(p.sq_i8);
    auto const dxs_f   = to_device(x.sq_f16);
    auto const dps_f   = to_device(p.sq_f16);
    auto const d_bound = to_device(bound);
    std::vector<rmm::device_uvector<int64_t>> d_rows;
    std::vector<bound_slice> slices;
    for (auto const& c : cuts) {
      d_rows.push_back(to_device(c.rows));
      void const* xs  = f16 ? static_cast<void const*>(dx_f16.data() + c.first * d)
                            : static_cast<void const*>(dx_i8.data() + c.first * d);
      void const* xsq = f16 ? static_cast<void const*>(dxs_f.data() + c.first)
                            : static_cast<void const*>(dxs_i8.data() + c.first);
      slices.push_back(bound_slice{xs,
                                   xsq,
                                   c.n,
                                   10 * c.first,
                                   nullptr,
                                   d_rows.back().data(),
                                   static_cast<int64_t>(c.rows.size())});
    }
    bound_candidates out(200000, stream(), mr);
    REQUIRE(cudaMemsetAsync(out.count.data(), 0, sizeof(unsigned long long), stream().value()) ==
            cudaSuccess);
    sirius::vss::bound_filter_group(
      slices,
      f16,
      f16 ? static_cast<void const*>(dp_f16.data()) : static_cast<void const*>(dp_i8.data()),
      f16 ? static_cast<void const*>(dps_f.data()) : static_cast<void const*>(dps_i8.data()),
      d,
      d_bound.data(),
      out,
      stream(),
      mr);
    auto const got = read_out(out);
    std::map<pair_key, float> seen;
    for (auto const& c : cuts) {
      auto const first = c.first;
      check_pairs(
        got,
        p,
        c.rows,
        x,
        first,
        c.n,
        d,
        bound,
        f16,
        [&](int64_t j) { return 10 * first + j; },
        seen);
    }
    REQUIRE(seen.size() == got.size());
  }
}

// Not run by default: the kernel's rate at join shapes (SIRIUS_VSS_BENCH_N corpus rows, default
// 1M, against 10k probe rows), against this device's measured tensor peaks.
TEST_CASE("bounded search throughput", "[.][vss_bench]")
{
  sirius::vss::device_rates peak;
  REQUIRE(sirius::vss::measure_device_rates(peak));
  auto const* env = std::getenv("SIRIUS_VSS_BENCH_N");
  int64_t const n = env != nullptr ? std::atoll(env) : 1'000'000;
  int64_t const m = 10'000;
  auto const mr   = cudf::get_current_device_resource_ref();
  std::mt19937 gen(5);
  for (bool const f16 : {false, true}) {
    for (int64_t const d : {96, 128, 256, 768, 960}) {
      if (!f16 && d > 256) { continue; }
      // Random rows straight on the device; a bound well under the typical distance lets a few
      // pairs in 10^4 through, about what a join's running bound does.
      std::vector<int64_t> rows(m);
      std::iota(rows.begin(), rows.end(), 0);
      auto const d_rows = to_device(rows);
      double typical;
      rmm::device_uvector<int8_t> x8(f16 ? 0 : n * d, stream()), p8(f16 ? 0 : m * d, stream());
      rmm::device_uvector<uint16_t> xh(f16 ? n * d : 0, stream()), ph(f16 ? m * d : 0, stream());
      std::vector<int32_t> sq8;
      std::vector<float> sqh;
      if (f16) {
        std::vector<uint16_t> h(n * d);
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        for (auto& e : h) {
          auto const v = __float2half(unit(gen));
          std::memcpy(&e, &v, sizeof(e));
        }
        REQUIRE(cudaMemcpy(xh.data(), h.data(), n * d * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(ph.data(), h.data(), m * d * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        sqh.assign(n, d / 3.f);
        typical = 2 * d / 3.0;
      } else {
        std::vector<int8_t> h(n * d);
        std::uniform_int_distribution<int> byte(-128, 127);
        for (auto& e : h) {
          e = static_cast<int8_t>(byte(gen));
        }
        REQUIRE(cudaMemcpy(x8.data(), h.data(), n * d, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(p8.data(), h.data(), m * d, cudaMemcpyHostToDevice) == cudaSuccess);
        sq8.assign(n, static_cast<int32_t>(d * 5461));
        typical = 2.0 * d * 5461;
      }
      auto const d_sq8 = to_device(sq8);
      auto const d_sqh = to_device(sqh);
      auto const bound = std::vector<float>(m, static_cast<float>(0.6 * typical));
      auto const d_bnd = to_device(bound);
      bound_candidates out(64 << 20, stream(), mr);
      auto run = [&] {
        REQUIRE(
          cudaMemsetAsync(out.count.data(), 0, sizeof(unsigned long long), stream().value()) ==
          cudaSuccess);
        if (f16) {
          sirius::vss::bound_filter_f16(xh.data(),
                                        d_sqh.data(),
                                        n,
                                        0,
                                        ph.data(),
                                        d_sqh.data(),
                                        d_rows.data(),
                                        m,
                                        d,
                                        d_bnd.data(),
                                        sirius::vss::float16_slack{},
                                        out,
                                        stream());
        } else {
          sirius::vss::bound_filter_int8(x8.data(),
                                         d_sq8.data(),
                                         n,
                                         0,
                                         nullptr,
                                         p8.data(),
                                         d_sq8.data(),
                                         d_rows.data(),
                                         m,
                                         d,
                                         d_bnd.data(),
                                         out,
                                         stream());
        }
      };
      run();
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      int const reps = 5;
      cudaEventRecord(a, stream().value());
      for (int r = 0; r < reps; ++r) {
        run();
      }
      cudaEventRecord(b, stream().value());
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      cudaEventDestroy(a);
      cudaEventDestroy(b);
      auto const s     = ms / 1e3 / reps;
      auto const rate  = 2.0 * m * n * d / s;
      auto const count = to_host(out.count, 1)[0];
      WARN((f16 ? "half" : "int8")
           << " d " << d << " m " << m << " n " << n << ": " << s * 1e3 << " ms, " << rate / 1e12
           << " Tops/s = " << 100 * rate / (f16 ? peak.f16 : peak.int8) << " % of peak, " << count
           << " pairs kept");
    }
  }
}

// Not run by default: the grouped launch at an approximate join's shape, many lists of a few
// thousand rows, each searched by the few hundred probe rows routed to it.
TEST_CASE("grouped bounded search throughput", "[.][vss_bench]")
{
  sirius::vss::device_rates peak;
  REQUIRE(sirius::vss::measure_device_rates(peak));
  int64_t const lists = 2048, list_rows = 2000, routed = 800, probes = 100'000;
  auto const mr = cudf::get_current_device_resource_ref();
  std::mt19937 gen(6);
  std::vector<int64_t> rows(lists * routed);
  std::uniform_int_distribution<int64_t> pick(0, probes - 1);
  for (auto& r : rows) {
    r = pick(gen);
  }
  auto const d_rows = to_device(rows);
  for (bool const f16 : {false, true}) {
    for (int64_t const d : {96, 128, 768}) {
      if (!f16 && d > 256) { continue; }
      int64_t const n = lists * list_rows;
      rmm::device_uvector<int8_t> x8(f16 ? 0 : n * d, stream()), p8(f16 ? 0 : probes * d, stream());
      rmm::device_uvector<uint16_t> xh(f16 ? n * d : 0, stream()),
        ph(f16 ? probes * d : 0, stream());
      double typical;
      if (f16) {
        std::vector<uint16_t> h(n * d);
        std::uniform_real_distribution<float> unit(-1.f, 1.f);
        for (auto& e : h) {
          auto const v = __float2half(unit(gen));
          std::memcpy(&e, &v, sizeof(e));
        }
        REQUIRE(cudaMemcpy(xh.data(), h.data(), n * d * 2, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(ph.data(), h.data(), probes * d * 2, cudaMemcpyHostToDevice) ==
                cudaSuccess);
        typical = 2 * d / 3.0;
      } else {
        std::vector<int8_t> h(n * d);
        std::uniform_int_distribution<int> byte(-128, 127);
        for (auto& e : h) {
          e = static_cast<int8_t>(byte(gen));
        }
        REQUIRE(cudaMemcpy(x8.data(), h.data(), n * d, cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(p8.data(), h.data(), probes * d, cudaMemcpyHostToDevice) == cudaSuccess);
        typical = 2.0 * d * 5461;
      }
      auto const d_sq8 =
        to_device(std::vector<int32_t>(f16 ? 0 : n, static_cast<int32_t>(d * 5461)));
      auto const d_sqh = to_device(std::vector<float>(f16 ? n : 0, d / 3.f));
      auto const d_bnd = to_device(std::vector<float>(probes, static_cast<float>(0.6 * typical)));
      std::vector<bound_slice> slices;
      for (int64_t l = 0; l < lists; ++l) {
        void const* xs  = f16 ? static_cast<void const*>(xh.data() + l * list_rows * d)
                              : static_cast<void const*>(x8.data() + l * list_rows * d);
        void const* xsq = f16 ? static_cast<void const*>(d_sqh.data() + l * list_rows)
                              : static_cast<void const*>(d_sq8.data() + l * list_rows);
        slices.push_back(bound_slice{
          xs, xsq, list_rows, l * list_rows, nullptr, d_rows.data() + l * routed, routed});
      }
      bound_candidates out(64 << 20, stream(), mr);
      auto run = [&] {
        REQUIRE(
          cudaMemsetAsync(out.count.data(), 0, sizeof(unsigned long long), stream().value()) ==
          cudaSuccess);
        sirius::vss::bound_filter_group(
          slices,
          f16,
          f16 ? static_cast<void const*>(ph.data()) : static_cast<void const*>(p8.data()),
          f16 ? static_cast<void const*>(d_sqh.data()) : static_cast<void const*>(d_sq8.data()),
          d,
          d_bnd.data(),
          out,
          stream(),
          mr);
      };
      run();
      cudaEvent_t a, b;
      cudaEventCreate(&a);
      cudaEventCreate(&b);
      int const reps = 5;
      cudaEventRecord(a, stream().value());
      for (int r = 0; r < reps; ++r) {
        run();
      }
      cudaEventRecord(b, stream().value());
      cudaEventSynchronize(b);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      cudaEventDestroy(a);
      cudaEventDestroy(b);
      auto const s    = ms / 1e3 / reps;
      auto const rate = 2.0 * lists * list_rows * routed * d / s;
      WARN("grouped " << (f16 ? "half" : "int8") << " d " << d << ", " << lists << " lists of "
                      << list_rows << " rows x " << routed << " probe rows: " << s * 1e3 << " ms, "
                      << rate / 1e12 << " Tops/s = " << 100 * rate / (f16 ? peak.f16 : peak.int8)
                      << " % of peak");
    }
  }
}
