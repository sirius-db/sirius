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

//===----------------------------------------------------------------------===//
// Expression evaluator microbenchmark across all four execution strategies.
//
// Tagged `[!benchmark]` so default unittest runs skip it. Run with
//   sirius_unittest "[expr_bench]"
//
// Environment:
//   SIRIUS_EXPR_BENCH_ROWS        comma list of row counts (default "1048576,16777216")
//   SIRIUS_EXPR_BENCH_STRATEGIES  comma list of strategies (default all four)
//   SIRIUS_EXPR_BENCH_CASES       comma list of case-name substrings to run (default all)
//   SIRIUS_EXPR_BENCH_ITERS       timed iterations per case (default 20)
//   SIRIUS_EXPR_BENCH_WARMUP      minimum warmup iterations after the cold call (default 3)
//   SIRIUS_EXPR_BENCH_WARMUP_MS   minimum warmup wall time in ms (default 300)
//   SIRIUS_EXPR_BENCH_VERIFY      1 (default) compares every strategy's result checksum against
//                                 the first listed strategy (`ref`; `noref` when it failed);
//                                 0 disables the host readback
//   SIRIUS_EXPR_BENCH_CSV         append one CSV row per measurement to this file
//
// Each measurement reports the cold first call, then the median and minimum wall time of a
// steady-state call. For NVRTC_JIT the cold call is a full compile: the in-memory kernel cache
// is cleared before it and the disk cache is disabled. For AST_JIT only the first use of each
// distinct cuDF kernel in the process compiles (cuDF's in-memory cache cannot be cleared; its
// disk cache is pointed at a fresh temporary directory). Wall time includes the host-side
// synchronizations the evaluators perform internally plus one stream sync.
//===----------------------------------------------------------------------===//

#include "codegen/jit/kernel_cache.hpp"
#include "expression_evaluator/ast_test_support.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <catch.hpp>
#include <cuda_profiler_api.h>
#include <expression/ast/node.hpp>
#include <expression/ast/utils.hpp>
#include <expression/join_condition.hpp>
#include <expression/value.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/expression_evaluator_strategy.hpp>
#include <helper/logical_type.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace sirius::expr_test;
using sirius::logical_type;
using sirius::type_id;
using strategy_t = sirius::expression_evaluator_strategy;

namespace {

//===----------------------------------------------------------------------===//
// Environment helpers
//===----------------------------------------------------------------------===//

std::vector<std::string> split_csv(std::string const& s)
{
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) { out.push_back(item); }
  }
  return out;
}

std::string env_or(char const* name, std::string const& dflt)
{
  char const* v = std::getenv(name);
  return (v != nullptr && *v != '\0') ? std::string(v) : dflt;
}

//===----------------------------------------------------------------------===//
// Synthetic input table
//
//   0  INT32   non-null        uniform [0, 1e6)
//   1  INT64   10% nulls       uniform [0, 1e9)
//   2  DOUBLE  non-null        uniform [1, 1000)
//   3  DEC(15,2) non-null      price   [900.00, 105000.00]
//   4  DEC(15,2) non-null      discount [0.00, 0.10]
//   5  DEC(15,2) non-null      tax      [0.00, 0.08]
//   6  DATE    non-null        days in [1992-01-01, 1998-12-31]
//   7  STRING  non-null        7-word ship-mode vocabulary
//   8  STRING  non-null        random lowercase 10..40 chars, 5% contain "special"
//   9  INT32   50% nulls       uniform [0, 1e6)
//   10 INT64   non-null        quantity [1, 50]
//===----------------------------------------------------------------------===//

struct bench_input {
  std::vector<std::unique_ptr<cudf::column>> columns;
  cudf::table_view view;
  std::size_t total_bytes{0};
};

rmm::device_buffer upload_mask(std::vector<bool> const& valids,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr,
                               cudf::size_type& null_count)
{
  auto const n = static_cast<cudf::size_type>(valids.size());
  // cuDF requires null masks padded to a 64-byte multiple, not just to whole words.
  auto const bytes = cudf::bitmask_allocation_size_bytes(n);
  std::vector<cudf::bitmask_type> host(bytes / sizeof(cudf::bitmask_type), 0);
  null_count = 0;
  for (cudf::size_type i = 0; i < n; ++i) {
    if (valids[i]) {
      host[i / 32] |= (1u << (i % 32));
    } else {
      ++null_count;
    }
  }
  return rmm::device_buffer(host.data(), bytes, stream, mr);
}

template <typename T>
std::unique_ptr<cudf::column> upload_fixed(cudf::data_type type,
                                           std::vector<T> const& values,
                                           std::vector<bool> const* valids,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<cudf::size_type>(values.size());
  rmm::device_buffer data(values.data(), sizeof(T) * values.size(), stream, mr);
  rmm::device_buffer mask(0, stream, mr);
  cudf::size_type null_count = 0;
  if (valids != nullptr) { mask = upload_mask(*valids, stream, mr, null_count); }
  return std::make_unique<cudf::column>(type, n, std::move(data), std::move(mask), null_count);
}

std::unique_ptr<cudf::column> upload_strings(std::vector<std::string> const& strs,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const n = static_cast<cudf::size_type>(strs.size());
  std::vector<cudf::size_type> offsets(n + 1, 0);
  std::size_t total = 0;
  for (cudf::size_type i = 0; i < n; ++i) {
    offsets[i] = static_cast<cudf::size_type>(total);
    total += strs[i].size();
  }
  offsets[n] = static_cast<cudf::size_type>(total);
  std::string chars;
  chars.reserve(total);
  for (auto const& s : strs) {
    chars += s;
  }
  auto offsets_col = cudf::make_numeric_column(
    cudf::data_type{cudf::type_id::INT32}, n + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpyAsync(offsets_col->mutable_view().data<cudf::size_type>(),
                  offsets.data(),
                  offsets.size() * sizeof(cudf::size_type),
                  cudaMemcpyHostToDevice,
                  stream.value());
  rmm::device_buffer chars_buf(chars.data(), chars.size(), stream, mr);
  // Pageable sources may be read asynchronously (HMM/ATS); the local buffers must outlive the copy.
  stream.synchronize();
  return cudf::make_strings_column(
    n, std::move(offsets_col), std::move(chars_buf), 0, rmm::device_buffer{0, stream, mr});
}

bench_input make_bench_input(cudf::size_type rows,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  std::mt19937_64 rng(0x5EED5EEDULL);
  std::uniform_int_distribution<std::int32_t> d_int(0, 999'999);
  std::uniform_int_distribution<std::int64_t> d_big(0, 999'999'999LL);
  std::uniform_real_distribution<double> d_dbl(1.0, 1000.0);
  std::uniform_int_distribution<std::int64_t> d_price(90'000, 10'500'000);
  std::uniform_int_distribution<std::int64_t> d_disc(0, 10);
  std::uniform_int_distribution<std::int64_t> d_tax(0, 8);
  std::uniform_int_distribution<std::int32_t> d_date(8036, 10591);  // 1992-01-01 .. 1998-12-31
  std::uniform_int_distribution<int> d_mode(0, 6);
  std::uniform_int_distribution<int> d_len(10, 40);
  std::uniform_int_distribution<int> d_alpha(0, 25);
  std::uniform_int_distribution<int> d_pct(0, 99);
  std::uniform_int_distribution<std::int64_t> d_qty(1, 50);

  static constexpr std::array<char const*, 7> modes{
    "MAIL", "SHIP", "RAIL", "TRUCK", "AIR", "FOB", "REG AIR"};

  std::vector<std::int32_t> c0(rows), c9(rows);
  std::vector<std::int64_t> c1(rows), c3(rows), c4(rows), c5(rows), c10(rows);
  std::vector<double> c2(rows);
  std::vector<std::int32_t> c6(rows);
  std::vector<std::string> c7(rows), c8(rows);
  std::vector<bool> v1(rows), v9(rows);

  for (cudf::size_type i = 0; i < rows; ++i) {
    c0[i]         = d_int(rng);
    c1[i]         = d_big(rng);
    v1[i]         = d_pct(rng) >= 10;
    c2[i]         = d_dbl(rng);
    c3[i]         = d_price(rng);
    c4[i]         = d_disc(rng);
    c5[i]         = d_tax(rng);
    c6[i]         = d_date(rng);
    c7[i]         = modes[static_cast<std::size_t>(d_mode(rng))];
    c9[i]         = d_int(rng);
    v9[i]         = d_pct(rng) >= 50;
    c10[i]        = d_qty(rng);
    int const len = d_len(rng);
    std::string s;
    s.reserve(static_cast<std::size_t>(len));
    for (int k = 0; k < len; ++k) {
      s.push_back(static_cast<char>('a' + d_alpha(rng)));
    }
    if (d_pct(rng) < 5) { s.replace(static_cast<std::size_t>(len / 2) - 3, 7, "special"); }
    c8[i] = std::move(s);
  }

  bench_input in;
  auto const dec_type = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  in.columns.push_back(
    upload_fixed(cudf::data_type{cudf::type_id::INT32}, c0, nullptr, stream, mr));
  in.columns.push_back(upload_fixed(cudf::data_type{cudf::type_id::INT64}, c1, &v1, stream, mr));
  in.columns.push_back(
    upload_fixed(cudf::data_type{cudf::type_id::FLOAT64}, c2, nullptr, stream, mr));
  in.columns.push_back(upload_fixed(dec_type, c3, nullptr, stream, mr));
  in.columns.push_back(upload_fixed(dec_type, c4, nullptr, stream, mr));
  in.columns.push_back(upload_fixed(dec_type, c5, nullptr, stream, mr));
  in.columns.push_back(
    upload_fixed(cudf::data_type{cudf::type_id::TIMESTAMP_DAYS}, c6, nullptr, stream, mr));
  in.columns.push_back(upload_strings(c7, stream, mr));
  in.columns.push_back(upload_strings(c8, stream, mr));
  in.columns.push_back(upload_fixed(cudf::data_type{cudf::type_id::INT32}, c9, &v9, stream, mr));
  in.columns.push_back(
    upload_fixed(cudf::data_type{cudf::type_id::INT64}, c10, nullptr, stream, mr));
  stream.synchronize();

  std::vector<cudf::column_view> views;
  for (auto const& c : in.columns) {
    views.push_back(c->view());
  }
  in.view = cudf::table_view(views);
  // The host vectors above die with this frame; wait for every upload to finish reading them.
  stream.synchronize();
  return in;
}

std::size_t column_bytes(cudf::column_view const& col, rmm::cuda_stream_view stream)
{
  std::size_t bytes = 0;
  if (col.type().id() == cudf::type_id::STRING) {
    cudf::strings_column_view const scv(col);
    bytes += static_cast<std::size_t>(scv.chars_size(stream));
    bytes += static_cast<std::size_t>(col.size() + 1) * sizeof(cudf::size_type);
  } else {
    bytes += static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
  }
  if (col.nullable()) { bytes += static_cast<std::size_t>(col.size()) / 8; }
  return bytes;
}

//===----------------------------------------------------------------------===//
// Expression builders
//===----------------------------------------------------------------------===//

using node_ptr = std::unique_ptr<ast_node>;

logical_type const t_int    = logical_type::make(type_id::INTEGER);
logical_type const t_big    = logical_type::make(type_id::BIGINT);
logical_type const t_dbl    = logical_type::make(type_id::DOUBLE);
logical_type const t_bool   = logical_type::make(type_id::BOOLEAN);
logical_type const t_date   = logical_type::make(type_id::DATE);
logical_type const t_str    = logical_type::make(type_id::VARCHAR);
logical_type const t_dec152 = logical_type::make_decimal(15, 2);
logical_type const t_dec162 = logical_type::make_decimal(16, 2);
logical_type const t_dec314 = logical_type::make_decimal(31, 4);
logical_type const t_dec386 = logical_type::make_decimal(38, 6);

node_ptr ref_int32() { return make_ref_typed(0, t_int); }
node_ptr ref_int64_nullable() { return make_ref_typed(1, t_big); }
node_ptr ref_double() { return make_ref_typed(2, t_dbl); }
node_ptr ref_price() { return make_ref_typed(3, t_dec152); }
node_ptr ref_discount() { return make_ref_typed(4, t_dec152); }
node_ptr ref_tax() { return make_ref_typed(5, t_dec152); }
node_ptr ref_date() { return make_ref_typed(6, t_date); }
node_ptr ref_mode() { return make_ref_typed(7, t_str); }
node_ptr ref_comment() { return make_ref_typed(8, t_str); }
node_ptr ref_int32_nullable() { return make_ref_typed(9, t_int); }
node_ptr ref_qty() { return make_ref_typed(10, t_big); }

node_ptr make_date_const(std::int32_t days)
{
  return std::make_unique<ast_node>(
    sirius::ast::constant{sirius::value{sirius::date_value{days}}, t_date});
}

node_ptr bin(sirius::function_id fid, node_ptr l, node_ptr r, logical_type ret)
{
  std::vector<node_ptr> args;
  args.push_back(std::move(l));
  args.push_back(std::move(r));
  return make_func(fid, std::move(args), ret);
}

node_ptr fn1(sirius::function_id fid, node_ptr a, logical_type ret)
{
  std::vector<node_ptr> args;
  args.push_back(std::move(a));
  return make_func(fid, std::move(args), ret);
}

node_ptr cmp(sirius::comparison_type op, node_ptr l, node_ptr r)
{
  return make_cmp(op, std::move(l), std::move(r));
}

node_ptr conj_and(std::vector<node_ptr> children)
{
  return make_conj(sirius::ast::conjunction::kind::op_and, std::move(children));
}

template <typename... Ns>
std::vector<node_ptr> list(Ns&&... ns)
{
  std::vector<node_ptr> v;
  (v.push_back(std::forward<Ns>(ns)), ...);
  return v;
}

// disc_price = price * (1.00 - discount)  -> DECIMAL(31,4), as DuckDB types it for TPC-H Q1.
node_ptr disc_price()
{
  return bin(sirius::function_id::mul,
             ref_price(),
             bin(sirius::function_id::sub, make_dec64_const(100, 15, 2), ref_discount(), t_dec162),
             t_dec314);
}

enum class case_kind { project, filter_one_col, filter_all_cols };

struct bench_case {
  std::string name;
  case_kind kind;
  std::function<std::vector<node_ptr>()> build;
};

std::vector<bench_case> all_cases()
{
  using fid = sirius::function_id;
  using ct  = sirius::comparison_type;
  std::vector<bench_case> cases;
  auto project = [&](std::string name, std::function<std::vector<node_ptr>()> b) {
    cases.push_back({std::move(name), case_kind::project, std::move(b)});
  };
  auto filter = [&](std::string name, std::function<node_ptr()> b, bool all_cols = false) {
    cases.push_back({std::move(name),
                     all_cols ? case_kind::filter_all_cols : case_kind::filter_one_col,
                     [b = std::move(b)]() { return list(b()); }});
  };

  project("P01_int_add_const",
          [] { return list(bin(fid::add, ref_int32(), make_int_const(1), t_int)); });
  project("P02_int_mul_add_nullable", [] {
    return list(
      bin(fid::add, bin(fid::mul, ref_int32(), ref_qty(), t_big), ref_int64_nullable(), t_big));
  });
  project("P03_cmp_and", [] {
    return list(conj_and(list(cmp(ct::gt, ref_int32(), make_int_const(500'000)),
                              cmp(ct::lt, ref_qty(), make_bigint_const(25)))));
  });
  project("P04_dec_disc_price", [] { return list(disc_price()); });
  project("P05_dec_charge", [] {
    return list(bin(fid::mul,
                    disc_price(),
                    bin(fid::add, make_dec64_const(100, 15, 2), ref_tax(), t_dec162),
                    t_dec386));
  });
  project("P06_case_when", [] {
    std::vector<sirius::ast::case_expr::when_then> whens;
    whens.push_back({cmp(ct::gt, ref_int32(), make_int_const(500'000)), ref_qty()});
    return list(std::make_unique<ast_node>(
      sirius::ast::case_expr{std::move(whens), ref_int64_nullable(), t_big}));
  });
  project("P07_cast_div_double", [] {
    return list(bin(fid::div, make_cast(ref_int32(), t_dbl, false), ref_double(), t_dbl));
  });
  project("P08_str_eq_const",
          [] { return list(cmp(ct::equal, ref_mode(), make_str_const("MAIL"))); });
  project("P09_str_like_contains",
          [] { return list(bin(fid::like, ref_comment(), make_str_const("%special%"), t_bool)); });
  project("P10_str_in_list", [] {
    return list(make_in(ref_mode(), list(make_str_const("MAIL"), make_str_const("SHIP")), false));
  });
  project("P11_strlen", [] { return list(fn1(fid::strlen, ref_comment(), t_big)); });
  project("P12_date_year", [] { return list(fn1(fid::year, ref_date(), t_big)); });
  project("P13_multi_out6", [] {
    return list(bin(fid::add, ref_int32(), make_int_const(1), t_int),
                bin(fid::mul, ref_int32(), make_int_const(2), t_int),
                bin(fid::sub, ref_int64_nullable(), make_cast(ref_int32(), t_big, false), t_big),
                disc_price(),
                fn1(fid::year, ref_date(), t_big),
                cmp(ct::gt, make_cast(ref_int32(), t_big, false), ref_qty()));
  });
  project("P14_arith_chain10", [] {
    node_ptr e = make_cast(ref_int32(), t_big, false);
    e          = bin(fid::add, std::move(e), make_bigint_const(1), t_big);
    e          = bin(fid::mul, std::move(e), make_bigint_const(3), t_big);
    e          = bin(fid::sub, std::move(e), ref_qty(), t_big);
    e          = bin(fid::mul, std::move(e), make_bigint_const(2), t_big);
    e          = bin(fid::add, std::move(e), make_bigint_const(7), t_big);
    e          = bin(fid::mul, std::move(e), make_bigint_const(5), t_big);
    e          = bin(fid::sub, std::move(e), make_bigint_const(11), t_big);
    e          = bin(fid::add, std::move(e), ref_qty(), t_big);
    e          = bin(fid::mul, std::move(e), make_bigint_const(13), t_big);
    e          = bin(fid::add, std::move(e), make_bigint_const(17), t_big);
    return list(std::move(e));
  });
  project("P15_coalesce", [] {
    return list(make_coalesce(
      list(
        make_cast(ref_int32_nullable(), t_big, false), ref_int64_nullable(), make_bigint_const(0)),
      t_big));
  });
  project("P16_try_cast_narrow",
          [] { return list(make_cast(ref_int64_nullable(), t_int, /*try_cast=*/true)); });
  project("P17_nullable_add",
          [] { return list(bin(fid::add, ref_int32_nullable(), make_int_const(1), t_int)); });
  project("P18_int_mul_int64", [] {
    return list(bin(fid::mul, make_cast(ref_int32(), t_big, false), ref_qty(), t_big));
  });

  auto q6_pred = [] {
    return conj_and(
      list(cmp(ct::ge, ref_date(), make_date_const(8766)),  // 1994-01-01
           cmp(ct::lt, ref_date(), make_date_const(9131)),  // 1995-01-01
           make_between(
             ref_discount(), make_dec64_const(5, 15, 2), make_dec64_const(7, 15, 2), true, true),
           cmp(ct::lt, ref_qty(), make_bigint_const(24))));
  };
  filter("F01_q6_pred", q6_pred);
  filter("F02_half", [] { return cmp(ct::lt, ref_int32(), make_int_const(500'000)); });
  filter("F03_rare_1pct", [] { return cmp(ct::lt, ref_int32(), make_int_const(10'000)); });
  filter("F04_most_99pct", [] { return cmp(ct::ge, ref_int32(), make_int_const(10'000)); });
  filter("F05_str_eq", [] { return cmp(ct::equal, ref_mode(), make_str_const("MAIL")); });
  filter("F06_str_like",
         [] { return bin(fid::like, ref_comment(), make_str_const("%special%"), t_bool); });
  filter("F07_q6_pred_wide", q6_pred, /*all_cols=*/true);
  filter("F08_nullable_pred",
         [] { return cmp(ct::lt, ref_int32_nullable(), make_int_const(500'000)); });
  return cases;
}

//===----------------------------------------------------------------------===//
// Result verification: a host checksum over valid payload bytes of each output column.
//===----------------------------------------------------------------------===//

std::uint64_t checksum_table(cudf::table_view const& tv, rmm::cuda_stream_view stream)
{
  std::uint64_t h = 1469598103934665603ULL;
  auto mix        = [&](std::uint64_t v) {
    h ^= v;
    h *= 1099511628211ULL;
  };
  mix(static_cast<std::uint64_t>(tv.num_rows()));
  for (cudf::size_type c = 0; c < tv.num_columns(); ++c) {
    auto const col = tv.column(c);
    mix(static_cast<std::uint64_t>(col.null_count()));
    if (col.size() == 0) { continue; }
    if (col.type().id() == cudf::type_id::STRING) {
      // Strings: hash the row lengths, so a wrong gather of string rows fails verification.
      cudf::strings_column_view const scv(col);
      auto const offsets = scv.offsets();
      std::vector<std::int64_t> host_offsets(static_cast<std::size_t>(col.size()) + 1);
      if (offsets.type().id() == cudf::type_id::INT64) {
        cudaMemcpy(host_offsets.data(),
                   offsets.data<std::int64_t>() + col.offset(),
                   host_offsets.size() * sizeof(std::int64_t),
                   cudaMemcpyDeviceToHost);
      } else {
        std::vector<std::int32_t> narrow(host_offsets.size());
        cudaMemcpy(narrow.data(),
                   offsets.data<std::int32_t>() + col.offset(),
                   narrow.size() * sizeof(std::int32_t),
                   cudaMemcpyDeviceToHost);
        std::copy(narrow.begin(), narrow.end(), host_offsets.begin());
      }
      for (std::size_t i = 0; i + 1 < host_offsets.size(); ++i) {
        mix(static_cast<std::uint64_t>(host_offsets[i + 1] - host_offsets[i]));
      }
      continue;
    }
    auto const width = cudf::size_of(col.type());
    std::vector<std::uint8_t> host(static_cast<std::size_t>(col.size()) * width);
    cudaMemcpyAsync(
      host.data(),
      static_cast<std::uint8_t const*>(col.head()) + static_cast<std::size_t>(col.offset()) * width,
      host.size(),
      cudaMemcpyDeviceToHost,
      stream.value());
    auto const valids = copy_valids_to_host(col);
    stream.synchronize();
    for (cudf::size_type i = 0; i < col.size(); ++i) {
      if (!valids[i]) {
        mix(0xFFULL);
        continue;
      }
      std::uint64_t v = 0;
      std::memcpy(
        &v, host.data() + static_cast<std::size_t>(i) * width, std::min<std::size_t>(width, 8));
      mix(v);
      if (width == 16) {
        std::memcpy(&v, host.data() + static_cast<std::size_t>(i) * width + 8, 8);
        mix(v);
      }
    }
  }
  return h;
}

struct measurement {
  double cold_ms{0};
  double median_ms{0};
  double min_ms{0};
  cudf::size_type rows_out{0};
  std::uint64_t checksum{0};
};

double now_ms()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

measurement run_case(bench_case const& bc,
                     bench_input const& in,
                     strategy_t strategy,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr,
                     int iters,
                     int warmup,
                     double warmup_min_ms,
                     bool verify)
{
  auto exprs = bc.build();
  duckdb::vector<std::unique_ptr<ast_node>> owned;
  for (auto& e : exprs) {
    owned.push_back(std::move(e));
  }
  sirius::expression_evaluator exec(owned,
                                    mr,
                                    ::cuda::stream_ref{stream.value()},
                                    sirius::expression_query_policy{strategy},
                                    sirius::expression_evaluator::default_min_ast_size,
                                    /*like_swar_fastpath=*/true);

  std::vector<cudf::size_type> const one_col{0};
  auto run_once = [&]() -> std::unique_ptr<cudf::table> {
    std::unique_ptr<cudf::table> out;
    switch (bc.kind) {
      case case_kind::project: out = exec.evaluate(in.view); break;
      case case_kind::filter_one_col: out = exec.select(in.view, one_col); break;
      case case_kind::filter_all_cols: out = exec.select(in.view); break;
    }
    stream.synchronize();
    return out;
  };

  measurement m;
  nvtx3::scoped_range case_range{bc.name + "/" + std::string(sirius::strategy_to_string(strategy))};
  {
    nvtx3::scoped_range cold_range{"cold"};
    // The process-wide kernel cache would otherwise serve a kernel compiled for an earlier case
    // or row count with the same generated source. Safe here because run_once synchronizes the
    // stream, so no launch from the previous case still uses a cached kernel.
    if (strategy == strategy_t::NVRTC_JIT) { codegen::jit::KernelCache::instance().clear(); }
    double const t0 = now_ms();
    auto out        = run_once();
    m.cold_ms       = now_ms() - t0;
    m.rows_out      = out->num_rows();
    if (verify) { m.checksum = checksum_table(out->view(), stream); }
  }
  // The GPU idles at a low clock between cases (cold compiles leave it idle for most of a
  // second), so warm up for a minimum wall time as well as a minimum number of calls.
  {
    double const warm_start = now_ms();
    for (int i = 0; i < warmup || now_ms() - warm_start < warmup_min_ms; ++i) {
      (void)run_once();
    }
  }
  std::vector<double> times;
  times.reserve(static_cast<std::size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    nvtx3::scoped_range iter_range{"iter"};
    double const t0 = now_ms();
    auto out        = run_once();
    times.push_back(now_ms() - t0);
  }
  std::sort(times.begin(), times.end());
  m.min_ms    = times.front();
  m.median_ms = times[times.size() / 2];
  return m;
}

char const* strategy_name(strategy_t s) { return sirius::strategy_to_string(s).data(); }

}  // namespace

TEST_CASE("bench expression evaluator strategies", "[!benchmark][expr_bench]")
{
  // The on-disk JIT caches would hide compile time in the cold measurement.
  setenv("SIMPATICO_JIT_CACHE_DIR", "off", /*overwrite=*/0);
  std::filesystem::path cudf_cache_dir;
  if (std::getenv("LIBCUDF_KERNEL_CACHE_PATH") == nullptr) {
    std::string tmpl = (std::filesystem::temp_directory_path() / "expr_bench_cudf_XXXXXX").string();
    char* const made = mkdtemp(tmpl.data());
    REQUIRE(made != nullptr);
    cudf_cache_dir = made;
    setenv("LIBCUDF_KERNEL_CACHE_PATH", made, /*overwrite=*/1);
  }
  struct cache_dir_cleanup {
    std::filesystem::path dir;
    ~cache_dir_cleanup()
    {
      if (!dir.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
      }
    }
  } const remove_cudf_cache_dir{cudf_cache_dir};

  auto const row_list      = split_csv(env_or("SIRIUS_EXPR_BENCH_ROWS", "1048576,16777216"));
  auto const strategy_list = split_csv(
    env_or("SIRIUS_EXPR_BENCH_STRATEGIES", "materialize,ast_interpret,ast_jit,nvrtc_jit"));
  auto const case_filters = split_csv(env_or("SIRIUS_EXPR_BENCH_CASES", ""));
  int const iters         = std::stoi(env_or("SIRIUS_EXPR_BENCH_ITERS", "20"));
  int const warmup        = std::stoi(env_or("SIRIUS_EXPR_BENCH_WARMUP", "3"));
  double const warmup_ms  = std::stod(env_or("SIRIUS_EXPR_BENCH_WARMUP_MS", "300"));
  bool const verify       = env_or("SIRIUS_EXPR_BENCH_VERIFY", "1") != "0";
  auto const csv_path     = env_or("SIRIUS_EXPR_BENCH_CSV", "");

  std::vector<strategy_t> strategies;
  for (auto const& s : strategy_list) {
    strategy_t parsed;
    REQUIRE(sirius::string_to_strategy(s, parsed));
    strategies.push_back(parsed);
  }

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_async_resource_ref mr_ref{mr};

  std::ofstream csv;
  if (!csv_path.empty()) {
    bool const exists = std::ifstream(csv_path).good();
    csv.open(csv_path, std::ios::app);
    if (!exists) {
      csv
        << "case,kind,rows,strategy,cold_ms,median_ms,min_ms,input_gbps,rows_out,checksum,match\n";
    }
  }

  // Under nsys the CUDA trace buffers are flushed on cudaProfilerStop(); the test binary exits
  // without running static destructors, so flush explicitly before leaving the test case.
  struct profiler_stop {
    ~profiler_stop() { (void)cudaProfilerStop(); }
  } const stop_profiler_on_exit;
  (void)cudaProfilerStart();

  auto const cases = all_cases();
  for (auto const& rows_str : row_list) {
    auto const rows = static_cast<cudf::size_type>(std::stoll(rows_str));
    auto input      = make_bench_input(rows, stream, mr_ref);
    std::printf("[expr_bench] rows=%d input_columns=%d\n", rows, input.view.num_columns());

    for (auto const& bc : cases) {
      if (!case_filters.empty()) {
        bool keep = false;
        for (auto const& f : case_filters) {
          if (bc.name.find(f) != std::string::npos) { keep = true; }
        }
        if (!keep) { continue; }
      }
      // Bytes touched: referenced input columns + output. Approximated as the referenced inputs
      // for projections (outputs are small) and reported only as a relative throughput figure.
      std::size_t bytes_in = 0;
      {
        auto exprs = bc.build();
        std::vector<bool> referenced(static_cast<std::size_t>(input.view.num_columns()), false);
        for (auto const& e : exprs) {
          sirius::ast::visit_references(*e, [&](sirius::ast::reference const& r) {
            if (r.column_index < referenced.size()) { referenced[r.column_index] = true; }
          });
        }
        for (std::size_t c = 0; c < referenced.size(); ++c) {
          if (referenced[c]) {
            bytes_in += column_bytes(input.view.column(static_cast<cudf::size_type>(c)), stream);
          }
        }
      }

      // The first listed strategy is the reference.
      std::optional<std::uint64_t> reference_checksum;
      for (auto const strategy : strategies) {
        measurement m;
        std::string status = "ok";
        try {
          m = run_case(bc, input, strategy, stream, mr_ref, iters, warmup, warmup_ms, verify);
        } catch (std::exception const& e) {
          status = std::string("error: ") + e.what();
          std::printf("[expr_bench] case=%s rows=%d strategy=%s %s\n",
                      bc.name.c_str(),
                      rows,
                      strategy_name(strategy),
                      status.c_str());
          continue;
        }
        std::string match = "-";
        if (verify) {
          if (strategy == strategies.front()) {
            reference_checksum = m.checksum;
            match              = "ref";
          } else if (!reference_checksum) {
            match = "noref";
          } else {
            match = (*reference_checksum == m.checksum) ? "match" : "MISMATCH";
          }
        }
        double const gbps = (bytes_in / 1e9) / (m.median_ms / 1e3);
        std::printf(
          "[expr_bench] case=%-24s rows=%9d strategy=%-13s cold_ms=%9.3f med_ms=%8.3f min_ms=%8.3f "
          "input_gbps=%7.1f rows_out=%9d chk=%016llx %s\n",
          bc.name.c_str(),
          rows,
          strategy_name(strategy),
          m.cold_ms,
          m.median_ms,
          m.min_ms,
          gbps,
          m.rows_out,
          static_cast<unsigned long long>(m.checksum),
          match.c_str());
        if (csv.is_open()) {
          csv << bc.name << ',' << static_cast<int>(bc.kind) << ',' << rows << ','
              << strategy_name(strategy) << ',' << m.cold_ms << ',' << m.median_ms << ','
              << m.min_ms << ',' << gbps << ',' << m.rows_out << ',' << std::hex << m.checksum
              << std::dec << ',' << match << '\n';
        }
        std::fflush(stdout);
      }
    }
  }
}
