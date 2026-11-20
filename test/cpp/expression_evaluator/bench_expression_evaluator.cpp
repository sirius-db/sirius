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

//===----------------------------------------------------------------------===//
// Expression evaluator microbenchmark across all four execution strategies.
//
// Tagged `[!benchmark]` so default unittest runs skip it. Run with
//   sirius_unittest "[expr_bench]"
//
// Case groups (the prefix of a case name):
//   P, F, B, S   single projections, filters, Materialization Boundaries, string constants
//   L            complexity ladders: one expression shape at increasing size
//                (LA/LN/LD/LF chain depth, LW output count, LC/LO conjunct count, LK CASE arms,
//                LX filter selectivity, LG operand cost behind a selective AND operand)
//   H            TPC-H projection lists and predicates, named H<query>_...
//   DS           TPC-DS projection lists and predicates over nullable columns, named DS<query>_...
//
// Environment:
//   SIRIUS_EXPR_BENCH_ROWS        comma list of row counts (default "1048576,16777216")
//   SIRIUS_EXPR_BENCH_STRATEGIES  comma list of strategies (default all four)
//   SIRIUS_EXPR_BENCH_CASES       comma list of case selectors (default all). A selector picks the
//                                 case with exactly that name; when there is none, every case whose
//                                 name starts with it; when there is none, every case whose name
//                                 contains it
//   SIRIUS_EXPR_BENCH_ITERS       minimum timed iterations per case (default 20)
//   SIRIUS_EXPR_BENCH_MIN_MS      minimum wall time of the timed loop in ms (default 200); the loop
//                                 runs until both minimums are met
//   SIRIUS_EXPR_BENCH_WARMUP      minimum warmup iterations after the cold call (default 3)
//   SIRIUS_EXPR_BENCH_WARMUP_MS   minimum warmup wall time in ms (default 300)
//   SIRIUS_EXPR_BENCH_COLD_REPS   NVRTC_JIT cold calls per case (default 1), each a new compile
//   SIRIUS_EXPR_BENCH_VERIFY      1 (default) compares every strategy's result checksum against
//                                 the first listed strategy (`ref`; `noref` when it failed);
//                                 0 disables the host readback
//   SIRIUS_EXPR_BENCH_FUSE_MULTIPLY_ADD  1 runs NVRTC_JIT as the `nvrtc_jit_fuse_multiply_add`
//                                 setting does (default 0) and reports it as `nvrtc_jit+fma`; a
//                                 result computed from a DOUBLE or FLOAT multiplication and an
//                                 addition or subtraction of its product may then differ from the
//                                 reference strategy
//   SIRIUS_EXPR_BENCH_CSV         append one CSV row per measurement to this file
//   CUDA_CACHE_DISABLE            export 1 so that an NVRTC_JIT cold call is a full compile (see
//                                 below); CUDA reads it at start-up, so the benchmark cannot set it
//
// Each measurement reports the cold first call, then the median and minimum wall time of a
// steady-state call. For NVRTC_JIT the in-memory kernel cache is cleared before every cold call and
// the kernel disk cache is disabled, whatever SIMPATICO_JIT_CACHE_DIR says, so NVRTC compiles every
// time. NVRTC itself keeps the generated code of every source it has compiled in the CUDA compute
// cache (`~/.nv/ComputeCache`): unless the process was started with CUDA_CACHE_DISABLE=1, a source
// compiled before on this machine costs only NVRTC's front end, a fraction of a full compile. The
// `cuda_cache` column records which of the two a row measured. With several cold calls the reported
// cold time is their median, and the compile stages, kernel count and source size are those of that
// median call. For AST_JIT only the first use of each distinct cuDF kernel in the process compiles
// (cuDF's in-memory cache cannot be cleared; its disk cache is pointed at a fresh temporary
// directory). Wall time includes the host-side synchronizations the evaluators perform internally
// plus one stream sync; `ret_ms` is the part of it spent before evaluate() or select() returns,
// which excludes that final sync.
//
// CSV columns after `match`: status (`ok` or `error`), cold_min_ms, cold_reps, the cold call split
// into cold_create_ms (nvrtcCreateProgram), cold_compile_ms (nvrtcCompileProgram), cold_cubin_ms
// (cubin retrieval), cold_load_ms (cuLibraryLoadData) and cold_other_ms (everything else:
// translation, launch, execution), then kernels and src_bytes (kernels compiled by the cold call
// and their total source size), ret_ms, ret_min_ms, iters, outputs, gather_cols, nvrtc_ran (kernels
// > 0), nvrtc_path, cuda_cache (`on` or `off`) and detail (the error text). nvrtc_path is how
// NVRTC_JIT plans the case: `fused` (one kernel computes every expression), `partial` (the kernel
// computes some expressions and the materialize path the others), either with `+boundary` when the
// kernel reads Materialization Boundary columns, or `materialize` (no kernel for the expressions
// themselves; the materialize path may still compile kernels for sub-expressions).
//===----------------------------------------------------------------------===//

#include "codegen/jit/kernel_cache.hpp"
#include "codegen/jit/nvrtc_compiler.hpp"
#include "expression_evaluator/ast_test_support.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/cudf_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

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
#include <expression_evaluator/nvrtc/supportability.hpp>
#include <helper/logical_type.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
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
// Columns 0..10 share one generator and never change:
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
//
// Every later column is drawn from a generator of its own, seeded by its index, so adding a
// column changes no other column. They are declared in `col` below with their contents.
//===----------------------------------------------------------------------===//

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
logical_type const t_dec72  = logical_type::make_decimal(7, 2);
logical_type const t_dec184 = logical_type::make_decimal(18, 4);
logical_type const t_dec382 = logical_type::make_decimal(38, 2);

logical_type dec(std::uint8_t precision, std::uint8_t scale)
{
  return logical_type::make_decimal(precision, scale);
}

struct column_def {
  std::uint32_t index;
  logical_type type;
};

namespace col {

column_def const price{3, t_dec152};
column_def const discount{4, t_dec152};
column_def const tax{5, t_dec152};
column_def const shipdate{6, t_date};
column_def const shipmode{7, t_str};

// Ladder inputs.
column_def const la{11, t_big};  // non-null, uniform [0, 1000)
column_def const lb{12, t_big};  // non-null, uniform [1, 1000]
column_def const ln{13, t_big};  // 20% nulls, uniform [0, 1000)
column_def const ld{14, t_dbl};  // non-null, uniform [1, 100)
// Sixteen non-null INT32 columns 15..30, each uniform [0, 1000).
constexpr int key_count = 16;
column_def key(int i) { return {static_cast<std::uint32_t>(15 + i), t_int}; }

// TPC-H shaped columns, all non-null.
column_def const l_quantity{31, t_dec152};     // whole units 1.00..50.00
column_def const l_commitdate{32, t_date};     // shipdate + [-91, 89] days
column_def const l_receiptdate{33, t_date};    // shipdate + [1, 30] days
column_def const o_orderdate{34, t_date};      // 1992-01-01..1998-08-02
column_def const ps_supplycost{35, t_dec152};  // [1.00, 1000.00]
column_def const c_acctbal{36, t_dec152};      // [-999.99, 9999.99]
column_def const ps_availqty{37, t_big};       // [1, 9999]
column_def const p_size{38, t_int};            // [1, 50]
column_def const l_returnflag{39, t_str};      // R, A, N
column_def const l_shipinstruct{40, t_str};    // 4 values
column_def const o_orderpriority{41, t_str};   // 5 values
column_def const c_mktsegment{42, t_str};      // 5 values
column_def const p_type{43, t_str};            // 6 x 5 x 5 three-word types
column_def const p_brand{44, t_str};           // Brand#11..Brand#55
column_def const p_container{45, t_str};       // 5 x 8 two-word containers
column_def const p_name{46, t_str};            // five colour words
column_def const n_name{47, t_str};            // 25 nations
column_def const c_phone{48, t_str};           // CC-ddd-ddd-dddd, CC in 10..34
column_def const o_comment{49, t_str};         // 6..12 words of the TPC-H text vocabulary

// TPC-DS shaped columns; fact columns are nullable as in TPC-DS.
column_def const ss_quantity{50, t_int};           // 4% nulls, [1, 100]
column_def const ss_wholesale_cost{51, t_dec72};   // 4% nulls, [1.00, 100.00]
column_def const ss_list_price{52, t_dec72};       // 4% nulls, [1.00, 200.00]
column_def const ss_sales_price{53, t_dec72};      // 4% nulls, [0.00, 200.00]
column_def const ss_ext_sales_price{54, t_dec72};  // 4% nulls, [0.00, 20000.00]
column_def const ss_coupon_amt{55, t_dec72};       // 4% nulls, [0.00, 10000.00]
column_def const ss_net_paid{56, t_dec72};         // 4% nulls, [0.00, 20000.00]
column_def const ss_net_profit{57, t_dec72};       // 4% nulls, [-10000.00, 10000.00]
column_def const sr_return_amt{58, t_dec72};       // 90% nulls (outer-join miss), [0.00, 10000.00]
column_def const sr_return_quantity{59, t_int};    // 90% nulls, [1, 100]
column_def const ss_customer_sk{60, t_int};        // 4% nulls, [1, 100000]
column_def const ws_sold_date_sk{61, t_int};       // 1% nulls, [2450815, 2452642]
column_def const ws_ship_date_sk{62, t_int};       // 1% nulls, sold date + [1, 150]
column_def const d_year{63, t_int};                // [1998, 2002]
column_def const d_moy{64, t_int};                 // [1, 12]
column_def const d_dom{65, t_int};                 // [1, 28]
column_def const t_hour{66, t_int};                // [0, 23]
column_def const t_minute{67, t_int};              // [0, 59]
column_def const hd_dep_count{68, t_int};          // [0, 9]
column_def const hd_vehicle_count{69, t_int};      // [-1, 4]
column_def const sum_sales{70, t_dec382};          // 2% nulls, [0.01, 10000000.00]
column_def const sum_sales2{71, t_dec382};         // 2% nulls, [0.01, 10000000.00]
column_def const avg_sales{72, t_dbl};             // 2% nulls, sum_sales x [0.8, 1.2], 5% zero
column_def const d_day_name{73, t_str};            // 7 values
column_def const cd_marital_status{74, t_str};     // 5 values
column_def const cd_education_status{75, t_str};   // 7 values
column_def const cd_gender{76, t_str};             // M, F
column_def const hd_buy_potential{77, t_str};      // 6 values
column_def const ca_state{78, t_str};              // 3% nulls, 50 values
column_def const ca_zip{79, t_str};      // 3% nulls, ddddd-dddd with 400 distinct prefixes
column_def const i_category{80, t_str};  // 2% nulls, 10 values
column_def const i_class{81, t_str};     // 2% nulls, 16 values
column_def const i_brand{82, t_str};     // 2% nulls, 24 values

constexpr std::uint32_t base_count = 11;
constexpr std::uint32_t count      = 83;

}  // namespace col

struct bench_input {
  // One entry per input column; null for a column no selected case uses.
  std::vector<std::unique_ptr<cudf::column>> columns;
  // Zeroed storage every unused column's view aliases, so the table keeps its shape without
  // holding device memory for data nothing reads.
  rmm::device_buffer placeholder;
  cudf::table_view view;
  // Device bytes of each column, for the relative throughput figure.
  std::vector<std::size_t> column_bytes;
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
  rmm::device_buffer mask(host.data(), bytes, stream, mr);
  // Pageable sources may be read asynchronously (HMM/ATS); the local buffer must outlive the copy.
  stream.synchronize();
  return mask;
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

// Host-side contents of one appended column.
struct host_column {
  cudf::data_type type{cudf::type_id::EMPTY};
  std::vector<char> data;                // fixed-width values, or the characters of a STRING column
  std::vector<cudf::size_type> offsets;  // STRING only: rows + 1 entries
  std::vector<bool> valids;              // empty when the column has no nulls
};

std::unique_ptr<cudf::column> upload_host_column(host_column const& host,
                                                 cudf::size_type rows,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  rmm::device_buffer mask(0, stream, mr);
  cudf::size_type null_count = 0;
  if (!host.valids.empty()) { mask = upload_mask(host.valids, stream, mr, null_count); }
  rmm::device_buffer data(host.data.data(), host.data.size(), stream, mr);
  if (host.type.id() != cudf::type_id::STRING) {
    return std::make_unique<cudf::column>(
      host.type, rows, std::move(data), std::move(mask), null_count);
  }
  rmm::device_buffer offsets(
    host.offsets.data(), host.offsets.size() * sizeof(cudf::size_type), stream, mr);
  auto offsets_col = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                    rows + 1,
                                                    std::move(offsets),
                                                    rmm::device_buffer{0, stream, mr},
                                                    0);
  return cudf::make_strings_column(
    rows, std::move(offsets_col), std::move(data), null_count, std::move(mask));
}

std::mt19937_64 column_rng(std::uint32_t column)
{
  return std::mt19937_64{0xC01A5EEDULL + 0x9E3779B97F4A7C15ULL * column};
}

// Fills a fixed-width column with `draw(rng, row)`; the drawn type must be the column's storage
// type.
template <typename Draw>
host_column generate_fixed(column_def const& def, cudf::size_type rows, int null_pct, Draw draw)
{
  using value_type = std::invoke_result_t<Draw&, std::mt19937_64&, cudf::size_type>;
  host_column out;
  out.type = sirius::require_cudf_type(def.type);
  if (cudf::size_of(out.type) != sizeof(value_type)) {
    throw std::logic_error("bench column " + std::to_string(def.index) +
                           ": storage width mismatch");
  }
  auto rng = column_rng(def.index);
  std::uniform_int_distribution<int> pct(0, 99);
  out.data.resize(sizeof(value_type) * static_cast<std::size_t>(rows));
  if (null_pct > 0) { out.valids.resize(static_cast<std::size_t>(rows)); }
  for (cudf::size_type i = 0; i < rows; ++i) {
    value_type const v = draw(rng, i);
    std::memcpy(out.data.data() + sizeof(value_type) * static_cast<std::size_t>(i), &v, sizeof(v));
    if (null_pct > 0) { out.valids[static_cast<std::size_t>(i)] = pct(rng) >= null_pct; }
  }
  return out;
}

// Fills a STRING column; `append(rng, chars)` appends the characters of one valid row.
template <typename Append>
host_column generate_strings(column_def const& def,
                             cudf::size_type rows,
                             int null_pct,
                             Append append)
{
  host_column out;
  out.type = cudf::data_type{cudf::type_id::STRING};
  auto rng = column_rng(def.index);
  std::uniform_int_distribution<int> pct(0, 99);
  out.offsets.resize(static_cast<std::size_t>(rows) + 1);
  if (null_pct > 0) { out.valids.resize(static_cast<std::size_t>(rows)); }
  for (cudf::size_type i = 0; i < rows; ++i) {
    out.offsets[static_cast<std::size_t>(i)] = static_cast<cudf::size_type>(out.data.size());
    bool const valid                         = null_pct == 0 || pct(rng) >= null_pct;
    if (null_pct > 0) { out.valids[static_cast<std::size_t>(i)] = valid; }
    if (valid) { append(rng, out.data); }
  }
  out.offsets[static_cast<std::size_t>(rows)] = static_cast<cudf::size_type>(out.data.size());
  return out;
}

template <typename T>
auto uniform(T lo, T hi)
{
  return [dist = std::uniform_int_distribution<T>(lo, hi)](
           std::mt19937_64& rng, cudf::size_type) mutable { return dist(rng); };
}

void append_text(std::vector<char>& out, std::string_view text)
{
  out.insert(out.end(), text.begin(), text.end());
}

template <std::size_t N>
std::string_view pick(std::array<std::string_view, N> const& words, std::mt19937_64& rng)
{
  return words[std::uniform_int_distribution<std::size_t>(0, N - 1)(rng)];
}

// Appends one uniformly drawn word per row.
template <std::size_t N>
auto vocabulary(std::array<std::string_view, N> const& words)
{
  return
    [&words](std::mt19937_64& rng, std::vector<char>& out) { append_text(out, pick(words, rng)); };
}

constexpr auto ship_instructs =
  std::to_array<std::string_view>({"DELIVER IN PERSON", "COLLECT COD", "NONE", "TAKE BACK RETURN"});
constexpr auto return_flags = std::to_array<std::string_view>({"R", "A", "N"});
constexpr auto order_priorities =
  std::to_array<std::string_view>({"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"});
constexpr auto market_segments = std::to_array<std::string_view>(
  {"AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"});
constexpr auto type_sizes =
  std::to_array<std::string_view>({"STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO"});
constexpr auto type_finishes =
  std::to_array<std::string_view>({"ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED"});
constexpr auto type_metals =
  std::to_array<std::string_view>({"TIN", "NICKEL", "BRASS", "STEEL", "COPPER"});
constexpr auto container_sizes =
  std::to_array<std::string_view>({"SM", "LG", "MED", "JUMBO", "WRAP"});
constexpr auto container_kinds =
  std::to_array<std::string_view>({"CASE", "BOX", "BAG", "JAR", "PKG", "PACK", "CAN", "DRUM"});
constexpr auto colours = std::to_array<std::string_view>(
  {"almond",   "antique",   "aquamarine", "azure",    "beige",     "bisque",     "black",
   "blanched", "blue",      "blush",      "brown",    "burlywood", "chartreuse", "chocolate",
   "coral",    "cornsilk",  "cream",      "cyan",     "dark",      "deep",       "dim",
   "dodger",   "drab",      "firebrick",  "floral",   "forest",    "frosted",    "gainsboro",
   "ghost",    "goldenrod", "green",      "grey",     "honeydew",  "hot",        "indian",
   "ivory",    "khaki",     "lace",       "lavender", "yellow"});
constexpr auto nations = std::to_array<std::string_view>(
  {"ALGERIA", "ARGENTINA", "BRAZIL",         "CANADA",       "EGYPT", "ETHIOPIA", "FRANCE",
   "GERMANY", "INDIA",     "INDONESIA",      "IRAN",         "IRAQ",  "JAPAN",    "JORDAN",
   "KENYA",   "MOROCCO",   "MOZAMBIQUE",     "PERU",         "CHINA", "ROMANIA",  "SAUDI ARABIA",
   "VIETNAM", "RUSSIA",    "UNITED KINGDOM", "UNITED STATES"});
constexpr auto comment_words = std::to_array<std::string_view>(
  {"packages", "requests", "accounts",     "deposits",     "foxes",   "ideas",     "theodolites",
   "pinto",    "beans",    "instructions", "dependencies", "excuses", "platelets", "asymptotes",
   "courts",   "dolphins", "sleep",        "wake",         "are",     "cajole",    "haggle",
   "nag",      "use",      "boost",        "furiously",    "slyly",   "carefully", "blithely",
   "quickly",  "fluffily", "slowly",       "quietly",      "regular", "final",     "ironic",
   "even",     "bold",     "silent",       "special",      "pending", "unusual",   "express",
   "above",    "against",  "along",        "among",        "around",  "the"});
constexpr auto day_names = std::to_array<std::string_view>(
  {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"});
constexpr auto marital_statuses   = std::to_array<std::string_view>({"M", "S", "D", "W", "U"});
constexpr auto education_statuses = std::to_array<std::string_view>(
  {"Primary", "Secondary", "College", "2 yr Degree", "4 yr Degree", "Advanced Degree", "Unknown"});
constexpr auto genders        = std::to_array<std::string_view>({"M", "F"});
constexpr auto buy_potentials = std::to_array<std::string_view>(
  {">10000", "Unknown", "1001-5000", "501-1000", "0-500", "5001-10000"});
constexpr auto states = std::to_array<std::string_view>(
  {"AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
   "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
   "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "OH", "OK", "OR", "PA", "RI",
   "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"});
constexpr auto item_categories = std::to_array<std::string_view>({"Books",
                                                                  "Children",
                                                                  "Electronics",
                                                                  "Home",
                                                                  "Jewelry",
                                                                  "Men",
                                                                  "Music",
                                                                  "Shoes",
                                                                  "Sports",
                                                                  "Women"});

constexpr auto item_classes = std::to_array<std::string_view>({"personal",
                                                               "portable",
                                                               "reference",
                                                               "self-help",
                                                               "accessories",
                                                               "classical",
                                                               "fragrances",
                                                               "pants",
                                                               "computers",
                                                               "stereo",
                                                               "football",
                                                               "shirts",
                                                               "dresses",
                                                               "rock",
                                                               "country",
                                                               "bedding"});

constexpr auto item_brands = std::to_array<std::string_view>(
  {"scholaramalgamalg #14", "scholaramalgamalg #7", "exportiunivamalg #9", "scholaramalgamalg #9",
   "amalgimporto #1",       "edu packscholar #1",   "exportiimporto #1",   "importoamalg #1",
   "amalgamalg #1",         "edu packamalg #2",     "exportiamalg #1",     "importoexporti #1",
   "corpmaxi #7",           "univunivamalg #4",     "namelessmaxi #3",     "amalgscholar #2",
   "brandbrand #5",         "corpunivamalg #6",     "maxinameless #8",     "scholarbrand #2",
   "exportischolar #2",     "importoedu pack #1",   "edu packimporto #2",  "amalgexporti #1"});

// The five-digit prefix of the k-th distinct zip code, k in [0, 400).
std::int32_t zip_prefix(int k) { return 10'000 + (k * 223) % 90'000; }

void append_digits(std::vector<char>& out, std::int64_t value, int width)
{
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%0*lld", width, static_cast<long long>(value));
  append_text(out, buf);
}

// Runs every task on a pool of threads and rethrows the first exception a task threw.
void run_parallel(std::vector<std::function<void()>> const& tasks)
{
  std::atomic<std::size_t> next{0};
  std::mutex error_mutex;
  std::exception_ptr error;
  {
    std::vector<std::jthread> workers;
    auto const worker_count =
      std::min<std::size_t>(tasks.size(), std::max(1U, std::thread::hardware_concurrency()));
    for (std::size_t w = 0; w < worker_count; ++w) {
      workers.emplace_back([&] {
        for (auto i = next.fetch_add(1); i < tasks.size(); i = next.fetch_add(1)) {
          try {
            tasks[i]();
          } catch (...) {
            std::lock_guard<std::mutex> const lock(error_mutex);
            if (!error) { error = std::current_exception(); }
          }
        }
      });
    }
  }
  if (error) { std::rethrow_exception(error); }
}

// Generates columns `col::base_count`..`col::count - 1`; the entries below `col::base_count` stay
// empty. `shipdate` is the host copy of column 6, which the commit and receipt dates follow.
std::vector<host_column> generate_appended_columns(cudf::size_type rows,
                                                   std::span<std::int32_t const> shipdate,
                                                   std::vector<bool> const& used)
{
  std::vector<host_column> host(col::count);
  std::vector<std::function<void()>> tasks;
  // Every column records its type; only a used column is filled.
  auto record_type = [&](column_def const& def) {
    host[def.index].type = sirius::require_cudf_type(def.type);
  };
  auto fixed = [&](column_def const& def, int null_pct, auto draw) {
    record_type(def);
    if (!used[def.index]) { return; }
    tasks.emplace_back([&host, def, rows, null_pct, draw] {
      host[def.index] = generate_fixed(def, rows, null_pct, draw);
    });
  };
  auto strings = [&](column_def const& def, int null_pct, auto append) {
    record_type(def);
    if (!used[def.index]) { return; }
    tasks.emplace_back([&host, def, rows, null_pct, append] {
      host[def.index] = generate_strings(def, rows, null_pct, append);
    });
  };
  using i32 = std::int32_t;
  using i64 = std::int64_t;

  fixed(col::la, 0, uniform<i64>(0, 999));
  fixed(col::lb, 0, uniform<i64>(1, 1000));
  fixed(col::ln, 20, uniform<i64>(0, 999));
  fixed(col::ld,
        0,
        [dist = std::uniform_real_distribution<double>(1.0, 100.0)](
          std::mt19937_64& rng, cudf::size_type) mutable { return dist(rng); });
  for (int k = 0; k < col::key_count; ++k) {
    fixed(col::key(k), 0, uniform<i32>(0, 999));
  }

  fixed(col::l_quantity,
        0,
        [dist = std::uniform_int_distribution<i64>(1, 50)](
          std::mt19937_64& rng, cudf::size_type) mutable { return dist(rng) * 100; });
  fixed(col::l_commitdate,
        0,
        [shipdate, dist = std::uniform_int_distribution<i32>(-91, 89)](
          std::mt19937_64& rng, cudf::size_type row) mutable {
          return shipdate[static_cast<std::size_t>(row)] + dist(rng);
        });
  fixed(col::l_receiptdate,
        0,
        [shipdate, dist = std::uniform_int_distribution<i32>(1, 30)](std::mt19937_64& rng,
                                                                     cudf::size_type row) mutable {
          return shipdate[static_cast<std::size_t>(row)] + dist(rng);
        });
  fixed(col::o_orderdate, 0, uniform<i32>(8036, 10440));
  fixed(col::ps_supplycost, 0, uniform<i64>(100, 100'000));
  fixed(col::c_acctbal, 0, uniform<i64>(-99'999, 999'999));
  fixed(col::ps_availqty, 0, uniform<i64>(1, 9999));
  fixed(col::p_size, 0, uniform<i32>(1, 50));
  strings(col::l_returnflag, 0, vocabulary(return_flags));
  strings(col::l_shipinstruct, 0, vocabulary(ship_instructs));
  strings(col::o_orderpriority, 0, vocabulary(order_priorities));
  strings(col::c_mktsegment, 0, vocabulary(market_segments));
  strings(col::p_type, 0, [](std::mt19937_64& rng, std::vector<char>& out) {
    append_text(out, pick(type_sizes, rng));
    out.push_back(' ');
    append_text(out, pick(type_finishes, rng));
    out.push_back(' ');
    append_text(out, pick(type_metals, rng));
  });
  strings(col::p_brand,
          0,
          [dist = std::uniform_int_distribution<int>(1, 5)](std::mt19937_64& rng,
                                                            std::vector<char>& out) mutable {
            append_text(out, "Brand#");
            out.push_back(static_cast<char>('0' + dist(rng)));
            out.push_back(static_cast<char>('0' + dist(rng)));
          });
  strings(col::p_container, 0, [](std::mt19937_64& rng, std::vector<char>& out) {
    append_text(out, pick(container_sizes, rng));
    out.push_back(' ');
    append_text(out, pick(container_kinds, rng));
  });
  strings(col::p_name, 0, [](std::mt19937_64& rng, std::vector<char>& out) {
    for (int w = 0; w < 5; ++w) {
      if (w > 0) { out.push_back(' '); }
      append_text(out, pick(colours, rng));
    }
  });
  strings(col::n_name, 0, vocabulary(nations));
  strings(col::c_phone,
          0,
          [country = std::uniform_int_distribution<int>(10, 34),
           part    = std::uniform_int_distribution<int>(0, 9999)](std::mt19937_64& rng,
                                                               std::vector<char>& out) mutable {
            append_digits(out, country(rng), 2);
            out.push_back('-');
            append_digits(out, part(rng) % 1000, 3);
            out.push_back('-');
            append_digits(out, part(rng) % 1000, 3);
            out.push_back('-');
            append_digits(out, part(rng), 4);
          });
  // One row in a hundred carries "Customer ... Complaints", as TPC-H supplier comments do.
  strings(col::o_comment,
          0,
          [length = std::uniform_int_distribution<int>(6, 12),
           pct    = std::uniform_int_distribution<int>(0, 99)](std::mt19937_64& rng,
                                                            std::vector<char>& out) mutable {
            int const words      = length(rng);
            bool const complaint = pct(rng) == 0;
            for (int w = 0; w < words; ++w) {
              if (w > 0) { out.push_back(' '); }
              if (complaint && w == 1) {
                append_text(out, "Customer");
              } else if (complaint && w == words - 2) {
                append_text(out, "Complaints");
              } else {
                append_text(out, pick(comment_words, rng));
              }
            }
          });

  fixed(col::ss_quantity, 4, uniform<i32>(1, 100));
  fixed(col::ss_wholesale_cost, 4, uniform<i32>(100, 10'000));
  fixed(col::ss_list_price, 4, uniform<i32>(100, 20'000));
  fixed(col::ss_sales_price, 4, uniform<i32>(0, 20'000));
  fixed(col::ss_ext_sales_price, 4, uniform<i32>(0, 2'000'000));
  fixed(col::ss_coupon_amt, 4, uniform<i32>(0, 1'000'000));
  fixed(col::ss_net_paid, 4, uniform<i32>(0, 2'000'000));
  fixed(col::ss_net_profit, 4, uniform<i32>(-1'000'000, 1'000'000));
  fixed(col::sr_return_amt, 90, uniform<i32>(0, 1'000'000));
  fixed(col::sr_return_quantity, 90, uniform<i32>(1, 100));
  fixed(col::ss_customer_sk, 4, uniform<i32>(1, 100'000));
  // The ship date follows the sold date of the same row, so one generator draws both.
  record_type(col::ws_sold_date_sk);
  record_type(col::ws_ship_date_sk);
  if (used[col::ws_sold_date_sk.index] || used[col::ws_ship_date_sk.index]) {
    tasks.emplace_back([&host, rows] {
      std::vector<i32> sold(static_cast<std::size_t>(rows));
      host[col::ws_sold_date_sk.index] =
        generate_fixed(col::ws_sold_date_sk,
                       rows,
                       1,
                       [&sold, dist = std::uniform_int_distribution<i32>(2'450'815, 2'452'642)](
                         std::mt19937_64& rng, cudf::size_type row) mutable {
                         return sold[static_cast<std::size_t>(row)] = dist(rng);
                       });
      host[col::ws_ship_date_sk.index] =
        generate_fixed(col::ws_ship_date_sk,
                       rows,
                       1,
                       [&sold, dist = std::uniform_int_distribution<i32>(1, 150)](
                         std::mt19937_64& rng, cudf::size_type row) mutable {
                         return sold[static_cast<std::size_t>(row)] + dist(rng);
                       });
    });
  }
  fixed(col::d_year, 0, uniform<i32>(1998, 2002));
  fixed(col::d_moy, 0, uniform<i32>(1, 12));
  fixed(col::d_dom, 0, uniform<i32>(1, 28));
  fixed(col::t_hour, 0, uniform<i32>(0, 23));
  fixed(col::t_minute, 0, uniform<i32>(0, 59));
  fixed(col::hd_dep_count, 0, uniform<i32>(0, 9));
  fixed(col::hd_vehicle_count, 0, uniform<i32>(-1, 4));
  // The average follows the sum of the same row, so one generator draws both.
  record_type(col::sum_sales);
  record_type(col::avg_sales);
  if (used[col::sum_sales.index] || used[col::avg_sales.index]) {
    tasks.emplace_back([&host, rows] {
      std::vector<i64> sums(static_cast<std::size_t>(rows));
      host[col::sum_sales.index] = generate_fixed(
        col::sum_sales,
        rows,
        2,
        [&sums, dist = std::uniform_int_distribution<i64>(1, 1'000'000'000)](
          std::mt19937_64& rng, cudf::size_type row) mutable {
          return static_cast<__int128_t>(sums[static_cast<std::size_t>(row)] = dist(rng));
        });
      host[col::avg_sales.index] = generate_fixed(
        col::avg_sales,
        rows,
        2,
        [&sums,
         ratio = std::uniform_real_distribution<double>(0.8, 1.2),
         pct   = std::uniform_int_distribution<int>(0, 99)](std::mt19937_64& rng,
                                                          cudf::size_type row) mutable {
          double const scaled =
            static_cast<double>(sums[static_cast<std::size_t>(row)]) / 100.0 * ratio(rng);
          return pct(rng) < 5 ? 0.0 : scaled;
        });
    });
  }
  fixed(col::sum_sales2,
        2,
        [dist = std::uniform_int_distribution<i64>(1, 1'000'000'000)](std::mt19937_64& rng,
                                                                      cudf::size_type) mutable {
          return static_cast<__int128_t>(dist(rng));
        });
  strings(col::d_day_name, 0, vocabulary(day_names));
  strings(col::cd_marital_status, 0, vocabulary(marital_statuses));
  strings(col::cd_education_status, 0, vocabulary(education_statuses));
  strings(col::cd_gender, 0, vocabulary(genders));
  strings(col::hd_buy_potential, 0, vocabulary(buy_potentials));
  strings(col::ca_state, 3, vocabulary(states));
  strings(col::ca_zip,
          3,
          [prefix = std::uniform_int_distribution<int>(0, 399),
           plus4  = std::uniform_int_distribution<int>(0, 9999)](std::mt19937_64& rng,
                                                                std::vector<char>& out) mutable {
            append_digits(out, zip_prefix(prefix(rng)), 5);
            out.push_back('-');
            append_digits(out, plus4(rng), 4);
          });
  strings(col::i_category, 2, vocabulary(item_categories));
  strings(col::i_class, 2, vocabulary(item_classes));
  strings(col::i_brand, 2, vocabulary(item_brands));

  run_parallel(tasks);
  return host;
}

// The bytes a placeholder view of `rows` values of `type` reads: a STRING one reads its offsets.
std::size_t placeholder_size(cudf::data_type type, cudf::size_type rows)
{
  auto const n = static_cast<std::size_t>(rows);
  return type.id() == cudf::type_id::STRING ? (n + 1) * sizeof(std::int32_t)
                                            : n * cudf::size_of(type);
}

// A view of `rows` zero values (empty strings for STRING) over `storage`.
cudf::column_view placeholder_view(cudf::data_type type,
                                   cudf::size_type rows,
                                   rmm::device_buffer const& storage)
{
  if (type.id() != cudf::type_id::STRING) {
    return cudf::column_view(type, rows, storage.data(), nullptr, 0);
  }
  cudf::column_view const offsets(
    cudf::data_type{cudf::type_id::INT32}, rows + 1, storage.data(), nullptr, 0);
  return cudf::column_view(type, rows, storage.data(), nullptr, 0, 0, {offsets});
}

// Builds the input table. Only the columns marked in `used` are uploaded; the others are views of
// one placeholder as large as the widest of them, so a run of a few cases holds little device
// memory beyond what those cases read.
bench_input make_bench_input(cudf::size_type rows,
                             std::vector<bool> const& used,
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

  auto const appended = generate_appended_columns(rows, c6, used);

  bench_input in;
  auto const dec_type  = cudf::data_type{cudf::type_id::DECIMAL64, -2};
  auto const int_type  = cudf::data_type{cudf::type_id::INT32};
  auto const big_type  = cudf::data_type{cudf::type_id::INT64};
  auto const dbl_type  = cudf::data_type{cudf::type_id::FLOAT64};
  auto const date_type = cudf::data_type{cudf::type_id::TIMESTAMP_DAYS};
  auto const str_type  = cudf::data_type{cudf::type_id::STRING};
  std::vector<cudf::data_type> types{int_type,
                                     big_type,
                                     dbl_type,
                                     dec_type,
                                     dec_type,
                                     dec_type,
                                     date_type,
                                     str_type,
                                     str_type,
                                     int_type,
                                     big_type};
  using uploader = std::function<std::unique_ptr<cudf::column>()>;
  std::vector<uploader> const base_uploads{
    [&] { return upload_fixed(int_type, c0, nullptr, stream, mr); },
    [&] { return upload_fixed(big_type, c1, &v1, stream, mr); },
    [&] { return upload_fixed(dbl_type, c2, nullptr, stream, mr); },
    [&] { return upload_fixed(dec_type, c3, nullptr, stream, mr); },
    [&] { return upload_fixed(dec_type, c4, nullptr, stream, mr); },
    [&] { return upload_fixed(dec_type, c5, nullptr, stream, mr); },
    [&] { return upload_fixed(date_type, c6, nullptr, stream, mr); },
    [&] { return upload_strings(c7, stream, mr); },
    [&] { return upload_strings(c8, stream, mr); },
    [&] { return upload_fixed(int_type, c9, &v9, stream, mr); },
    [&] { return upload_fixed(big_type, c10, nullptr, stream, mr); }};
  for (std::uint32_t c = col::base_count; c < col::count; ++c) {
    types.push_back(appended[c].type);
  }
  in.columns.resize(col::count);
  for (std::uint32_t c = 0; c < col::count; ++c) {
    if (!used[c]) { continue; }
    in.columns[c] =
      c < col::base_count ? base_uploads[c]() : upload_host_column(appended[c], rows, stream, mr);
  }
  std::size_t placeholder_bytes = 0;
  for (std::uint32_t c = 0; c < col::count; ++c) {
    if (!used[c]) {
      placeholder_bytes = std::max(placeholder_bytes, placeholder_size(types[c], rows));
    }
  }
  in.placeholder = rmm::device_buffer(placeholder_bytes, stream, mr);
  if (placeholder_bytes > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(in.placeholder.data(), 0, placeholder_bytes, stream.value()));
  }
  // The host vectors above die with this frame; wait for every upload to finish reading them.
  stream.synchronize();

  std::vector<cudf::column_view> views;
  for (std::uint32_t c = 0; c < col::count; ++c) {
    if (in.columns[c] == nullptr) {
      views.push_back(placeholder_view(types[c], rows, in.placeholder));
      in.column_bytes.push_back(0);
      continue;
    }
    views.push_back(in.columns[c]->view());
    in.column_bytes.push_back(column_bytes(in.columns[c]->view(), stream));
  }
  in.view = cudf::table_view(views);
  return in;
}

//===----------------------------------------------------------------------===//
// Expression builders
//===----------------------------------------------------------------------===//

using node_ptr = std::unique_ptr<ast_node>;

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

node_ptr ref(column_def const& def) { return make_ref_typed(def.index, def.type); }

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

node_ptr conj_or(std::vector<node_ptr> children)
{
  return make_conj(sirius::ast::conjunction::kind::op_or, std::move(children));
}

node_ptr logical_not(node_ptr child)
{
  return make_unary(sirius::ast::unary_op::kind::op_not, std::move(child));
}

node_ptr cast_to(node_ptr child, logical_type type)
{
  return make_cast(std::move(child), std::move(type), /*try_cast=*/false);
}

// A DECIMAL constant in the narrowest payload Sirius carries for `precision`.
node_ptr dec_const(std::int64_t unscaled, std::uint8_t precision, std::uint8_t scale)
{
  if (precision <= logical_type::decimal_max_precision_int32) {
    return make_dec32_const(static_cast<std::int32_t>(unscaled), precision, scale);
  }
  if (precision <= logical_type::decimal_max_precision_int64) {
    return make_dec64_const(unscaled, precision, scale);
  }
  return make_dec128_const(unscaled, precision, scale);
}

// A constant of `def`'s DECIMAL type, which is how DuckDB types the constant side of a comparison.
node_ptr dec_const_of(column_def const& def, std::int64_t unscaled)
{
  return dec_const(unscaled, def.type.decimal_precision(), def.type.decimal_scale());
}

node_ptr date_const(int year, unsigned month, unsigned day)
{
  using namespace std::chrono;
  auto const days =
    sys_days{std::chrono::year{year} / std::chrono::month{month} / std::chrono::day{day}};
  return make_date_const(static_cast<std::int32_t>(days.time_since_epoch().count()));
}

node_ptr str_eq(column_def const& def, char const* text)
{
  return cmp(sirius::comparison_type::equal, ref(def), make_str_const(text));
}

node_ptr str_in(node_ptr probe, std::initializer_list<char const*> texts)
{
  std::vector<node_ptr> values;
  for (auto const* text : texts) {
    values.push_back(make_str_const(text));
  }
  return make_in(std::move(probe), std::move(values), /*negated=*/false);
}

node_ptr int_in(node_ptr probe, std::initializer_list<std::int32_t> numbers)
{
  std::vector<node_ptr> values;
  for (auto const number : numbers) {
    values.push_back(make_int_const(number));
  }
  return make_in(std::move(probe), std::move(values), /*negated=*/false);
}

// `lo <= def AND def <= hi` as two comparisons, which is how DuckDB plans most BETWEEN predicates.
node_ptr in_range(column_def const& def, node_ptr lo, node_ptr hi)
{
  return conj_and(list(cmp(sirius::comparison_type::ge, ref(def), std::move(lo)),
                       cmp(sirius::comparison_type::le, ref(def), std::move(hi))));
}

node_ptr int_range(column_def const& def, std::int32_t lo, std::int32_t hi)
{
  return in_range(def, make_int_const(lo), make_int_const(hi));
}

node_ptr dec_range(column_def const& def, std::int64_t lo_unscaled, std::int64_t hi_unscaled)
{
  return in_range(def, dec_const_of(def, lo_unscaled), dec_const_of(def, hi_unscaled));
}

node_ptr case_when(node_ptr when, node_ptr then, node_ptr otherwise, logical_type type)
{
  std::vector<sirius::ast::case_expr::when_then> whens;
  whens.push_back({std::move(when), std::move(then)});
  return std::make_unique<ast_node>(
    sirius::ast::case_expr{std::move(whens), std::move(otherwise), std::move(type)});
}

node_ptr year_of(column_def const& def) { return fn1(sirius::function_id::year, ref(def), t_big); }

node_ptr to_double(node_ptr child) { return cast_to(std::move(child), t_dbl); }

// l_extendedprice * (1.00 - l_discount) -> DECIMAL(18,4): DuckDB caps the product of two
// DECIMALs that each fit 64 bits at 18 digits and checks the multiplication for overflow.
node_ptr volume()
{
  return bin(sirius::function_id::mul,
             ref(col::price),
             bin(sirius::function_id::sub, dec_const(100, 16, 2), ref(col::discount), t_dec162),
             t_dec184);
}

enum class case_kind { project, filter_one_col, filter_all_cols, filter_cols };

struct bench_case {
  std::string name;
  case_kind kind;
  std::function<std::vector<node_ptr>()> build;
  // The input columns a filter case gathers for the surviving rows.
  std::vector<cudf::size_type> gather;
};

// Collects cases as they are declared.
class case_list {
 public:
  void project(std::string name, std::function<std::vector<node_ptr>()> build)
  {
    _cases.push_back({std::move(name), case_kind::project, std::move(build), {}});
  }

  // A filter that gathers column 0, or with `all_cols` the eleven columns 0..10.
  void filter(std::string name, std::function<node_ptr()> build, bool all_cols = false)
  {
    std::vector<cudf::size_type> gather{0};
    if (all_cols) {
      gather.resize(col::base_count);
      std::iota(gather.begin(), gather.end(), 0);
    }
    _cases.push_back({std::move(name),
                      all_cols ? case_kind::filter_all_cols : case_kind::filter_one_col,
                      [build = std::move(build)]() { return list(build()); },
                      std::move(gather)});
  }

  // A filter that gathers the listed columns.
  void filter(std::string name,
              std::initializer_list<column_def> gather,
              std::function<node_ptr()> build)
  {
    std::vector<cudf::size_type> indices;
    for (auto const& def : gather) {
      indices.push_back(static_cast<cudf::size_type>(def.index));
    }
    _cases.push_back({std::move(name),
                      case_kind::filter_cols,
                      [build = std::move(build)]() { return list(build()); },
                      std::move(indices)});
  }

  [[nodiscard]] std::vector<bench_case> release() { return std::move(_cases); }

 private:
  std::vector<bench_case> _cases;
};

void add_basic_cases(case_list& cases)
{
  using fid    = sirius::function_id;
  using ct     = sirius::comparison_type;
  auto project = [&](std::string name, std::function<std::vector<node_ptr>()> b) {
    cases.project(std::move(name), std::move(b));
  };
  auto filter = [&](std::string name, std::function<node_ptr()> b, bool all_cols = false) {
    cases.filter(std::move(name), std::move(b), all_cols);
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
  // A nullable result that can carry Overflow Poison: the NVRTC Sink reads the poison of every row.
  project("P19_nullable_add_abs", [] {
    return list(bin(fid::add, ref_int32_nullable(), fn1(fid::abs, ref_int32(), t_int), t_int));
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
  filter("F09_nullable_overflow_pred", [] {
    return cmp(
      ct::gt, bin(fid::add, ref_int32_nullable(), ref_int32(), t_int), make_int_const(1'000'000));
  });

  // Materialization Boundaries: upper() is evaluated into a column the fused kernel reads.
  auto upper_mode = [] { return fn1(fid::upper, ref_mode(), t_str); };
  filter("B01_filter_upper_eq_and_add_gt", [=] {
    return conj_and(list(cmp(ct::equal, upper_mode(), make_str_const("MAIL")),
                         cmp(ct::gt,
                             bin(fid::add, make_cast(ref_int32(), t_big, false), ref_qty(), t_big),
                             make_bigint_const(500'000))));
  });
  project("B02_mixed_varchar_and_fixed", [=] {
    return list(upper_mode(),
                bin(fid::add, fn1(fid::strlen, upper_mode(), t_big), ref_qty(), t_big));
  });
  project("B03_boundary_under_varchar_case", [=] {
    std::vector<sirius::ast::case_expr::when_then> whens;
    whens.push_back({cmp(ct::equal, upper_mode(), make_str_const("MAIL")), ref_comment()});
    return list(
      std::make_unique<ast_node>(sirius::ast::case_expr{std::move(whens), ref_mode(), t_str}));
  });

  // String constants: every launch reads its kernel's string-constant bytes from the device.
  project("S01_str_consts_project", [] {
    return list(cmp(ct::equal, ref_mode(), make_str_const("MAIL")),
                cmp(ct::equal, ref_mode(), make_str_const("REG AIR")),
                bin(fid::contains, ref_comment(), make_str_const("special"), t_bool),
                bin(fid::prefix, ref_comment(), make_str_const("ab"), t_bool));
  });
  filter("S02_str_consts_filter", [] {
    return make_conj(sirius::ast::conjunction::kind::op_or,
                     list(cmp(ct::equal, ref_mode(), make_str_const("MAIL")),
                          cmp(ct::equal, ref_mode(), make_str_const("TRUCK")),
                          bin(fid::contains, ref_comment(), make_str_const("special"), t_bool),
                          bin(fid::suffix, ref_comment(), make_str_const("xyz"), t_bool)));
  });
}

//===----------------------------------------------------------------------===//
// Complexity ladders: one expression shape at increasing size.
//===----------------------------------------------------------------------===//

// A checked BIGINT chain of `depth` operations over two columns. The cycle `+ b, * 3, - a, + k`
// keeps a column operand in every other step, so no step folds into its neighbour, and grows the
// value by about 3x per cycle, far from overflow at depth 32.
node_ptr bigint_chain(int depth, column_def const& a, column_def const& b)
{
  using fid  = sirius::function_id;
  node_ptr e = ref(a);
  for (int i = 0; i < depth; ++i) {
    switch (i % 4) {
      case 0: e = bin(fid::add, std::move(e), ref(b), t_big); break;
      case 1: e = bin(fid::mul, std::move(e), make_bigint_const(3), t_big); break;
      case 2: e = bin(fid::sub, std::move(e), ref(a), t_big); break;
      default: e = bin(fid::add, std::move(e), make_bigint_const(7 + i), t_big); break;
    }
  }
  return e;
}

// A DOUBLE chain of `depth` operations with the cycle `/ d, + 1.5, * x, - d`.
node_ptr double_chain(int depth)
{
  using fid  = sirius::function_id;
  node_ptr e = ref_double();
  for (int i = 0; i < depth; ++i) {
    switch (i % 4) {
      case 0: e = bin(fid::div, std::move(e), ref(col::ld), t_dbl); break;
      case 1: e = bin(fid::add, std::move(e), make_double_const(1.5), t_dbl); break;
      case 2: e = bin(fid::mul, std::move(e), ref_double(), t_dbl); break;
      default: e = bin(fid::sub, std::move(e), ref(col::ld), t_dbl); break;
    }
  }
  return e;
}

// ps_supplycost * l_quantity -> DECIMAL(18,4).
node_ptr supply_cost()
{
  return bin(sirius::function_id::mul, ref(col::ps_supplycost), ref(col::l_quantity), t_dec184);
}

// TPC-H Q9 amount: both DECIMAL(18,4) products widen to DECIMAL(19,4), a 128-bit type.
node_ptr q9_amount()
{
  return bin(sirius::function_id::sub,
             cast_to(volume(), dec(19, 4)),
             cast_to(supply_cost(), dec(19, 4)),
             dec(19, 4));
}

void add_ladder_cases(case_list& cases)
{
  using fid = sirius::function_id;
  using ct  = sirius::comparison_type;

  for (int depth : {1, 2, 4, 8, 16, 32}) {
    cases.project("LA_arith_d" + std::to_string(depth),
                  [depth] { return list(bigint_chain(depth, col::la, col::lb)); });
  }
  for (int depth : {1, 4, 16}) {
    cases.project("LN_arith_null_d" + std::to_string(depth),
                  [depth] { return list(bigint_chain(depth, column_def{1, t_big}, col::ln)); });
  }

  // DECIMAL chains as DuckDB types them: 64-bit while every operand fits 18 digits, 128-bit after.
  cases.project("LD_dec_d1", [] {
    return list(bin(fid::sub, dec_const(100, 16, 2), ref(col::discount), t_dec162));
  });
  cases.project("LD_dec_d2", [] { return list(volume()); });
  cases.project("LD_dec_d4", [] { return list(q9_amount()); });

  for (int depth : {1, 4, 16}) {
    cases.project("LF_dbl_d" + std::to_string(depth),
                  [depth] { return list(double_chain(depth)); });
  }

  // N independent outputs `(la + k) * (k + 1) - lb`.
  for (int width : {1, 2, 4, 8, 16}) {
    cases.project("LW_width" + std::to_string(width), [width] {
      std::vector<node_ptr> outputs;
      for (int k = 1; k <= width; ++k) {
        outputs.push_back(bin(fid::sub,
                              bin(fid::mul,
                                  bin(fid::add, ref(col::la), make_bigint_const(k), t_big),
                                  make_bigint_const(k + 1),
                                  t_big),
                              ref(col::lb),
                              t_big));
      }
      return outputs;
    });
  }

  // AND of N comparisons over N columns, each true for 93% of rows.
  for (int n : {1, 2, 4, 8, 16}) {
    cases.filter("LC_conj" + std::to_string(n), [n] {
      std::vector<node_ptr> terms;
      for (int k = 0; k < n; ++k) {
        terms.push_back(cmp(ct::lt, ref(col::key(k)), make_int_const(930)));
      }
      return conj_and(std::move(terms));
    });
  }
  // OR of N comparisons over N columns, each true for 7% of rows.
  for (int n : {2, 4, 8}) {
    cases.filter("LO_disj" + std::to_string(n), [n] {
      std::vector<node_ptr> terms;
      for (int k = 0; k < n; ++k) {
        terms.push_back(cmp(ct::lt, ref(col::key(k)), make_int_const(70)));
      }
      return conj_or(std::move(terms));
    });
  }

  // AND whose first operand is true for 1% of rows, so most warps hold no row that needs the later
  // operands. None of the later operands can raise a SQL error or reads a string: three cheap
  // comparisons in one case, a date part, a DOUBLE chain and a 128-bit comparison in the other.
  cases.filter("LG_rare_and_cheap", [] {
    std::vector<node_ptr> terms;
    terms.push_back(cmp(ct::lt, ref(col::key(0)), make_int_const(10)));
    for (int k = 1; k < 4; ++k) {
      terms.push_back(cmp(ct::lt, ref(col::key(k)), make_int_const(930)));
    }
    return conj_and(std::move(terms));
  });
  cases.filter("LG_rare_and_costly", [] {
    return conj_and(list(cmp(ct::lt, ref(col::key(0)), make_int_const(10)),
                         cmp(ct::ge, year_of(col::o_orderdate), make_bigint_const(1993)),
                         cmp(ct::gt, double_chain(8), make_double_const(0.0)),
                         cmp(ct::gt, ref(col::sum_sales), ref(col::sum_sales2))));
  });

  // CASE with N arms, each taken by 10% of rows.
  for (int arms : {1, 2, 4, 8}) {
    cases.project("LK_case_arms" + std::to_string(arms), [arms] {
      std::vector<sirius::ast::case_expr::when_then> whens;
      for (int k = 0; k < arms; ++k) {
        whens.push_back({cmp(ct::lt, ref(col::key(0)), make_int_const(100 * (k + 1))),
                         bin(fid::add, ref(col::la), make_bigint_const(k + 1), t_big)});
      }
      return list(
        std::make_unique<ast_node>(sirius::ast::case_expr{std::move(whens), ref(col::lb), t_big}));
    });
  }

  // One predicate shape at increasing selectivity.
  struct selectivity {
    char const* label;
    std::int32_t bound;
  };
  for (auto const sel : {selectivity{"0001", 1},
                         selectivity{"01", 10},
                         selectivity{"10", 100},
                         selectivity{"50", 500},
                         selectivity{"90", 900},
                         selectivity{"100", 1000}}) {
    cases.filter(std::string("LX_sel_") + sel.label,
                 [sel] { return cmp(ct::lt, ref(col::key(0)), make_int_const(sel.bound)); });
  }
  for (auto const sel : {selectivity{"01", 10}, selectivity{"50", 500}, selectivity{"90", 900}}) {
    cases.filter(
      std::string("LX_wide_sel_") + sel.label,
      [sel] { return cmp(ct::lt, ref(col::key(0)), make_int_const(sel.bound)); },
      /*all_cols=*/true);
  }
}

//===----------------------------------------------------------------------===//
// TPC-H: the projection lists and predicates DuckDB plans for the queries, with its types.
// A projection case holds only computed expressions, since a projection passes plain column
// references through without evaluating them.
//===----------------------------------------------------------------------===//

void add_tpch_cases(case_list& cases)
{
  using fid         = sirius::function_id;
  using ct          = sirius::comparison_type;
  auto one_plus_tax = [] { return bin(fid::add, dec_const(100, 16, 2), ref(col::tax), t_dec162); };
  auto like         = [](column_def const& def, char const* pattern) {
    return bin(fid::like, ref(def), make_str_const(pattern), t_bool);
  };
  auto prefix = [](column_def const& def, char const* text) {
    return bin(fid::prefix, ref(def), make_str_const(text), t_bool);
  };

  cases.filter("H01_q1_filter",
               {col::l_returnflag, col::l_quantity, col::price, col::discount, col::tax},
               [] { return cmp(ct::le, ref(col::shipdate), date_const(1998, 9, 2)); });
  cases.project("H01_q1_project", [=] {
    return list(volume(), bin(fid::mul, volume(), one_plus_tax(), dec(18, 6)));
  });

  cases.filter("H02_q2_filter_part", [] {
    return conj_and(list(cmp(ct::equal, ref(col::p_size), make_int_const(15)),
                         bin(fid::suffix, ref(col::p_type), make_str_const("BRASS"), t_bool)));
  });

  cases.filter("H03_q3_filter_orders",
               [] { return cmp(ct::lt, ref(col::o_orderdate), date_const(1995, 3, 15)); });
  cases.filter("H03_q3_filter_customer", [] { return str_eq(col::c_mktsegment, "BUILDING"); });
  cases.project("H03_q3_revenue", [] { return list(volume()); });

  cases.filter("H04_q4_filter_lineitem",
               [] { return cmp(ct::lt, ref(col::l_commitdate), ref(col::l_receiptdate)); });
  cases.filter("H04_q4_filter_orders", [] {
    return conj_and(list(cmp(ct::ge, ref(col::o_orderdate), date_const(1993, 7, 1)),
                         cmp(ct::lt, ref(col::o_orderdate), date_const(1993, 10, 1))));
  });

  cases.filter("H06_q6_filter", {col::price, col::discount}, [] {
    return conj_and(list(cmp(ct::ge, ref(col::shipdate), date_const(1994, 1, 1)),
                         cmp(ct::lt, ref(col::shipdate), date_const(1995, 1, 1)),
                         cmp(ct::ge, ref(col::discount), dec_const_of(col::discount, 5)),
                         cmp(ct::le, ref(col::discount), dec_const_of(col::discount, 7)),
                         cmp(ct::lt, ref(col::l_quantity), dec_const_of(col::l_quantity, 2400))));
  });
  cases.project("H06_q6_revenue",
                [] { return list(bin(fid::mul, ref(col::price), ref(col::discount), t_dec184)); });

  cases.filter("H07_q7_filter_nation", [] {
    return conj_or(list(str_eq(col::n_name, "FRANCE"), str_eq(col::n_name, "GERMANY")));
  });
  cases.filter("H07_q7_filter_lineitem", {col::price, col::discount, col::shipdate}, [] {
    return conj_and(list(cmp(ct::ge, ref(col::shipdate), date_const(1995, 1, 1)),
                         cmp(ct::le, ref(col::shipdate), date_const(1996, 12, 31))));
  });
  cases.project("H07_q7_project", [] { return list(year_of(col::shipdate), volume()); });

  cases.project("H08_q8_project", [] {
    return list(year_of(col::o_orderdate),
                case_when(str_eq(col::n_name, "BRAZIL"), volume(), dec_const(0, 18, 4), t_dec184),
                volume());
  });

  cases.filter("H09_q9_filter_part", [] {
    return bin(fid::contains, ref(col::p_name), make_str_const("green"), t_bool);
  });
  cases.project("H09_q9_project", [] { return list(year_of(col::o_orderdate), q9_amount()); });

  cases.filter("H10_q10_filter_lineitem", {col::price, col::discount}, [] {
    return str_eq(col::l_returnflag, "R");
  });

  // ps_supplycost * ps_availqty: the BIGINT operand makes both sides DECIMAL(34,x).
  cases.project("H11_q11_value", [] {
    return list(bin(fid::mul,
                    cast_to(ref(col::ps_supplycost), dec(34, 2)),
                    cast_to(ref(col::ps_availqty), dec(34, 0)),
                    dec(34, 2)));
  });

  cases.filter("H12_q12_filter", {col::shipmode}, [] {
    return conj_and(list(str_in(ref(col::shipmode), {"MAIL", "SHIP"}),
                         cmp(ct::lt, ref(col::l_commitdate), ref(col::l_receiptdate)),
                         cmp(ct::lt, ref(col::shipdate), ref(col::l_commitdate)),
                         cmp(ct::ge, ref(col::l_receiptdate), date_const(1994, 1, 1)),
                         cmp(ct::lt, ref(col::l_receiptdate), date_const(1995, 1, 1))));
  });
  // The filter DuckDB leaves above the scan once the date ranges are pushed into it.
  cases.filter("H12_q12_filter_residual", {col::shipmode}, [] {
    return conj_and(
      list(cmp(ct::lt, ref(col::l_commitdate), ref(col::l_receiptdate)),
           cmp(ct::lt, ref(col::shipdate), ref(col::l_commitdate)),
           conj_or(list(str_eq(col::shipmode, "MAIL"), str_eq(col::shipmode, "SHIP")))));
  });
  cases.project("H12_q12_case", [] {
    auto const differs = [](char const* text) {
      return cmp(ct::not_equal, ref(col::o_orderpriority), make_str_const(text));
    };
    return list(case_when(conj_or(list(str_eq(col::o_orderpriority, "1-URGENT"),
                                       str_eq(col::o_orderpriority, "2-HIGH"))),
                          make_int_const(1),
                          make_int_const(0),
                          t_int),
                case_when(conj_and(list(differs("1-URGENT"), differs("2-HIGH"))),
                          make_int_const(1),
                          make_int_const(0),
                          t_int));
  });

  cases.filter("H13_q13_filter", [] {
    return bin(fid::not_like, ref(col::o_comment), make_str_const("%special%requests%"), t_bool);
  });

  cases.project("H14_q14_project", [=] {
    return list(case_when(prefix(col::p_type, "PROMO"), volume(), dec_const(0, 18, 4), t_dec184),
                volume());
  });

  cases.filter("H16_q16_filter_part", [=] {
    return conj_and(list(cmp(ct::not_equal, ref(col::p_brand), make_str_const("Brand#45")),
                         logical_not(prefix(col::p_type, "MEDIUM POLISHED")),
                         int_in(ref(col::p_size), {49, 14, 23, 45, 19, 3, 36, 9})));
  });
  cases.filter("H16_q16_filter_supplier",
               [=] { return like(col::o_comment, "%Customer%Complaints%"); });

  cases.filter("H17_q17_filter_part", [] {
    return conj_and(list(str_eq(col::p_brand, "Brand#23"), str_eq(col::p_container, "MED BOX")));
  });
  // l_quantity < 0.2 * avg(l_quantity): the correlated average arrives as a DOUBLE column.
  cases.filter("H17_q17_filter_quantity", {col::price}, [] {
    return cmp(ct::lt, to_double(ref(col::l_quantity)), ref(col::ld));
  });

  cases.filter("H18_q18_having", [] {
    return cmp(ct::gt, ref(col::sum_sales), dec_const_of(col::sum_sales, 30'000));
  });

  auto q19_branch = [](char const* brand,
                       std::initializer_list<char const*> containers,
                       std::int64_t quantity,
                       std::int32_t max_size) {
    return conj_and(
      list(str_eq(col::p_brand, brand),
           str_in(ref(col::p_container), containers),
           cmp(ct::ge, ref(col::l_quantity), dec_const_of(col::l_quantity, quantity * 100)),
           cmp(ct::le, ref(col::l_quantity), dec_const_of(col::l_quantity, (quantity + 10) * 100)),
           make_between(ref(col::p_size), make_int_const(1), make_int_const(max_size), true, true),
           str_in(ref(col::shipmode), {"AIR", "REG AIR"}),
           str_eq(col::l_shipinstruct, "DELIVER IN PERSON")));
  };
  cases.filter("H19_q19_filter", {col::price, col::discount}, [=] {
    return conj_or(
      list(q19_branch("Brand#12", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}, 1, 5),
           q19_branch("Brand#23", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}, 10, 10),
           q19_branch("Brand#34", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}, 20, 15)));
  });
  // The parts of Q19 DuckDB pushes below the join: one filter per table.
  cases.filter("H19_q19_filter_lineitem", {col::price, col::discount, col::l_quantity}, [] {
    return conj_and(
      list(conj_or(list(dec_range(col::l_quantity, 100, 1100),
                        dec_range(col::l_quantity, 1000, 2000),
                        dec_range(col::l_quantity, 2000, 3000))),
           conj_or(list(str_eq(col::shipmode, "AIR"), str_eq(col::shipmode, "REG AIR"))),
           str_eq(col::l_shipinstruct, "DELIVER IN PERSON")));
  });
  auto q19_part =
    [](char const* brand, std::initializer_list<char const*> containers, std::int32_t max_size) {
      std::vector<node_ptr> any_container;
      for (auto const* container : containers) {
        any_container.push_back(str_eq(col::p_container, container));
      }
      return conj_and(list(cmp(ct::le, ref(col::p_size), make_int_const(max_size)),
                           cmp(ct::ge, ref(col::p_size), make_int_const(1)),
                           str_eq(col::p_brand, brand),
                           conj_or(std::move(any_container))));
    };
  cases.filter("H19_q19_filter_part", {col::p_brand, col::p_size, col::p_container}, [=] {
    return conj_or(list(q19_part("Brand#12", {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}, 5),
                        q19_part("Brand#23", {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}, 10),
                        q19_part("Brand#34", {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}, 15)));
  });

  // p_name LIKE 'forest%', which DuckDB plans as a string range.
  cases.filter("H20_q20_filter_part", [] {
    return conj_and(list(cmp(ct::ge, ref(col::p_name), make_str_const("forest")),
                         cmp(ct::lt, ref(col::p_name), make_str_const("foresu"))));
  });
  // 0.5 * sum(l_quantity) -> DECIMAL(38,3).
  cases.project("H20_q20_half_sum", [] {
    return list(bin(fid::mul, dec_const(5, 38, 1), ref(col::sum_sales), dec(38, 3)));
  });

  cases.filter("H21_q21_filter_lineitem",
               [] { return cmp(ct::gt, ref(col::l_receiptdate), ref(col::l_commitdate)); });

  auto country_code = [] {
    return make_func(
      fid::substring, list(ref(col::c_phone), make_bigint_const(1), make_bigint_const(2)), t_str);
  };
  cases.filter("H22_q22_filter_phone", {col::c_phone, col::c_acctbal}, [=] {
    return str_in(country_code(), {"13", "31", "23", "29", "30", "18", "17"});
  });
  cases.filter("H22_q22_filter_acctbal", {col::c_acctbal}, [] {
    return cmp(ct::gt, ref(col::c_acctbal), dec_const_of(col::c_acctbal, 0));
  });
  cases.project("H22_q22_cntrycode", [=] { return list(country_code()); });
}

//===----------------------------------------------------------------------===//
// TPC-DS: representative projection lists and predicates over nullable fact columns, with
// DuckDB's types. Each case is named after one query; its comment names the other queries with
// the same shape.
//===----------------------------------------------------------------------===//

void add_tpcds_cases(case_list& cases)
{
  using fid   = sirius::function_id;
  using ct    = sirius::comparison_type;
  auto int_eq = [](column_def const& def, std::int32_t value) {
    return cmp(ct::equal, ref(def), make_int_const(value));
  };
  auto coalesce_zero = [](node_ptr value, logical_type type) {
    node_ptr zero = type.id() == type_id::DECIMAL
                      ? dec_const(0, type.decimal_precision(), type.decimal_scale())
                      : make_int_const(0);
    return make_coalesce(list(std::move(value), std::move(zero)), type);
  };
  // CASE WHEN avg > 0.0 THEN abs(sum - avg) / avg ELSE NULL END > 0.1 (Q47, Q53, Q57, Q63, Q89).
  auto deviation_exceeds_tenth = [] {
    auto deviation = bin(
      fid::div,
      fn1(
        fid::abs, bin(fid::sub, to_double(ref(col::sum_sales)), ref(col::avg_sales), t_dbl), t_dbl),
      ref(col::avg_sales),
      t_dbl);
    return cmp(ct::gt,
               case_when(cmp(ct::gt, ref(col::avg_sales), make_double_const(0.0)),
                         std::move(deviation),
                         make_null_const(t_dbl),
                         t_dbl),
               make_double_const(0.1));
  };
  auto zip5 = [] {
    return make_func(
      fid::substring, list(ref(col::ca_zip), make_bigint_const(1), make_bigint_const(5)), t_str);
  };
  auto zip_in = [=](int count) {
    std::vector<node_ptr> values;
    for (int k = 0; k < count; ++k) {
      values.push_back(make_str_const(std::to_string(zip_prefix(k * 7))));
    }
    return make_in(zip5(), std::move(values), /*negated=*/false);
  };
  std::initializer_list<column_def> const fact_columns{
    col::ss_customer_sk, col::ss_quantity, col::ss_ext_sales_price, col::ss_wholesale_cost};

  // Q2: round(sum1 / sum2, 2) over weekly sales sums.
  cases.project("DS02_q2_ratio_round", [] {
    auto ratio = [](column_def const& num, column_def const& den) {
      return make_func(
        fid::round,
        list(bin(fid::div, to_double(ref(num)), to_double(ref(den)), t_dbl), make_int_const(2)),
        t_dbl);
    };
    return list(ratio(col::sum_sales, col::sum_sales2), ratio(col::sum_sales2, col::sum_sales));
  });

  // Q5: sales - coalesce(returns, 0) over DECIMAL(38,2) sums.
  cases.project("DS05_q5_coalesce_sub", [=] {
    return list(
      bin(fid::sub, ref(col::sum_sales), coalesce_zero(ref(col::sum_sales2), t_dec382), t_dec382));
  });

  // Q7 (also Q26, Q27): string equalities on customer demographics.
  cases.filter("DS07_q7_filter_demographics", [] {
    return conj_and(list(str_eq(col::cd_gender, "M"),
                         str_eq(col::cd_marital_status, "S"),
                         str_eq(col::cd_education_status, "College")));
  });

  // Q8: substr(ca_zip, 1, 5) IN (...) and the substr(ca_zip, 1, 2) join key.
  cases.filter("DS08_q8_zip_in20", [=] { return zip_in(20); });
  cases.project("DS08_q8_zip_prefix", [] {
    return list(make_func(
      fid::substring, list(ref(col::ca_zip), make_bigint_const(1), make_bigint_const(2)), t_str));
  });

  // Q9: quantity bucket predicates, and a CASE per bucket.
  cases.filter(
    "DS09_q9_filter_bucket", fact_columns, [] { return int_range(col::ss_quantity, 1, 20); });
  cases.project("DS09_q9_bucket_case", [] {
    std::vector<node_ptr> outputs;
    for (int bucket = 0; bucket < 5; ++bucket) {
      outputs.push_back(case_when(int_range(col::ss_quantity, 20 * bucket + 1, 20 * bucket + 20),
                                  ref(col::ss_coupon_amt),
                                  ref(col::ss_net_paid),
                                  t_dec72));
    }
    return outputs;
  });

  // Q13 (also Q48, Q85): disjunctive demographic filters.
  cases.filter("DS13_q13_filter_scan", fact_columns, [] {
    return conj_and(list(conj_or(list(dec_range(col::ss_sales_price, 10'000, 15'000),
                                      dec_range(col::ss_sales_price, 5'000, 10'000),
                                      dec_range(col::ss_sales_price, 15'000, 20'000))),
                         conj_or(list(dec_range(col::ss_net_profit, 10'000, 20'000),
                                      dec_range(col::ss_net_profit, 15'000, 30'000),
                                      dec_range(col::ss_net_profit, 5'000, 25'000)))));
  });
  cases.filter("DS13_q13_filter_demographics", fact_columns, [=] {
    auto branch =
      [=](
        std::int32_t dep_count, std::int64_t price_lo, char const* marital, char const* education) {
        return conj_and(list(int_eq(col::hd_dep_count, dep_count),
                             dec_range(col::ss_sales_price, price_lo, price_lo + 5'000),
                             str_eq(col::cd_marital_status, marital),
                             str_eq(col::cd_education_status, education)));
      };
    return conj_or(list(branch(3, 10'000, "M", "Advanced Degree"),
                        branch(1, 5'000, "S", "College"),
                        branch(1, 15'000, "W", "2 yr Degree")));
  });
  cases.filter("DS13_q13_filter_address", fact_columns, [] {
    auto branch = [](std::int64_t profit_lo,
                     std::int64_t profit_hi,
                     std::initializer_list<char const*> state_list) {
      return conj_and(list(dec_range(col::ss_net_profit, profit_lo, profit_hi),
                           str_in(ref(col::ca_state), state_list)));
    };
    return conj_or(list(branch(10'000, 20'000, {"TX", "OH", "CO"}),
                        branch(15'000, 30'000, {"OR", "NM", "KY"}),
                        branch(5'000, 25'000, {"VA", "CA", "MS"})));
  });

  // Q21: CASE WHEN d_date < c THEN quantity ELSE 0 END, before and after.
  cases.project("DS21_q21_case", [] {
    return list(case_when(cmp(ct::lt, ref(col::shipdate), date_const(1995, 3, 11)),
                          ref(col::ss_quantity),
                          make_int_const(0),
                          t_int),
                case_when(cmp(ct::ge, ref(col::shipdate), date_const(1995, 3, 11)),
                          ref(col::ss_quantity),
                          make_int_const(0),
                          t_int));
  });

  // Q28: a quantity bucket and an OR of three price ranges.
  cases.filter("DS28_q28_filter", fact_columns, [] {
    return conj_and(list(int_range(col::ss_quantity, 0, 5),
                         conj_or(list(dec_range(col::ss_list_price, 800, 1'800),
                                      dec_range(col::ss_coupon_amt, 45'900, 145'900),
                                      dec_range(col::ss_wholesale_cost, 5'700, 7'700)))));
  });

  // Q34 (also Q73): household filter with a guarded ratio, date_dim filter, HAVING range.
  cases.filter("DS34_q34_filter_household", [] {
    auto ratio = bin(
      fid::div,
      to_double(bin(
        fid::mul, cast_to(ref(col::hd_dep_count), dec(14, 0)), dec_const(1000, 14, 3), dec(14, 3))),
      to_double(ref(col::hd_vehicle_count)),
      t_dbl);
    return conj_and(list(conj_or(list(str_eq(col::hd_buy_potential, ">10000"),
                                      str_eq(col::hd_buy_potential, "Unknown"))),
                         cmp(ct::gt,
                             case_when(cmp(ct::gt, ref(col::hd_vehicle_count), make_int_const(0)),
                                       std::move(ratio),
                                       make_null_const(t_dbl),
                                       t_dbl),
                             make_double_const(1.2))));
  });
  cases.filter("DS34_q34_filter_date", [] {
    return conj_and(list(conj_or(list(int_range(col::d_dom, 1, 3), int_range(col::d_dom, 25, 28))),
                         int_range(col::d_year, 1999, 2001)));
  });
  cases.filter("DS34_q34_having", [] {
    return make_between(ref(col::la), make_bigint_const(15), make_bigint_const(20), true, true);
  });

  // Q36 (also Q70, Q86): gross margin as a ratio of two sums.
  cases.project("DS36_q36_margin", [] {
    return list(
      bin(fid::div, to_double(ref(col::sum_sales)), to_double(ref(col::sum_sales2)), t_dbl));
  });

  // Q40: CASE on a date over price - coalesce(refund, 0).
  cases.project("DS40_q40_case", [=] {
    auto net = [=] {
      return bin(fid::sub,
                 cast_to(ref(col::ss_sales_price), dec(13, 2)),
                 coalesce_zero(cast_to(ref(col::sr_return_amt), dec(12, 2)), dec(12, 2)),
                 dec(13, 2));
    };
    return list(case_when(cmp(ct::lt, ref(col::shipdate), date_const(1995, 3, 11)),
                          net(),
                          dec_const(0, 13, 2),
                          dec(13, 2)),
                case_when(cmp(ct::ge, ref(col::shipdate), date_const(1995, 3, 11)),
                          net(),
                          dec_const(0, 13, 2),
                          dec(13, 2)));
  });

  // Q43 (also Q2, Q59): one CASE per day name, NULL otherwise.
  cases.project("DS43_q43_day_case", [] {
    std::vector<node_ptr> outputs;
    for (auto const day : day_names) {
      outputs.push_back(case_when(str_eq(col::d_day_name, std::string(day).c_str()),
                                  ref(col::ss_sales_price),
                                  make_null_const(t_dec72),
                                  t_dec72));
    }
    return outputs;
  });

  // Q45: substr(ca_zip, 1, 5) IN (nine zip codes).
  cases.filter("DS45_q45_zip_in9", [=] { return zip_in(9); });

  // Q47 (also Q57): date_dim filter, deviation filter, and the deviation projection.
  cases.filter("DS47_q47_filter_date", [=] {
    return conj_or(list(int_eq(col::d_year, 1999),
                        conj_and(list(int_eq(col::d_year, 1998), int_eq(col::d_moy, 12))),
                        conj_and(list(int_eq(col::d_year, 2000), int_eq(col::d_moy, 1)))));
  });
  cases.filter("DS47_q47_filter_deviation", [=] {
    return conj_and(list(deviation_exceeds_tenth(),
                         int_eq(col::d_year, 1999),
                         cmp(ct::gt, ref(col::avg_sales), make_double_const(0.0))));
  });
  cases.project("DS47_q47_difference", [] {
    return list(bin(fid::sub, to_double(ref(col::sum_sales)), ref(col::avg_sales), t_dbl));
  });

  // Q49: the coalesce(x, 0) arguments of its sums.
  cases.project("DS49_q49_coalesce", [=] {
    return list(coalesce_zero(ref(col::sr_return_quantity), t_int),
                coalesce_zero(ref(col::ss_quantity), t_int),
                coalesce_zero(cast_to(ref(col::sr_return_amt), dec(12, 2)), dec(12, 2)),
                coalesce_zero(cast_to(ref(col::ss_net_paid), dec(12, 2)), dec(12, 2)));
  });

  // Q53 (also Q63): two branches of three string IN lists, and the deviation filter.
  cases.filter("DS53_q53_filter_item", [] {
    auto branch = [](std::initializer_list<char const*> categories,
                     std::initializer_list<char const*> classes,
                     std::initializer_list<char const*> brands) {
      return conj_and(list(str_in(ref(col::i_category), categories),
                           str_in(ref(col::i_class), classes),
                           str_in(ref(col::i_brand), brands)));
    };
    return conj_or(list(
      branch({"Books", "Children", "Electronics"},
             {"personal", "portable", "reference", "self-help"},
             {"scholaramalgamalg #14",
              "scholaramalgamalg #7",
              "exportiunivamalg #9",
              "scholaramalgamalg #9"}),
      branch({"Women", "Music", "Men"},
             {"accessories", "classical", "fragrances", "pants"},
             {"amalgimporto #1", "edu packscholar #1", "exportiimporto #1", "importoamalg #1"})));
  });
  cases.filter("DS53_q53_filter_deviation", [=] { return deviation_exceeds_tenth(); });

  // Q62 (also Q99): the ship delay, and one CASE per delay bucket.
  auto ship_delay = [] {
    return bin(fid::sub, ref(col::ws_ship_date_sk), ref(col::ws_sold_date_sk), t_int);
  };
  cases.project("DS62_q62_delay", [=] { return list(ship_delay()); });
  cases.project("DS62_q62_bucket_case", [=] {
    auto flag = [](node_ptr when) {
      return case_when(std::move(when), make_int_const(1), make_int_const(0), t_int);
    };
    auto between = [=](std::int32_t above, std::int32_t up_to) {
      return conj_and(list(cmp(ct::gt, ship_delay(), make_int_const(above)),
                           cmp(ct::le, ship_delay(), make_int_const(up_to))));
    };
    return list(flag(cmp(ct::le, ship_delay(), make_int_const(30))),
                flag(between(30, 60)),
                flag(between(60, 90)),
                flag(between(90, 120)),
                flag(cmp(ct::gt, ship_delay(), make_int_const(120))));
  });

  // Q66: the twelve month flags DuckDB projects, and the twelve CASE sums they select.
  cases.project("DS66_q66_month_flags", [=] {
    std::vector<node_ptr> outputs;
    for (int month = 1; month <= 12; ++month) {
      outputs.push_back(int_eq(col::d_moy, month));
    }
    return outputs;
  });
  cases.project("DS66_q66_month_case", [=] {
    std::vector<node_ptr> outputs;
    for (int month = 1; month <= 12; ++month) {
      outputs.push_back(case_when(int_eq(col::d_moy, month),
                                  bin(fid::mul,
                                      cast_to(ref(col::ss_ext_sales_price), dec(17, 2)),
                                      cast_to(ref(col::ss_quantity), dec(17, 0)),
                                      dec(17, 2)),
                                  dec_const(0, 17, 2),
                                  dec(17, 2)));
    }
    return outputs;
  });

  // Q75: sales net of returns, per row.
  cases.project("DS75_q75_net", [=] {
    return list(
      bin(
        fid::sub, ref(col::ss_quantity), coalesce_zero(ref(col::sr_return_quantity), t_int), t_int),
      bin(fid::sub,
          ref(col::ss_ext_sales_price),
          coalesce_zero(ref(col::sr_return_amt), t_dec72),
          dec(8, 2)));
  });

  // Q78: coalesce(x, 0) in a filter and in a sum of two channels.
  cases.filter("DS78_q78_filter", fact_columns, [=] {
    auto positive = [=](column_def const& def) {
      return cmp(ct::gt, coalesce_zero(ref(def), t_int), make_int_const(0));
    };
    return conj_or(list(positive(col::sr_return_quantity), positive(col::ss_quantity)));
  });
  cases.project("DS78_q78_coalesce_add", [=] {
    return list(bin(fid::add,
                    coalesce_zero(ref(col::ss_quantity), t_int),
                    coalesce_zero(ref(col::sr_return_quantity), t_int),
                    t_int));
  });

  // Q88 (also Q96): household and time-of-day filters.
  cases.filter("DS88_q88_filter_household", [=] {
    auto branch = [=](std::int32_t dep_count, std::int32_t max_vehicles) {
      return conj_and(list(int_eq(col::hd_dep_count, dep_count),
                           cmp(ct::le, ref(col::hd_vehicle_count), make_int_const(max_vehicles))));
    };
    return conj_or(list(branch(4, 6), branch(2, 4), branch(0, 2)));
  });
  cases.filter("DS88_q88_filter_time", [=] {
    return conj_and(
      list(int_eq(col::t_hour, 8), cmp(ct::ge, ref(col::t_minute), make_int_const(30))));
  });
}

std::vector<bench_case> all_cases()
{
  case_list cases;
  add_basic_cases(cases);
  add_ladder_cases(cases);
  add_tpch_cases(cases);
  add_tpcds_cases(cases);
  return cases.release();
}

// Applies the SIRIUS_EXPR_BENCH_CASES selectors as the file header defines them.
std::vector<bool> select_cases(std::vector<bench_case> const& cases,
                               std::vector<std::string> const& selectors)
{
  std::vector<bool> selected(cases.size(), selectors.empty());
  for (auto const& selector : selectors) {
    using matcher = bool (*)(std::string const&, std::string const&);
    constexpr std::array<matcher, 3> matchers{
      [](std::string const& name, std::string const& s) { return name == s; },
      [](std::string const& name, std::string const& s) { return name.starts_with(s); },
      [](std::string const& name, std::string const& s) {
        return name.find(s) != std::string::npos;
      }};
    bool found = false;
    for (auto const matches : matchers) {
      for (std::size_t i = 0; i < cases.size(); ++i) {
        if (matches(cases[i].name, selector)) { selected[i] = found = true; }
      }
      if (found) { break; }
    }
    if (!found) { std::printf("[expr_bench] selector '%s' matches no case\n", selector.c_str()); }
  }
  return selected;
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

//===----------------------------------------------------------------------===//
// Measurement
//===----------------------------------------------------------------------===//

struct bench_config {
  int iters{20};
  double min_ms{200};
  int warmup{3};
  double warmup_min_ms{300};
  int cold_reps{1};
  bool verify{true};
  bool fuse_multiply_add{false};
};

// One cold call: its wall time and what the JIT pipeline spent inside it.
struct cold_call {
  double total_ms{0};
  double create_ms{0};
  double compile_ms{0};
  double cubin_ms{0};
  double load_ms{0};
  std::uint64_t kernels{0};
  std::uint64_t source_bytes{0};

  // Translation, launch and execution: the cold call outside the JIT pipeline stages.
  [[nodiscard]] double other_ms() const
  {
    return total_ms - create_ms - compile_ms - cubin_ms - load_ms;
  }
};

struct measurement {
  cold_call cold;  // the cold call of median wall time
  double cold_min_ms{0};
  int cold_reps{0};
  double median_ms{0};
  double min_ms{0};
  double return_median_ms{0};
  double return_min_ms{0};
  int iters{0};
  cudf::size_type rows_out{0};
  std::uint64_t checksum{0};
};

double now_ms()
{
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

double median_of(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

cold_call cold_call_between(codegen::jit::CompileStats const& before,
                            codegen::jit::CompileStats const& after,
                            double total_ms)
{
  auto const ms = [](std::uint64_t from, std::uint64_t to) {
    return static_cast<double>(to - from) / 1000.0;
  };
  return {total_ms,
          ms(before.create_program_us, after.create_program_us),
          ms(before.compile_program_us, after.compile_program_us),
          ms(before.get_cubin_us, after.get_cubin_us),
          ms(before.load_us, after.load_us),
          after.compiles - before.compiles,
          after.source_bytes - before.source_bytes};
}

measurement run_case(bench_case const& bc,
                     bench_input const& in,
                     strategy_t strategy,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr,
                     bench_config const& config)
{
  auto exprs = bc.build();
  duckdb::vector<std::unique_ptr<ast_node>> owned;
  for (auto& e : exprs) {
    owned.push_back(std::move(e));
  }
  auto make_evaluator = [&] {
    return std::make_unique<sirius::expression_evaluator>(
      owned,
      mr,
      ::cuda::stream_ref{stream.value()},
      sirius::expression_query_policy{.strategy                    = strategy,
                                      .nvrtc_jit_fuse_multiply_add = config.fuse_multiply_add},
      sirius::expression_evaluator::default_min_ast_size,
      /*like_swar_fastpath=*/true);
  };
  auto evaluate = [&](sirius::expression_evaluator& exec) -> std::unique_ptr<cudf::table> {
    return bc.kind == case_kind::project ? exec.evaluate(in.view) : exec.select(in.view, bc.gather);
  };

  measurement m;
  nvtx3::scoped_range case_range{bc.name + "/" + std::string(sirius::strategy_to_string(strategy))};
  // Only NVRTC_JIT can compile again: every repetition starts from a new evaluator and an empty
  // kernel cache.
  bool const recompiles = strategy == strategy_t::NVRTC_JIT;
  m.cold_reps           = recompiles ? config.cold_reps : 1;
  std::unique_ptr<sirius::expression_evaluator> exec;
  std::vector<cold_call> cold_calls;
  for (int rep = 0; rep < m.cold_reps; ++rep) {
    nvtx3::scoped_range cold_range{"cold"};
    exec = make_evaluator();
    // The process-wide kernel cache would otherwise serve a kernel compiled for an earlier case
    // or row count with the same generated source. Safe here because every call synchronizes the
    // stream, so no launch from the previous case still uses a cached kernel.
    if (recompiles) { codegen::jit::KernelCache::instance().clear(); }
    auto const before = codegen::jit::compile_stats();
    double const t0   = now_ms();
    auto out          = evaluate(*exec);
    stream.synchronize();
    cold_calls.push_back(cold_call_between(before, codegen::jit::compile_stats(), now_ms() - t0));
    m.rows_out = out->num_rows();
    if (config.verify && rep + 1 == m.cold_reps) {
      m.checksum = checksum_table(out->view(), stream);
    }
  }
  std::sort(cold_calls.begin(), cold_calls.end(), [](cold_call const& a, cold_call const& b) {
    return a.total_ms < b.total_ms;
  });
  m.cold        = cold_calls[cold_calls.size() / 2];
  m.cold_min_ms = cold_calls.front().total_ms;

  // The GPU idles at a low clock between cases (cold compiles leave it idle for most of a
  // second), so warm up for a minimum wall time as well as a minimum number of calls.
  {
    double const warm_start = now_ms();
    for (int i = 0; i < config.warmup || now_ms() - warm_start < config.warmup_min_ms; ++i) {
      auto out = evaluate(*exec);
      stream.synchronize();
    }
  }
  // A call of tens of microseconds needs thousands of samples for a stable median, so the loop
  // also runs for a minimum wall time; the cap bounds the sample vectors.
  int const max_iters = std::max(config.iters, 100'000);
  std::vector<double> total_ms;
  std::vector<double> return_ms;
  double const loop_start = now_ms();
  for (int i = 0; i < max_iters && (i < config.iters || now_ms() - loop_start < config.min_ms);
       ++i) {
    nvtx3::scoped_range iter_range{"iter"};
    double const t0 = now_ms();
    auto out        = evaluate(*exec);
    double const t1 = now_ms();
    stream.synchronize();
    total_ms.push_back(now_ms() - t0);
    return_ms.push_back(t1 - t0);
  }
  m.iters            = static_cast<int>(total_ms.size());
  m.min_ms           = *std::min_element(total_ms.begin(), total_ms.end());
  m.median_ms        = median_of(std::move(total_ms));
  m.return_min_ms    = *std::min_element(return_ms.begin(), return_ms.end());
  m.return_median_ms = median_of(std::move(return_ms));
  return m;
}

// How NVRTC_JIT plans the case over `input`; see `nvrtc_path` in the file header.
std::string nvrtc_path(bench_case const& bc, cudf::table_view const& input)
{
  auto const exprs = bc.build();
  std::vector<sirius::ast::node const*> roots;
  for (auto const& e : exprs) {
    roots.push_back(e.get());
  }
  auto const plan = bc.kind == case_kind::project
                      ? sirius::nvrtc::plan_project_sink(roots, input)
                      : sirius::nvrtc::plan_filter_sink(*roots.front(), input);
  if (!plan) { return "materialize"; }
  std::string path = plan->roots.size() == roots.size() ? "fused" : "partial";
  if (!plan->is_direct()) { path += "+boundary"; }
  return path;
}

// Marks in `referenced` the input columns the case's expressions reference.
void mark_referenced_columns(bench_case const& bc, std::vector<bool>& referenced)
{
  auto const exprs = bc.build();
  for (auto const& e : exprs) {
    sirius::ast::visit_references(*e, [&](sirius::ast::reference const& r) {
      if (r.column_index < referenced.size()) { referenced[r.column_index] = true; }
    });
  }
}

// The input columns the selected cases read: the ones their expressions reference and the ones
// their filters gather.
std::vector<bool> used_columns(std::vector<bench_case> const& cases,
                               std::vector<bool> const& selected)
{
  std::vector<bool> used(col::count, false);
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (!selected[i]) { continue; }
    mark_referenced_columns(cases[i], used);
    for (auto const c : cases[i].gather) {
      used[static_cast<std::size_t>(c)] = true;
    }
  }
  return used;
}

// The device bytes of the input columns the case's expressions reference.
std::size_t referenced_bytes(bench_case const& bc, bench_input const& in)
{
  std::vector<bool> referenced(in.column_bytes.size(), false);
  mark_referenced_columns(bc, referenced);
  std::size_t bytes = 0;
  for (std::size_t c = 0; c < referenced.size(); ++c) {
    if (referenced[c]) { bytes += in.column_bytes[c]; }
  }
  return bytes;
}

// The first line of `text` without the characters that would break a CSV row.
std::string csv_field(std::string text)
{
  text = text.substr(0, text.find('\n'));
  std::replace(text.begin(), text.end(), ',', ';');
  std::replace(text.begin(), text.end(), '"', '\'');
  return text;
}

// The name a measurement is reported under. NVRTC_JIT with fused multiply-add has a name of its
// own, so that its rows are never compared as ordinary `nvrtc_jit` ones.
std::string strategy_label(strategy_t s, bench_config const& config)
{
  std::string label{sirius::strategy_to_string(s)};
  if (s == strategy_t::NVRTC_JIT && config.fuse_multiply_add) { label += "+fma"; }
  return label;
}

}  // namespace

TEST_CASE("bench expression evaluator strategies", "[!benchmark][expr_bench]")
{
  // The on-disk JIT caches would hide compile time in the cold measurement: a kernel compiled for
  // one row count would be read back from disk for the next.
  setenv("SIMPATICO_JIT_CACHE_DIR", "off", /*overwrite=*/1);
  bool const compute_cache_off = env_or("CUDA_CACHE_DISABLE", "0") == "1";
  if (!compute_cache_off) {
    std::printf(
      "[expr_bench] the CUDA compute cache is enabled: an NVRTC_JIT cold call of a source "
      "compiled before on this machine skips most of the compile; start the benchmark "
      "with CUDA_CACHE_DISABLE=1 to time full compiles\n");
  }
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
  auto const selectors = split_csv(env_or("SIRIUS_EXPR_BENCH_CASES", ""));
  auto const csv_path  = env_or("SIRIUS_EXPR_BENCH_CSV", "");
  bench_config config;
  config.iters             = std::stoi(env_or("SIRIUS_EXPR_BENCH_ITERS", "20"));
  config.min_ms            = std::stod(env_or("SIRIUS_EXPR_BENCH_MIN_MS", "200"));
  config.warmup            = std::stoi(env_or("SIRIUS_EXPR_BENCH_WARMUP", "3"));
  config.warmup_min_ms     = std::stod(env_or("SIRIUS_EXPR_BENCH_WARMUP_MS", "300"));
  config.cold_reps         = std::max(1, std::stoi(env_or("SIRIUS_EXPR_BENCH_COLD_REPS", "1")));
  config.verify            = env_or("SIRIUS_EXPR_BENCH_VERIFY", "1") != "0";
  config.fuse_multiply_add = env_or("SIRIUS_EXPR_BENCH_FUSE_MULTIPLY_ADD", "0") == "1";

  std::vector<strategy_t> strategies;
  for (auto const& s : strategy_list) {
    strategy_t parsed;
    REQUIRE(sirius::string_to_strategy(s, parsed));
    strategies.push_back(parsed);
  }

  rmm::cuda_stream stream;
  rmm::mr::cuda_async_memory_resource mr;
  rmm::device_async_resource_ref mr_ref{mr};
  // cuDF allocates its internal temporaries from the current device resource. The engine runs on
  // a pool; leaving the default here would time raw cudaMalloc/cudaFree in every cuDF call.
  struct current_resource_guard {
    explicit current_resource_guard(rmm::device_async_resource_ref resource)
      : previous(cudf::set_current_device_resource(
          ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
    {
    }
    ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(previous)); }
    ::cuda::mr::any_resource<::cuda::mr::device_accessible> previous;
  } const pool_as_current_resource{mr_ref};

  std::ofstream csv;
  if (!csv_path.empty()) {
    bool const exists = std::ifstream(csv_path).good();
    csv.open(csv_path, std::ios::app);
    if (!exists) {
      csv << "case,kind,rows,strategy,cold_ms,median_ms,min_ms,input_gbps,rows_out,checksum,match,"
             "status,cold_min_ms,cold_reps,cold_create_ms,cold_compile_ms,cold_cubin_ms,"
             "cold_load_ms,cold_other_ms,kernels,src_bytes,ret_ms,ret_min_ms,iters,outputs,"
             "gather_cols,nvrtc_ran,nvrtc_path,cuda_cache,detail\n";
    }
  }

  // Under nsys the CUDA trace buffers are flushed on cudaProfilerStop(); the test binary exits
  // without running static destructors, so flush explicitly before leaving the test case.
  struct profiler_stop {
    ~profiler_stop() { (void)cudaProfilerStop(); }
  } const stop_profiler_on_exit;
  (void)cudaProfilerStart();

  auto const cases    = all_cases();
  auto const selected = select_cases(cases, selectors);
  auto const used     = used_columns(cases, selected);
  for (auto const& rows_str : row_list) {
    auto const rows = static_cast<cudf::size_type>(std::stoll(rows_str));
    auto input      = make_bench_input(rows, used, stream, mr_ref);
    std::printf("[expr_bench] rows=%d input_columns=%d uploaded=%td fuse_multiply_add=%d\n",
                rows,
                input.view.num_columns(),
                std::count(used.begin(), used.end(), true),
                config.fuse_multiply_add ? 1 : 0);

    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
      if (!selected[case_index]) { continue; }
      auto const& bc = cases[case_index];
      // Bytes touched: referenced input columns + output. Approximated as the referenced inputs
      // for projections (outputs are small) and reported only as a relative throughput figure.
      std::size_t const bytes_in = referenced_bytes(bc, input);
      std::size_t const outputs  = bc.build().size();

      // The first listed strategy is the reference.
      std::optional<std::uint64_t> reference_checksum;
      for (auto const strategy : strategies) {
        std::string const path =
          strategy == strategy_t::NVRTC_JIT ? nvrtc_path(bc, input.view) : std::string("-");
        std::string const label = strategy_label(strategy, config);
        measurement m;
        std::string error;
        try {
          m = run_case(bc, input, strategy, stream, mr_ref, config);
        } catch (std::exception const& e) {
          error = e.what();
          std::printf("[expr_bench] case=%-30s rows=%9d strategy=%-13s path=%s error: %s\n",
                      bc.name.c_str(),
                      rows,
                      label.c_str(),
                      path.c_str(),
                      csv_field(error).c_str());
        }
        bool const failed = !error.empty();
        std::string match = "-";
        if (config.verify && !failed) {
          if (strategy == strategies.front()) {
            reference_checksum = m.checksum;
            match              = "ref";
          } else if (!reference_checksum) {
            match = "noref";
          } else {
            match = (*reference_checksum == m.checksum) ? "match" : "MISMATCH";
          }
        }
        double const gbps = failed ? 0.0 : (bytes_in / 1e9) / (m.median_ms / 1e3);
        if (!failed) {
          std::printf(
            "[expr_bench] case=%-30s rows=%9d strategy=%-13s cold_ms=%9.3f med_ms=%8.3f "
            "min_ms=%8.3f input_gbps=%7.1f rows_out=%9d chk=%016llx %-8s ret_ms=%8.3f iters=%6d "
            "cold[min=%.3f compile=%.3f load=%.3f other=%.3f kernels=%llu src=%llu] path=%s\n",
            bc.name.c_str(),
            rows,
            label.c_str(),
            m.cold.total_ms,
            m.median_ms,
            m.min_ms,
            gbps,
            m.rows_out,
            static_cast<unsigned long long>(m.checksum),
            match.c_str(),
            m.return_median_ms,
            m.iters,
            m.cold_min_ms,
            m.cold.create_ms + m.cold.compile_ms + m.cold.cubin_ms,
            m.cold.load_ms,
            m.cold.other_ms(),
            static_cast<unsigned long long>(m.cold.kernels),
            static_cast<unsigned long long>(m.cold.source_bytes),
            path.c_str());
        }
        if (csv.is_open()) {
          csv << bc.name << ',' << static_cast<int>(bc.kind) << ',' << rows << ',' << label << ','
              << m.cold.total_ms << ',' << m.median_ms << ',' << m.min_ms << ',' << gbps << ','
              << m.rows_out << ',' << std::hex << m.checksum << std::dec << ',' << match << ','
              << (failed ? "error" : "ok") << ',' << m.cold_min_ms << ',' << m.cold_reps << ','
              << m.cold.create_ms << ',' << m.cold.compile_ms << ',' << m.cold.cubin_ms << ','
              << m.cold.load_ms << ',' << (failed ? 0.0 : m.cold.other_ms()) << ','
              << m.cold.kernels << ',' << m.cold.source_bytes << ',' << m.return_median_ms << ','
              << m.return_min_ms << ',' << m.iters << ',' << outputs << ',' << bc.gather.size()
              << ',' << (m.cold.kernels > 0 ? 1 : 0) << ',' << path << ','
              << (compute_cache_off ? "off" : "on") << ',' << csv_field(error) << '\n';
          csv.flush();
        }
        std::fflush(stdout);
      }
    }
  }
}
