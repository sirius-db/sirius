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

// Draw kernels of `sirius_generate`. Every value is a pure function of
// (seed, batch_index, row, column_index, salt) through detail::counter_hash, so no state is
// carried from one row or one batch to the next. The value semantics are documented in
// generator_spec.hpp; the details that table leaves open are fixed here:
//
// - uniform int64: `lo + mulhi64(hash, hi - lo + 1)` (the hash itself when the range is all of
//   int64), which is overflow-free for any `lo <= hi`.
// - zipf: round 0 of the rejection loop draws from salt VALUE_A and round r > 0 from salt
//   `zipf_round_salt + r`, which no other stream uses. After `zipf_max_rounds` rejections the
//   rank is 1.
// - double parameters that mean integers (bounds, start, step, D, N, cardinality) are rounded to
//   the nearest int64 and saturate at its limits.

#include "op/scan/generator/generator_detail.cuh"

#include <cudf/null_mask.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan::generator {

namespace {

/// Rejection rounds of the zipf sampler before it gives up and returns rank 1. The acceptance
/// probability of one round is well above 1/2, so the fallback is never reached in practice.
constexpr std::uint32_t zipf_max_rounds = 64;

/// Salt of zipf rejection round `r > 0` is `zipf_round_salt + r`: above every `STR_CHAR_BASE + i`.
constexpr std::uint32_t zipf_round_salt = 0x80000000U;

constexpr double two_pi = 6.283185307179586476925286766559;

/// Rounds `value` to the nearest int64, saturating at the int64 limits.
std::int64_t to_int64(double value, const column_spec& column, const char* what)
{
  if (!std::isfinite(value)) {
    throw std::invalid_argument("sirius_generate: column '" + column.name + "': " + what +
                                " must be finite");
  }
  const double rounded = std::round(value);
  if (rounded >= 0x1.0p63) { return std::numeric_limits<std::int64_t>::max(); }
  if (rounded < -0x1.0p63) { return std::numeric_limits<std::int64_t>::min(); }
  return static_cast<std::int64_t>(rounded);
}

// Rejection-inversion zipf sampler (Hormann & Derflinger 1996), following Apache Commons'
// RejectionInversionZipfSampler. `h(x) = x^-s`, `h_integral` is its antiderivative.

/// `log1p(x) / x`, continuous at 0.
CUDF_HOST_DEVICE inline double zipf_helper1(double x) noexcept
{
  return std::abs(x) > 1e-8 ? std::log1p(x) / x : 1.0 - x * (0.5 - x * (1.0 / 3.0 - 0.25 * x));
}

/// `expm1(x) / x`, continuous at 0.
CUDF_HOST_DEVICE inline double zipf_helper2(double x) noexcept
{
  return std::abs(x) > 1e-8 ? std::expm1(x) / x
                            : 1.0 + x * 0.5 * (1.0 + x * (1.0 / 3.0) * (1.0 + 0.25 * x));
}

CUDF_HOST_DEVICE inline double zipf_h(double x, double s) noexcept
{
  return std::exp(-s * std::log(x));
}

CUDF_HOST_DEVICE inline double zipf_h_integral(double x, double s) noexcept
{
  const double log_x = std::log(x);
  return zipf_helper2((1.0 - s) * log_x) * log_x;
}

CUDF_HOST_DEVICE inline double zipf_h_integral_inverse(double x, double s) noexcept
{
  double t = x * (1.0 - s);
  if (t < -1.0) { t = -1.0; }  // only reachable through rounding
  return std::exp(zipf_helper1(t) * x);
}

/// Everything a draw kernel needs, by value. Which fields are meaningful depends on `dist`.
struct draw_params {
  std::uint64_t seed{0};
  std::uint64_t batch{0};
  std::uint64_t first_row{0};  // global row of row 0: batch * rows_per_batch
  std::uint32_t column{0};
  distribution dist{distribution::uniform};
  double p1{0};                // uniform lo, normal mean, sequential start (double draws)
  double p2{0};                // uniform hi, normal stddev, sequential step (double draws)
  std::int64_t lo{0};          // uniform lo / sequential start (int draws)
  std::uint64_t range{0};      // uniform: hi - lo + 1, 0 when that is 2^64
  std::int64_t step{1};        // sequential step (int draws)
  std::int64_t distinct{1};    // few_distinct D / zipf N
  std::int64_t clamp{0};       // cardinality clamp C, 0 = none (int draws only)
  std::int64_t clamp_base{0};  // lo' of the clamp
  double zipf_s{0};
  double zipf_h_x1{0};       // h_integral(1.5) - 1
  double zipf_h_n{0};        // h_integral(N + 0.5)
  double zipf_threshold{0};  // 2 - h_integral_inverse(h_integral(2.5) - h(2))

  [[nodiscard]] CUDF_HOST_DEVICE std::uint64_t hash(std::uint64_t row,
                                                    std::uint32_t salt) const noexcept
  {
    return detail::counter_hash(seed, batch, row, column, salt);
  }
};

/// Standard normal deviate of `row` (Box-Muller over salts VALUE_A and VALUE_B).
__device__ inline double standard_normal(const draw_params& p, std::uint64_t row) noexcept
{
  double u1 = detail::to_unit(p.hash(row, detail::VALUE_A));
  if (u1 == 0.0) { u1 = 0x1.0p-53; }
  const double u2 = detail::to_unit(p.hash(row, detail::VALUE_B));
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
}

/// llround that saturates at the int64 limits (NaN gives 0).
__device__ inline std::int64_t saturating_llround(double value) noexcept
{
  if (value != value) { return 0; }
  if (value >= 0x1.0p63) { return std::numeric_limits<std::int64_t>::max(); }
  if (value <= -0x1.0p63) { return std::numeric_limits<std::int64_t>::min(); }
  return std::llround(value);
}

/// Zipf rank of `row` in [1, N].
__device__ inline std::int64_t zipf_rank(const draw_params& p, std::uint64_t row) noexcept
{
  const auto n = static_cast<double>(p.distinct);
  for (std::uint32_t round = 0; round < zipf_max_rounds; ++round) {
    const std::uint32_t salt = round == 0 ? detail::VALUE_A : zipf_round_salt + round;
    const double u = p.zipf_h_n + detail::to_unit(p.hash(row, salt)) * (p.zipf_h_x1 - p.zipf_h_n);
    const double x = zipf_h_integral_inverse(u, p.zipf_s);
    double k       = std::floor(x + 0.5);
    if (!(k >= 1.0)) {
      k = 1.0;
    } else if (k > n) {
      k = n;
    }
    if (k - x <= p.zipf_threshold ||
        u >= zipf_h_integral(k + 0.5, p.zipf_s) - zipf_h(k, p.zipf_s)) {
      const auto rank = static_cast<std::int64_t>(k);
      return rank > p.distinct ? p.distinct : rank;  // N may not be exact as a double
    }
  }
  return 1;
}

/// `base + ((value - base) mod c + c) mod c` without intermediate overflow (the final sum wraps).
__device__ inline std::int64_t clamp_to_cardinality(std::int64_t value,
                                                    std::int64_t base,
                                                    std::int64_t c) noexcept
{
  std::int64_t a = value % c;
  if (a < 0) { a += c; }
  std::int64_t b = base % c;
  if (b < 0) { b += c; }
  const std::int64_t m = a >= b ? a - b : a - b + c;
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(base) +
                                   static_cast<std::uint64_t>(m));
}

/// The int64 draw of one row (INTEGER, BIGINT and VARCHAR columns).
struct int_draw_fn {
  draw_params p;

  __device__ std::int64_t operator()(std::int64_t index) const noexcept
  {
    const auto row = static_cast<std::uint64_t>(index);
    std::int64_t value{0};
    switch (p.dist) {
      case distribution::uniform: {
        const std::uint64_t h      = p.hash(row, detail::VALUE_A);
        const std::uint64_t offset = p.range == 0 ? h : __umul64hi(h, p.range);
        value = static_cast<std::int64_t>(static_cast<std::uint64_t>(p.lo) + offset);
        break;
      }
      case distribution::normal:
        value = saturating_llround(p.p1 + p.p2 * standard_normal(p, row));
        break;
      case distribution::zipf: return zipf_rank(p, row);
      case distribution::few_distinct:
        return static_cast<std::int64_t>(p.hash(row, detail::VALUE_A) %
                                         static_cast<std::uint64_t>(p.distinct));
      case distribution::sequential:
        value = static_cast<std::int64_t>(static_cast<std::uint64_t>(p.lo) +
                                          static_cast<std::uint64_t>(p.step) * (p.first_row + row));
        break;
      case distribution::sequential_desc:
        value = static_cast<std::int64_t>(static_cast<std::uint64_t>(p.lo) -
                                          static_cast<std::uint64_t>(p.step) * (p.first_row + row));
        break;
    }
    return p.clamp > 0 ? clamp_to_cardinality(value, p.clamp_base, p.clamp) : value;
  }
};

/// The double draw of one row (DOUBLE columns).
struct double_draw_fn {
  draw_params p;

  __device__ double operator()(std::int64_t index) const noexcept
  {
    const auto row = static_cast<std::uint64_t>(index);
    switch (p.dist) {
      case distribution::uniform: {
        const double value = p.p1 + detail::to_unit(p.hash(row, detail::VALUE_A)) * (p.p2 - p.p1);
        // Rounding can land on hi; keep the interval half-open.
        return value >= p.p2 && p.p2 > p.p1 ? std::nextafter(p.p2, p.p1) : value;
      }
      case distribution::normal: return p.p1 + p.p2 * standard_normal(p, row);
      case distribution::zipf: return static_cast<double>(zipf_rank(p, row));
      case distribution::few_distinct:
        return static_cast<double>(p.hash(row, detail::VALUE_A) %
                                   static_cast<std::uint64_t>(p.distinct));
      case distribution::sequential: return p.p1 + p.p2 * static_cast<double>(p.first_row + row);
      case distribution::sequential_desc:
        return p.p1 - p.p2 * static_cast<double>(p.first_row + row);
    }
    return 0.0;
  }
};

/// One word of the validity mask: bit `b` is set iff row `32 * word + b` exists and is not null.
struct validity_word_fn {
  draw_params p;
  double null_fraction;
  std::int64_t rows;

  __device__ cudf::bitmask_type operator()(std::int64_t word) const noexcept
  {
    constexpr std::int64_t bits_per_word = 8 * sizeof(cudf::bitmask_type);
    cudf::bitmask_type bits{0};
    for (std::int64_t bit = 0; bit < bits_per_word; ++bit) {
      const std::int64_t row = word * bits_per_word + bit;
      if (row < rows && !(detail::to_unit(p.hash(static_cast<std::uint64_t>(row),
                                                 detail::NULL_FLAG)) < null_fraction)) {
        bits |= cudf::bitmask_type{1} << bit;
      }
    }
    return bits;
  }
};

struct narrow_fn {
  __device__ std::int32_t operator()(std::int64_t value) const noexcept
  {
    return static_cast<std::int32_t>(static_cast<std::uint32_t>(value));
  }
};

cudf::size_type checked_rows(const generator_spec& spec)
{
  if (spec.rows_per_batch >
      static_cast<std::uint64_t>(std::numeric_limits<cudf::size_type>::max())) {
    throw std::invalid_argument("sirius_generate: rows_per_batch " +
                                std::to_string(spec.rows_per_batch) +
                                " exceeds the largest cudf column");
  }
  return static_cast<cudf::size_type>(spec.rows_per_batch);
}

/// Validates the column and turns it into kernel parameters.
draw_params make_params(const generator_spec& spec,
                        std::size_t column_index,
                        std::uint64_t batch_index)
{
  const column_spec& column = spec.columns[column_index];
  const bool as_double      = column.type == value_type::float64;
  auto fail                 = [&](const std::string& what) {
    throw std::invalid_argument("sirius_generate: column '" + column.name + "': " + what);
  };

  draw_params p;
  p.seed      = spec.seed;
  p.batch     = batch_index;
  p.first_row = batch_index * spec.rows_per_batch;
  p.column    = static_cast<std::uint32_t>(column_index);
  p.dist      = column.dist;
  p.p1        = column.p1;
  p.p2        = column.p2;

  // The cardinality clamp applies to the int64 draws of uniform / normal / sequential* only.
  auto set_clamp = [&](std::int64_t base) {
    if (as_double || !(column.cardinality > 0)) { return; }
    p.clamp      = to_int64(column.cardinality, column, "cardinality");
    p.clamp_base = base;
  };

  switch (column.dist) {
    case distribution::uniform: {
      if (!std::isfinite(column.p1) || !std::isfinite(column.p2) || column.p2 < column.p1) {
        fail("uniform needs finite bounds with lo <= hi");
      }
      if (!as_double) {
        p.lo          = to_int64(column.p1, column, "lo");
        const auto hi = to_int64(column.p2, column, "hi");
        p.range       = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(p.lo) + 1;
        set_clamp(p.lo);
      }
      break;
    }
    case distribution::normal: {
      if (!std::isfinite(column.p1) || !std::isfinite(column.p2)) {
        fail("normal needs a finite mean and stddev");
      }
      set_clamp(0);
      break;
    }
    case distribution::zipf: {
      if (!std::isfinite(column.p1) || !(column.p1 > 0)) { fail("zipf needs an exponent s > 0"); }
      if (!(column.cardinality >= 1)) { fail("zipf needs cardinality >= 1"); }
      p.distinct       = to_int64(column.cardinality, column, "cardinality");
      p.zipf_s         = column.p1;
      p.zipf_h_x1      = zipf_h_integral(1.5, p.zipf_s) - 1.0;
      p.zipf_h_n       = zipf_h_integral(static_cast<double>(p.distinct) + 0.5, p.zipf_s);
      p.zipf_threshold = 2.0 - zipf_h_integral_inverse(
                                 zipf_h_integral(2.5, p.zipf_s) - zipf_h(2.0, p.zipf_s), p.zipf_s);
      break;
    }
    case distribution::few_distinct: {
      if (!(column.p1 >= 1)) { fail("few_distinct needs D >= 1"); }
      p.distinct = to_int64(column.p1, column, "D");
      break;
    }
    case distribution::sequential:
    case distribution::sequential_desc: {
      if (!std::isfinite(column.p1) || !std::isfinite(column.p2)) {
        fail("sequential needs a finite start and step");
      }
      if (column.p2 == 0) { p.p2 = 1; }
      if (!as_double) {
        p.lo   = to_int64(column.p1, column, "start");
        p.step = to_int64(p.p2, column, "step");
        if (p.step == 0) { p.step = 1; }  // a step that rounds to 0
        set_clamp(p.lo);
      }
      break;
    }
    default: fail("unknown distribution");
  }
  return p;
}

}  // namespace

namespace detail {

drawn_column draw_column(const generator_spec& spec,
                         std::size_t column_index,
                         std::uint64_t batch_index,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  if (column_index >= spec.columns.size()) {
    throw std::out_of_range("sirius_generate: column " + std::to_string(column_index) +
                            " is out of range (" + std::to_string(spec.columns.size()) +
                            " columns)");
  }
  const cudf::size_type rows = checked_rows(spec);
  const column_spec& column  = spec.columns[column_index];
  const draw_params params   = make_params(spec, column_index, batch_index);
  const bool as_double       = column.type == value_type::float64;
  const auto policy          = rmm::exec_policy_nosync(stream, mr);

  drawn_column out{rmm::device_uvector<std::int64_t>(as_double ? 0 : rows, stream, mr),
                   rmm::device_uvector<double>(as_double ? rows : 0, stream, mr),
                   rmm::device_buffer{0, stream, mr},
                   0};
  const thrust::counting_iterator<std::int64_t> first_row{0};
  if (as_double) {
    thrust::transform(
      policy, first_row, first_row + rows, out.doubles.begin(), double_draw_fn{params});
  } else {
    thrust::transform(policy, first_row, first_row + rows, out.ints.begin(), int_draw_fn{params});
  }

  if (column.null_fraction > 0 && rows > 0) {
    out.null_mask    = cudf::create_null_mask(rows, cudf::mask_state::UNINITIALIZED, stream, mr);
    auto* const bits = static_cast<cudf::bitmask_type*>(out.null_mask.data());
    // Every word of the (padded) allocation is written; bits past `rows` are 0.
    const auto words = static_cast<std::int64_t>(out.null_mask.size() / sizeof(cudf::bitmask_type));
    thrust::transform(policy,
                      first_row,
                      first_row + words,
                      bits,
                      validity_word_fn{params, column.null_fraction, rows});
    out.null_count = cudf::null_count(bits, 0, rows, stream);
  }
  return out;
}

}  // namespace detail

std::unique_ptr<cudf::table> generate_batch(const generator_spec& spec,
                                            std::uint64_t batch_index,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  if (batch_index >= spec.num_batches) {
    throw std::out_of_range("sirius_generate: batch " + std::to_string(batch_index) +
                            " is out of range (" + std::to_string(spec.num_batches) + " batches)");
  }
  const cudf::size_type rows = checked_rows(spec);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.reserve(spec.columns.size());
  for (std::size_t i = 0; i < spec.columns.size(); ++i) {
    const column_spec& column = spec.columns[i];
    auto drawn                = detail::draw_column(spec, i, batch_index, stream, mr);
    switch (column.type) {
      case value_type::int32: {
        rmm::device_uvector<std::int32_t> narrowed(rows, stream, mr);
        thrust::transform(rmm::exec_policy_nosync(stream, mr),
                          drawn.ints.begin(),
                          drawn.ints.end(),
                          narrowed.begin(),
                          narrow_fn{});
        columns.push_back(std::make_unique<cudf::column>(
          std::move(narrowed), std::move(drawn.null_mask), drawn.null_count));
        break;
      }
      case value_type::int64:
        columns.push_back(std::make_unique<cudf::column>(
          std::move(drawn.ints), std::move(drawn.null_mask), drawn.null_count));
        break;
      case value_type::float64:
        columns.push_back(std::make_unique<cudf::column>(
          std::move(drawn.doubles), std::move(drawn.null_mask), drawn.null_count));
        break;
      case value_type::varchar:
        // format_varchar clamps negative keys to 0 itself.
        columns.push_back(detail::format_varchar(drawn.ints,
                                                 std::move(drawn.null_mask),
                                                 drawn.null_count,
                                                 column,
                                                 spec.seed,
                                                 batch_index,
                                                 static_cast<std::uint32_t>(i),
                                                 stream,
                                                 mr));
        break;
    }
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

}  // namespace sirius::op::scan::generator
