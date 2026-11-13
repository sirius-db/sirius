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

// Differential tests: windowed_merger with the real bounded_kway_merge and real window buffers
// over runs in the GPU, HOST and DISK tiers, against a full sort of the same rows.

#include "catch.hpp"
#include "op/merge/windowed_merge.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/data/row_range_read.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <data/sirius_converter_registry.hpp>
#include <data/sirius_row_range_reader_registry.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace sirius::op::merge;
using namespace sirius::test::merge_utils;
using cucascade::memory::Tier;
using sirius::test::operator_utils::get_resource_ref;

namespace {

using rng_t = std::mt19937_64;

constexpr std::size_t kUnlimitedOutput = std::numeric_limits<std::size_t>::max();

/// Uniform in [0, n); plain modulo so a seed gives the same case on every platform.
std::size_t pick(rng_t& rng, std::size_t n) { return static_cast<std::size_t>(rng() % n); }

// ---------------------------------------------------------------------------
// Memory spaces: GPU, HOST and DISK
// ---------------------------------------------------------------------------

/**
 * @brief A memory manager with a 256 MiB GPU space, a host space and a disk space in a temporary
 * directory, the registries the merger reads through, and the stream the merge runs on.
 *
 * tiered_memory_fixture has no DISK tier, so this mirrors it with a disk mounting point. Declare
 * it before anything that holds batches of its spaces.
 */
struct differential_fixture {
  std::filesystem::path spill_dir;
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu  = nullptr;
  cucascade::memory::memory_space* host = nullptr;
  cucascade::memory::memory_space* disk = nullptr;
  rmm::cuda_stream stream;

  differential_fixture()
  {
    sirius::converter_registry::reset_for_testing();
    std::string tmp_template =
      (std::filesystem::temp_directory_path() / "sirius_merge_differential_XXXXXX").string();
    REQUIRE(mkdtemp(tmp_template.data()) != nullptr);
    spill_dir = tmp_template;

    cucascade::memory::reservation_manager_configurator builder;
    builder.set_number_of_gpus(1)
      .set_gpu_usage_limit(std::size_t{256} << 20)
      .set_reservation_fraction_per_gpu(0.75)
      .set_per_numa_region_capacity(std::size_t{1} << 30)
      .use_gpu_id_as_host_id()
      .set_reservation_fraction_per_numa_region(0.75)
      .set_disk_mounting_point(0, std::size_t{1} << 30, spill_dir.string());
    manager = std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    sirius::converter_registry::initialize();
    // Left initialized: a live SiriusContext serves its readers from this registry.
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }
    gpu  = space_of(Tier::GPU);
    host = space_of(Tier::HOST);
    disk = space_of(Tier::DISK);
    REQUIRE(gpu != nullptr);
    REQUIRE(host != nullptr);
    REQUIRE(disk != nullptr);
  }

  ~differential_fixture()
  {
    if (manager) {
      manager->shutdown();
      sirius::converter_registry::shutdown();
    }
    std::error_code ignored;
    std::filesystem::remove_all(spill_dir, ignored);
  }

  [[nodiscard]] rmm::cuda_stream_view stream_view() const
  {
    return rmm::cuda_stream_view{stream.value()};
  }

 private:
  cucascade::memory::memory_space* space_of(Tier tier)
  {
    auto spaces = manager->get_memory_spaces_for_tier(tier);
    return spaces.empty() ? nullptr : const_cast<cucascade::memory::memory_space*>(spaces[0]);
  }
};

const char* tier_name(Tier tier)
{
  switch (tier) {
    case Tier::GPU: return "GPU";
    case Tier::HOST: return "HOST";
    case Tier::DISK: return "DISK";
    default: return "?";
  }
}

Tier tier_of(const cucascade::data_batch& batch)
{
  return batch.to_read_only().get_memory_space()->get_tier();
}

// ---------------------------------------------------------------------------
// Columns from host values
// ---------------------------------------------------------------------------

/// The null mask of @p valid (true == valid) and its null count; no mask when @p valid is empty.
std::pair<rmm::device_buffer, cudf::size_type> make_mask(const std::vector<bool>& valid)
{
  if (valid.empty()) { return {rmm::device_buffer{}, 0}; }
  std::vector<cudf::bitmask_type> words(
    cudf::bitmask_allocation_size_bytes(static_cast<cudf::size_type>(valid.size())) /
      sizeof(cudf::bitmask_type),
    0);
  cudf::size_type nulls = 0;
  for (std::size_t i = 0; i < valid.size(); ++i) {
    if (valid[i]) {
      words[i / 32] |= cudf::bitmask_type{1} << (i % 32);
    } else {
      ++nulls;
    }
  }
  return {rmm::device_buffer(words.data(),
                             words.size() * sizeof(cudf::bitmask_type),
                             cudf::get_default_stream(),
                             cudf::get_current_device_resource_ref()),
          nulls};
}

/// A fixed-width column of @p type; a non-empty @p valid makes it nullable.
template <typename T>
std::unique_ptr<cudf::column> fixed_column(cudf::data_type type,
                                           const std::vector<T>& values,
                                           const std::vector<bool>& valid = {})
{
  rmm::device_buffer data(values.data(),
                          values.size() * sizeof(T),
                          cudf::get_default_stream(),
                          cudf::get_current_device_resource_ref());
  auto [mask, nulls] = make_mask(valid);
  return std::make_unique<cudf::column>(
    type, static_cast<cudf::size_type>(values.size()), std::move(data), std::move(mask), nulls);
}

std::unique_ptr<cudf::column> int64_column(const std::vector<std::int64_t>& values,
                                           const std::vector<bool>& valid = {})
{
  return fixed_column(cudf::data_type{cudf::type_id::INT64}, values, valid);
}

/// A STRING column; a non-empty @p valid makes it nullable (null rows keep their bytes).
std::unique_ptr<cudf::column> string_column(const std::vector<std::string>& values,
                                            const std::vector<bool>& valid = {})
{
  auto col = sirius::test::operator_utils::make_string_column(
    values, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  if (!valid.empty()) {
    auto [mask, nulls] = make_mask(valid);
    col->set_null_mask(std::move(mask), nulls);
  }
  return col;
}

/// Row validity with about @p null_percent % nulls; empty (not nullable) when it is negative.
std::vector<bool> draw_validity(std::size_t rows, int null_percent, rng_t& rng)
{
  std::vector<bool> valid;
  if (null_percent < 0) { return valid; }
  valid.resize(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    valid[i] = static_cast<int>(pick(rng, 100)) >= null_percent;
  }
  return valid;
}

// ---------------------------------------------------------------------------
// Random column generators
// ---------------------------------------------------------------------------

/// Builds one column of a run: (rows, rng) -> column. The same factory builds the column of every
/// run of a case, so the runs share its value pool and tie with each other.
using column_factory = std::function<std::unique_ptr<cudf::column>(std::size_t, rng_t&)>;

/// Values drawn from a small pool (heavy ties), or fresh ones when the pool is empty.
template <typename T>
struct value_source {
  std::vector<T> pool;
  std::function<T(rng_t&)> fresh;

  T operator()(rng_t& rng) const
  {
    return pool.empty() ? fresh(rng) : pool[pick(rng, pool.size())];
  }
};

/// A source of @p cardinality distinct values (0: unbounded), preferring the special values.
template <typename T>
value_source<T> make_source(std::vector<T> specials,
                            std::function<T(rng_t&)> fresh,
                            std::size_t cardinality,
                            rng_t& rng)
{
  value_source<T> source;
  if (cardinality > 0) {
    for (std::size_t i = specials.size(); i > 1; --i) {
      std::swap(specials[i - 1], specials[pick(rng, i)]);
    }
    specials.resize(std::min(specials.size(), cardinality));
    source.pool = std::move(specials);
    while (source.pool.size() < cardinality) {
      source.pool.push_back(fresh(rng));
    }
  } else {
    // One value in eight is a special one, so they also meet the unbounded values.
    source.pool.clear();
    auto shared = std::make_shared<std::vector<T>>(std::move(specials));
    fresh       = [shared, fresh](rng_t& g) {
      return pick(g, 8) == 0 ? (*shared)[pick(g, shared->size())] : fresh(g);
    };
  }
  source.fresh = std::move(fresh);
  return source;
}

template <typename T>
column_factory fixed_factory(cudf::data_type type, value_source<T> source, int null_percent)
{
  return [type, source = std::move(source), null_percent](std::size_t rows, rng_t& rng) {
    std::vector<T> values(rows);
    for (auto& value : values) {
      value = source(rng);
    }
    return fixed_column(type, values, draw_validity(rows, null_percent, rng));
  };
}

column_factory string_factory(value_source<std::string> source, int null_percent)
{
  return [source = std::move(source), null_percent](std::size_t rows, rng_t& rng) {
    std::vector<std::string> values(rows);
    for (auto& value : values) {
      value = source(rng);
    }
    return string_column(values, draw_validity(rows, null_percent, rng));
  };
}

/// Short strings over {a, b, c, é}: many shared prefixes, and bytes above 0x7F.
std::string random_string(rng_t& rng, std::size_t max_tokens)
{
  static const std::vector<std::string> tokens{"a", "b", "c", "\xc3\xa9"};
  std::string out;
  for (auto n = pick(rng, max_tokens + 1); n > 0; --n) {
    out += tokens[pick(rng, tokens.size())];
  }
  return out;
}

const cudf::data_type kDecimalType{cudf::type_id::DECIMAL64, -2};

/// The data type the generator of @p id produces.
cudf::data_type type_of(cudf::type_id id)
{
  return id == cudf::type_id::DECIMAL64 ? kDecimalType : cudf::data_type{id};
}

/// A generator of @p id columns: INT32, INT64, FLOAT64 (NaN, ±0, ±inf), STRING or DECIMAL64.
column_factory random_factory(cudf::type_id id,
                              std::size_t cardinality,
                              int null_percent,
                              rng_t& rng)
{
  using i32 = std::int32_t;
  using i64 = std::int64_t;
  switch (id) {
    case cudf::type_id::INT32:
      return fixed_factory<i32>(
        type_of(id),
        make_source<i32>(
          {std::numeric_limits<i32>::min(), std::numeric_limits<i32>::max(), -1, 0, 1},
          [](rng_t& g) { return static_cast<i32>(g()); },
          cardinality,
          rng),
        null_percent);
    case cudf::type_id::INT64:
      return fixed_factory<i64>(
        type_of(id),
        make_source<i64>(
          {std::numeric_limits<i64>::min(), std::numeric_limits<i64>::max(), -1, 0, 1},
          [](rng_t& g) { return static_cast<i64>(g()); },
          cardinality,
          rng),
        null_percent);
    case cudf::type_id::DECIMAL64:
      return fixed_factory<i64>(
        type_of(id),
        make_source<i64>(
          {std::numeric_limits<i64>::min(), std::numeric_limits<i64>::max(), -100, 0, 99},
          [](rng_t& g) { return static_cast<i64>(g() % 2'000'001) - 1'000'000; },
          cardinality,
          rng),
        null_percent);
    case cudf::type_id::FLOAT64:
      return fixed_factory<double>(
        type_of(id),
        make_source<double>(
          {std::numeric_limits<double>::quiet_NaN(),
           -0.0,
           0.0,
           std::numeric_limits<double>::infinity(),
           -std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::denorm_min(),
           -1.5},
          [](rng_t& g) { return (static_cast<double>(g() % 2'000'001) - 1'000'000.0) / 8.0; },
          cardinality,
          rng),
        null_percent);
    case cudf::type_id::STRING:
      return string_factory(make_source<std::string>(
                              {"",
                               "a",
                               "aa",
                               "ab",
                               "b",
                               "\xc3\xa9",
                               "\xc3\xa9"
                               "e",
                               "\x7f",
                               std::string(40, 'a'),
                               std::string(40, 'a') + "b"},
                              [](rng_t& g) { return random_string(g, 20); },
                              cardinality,
                              rng),
                            null_percent);
    default: throw std::invalid_argument("random_factory: unsupported type");
  }
}

const char* type_name(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    default: return "?";
  }
}

// ---------------------------------------------------------------------------
// A merge case: layout, runs, merge, comparison with the oracle
// ---------------------------------------------------------------------------

/// What the merger fetches, and where those columns are in the source batches.
struct case_layout {
  std::vector<cudf::data_type> types;    ///< fetched columns, in fetch order
  merge_key_spec keys;                   ///< key_cols index the fetched columns
  cudf::size_type id_col{0};             ///< fetched INT64 column holding a unique row id
  std::vector<std::size_t> fetch_cols;   ///< source column of each fetched column
  std::vector<bool> source_junk_string;  ///< per source column: an unfetched one is a STRING

  [[nodiscard]] std::size_t source_columns() const { return source_junk_string.size(); }
};

/// The source batches hold exactly the fetched columns, in fetch order.
void set_identity_source(case_layout& layout)
{
  layout.fetch_cols.resize(layout.types.size());
  for (std::size_t i = 0; i < layout.fetch_cols.size(); ++i) {
    layout.fetch_cols[i] = i;
  }
  layout.source_junk_string.assign(layout.types.size(), false);
}

/// The source batches hold the fetched columns in a random order among 0-2 unfetched columns.
void set_random_source(case_layout& layout, rng_t& rng)
{
  const auto source_cols = layout.types.size() + pick(rng, 3);
  std::vector<std::size_t> positions(source_cols);
  for (std::size_t i = 0; i < source_cols; ++i) {
    positions[i] = i;
  }
  for (std::size_t i = source_cols; i > 1; --i) {
    std::swap(positions[i - 1], positions[pick(rng, i)]);
  }
  layout.fetch_cols.assign(positions.begin(),
                           positions.begin() + static_cast<std::ptrdiff_t>(layout.types.size()));
  layout.source_junk_string.resize(source_cols);
  for (std::size_t i = 0; i < source_cols; ++i) {
    layout.source_junk_string[i] = pick(rng, 2) == 0;
  }
}

/// @p table with FLOAT64 and DECIMAL64 columns viewed as INT64, so NaN equals NaN and -0 differs
/// from +0.
std::vector<cudf::column_view> as_bits(const cudf::table_view& table)
{
  std::vector<cudf::column_view> columns;
  for (const auto& column : table) {
    const auto id = column.type().id();
    columns.push_back(id == cudf::type_id::FLOAT64 || id == cudf::type_id::DECIMAL64
                        ? cudf::bit_cast(column, cudf::data_type{cudf::type_id::INT64})
                        : column);
  }
  return columns;
}

/// Same types, row count and values, bit for bit (NULL equals NULL).
bool tables_identical(const cudf::table_view& lhs, const cudf::table_view& rhs)
{
  if (lhs.num_columns() != rhs.num_columns()) { return false; }
  for (cudf::size_type c = 0; c < lhs.num_columns(); ++c) {
    if (lhs.column(c).type() != rhs.column(c).type()) { return false; }
  }
  return tables_equal(cudf::table_view{as_bits(lhs)}, cudf::table_view{as_bits(rhs)});
}

struct merge_outcome {
  std::size_t calls{0};  ///< run() calls it took
  std::size_t rows{0};   ///< rows of all runs
  merge_round_stats stats;
};

/// The runs of one merge and everything that has to outlive the merger.
struct merge_case {
  differential_fixture& fx;
  case_layout layout;
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  std::vector<std::vector<Tier>> tiers;  ///< per run, per batch
  std::vector<merge_input_cursor> cursors;

  merge_case(differential_fixture& fixture, case_layout case_layout)
    : fx(fixture), layout(std::move(case_layout))
  {
  }

  /**
   * @brief Add a run: sort @p fetched by the keys, split it into batches of @p batch_rows rows and
   * move each batch to its tier.
   *
   * @param fetched The run's rows, unsorted, one column per fetched column.
   * @param batch_tiers One tier per batch; an empty batch stays on the GPU.
   * @param disk_via_host Spill DISK batches GPU -> HOST -> DISK instead of GPU -> DISK.
   */
  void add_run(std::vector<std::unique_ptr<cudf::column>> fetched,
               const std::vector<std::size_t>& batch_rows,
               std::vector<Tier> batch_tiers,
               bool disk_via_host = false)
  {
    REQUIRE(fetched.size() == layout.types.size());
    REQUIRE(batch_rows.size() == batch_tiers.size());
    const auto rows = static_cast<std::size_t>(fetched.front()->size());

    std::vector<std::unique_ptr<cudf::column>> source(layout.source_columns());
    for (std::size_t f = 0; f < fetched.size(); ++f) {
      REQUIRE(fetched[f]->type() == layout.types[f]);
      source[layout.fetch_cols[f]] = std::move(fetched[f]);
    }
    for (std::size_t s = 0; s < source.size(); ++s) {
      if (source[s]) { continue; }
      if (layout.source_junk_string[s]) {
        std::vector<std::string> junk(rows);
        for (std::size_t i = 0; i < rows; ++i) {
          junk[i] = "junk-" + std::to_string(i % 97);
        }
        source[s] = string_column(junk);
      } else {
        source[s] = int64_column(std::vector<std::int64_t>(rows, -7));
      }
    }
    const cudf::table unsorted(std::move(source));

    merge_key_spec source_keys = layout.keys;
    for (auto& key : source_keys.key_cols) {
      key = static_cast<cudf::size_type>(layout.fetch_cols[static_cast<std::size_t>(key)]);
    }
    auto sorted = stable_sort(unsorted.view(), source_keys);
    runs.push_back(make_gpu_run(repository, *fx.gpu, sorted->view(), batch_rows));
    // The batches were copied on the default stream; the converters run on fx.stream.
    cudf::get_default_stream().sync();

    auto& run = runs.back();
    for (std::size_t b = 0; b < run.batches.size(); ++b) {
      if (batch_rows[b] == 0) { batch_tiers[b] = Tier::GPU; }
      if (batch_tiers[b] != Tier::GPU) {
        auto mut = run.batches[b]->to_mutable();
        if (batch_tiers[b] == Tier::HOST || disk_via_host) {
          mut.convert_to<cucascade::host_data_representation>(
            sirius::converter_registry::get(), fx.host, fx.stream);
          fx.stream.synchronize();
        }
        if (batch_tiers[b] == Tier::DISK) {
          mut.convert_to<cucascade::disk_data_representation>(
            sirius::converter_registry::get(), fx.disk, fx.stream);
          fx.stream.synchronize();
        }
      }
      REQUIRE(tier_of(*run.batches[b]) == batch_tiers[b]);
    }
    tiers.push_back(std::move(batch_tiers));
  }

  /// The fetched columns of every run, concatenated in run order and stably sorted by the keys.
  [[nodiscard]] std::unique_ptr<cudf::table> oracle() const
  {
    const std::vector<cudf::size_type> select(layout.fetch_cols.begin(), layout.fetch_cols.end());
    std::vector<cudf::table_view> views;
    for (const auto& run : runs) {
      views.push_back(run.rows->view().select(select));
    }
    return expected_full_sort(std::span<const cudf::table_view>(views), layout.keys);
  }

  /**
   * @brief Merge every run with the real kernel and compare the concatenated output of all run()
   * calls with the oracle.
   *
   * @param fresh_merger_each_call Build a new merger over the same cursors for every run() call,
   *        as consecutive tasks of the operator do.
   */
  merge_outcome merge_and_check(window_config cfg, bool fresh_merger_each_call = false)
  {
    merge_outcome outcome;
    std::size_t spilled_rows = 0;
    for (std::size_t r = 0; r < runs.size(); ++r) {
      outcome.rows += runs[r].run.total_rows();
      for (std::size_t b = 0; b < tiers[r].size(); ++b) {
        if (tiers[r][b] != Tier::GPU) { spilled_rows += runs[r].run.batch_rows[b]; }
      }
    }
    CAPTURE(cfg.window_rows, cfg.output_target_bytes, fresh_merger_each_call, outcome.rows);

    cursors = make_cursors(runs);
    std::vector<std::unique_ptr<cudf::table>> chunks;
    std::vector<cudf::table_view> chunk_views;
    {
      std::optional<windowed_merger> merger;
      while (!all_exhausted(cursors)) {
        if (fresh_merger_each_call || !merger) {
          merger.emplace(run_batches(runs),
                         cursors,
                         layout.fetch_cols,
                         layout.types,
                         layout.keys,
                         cfg,
                         sirius::row_range_reader_registry::get(),
                         *fx.gpu);
        }
        chunks.push_back(merger->run(fx.stream_view(), get_resource_ref(*fx.gpu), &outcome.stats));
        chunk_views.push_back(chunks.back()->view());
        ++outcome.calls;
        if (chunks.back()->num_rows() == 0) {
          FAIL_CHECK("run() returned no rows although a cursor is not exhausted");
          return outcome;
        }
      }
    }
    fx.stream.synchronize();

    // Bookkeeping: every row came out once, through window buffers where the run was spilled.
    CHECK(outcome.stats.rows_out == outcome.rows);
    CHECK(outcome.stats.rounds >= outcome.calls);
    CHECK(outcome.stats.fallback_conversions == 0);
    CHECK(outcome.stats.refill_rows >= spilled_rows);
    if (spilled_rows == 0) { CHECK(outcome.stats.refill_rows == 0); }
    for (std::size_t r = 0; r < runs.size(); ++r) {
      for (std::size_t b = 0; b < runs[r].batches.size(); ++b) {
        CAPTURE(r, b);
        CHECK(tier_of(*runs[r].batches[b]) == tiers[r][b]);
        CHECK(runs[r].batches[b]->try_to_mutable().has_value());  // no read lock left behind
      }
    }

    const auto expected = oracle();
    const auto actual   = cudf::concatenate(chunk_views);
    const auto out      = actual->view();
    REQUIRE(out.num_columns() == expected->num_columns());
    for (cudf::size_type c = 0; c < out.num_columns(); ++c) {
      CAPTURE(c);
      CHECK(out.column(c).type() == expected->view().column(c).type());
    }
    CHECK(out.num_rows() == expected->num_rows());
    if (out.num_rows() != expected->num_rows()) { return outcome; }

    // 1. The output is in key order.
    CHECK(cudf::is_sorted(
      out.select(layout.keys.key_cols), layout.keys.column_order, layout.keys.null_precedence));

    // 2. It holds exactly the input rows: ordered by the unique row id, both tables are the same.
    const auto by_id = [&](const cudf::table_view& table) {
      return cudf::sort_by_key(table, table.select({layout.id_col}));
    };
    CHECK(tables_identical(by_id(out)->view(), by_id(expected->view())->view()));

    // 3. Ties are in (run, row) order, as bounded_kway_merge documents: the output IS the stable
    //    sort of the runs concatenated in run order.
    const bool stable = tables_identical(out, expected->view());
    if (!stable) {
      const auto actual_ids =
        sirius::test::operator_utils::copy_column_to_host<std::int64_t>(out.column(layout.id_col));
      const auto expected_ids = sirius::test::operator_utils::copy_column_to_host<std::int64_t>(
        expected->view().column(layout.id_col));
      const auto mismatch =
        std::mismatch(actual_ids.begin(), actual_ids.end(), expected_ids.begin()).first -
        actual_ids.begin();
      const auto row = static_cast<std::size_t>(mismatch);
      if (row < actual_ids.size()) {
        INFO("first row whose id differs from the stable sort: "
             << row << ", actual id " << actual_ids[row] << ", expected id " << expected_ids[row]);
        CHECK(stable);
      } else {
        INFO("row ids are in the stable order, but a column's values differ");
        CHECK(stable);
      }
    }
    return outcome;
  }
};

// ---------------------------------------------------------------------------
// Seeded random cases
// ---------------------------------------------------------------------------

struct random_options {
  std::size_t min_rows{1};     ///< rows per run, lower bound
  std::size_t max_rows{0};     ///< rows per run, upper bound; 0 picks 64, 2 500 or 20 000
  std::size_t min_batches{1};  ///< batches per run
  std::size_t max_batches{1};  ///< batches per run
  bool tier_per_batch{false};  ///< a tier per batch instead of one per run
  std::optional<std::size_t> output_target;
  std::optional<bool> fresh_merger_each_call;
};

/// @p rows split at random points into @p batches batches (some may be empty).
std::vector<std::size_t> split_rows(std::size_t rows, std::size_t batches, rng_t& rng)
{
  std::vector<std::size_t> cuts{0, rows};
  for (std::size_t i = 1; i < batches; ++i) {
    cuts.push_back(pick(rng, rows + 1));
  }
  std::sort(cuts.begin(), cuts.end());
  std::vector<std::size_t> sizes;
  for (std::size_t i = 1; i < cuts.size(); ++i) {
    sizes.push_back(cuts[i] - cuts[i - 1]);
  }
  return sizes;
}

Tier random_tier(rng_t& rng)
{
  static constexpr Tier all[] = {Tier::GPU, Tier::HOST, Tier::DISK};
  return all[pick(rng, 3)];
}

/**
 * @brief Build the case of @p seed, merge it and compare it with the oracle.
 *
 * k in [1, 12] runs; n in {1024, 4096}; 1-3 keys of INT32 / INT64 / FLOAT64 / STRING / DECIMAL64
 * (the first key's type cycles with the seed) in random order and null precedence, from pools of
 * 1, 2, 7 or 100 values or unbounded; a unique INT64 row id and optional STRING / INT32 payloads;
 * source batches with the columns in another order among unfetched ones.
 *
 * @param description Set to the case's parameters before anything is run.
 */
merge_outcome run_random_case(differential_fixture& fx,
                              std::uint64_t seed,
                              const random_options& options,
                              std::string& description)
{
  static constexpr cudf::type_id key_types[]   = {cudf::type_id::INT32,
                                                  cudf::type_id::INT64,
                                                  cudf::type_id::FLOAT64,
                                                  cudf::type_id::STRING,
                                                  cudf::type_id::DECIMAL64};
  static constexpr std::size_t cardinalities[] = {1, 2, 7, 100, 0};
  static constexpr int null_percents[]         = {-1, 0, 10, 60};

  rng_t rng(seed * 0x9E3779B97F4A7C15ULL + 0x5DEECE66DULL);
  std::ostringstream text;
  text << "seed " << seed << ":";

  // Columns in "role" order: keys, row id, payloads; then shuffled into the fetch order.
  const auto num_keys = 1 + pick(rng, 3);
  std::vector<column_factory> role_factories;
  std::vector<cudf::data_type> role_types;
  merge_key_spec role_keys;
  for (std::size_t key = 0; key < num_keys; ++key) {
    const auto id          = key == 0 ? key_types[seed % 5] : key_types[pick(rng, 5)];
    const auto cardinality = cardinalities[pick(rng, 5)];
    const auto nulls       = null_percents[pick(rng, 4)];
    role_factories.push_back(random_factory(id, cardinality, nulls, rng));
    role_types.push_back(type_of(id));
    role_keys.column_order.push_back(pick(rng, 2) == 0 ? cudf::order::ASCENDING
                                                       : cudf::order::DESCENDING);
    role_keys.null_precedence.push_back(pick(rng, 2) == 0 ? cudf::null_order::BEFORE
                                                          : cudf::null_order::AFTER);
    text << " key" << key << "=" << type_name(id)
         << (role_keys.column_order.back() == cudf::order::ASCENDING ? " ASC" : " DESC")
         << (role_keys.null_precedence.back() == cudf::null_order::BEFORE ? " nulls-before"
                                                                          : " nulls-after")
         << " card=" << cardinality << " null%=" << nulls << ";";
  }
  const auto id_role = role_factories.size();
  auto next_id       = std::make_shared<std::int64_t>(0);
  role_factories.emplace_back([next_id](std::size_t rows, rng_t&) {
    std::vector<std::int64_t> ids(rows);
    for (auto& id : ids) {
      id = (*next_id)++;
    }
    return int64_column(ids);
  });
  role_types.emplace_back(cudf::type_id::INT64);
  if (pick(rng, 2) == 0) {
    role_factories.push_back(random_factory(cudf::type_id::STRING, 0, 10, rng));
    role_types.emplace_back(cudf::type_id::STRING);
    text << " +STRING payload;";
  }
  if (pick(rng, 2) == 0) {
    role_factories.push_back(random_factory(cudf::type_id::INT32, 0, 10, rng));
    role_types.emplace_back(cudf::type_id::INT32);
    text << " +INT32 payload;";
  }

  // fetch_position[role] is the fetched column of that role.
  std::vector<std::size_t> fetch_position(role_factories.size());
  for (std::size_t i = 0; i < fetch_position.size(); ++i) {
    fetch_position[i] = i;
  }
  for (std::size_t i = fetch_position.size(); i > 1; --i) {
    std::swap(fetch_position[i - 1], fetch_position[pick(rng, i)]);
  }
  case_layout layout;
  layout.types.resize(role_types.size());
  std::vector<column_factory> factories(role_factories.size());
  for (std::size_t role = 0; role < role_factories.size(); ++role) {
    layout.types[fetch_position[role]] = role_types[role];
    factories[fetch_position[role]]    = role_factories[role];
  }
  layout.keys = role_keys;
  for (std::size_t key = 0; key < num_keys; ++key) {
    layout.keys.key_cols.push_back(static_cast<cudf::size_type>(fetch_position[key]));
  }
  layout.id_col = static_cast<cudf::size_type>(fetch_position[id_role]);
  set_random_source(layout, rng);

  const auto num_runs = 1 + pick(rng, 12);
  window_config cfg;
  cfg.window_rows                        = pick(rng, 2) == 0 ? 1024 : 4096;
  static constexpr std::size_t targets[] = {kUnlimitedOutput, std::size_t{1} << 20, 65536};
  cfg.output_target_bytes                = options.output_target.value_or(targets[pick(rng, 3)]);
  const bool fresh = options.fresh_merger_each_call.value_or(pick(rng, 2) == 0);
  static constexpr std::size_t row_caps[] = {64, 2500, 2500, 20000, 20000};
  const auto max_rows = options.max_rows != 0 ? options.max_rows : row_caps[pick(rng, 5)];
  text << " k=" << num_runs << " n=" << cfg.window_rows << " target=" << cfg.output_target_bytes
       << " fresh=" << fresh << " fetch_cols=[";
  for (const auto col : layout.fetch_cols) {
    text << col << " ";
  }
  text << "] of " << layout.source_columns() << "; runs:";

  // Decide every run's shape first, so the description is complete before anything can fail.
  struct run_shape {
    std::vector<std::size_t> batch_rows;
    std::vector<Tier> tiers;
    bool disk_via_host{false};
  };
  std::vector<run_shape> shapes(num_runs);
  for (auto& shape : shapes) {
    const auto rows =
      options.min_rows + pick(rng, std::max(max_rows, options.min_rows) - options.min_rows + 1);
    const auto batches =
      options.min_batches + pick(rng, options.max_batches - options.min_batches + 1);
    shape.batch_rows = split_rows(rows, batches, rng);
    const auto tier  = random_tier(rng);
    for (std::size_t b = 0; b < shape.batch_rows.size(); ++b) {
      shape.tiers.push_back(options.tier_per_batch ? random_tier(rng) : tier);
    }
    shape.disk_via_host = pick(rng, 2) == 0;
    text << " {";
    for (std::size_t b = 0; b < shape.batch_rows.size(); ++b) {
      text << (b == 0 ? "" : ",") << shape.batch_rows[b] << tier_name(shape.tiers[b]);
    }
    text << "}";
  }
  description = text.str();
  INFO(description);

  merge_case merge(fx, std::move(layout));
  for (auto& shape : shapes) {
    std::size_t rows = 0;
    for (const auto batch : shape.batch_rows) {
      rows += batch;
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    for (const auto& factory : factories) {
      columns.push_back(factory(rows, rng));
    }
    merge.add_run(std::move(columns), shape.batch_rows, shape.tiers, shape.disk_via_host);
  }
  return merge.merge_and_check(cfg, fresh);
}

/// Run @p seeds through run_random_case; a case that throws fails and the next one still runs.
template <typename Check>
void run_random_cases(std::uint64_t first_seed,
                      std::uint64_t num_seeds,
                      const random_options& options,
                      Check&& check)
{
  differential_fixture fx;
  for (auto seed = first_seed; seed < first_seed + num_seeds; ++seed) {
    std::string description;
    try {
      const auto outcome = run_random_case(fx, seed, options, description);
      INFO(description);
      check(outcome);
    } catch (const std::exception& e) {
      FAIL_CHECK(description << "\n  threw: " << e.what());
    }
  }
}

// ---------------------------------------------------------------------------
// Hand-built (key INT64, id INT64, payload STRING) cases
// ---------------------------------------------------------------------------

/// Fetched columns (key INT64 nullable, id INT64, payload STRING) read from (payload, key, id).
case_layout key_id_payload_layout(cudf::order order = cudf::order::ASCENDING)
{
  case_layout layout;
  layout.types = {cudf::data_type{cudf::type_id::INT64},
                  cudf::data_type{cudf::type_id::INT64},
                  cudf::data_type{cudf::type_id::STRING}};
  layout.keys  = {
     .key_cols = {0}, .column_order = {order}, .null_precedence = {cudf::null_order::AFTER}};
  layout.id_col             = 1;
  layout.fetch_cols         = {1, 2, 0};
  layout.source_junk_string = {false, false, false};
  return layout;
}

/// Adds runs of (key, id, payload) rows with ids that are unique across the case.
struct keyed_runs {
  merge_case merge;
  std::int64_t next_id{0};

  keyed_runs(differential_fixture& fx, case_layout layout) : merge(fx, std::move(layout)) {}

  /// @p key_valid empty: no null keys. The payload of a row is @p payload(row).
  template <typename Payload>
  void add(const std::vector<std::int64_t>& keys,
           const std::vector<bool>& key_valid,
           Payload&& payload,
           const std::vector<std::size_t>& batch_rows,
           const std::vector<Tier>& batch_tiers)
  {
    std::vector<std::int64_t> ids(keys.size());
    std::vector<std::string> payloads(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
      ids[i]      = next_id++;
      payloads[i] = payload(i);
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column(keys, key_valid));
    columns.push_back(int64_column(ids));
    columns.push_back(string_column(payloads));
    merge.add_run(std::move(columns), batch_rows, batch_tiers);
  }

  void add(const std::vector<std::int64_t>& keys, Tier tier)
  {
    add(keys,
        {},
        [&](std::size_t i) { return "p" + std::to_string(keys[i] % 13); },
        {keys.size()},
        {tier});
  }
};

constexpr Tier kTierCycle[] = {Tier::HOST, Tier::DISK, Tier::GPU};

}  // namespace

// ---------------------------------------------------------------------------
// 1. Seeded random cases
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge equals a full sort on seeded random mixed-tier runs",
          "[windowed_merge_differential]")
{
  // 60 seeds; the first key's type cycles through INT32, INT64, FLOAT64, STRING, DECIMAL64.
  run_random_cases(0, 60, random_options{}, [](const merge_outcome&) {});
}

// ---------------------------------------------------------------------------
// 2. STRING columns that truncate windows by bytes
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge is exact when long STRING payloads truncate windows by bytes",
          "[windowed_merge_differential]")
{
  // Most payloads are a few bytes, so the per-row estimate the merger sizes the windows with is
  // small; 2% of the keys (adjacent once sorted) carry 1.5-2.5 KB payloads and one row carries
  // 150 KB, more than a whole 1024-row window was sized for.
  const auto window_rows = GENERATE(std::size_t{1024}, std::size_t{4096});
  CAPTURE(window_rows);
  differential_fixture fx;
  keyed_runs runs(fx, key_id_payload_layout());
  rng_t rng(20261112);
  constexpr std::size_t rows_per_run = 6000;
  constexpr Tier run_tiers[]         = {Tier::HOST, Tier::DISK, Tier::HOST, Tier::DISK, Tier::GPU};
  for (const auto tier : run_tiers) {
    std::vector<std::int64_t> keys(rows_per_run);
    std::vector<std::size_t> lengths(rows_per_run);
    for (std::size_t i = 0; i < rows_per_run; ++i) {
      keys[i]    = static_cast<std::int64_t>(pick(rng, 1'000'000));
      lengths[i] = keys[i] >= 400'000 && keys[i] < 420'000 ? 1500 + pick(rng, 1001) : pick(rng, 4);
    }
    lengths[pick(rng, rows_per_run)] = 150'000;
    runs.add(keys,
             {},
             [&](std::size_t i) {
               return std::string(lengths[i], static_cast<char>('a' + (keys[i] % 26)));
             },
             {rows_per_run},
             {tier});
  }
  const auto outcome =
    runs.merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = 1 << 20});
  CHECK(outcome.calls > 1);
  // 24 000 spilled rows fit 6 windows of 4096 rows per run; byte-truncated refills need more.
  CHECK(outcome.stats.rounds > 6);
}

TEST_CASE("windowed merge is exact with long STRING keys", "[windowed_merge_differential]")
{
  // STRING key, DESC, nullable. 5% of the keys are about 2 KB and share a 1990-byte prefix, so
  // they are adjacent in the runs and differ only in their last bytes.
  const auto window_rows = GENERATE(std::size_t{1024}, std::size_t{4096});
  CAPTURE(window_rows);
  differential_fixture fx;
  case_layout layout;
  layout.types  = {cudf::data_type{cudf::type_id::INT64}, cudf::data_type{cudf::type_id::STRING}};
  layout.keys   = {.key_cols        = {1},
                   .column_order    = {cudf::order::DESCENDING},
                   .null_precedence = {cudf::null_order::BEFORE}};
  layout.id_col = 0;
  set_identity_source(layout);
  merge_case merge(fx, std::move(layout));

  rng_t rng(77);
  const std::string prefix(1990, 'm');
  std::int64_t next_id       = 0;
  constexpr Tier run_tiers[] = {Tier::DISK, Tier::HOST, Tier::GPU, Tier::HOST};
  for (const auto tier : run_tiers) {
    constexpr std::size_t rows = 4000;
    std::vector<std::int64_t> ids(rows);
    std::vector<std::string> keys(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      ids[i]  = next_id++;
      keys[i] = (pick(rng, 20) == 0 ? prefix : std::string{}) + std::to_string(pick(rng, 500));
    }
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(int64_column(ids));
    columns.push_back(string_column(keys, draw_validity(rows, 5, rng)));
    merge.add_run(std::move(columns), {rows}, {tier});
  }
  merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = kUnlimitedOutput});
}

// ---------------------------------------------------------------------------
// 3. Multi-batch runs (intermediate-run shape)
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge equals a full sort on multi-batch runs with a tier per batch",
          "[windowed_merge_differential]")
{
  random_options options;
  options.min_rows       = 2000;
  options.max_rows       = 20000;
  options.min_batches    = 2;
  options.max_batches    = 8;
  options.tier_per_batch = true;
  run_random_cases(1000, 12, options, [](const merge_outcome&) {});
}

TEST_CASE("windowed merge handles batches of exactly, just under and just over a window",
          "[windowed_merge_differential]")
{
  constexpr std::size_t n = 1024;
  differential_fixture fx;
  keyed_runs runs(fx, key_id_payload_layout());
  rng_t rng(5);
  const auto payload = [](std::size_t i) { return "row" + std::to_string(i % 101); };
  // Batch sizes around the window size, a one-row batch, an empty (GPU) batch, and tiers that
  // change from batch to batch inside one run.
  const std::vector<std::vector<std::size_t>> batch_rows{
    {n, 1, n - 1, n + 1, 0, 3 * n + 7},
    {1, 1, 1, 2 * n},
    {n, n, n},
    {5 * n + 13},
  };
  const std::vector<std::vector<Tier>> batch_tiers{
    {Tier::HOST, Tier::DISK, Tier::GPU, Tier::DISK, Tier::GPU, Tier::HOST},
    {Tier::DISK, Tier::HOST, Tier::GPU, Tier::HOST},
    {Tier::DISK, Tier::DISK, Tier::DISK},
    {Tier::GPU},
  };
  for (std::size_t r = 0; r < batch_rows.size(); ++r) {
    std::size_t rows = 0;
    for (const auto batch : batch_rows[r]) {
      rows += batch;
    }
    std::vector<std::int64_t> keys(rows);
    for (auto& key : keys) {
      key = static_cast<std::int64_t>(pick(rng, 3000));  // every key about 5 times: ties
    }
    runs.add(keys, draw_validity(rows, 3, rng), payload, batch_rows[r], batch_tiers[r]);
  }
  const bool fresh = GENERATE(false, true);
  const auto outcome =
    runs.merge.merge_and_check({.window_rows = n, .output_target_bytes = 50000}, fresh);
  CHECK(outcome.calls > 1);
}

// ---------------------------------------------------------------------------
// 4. Several run() calls with a small output target
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge outputs of several run() calls concatenate to the full sort",
          "[windowed_merge_differential]")
{
  // Every run has more rows than the largest window, so even one run needs two rounds, and with
  // these targets a run() call stops after very few rounds.
  const auto target = GENERATE(std::size_t{1}, std::size_t{4096}, std::size_t{32768});
  const bool fresh  = GENERATE(false, true);
  CAPTURE(target, fresh);
  random_options options;
  options.min_rows               = 5000;
  options.max_rows               = 12000;
  options.max_batches            = 2;
  options.output_target          = target;
  options.fresh_merger_each_call = fresh;
  const std::uint64_t first_seed = 2000 + (target % 97) * 10 + (fresh ? 5 : 0);
  run_random_cases(first_seed, 3, options, [&](const merge_outcome& outcome) {
    CHECK(outcome.calls > 1);
    // A 1-byte target is reached by the first round of every call.
    if (target == 1) { CHECK(outcome.stats.rounds == outcome.calls); }
  });
}

// ---------------------------------------------------------------------------
// 5. Already-sorted, reverse-sorted and single-distinct-key datasets
// ---------------------------------------------------------------------------

TEST_CASE("windowed merge of an already sorted dataset", "[windowed_merge_differential]")
{
  constexpr std::size_t num_runs     = 6;
  constexpr std::size_t rows_per_run = 5000;
  const auto window_rows             = GENERATE(std::size_t{1024}, std::size_t{4096});
  const auto order                   = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  differential_fixture fx;

  SECTION("runs are consecutive key ranges, in order")
  {
    // The merge drains run 0, then run 1, ...: all other windows wait, truncated.
    keyed_runs runs(fx, key_id_payload_layout(order));
    for (std::size_t r = 0; r < num_runs; ++r) {
      // With DESC keys the first run holds the largest keys, so the data is "sorted" either way.
      const auto range = order == cudf::order::ASCENDING ? r : num_runs - 1 - r;
      std::vector<std::int64_t> keys(rows_per_run);
      for (std::size_t i = 0; i < rows_per_run; ++i) {
        keys[i] = static_cast<std::int64_t>(range * rows_per_run + i);
      }
      runs.add(keys, kTierCycle[r % 3]);
    }
    runs.merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = 200000});
  }

  SECTION("runs interleave row by row")
  {
    // Run r holds the keys congruent to r: every round takes the same share of every window.
    keyed_runs runs(fx, key_id_payload_layout(order));
    for (std::size_t r = 0; r < num_runs; ++r) {
      std::vector<std::int64_t> keys(rows_per_run);
      for (std::size_t i = 0; i < rows_per_run; ++i) {
        keys[i] = static_cast<std::int64_t>(i * num_runs + r);
      }
      runs.add(keys, kTierCycle[r % 3]);
    }
    runs.merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = 200000});
  }
}

TEST_CASE("windowed merge of a reverse sorted dataset", "[windowed_merge_differential]")
{
  // The dataset is in the opposite of the requested order, so the LAST run holds the first keys
  // of the output and the first run is consumed last.
  constexpr std::size_t num_runs     = 6;
  constexpr std::size_t rows_per_run = 5000;
  const auto window_rows             = GENERATE(std::size_t{1024}, std::size_t{4096});
  const auto order                   = GENERATE(cudf::order::ASCENDING, cudf::order::DESCENDING);
  differential_fixture fx;
  keyed_runs runs(fx, key_id_payload_layout(order));
  for (std::size_t r = 0; r < num_runs; ++r) {
    const auto range = order == cudf::order::ASCENDING ? num_runs - 1 - r : r;
    std::vector<std::int64_t> keys(rows_per_run);
    for (std::size_t i = 0; i < rows_per_run; ++i) {
      keys[i] = static_cast<std::int64_t>(range * rows_per_run + (rows_per_run - 1 - i));
    }
    runs.add(keys, kTierCycle[r % 3]);
  }
  runs.merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = 200000});
}

TEST_CASE("windowed merge of a dataset with a single distinct key", "[windowed_merge_differential]")
{
  // Every row ties with every other one: the output order is decided by (run, row) alone.
  const auto window_rows = GENERATE(std::size_t{1024}, std::size_t{4096});
  const bool null_key    = GENERATE(false, true);
  CAPTURE(window_rows, null_key);
  differential_fixture fx;
  keyed_runs runs(fx, key_id_payload_layout());
  const std::vector<std::size_t> run_rows{7000, 1, 4096, 1024, 9000, 3000, 2};
  for (std::size_t r = 0; r < run_rows.size(); ++r) {
    const auto rows = run_rows[r];
    runs.add(std::vector<std::int64_t>(rows, 42),
             null_key ? std::vector<bool>(rows, false) : std::vector<bool>{},
             [&](std::size_t i) { return "r" + std::to_string(r) + "-" + std::to_string(i); },
             {rows},
             {kTierCycle[r % 3]});
  }
  const auto outcome =
    runs.merge.merge_and_check({.window_rows = window_rows, .output_target_bytes = 100000});
  CHECK(outcome.calls > 1);
}
