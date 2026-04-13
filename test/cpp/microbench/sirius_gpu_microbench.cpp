/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License.
 *
 * libcudf GPU microbenchmarks (Google Benchmark). Timed regions mirror hot
 * operator paths: hash join, groupby sum, sort keys, boolean filter, optional Parquet read.
 *
 * Run:
 *   sirius_gpu_microbench --benchmark_format=json
 *   sirius_gpu_microbench --benchmark_filter='BM_HashJoin|BM_GroupBySum'
 *
 * TPC-H–style Parquet column read (optional):
 *   export SIRIUS_MICROBENCH_PARQUET_FILE=$PWD/test_datasets/tpch_parquet_sf1/lineitem.parquet
 *   export SIRIUS_MICROBENCH_PARQUET_COLUMN=l_orderkey
 */

#include "microbench_data.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream.hpp>

#include <benchmark/benchmark.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace {

void BM_HashJoin(benchmark::State& state)
{
  rmm::cuda_stream stream;
  auto const build_n     = static_cast<cudf::size_type>(state.range(0));
  auto const probe_n     = static_cast<cudf::size_type>(state.range(1));
  std::int32_t const ndv = std::max(
    std::int32_t{1}, std::min(static_cast<std::int32_t>(std::min(build_n, probe_n)), 1 << 20));

  auto build_key = sirius::microbench::make_modulo_int32_keys(build_n, ndv, stream);
  auto probe_key = sirius::microbench::make_modulo_int32_keys(probe_n, ndv, stream);
  cudf::table_view const build_tv{{build_key->view()}};
  cudf::table_view const probe_tv{{probe_key->view()}};

  for (auto _ : state) {
    cudf::hash_join hj(build_tv, cudf::null_equality::UNEQUAL, stream);
    stream.synchronize();
    auto joined = hj.inner_join(probe_tv, std::nullopt, stream);
    stream.synchronize();
    benchmark::DoNotOptimize(joined.first);
    benchmark::DoNotOptimize(joined.second);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(probe_n));
}

void BM_GroupBySum(benchmark::State& state)
{
  rmm::cuda_stream stream;
  auto const n   = static_cast<cudf::size_type>(state.range(0));
  auto const ndv = std::max(std::int32_t{1}, static_cast<std::int32_t>(state.range(1)));
  auto keys      = sirius::microbench::make_modulo_int32_keys(n, ndv, stream);
  auto vals      = sirius::microbench::make_int64_ones(n, stream);

  cudf::groupby::aggregation_request req;
  req.values = vals->view();
  req.aggregations.push_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
  std::vector<cudf::groupby::aggregation_request> reqs;
  reqs.push_back(std::move(req));

  for (auto _ : state) {
    cudf::groupby::groupby gb(cudf::table_view{{keys->view()}}, cudf::null_policy::INCLUDE);
    auto out = gb.aggregate(cudf::host_span<cudf::groupby::aggregation_request const>(
                              reqs.data(), static_cast<std::size_t>(reqs.size())),
                            stream);
    stream.synchronize();
    benchmark::DoNotOptimize(out.first);
    benchmark::DoNotOptimize(out.second);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_SortKeys(benchmark::State& state)
{
  rmm::cuda_stream stream;
  auto const n = static_cast<cudf::size_type>(state.range(0));
  std::int32_t const ndv =
    std::max(std::int32_t{1}, std::min(static_cast<std::int32_t>(n), 1 << 18));
  auto keys = sirius::microbench::make_modulo_int32_keys(n, ndv, stream);
  cudf::table_view const key_tbl{{keys->view()}};
  std::vector<cudf::order> const orders{cudf::order::ASCENDING};
  std::vector<cudf::null_order> const null_orders{cudf::null_order::AFTER};

  for (auto _ : state) {
    auto sorted_ix = cudf::sorted_order(key_tbl, orders, null_orders, stream);
    stream.synchronize();
    benchmark::DoNotOptimize(sorted_ix);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_FilterMask(benchmark::State& state)
{
  rmm::cuda_stream stream;
  auto const n       = static_cast<cudf::size_type>(state.range(0));
  int const permille = static_cast<int>(state.range(1));
  auto payload       = sirius::microbench::make_modulo_int32_keys(n, 100000, stream);
  auto mask          = sirius::microbench::make_sparse_bool_mask(n, permille, stream);
  cudf::table_view const tbl{{payload->view()}};

  for (auto _ : state) {
    auto out = cudf::apply_boolean_mask(tbl, mask->view(), stream);
    stream.synchronize();
    benchmark::DoNotOptimize(out);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(n));
}

void BM_ParquetReadColumn(benchmark::State& state)
{
  char const* path = std::getenv("SIRIUS_MICROBENCH_PARQUET_FILE");
  char const* col  = std::getenv("SIRIUS_MICROBENCH_PARQUET_COLUMN");
  if (path == nullptr || col == nullptr) {
    state.SkipWithError("Set SIRIUS_MICROBENCH_PARQUET_FILE and SIRIUS_MICROBENCH_PARQUET_COLUMN");
    return;
  }

  rmm::cuda_stream stream;
  auto const max_rows = static_cast<cudf::size_type>(state.range(0));

  for (auto _ : state) {
    auto opt = sirius::microbench::try_read_parquet_column(path, col, max_rows, stream);
    stream.synchronize();
    benchmark::DoNotOptimize(opt);
  }
}

}  // namespace

// All parameter packs registered here; daily CI uses --benchmark_filter to run a subset.
BENCHMARK(BM_HashJoin)
  ->Args({1 << 16, 1 << 20})
  ->Args({1 << 17, 1 << 21})
  ->Args({1 << 18, 1 << 22})
  ->Args({1 << 19, 1 << 23})
  ->ArgNames({"build_rows", "probe_rows"})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_GroupBySum)
  ->Args({1 << 20, 100000})
  ->Args({1 << 22, 200000})
  ->Args({1 << 21, 500000})
  ->Args({1 << 23, 800000})
  ->ArgNames({"rows", "num_groups"})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_SortKeys)
  ->Arg(1 << 20)
  ->Arg(1 << 22)
  ->Arg(1 << 23)
  ->ArgName("rows")
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_FilterMask)
  ->Args({1 << 20, 50})
  ->Args({1 << 22, 50})
  ->Args({1 << 23, 100})
  ->Args({1 << 22, 200})
  ->ArgNames({"rows", "permille_true"})
  ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_ParquetReadColumn)
  ->Arg(1 << 20)
  ->Arg(1 << 22)
  ->ArgName("max_rows")
  ->Unit(benchmark::kMillisecond);
