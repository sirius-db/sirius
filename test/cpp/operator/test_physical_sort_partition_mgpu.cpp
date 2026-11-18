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

// Multi-GPU tests of the range partitioning of ORDER BY: SORT_SAMPLE computes the partition
// boundaries once, and SORT_PARTITION cuts every sorted batch at them on whichever GPU holds the
// batch. The boundaries must be complete, and readable on every GPU, when they are published.

#include "helper/type_conversions.hpp"
#include "mgpu_test_utils.hpp"
#include "operator_test_utils.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

#include <cuda_runtime.h>

#include <catch.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <op/sirius_physical_merge_sort.hpp>
#include <op/sirius_physical_sort_partition.hpp>
#include <op/sirius_physical_sort_sample.hpp>
#include <unistd.h>
#include <utils/log_test_utils.hpp>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sirius::test::mgpu;

namespace {

constexpr std::uint64_t kBatches      = 48;
constexpr std::uint64_t kRowsPerBatch = 2'000'000;
constexpr std::uint64_t kRows         = kBatches * kRowsPerBatch;
// BIGINT key + DOUBLE payload, no nulls.
constexpr std::uint64_t kBytesPerRow   = 16;
constexpr std::uint64_t kBatchBytes    = kRowsPerBatch * kBytesPerRow;
constexpr std::uint64_t kSampleBatches = 12;

struct sink_summary {
  std::uint64_t rows{0};
  std::uint64_t bytes{0};
  std::uint64_t order_violations{0};
  std::uint64_t order_checked{0};
  std::uint64_t ranges_checked{0};
};

void run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO(sql << ": " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
}

/// Sorts the generated rows into @p partitions range partitions and returns the sink's summary.
sink_summary sort_and_check(duckdb::Connection& con, std::size_t partitions, std::uint64_t seed)
{
  // The sample is a quarter of the batches: the boundary task merges them before it computes the
  // boundaries, and the other batches are partitioned by tasks that start, on both GPUs, as soon
  // as the boundaries are published.
  run_ok(con, "SET sort_sample_bytes = " + std::to_string(kSampleBatches * kBatchBytes) + ";");
  run_ok(con,
         "SET max_sort_partition_bytes = " +
           std::to_string((kRows * kBytesPerRow + partitions - 1) / partitions) + ";");
  auto const sql =
    "COPY (SELECT * FROM sirius_generate(num_batches := " + std::to_string(kBatches) +
    ", rows_per_batch := " + std::to_string(kRowsPerBatch) + ", seed := " + std::to_string(seed) +
    ", columns := [{name: 'k', type: 'BIGINT', dist: 'uniform', p1: 0, p2: 1000000000000000}, "
    "{name: 'v', type: 'DOUBLE', dist: 'uniform', p1: 0, p2: 1}]) ORDER BY k) "
    "TO '/dev/null' (FORMAT sirius_discard, check_order true)";
  INFO(sql);

  std::vector<sirius::test::recording_log_sink::record> records;
  duckdb::unique_ptr<duckdb::QueryResult> result;
  sirius::op::sirius_physical_merge_sort::reset_process_counters();
  {
    sirius::test::scoped_recording_log_sink logs("info");
    result  = con.Query(sql);
    records = logs.records();
  }
  REQUIRE(result);
  if (result->HasError()) { UNSCOPED_INFO("GPU error: " << result->GetError()); }
  REQUIRE_FALSE(result->HasError());
  REQUIRE(result->RowCount() == 1);
  REQUIRE(result->Collection().GetValue(0, 0).GetValue<std::int64_t>() ==
          static_cast<std::int64_t>(kRows));

  // The sort really was cut into the requested number of range partitions.
  auto const merges = sirius::op::sirius_physical_merge_sort::process_counters();
  REQUIRE(merges.windowed_partitions + merges.one_shot_partitions == partitions);

  sink_summary summary;
  std::size_t summaries = 0;
  for (auto const& record : records) {
    if (record.message.rfind("[discard_sink] ", 0) != 0) { continue; }
    INFO(record.message);
    REQUIRE(std::sscanf(record.message.c_str(),
                        "[discard_sink] rows=%" SCNu64 " bytes=%" SCNu64
                        " order_violations=%" SCNu64 " order_checked=%" SCNu64
                        " ranges_checked=%" SCNu64,
                        &summary.rows,
                        &summary.bytes,
                        &summary.order_violations,
                        &summary.order_checked,
                        &summary.ranges_checked) == 5);
    summaries++;
  }
  REQUIRE(summaries == 1);
  return summary;
}

}  // namespace

TEST_CASE("ORDER BY on two GPUs is globally ordered across range partitions",
          "[mgpu][operator-mgpu][order][sort_partition][gpu_execution][regression][multi_gpu]")
{
  if (!require_two_gpus()) return;

  auto tmp =
    fs::temp_directory_path() / ("sirius-mgpu-sort-partition-" + std::to_string(::getpid()));
  std::error_code ec;
  fs::remove_all(tmp, ec);
  fs::create_directories(tmp);
  auto yaml = tmp / "mgpu.yaml";

  mgpu_env_params params;
  params.cache                    = "none";
  params.pipeline_num_threads     = 4;
  params.task_creator_num_threads = 4;
  write_mgpu_yaml(yaml, params);
  REQUIRE(fs::exists(yaml));

  std::map<int, size_t> tasks_per_gpu;
  {
    scoped_mgpu_env env(yaml);
    auto con = std::make_unique<duckdb::Connection>(env.make_connection());
    run_ok(*con, "SET enable_duckdb_fallback = false;");
    run_ok(*con, "SET gpu_execution = true;");

    for (std::size_t partitions : {std::size_t{2}, std::size_t{3}, std::size_t{8}}) {
      // Timing-dependent: repeat each partition count.
      for (std::uint64_t repeat = 0; repeat < 3; ++repeat) {
        CAPTURE(partitions, repeat);
        auto const summary = sort_and_check(*con, partitions, 42 + repeat);
        CHECK(summary.rows == kRows);
        REQUIRE(summary.order_checked == 1);
        CHECK(summary.ranges_checked > 0);
        CHECK(summary.order_violations == 0);
      }
    }

    auto& scheduler = env.get_task_scheduler(*con);
    con.reset();
    scheduler.visit_executors(
      [&](int device_id, const sirius::pipeline::gpu_pipeline_executor& exec) {
        tasks_per_gpu[device_id] = exec.get_metrics().tasks_executed;
      });
  }

  // Both GPUs took part, so batches were partitioned on a GPU other than the boundary task's.
  INFO("gpu0 tasks=" << tasks_per_gpu[0] << " gpu1 tasks=" << tasks_per_gpu[1]);
  REQUIRE(tasks_per_gpu.at(0) >= 1);
  REQUIRE(tasks_per_gpu.at(1) >= 1);

  fs::remove_all(tmp, ec);
}

namespace {

using sirius::test::operator_utils::initialize_memory_manager;
using sirius::test::operator_utils::make_numeric_batch;

/// The BIGINT keys of column 0 of @p batch, read with the batch's GPU current.
std::vector<std::int64_t> read_keys(const std::shared_ptr<cucascade::data_batch>& batch)
{
  auto read_only  = batch->to_read_only();
  auto const view = sirius::get_cudf_table_view(read_only);
  std::vector<std::int64_t> host(static_cast<std::size_t>(view.num_rows()));
  if (!host.empty()) {
    REQUIRE(cudaMemcpy(host.data(),
                       view.column(0).data<std::int64_t>(),
                       host.size() * sizeof(std::int64_t),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
  }
  return host;
}

}  // namespace

// The boundary task runs on GPU 0 on a stream that still has work queued, and a batch is
// partitioned on GPU 1 (and on GPU 0 on another stream) the moment the boundaries are published.
TEST_CASE("SORT_PARTITION cuts batches on every GPU at the boundaries SORT_SAMPLE published",
          "[mgpu][operator-mgpu][order][sort_partition][regression][multi_gpu]")
{
  if (!require_two_gpus()) return;

  constexpr std::int64_t kSampleRows = 24'000;
  constexpr std::int64_t kBatchRows  = 10'000;
  auto const partitions              = GENERATE(std::size_t{2}, std::size_t{3}, std::size_t{8});
  CAPTURE(partitions);

  auto manager =
    initialize_memory_manager(2, {.gpu_bytes = std::size_t{1} << 30, .divide_among_gpus = false});
  auto* space0 = manager->get_memory_space(cucascade::memory::Tier::GPU, 0);
  auto* space1 = manager->get_memory_space(cucascade::memory::Tier::GPU, 1);
  REQUIRE(space0 != nullptr);
  REQUIRE(space1 != nullptr);

  auto make_orders = [] {
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(duckdb::BoundOrderByNode(
      duckdb::OrderType::ASCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
    return orders;
  };
  auto const types =
    sirius::from_duckdb_vec(duckdb::vector<duckdb::LogicalType>{duckdb::LogicalType::BIGINT});

  // The streams must outlive everything allocated on them.
  std::optional<rmm::cuda_stream> sample_stream;
  std::optional<rmm::cuda_stream> other_stream0;
  std::optional<rmm::cuda_stream> stream1;
  {
    rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
    sample_stream.emplace();
    other_stream0.emplace();
  }
  {
    rmm::cuda_set_device_raii on_gpu1{rmm::cuda_device_id{1}};
    stream1.emplace();
  }

  // The sample: keys 0, 1, ..., sorted. Boundary i is the key i * kSampleRows / partitions.
  std::vector<std::int64_t> sample_keys(kSampleRows);
  std::iota(sample_keys.begin(), sample_keys.end(), std::int64_t{0});
  // Batches to partition: every key range is hit, on both GPUs.
  std::vector<std::int64_t> keys0(kBatchRows);
  std::vector<std::int64_t> keys1(kBatchRows);
  for (std::int64_t i = 0; i < kBatchRows; ++i) {
    keys0[static_cast<std::size_t>(i)] = (i * kSampleRows) / kBatchRows;
    keys1[static_cast<std::size_t>(i)] = (i * kSampleRows) / kBatchRows + 1;
  }

  std::shared_ptr<cucascade::data_batch> sample_batch;
  std::shared_ptr<cucascade::data_batch> batch0;
  std::shared_ptr<cucascade::data_batch> batch1;
  std::size_t sample_bytes = 0;
  {
    rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
    sample_batch = make_numeric_batch<std::int64_t>(*space0, sample_keys, cudf::type_id::INT64);
    batch0       = make_numeric_batch<std::int64_t>(*space0, keys0, cudf::type_id::INT64);
    sample_bytes = sample_batch->to_read_only().get_data()->get_size_in_bytes();
  }
  {
    rmm::cuda_set_device_raii on_gpu1{rmm::cuda_device_id{1}};
    batch1 = make_numeric_batch<std::int64_t>(*space1, keys1, cudf::type_id::INT64);
  }

  sirius::op::sirius_physical_sort_sample sample_op(types,
                                                    make_orders(),
                                                    static_cast<std::size_t>(kSampleRows),
                                                    sample_bytes,
                                                    (sample_bytes + partitions - 1) / partitions);
  sirius::op::sirius_physical_sort_partition partition_op(
    types, make_orders(), duckdb::vector<std::size_t>{0}, static_cast<std::size_t>(kSampleRows));
  partition_op.set_sample_op(&sample_op);

  // (partition index, keys) of every slice, from both GPUs.
  std::vector<std::vector<std::int64_t>> rows_of_partition(partitions);
  auto partition_batch = [&](const std::shared_ptr<cucascade::data_batch>& batch,
                             rmm::cuda_stream_view stream,
                             const std::vector<std::int64_t>& expected_keys) {
    auto output = partition_op.execute(sirius::op::pipelineable_operator_data({batch}), stream);
    auto const& slices = dynamic_cast<const sirius::op::sort_partition_output&>(*output);
    REQUIRE(slices.get_data_batches().size() == slices.partitions.size());
    std::vector<std::int64_t> seen;
    for (std::size_t i = 0; i < slices.partitions.size(); ++i) {
      REQUIRE(slices.partitions[i] < partitions);
      if (i > 0) { REQUIRE(slices.partitions[i] > slices.partitions[i - 1]); }
      auto const keys = read_keys(slices.get_data_batches()[i]);
      seen.insert(seen.end(), keys.begin(), keys.end());
      auto& all = rows_of_partition[slices.partitions[i]];
      all.insert(all.end(), keys.begin(), keys.end());
    }
    // Every row is in exactly one slice, in the batch's order.
    REQUIRE(seen == expected_keys);
  };

  {
    rmm::cuda_set_device_raii on_gpu0{rmm::cuda_device_id{0}};
    // Queue work on the boundary task's stream, as the merge of the sample batches and the
    // other tasks of a busy GPU do in a query: what execute() queues runs after it.
    rmm::device_buffer scratch(std::size_t{256} << 20, sample_stream->view());
    for (int i = 0; i < 40; ++i) {
      REQUIRE(cudaMemsetAsync(scratch.data(), i, scratch.size(), sample_stream->value()) ==
              cudaSuccess);
    }
    (void)sample_op.execute(sirius::op::pipelineable_operator_data({sample_batch}),
                            sample_stream->view());
    REQUIRE(sample_op.boundaries_computed());
    REQUIRE(sample_op.get_num_partitions() == partitions);

    // At once, on GPU 1 ...
    {
      rmm::cuda_set_device_raii on_gpu1{rmm::cuda_device_id{1}};
      partition_batch(batch1, stream1->view(), keys1);
    }
    // ... and on GPU 0 on another stream.
    partition_batch(batch0, other_stream0->view(), keys0);
    REQUIRE(cudaStreamSynchronize(sample_stream->value()) == cudaSuccess);
  }

  // Global order across partitions: partition p holds exactly the keys in
  // [boundary p-1, boundary p), so concatenating the partitions in index order is ordered.
  for (std::size_t p = 0; p < partitions; ++p) {
    CAPTURE(p);
    auto const lower = static_cast<std::int64_t>(p * kSampleRows / partitions);
    auto const upper = static_cast<std::int64_t>((p + 1) * kSampleRows / partitions);
    REQUIRE_FALSE(rows_of_partition[p].empty());
    auto const [lo, hi] =
      std::minmax_element(rows_of_partition[p].begin(), rows_of_partition[p].end());
    if (p > 0) { CHECK(*lo >= lower); }
    if (p + 1 < partitions) { CHECK(*hi < upper); }
  }

  partition_op.on_finalize_operator();
  CHECK(sample_op.num_boundary_rows() == 0);
}
