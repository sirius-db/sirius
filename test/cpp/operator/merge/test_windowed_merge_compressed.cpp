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

// The windowed merge over spill-compressed runs: every run batch is a Simpatico-compressed
// representation in the device, host or disk tier and is read by row range through the
// compressed readers, not converted whole to the GPU.

#include "catch.hpp"
#include "creator/task_creator.hpp"
#include "data/convertible_data_batch.hpp"
#include "data/data_batch_utils.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "data/sirius_converter_registry.hpp"
#include "data/sirius_row_range_reader_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/config.hpp"
#include "helper/type_conversions.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/merge/bounded_kway_merge.hpp"
#include "op/merge/merge_chunk_position.hpp"
#include "op/merge/merge_config.hpp"
#include "op/merge/windowed_merge.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator/merge/merge_test_utils.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/completion_handler.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/query_retirement_test_utils.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "pipeline/task_scheduler.hpp"
#include "sirius_config.hpp"
#include "utils/telemetry_utils.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <api/chunk_decode.hpp>
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <compression/compressed_disk_representation.hpp>
#include <compression/compressed_payload_sources.hpp>
#include <compression/compressed_representation.hpp>
#include <compression/device_compressed_blob.hpp>
#include <compression/spill_compression.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <typeindex>
#include <typeinfo>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace sirius::test::merge_utils;
using cucascade::memory::Tier;

namespace {

namespace utils = sirius::test::operator_utils;
namespace pl    = sirius::pipeline;

constexpr std::size_t kMiB = std::size_t{1} << 20;
constexpr std::size_t kGiB = std::size_t{1} << 30;

bool has_gpu()
{
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

template <typename Representation>
bool holds(cucascade::data_batch& batch)
{
  auto ro = batch.to_read_only();
  return dynamic_cast<Representation const*>(ro.get_data()) != nullptr;
}

std::type_index representation_of(cucascade::data_batch& batch)
{
  auto ro          = batch.to_read_only();
  auto const& data = *ro.get_data();
  return std::type_index{typeid(data)};
}

// ---------------------------------------------------------------------------
// Compressed batches in the three tiers
// ---------------------------------------------------------------------------

enum class placement { device, host, disk };

/// Spills the GPU batch @p batch compressed, as a downgrade does, and leaves it in @p where.
void compress_batch(tiers_fixture& fx,
                    const std::shared_ptr<cucascade::data_batch>& batch,
                    placement where)
{
  // The batch was built on the default stream; the converters run on the fixture's.
  cudf::get_default_stream().sync();
  sirius::convertible_data_batch convertible(batch);
  REQUIRE(convertible.convert({fx.host}, fx.stream_ref(), *fx.manager, true).has_value());
  fx.stream.synchronize();
  REQUIRE(holds<sirius::compressed_host_representation>(*batch));
  if (where == placement::host) { return; }

  if (where == placement::disk) {
    REQUIRE(convertible.convert({fx.disk}, fx.stream_ref(), *fx.manager, true).has_value());
    fx.stream.synchronize();
    REQUIRE(holds<sirius::compressed_disk_representation>(*batch));
    return;
  }

  // The device tier: the compressed table rebuilt on the device from the pinned payload.
  auto mut           = batch->to_mutable();
  auto const& host   = mut.get_data()->cast<sirius::compressed_host_representation>();
  auto const& blocks = host.payload();
  auto blob          = std::make_shared<sirius::compressed_device_blob>();
  std::string error;
  simpatico::payload_fetch_fn fetch =
    [&blocks](std::uint64_t offset, std::size_t size, void* dst, ::cuda::stream_ref stream) {
      sirius::copy_pinned_blocks_to_device(blocks, offset, dst, size, stream);
    };
  blob->table = simpatico::read_compressed_table_from_memory(
    host.header(), fetch, fx.stream_ref(), fx.mr(), &error);
  REQUIRE(error.empty());
  fx.stream.synchronize();
  auto device = std::make_unique<sirius::compressed_device_representation>(
    *fx.gpu,
    std::move(blob),
    host.column_names(),
    host.get_size_in_bytes(),
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  mut.set_data(std::move(device));
}

/// The plans of the compressed batches of a test: every node op, and whether a STRING column
/// was stored as is.
struct plan_census {
  std::set<std::string> ops;
  bool stored_string           = false;
  std::size_t nullable_columns = 0;

  void add(cucascade::data_batch& batch)
  {
    auto ro = batch.to_read_only();
    auto const* data =
      dynamic_cast<sirius::simpatico_compressed_representation const*>(ro.get_data());
    REQUIRE(data != nullptr);
    auto const description = sirius::compression::describe(*data);
    for (auto const& column : description->columns) {
      bool encoded_string = false;
      for (auto const& node : column.nodes) {
        ops.insert(node.op);
        encoded_string = encoded_string || node.op == "dictionary" || node.op == "str_split";
      }
      if (column.dtype.id() == cudf::type_id::STRING && !encoded_string) { stored_string = true; }
      if (column.null_count > 0) { ++nullable_columns; }
    }
  }
};

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

merge::merge_key_spec ascending_key(cudf::size_type col = 0)
{
  return {.key_cols        = {col},
          .column_order    = {cudf::order::ASCENDING},
          .null_precedence = {cudf::null_order::AFTER}};
}

/// Gives @p column a null mask: row i is NULL when !valid[i].
void set_validity(cudf::column& column, const std::vector<bool>& valid)
{
  std::vector<std::uint8_t> flags(valid.begin(), valid.end());
  auto bools         = make_column(flags, cudf::type_id::BOOL8);
  auto [mask, nulls] = cudf::bools_to_mask(bools->view());
  column.set_null_mask(std::move(*mask), nulls);
}

std::vector<bool> random_validity(std::mt19937& rng, std::size_t rows)
{
  std::vector<bool> valid(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    valid[r] = rng() % 5 != 0;
  }
  return valid;
}

/// Sorted values with irregular steps of 0..15, so runs share keys and a run repeats some.
std::vector<std::int64_t> sorted_keys(std::mt19937& rng, std::size_t rows)
{
  std::vector<std::int64_t> keys(rows);
  auto value = static_cast<std::int64_t>(rng() % 64);
  for (auto& key : keys) {
    value += static_cast<std::int64_t>(rng() % 16);
    key = value;
  }
  return keys;
}

/// Eight distinct values. A NULL row holds an empty string.
std::vector<std::string> low_cardinality_strings(std::mt19937& rng,
                                                 std::size_t rows,
                                                 const std::vector<bool>& valid = {})
{
  static const char* const kValues[] = {
    "RETURNED", "NONE", "DELIVER IN PERSON", "", "COLLECT COD", "TAKE BACK RETURN", "AIR", "x"};
  std::vector<std::string> strings(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    if (!valid.empty() && !valid[r]) { continue; }
    strings[r] = kValues[rng() % 8];
  }
  return strings;
}

/// Random letters of 0..39 bytes: nearly all distinct, about 1 in 40 empty.
std::vector<std::string> high_cardinality_strings(std::mt19937& rng,
                                                  std::size_t rows,
                                                  const std::vector<bool>& valid = {})
{
  std::vector<std::string> strings(rows);
  for (std::size_t r = 0; r < rows; ++r) {
    if (!valid.empty() && !valid[r]) { continue; }
    strings[r].resize(rng() % 40);
    for (auto& c : strings[r]) {
      c = static_cast<char>('a' + rng() % 26);
    }
  }
  return strings;
}

// Columns of a wide run: one per spill plan, plus nullable variants. A STRING column of a batch
// under 4,096 rows is stored as is.
constexpr std::size_t kKey         = 0;  // INT64 sorted, the last rows NULL -> delta
constexpr std::size_t kRuns        = 1;  // INT32 long runs -> delta -> rle
constexpr std::size_t kRandom      = 2;  // INT64 random -> bitpack
constexpr std::size_t kDict        = 3;  // STRING, 8 distinct values -> dictionary
constexpr std::size_t kSplit       = 4;  // STRING, nearly all distinct -> str_split
constexpr std::size_t kRandomNulls = 5;  // INT32 random, nullable -> bitpack
constexpr std::size_t kDictNulls   = 6;  // STRING low cardinality, nullable -> dictionary
constexpr std::size_t kSplitNulls  = 7;  // STRING high cardinality, nullable -> str_split
constexpr std::size_t kId          = 8;  // INT64, unique over the runs
constexpr std::int64_t kIdStride   = 1'000'000;

const std::vector<std::size_t> kWideCols{0, 1, 2, 3, 4, 5, 6, 7, 8};
const std::vector<cudf::data_type> kWideTypes{cudf::data_type{cudf::type_id::INT64},
                                              cudf::data_type{cudf::type_id::INT32},
                                              cudf::data_type{cudf::type_id::INT64},
                                              cudf::data_type{cudf::type_id::STRING},
                                              cudf::data_type{cudf::type_id::STRING},
                                              cudf::data_type{cudf::type_id::INT32},
                                              cudf::data_type{cudf::type_id::STRING},
                                              cudf::data_type{cudf::type_id::STRING},
                                              cudf::data_type{cudf::type_id::INT64}};

/// One run of the wide schema, sorted by its key (NULLs last): the last @p null_keys keys are
/// NULL.
std::unique_ptr<cudf::table> wide_run(std::size_t run_index,
                                      std::size_t rows,
                                      std::size_t null_keys)
{
  std::mt19937 rng(static_cast<std::uint32_t>(20261112 + run_index));
  std::vector<std::unique_ptr<cudf::column>> columns;

  columns.push_back(make_column(sorted_keys(rng, rows), cudf::type_id::INT64));
  {
    // A mask also when the run has no NULL key.
    std::vector<bool> valid(rows, true);
    std::fill(valid.end() - static_cast<std::ptrdiff_t>(null_keys), valid.end(), false);
    set_validity(*columns.back(), valid);
  }
  {
    std::vector<std::int32_t> runs(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      runs[i] = static_cast<std::int32_t>(i / 500) * 3;
    }
    columns.push_back(make_column(runs, cudf::type_id::INT32));
  }
  {
    std::vector<std::int64_t> random(rows);
    for (auto& v : random) {
      v = static_cast<std::int64_t>((static_cast<std::uint64_t>(rng()) << 9) ^ rng()) - 12345;
    }
    columns.push_back(make_column(random, cudf::type_id::INT64));
  }
  columns.push_back(make_string_column(low_cardinality_strings(rng, rows)));
  columns.push_back(make_string_column(high_cardinality_strings(rng, rows)));
  {
    std::vector<std::int32_t> random(rows);
    for (auto& v : random) {
      v = static_cast<std::int32_t>(rng() & 0xFFFFF);
    }
    columns.push_back(make_column(random, cudf::type_id::INT32));
    set_validity(*columns.back(), random_validity(rng, rows));
  }
  {
    auto const valid = random_validity(rng, rows);
    columns.push_back(make_string_column(low_cardinality_strings(rng, rows, valid)));
    set_validity(*columns.back(), valid);
  }
  {
    auto const valid = random_validity(rng, rows);
    columns.push_back(make_string_column(high_cardinality_strings(rng, rows, valid)));
    set_validity(*columns.back(), valid);
  }
  {
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      ids[i] = static_cast<std::int64_t>(run_index) * kIdStride + static_cast<std::int64_t>(i);
    }
    columns.push_back(make_column(ids, cudf::type_id::INT64));
  }
  return std::make_unique<cudf::table>(std::move(columns));
}

/// An unsorted (key INT64 in [0, key_range), id INT64) table; ids are unique from @p first_id on.
std::unique_ptr<cudf::table> random_rows(std::size_t rows,
                                         std::int64_t key_range,
                                         std::int64_t first_id,
                                         std::uint32_t seed)
{
  std::mt19937 gen(seed);
  std::uniform_int_distribution<std::int64_t> dist(0, key_range - 1);
  std::vector<std::int64_t> keys(rows);
  std::vector<std::int64_t> ids(rows);
  for (std::size_t i = 0; i < rows; ++i) {
    keys[i] = dist(gen);
    ids[i]  = first_id + static_cast<std::int64_t>(i);
  }
  return make_table(make_column(keys, cudf::type_id::INT64),
                    make_column(ids, cudf::type_id::INT64));
}

const std::vector<std::size_t> kTwoCols{0, 1};
const std::vector<cudf::data_type> kTwoInt64{cudf::data_type{cudf::type_id::INT64},
                                             cudf::data_type{cudf::type_id::INT64}};

// ---------------------------------------------------------------------------
// Runs of compressed batches
// ---------------------------------------------------------------------------

/// Rows and batches of one wide run.
struct run_shape {
  std::vector<std::size_t> batch_rows;
  std::size_t null_keys;

  [[nodiscard]] std::size_t rows() const
  {
    return std::accumulate(batch_rows.begin(), batch_rows.end(), std::size_t{0});
  }
};

/// Wide runs whose batches are all compressed, spread over the three tiers.
/// Declared after the fixture: the batches must go before the memory manager does.
struct compressed_runs {
  tiers_fixture& fx;
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  std::vector<merge::merge_input_cursor> cursors;
  std::map<placement, std::size_t> batches_in;
  plan_census plans;
  std::size_t total_rows = 0;

  /// Batch b (counted over all runs) goes to the tier b % 3 for the first three, then to a
  /// random one.
  compressed_runs(tiers_fixture& fixture, const std::vector<run_shape>& shapes, std::uint32_t seed)
    : fx(fixture)
  {
    std::mt19937 rng(seed);
    std::size_t batch_index = 0;
    for (std::size_t r = 0; r < shapes.size(); ++r) {
      auto const& shape = shapes[r];
      auto const table  = wide_run(r, shape.rows(), shape.null_keys);
      runs.push_back(make_gpu_run(repository, *fx.gpu, table->view(), shape.batch_rows));
      for (auto const& batch : runs.back().batches) {
        auto const where =
          static_cast<placement>(batch_index < 3 ? batch_index : std::size_t{rng() % 3});
        compress_batch(fx, batch, where);
        ++batches_in[where];
        plans.add(*batch);
        ++batch_index;
      }
      total_rows += shape.rows();
    }
    cursors = make_cursors(runs);
  }

  /// The representation type of every batch, in run order.
  [[nodiscard]] std::vector<std::type_index> representations() const
  {
    std::vector<std::type_index> types;
    for (auto const& run : runs) {
      for (auto const& batch : run.batches) {
        types.push_back(representation_of(*batch));
      }
    }
    return types;
  }

  merge::windowed_merger make_merger(
    const merge::merge_key_spec& keys,
    merge::window_config cfg,
    std::vector<std::size_t> fetch_cols  = kWideCols,
    std::vector<cudf::data_type> types   = kWideTypes,
    merge::bounded_merge_fn merge_kernel = merge::bounded_kway_merge)
  {
    return merge::windowed_merger(run_batches(runs),
                                  cursors,
                                  std::move(fetch_cols),
                                  std::move(types),
                                  keys,
                                  cfg,
                                  sirius::row_range_reader_registry::get(),
                                  *fx.gpu,
                                  std::move(merge_kernel));
  }
};

// ---------------------------------------------------------------------------
// The operator in a real pipeline, with its input spilled by the downgrade executor
// ---------------------------------------------------------------------------

/// Longest a merge may take before the test gives up on it.
constexpr std::chrono::seconds kWatchdog{20};

/// Pipeline sink with the materialized result collector's publication contract: one batch per
/// positioned publication, every position at most once. Keeps the chunks by position.
class ordered_collector final : public sirius_physical_operator {
 public:
  ordered_collector()
    : sirius_physical_operator(SiriusPhysicalOperatorType::RESULT_COLLECTOR, {}, 0)
  {
  }

  sirius::OrderPreservationType output_order_preservation() const override
  {
    return sirius::OrderPreservationType::FIXED_ORDER;
  }

  bool is_sink() const override { return true; }

  void sink(const operator_data& input, ::cuda::stream_ref) override
  {
    auto const& data    = dynamic_cast<const pipelineable_operator_data&>(input);
    auto const& batches = data.get_data_batches();
    if (batches.empty()) { return; }
    auto const position = data.get_position();
    if (!position) {
      throw sirius::internal_exception("MERGE_SORT published a result without a position");
    }
    if (batches.size() != 1) {
      throw sirius::internal_exception("Positioned result requires one output batch");
    }
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_chunks.emplace(*position, batches.front()).second) {
      throw sirius::internal_exception("Duplicate query result position");
    }
  }

  /// The published chunks, by position.
  [[nodiscard]] std::map<std::size_t, std::shared_ptr<cucascade::data_batch>> chunks() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _chunks;
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _chunks.clear();
  }

 private:
  mutable std::mutex _mutex;
  std::map<std::size_t, std::shared_ptr<cucascade::data_batch>> _chunks;
};

/// The upstream SORT_PARTITION pipeline: already finished, so the FULL barrier is open.
class finished_pipeline final : public pl::sirius_pipeline {
 public:
  finished_pipeline() : sirius_pipeline{pl::pipeline_build_context{nullptr}} {}
  [[nodiscard]] bool is_pipeline_finished() const override { return true; }
};

struct order_by_harness;

/// Task creator whose requests are served by the harness, the way a creation worker serves them.
class pumping_task_creator final : public sirius::creator::task_creator {
 public:
  pumping_task_creator(sirius::exec::query_lifecycle_registry& lifecycle,
                       sirius::memory::sirius_memory_reservation_manager& manager,
                       order_by_harness& harness)
    : task_creator(lifecycle, sirius::creator::task_creator_config{}, manager), _harness(harness)
  {
  }

  void schedule(sirius_physical_operator* request) override;

 private:
  order_by_harness& _harness;
};

/**
 * @brief A windowed ORDER BY over (key BIGINT, id BIGINT, tag VARCHAR): the MERGE_SORT is the
 * source of a real pipeline whose sink is an ordered collector, with a real task scheduler and
 * GPU executor, and its input repository is registered for the downgrade executor.
 *
 * Only the task creator's queue is replaced, by a pump thread that does what a creation worker
 * does with each `schedule(op)` request.
 */
struct order_by_harness {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu = nullptr;
  sirius::data::data_repository_manager_registry repo_registry;
  /// Owned by the query's repository manager.
  cucascade::shared_data_repository* input = nullptr;
  std::unique_ptr<sirius_physical_merge_sort> merge;
  ordered_collector collector;
  std::shared_ptr<finished_pipeline> producer = std::make_shared<finished_pipeline>();
  std::shared_ptr<pl::sirius_pipeline> pipeline;
  std::shared_ptr<pl::completion_handler> completion = std::make_shared<pl::completion_handler>();
  std::future<void> completion_result                = completion->get_awaitable();
  std::shared_ptr<pl::sirius_pipeline_task_global_state> global;
  sirius::exec::query_lifecycle_registry lifecycle;
  std::unique_ptr<pl::task_scheduler> scheduler;
  std::unique_ptr<pumping_task_creator> creator;
  /// The rows of each run, in run order (the oracle's input).
  std::vector<std::unique_ptr<cudf::table>> run_rows;

  order_by_harness()
    : manager{utils::initialize_memory_manager(1,
                                               {.gpu_bytes                = 512 * kMiB,
                                                .gpu_reservation_fraction = 0.9,
                                                .divide_among_gpus        = false,
                                                .per_stream_tracking      = false})}
  {
    gpu = const_cast<cucascade::memory::memory_space*>(manager->get_memory_space(Tier::GPU, 0));
    REQUIRE(gpu != nullptr);
    if (!sirius::row_range_reader_registry::is_initialized()) {
      sirius::row_range_reader_registry::initialize();
    }

    duckdb::vector<duckdb::LogicalType> types{
      duckdb::LogicalType::BIGINT, duckdb::LogicalType::BIGINT, duckdb::LogicalType::VARCHAR};
    duckdb::vector<duckdb::BoundOrderByNode> orders;
    orders.push_back(duckdb::BoundOrderByNode(
      duckdb::OrderType::ASCENDING,
      duckdb::OrderByNullType::NULLS_LAST,
      duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0)));
    merge = std::make_unique<sirius_physical_merge_sort>(sirius::from_duckdb_vec(std::move(types)),
                                                         std::move(orders),
                                                         duckdb::vector<std::size_t>{0, 1, 2},
                                                         0);
    merge->operator_id    = 1;
    collector.operator_id = 2;

    pipeline = std::make_shared<pl::sirius_pipeline>(pl::pipeline_build_context{nullptr});
    pipeline->set_pipeline_id(7);
    pl::sirius_pipeline_build_state state;
    state.set_pipeline_source(*pipeline, *merge);
    state.set_pipeline_operators(*pipeline, {*merge});
    state.set_pipeline_sink(*pipeline, &collector, 1);
    merge->set_pipeline(pipeline);

    // What repository wiring does for the input of a windowed sort.
    auto owner             = std::make_unique<cucascade::shared_data_repository>();
    input                  = owner.get();
    auto const policy      = merge->input_residency_policy("default");
    auto upstream          = std::make_unique<sirius_physical_operator::port>();
    upstream->type         = MemoryBarrierType::FULL;
    upstream->repo         = input;
    upstream->src_pipeline = producer;
    upstream->residency    = policy;
    merge->add_port("default", std::move(upstream));
    input->set_downgrade_priority(policy.downgrade_priority);
    repo_registry.create_for_query(pipeline->get_query_id())
      ->add_new_repository(merge->operator_id, "default", std::move(owner));

    global = std::make_shared<pl::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
    global->set_completion_handler(completion);

    sirius::exec::thread_pool_config config;
    config.num_threads        = 1;
    config.thread_name_prefix = "merge-compressed";
    lifecycle.open_query(pipeline->get_query_id());
    scheduler = std::make_unique<pl::task_scheduler>(
      lifecycle, config, *manager, sirius::test::make_test_telemetry_context());
    creator = std::make_unique<pumping_task_creator>(lifecycle, *manager, *this);
    creator->set_task_scheduler(*scheduler);
    scheduler->set_task_creator(*creator);
    pipeline->set_task_creator(creator.get());
    scheduler->start();
    _pump = std::thread([this] { pump(); });
  }

  ~order_by_harness()
  {
    stop_pump();
    if (!_retired) {
      try {
        sirius::test::retire_scheduler_query(
          lifecycle, *scheduler, pipeline->get_query_id(), false, creator.get());
      } catch (...) {
      }
    }
    scheduler->stop();
    // Task output buffers retain executor streams, so free them before the scheduler's stream pool.
    collector.clear();
    while (input->pop_next_data_batch(0)) {}
  }

  /// Applies the sort_merge_* settings the way the plan generator does, from operator_params.
  void configure(const sirius::operator_params& params)
  {
    merge->set_merge_settings(merge::merge_settings::from(params));
    merge->set_device_budget_provider([](const cucascade::memory::memory_space*) {
      return merge::device_budget{
        .gpu_available_bytes = 8 * kGiB, .gpu_total_bytes = 8 * kGiB, .concurrent_tasks = 1};
    });
  }

  /// Adds one sorted run of @p rows rows as a GPU batch: keys shared with the other runs, ids
  /// unique over the runs and a tag of random letters.
  void add_run(std::size_t run_index, std::size_t rows)
  {
    std::mt19937 rng(static_cast<std::uint32_t>(977 + run_index));
    std::vector<std::int64_t> ids(rows);
    for (std::size_t i = 0; i < rows; ++i) {
      ids[i] = static_cast<std::int64_t>(run_index) * kIdStride + static_cast<std::int64_t>(i);
    }
    auto table  = make_table(make_column(sorted_keys(rng, rows), cudf::type_id::INT64),
                            make_column(ids, cudf::type_id::INT64),
                            make_string_column(high_cardinality_strings(rng, rows)));
    auto stream = utils::default_stream();
    auto batch  = sirius::make_data_batch(
      std::make_unique<cudf::table>(table->view(), stream, utils::get_resource_ref(*gpu)),
      *gpu,
      stream,
      sirius::telemetry::batch_telemetry_info{});
    stream.sync();
    input->add_data_batch(batch);
    _inputs.push_back(batch);
    run_rows.push_back(std::move(table));
  }

  /// Input runs that hold a @p Representation right now.
  template <typename Representation>
  [[nodiscard]] std::size_t count_inputs_holding() const
  {
    std::size_t count = 0;
    for (auto const& weak : _inputs) {
      auto batch = weak.lock();
      REQUIRE(batch);
      if (holds<Representation>(*batch)) { ++count; }
    }
    return count;
  }

  /// Has the downgrade executor of the GPU space spill every input run.
  void spill_inputs()
  {
    std::size_t input_bytes = 0;
    for (auto const& weak : _inputs) {
      auto batch = weak.lock();
      REQUIRE(batch);
      input_bytes += sirius::get_batch_rows_and_bytes(*batch).bytes;
    }
    sirius::exec::downgrade_executor_config config{
      .thread_pool    = {.num_threads = 1, .thread_name_prefix = "downgrade"},
      .monitor_period = std::chrono::milliseconds{0}};
    sirius::parallel::downgrade_executor downgrader(
      lifecycle,
      config,
      repo_registry,
      cucascade::memory::memory_space_id(Tier::GPU, 0),
      gpu,
      *manager);
    downgrader.start();
    for (int request = 0; request < 8; ++request) {
      if (count_inputs_holding<cucascade::gpu_table_representation>() == 0) { break; }
      downgrader.request_free_memory_and_wait(input_bytes);
    }
    downgrader.stop();
  }

  /// Hands the MERGE_SORT to the task creator, as the upstream pipeline does when it finishes,
  /// and waits under a watchdog until the pipeline has finished.
  void run()
  {
    creator->schedule(merge.get());
    auto const deadline = std::chrono::steady_clock::now() + kWatchdog;
    while (!pipeline->is_pipeline_finished() && !completion->has_error()) {
      if (std::chrono::steady_clock::now() >= deadline) {
        FAIL("the merge did not finish within the watchdog: "
             << pipeline->get_tasks_created() << " tasks created, "
             << pipeline->get_tasks_completed() << " completed, " << input->total_size()
             << " batches left in the input repository");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds{2});
    }
    if (completion->has_error()) {
      std::string what = "unknown error";
      if (completion_result.wait_for(std::chrono::seconds{5}) == std::future_status::ready) {
        try {
          completion_result.get();
        } catch (const std::exception& e) {
          what = e.what();
        } catch (...) {
        }
      }
      FAIL("a merge task failed: " << what);
    }
    REQUIRE(pipeline->get_tasks_created() == pipeline->get_tasks_completed());
    stop_pump();
    sirius::test::retire_scheduler_query(
      lifecycle, *scheduler, pipeline->get_query_id(), true, creator.get());
    _retired = true;
  }

  /// The collected chunks, concatenated in position order.
  [[nodiscard]] std::unique_ptr<cudf::table> collected() const
  {
    std::vector<std::unique_ptr<cudf::table>> tables;
    std::vector<cudf::table_view> views;
    std::size_t next_chunk = 0;
    for (auto const& [position, batch] : collector.chunks()) {
      REQUIRE(position == merge::make_merge_chunk_position(0, next_chunk));
      ++next_chunk;
      tables.push_back(std::make_unique<cudf::table>(sirius::get_cudf_table_view(*batch)));
      views.push_back(tables.back()->view());
    }
    REQUIRE_FALSE(views.empty());
    return cudf::concatenate(views);
  }

  /// The runs concatenated in run order, then stably sorted by the key.
  [[nodiscard]] std::unique_ptr<cudf::table> expected() const
  {
    std::vector<cudf::table_view> views;
    for (auto const& rows : run_rows) {
      views.push_back(rows->view());
    }
    return expected_full_sort(std::span<const cudf::table_view>(views), ascending_key());
  }

  /// Requires that every input batch left the repository and was freed.
  void require_inputs_popped()
  {
    REQUIRE(input->total_size() == 0);
    for (auto const& batch : _inputs) {
      REQUIRE(batch.expired());
    }
  }

  // Called by pumping_task_creator::schedule, from any thread.
  void request(sirius_physical_operator* node)
  {
    // The pipeline's consumers are notified through the same creator; only the MERGE_SORT has
    // tasks to create here.
    if (node != merge.get()) { return; }
    {
      std::lock_guard<std::mutex> lock(_pump_mutex);
      ++_pending;
    }
    _pump_wake.notify_one();
  }

 private:
  void pump()
  {
    while (true) {
      {
        std::unique_lock<std::mutex> lock(_pump_mutex);
        _pump_wake.wait(lock, [this] { return _pending > 0 || _stop; });
        if (_stop) { return; }
        --_pending;
      }
      try {
        create_tasks();
      } catch (...) {
        completion->report_error(std::current_exception());
      }
    }
  }

  /// One creation worker's turn for the MERGE_SORT (task_creator::manager_loop).
  void create_tasks()
  {
    auto const hint = merge->get_next_task_hint();
    if (hint.has_value() && hint->hint == TaskCreationHint::READY) {
      while (!merge->all_ports_empty()) {
        auto task_lock = pipeline->get_task_creation_lock();
        if (pipeline->is_pipeline_finished()) { break; }
        auto data          = merge->get_next_task_input_data();
        auto* pipelineable = dynamic_cast<pipelineable_operator_data*>(data.get());
        if (!data || (pipelineable && pipelineable->get_data_batches().empty())) { break; }
        auto const device = data->get_preferred_device_id();
        auto local        = std::make_unique<pl::gpu_pipeline_task_local_state>(std::move(data));
        local->set_preferred_device_id(device.value_or(0));
        auto task =
          std::make_unique<pl::gpu_pipeline_task>(creator->get_next_task_id(),
                                                  std::vector<cucascade::shared_data_repository*>{},
                                                  std::move(local),
                                                  global);
        task_lock.unlock();
        scheduler->schedule(std::move(task));
      }
    }
    pipeline->update_pipeline_status(false);
  }

  void stop_pump()
  {
    {
      std::lock_guard<std::mutex> lock(_pump_mutex);
      _stop = true;
    }
    _pump_wake.notify_one();
    if (_pump.joinable()) { _pump.join(); }
  }

  std::vector<std::weak_ptr<cucascade::data_batch>> _inputs;
  std::mutex _pump_mutex;
  std::condition_variable _pump_wake;
  std::size_t _pending = 0;
  bool _stop           = false;
  bool _retired        = false;
  std::thread _pump;
};

void pumping_task_creator::schedule(sirius_physical_operator* request)
{
  _harness.request(request);
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. Every spill plan, every tier
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger merges compressed runs of every spill plan in the three tiers",
          "[windowed_merge_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  scoped_spill_compression compression;
  // Six runs, ten batches. The 3,000- and 4,000-row batches store their STRING columns as is.
  compressed_runs runs(fx,
                       {{.batch_rows = {20500}, .null_keys = 300},
                        {.batch_rows = {4500, 4500}, .null_keys = 0},
                        {.batch_rows = {3000}, .null_keys = 40},
                        {.batch_rows = {4000, 4000, 4000}, .null_keys = 5000},
                        {.batch_rows = {5000}, .null_keys = 0},
                        {.batch_rows = {9000, 7000}, .null_keys = 1}},
                       1);
  REQUIRE(runs.runs.size() == 6);
  CHECK(runs.batches_in[placement::device] > 0);
  CHECK(runs.batches_in[placement::host] > 0);
  CHECK(runs.batches_in[placement::disk] > 0);
  for (auto const* op : {"delta", "rle", "bitpack", "dictionary", "str_split"}) {
    CAPTURE(op);
    CHECK(runs.plans.ops.contains(op));
  }
  CHECK(runs.plans.stored_string);
  CHECK(runs.plans.nullable_columns > 0);
  auto const representations = runs.representations();
  auto const keys            = ascending_key();
  auto const expected        = expected_full_sort(runs.runs, keys);

  // 3,000-row windows start and end inside a 1,024-row chunk.
  auto merger = runs.make_merger(keys, {.window_rows = 3000, .output_target_bytes = 512 * 1024});
  merge::merge_round_stats stats;
  auto const collected = collect_full_sort(merger, runs.cursors, fx.stream_view(), fx.mr(), &stats);

  CHECK(collected.chunks.size() > 3);
  REQUIRE(collected.table->num_rows() == expected->num_rows());
  CHECK(tables_equal(collected.table->view(), expected->view()));
  CHECK(all_exhausted(runs.cursors));
  CHECK(stats.rows_out == runs.total_rows);
  // Every row came through a window buffer, once, and no batch was converted whole.
  CHECK(stats.fallback_conversions == 0);
  CHECK(stats.refill_rows == runs.total_rows);
  CHECK(runs.representations() == representations);
}

// ---------------------------------------------------------------------------
// 2. A plan the readers refuse: whole-batch fallback
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger converts a compressed run once when its plan cannot be read by rows",
          "[windowed_merge_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  scoped_spill_compression compression;
  auto const keys = ascending_key();
  cucascade::shared_data_repository repository;
  std::vector<test_run> runs;
  runs.push_back(
    make_sorted_gpu_run(repository, *fx.gpu, random_rows(6000, 40000, 0, 21)->view(), keys));
  runs.push_back(
    make_sorted_gpu_run(repository, *fx.gpu, random_rows(5500, 40000, 100000, 22)->view(), keys));
  compress_batch(fx, runs[1].batches[0], placement::host);
  {
    // Column 0 with a spill plan, column 1 with a plan the spill never picks.
    auto sorted = stable_sort(random_rows(5000, 40000, 200000, 23)->view(), keys);
    const std::string plan =
      "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
      "---\n"
      "input -> bitcomp\n";
    auto blob   = std::make_shared<sirius::compressed_device_blob>();
    blob->table = simpatico::compress_with_plan(sorted->view(), plan, fx.stream_ref(), fx.mr(), {});
    fx.stream.synchronize();
    auto batch =
      cucascade::data_batch::make(sirius::get_next_batch_id(),
                                  std::make_unique<sirius::compressed_device_representation>(
                                    *fx.gpu,
                                    std::move(blob),
                                    std::vector<std::string>{"key", "id"},
                                    sorted->alloc_size(),
                                    sorted->alloc_size(),
                                    sorted->num_rows()));
    repository.add_data_batch(batch);
    test_run run;
    run.run.batch_ids.push_back(batch->get_batch_id());
    run.run.batch_rows.push_back(static_cast<std::size_t>(sorted->num_rows()));
    run.batches.push_back(std::move(batch));
    run.rows = std::move(sorted);
    runs.push_back(std::move(run));
  }
  auto& refused       = *runs[2].batches[0];
  auto const& readers = sirius::row_range_reader_registry::get();
  // The representation has a reader; it is this batch's plan the reader refuses.
  REQUIRE(readers.has_reader(*refused.to_read_only().get_data()));
  auto const id       = refused.get_batch_id();
  auto const expected = expected_full_sort(runs, keys);

  auto cursors = make_cursors(runs);
  merge::windowed_merger merger(run_batches(runs),
                                cursors,
                                kTwoCols,
                                kTwoInt64,
                                keys,
                                {.window_rows = 1024, .output_target_bytes = 32 * 1024},
                                readers,
                                *fx.gpu);
  merge::merge_round_stats stats;
  auto const collected = collect_full_sort(merger, cursors, fx.stream_view(), fx.mr(), &stats);

  CHECK(collected.chunks.size() > 3);
  REQUIRE(collected.table->num_rows() == expected->num_rows());
  CHECK(tables_equal(collected.table->view(), expected->view()));
  // Exactly the refused batch was converted, in place; the readable compressed run was not.
  CHECK(stats.fallback_conversions == 1);
  CHECK(holds<cucascade::gpu_table_representation>(refused));
  CHECK(refused.to_read_only().get_memory_space() == fx.gpu);
  CHECK(refused.get_batch_id() == id);
  CHECK(repository.get_data_batch_by_id(id, 0).get() == &refused);
  CHECK(holds<sirius::compressed_host_representation>(*runs[1].batches[0]));
  CHECK(stats.refill_rows == 5500);
}

// ---------------------------------------------------------------------------
// 3. Operator level: runs spilled by the downgrade executor with compression on
// ---------------------------------------------------------------------------

namespace {

/// A windowed ORDER BY over five runs the downgrade executor spilled compressed and, with
/// @p gpu_runs, that many more runs still on the GPU.
void check_order_by_over_spilled_runs(std::size_t gpu_runs)
{
  constexpr std::size_t spilled_runs = 5;
  order_by_harness test;
  sirius::operator_params params;
  params.sort_merge_mode_setting = sirius::sort_merge_mode::windowed;
  params.sort_merge_window_rows  = 1024;
  params.sort_merge_output_bytes = 256 * 1024;
  test.configure(params);
  std::size_t spilled_rows = 0;
  for (std::size_t run = 0; run < spilled_runs; ++run) {
    auto const rows = 6000 + 700 * run;
    test.add_run(run, rows);
    spilled_rows += rows;
  }
  {
    // The settings of `spill_compression = true` with `spill_min_batch_bytes` lowered: the
    // size and headroom checks of the encode keep their defaults.
    scoped_spill_compression compression{
      sirius::compression::spill_compression_settings{}.max_compressed_fraction};
    test.spill_inputs();
  }
  REQUIRE(test.count_inputs_holding<sirius::compressed_host_representation>() == spilled_runs);
  std::size_t total_rows = spilled_rows;
  for (std::size_t run = spilled_runs; run < spilled_runs + gpu_runs; ++run) {
    test.add_run(run, 5000);
    total_rows += 5000;
  }
  auto const expected = test.expected();

  test.run();

  auto const counters = test.merge->counters();
  REQUIRE(counters.windowed_partitions == 1);
  CHECK(counters.one_shot_partitions == 0);
  CHECK(counters.intermediate_tasks == 0);
  CHECK(counters.mergers_built == 1);
  CHECK(counters.rounds.rows_out == total_rows);
  // Every spilled run was read by row range from its compressed form: none was converted
  // whole, and no row was read twice. The GPU runs are sliced, not read.
  CHECK(counters.rounds.fallback_conversions == 0);
  CHECK(counters.rounds.refill_rows >= spilled_rows);
  CHECK(counters.rounds.refill_rows <= spilled_rows + spilled_rows / 100);

  auto const merged = test.collected();
  REQUIRE(merged->num_rows() == expected->num_rows());
  CHECK(tables_equal(merged->view(), expected->view()));
  CHECK(test.collector.chunks().size() > 3);
  CHECK(test.pipeline->is_pipeline_finished());
  test.require_inputs_popped();
}

}  // namespace

TEST_CASE("windowed ORDER BY reads the runs the downgrade executor spilled compressed",
          "[windowed_merge_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  // One run stays on the GPU and is sliced in place next to the compressed ones.
  check_order_by_over_spilled_runs(1);
}

TEST_CASE("windowed ORDER BY reads a partition whose runs were all spilled compressed",
          "[windowed_merge_compressed][windowed_merge_all_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  // No run exposes a table: the operator takes the partition's schema from a compressed header.
  check_order_by_over_spilled_runs(0);
}

// ---------------------------------------------------------------------------
// 4. Windows of one chunk: reads cross chunk boundaries all the time
// ---------------------------------------------------------------------------

TEST_CASE("windowed_merger reads compressed runs through 1024-row windows",
          "[windowed_merge_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  constexpr std::size_t window_rows = 1024;
  tiers_fixture fx;
  scoped_spill_compression compression;
  // Batches that neither start nor end on a 1,024-row chunk of their run's rows.
  compressed_runs runs(fx,
                       {{.batch_rows = {5000, 4321}, .null_keys = 100},
                        {.batch_rows = {10007}, .null_keys = 0},
                        {.batch_rows = {4097, 1023, 4500}, .null_keys = 2000},
                        {.batch_rows = {8191}, .null_keys = 7}},
                       2);
  CHECK(runs.batches_in[placement::device] > 0);
  CHECK(runs.batches_in[placement::host] > 0);
  CHECK(runs.batches_in[placement::disk] > 0);
  auto const representations = runs.representations();

  // A projection in another order than the source's, with the key second.
  const std::vector<std::size_t> fetch_cols{kId, kKey, kSplitNulls, kRandomNulls, kDict};
  std::vector<cudf::data_type> fetch_types;
  std::vector<cudf::size_type> select;
  for (auto const col : fetch_cols) {
    fetch_types.push_back(kWideTypes[col]);
    select.push_back(static_cast<cudf::size_type>(col));
  }
  auto const expected = expected_full_sort(runs.runs, ascending_key(kKey));

  std::size_t rounds         = 0;
  cudf::size_type max_window = 0;
  auto const counting_kernel = [&](std::span<const cudf::table_view> windows,
                                   const merge::merge_key_spec& key_spec,
                                   std::span<const std::uint8_t> truncated,
                                   cudf::size_type max_rows,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr) {
    ++rounds;
    for (auto const& window : windows) {
      max_window = std::max(max_window, window.num_rows());
    }
    return merge::bounded_kway_merge(windows, key_spec, truncated, max_rows, stream, mr);
  };
  auto merger = runs.make_merger(ascending_key(1),
                                 {.window_rows = window_rows, .output_target_bytes = 64 * 1024},
                                 fetch_cols,
                                 fetch_types,
                                 counting_kernel);
  merge::merge_round_stats stats;
  auto const collected = collect_full_sort(merger, runs.cursors, fx.stream_view(), fx.mr(), &stats);

  REQUIRE(collected.table->num_rows() == expected->num_rows());
  CHECK(tables_equal(collected.table->view(), expected->view().select(select)));
  CHECK(all_exhausted(runs.cursors));
  CHECK(max_window == window_rows);
  // A round emits at most one window per run, so the merge takes many rounds and refills.
  CHECK(stats.rounds == rounds);
  CHECK(rounds >= runs.total_rows / (runs.runs.size() * window_rows));
  CHECK(collected.chunks.size() > 3);
  CHECK(stats.rows_out == runs.total_rows);
  CHECK(stats.fallback_conversions == 0);
  CHECK(stats.refill_rows == runs.total_rows);
  CHECK(runs.representations() == representations);
}

// ---------------------------------------------------------------------------
// 5. Where the readers' device memory comes from
// ---------------------------------------------------------------------------

namespace {

/// Makes @p resource the current device resource until the end of the scope.
class current_resource_guard {
 public:
  explicit current_resource_guard(rmm::device_async_resource_ref resource)
    : _previous(cudf::set_current_device_resource(
        ::cuda::mr::any_resource<::cuda::mr::device_accessible>{resource}))
  {
  }
  current_resource_guard(const current_resource_guard&)            = delete;
  current_resource_guard& operator=(const current_resource_guard&) = delete;
  ~current_resource_guard() { (void)cudf::set_current_device_resource(std::move(_previous)); }

 private:
  ::cuda::mr::any_resource<::cuda::mr::device_accessible> _previous;
};

}  // namespace

TEST_CASE("windowed_merger keeps the compressed readers' memory in the task's GPU space",
          "[windowed_merge_compressed]")
{
  if (!has_gpu()) { SKIP("No GPU available"); }
  tiers_fixture fx;
  scoped_spill_compression compression;
  // Dictionary, bitpack and delta columns: their readers keep decode state on the device from
  // one refill to the next, and so from one run() to the next.
  compressed_runs runs(fx,
                       {{.batch_rows = {9000}, .null_keys = 50},
                        {.batch_rows = {7000}, .null_keys = 0},
                        {.batch_rows = {8000}, .null_keys = 3}},
                       3);
  CHECK(runs.plans.ops.contains("dictionary"));
  auto const keys     = ascending_key();
  auto const expected = expected_full_sort(runs.runs, keys);
  fx.stream.synchronize();

  // What the merge leaves allocated in the current device resource, the one a reader without a
  // scratch reservation uses.
  rmm::mr::statistics_resource_adaptor current{
    ::cuda::mr::any_resource<::cuda::mr::device_accessible>{
      cudf::get_current_device_resource_ref()}};
  auto const available_before = fx.gpu->get_available_memory();
  std::vector<cudf::table_view> views;
  std::vector<std::unique_ptr<cudf::table>> chunks;
  {
    auto merger = runs.make_merger(keys, {.window_rows = 1024, .output_target_bytes = 1});
    merge::merge_round_stats stats;
    {
      const current_resource_guard guard{rmm::device_async_resource_ref{current}};
      while (!all_exhausted(runs.cursors)) {
        REQUIRE(chunks.size() < 1000);
        chunks.push_back(merger.run(fx.stream_view(), fx.mr(), &stats));
        views.push_back(chunks.back()->view());
        // Between two run() calls the merger holds its windows and the readers' state: all of
        // it in the GPU space, none in the current device resource.
        CHECK(current.get_bytes_counter().value == 0);
        if (!all_exhausted(runs.cursors)) {
          CHECK(fx.gpu->get_available_memory() < available_before);
        }
      }
    }
    CHECK(chunks.size() > 3);
    CHECK(stats.fallback_conversions == 0);
    CHECK(stats.refill_rows == runs.total_rows);
    auto const merged = cudf::concatenate(views, fx.stream_view(), fx.mr());
    REQUIRE(merged->num_rows() == expected->num_rows());
    CHECK(tables_equal(merged->view(), expected->view()));
    views.clear();
    chunks.clear();
  }
  // The merger is gone, and with it every window and reader state.
  fx.stream.synchronize();
  CHECK(fx.gpu->get_available_memory() == available_before);
}
