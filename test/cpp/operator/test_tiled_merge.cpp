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

#include "catch.hpp"
#include "data/convertible_data_batch.hpp"
#include "op/sirius_physical_merge_sort.hpp"
#include "operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/oom_reschedule_exception.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "utils/log_test_utils.hpp"
#include "utils/telemetry_utils.hpp"
#include "utils/test_physical_operator.hpp"

#include <future>
#include <numeric>
#include <set>

using namespace sirius::op;
using namespace sirius::test::operator_utils;
namespace {
duckdb::vector<duckdb::BoundOrderByNode> ascending()
{
  duckdb::vector<duckdb::BoundOrderByNode> orders;
  orders.emplace_back(
    duckdb::OrderType::ASCENDING,
    duckdb::OrderByNullType::NULLS_LAST,
    duckdb::make_uniq<duckdb::BoundReferenceExpression>(duckdb::LogicalType::BIGINT, 0));
  return orders;
}
class injected_merge : public sirius_physical_merge_sort {
 public:
  using sirius_physical_merge_sort::sirius_physical_merge_sort;
  bool fail_compute{false};
  std::unique_ptr<operator_data> execute(operator_data const& input,
                                         ::cuda::stream_ref stream) override
  {
    auto output = sirius_physical_merge_sort::execute(input, stream);
    if (std::exchange(fail_compute, false)) {
      throw rmm::out_of_memory("injected before publication");
    }
    return output;
  }
};

struct tiled_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu;
  rmm::cuda_stream stream;
  cucascade::shared_data_repository_manager repositories;
  injected_merge merge{{sirius::logical_type::make(sirius::type_id::BIGINT)}, ascending(), {0}, 0};
  sirius::op::test_physical_operator destination{SiriusPhysicalOperatorType::FILTER, {}, 0};
  cucascade::shared_data_repository output;
  cucascade::shared_data_repository* input;
  std::shared_ptr<sirius::pipeline::sirius_pipeline> pipeline;
  std::shared_ptr<sirius::pipeline::sirius_pipeline_task_global_state> global;
  uint64_t task_id{0};
  std::size_t retry_floor{0};

  explicit tiled_fixture(std::size_t gpus = 1, memory_manager_options options = {})
    : manager(sirius::test::operator_utils::initialize_memory_manager(gpus, options)),
      gpu(manager->get_memory_space(cucascade::memory::Tier::GPU, 0))
  {
    merge.operator_id       = 1;
    merge.chunk_bytes       = 24;
    merge.reservation_limit = gpu->get_max_memory();
    repositories.add_new_repository(
      1, "default", std::make_unique<cucascade::shared_data_repository>());
    input = repo("default");
    merge.register_scratch(repositories);
    merge.add_port("default",
                   std::make_unique<sirius_physical_operator::port>(
                     MemoryBarrierType::FULL, input, nullptr, nullptr));
    destination.add_port("default",
                         std::make_unique<sirius_physical_operator::port>(
                           MemoryBarrierType::FULL, &output, nullptr, nullptr));
    merge.add_next_port_after_sink({&destination, "default"});
    pipeline = std::make_shared<sirius::pipeline::sirius_pipeline>(
      sirius::pipeline::pipeline_build_context{nullptr});
    pipeline->set_pipeline_id(0);
    sirius::pipeline::sirius_pipeline_build_state build;
    build.set_pipeline_source(*pipeline, merge);
    build.add_pipeline_operator(*pipeline, merge);
    build.set_pipeline_sink(*pipeline, &merge, 0);
    global = std::make_shared<sirius::pipeline::sirius_pipeline_task_global_state>(
      pipeline, sirius::test::make_test_telemetry_context());
  }
  cucascade::shared_data_repository* repo(char const* name)
  {
    return repositories.get_repository(1, name).get();
  }
  void add(std::vector<int64_t> values, std::size_t range = 0)
  {
    input->add_data_batch(make_numeric_batch<int64_t>(*gpu, values, cudf::type_id::INT64), range);
  }
  std::unique_ptr<sirius::pipeline::gpu_pipeline_task> task(
    std::unique_ptr<operator_data> data, cucascade::memory::memory_space* target = nullptr)
  {
    if (!target) { target = gpu; }
    auto task = std::make_unique<sirius::pipeline::gpu_pipeline_task>(
      ++task_id,
      repositories.get_repositories(),
      std::make_unique<sirius::pipeline::gpu_pipeline_task_local_state>(std::move(data)),
      global);
    auto info             = task->get_estimated_reservation_size_info(target);
    info.reservation_size = std::max(info.reservation_size, retry_floor);
    auto reservation =
      target->make_reservation_or_null(std::min(info.reservation_size, target->get_max_memory()));
    if (!reservation) {
      // Model downgrade before reservation; ordinary preparation restores only
      // this task's inputs afterward.
      auto candidates = repositories.get_repositories();
      candidates.push_back(&output);
      auto* host = manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
      for (auto* repository : candidates) {
        sirius::convertible_data_batch_provider provider(repository);
        for (auto& candidate : provider.get_all_convertible(target, true, false)) {
          REQUIRE(candidate->convert({host}, stream, *manager, true).has_value());
          info                  = task->get_estimated_reservation_size_info(target);
          info.reservation_size = std::max(info.reservation_size, retry_floor);
          reservation           = target->make_reservation_or_null(
            std::min(info.reservation_size, target->get_max_memory()));
          if (reservation) { break; }
        }
        if (reservation) { break; }
      }
    }
    REQUIRE(reservation);
    auto* local =
      dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(task->local_state());
    local->set_reservation(std::move(reservation), info);
    task->telemetry_fsm().queued(
      {.queue = {.target = quent::task_queue::TaskQueueId{quent::nil_uuid()},
                 .data   = {.entries = 1}}});
    task->telemetry_fsm().reserving(
      {.requested_bytes      = 0,
       .input_basis          = 0,
       .peak_estimate        = 0,
       .bytes_to_materialize = 0,
       .manager_thread       = {
               .target = quent::task_manager_loop_thread::TaskManagerLoopThreadId{quent::nil_uuid()},
               .data   = {}}});
    return task;
  }
  void drain(bool reverse = false)
  {
    for (std::size_t guard = 0; !merge.all_ports_empty(); ++guard) {
      REQUIRE(guard < 20000);
      REQUIRE(merge.get_next_task_hint());
      std::vector<std::unique_ptr<sirius::pipeline::gpu_pipeline_task>> claimed;
      while (auto data = merge.get_next_task_input_data()) {
        claimed.push_back(task(std::move(data)));
        if (!reverse) { break; }
      }
      if (claimed.empty() && !merge.all_ports_empty()) {
        REQUIRE_FALSE(merge.get_next_task_hint());
      }
      if (reverse) { std::reverse(claimed.begin(), claimed.end()); }
      for (auto& item : claimed) {
        for (int retry = 0;; ++retry) {
          REQUIRE(retry < 24);
          try {
            item->execute(stream);
            break;
          } catch (sirius::pipeline::oom_reschedule_exception& error) {
            auto input = error.release_intermediate_data();
            dynamic_cast<pipelineable_operator_data&>(*input).remove_read_only_lock();
            retry_floor =
              dynamic_cast<sirius::pipeline::gpu_pipeline_task_local_state*>(item->local_state())
                ->get_retry_reservation_floor();
            item.reset();
            item = task(std::move(input));
          }
        }
        item.reset();
        retry_floor = 0;
      }
    }
    pipeline->update_pipeline_status(false);
    CHECK(pipeline->is_pipeline_finished());
    for (auto* storage : repositories.get_repositories()) {
      CHECK(storage->all_empty());
    }
  }
  std::vector<int64_t> rows()
  {
    std::vector<int64_t> values;
    auto positions = output.get_positions();
    for (std::size_t i = 0; i < positions.size(); ++i) {
      CHECK(positions[i] == i);
      pipelineable_operator_data restored({output.get_data_batch_by_position(positions[i])});
      restored.prepare_for_processing(gpu, stream);
      stream.synchronize();
      auto read  = restored.get_read_only_batches().front();
      auto chunk = copy_column_to_host<int64_t>(sirius::get_cudf_table_view(read).column(0));
      values.insert(values.end(), chunk.begin(), chunk.end());
      stream.synchronize();
    }
    return values;
  }
};
}  // namespace

TEST_CASE("independent tiled merges publish ordered ranges after all leaves finish",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.input->set_num_partitions(5);
  f.add({1, 4, 7}, 1);
  f.add({2, 5, 8}, 1);
  f.add({3, 6, 9}, 1);
  f.add({10, 13, 16}, 3);
  f.add({11, 14, 17}, 3);
  f.add({12, 15, 18}, 3);
  f.add({}, 4);
  f.drain(true);
  std::vector<int64_t> expected(18);
  std::iota(expected.begin(), expected.end(), 1);
  CHECK(f.rows() == expected);
}

TEST_CASE("tiled merge handles empty and single-contribution inputs", "[tiled_merge]")
{
  auto rows = GENERATE(0, 1, 101);
  tiled_fixture f;
  std::vector<int64_t> expected(rows);
  std::iota(expected.begin(), expected.end(), 0);
  f.add(expected);
  f.drain();
  CHECK(f.rows() == expected);
}

TEST_CASE("preparation subdivision commits only through ordinary publication", "[tiled_merge]")
{
  tiled_fixture f;
  f.merge.chunk_bytes = 128;
  f.add({1, 3, 5});
  f.add({2, 4, 6});
  auto data = f.merge.get_next_task_input_data();
  REQUIRE(data);
  f.merge.note_task_reservation(*data, f.merge.reservation_limit, f.merge.reservation_limit);
  auto replacement = f.merge.on_preparation_oom(*data);
  REQUIRE(replacement);
  CHECK(f.input->total_size() == 2);
  data.reset();
  auto task = f.task(std::move(replacement));
  task->execute(f.stream);
  task.reset();
  CHECK(f.input->total_size() == 2);  // splitting itself retires no owning sources
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6});
}

TEST_CASE("an unrestorable sort row fails before claiming work", "[tiled_merge]")
{
  tiled_fixture f;
  f.merge.reservation_limit = 128;
  f.input->add_data_batch(make_string_batch(*f.gpu, {std::string(200, 'x')}));
  CHECK_THROWS_AS(f.merge.get_next_task_input_data(), sirius::invalid_input_exception);
  CHECK(f.input->total_size() == 1);
  CHECK(f.repo("sort_tile_inputs")->all_empty());
}

namespace {
class rejecting_tiled_destination : public cucascade::shared_data_repository {
 public:
  void add_data_batch_at_position_unlocked(std::shared_ptr<cucascade::data_batch> const&,
                                           std::size_t,
                                           std::size_t) override
  {
    throw rmm::out_of_memory("injected final fan-out failure");
  }
};
}  // namespace

TEST_CASE("tiled final fan-out failure retains its result and is never replayed", "[tiled_merge]")
{
  rejecting_tiled_destination rejecting;
  sirius::op::test_physical_operator consumer{SiriusPhysicalOperatorType::FILTER, {}, 0};
  tiled_fixture f;
  consumer.add_port("default",
                    std::make_unique<sirius_physical_operator::port>(
                      MemoryBarrierType::FULL, &rejecting, nullptr, nullptr));
  f.merge.add_next_port_after_sink({&consumer, "default"});
  f.add({1, 2, 3});
  std::unique_ptr<operator_data> final;
  for (int guard = 0; !final; ++guard) {
    REQUIRE(guard < 20);
    auto data = f.merge.get_next_task_input_data();
    if (!data) { continue; }
    if (dynamic_cast<pipelineable_operator_data const&>(*data).get_position()) {
      final = std::move(data);
      break;
    }
    auto task = f.task(std::move(data));
    task->execute(f.stream);
  }
  auto* results = f.repo("sort_tile_results");
  REQUIRE(results->total_size() == 1);
  std::weak_ptr<cucascade::data_batch> source = results->get_data_batch_by_position(0, 0);
  auto task                                   = f.task(std::move(final));
  CHECK_THROWS_AS(task->execute(f.stream), rmm::out_of_memory);
  task.reset();
  CHECK(f.output.total_size() == 1);
  CHECK(results->total_size() == 1);
  CHECK_FALSE(f.merge.get_next_task_input_data());
  CHECK_FALSE(f.merge.all_ports_empty());
  (void)f.repositories.clear_all_repositories();
  CHECK_FALSE(source.expired());
  (void)f.output.pop_next_data_batch_with_position();
  CHECK(source.expired());
}

TEST_CASE("tiled final projection removes hidden keys after positioned merge", "[tiled_merge]")
{
  tiled_fixture f;
  f.merge.chunk_bytes = 48;
  f.merge.set_final_projections({1}, {sirius::logical_type::make(sirius::type_id::BIGINT)});
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {1, 3, 5}, {10, 30, 50}, cudf::type_id::INT64));
  f.input->add_data_batch(
    make_two_column_batch<int64_t, int64_t>(*f.gpu, {2, 4, 6}, {20, 40, 60}, cudf::type_id::INT64));
  f.drain(true);
  CHECK(f.rows() == std::vector<int64_t>{10, 20, 30, 40, 50, 60});
}

TEST_CASE("variable-width tiled output conserves payloads under reservation admission",
          "[tiled_merge]")
{
  memory_manager_options options;
  options.gpu_bytes                = 512 * 1024;
  options.gpu_reservation_fraction = 0.25;
  tiled_fixture f(1, options);
  f.merge.chunk_bytes = 64 * 1024;
  auto* tracker =
    f.gpu->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  REQUIRE(tracker);
  tracker->set_default_policy(std::make_unique<cucascade::memory::fail_reservation_limit_policy>());
  for (int64_t side = 0; side < 2; ++side) {
    std::vector<int64_t> keys(500);
    std::vector<std::string> payload(500, "a");
    for (int64_t i = 0; i < 500; ++i) {
      keys[i] = 2 * i + side;
    }
    payload.back() = std::string(20000, 'z');
    auto batch =
      make_two_column_batch<int64_t, std::string>(*f.gpu, keys, payload, cudf::type_id::STRING);
    default_stream().sync();
    CHECK(copy_column_to_host<std::string>(sirius::get_cudf_table_view(*batch).column(1)).back() ==
          payload.back());
    auto last = cudf::slice(sirius::get_cudf_table_view(*batch), {499, 500}, default_stream())[0];
    CHECK(copy_column_to_host<std::string>(last.column(1)).front() == payload.back());
    f.input->add_data_batch(batch);
  }
  sirius::test::scoped_recording_log_sink logs("debug");
  f.drain();
  std::vector<int64_t> expected(1000);
  std::iota(expected.begin(), expected.end(), 0);
  CHECK(f.rows() == expected);
  uint64_t subdivisions = 0;
  for (auto const& log : logs.records()) {
    subdivisions += log.message.find("sort.tile_refine") != std::string::npos;
  }
  CHECK(subdivisions == 0);
  std::size_t payload_rows = 0, large_rows = 0;
  for (auto position : f.output.get_positions()) {
    pipelineable_operator_data restored({f.output.get_data_batch_by_position(position)});
    restored.prepare_for_processing(f.gpu, f.stream);
    f.stream.synchronize();
    auto ro     = restored.get_read_only_batches().front();
    auto values = copy_column_to_host<std::string>(sirius::get_cudf_table_view(ro).column(1));
    auto keys   = copy_column_to_host<int64_t>(sirius::get_cudf_table_view(ro).column(0));
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (keys[i] >= 998) {
        INFO("key=" << keys[i] << " payload size=" << values[i].size()
                    << " first=" << (values[i].empty() ? -1 : static_cast<int>(values[i].front())));
        CHECK(values[i].size() == 20000);
      }
    }
    payload_rows += values.size();
    large_rows += std::count(values.begin(), values.end(), std::string(20000, 'z'));
    f.stream.synchronize();
  }
  CHECK(payload_rows == 1000);
  CHECK(large_rows == 2);
  std::cout << "tiled_runtime_capacity reservation=" << f.merge.reservation_limit
            << " task_attempts=" << f.task_id << " subdivisions=" << subdivisions
            << " rows=" << payload_rows << " wide_rows=" << large_rows << std::endl;
}

TEST_CASE("tiled extraction publishes independent pieces through real task completion",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  std::vector<std::weak_ptr<cucascade::data_batch>> originals;
  for (auto id : f.input->get_batch_ids()) {
    originals.push_back(f.input->get_data_batch_by_id(id));
  }
  f.drain(true);
  CHECK(f.input->all_empty());
  CHECK(f.repo("sort_tile_samples")->all_empty());
  CHECK(f.repo("sort_tile_splitters")->all_empty());
  CHECK(f.repo("sort_tile_pieces")->all_empty());
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
  for (auto const& source : originals) {
    CHECK(source.expired());
  }
}

TEST_CASE("tiled preprocessing normalizes coalesces and preserves ranges", "[tiled_merge]")
{
  tiled_fixture f;
  f.merge.chunk_bytes = 64;
  f.input->set_num_partitions(4);
  for (int64_t i = 0; i < 32; ++i) {
    f.add({i}, 0);
  }
  std::vector<int64_t> large(1200);
  std::iota(large.begin(), large.end(), 100);
  f.add(large, 2);
  f.add({}, 3);
  f.drain(true);
  auto expected = large;
  for (int64_t i = 0; i < 32; ++i) {
    expected.push_back(i);
  }
  std::sort(expected.begin(), expected.end());
  CHECK(f.rows() == expected);
  CHECK(f.repo("sort_tile_inputs")->all_empty());
  CHECK(f.input->all_empty());
}

TEST_CASE("tiled task metadata survives replacement and prepublication retry", "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  auto data = f.merge.get_next_task_input_data();
  REQUIRE(data);
  auto const& typed = dynamic_cast<pipelineable_operator_data const&>(*data);
  auto replacement  = typed.with_batches_preserving_rows(typed.get_data_batches());
  data.reset();
  auto work            = f.task(std::move(replacement));
  f.merge.fail_compute = true;
  std::unique_ptr<operator_data> retry;
  try {
    work->execute(f.stream);
  } catch (sirius::pipeline::oom_reschedule_exception& error) {
    retry = error.release_intermediate_data();
  }
  REQUIRE(retry);
  CHECK(f.repo("sort_tile_samples")->all_empty());
  CHECK(f.input->total_size() == 2);
  dynamic_cast<pipelineable_operator_data&>(*retry).remove_read_only_lock();
  work.reset();
  auto resumed = f.task(std::move(retry));
  resumed->execute(f.stream);
  resumed.reset();
  f.drain(true);
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 4, 5, 7, 8});
}

TEST_CASE("extraction pieces spill independently while source inputs remain registered",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  auto* pieces = f.repo("sort_tile_pieces");
  for (int guard = 0; pieces->all_empty(); ++guard) {
    REQUIRE(guard < 30);
    auto data = f.merge.get_next_task_input_data();
    REQUIRE(data);
    auto work = f.task(std::move(data));
    work->execute(f.stream);
  }
  auto* host = f.manager->get_memory_space(cucascade::memory::Tier::HOST, 0);
  {
    sirius::convertible_data_batch_provider sources(f.input);
    auto candidates = sources.get_all_convertible(f.gpu, true, false);
    REQUIRE_FALSE(candidates.empty());
    // A published borrowed slice would retain a source read lock and prevent
    // this nonblocking source downgrade. An owning piece does not.
    REQUIRE(candidates.front()->convert({host}, f.stream, *f.manager, false).has_value());
  }
  {
    sirius::convertible_data_batch_provider candidates(pieces);
    auto candidate = candidates.get_next_convertible(f.gpu, true);
    REQUIRE(candidate);
    REQUIRE(candidate->convert({host}, f.stream, *f.manager, false).has_value());
  }
  f.drain();
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 4, 5, 7, 8});
}

TEST_CASE("concurrent tiled claims cannot duplicate sample work", "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  f.add({3, 6, 9});
  std::vector<std::future<std::unique_ptr<operator_data>>> claims;
  for (int i = 0; i < 8; ++i) {
    claims.push_back(
      std::async(std::launch::async, [&] { return f.merge.get_next_task_input_data(); }));
  }
  std::set<uint64_t> ids;
  std::vector<std::unique_ptr<operator_data>> claimed;
  for (auto& claim : claims) {
    auto data = claim.get();
    if (!data) { continue; }
    claimed.push_back(std::move(data));
  }
  for (auto& data : claimed) {
    auto const& batches = dynamic_cast<pipelineable_operator_data const&>(*data).get_data_batches();
    REQUIRE(batches.size() == 1);
    CHECK(ids.insert(batches.front()->get_batch_id()).second);
    auto task = f.task(std::move(data));
    task->execute(f.stream);
  }
  REQUIRE(ids.size() == 3);
  f.drain(true);
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST_CASE("partial tile-piece sink failure retains sources until drained query retirement",
          "[tiled_merge]")
{
  tiled_fixture f;
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  auto original_id                              = f.input->get_batch_ids().front();
  std::weak_ptr<cucascade::data_batch> original = f.input->get_data_batch_by_id(original_id);
  for (int step = 0; f.repo("sort_tile_splitters")->all_empty(); ++step) {
    REQUIRE(step < 10);
    auto data = f.merge.get_next_task_input_data();
    REQUIRE(data);
    auto task = f.task(std::move(data));
    task->execute(f.stream);
  }
  auto data = f.merge.get_next_task_input_data();
  REQUIRE(data);
  auto* pieces = f.repo("sort_tile_pieces");
  REQUIRE(pieces->num_partitions() >= 2);
  // The first tile publishes, then an ordinary duplicate-position error aborts
  // the second insertion. No completion callback or allocation retry follows.
  pieces->add_data_batch_at_position(
    make_numeric_batch<int64_t>(*f.gpu, {99}, cudf::type_id::INT64), 0, 1);
  auto task = f.task(std::move(data));
  CHECK_THROWS_AS(task->execute(f.stream), cucascade::repository_duplicate_error);
  task.reset();
  CHECK(pieces->get_data_batch_by_position(0, 0));
  CHECK(f.input->total_size() == 2);
  CHECK_FALSE(f.merge.all_ports_empty());
  CHECK_FALSE(f.pipeline->is_pipeline_finished());
  std::weak_ptr<cucascade::data_batch> published = pieces->get_data_batch_by_position(0, 0);
  (void)f.repositories.clear_all_repositories();  // no active task/read handles remain
  CHECK(original.expired());
  CHECK(published.expired());
}

TEST_CASE("tiled sample and extraction work restores backing across GPUs",
          "[tiled_merge][multi_gpu]")
{
  int count = 0;
  REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess);
  if (count < 2) { SKIP("requires two GPUs"); }
  std::optional<rmm::cuda_stream> remote_stream;
  tiled_fixture f(2);
  auto* remote = f.manager->get_memory_space(cucascade::memory::Tier::GPU, 1);
  {
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{1}};
    remote_stream.emplace();
  }
  f.add({1, 4, 7});
  f.add({2, 5, 8});
  for (std::size_t guard = 0; !f.merge.all_ports_empty(); ++guard) {
    REQUIRE(guard < 100);
    auto data = f.merge.get_next_task_input_data();
    if (!data) { continue; }
    auto target = guard % 2 ? remote : f.gpu;
    rmm::cuda_set_device_raii device{rmm::cuda_device_id{static_cast<int>(guard % 2)}};
    auto task = f.task(std::move(data), target);
    task->execute(guard % 2 ? ::cuda::stream_ref{*remote_stream} : ::cuda::stream_ref{f.stream});
  }
  for (auto* repo : f.repositories.get_repositories()) {
    CHECK(repo->all_empty());
  }
  CHECK(f.rows() == std::vector<int64_t>{1, 2, 4, 5, 7, 8});
}

TEST_CASE("cold estimates do not refuse fitting order or coalescing tasks", "[tiled_merge]")
{
  auto chunk = GENERATE(512, 1024);
  memory_manager_options options;
  options.gpu_bytes                = 512 * 1024;
  options.gpu_reservation_fraction = 0.0625;  // 32 KiB: below even the cold estimate's base
  tiled_fixture f(1, options);
  f.merge.chunk_bytes = chunk;
  // A planner default larger than the selected GPU must not prevent exhausted recovery.
  f.merge.reservation_limit = std::numeric_limits<std::size_t>::max();
  auto* tracker =
    f.gpu->get_memory_resource_as<cucascade::memory::reservation_aware_resource_adaptor>();
  tracker->set_default_policy(std::make_unique<cucascade::memory::fail_reservation_limit_policy>());
  std::vector<int64_t> left(64), right(64), expected(128);
  for (int i = 0; i < 64; ++i) {
    left[i]  = 2 * i;
    right[i] = 2 * i + 1;
  }
  std::iota(expected.begin(), expected.end(), 0);
  f.add(left);
  f.add(right);
  sirius::test::scoped_recording_log_sink logs("trace");
  f.drain();
  CHECK(f.rows() == expected);
  std::size_t orders = 0, coalesces = 0, refinements = 0;
  for (auto const& log : logs.records()) {
    orders += log.message.find("sort.tile_work kind=order_samples") != std::string::npos;
    coalesces += log.message.find("sort.tile_work kind=coalesce") != std::string::npos;
    refinements += log.message.find("sort.tile_refine") != std::string::npos;
  }
  CHECK(orders == (chunk == 512 ? 1 : 0));
  CHECK(coalesces == (chunk == 1024 ? 1 : 0));
  CHECK(refinements == 0);
}
