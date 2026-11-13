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

#include "catch.hpp"
#include "data/data_repository_manager_registry.hpp"
#include "downgrade/downgrade_executor.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "op/sirius_physical_operator.hpp"
#include "operator/operator_test_utils.hpp"
#include "query_id.hpp"
#include "utils/sirius_test_env.hpp"
#include "utils/test_physical_operator.hpp"

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

using namespace sirius::op;
using namespace std::chrono_literals;
using sirius::test::operator_utils::tiered_memory_fixture;

namespace {

/// Attach a port backed by @p repo, with the residency policy set by hand (no wiring involved).
void attach_port(sirius_physical_operator& op,
                 std::string_view port_id,
                 cucascade::shared_data_repository& repo,
                 bool materialize_inputs_on_gpu)
{
  auto port                                 = std::make_unique<sirius_physical_operator::port>();
  port->type                                = MemoryBarrierType::PIPELINE;
  port->repo                                = &repo;
  port->residency.materialize_inputs_on_gpu = materialize_inputs_on_gpu;
  op.add_port(port_id, std::move(port));
}

/// The default task input of @p op, as the pipelineable payload the base class builds.
std::unique_ptr<pipelineable_operator_data> next_input(sirius_physical_operator& op)
{
  auto input = op.get_next_task_input_data();
  REQUIRE(input != nullptr);
  auto* pipelineable = dynamic_cast<pipelineable_operator_data*>(input.get());
  REQUIRE(pipelineable != nullptr);
  std::ignore = input.release();
  return std::unique_ptr<pipelineable_operator_data>(pipelineable);
}

}  // namespace

TEST_CASE("default ports build a task input that materializes on the GPU",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  cucascade::shared_data_repository repo;
  test_physical_operator op;
  auto port  = std::make_unique<sirius_physical_operator::port>();
  port->type = MemoryBarrierType::PIPELINE;
  port->repo = &repo;
  op.add_port("input", std::move(port));
  repo.add_data_batch(f.make_gpu_batch(stream));

  REQUIRE(next_input(op)->materializes_on_gpu());
}

TEST_CASE("a single opted-in port builds a task input that does not materialize on the GPU",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  cucascade::shared_data_repository repo;
  test_physical_operator op;
  attach_port(op, "input", repo, false);
  auto batch = f.make_gpu_batch(stream);
  repo.add_data_batch(batch);

  auto input = next_input(op);
  REQUIRE_FALSE(input->materializes_on_gpu());
  REQUIRE(input->get_data_batches().size() == 1);
  REQUIRE(input->get_data_batches()[0] == batch);
}

TEST_CASE("mixed ports build a task input that materializes on the GPU", "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  cucascade::shared_data_repository opted_in_repo;
  cucascade::shared_data_repository default_repo;
  test_physical_operator op;
  attach_port(op, "opted_in", opted_in_repo, false);
  attach_port(op, "default", default_repo, true);

  SECTION("both ports contribute")
  {
    opted_in_repo.add_data_batch(f.make_gpu_batch(stream));
    default_repo.add_data_batch(f.make_gpu_batch(stream));
    auto input = next_input(op);
    REQUIRE(input->get_data_batches().size() == 2);
    REQUIRE(input->materializes_on_gpu());
  }
  SECTION("only the opted-in port contributes")
  {
    // A port that contributed no batch has no say in the flag.
    opted_in_repo.add_data_batch(f.make_gpu_batch(stream));
    auto input = next_input(op);
    REQUIRE(input->get_data_batches().size() == 1);
    REQUIRE_FALSE(input->materializes_on_gpu());
  }
}

TEST_CASE("prepare locks a host batch in place when the input does not materialize on the GPU",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;
  auto batch           = f.make_host_batch(stream);
  const auto source_id = batch->get_batch_id();

  pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  input.set_materialize_on_gpu(false);

  const auto gpu_available_before = f.gpu0->get_available_memory();
  input.prepare_for_processing(f.gpu0, stream);
  stream.synchronize();

  // Same batch, still on the host, read-locked; nothing was allocated on the GPU for it.
  REQUIRE(input.get_data_batches().at(0) == batch);
  REQUIRE(input.original_batch_ids() == std::vector<std::uint64_t>{source_id});
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);
  REQUIRE(f.gpu0->get_available_memory() == gpu_available_before);
  {
    auto ro_batches = input.get_read_only_batches();
    REQUIRE(ro_batches.size() == 1);
    REQUIRE(ro_batches[0].get_batch_id() == source_id);
    REQUIRE(ro_batches[0].get_current_tier() == cucascade::memory::Tier::HOST);
    REQUIRE(ro_batches[0].get_memory_space()->get_id() == f.host0->get_id());
    REQUIRE(dynamic_cast<const cucascade::host_data_representation*>(ro_batches[0].get_data()) !=
            nullptr);
  }
  // The stored locks keep the batch pinned after the transient copies above are gone.
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  input.remove_read_only_lock();
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
  REQUIRE(batch->try_to_mutable().has_value());
}

TEST_CASE("prepare locks a GPU batch in place when the input does not materialize on the GPU",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;
  auto batch           = f.make_gpu_batch(stream);
  const auto source_id = batch->get_batch_id();

  pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  input.set_materialize_on_gpu(false);

  const auto gpu_available_before = f.gpu0->get_available_memory();
  input.prepare_for_processing(f.gpu0, stream);
  stream.synchronize();

  // Same batch, already where the task runs, read-locked; no copy was made of it.
  REQUIRE(input.get_data_batches().size() == 1);
  REQUIRE(input.get_data_batches()[0] == batch);
  REQUIRE(input.original_batch_ids() == std::vector<std::uint64_t>{source_id});
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);
  REQUIRE(f.gpu0->get_available_memory() == gpu_available_before);
  {
    auto ro_batches = input.get_read_only_batches();
    REQUIRE(ro_batches.size() == 1);
    REQUIRE(ro_batches[0].get_batch_id() == source_id);
    REQUIRE(ro_batches[0].get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro_batches[0].get_memory_space()->get_id() == f.gpu0->get_id());
  }
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  input.remove_read_only_lock();
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE(
  "prepare leaves a batch on another GPU in place when the input does not materialize on "
  "the GPU",
  "[input_residency_policy][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }
  tiered_memory_fixture f;
  REQUIRE(f.setup(2));
  rmm::cuda_set_device_raii source_device{rmm::cuda_device_id{f.gpu1->get_device_id()}};
  rmm::cuda_stream source_stream;
  auto batch           = f.make_gpu_batch(*f.gpu1, source_stream);
  const auto source_id = batch->get_batch_id();
  source_stream.synchronize();

  pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  input.set_materialize_on_gpu(false);

  // The task runs on GPU 0 with a stream of that device.
  rmm::cuda_set_device_raii target_device{rmm::cuda_device_id{f.gpu0->get_device_id()}};
  rmm::cuda_stream target_stream;
  const auto gpu0_available_before = f.gpu0->get_available_memory();
  const auto gpu1_available_before = f.gpu1->get_available_memory();
  input.prepare_for_processing(f.gpu0, target_stream);
  target_stream.synchronize();

  // Same batch, still on GPU 1, read-locked; it was not cloned and GPU 0 holds nothing for it.
  REQUIRE(input.get_data_batches().size() == 1);
  REQUIRE(input.get_data_batches()[0] == batch);
  REQUIRE(input.original_batch_ids() == std::vector<std::uint64_t>{source_id});
  REQUIRE(batch->get_state() == cucascade::batch_state::read_only);
  REQUIRE(f.gpu0->get_available_memory() == gpu0_available_before);
  REQUIRE(f.gpu1->get_available_memory() == gpu1_available_before);
  {
    auto ro_batches = input.get_read_only_batches();
    REQUIRE(ro_batches.size() == 1);
    REQUIRE(ro_batches[0].get_batch_id() == source_id);
    REQUIRE(ro_batches[0].get_current_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(ro_batches[0].get_memory_space()->get_id() == f.gpu1->get_id());
  }
  REQUIRE_FALSE(batch->try_to_mutable().has_value());

  input.remove_read_only_lock();
  REQUIRE(batch->get_state() == cucascade::batch_state::idle);
}

TEST_CASE("prepare of an input that does not materialize on the GPU races a downgrade safely",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  const auto query_id = sirius::make_query_id(1);
  sirius::exec::query_lifecycle_registry lifecycle;
  lifecycle.open_query(query_id);
  sirius::data::data_repository_manager_registry repo_registry;
  auto& repo_mgr  = *repo_registry.create_for_query(query_id);
  auto owned_repo = std::make_unique<cucascade::shared_data_repository>();
  auto* repo      = owned_repo.get();
  repo_mgr.add_new_repository(1, "out", std::move(owned_repo));

  // monitor_period 0: the executor downgrades only on request. It joins its threads on
  // destruction, before the repositories and the memory manager above go away.
  sirius::parallel::downgrade_executor executor(
    lifecycle,
    sirius::exec::downgrade_executor_config{
      .thread_pool = {.num_threads = 1, .thread_name_prefix = "downgrade"}, .monitor_period = 0ms},
    repo_registry,
    f.gpu0->get_id(),
    f.gpu0,
    *f.manager);
  executor.start();

  constexpr std::size_t free_everything = std::size_t{1} << 30;
  constexpr int iterations              = 40;
  for (int i = 0; i < iterations; ++i) {
    CAPTURE(i);
    auto batch = f.make_gpu_batch(stream);
    repo->add_data_batch(batch);
    pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
    input.set_materialize_on_gpu(false);

    // The executor's threads downgrade the batch while this thread prepares it. The head start
    // the downgrade gets grows with i, so the prepare lands before, during and after it.
    auto downgrade        = executor.request_free_memory(free_everything);
    const auto head_start = std::chrono::steady_clock::now() + i * 10us;
    while (std::chrono::steady_clock::now() < head_start) {}
    std::exception_ptr prepare_error;
    try {
      input.prepare_for_processing(f.gpu0, stream);
    } catch (...) {
      prepare_error = std::current_exception();
    }
    REQUIRE(downgrade.wait_for(30s) == std::future_status::ready);
    REQUIRE_NOTHROW(downgrade.get());
    if (prepare_error) { REQUIRE_NOTHROW(std::rethrow_exception(prepare_error)); }

    // Whichever side won, the input holds a lock on the batch in the tier it ended up in.
    REQUIRE(input.get_data_batches().at(0) == batch);
    cucascade::memory::Tier locked_tier{};
    {
      auto ro_batches = input.get_read_only_batches();
      REQUIRE(ro_batches.size() == 1);
      REQUIRE(ro_batches[0].get_memory_space() != nullptr);
      locked_tier = ro_batches[0].get_current_tier();
    }
    const bool on_gpu = locked_tier == cucascade::memory::Tier::GPU;
    REQUIRE((on_gpu || locked_tier == cucascade::memory::Tier::HOST));

    // While the lock is held the batch cannot be taken exclusively, so a downgrade skips it.
    REQUIRE_FALSE(batch->try_to_mutable().has_value());
    const auto freed_while_locked = executor.request_free_memory_and_wait(free_everything);
    REQUIRE(freed_while_locked == 0);
    REQUIRE(input.get_read_only_batches().at(0).get_current_tier() == locked_tier);

    // Released, it is downgradable again; this also leaves the repository with nothing on the
    // GPU for the next iteration.
    input.remove_read_only_lock();
    REQUIRE((executor.request_free_memory_and_wait(free_everything) > 0) == on_gpu);
    REQUIRE(batch->to_read_only().get_current_tier() == cucascade::memory::Tier::HOST);
  }
  executor.stop();
}

TEST_CASE("prepare converts a host batch to the GPU when the input materializes on the GPU",
          "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;
  auto batch           = f.make_host_batch(stream);
  const auto source_id = batch->get_batch_id();

  pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{batch});
  REQUIRE(input.materializes_on_gpu());
  input.prepare_for_processing(f.gpu0, stream);

  auto ro_batches = input.get_read_only_batches();
  REQUIRE(ro_batches.size() == 1);
  REQUIRE(ro_batches[0].get_batch_id() == source_id);
  REQUIRE(ro_batches[0].get_current_tier() == cucascade::memory::Tier::GPU);
  REQUIRE(ro_batches[0].get_memory_space()->get_id() == f.gpu0->get_id());
}

TEST_CASE("a row-preserving replacement keeps the materialize flag", "[input_residency_policy]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;
  auto original    = f.make_gpu_batch(stream);
  auto replacement = f.make_gpu_batch(stream);

  SECTION("pipelineable input")
  {
    pipelineable_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{original});
    REQUIRE(input.with_batches_preserving_rows({replacement})->materializes_on_gpu());
    input.set_materialize_on_gpu(false);
    auto replaced = input.with_batches_preserving_rows({replacement});
    REQUIRE_FALSE(replaced->materializes_on_gpu());
    REQUIRE(replaced->get_data_batches().at(0) == replacement);
  }
  SECTION("partitioned input")
  {
    partitioned_operator_data input(std::vector<std::shared_ptr<cucascade::data_batch>>{original});
    REQUIRE(input.with_batches_preserving_rows({replacement})->materializes_on_gpu());
    input.set_materialize_on_gpu(false);
    auto replaced = input.with_batches_preserving_rows({replacement});
    REQUIRE(replaced->get_type() == operator_data_type::PARTITIONED);
    REQUIRE_FALSE(replaced->materializes_on_gpu());
  }
}
