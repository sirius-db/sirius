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
#include "data/data_batch_utils.hpp"
#include "data/sirius_converter_registry.hpp"
#include "memory/sirius_memory_reservation_manager.hpp"
#include "op/sirius_physical_operator.hpp"
#include "utils/test_physical_operator.hpp"
#include "utils/utils.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

using namespace sirius::op;

namespace {

constexpr std::size_t kNumRows = 4096;

/// Memory manager fixture over one GPU space plus NUMA host spaces.
struct residency_fixture {
  std::unique_ptr<sirius::memory::sirius_memory_reservation_manager> manager;
  cucascade::memory::memory_space* gpu0  = nullptr;
  cucascade::memory::memory_space* host0 = nullptr;

  bool setup()
  {
    sirius::converter_registry::reset_for_testing();
    try {
      cucascade::memory::reservation_manager_configurator builder;
      builder.set_number_of_gpus(1)
        .set_gpu_usage_limit(256ULL << 20)
        .set_reservation_fraction_per_gpu(0.75)
        .set_per_numa_region_capacity(1ULL << 30)
        .use_numa_id_as_host_id()
        .track_reservation_per_stream(false)
        .set_reservation_fraction_per_numa_region(0.75);
      manager =
        std::make_unique<sirius::memory::sirius_memory_reservation_manager>(builder.build());
    } catch (cucascade::cuda_error const&) {
      throw;
    } catch (const std::exception&) {
      return false;
    }
    sirius::converter_registry::initialize();

    auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
    if (gpu_spaces.empty()) { return false; }
    gpu0             = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
    auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
    if (host_spaces.empty()) { return false; }
    host0 = const_cast<cucascade::memory::memory_space*>(host_spaces[0]);
    return true;
  }

  ~residency_fixture()
  {
    // The registry is initialized right after the manager is built, so both exist or neither.
    if (manager) {
      manager->shutdown();
      sirius::converter_registry::shutdown();
    }
  }

  /// Single INT64 column of random data, resident on the GPU space.
  std::shared_ptr<cucascade::data_batch> make_gpu_batch(::cuda::stream_ref stream)
  {
    auto table = sirius::create_cudf_table_with_random_data(kNumRows,
                                                            {cudf::data_type{cudf::type_id::INT64}},
                                                            {std::make_pair(0, 1000000)},
                                                            stream,
                                                            gpu0->get_default_allocator());
    stream.sync();
    return sirius::make_data_batch(
      std::move(table), *gpu0, stream, sirius::telemetry::batch_telemetry_info{});
  }

  /// GPU batch converted in place to the host representation (spilled-batch shape).
  std::shared_ptr<cucascade::data_batch> make_host_batch(::cuda::stream_ref stream)
  {
    auto batch     = make_gpu_batch(stream);
    auto& registry = sirius::converter_registry::get();
    {
      auto mut = batch->to_mutable();
      mut.convert_to<cucascade::host_data_representation>(registry, host0, stream);
    }
    stream.sync();
    return batch;
  }
};

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
  residency_fixture f;
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
  residency_fixture f;
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
  residency_fixture f;
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
  residency_fixture f;
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

TEST_CASE("prepare converts a host batch to the GPU when the input materializes on the GPU",
          "[input_residency_policy]")
{
  residency_fixture f;
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
  residency_fixture f;
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
