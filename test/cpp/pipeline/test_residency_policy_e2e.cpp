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
#include "op/sirius_physical_operator.hpp"
#include "operator/operator_test_utils.hpp"
#include "pipeline/gpu_pipeline_task.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "utils/test_physical_operator.hpp"

#include <rmm/cuda_stream.hpp>

#include <cuda/stream>

#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

using sirius::op::MemoryBarrierType;
using sirius::op::pipelineable_operator_data;
using sirius::op::port_residency_policy;
using sirius::op::sirius_physical_operator;
using sirius::op::test_physical_operator;
using sirius::pipeline::gpu_pipeline_task_local_state;
using sirius::pipeline::materialize_repository_wiring;
using sirius::pipeline::repository_wiring;
using sirius::pipeline::sirius_pipeline;
using sirius::pipeline::sirius_pipeline_build_state;
using sirius::test::operator_utils::tiered_memory_fixture;

namespace {

constexpr std::string_view kPort = "default";

/// Opts every input port into on-demand residency and downgrade-first.
class opted_in_operator : public test_physical_operator {
 public:
  using test_physical_operator::test_physical_operator;

  static constexpr port_residency_policy policy{.materialize_inputs_on_gpu = false,
                                                .downgrade_priority        = 100};

  port_residency_policy input_residency_policy(std::string_view) const override { return policy; }
};

/// A source pipeline feeding @p consumer through one PIPELINE-barrier port, wired by the real
/// materializer. The consumer is the destination pipeline's first operator.
struct wired_consumer {
  explicit wired_consumer(sirius_physical_operator& consumer)
  {
    auto src_pipeline  = make_pipeline(&source_sink, {}, /*pipeline_id=*/0);
    auto dest_pipeline = make_pipeline(&dest_sink, {&consumer}, /*pipeline_id=*/1);
    std::vector<repository_wiring> wirings = {
      {kPort, MemoryBarrierType::PIPELINE, &source_sink, src_pipeline, dest_pipeline}};
    // materialize_repository_wiring reads operator ids, so number the operators first.
    sirius::pipeline::assign_operator_ids(pipelines);
    materialize_repository_wiring(wirings, mgr);
    port = consumer.get_port(kPort);
  }

  std::shared_ptr<sirius_pipeline> make_pipeline(
    sirius_physical_operator* sink,
    const std::vector<sirius_physical_operator*>& operators,
    size_t pipeline_id)
  {
    auto pipeline = std::make_shared<sirius_pipeline>(build_ctx);
    pipelines.push_back(pipeline);
    sirius_pipeline_build_state build_state;
    build_state.set_pipeline_sink(*pipeline, sink, /*pipeline_idx=*/0);
    for (auto* op : operators) {
      build_state.add_pipeline_operator(*pipeline, *op);
    }
    pipeline->set_pipeline_id(pipeline_id);
    return pipeline;
  }

  sirius::pipeline::pipeline_build_context build_ctx{nullptr};
  std::vector<std::shared_ptr<sirius_pipeline>> pipelines;
  cucascade::shared_data_repository_manager mgr;
  test_physical_operator source_sink;
  test_physical_operator dest_sink;
  sirius_physical_operator::port* port = nullptr;
};

/// Push a HOST batch into the wired port and build the task local state around the operator's
/// default task input, as the task creator does.
struct task_under_test {
  task_under_test(tiered_memory_fixture& f,
                  sirius_physical_operator& consumer,
                  wired_consumer& wired,
                  ::cuda::stream_ref stream)
    : batch(f.make_host_batch(stream))
  {
    REQUIRE(wired.port != nullptr);
    REQUIRE(wired.port->repo != nullptr);
    wired.port->repo->add_data_batch(batch);

    auto task_input = consumer.get_next_task_input_data();
    REQUIRE(task_input != nullptr);
    input = dynamic_cast<pipelineable_operator_data*>(task_input.get());
    REQUIRE(input != nullptr);
    REQUIRE(input->get_data_batches().size() == 1);
    REQUIRE(input->get_data_batches()[0] == batch);
    local_state = std::make_unique<gpu_pipeline_task_local_state>(std::move(task_input));
  }

  std::shared_ptr<cucascade::data_batch> batch;
  pipelineable_operator_data* input = nullptr;  ///< Owned by local_state.
  std::unique_ptr<gpu_pipeline_task_local_state> local_state;
};

}  // namespace

TEST_CASE("an opted-in operator's host input stays in its tier from wiring to reservation",
          "[residency_policy_e2e]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  opted_in_operator consumer;
  wired_consumer wired(consumer);
  REQUIRE(wired.port != nullptr);
  CHECK(wired.port->residency == opted_in_operator::policy);
  CHECK(wired.port->repo->get_downgrade_priority() == 100);

  task_under_test task(f, consumer, wired, stream);
  REQUIRE_FALSE(task.input->materializes_on_gpu());
  CHECK(task.local_state->get_estimated_bytes_to_materialize_input(f.gpu0) == 0);

  const auto source_id            = task.batch->get_batch_id();
  const auto gpu_available_before = f.gpu0->get_available_memory();
  task.input->prepare_for_processing(f.gpu0, stream);
  stream.synchronize();

  // Same batch, still on the host, read-locked; nothing was allocated on the GPU for it.
  REQUIRE(task.input->get_data_batches().at(0) == task.batch);
  CHECK(f.gpu0->get_available_memory() == gpu_available_before);
  {
    auto ro_batches = task.input->get_read_only_batches();
    REQUIRE(ro_batches.size() == 1);
    CHECK(ro_batches[0].get_batch_id() == source_id);
    CHECK(ro_batches[0].get_current_tier() == cucascade::memory::Tier::HOST);
    CHECK(ro_batches[0].get_memory_space()->get_id() == f.host0->get_id());
    CHECK(dynamic_cast<const cucascade::host_data_representation*>(ro_batches[0].get_data()) !=
          nullptr);
  }
  CHECK(task.local_state->get_estimated_bytes_to_materialize_input(f.gpu0) == 0);
  task.input->remove_read_only_lock();
}

TEST_CASE("a default operator's host input is sized for and converted to the GPU",
          "[residency_policy_e2e]")
{
  tiered_memory_fixture f;
  REQUIRE(f.setup());
  rmm::cuda_stream stream;

  test_physical_operator consumer;
  wired_consumer wired(consumer);
  REQUIRE(wired.port != nullptr);
  CHECK(wired.port->residency == port_residency_policy{});
  CHECK(wired.port->repo->get_downgrade_priority() == 0);

  task_under_test task(f, consumer, wired, stream);
  REQUIRE(task.input->materializes_on_gpu());
  // The reservation is sized before prepare, while the batch is still on the host.
  CHECK(task.local_state->get_estimated_bytes_to_materialize_input(f.gpu0) > 0);

  const auto source_id = task.batch->get_batch_id();
  task.input->prepare_for_processing(f.gpu0, stream);
  stream.synchronize();

  {
    auto ro_batches = task.input->get_read_only_batches();
    REQUIRE(ro_batches.size() == 1);
    CHECK(ro_batches[0].get_batch_id() == source_id);
    CHECK(ro_batches[0].get_current_tier() == cucascade::memory::Tier::GPU);
    CHECK(ro_batches[0].get_memory_space()->get_id() == f.gpu0->get_id());
  }
  task.input->remove_read_only_lock();
}
