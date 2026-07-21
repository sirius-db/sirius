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

#include "telemetry/telemetry_context.hpp"

#include "log/logging.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_config.hpp"
#include "telemetry-bridge/gen/operator.rs.h"
#include "telemetry-bridge/gen/plan.rs.h"
#include "telemetry-bridge/gen/port.rs.h"

#include <unistd.h>

#include <format>
#include <memory>
#include <ranges>
#include <string>

namespace sirius::telemetry {

std::shared_ptr<const telemetry_context> telemetry_context::create(
  const sirius::telemetry_config& config,
  const cucascade::memory::memory_reservation_manager* manager,
  const std::vector<int>& gpu_device_ids)
{
  return std::shared_ptr<telemetry_context>(new telemetry_context(config, manager, gpu_device_ids));
}

telemetry_context::telemetry_context(const sirius::telemetry_config& config,
                                     const cucascade::memory::memory_reservation_manager* manager,
                                     const std::vector<int>& gpu_device_ids)
  : engine_uuid_(uuid::now_v7()),
    worker_uuid_(uuid::now_v7()),
    query_group_uuid_(uuid::now_v7()),
    shared_group_uuid_(uuid::now_v7()),
    context_(
      quent::create_context(config.enable_quent ? "ndjson" : "noop", config.output_directory)),
    engine_observer_(quent::engine::create_observer(*context_)),
    worker_observer_(quent::worker::create_observer(*context_)),
    query_group_observer_(quent::query_group::create_observer(*context_))
{
  engine_observer_->init(engine_uuid_,
                         quent::engine::Init{
                           .implementation =
                             quent::engine::Implementation{
                               .name              = config.engine_name,
                               .version           = "",
                               .custom_attributes = {},
                             },
                           .instance_name = config.engine_name,
                         });

  worker_observer_->init(worker_uuid_,
                         quent::worker::Init{
                           .parent_engine_id = engine_uuid_,
                           .instance_name    = std::format("worker-{}", getpid()),
                         });

  memory_context_ = std::make_shared<memory_context>(engine_uuid_, *context_, manager);

  // One session-scoped query group under this engine; every query in this context is reported
  // under it, so a whole run shows up as a single group rather than one group per query.
  query_group_observer_->declaration(
    query_group_uuid_,
    quent::query_group::Declaration{
      .instance_name = std::format("{}-session-{}", config.engine_name, getpid()),
      .engine_id     = engine_uuid_,
    });

  // Per-GPU device groups plus per-thread-type buckets underneath, so the
  // viewer renders threads as an engine -> gpu-N -> thread-type tree instead
  // of a flat sibling list. Threads with no single GPU go under `shared`.
  auto gpu_device_observer   = quent::gpu_device::create_observer(*context_);
  auto thread_group_observer = quent::thread_group::create_observer(*context_);

  thread_group_observer->declaration(shared_group_uuid_,
                                     quent::thread_group::Declaration{
                                       .instance_name   = "shared",
                                       .parent_group_id = engine_uuid_,
                                     });

  for (const int device_id : gpu_device_ids) {
    const gpu_device_group_ids ids{
      .device           = uuid::now_v7(),
      .executor_threads = uuid::now_v7(),
      .manager_threads  = uuid::now_v7(),
    };
    gpu_device_observer->declaration(ids.device,
                                     quent::gpu_device::Declaration{
                                       .instance_name   = std::format("gpu-{}", device_id),
                                       .parent_group_id = engine_uuid_,
                                       .ordinal         = static_cast<uint32_t>(device_id),
                                     });
    thread_group_observer->declaration(ids.executor_threads,
                                       quent::thread_group::Declaration{
                                         .instance_name   = "executor_thread",
                                         .parent_group_id = ids.device,
                                       });
    thread_group_observer->declaration(ids.manager_threads,
                                       quent::thread_group::Declaration{
                                         .instance_name   = "task_manager_loop_thread",
                                         .parent_group_id = ids.device,
                                       });
    gpu_group_ids_.emplace(device_id, ids);
  }

  SIRIUS_LOG_INFO("Telemetry context initialized (engine={}, {} GPU device group(s))",
                  config.engine_name,
                  gpu_group_ids_.size());
}

const uuid::UUID& telemetry_context::gpu_device_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.device;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

const uuid::UUID& telemetry_context::executor_thread_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.executor_threads;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

const uuid::UUID& telemetry_context::manager_thread_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.manager_threads;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

telemetry_context::~telemetry_context()
{
  memory_context_.reset();
  worker_observer_->exit(worker_uuid_);
  engine_observer_->exit(engine_uuid_);
}

void emit_plan_telemetry(
  const quent::Context& context,
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  const uuid::UUID plan_id,
  const query_telemetry_info telemetry_info)
{
  auto operator_obs = quent::operator_::create_observer(context);
  auto port_obs     = quent::port::create_observer(context);
  auto plan_obs     = quent::plan::create_observer(context);

  // Collect edges while iterating
  rust::Vec<quent::plan::Edges> edges;

  for (const auto& pipeline : pipelines) {
    const auto pipeline_uuid         = pipeline->pipeline_uuid();
    const auto operators             = pipeline->get_operators();
    const std::string operator_chain = [&operators]() {
      std::string chain{};
      for (const auto& name : operators | std::views::transform([](const auto& op) {
                                return std::format(
                                  "{}({})", op.get().get_name(), op.get().operator_id);
                              })) {
        if (chain.empty()) {
          chain = name;
          continue;
        }
        chain = std::format("{} -> {}", chain, name);
      }
      return chain;
    }();

    operator_obs->declaration(
      pipeline_uuid,
      quent::operator_::Declaration{
        .plan_id             = plan_id,
        .parent_operator_ids = {},
        .instance_name       = operator_chain,
        .type_name           = std::format("Pipeline Id {}", pipeline->get_pipeline_id()),
        .custom_attributes   = {},
      });

    // Receiver ports on pipeline source operators.
    if (auto source = pipeline->get_source()) {
      for (std::string_view port_id : source->get_port_ids()) {
        if (const op::sirius_physical_operator::port* port = source->get_port(port_id)) {
          port_obs->declaration(port->source_port_uuid,
                                quent::port::Declaration{
                                  .operator_id   = pipeline_uuid,
                                  .instance_name = std::format("{}_receiver", port_id),
                                });
        }
      }
    }

    // Sender ports on pipeline sink(last) operators.
    for (const auto& [next_operator, next_operator_port_name, pseudo_sink_port_uuid] :
         pipeline->get_next_ports_after_sink()) {
      // Declare the pseudo-sink port
      port_obs->declaration(pseudo_sink_port_uuid,
                            quent::port::Declaration{
                              .operator_id   = pipeline_uuid,
                              .instance_name = std::format("{}_sender", next_operator_port_name),
                            });

      // Find the target port on the downstream operator
      if (const op::sirius_physical_operator::port* target_port =
            next_operator->get_port(next_operator_port_name)) {
        edges.push_back(quent::plan::Edges{
          .source = pseudo_sink_port_uuid,
          .target = target_port->source_port_uuid,
        });
      }
    }
  }

  plan_obs->declaration(plan_id,
                        quent::plan::Declaration{
                          .parent =
                            quent::plan::Parent{
                              .query_id = telemetry_info.query_id,
                              .plan_id  = uuid::new_nil(),  // no parent plan
                            },
                          .instance_name = "pipeline_plan",
                          .edges         = std::move(edges),
                          .worker_id     = telemetry_info.worker_id,
                        });
}

}  // namespace sirius::telemetry
