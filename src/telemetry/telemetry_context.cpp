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
#include "telemetry/batch_telemetry.hpp"

#include <unistd.h>

#include <exception>
#include <format>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sirius::telemetry {
namespace {

void log_finalization_failure(std::string_view resource_kind, const char* error) noexcept
{
  try {
    if (error != nullptr) {
      SIRIUS_LOG_WARN("Failed to finalize {} telemetry: {}", resource_kind, error);
    } else {
      SIRIUS_LOG_WARN("Failed to finalize {} telemetry: unknown exception", resource_kind);
    }
  } catch (...) {
  }
}

}  // namespace

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
  : engine_uuid_(quent::now_v7()),
    worker_uuid_(quent::now_v7()),
    query_group_uuid_(quent::now_v7()),
    shared_group_uuid_(quent::now_v7()),
    engine_name_(config.engine_name),
    context_([&config] {
      if (!config.enable_quent) { return quent::Context::none(); }
      if (config.exporter == "ndjson") { return quent::Context::ndjson(config.output_directory); }
      if (config.exporter == "msgpack") { return quent::Context::msgpack(config.output_directory); }
      if (config.exporter == "postcard") {
        return quent::Context::postcard(config.output_directory);
      }
      throw std::invalid_argument(std::format("unknown Quent exporter: {}", config.exporter));
    }()),
    engine_observer_(context_.engine_observer()),
    worker_observer_(context_.worker_observer()),
    query_group_observer_(context_.query_group_observer()),
    engine_handle_(engine_observer_->handle(quent::engine::EngineId(engine_uuid_))),
    worker_handle_(worker_observer_->handle(quent::worker::WorkerId(worker_uuid_)))
{
  engine_handle_.init(quent::engine::Init{
    .implementation =
      quent::records::EngineImplementationAttributes{
        .name              = config.engine_name,
        .version           = "",
        .custom_attributes = {},
      },
    .instance_name = config.engine_name,
  });

  worker_handle_.init(quent::worker::Init{
    .parent_engine_id = quent::engine::EngineId(engine_uuid_),
    .instance_name    = std::format("worker-{}", getpid()),
  });

  memory_context_ = std::make_shared<memory_context>(engine_uuid_, context_, manager);

  // One session-scoped query group under this engine; every query in this context is reported
  // under it, so a whole run shows up as a single group rather than one group per query.
  query_group_observer_->handle(quent::query_group::QueryGroupId(query_group_uuid_))
    .declaration(quent::query_group::Declaration{
      .instance_name = std::format("{}-session-{}", config.engine_name, getpid()),
      .engine_id     = quent::engine::EngineId(engine_uuid_),
    });

  // Per-GPU device groups plus per-thread-type buckets underneath, so the
  // viewer renders threads as an engine -> gpu-N -> thread-type tree instead
  // of a flat sibling list. Threads with no single GPU go under `shared`.
  auto gpu_device_observer   = context_.gpu_device_observer();
  auto thread_group_observer = context_.thread_group_observer();

  thread_group_observer->handle(quent::thread_group::ThreadGroupId(shared_group_uuid_))
    .declaration(quent::thread_group::Declaration{
      .instance_name   = "shared",
      .parent_group_id = engine_uuid_,
      .engine_id       = quent::engine::EngineId(engine_uuid_),
    });

  for (const int device_id : gpu_device_ids) {
    const gpu_device_group_ids ids{
      .device           = quent::now_v7(),
      .executor_threads = quent::now_v7(),
      .manager_threads  = quent::now_v7(),
    };
    gpu_device_observer->handle(quent::gpu_device::GpuDeviceId(ids.device))
      .declaration(quent::gpu_device::Declaration{
        .instance_name   = std::format("gpu-{}", device_id),
        .parent_group_id = quent::engine::EngineId(engine_uuid_),
        .ordinal         = static_cast<uint32_t>(device_id),
      });
    thread_group_observer->handle(quent::thread_group::ThreadGroupId(ids.executor_threads))
      .declaration(quent::thread_group::Declaration{
        .instance_name   = "executor_thread",
        .parent_group_id = ids.device,
        .engine_id       = quent::engine::EngineId(engine_uuid_),
      });
    thread_group_observer->handle(quent::thread_group::ThreadGroupId(ids.manager_threads))
      .declaration(quent::thread_group::Declaration{
        .instance_name   = "task_manager_loop_thread",
        .parent_group_id = ids.device,
        .engine_id       = quent::engine::EngineId(engine_uuid_),
      });
    gpu_group_ids_.emplace(device_id, ids);
  }

  SIRIUS_LOG_INFO("Telemetry context initialized (engine={}, {} GPU device group(s))",
                  config.engine_name,
                  gpu_group_ids_.size());
}

quent::Uuid telemetry_context::query_group_id_for(
  const std::optional<std::string>& session_label) const
{
  if (!session_label.has_value() || session_label->empty()) { return query_group_uuid_; }
  const std::lock_guard lock(labeled_groups_mutex_);
  auto it = labeled_group_ids_.find(*session_label);
  if (it == labeled_group_ids_.end()) {
    auto group_uuid = quent::now_v7();
    query_group_observer_->handle(quent::query_group::QueryGroupId(group_uuid))
      .declaration(quent::query_group::Declaration{
        .instance_name = std::format("{}-{}", engine_name_, *session_label),
        .engine_id     = quent::engine::EngineId(engine_uuid_),
      });
    it = labeled_group_ids_.emplace(*session_label, std::move(group_uuid)).first;
  }
  return it->second;
}

const quent::Uuid& telemetry_context::gpu_device_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.device;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

const quent::Uuid& telemetry_context::executor_thread_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.executor_threads;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

const quent::Uuid& telemetry_context::manager_thread_group_id(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second.manager_threads;
  }
  SIRIUS_LOG_WARN("Telemetry: no device group declared for GPU {}; falling back to engine group",
                  device_id);
  return engine_uuid_;
}

telemetry_context::~telemetry_context() noexcept
{
  memory_context_.reset();
  try {
    worker_handle_.exit();
  } catch (const std::exception& error) {
    log_finalization_failure("worker", error.what());
  } catch (...) {
    log_finalization_failure("worker", nullptr);
  }
  try {
    engine_handle_.exit();
  } catch (const std::exception& error) {
    log_finalization_failure("engine", error.what());
  } catch (...) {
    log_finalization_failure("engine", nullptr);
  }
}

void emit_plan_telemetry(
  const quent::Context& context,
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  const quent::Uuid plan_id,
  const query_telemetry_info telemetry_info)
{
  auto operator_obs = context.operator_observer();
  auto port_obs     = context.port_observer();
  auto plan_obs     = context.plan_observer();

  // Collect edges while iterating
  std::vector<quent::records::Edge> edges;

  for (const auto& pipeline : pipelines) {
    const auto pipeline_uuid         = pipeline->pipeline_uuid();
    const auto operators             = pipeline->get_operators();
    const std::string operator_chain = [&operators]() {
      std::string chain{};
      for (const auto& name : operators | std::views::transform([](const auto& op) {
                                return std::format(
                                  "{}({})", op.get().get_name(), op.get().get_operator_id());
                              })) {
        if (chain.empty()) {
          chain = name;
          continue;
        }
        chain = std::format("{} -> {}", chain, name);
      }
      return chain;
    }();

    operator_obs->handle(quent::operator_::OperatorId(pipeline_uuid))
      .declaration(quent::operator_::Declaration{
        .plan_id             = quent::plan::PlanId(plan_id),
        .parent_operator_ids = {},
        .instance_name       = operator_chain,
        .type_name           = std::format("Pipeline Id {}", pipeline->get_pipeline_id()),
        .custom_attributes   = {},
      });

    // Receiver ports on pipeline source operators.
    if (auto source = pipeline->get_source()) {
      for (std::string_view port_id : source->get_port_ids()) {
        if (const op::sirius_physical_operator::port* port = source->get_port(port_id)) {
          port_obs->handle(quent::port::PortId(port->source_port_uuid))
            .declaration(quent::port::Declaration{
              .operator_id   = quent::operator_::OperatorId(pipeline_uuid),
              .instance_name = std::format("{}_receiver", port_id),
            });
          batch_telemetry_registry::instance().register_consumer_port(
            port->repo, telemetry_info.query_id, pipeline_uuid, port->source_port_uuid);
        }
      }
    }

    // Sender ports on pipeline sink(last) operators.
    for (const auto& [next_operator, next_operator_port_name, pseudo_sink_port_uuid] :
         pipeline->get_next_ports_after_sink()) {
      // Declare the pseudo-sink port
      port_obs->handle(quent::port::PortId(pseudo_sink_port_uuid))
        .declaration(quent::port::Declaration{
          .operator_id   = quent::operator_::OperatorId(pipeline_uuid),
          .instance_name = std::format("{}_sender", next_operator_port_name),
        });

      // Find the target port on the downstream operator
      if (const op::sirius_physical_operator::port* target_port =
            next_operator->get_port(next_operator_port_name)) {
        edges.push_back(quent::records::Edge{
          .source = quent::port::PortId(pseudo_sink_port_uuid),
          .target = quent::port::PortId(target_port->source_port_uuid),
        });
      }
    }
  }

  plan_obs->handle(quent::plan::PlanId(plan_id))
    .declaration(quent::plan::Declaration{
      .parent =
        quent::records::PlanParent{
          .query_id = quent::query::QueryId(telemetry_info.telemetry_query_id),
          .plan_id  = std::nullopt,
        },
      .instance_name = "pipeline_plan",
      .edges         = std::move(edges),
      .worker_id     = quent::worker::WorkerId(telemetry_info.worker_id),
    });
}

}  // namespace sirius::telemetry
