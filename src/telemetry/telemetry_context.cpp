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
#include "telemetry-bridge/gen/quent.hpp"

#include <unistd.h>

#include <format>
#include <memory>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <vector>

namespace sirius::telemetry {

quent::Context make_quent_context(const sirius::telemetry_config& config)
{
  if (not config.enable_quent) { return quent::Context::none(); }
  if (config.exporter == "ndjson") { return quent::Context::ndjson(config.output_directory); }
  if (config.exporter == "msgpack") { return quent::Context::msgpack(config.output_directory); }
  if (config.exporter == "postcard") { return quent::Context::postcard(config.output_directory); }
  throw std::invalid_argument(std::format("unknown Quent exporter: {}", config.exporter));
}

std::shared_ptr<const telemetry_context> telemetry_context::create(
  quent::Context&& context,
  const sirius::telemetry_config& config,
  const cucascade::memory::memory_reservation_manager* manager)
{
  return std::shared_ptr<telemetry_context>(
    new telemetry_context(std::move(context), config, manager));
}

telemetry_context::telemetry_context(quent::Context&& context,
                                     const telemetry_config& config,
                                     const cucascade::memory::memory_reservation_manager* manager)
  : engine_name_(config.engine_name),
    context_(std::move(context)),
    engine_handle_(context_.engine_observer()->handle()),
    worker_handle_(context_.worker_observer()->handle()),
    default_query_group_handle_(context_.query_group_observer()->handle()),
    shared_thread_group_handle_(context_.thread_group_observer()->handle())
{
  engine_handle_.init({.label = config.engine_name});
  worker_handle_.init({
    .parent_engine_id = engine_handle_.id(),
    .process_id       = std::format("{}", getpid()),
    .tag              = "worker",
  });

  // One session-scoped query group under this engine; every query in this context is reported
  // under it, so a whole run shows up as a single group rather than one group per query.
  default_query_group_handle_.declaration({
    .label     = std::format("{}-default-session-{}", config.engine_name, getpid()),
    .engine_id = engine_handle_.id(),
  });

  // Per-GPU device groups plus per-thread-type buckets underneath, so the
  // viewer renders threads as an engine -> gpu-N -> thread-type tree instead
  // of a flat sibling list. Threads with no single GPU go under `shared`.
  auto gpu_device_observer   = context_.gpu_device_observer();
  auto thread_group_observer = context_.thread_group_observer();

  shared_thread_group_handle_.declaration({
    .label         = "shared-thread-group",
    .worker_id     = worker_handle_.id(),
    .gpu_device_id = std::nullopt,
  });

  std::unordered_map<int, quent::gpu_device::GpuDeviceId> device_id_to_quent_id;
  if (manager) {
    for (const auto* gpu_memory_space :
         manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU)) {
      int device_id          = gpu_memory_space->get_device_id();
      auto gpu_device_handle = gpu_device_observer->handle();
      gpu_device_handle.declaration({
        .label     = std::format("gpu-{}", device_id),
        .worker_id = worker_handle_.id(),
        .ordinal   = static_cast<uint32_t>(device_id),
      });

      auto manager_thread_group = thread_group_observer->handle();
      manager_thread_group.declaration({
        .label         = std::format("gpu-{}-manager-threads", device_id),
        .worker_id     = worker_handle_.id(),
        .gpu_device_id = gpu_device_handle.id(),
      });

      auto executor_thread_group = thread_group_observer->handle();
      executor_thread_group.declaration({
        .label         = std::format("gpu-{}-executor-threads", device_id),
        .worker_id     = worker_handle_.id(),
        .gpu_device_id = gpu_device_handle.id(),
      });

      device_id_to_quent_id.emplace(device_id, gpu_device_handle.id());
      gpu_group_ids_.emplace(device_id,
                             gpu_device_telemtry_handles{
                               .device           = std::move(gpu_device_handle),
                               .manager_threads  = std::move(manager_thread_group),
                               .executor_threads = std::move(executor_thread_group),
                             });
    }
  }

  memory_context_ =
    std::make_shared<memory_context>(worker_handle_.id(), context_, manager, device_id_to_quent_id);

  SIRIUS_LOG_INFO("Telemetry context initialized (engine={}, {} GPU device group(s))",
                  config.engine_name,
                  gpu_group_ids_.size());
}

const quent::query_group::QueryGroupId telemetry_context::query_group_id_for(
  const std::optional<std::string>& session_label) const
{
  if (!session_label.has_value() || session_label->empty()) {
    return default_query_group_handle_.id();
  }
  const std::lock_guard lock(labeled_groups_mutex_);
  auto it = labeled_group_ids_.find(*session_label);
  if (it == labeled_group_ids_.end()) {
    auto session_query_group_handle = context_.query_group_observer()->handle();
    session_query_group_handle.declaration({
      .label     = std::format("{}-{}", engine_name_, *session_label),
      .engine_id = engine_handle_.id(),
    });
    it = labeled_group_ids_.emplace(*session_label, session_query_group_handle.id()).first;
  }
  return it->second;
}
const telemetry_context::gpu_device_telemtry_handles&
telemetry_context::gpu_device_telemetry_handles(int device_id) const
{
  if (const auto it = gpu_group_ids_.find(device_id); it != gpu_group_ids_.end()) {
    return it->second;
  }
  SIRIUS_LOG_WARN(
    "Telemetry: no device group declared for GPU {}; falling back to fallback GPU device group",
    device_id);
  return fallback_gpu_device_telemtry_handles();
}

telemetry_context::~telemetry_context()
{
  memory_context_.reset();
  for (auto& [device_id, handles] : gpu_group_ids_) {
    exit_from_destructor(handles.executor_threads, "gpu executor thread group");
    exit_from_destructor(handles.manager_threads, "gpu manager thread group");
  }
  if (fallback_gpu_device_handles_) {
    exit_from_destructor(fallback_gpu_device_handles_->executor_threads, "fallback executor group");
    exit_from_destructor(fallback_gpu_device_handles_->manager_threads, "fallback manager group");
  }
  exit_from_destructor(shared_thread_group_handle_, "shared thread group");
  exit_from_destructor(worker_handle_, "worker");
  exit_from_destructor(engine_handle_, "engine");
}

void emit_plan_telemetry(const quent::Context& context,
                         const std::vector<std::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
                         const quent::Uuid plan_id,
                         const query_telemetry_info telemetry_info)
{
  auto operator_observer                 = context.operator_observer();
  auto port_observer                     = context.port_observer();
  quent::Handle<quent::Plan> plan_handle = context.plan_observer()->handle();

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

    quent::Handle<quent::Operator> operator_handle =
      operator_observer->handle(quent::operator_::OperatorId{pipeline_uuid});
    operator_handle.declaration({
      .plan_id           = plan_handle.id(),
      .label             = operator_chain,
      .type_name         = std::format("Pipeline Id {}", pipeline->get_pipeline_id()),
      .custom_attributes = {},
    });

    // Receiver ports on pipeline source operators.
    if (auto source = pipeline->get_source()) {
      for (std::string_view port_id : source->get_port_ids()) {
        if (const op::sirius_physical_operator::port* port = source->get_port(port_id)) {
          quent::Handle<quent::Port> port_handle =
            port_observer->handle(quent::port::PortId{port->source_port_uuid});
          port_handle.declaration({
            .operator_id = operator_handle.id(),
            .label       = std::format("{}_receiver", port_id),
          });
        }
      }
    }

    // Sender ports on pipeline sink(last) operators.
    for (const auto& [next_operator, next_operator_port_name, pseudo_sink_port_uuid] :
         pipeline->get_next_ports_after_sink()) {
      // Declare the pseudo-sink port
      quent::Handle<quent::Port> pseudo_sink_port_handle =
        port_observer->handle(quent::port::PortId{pseudo_sink_port_uuid});
      pseudo_sink_port_handle.declaration({
        .operator_id = operator_handle.id(),
        .label       = std::format("{}_sender", next_operator_port_name),
      });

      // Find the target port on the downstream operator
      if (const op::sirius_physical_operator::port* target_port =
            next_operator->get_port(next_operator_port_name)) {
        edges.push_back(quent::records::Edge{
          .source = pseudo_sink_port_handle.id(),
          .target = quent::port::PortId(target_port->source_port_uuid),
        });
      }
    }
  }

  plan_handle.declaration({
    .query_id  = quent::query::QueryId{telemetry_info.telemetry_query_id},
    .label     = "pipeline_plan",
    .edges     = std::move(edges),
    .worker_id = quent::worker::WorkerId{telemetry_info.worker_id},
  });
}

}  // namespace sirius::telemetry
