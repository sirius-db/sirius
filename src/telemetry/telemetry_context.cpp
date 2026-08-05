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

#include "late_mat/column_origin.hpp"
#include "log/logging.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius_config.hpp"
#include "telemetry-bridge/gen/operator.rs.h"
#include "telemetry-bridge/gen/plan.rs.h"
#include "telemetry-bridge/gen/port.rs.h"
#include "telemetry/batch_telemetry.hpp"

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <cstdlib>
#include <format>
#include <memory>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>

namespace sirius::telemetry {

namespace {

/// Snapshot the resolved engine config + hardware info into the engine Init
/// event so traces are self-describing (thread counts, memory-space limits,
/// scan/cache/IO settings, GPU SM/clock info). One-time emission at engine
/// init; steady-state cost is zero.
quent::DynamicAttributes build_engine_custom_attributes(const sirius::sirius_config& config)
{
  quent::DynamicAttributes attrs;
  auto add_str = [&attrs](std::string key, std::string value) {
    attrs.string_attrs.push_back({std::move(key), std::move(value)});
  };
  auto add_i64 = [&attrs](std::string key, int64_t value) {
    attrs.i64_attrs.push_back({std::move(key), value});
  };
  auto add_f64 = [&attrs](std::string key, double value) {
    attrs.f64_attrs.push_back({std::move(key), value});
  };

  const auto& topo = config.get_hw_topology();
  add_str("host.name", topo.hostname);
  add_i64("hw.num_gpus", static_cast<int64_t>(topo.num_gpus));
  add_i64("hw.num_numa_nodes", topo.num_numa_nodes);
  add_i64("hw.host_cores", static_cast<int64_t>(std::thread::hardware_concurrency()));

  for (const auto& gpu : topo.gpus) {
    const auto prefix = std::format("gpu.{}", gpu.id);
    add_str(std::format("{}.name", prefix), gpu.name);
    add_i64(std::format("{}.numa_node", prefix), gpu.numa_node);
    // cudaDeviceGetAttribute rather than cudaDeviceProp fields: the clock-rate
    // prop members are deprecated/removed on newer CUDA toolkits.
    auto add_device_attr = [&](const char* name, cudaDeviceAttr attr) {
      int value = 0;
      if (::cudaDeviceGetAttribute(&value, attr, static_cast<int>(gpu.id)) == cudaSuccess) {
        add_i64(std::format("{}.{}", prefix, name), value);
      }
    };
    add_device_attr("sm_count", cudaDevAttrMultiProcessorCount);
    add_device_attr("sm_clock_khz", cudaDevAttrClockRate);
    add_device_attr("mem_clock_khz", cudaDevAttrMemoryClockRate);
    add_device_attr("mem_bus_width_bits", cudaDevAttrGlobalMemoryBusWidth);
  }

  for (const auto& space_config : config.get_memory_space_configs()) {
    if (const auto* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space_config)) {
      const auto prefix = std::format("memory.gpu{}", gpu->device_id);
      add_i64(std::format("{}.capacity_bytes", prefix), static_cast<int64_t>(gpu->memory_capacity));
      add_i64(std::format("{}.reservation_limit_bytes", prefix),
              static_cast<int64_t>(gpu->reservation_limit()));
    } else if (const auto* host =
                 std::get_if<cucascade::memory::host_memory_space_config>(&space_config)) {
      const auto prefix = std::format("memory.host{}", host->numa_id);
      add_i64(std::format("{}.capacity_bytes", prefix),
              static_cast<int64_t>(host->memory_capacity));
      add_i64(std::format("{}.reservation_limit_bytes", prefix),
              static_cast<int64_t>(host->reservation_limit()));
    } else if (const auto* disk =
                 std::get_if<cucascade::memory::disk_memory_space_config>(&space_config)) {
      add_i64(std::format("memory.disk{}.capacity_bytes", disk->disk_id),
              static_cast<int64_t>(disk->memory_capacity));
    }
  }

  add_i64("executor.num_threads", config.get_gpu_pipeline_executor_config().num_threads);
  add_i64("task_creator.num_threads", config.get_task_creator_config().thread_pool.num_threads);
  add_i64("downgrade.num_threads", config.get_downgrade_executor_config().thread_pool.num_threads);
  add_i64("downgrade.monitor_period_ms",
          config.get_downgrade_executor_config().monitor_period.count());

  const auto& scan = config.get_scan_manager_config();
  add_i64("scan_manager.num_threads", scan.thread_pool.num_threads);
  add_str("scan_manager.io_backend", scan.use_sirius_datasource ? "uring" : "kvikio");
  add_i64("scan_manager.uring_n_reactors", static_cast<int64_t>(scan.uring_n_reactors));
  add_i64("scan_manager.rest_n_reactors", static_cast<int64_t>(scan.rest_n_reactors));
  add_i64("scan_manager.prefetch_cache_enabled", scan.enable_prefetch_cache ? 1 : 0);
  add_i64("scan_manager.memory_prefetcher.enabled", scan.memory_prefetcher.enable ? 1 : 0);
  add_i64("scan_manager.memory_prefetcher.num_threads",
          static_cast<int64_t>(scan.memory_prefetcher.num_threads));
  add_i64("scan_manager.cache.inflight_io_chunk_budget",
          static_cast<int64_t>(scan.cache.inflight_io_chunk_budget));
  add_f64("scan_manager.cache.min_prefetching_budget_fraction",
          scan.cache.min_prefetching_budget_fraction);
  add_f64("scan_manager.cache.eviction_threshold_fraction", scan.cache.eviction_threshold_fraction);

  const auto& params = config.get_operator_params();
  add_i64("operator.scan_task_batch_size", static_cast<int64_t>(params.scan_task_batch_size));
  add_i64("operator.hash_partition_bytes", static_cast<int64_t>(params.hash_partition_bytes));

  add_i64("telemetry.batch_events", config.get_telemetry_config().enable_batch_events ? 1 : 0);

  // Experimental feature gates (env-driven, read with the same set-and-!="0"
  // convention as their in-engine readers): record whether the late-mat /
  // fused-scan-filter paths were lit so every trace is attributable to the
  // engine mode that produced it. late_mat.* values are EFFECTIVE (sub-gates
  // imply their parents; defer defaults ON under the main gate, matching
  // late_mat_defer_policy.cpp).
  auto env_flag_on = [](const char* name) {
    char const* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  };
  auto env_flag_default_on = [](const char* name) {
    char const* v = std::getenv(name);
    return v == nullptr || v[0] == '\0' || !(v[0] == '0' && v[1] == '\0');
  };
  add_i64("late_mat.enabled", late_mat::late_mat_enabled() ? 1 : 0);
  add_i64("late_mat.v2", late_mat::late_mat_v2_enabled() ? 1 : 0);
  add_i64("late_mat.v3", late_mat::late_mat_v3_enabled() ? 1 : 0);
  add_i64("late_mat.defer",
          late_mat::late_mat_enabled() && env_flag_default_on("SIRIUS_EXP_LATE_MAT_DEFER") ? 1 : 0);
  add_i64("late_mat.compressed",
          late_mat::late_mat_enabled() && env_flag_on("SIRIUS_EXP_LATE_MAT_COMPRESSED") ? 1 : 0);
  if (char const* cols = std::getenv("SIRIUS_LATE_MAT_PIN_UNIQUE_COLS");
      cols != nullptr && cols[0] != '\0') {
    add_str("late_mat.pin_unique_cols", cols);
  }
  add_i64("fused_scan_filter.enabled", env_flag_on("SIRIUS_EXP_FUSED_SCAN_FILTER") ? 1 : 0);

  return attrs;
}

}  // namespace

std::shared_ptr<const telemetry_context> telemetry_context::create(
  const sirius::telemetry_config& config,
  const cucascade::memory::memory_reservation_manager* manager,
  const std::vector<int>& gpu_device_ids,
  const sirius::sirius_config* full_config)
{
  return std::shared_ptr<telemetry_context>(
    new telemetry_context(config, manager, gpu_device_ids, full_config));
}

telemetry_context::telemetry_context(const sirius::telemetry_config& config,
                                     const cucascade::memory::memory_reservation_manager* manager,
                                     const std::vector<int>& gpu_device_ids,
                                     const sirius::sirius_config* full_config)
  : engine_uuid_(uuid::now_v7()),
    worker_uuid_(uuid::now_v7()),
    query_group_uuid_(uuid::now_v7()),
    shared_group_uuid_(uuid::now_v7()),
    context_(quent::create_context([&config] {
      if (!config.enable_quent) { return quent::ExporterOptions::none(); }
      if (config.exporter == "ndjson") {
        return quent::ExporterOptions::ndjson(config.output_directory);
      }
      if (config.exporter == "msgpack") {
        return quent::ExporterOptions::msgpack(config.output_directory);
      }
      if (config.exporter == "postcard") {
        return quent::ExporterOptions::postcard(config.output_directory);
      }
      throw std::invalid_argument(std::format("unknown Quent exporter: {}", config.exporter));
    }())),
    engine_observer_(quent::engine::create_observer(*context_)),
    worker_observer_(quent::worker::create_observer(*context_)),
    query_group_observer_(quent::query_group::create_observer(*context_))
{
  engine_observer_->init(
    engine_uuid_,
    quent::engine::Init{
      .implementation =
        quent::engine::Implementation{
          .name              = config.engine_name,
          .version           = "",
          .custom_attributes = full_config != nullptr ? build_engine_custom_attributes(*full_config)
                                                      : quent::DynamicAttributes{},
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
          batch_telemetry_registry::instance().register_consumer_port(
            port->repo, pipeline_uuid, port->source_port_uuid);
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
                              .query_id = telemetry_info.telemetry_query_id,
                              .plan_id  = uuid::new_nil(),  // no parent plan
                            },
                          .instance_name = "pipeline_plan",
                          .edges         = std::move(edges),
                          .worker_id     = telemetry_info.worker_id,
                        });
}

}  // namespace sirius::telemetry
