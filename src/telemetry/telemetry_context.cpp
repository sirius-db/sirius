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
#include "telemetry-bridge/gen/custom_attributes.rs.h"
#include "telemetry-bridge/gen/operator.rs.h"
#include "telemetry-bridge/gen/plan.rs.h"
#include "telemetry-bridge/gen/port.rs.h"

#include <unistd.h>

#include <cstdlib>
#include <ranges>
#include <string>

namespace sirius::telemetry {

telemetry_context::telemetry_context(std::optional<std::string> query_label)
  : engine_uuid_(uuid::now_v7()),
    worker_uuid_(uuid::now_v7()),
    context_(quent::create_context(uuid::now_v7(), "ndjson", "telemetry_data")),
    engine_observer_(quent::engine::create_observer(*context_)),
    worker_observer_(quent::worker::create_observer(*context_)),
    query_label_(std::move(query_label))
{
  // Engine init
  const char* env_name          = std::getenv("SIRIUS_ENGINE_NAME");
  const std::string engine_name = env_name ? env_name : "siriusDB";

  engine_observer_->init(engine_uuid_,
                         quent::engine::Init{
                           .implementation =
                             quent::engine::Implementation{
                               .name              = engine_name,
                               .version           = "",
                               .custom_attributes = {},
                             },
                           .instance_name = engine_name,
                         });

  worker_observer_->init(worker_uuid_,
                         quent::worker::Init{
                           .parent_engine_id = engine_uuid_,
                           .instance_name    = fmt::format("worker-{}", getpid()),
                         });

  SIRIUS_LOG_INFO("Telemetry context initialized (engine={})", engine_name);
}

telemetry_context::~telemetry_context()
{
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
    auto op_names                    = operators | std::views::transform([](const auto& op) {
                      return fmt::format("{}({})", op.get().get_name(), op.get().operator_id);
                    });
    const std::string operator_chain = fmt::format("{}", fmt::join(op_names, " -> "));
    operator_obs->declaration(
      pipeline_uuid,
      quent::operator_::Declaration{
        .plan_id             = plan_id,
        .parent_operator_ids = {},
        .instance_name       = operator_chain,
        .type_name           = fmt::format("Pipeline Id {}", pipeline->get_pipeline_id()),
        .custom_attributes   = {},
      });

    // Receiver ports on pipeline source operators.
    if (auto source = pipeline->get_source()) {
      for (std::string_view port_id : source->get_port_ids()) {
        if (const op::sirius_physical_operator::port* port = source->get_port(port_id)) {
          port_obs->declaration(port->source_port_uuid,
                                quent::port::Declaration{
                                  .operator_id   = pipeline_uuid,
                                  .instance_name = fmt::format("{}_receiver", port_id),
                                });
        }
      }
    }

    // Sender ports on pipline sink(last) operators.
    if (auto sink = pipeline->get_sink()) {
      // Delim joins are composite: pipeline wiring attaches next_port_after_sink to their
      // internal sub-operators, not the delim join itself. Mirror the special-casing in
      // sirius_pipeline_converter::setup_pipeline_parents() so telemetry sees all edges.
      std::vector<op::sirius_physical_operator*> next_port_sources;
      if (sink->type == op::SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN) {
        auto& right_delim = sink->Cast<op::sirius_physical_right_delim_join>();
        next_port_sources = {right_delim.partition_join, right_delim.distinct.get()};
      } else if (sink->type == op::SiriusPhysicalOperatorType::LEFT_DELIM_JOIN) {
        auto& left_delim  = sink->Cast<op::sirius_physical_left_delim_join>();
        next_port_sources = {left_delim.column_data_scan, left_delim.distinct.get()};
      } else {
        next_port_sources = {sink.get()};
      }

      for (op::sirius_physical_operator* next_port_source : next_port_sources) {
        for (const auto& [next_operator, next_operator_port_name, pseudo_sink_port_uuid] :
             next_port_source->get_next_port_after_sink()) {
          // Declare the pseudo-sink port
          port_obs->declaration(
            pseudo_sink_port_uuid,
            quent::port::Declaration{
              .operator_id   = pipeline_uuid,
              .instance_name = fmt::format("{}_sender", next_operator_port_name),
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
