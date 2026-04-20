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
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry-bridge/gen/custom_attributes.rs.h"
#include "telemetry-bridge/gen/operator.rs.h"
#include "telemetry-bridge/gen/plan.rs.h"
#include "telemetry-bridge/gen/port.rs.h"

#include <cstdlib>
#include <string>

namespace sirius::telemetry {

telemetry_context::telemetry_context()
  : engine_uuid_(uuid::now_v7()),
    worker_uuid_(uuid::now_v7()),
    context_(quent::create_context(uuid::now_v7(), "ndjson", "telemetry_data")),
    engine_observer_(quent::engine::create_observer()),
    worker_observer_(quent::worker::create_observer())
{
  // Engine init
  const char* env_name    = std::getenv("SIRIUS_ENGINE_NAME");
  std::string engine_name = env_name ? env_name : "siriusDB";

  engine_observer_->init(engine_uuid_,
                         quent::engine::Init{
                           .implementation =
                             quent::engine::Implementation{
                               .name              = rust::String(engine_name),
                               .version           = rust::String(""),
                               .custom_attributes = quent::CustomAttributes{},
                             },
                           .instance_name = rust::String(engine_name),
                         });

  // Worker init
  worker_observer_->init(worker_uuid_,
                         quent::worker::Init{
                           .parent_engine_id = engine_uuid_,
                           .instance_name    = rust::String("worker-0"),
                         });

  SIRIUS_LOG_INFO("Telemetry context initialized (engine={})", engine_name);
}

telemetry_context::~telemetry_context()
{
  worker_observer_->exit(worker_uuid_);
  engine_observer_->exit(engine_uuid_);
}

void emit_plan_telemetry(
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  const uuid::UUID plan_id,
  const telemetry_info telemetry_info)
{
  auto operator_obs = quent::operator_::create_observer();
  auto port_obs     = quent::port::create_observer();
  auto plan_obs     = quent::plan::create_observer();

  // Collect edges while iterating
  rust::Vec<quent::plan::Edges> edges;

  for (const auto& pipeline : pipelines) {
    auto pipeline_uuid = pipeline->pipeline_uuid();

    // Build operator name from the chain of operators
    std::string op_chain = "[";
    auto source          = pipeline->get_source();
    if (source) { op_chain += source->get_name(); }
    for (auto& op_ref : pipeline->get_operators()) {
      op_chain += " -> " + op_ref.get().get_name();
    }
    auto sink = pipeline->get_sink();
    if (sink) { op_chain += " -> " + sink->get_name(); }
    op_chain += "]";

    std::string instance_name =
      "Pipeline ID " + std::to_string(pipeline->get_pipeline_id()) + " " + op_chain;

    operator_obs->declaration(pipeline_uuid,
                              quent::operator_::Declaration{
                                .plan_id             = plan_id,
                                .parent_operator_ids = {},
                                .instance_name       = rust::String(instance_name),
                                .type_name           = rust::String(op_chain),
                                .custom_attributes   = quent::CustomAttributes{},
                              });
    
    // Source operator ports
    if (source) {
      for (auto port_id : source->get_port_ids()) {
        auto* p = source->get_port(port_id);
        if (p) {
          port_obs->declaration(p->source_port_uuid,
                                quent::port::Declaration{
                                  .operator_id   = pipeline_uuid,
                                  .instance_name = rust::String(std::string(port_id)),
                                });
        }
      }
    }

    // Pseudo-sink ports and edges
    if (sink) {
      for (auto& npi : sink->get_next_port_after_sink()) {
        // Declare the pseudo-sink port
        port_obs->declaration(
          npi.pseudo_sink_port_uuid,
          quent::port::Declaration{
            .operator_id   = pipeline_uuid,
            .instance_name = rust::String(std::string(npi.next_operator_port_name) + "_sink"),
          });

        // Find the target port on the downstream operator
        auto* target_port = npi.next_operator->get_port(npi.next_operator_port_name);
        if (target_port) {
          edges.push_back(quent::plan::Edges{
            .source = npi.pseudo_sink_port_uuid,
            .target = target_port->source_port_uuid,
          });
        }
      }
    }
  }

  // --- Plan declaration ---
  plan_obs->declaration(plan_id,
                        quent::plan::Declaration{
                          .parent =
                            quent::plan::Parent{
                              .query_id = telemetry_info.query_id,
                              .plan_id  = uuid::new_nil(),  // no parent plan
                            },
                          .instance_name = rust::String("pipeline_plan"),
                          .edges         = std::move(edges),
                          .worker_id     = telemetry_info.worker_id,
                        });
}

}  // namespace sirius::telemetry
