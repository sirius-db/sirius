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

#pragma once

#include "duckdb/common/common.hpp"
#include "telemetry-bridge/gen/context.rs.h"
#include "telemetry-bridge/gen/engine.rs.h"
#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry-bridge/gen/worker.rs.h"

#include <string>

namespace sirius::pipeline {
class sirius_pipeline;
}  // namespace sirius::pipeline

namespace sirius {
struct telemetry_config;
}  // namespace sirius

namespace sirius::telemetry {

/// Owns the top-level telemetry states for a single SiriusContext.
class telemetry_context {
 public:
  explicit telemetry_context(const sirius::telemetry_config& config);
  ~telemetry_context();

  // Non-copyable, non-movable (owns opaque Rust boxes)
  telemetry_context(const telemetry_context&)            = delete;
  telemetry_context& operator=(const telemetry_context&) = delete;
  telemetry_context(telemetry_context&&)                 = delete;
  telemetry_context& operator=(telemetry_context&&)      = delete;

  [[nodiscard]] const uuid::UUID& engine_id() const { return engine_uuid_; }
  [[nodiscard]] const uuid::UUID& worker_id() const { return worker_uuid_; }
  [[nodiscard]] const quent::Context& context() const { return *context_; }

 private:
  uuid::UUID engine_uuid_;
  uuid::UUID worker_uuid_;
  rust::Box<quent::Context> context_;
  rust::Box<quent::engine::EngineObserver> engine_observer_;
  rust::Box<quent::worker::WorkerObserver> worker_observer_;
};

// A POD to hold common identifiers for useful telemetry.
struct query_telemetry_info {
  uuid::UUID query_id;
  uuid::UUID worker_id;
};

/// Emit plan-level telemetry (operator declarations, port declarations, edges)
/// for the given set of pipelines. Called once during query construction.
void emit_plan_telemetry(
  const quent::Context& context,
  const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& pipelines,
  uuid::UUID plan_id,
  query_telemetry_info telemetry_info);

}  // namespace sirius::telemetry
