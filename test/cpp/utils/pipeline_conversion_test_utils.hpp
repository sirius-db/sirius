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

#include "op/sirius_physical_streaming_sink.hpp"
#include "query_id.hpp"

#include <cucascade/data/data_repository.hpp>
#include <duckdb/main/connection.hpp>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace sirius::pipeline {
struct pipeline_conversion_result;
}  // namespace sirius::pipeline

namespace sirius {
class sirius_engine;
}

namespace sirius::test {

//! Registers a data repository manager for a synthetic query and drops it on scope exit,
//! standing in for a real `SiriusContext::StandaloneQueryScope`.
//!
//! Tests that build a `sirius_engine` directly never open an execution window, so nothing
//! would otherwise create the manager the engine requires, nor clean it up afterwards.
//! Leaving repositories behind also breaks the *next* real GPU query in the process, since
//! operator ids restart at 0 for every plan and would collide on `{operator_id, port_id}`.
//!
//! Construct it before the engine and keep it alive for at least as long: pass `query_id()`
//! to the `sirius_engine` constructor. No-op when Sirius is not registered on the connection.
class scoped_test_query {
 public:
  explicit scoped_test_query(duckdb::ClientContext& context);
  ~scoped_test_query();

  scoped_test_query(const scoped_test_query&)            = delete;
  scoped_test_query& operator=(const scoped_test_query&) = delete;

  [[nodiscard]] sirius::query_id_t query_id() const noexcept { return query_id_; }

 private:
  [[nodiscard]] bool usable() const noexcept;

  duckdb::shared_ptr<duckdb::SiriusContext> ctx_;
  sirius::query_id_t query_id_;
};

//! Drive the full sirius planner + meta_pipeline + converter flow on `query` (with the
//! production optimizer disables) and return `dump_pipeline_conversion_result(...)`; stops
//! after `converter.convert()` — no GPU execution. Returns a string rather than the raw
//! result because the result references the function-local plan tree. Throws on
//! parse / bind / optimize errors.
std::string convert_query_to_dump(duckdb::Connection& con, const std::string& query);

//! Like `convert_query_to_dump`, but returns the raw scheduled order
//! (`dump_pipeline_schedule_raw`); comparing two calls is the schedule-determinism check.
std::string convert_query_to_raw_schedule(duckdb::Connection& con, const std::string& query);

//! Like `convert_query_to_dump`, but hands the raw `pipeline_conversion_result` to `consume`
//! while the plan tree and meta-pipelines are still alive; the result must not escape
//! `consume`. Lets tests inspect the real schedule order, which the dump sorts away.
void with_conversion_result(
  duckdb::Connection& con,
  const std::string& query,
  const std::function<void(pipeline::pipeline_conversion_result&)>& consume);

//! Initialize an engine for `query` and invoke `consume` while its plan is alive.
void with_initialized_engine(duckdb::Connection& con,
                             const std::string& query,
                             const std::function<void(sirius_engine&)>& consume);

//! Like `with_initialized_engine`, but roots the plan in a STREAMING_SINK over `output_repos`
//! instead of a RESULT_COLLECTOR — the shape a streaming fragment runs. The caller (standing in
//! for the fragment) owns the plan tree, so the engine borrows it via `initialize_internal`.
//! `consume` runs while both are alive.
void with_initialized_streaming_fragment(
  duckdb::Connection& con,
  const std::string& query,
  std::vector<std::shared_ptr<cucascade::shared_data_repository>> output_repos,
  const std::function<void(sirius_engine&, op::sirius_physical_streaming_sink&)>& consume);

//! Path to the canonical TPC-H queries (`test/tpch_performance/tpch_queries/orig/`).
std::filesystem::path tpch_queries_dir();

//! Read canonical TPC-H query `q` (1..22); throws if the file cannot be opened.
std::string read_tpch_query_file(int q);

}  // namespace sirius::test
