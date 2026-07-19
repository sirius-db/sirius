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

#include <duckdb/main/connection.hpp>

#include <filesystem>
#include <functional>
#include <string>

namespace sirius::pipeline {
struct pipeline_conversion_result;
}  // namespace sirius::pipeline

namespace sirius::test {

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

//! Path to the canonical TPC-H queries (`test/tpch_performance/tpch_queries/orig/`).
std::filesystem::path tpch_queries_dir();

//! Read canonical TPC-H query `q` (1..22); throws if the file cannot be opened.
std::string read_tpch_query_file(int q);

}  // namespace sirius::test
