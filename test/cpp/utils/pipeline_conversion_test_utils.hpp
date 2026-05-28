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

#include <string>

namespace sirius::test {

//! Drive the full sirius planner + meta_pipeline + converter flow on `query`
//! and return `dump_pipeline_conversion_result(...)` for the resulting conversion.
//! Mirrors the path `sirius_extension.cpp` and `sirius_engine::initialize_internal`
//! follow at runtime, but stops after `converter.convert()` — no GPU execution,
//! no materialize_repository_wiring side effects.
//!
//! Returns the dump *string* (not the raw `pipeline_conversion_result`) because the
//! result's pipelines reference operators in the plan tree, and the tree is owned
//! by a local in this function — extending its lifetime to the caller would require
//! returning both, and is easier to get wrong than just dumping while everything is
//! still alive.
//!
//! Optimizer disables match `SiriusTableFunctionData::PrepareConnection`
//! (IN_CLAUSE, COMPRESSED_MATERIALIZATION, STATISTICS_PROPAGATION). The
//! pre-existing settings are saved and restored even on error.
//!
//! Used by Sub-phase E.1's differential dump test to compare the conversion
//! result between `USE_TREE_BASED_PIPELINE_BUILD=false` and `=true`. The flag's
//! state at call time is what the converter and plan generator see; tests must
//! toggle it before each call.
//!
//! Throws on parse / bind / optimize errors. Iceberg queries are not supported
//! (no engine-side prefetch is performed) — TPC-H queries are unaffected.
std::string convert_query_to_dump(duckdb::Connection& con, const std::string& query);

}  // namespace sirius::test
