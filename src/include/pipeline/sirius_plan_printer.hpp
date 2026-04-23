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

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <string>
#include <vector>

namespace sirius::pipeline {

/// Configuration for box-drawing characters and layout used by the plan printer.
struct plan_printer_config {
  // Box-drawing characters (UTF-8)
  const char* LTCORNER   = "\xe2\x94\x8c";  // U+250C "┌"
  const char* RTCORNER   = "\xe2\x94\x90";  // U+2510 "┐"
  const char* LDCORNER   = "\xe2\x94\x94";  // U+2514 "└"
  const char* RDCORNER   = "\xe2\x94\x98";  // U+2518 "┘"
  const char* LMIDDLE    = "\xe2\x94\x9c";  // U+251C "├"
  const char* RMIDDLE    = "\xe2\x94\xa4";  // U+2524 "┤"
  const char* TMIDDLE    = "\xe2\x94\xac";  // U+252C "┬"
  const char* DMIDDLE    = "\xe2\x94\xb4";  // U+2534 "┴"
  const char* VERTICAL   = "\xe2\x94\x82";  // U+2502 "│"
  const char* HORIZONTAL = "\xe2\x94\x80";  // U+2500 "─"

  // Layout constraints
  size_t minimum_render_width = 26;
  size_t maximum_render_width = 240;
};

/// Produces text output showing pipeline operator chains and the full pipeline DAG
/// with DuckDB-style box-drawing characters.
///
/// Usage:
///   sirius_plan_printer printer(engine.new_scheduled);
///   std::string pipelines_view = printer.render_pipelines();
///   std::string dag_view       = printer.render_dag();
///   std::string combined       = printer.render();
class sirius_plan_printer {
 public:
  explicit sirius_plan_printer(
    const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines);

  /// Render a compact one-line-per-pipeline summary showing operator chains and dependencies.
  std::string render_pipelines() const;

  /// Render the full pipeline dependency DAG as a tree with box-drawing characters.
  std::string render_dag() const;

  /// Render both views concatenated (render_pipelines + render_dag).
  std::string render() const;

  /// Build operator chain string for a single pipeline (source -> ops -> sink).
  static std::string build_operator_chain(const sirius_pipeline& pipeline);

  /// Convert MemoryBarrierType to a human-readable string.
  static std::string barrier_type_to_string(op::MemoryBarrierType type);

  /// Format operator name with ID suffix: "HASH_JOIN (id=2)"
  static std::string format_operator_with_id(const op::sirius_physical_operator& op);

  /// Get enrichment detail lines for an operator.
  /// Returns empty vector for operators without enrichable properties.
  static std::vector<std::string> get_operator_detail_lines(const op::sirius_physical_operator& op);

 private:
  const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines_;
  plan_printer_config config_;

  /// Find the root pipeline (the one whose sink is RESULT_COLLECTOR).
  duckdb::shared_ptr<sirius_pipeline> find_root_pipeline() const;
};

}  // namespace sirius::pipeline
