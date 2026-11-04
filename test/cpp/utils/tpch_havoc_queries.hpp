/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * See the LICENSE file at the repo root for the full text.
 */

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sirius::test {

/// One distinct TPC-Havoc query: BenchBox's rewrite `variant` of TPC-H query `base_query`.
struct tpch_havoc_query {
  int base_query = 0;                   ///< TPC-H query number, 1-22.
  int variant    = 0;                   ///< BenchBox variant id, 1-10.
  std::string label;                    ///< "Q05_V03".
  std::string description;              ///< BenchBox's one-line description of the rewrite.
  std::string sql;                      ///< DuckDB SQL rendered for the requested scale factor.
  std::vector<std::string> duplicates;  ///< Labels of later variants with byte-identical SQL.
};

/// What happens when a query runs with `enable_duckdb_fallback = false`.
enum class tpch_havoc_outcome {
  gpu,             ///< Ran on the GPU and matched DuckDB on the CPU.
  mismatch,        ///< Ran on the GPU and differed from DuckDB on the CPU.
  plan_rejected,   ///< "GPU plan generation failed: ...".
  runtime_failed,  ///< "Sirius GPU execution failed: ...".
  skip,            ///< Not run: it crashes, hangs or is flaky.
};

std::string to_string(tpch_havoc_outcome outcome);

/// One entry of test/tpch_havoc/expectations.yaml: queries that share a root cause and an issue.
struct tpch_havoc_expectation {
  tpch_havoc_outcome outcome = tpch_havoc_outcome::gpu;
  std::string reason;  ///< Regex the error must contain; for `mismatch` and `skip`, a description.
  int issue = 0;       ///< Open sirius-db/sirius issue that tracks the fix.
  std::string scan;    ///< "native" or "parquet"; empty applies to both.
  std::string evaluator;             ///< "ast_interpret" or "ast_jit"; empty applies to both.
  std::vector<std::string> queries;  ///< Labels, such as "Q05_V03".

  bool applies_to(std::string const& scan_name, std::string const& evaluator_name) const;
};

/// test/tpch_havoc/expectations.yaml in the source tree.
std::filesystem::path tpch_havoc_expectations_path();

/// Loads the expected outcomes. Throws on a malformed entry: an unknown outcome, scan or
/// evaluator, a `gpu` outcome, a missing issue, or a reason that isn't a valid regex.
std::vector<tpch_havoc_expectation> load_tpch_havoc_expectations(std::filesystem::path const& path);

/// The entry that lists `label` for this scan and evaluator, or nullptr when the query is expected
/// to run on the GPU. Throws when more than one entry lists it.
tpch_havoc_expectation const* find_tpch_havoc_expectation(
  std::vector<tpch_havoc_expectation> const& expectations,
  std::string const& label,
  std::string const& scan,
  std::string const& evaluator);

/// Run on each connection before the queries. Q01_V07 uses the `x -> ...` lambda syntax, which
/// DuckDB accepts only with this (deprecated) setting.
inline constexpr char kTpchHavocSessionSetup[] = "SET lambda_syntax = 'ENABLE_SINGLE_ARROW'";

/// Directory with BenchBox's `q01.yaml` ... `q22.yaml`: the SIRIUS_TPCH_HAVOC_DIR environment
/// variable if set, else the copy CMake downloaded, else nullopt when the build was configured
/// with -DSIRIUS_TPCH_HAVOC=OFF.
std::optional<std::filesystem::path> tpch_havoc_dir();

/// Loads every variant in `dir`, renders Q11's `{q11_fraction}` (0.0001 / SF) for `scale_factor`,
/// and keeps one query per distinct SQL text: Q19-Q22 repeat the canonical query ten times.
/// Throws on a missing file or a template parameter it doesn't know.
std::vector<tpch_havoc_query> load_tpch_havoc_queries(std::filesystem::path const& dir,
                                                      double scale_factor);

}  // namespace sirius::test
