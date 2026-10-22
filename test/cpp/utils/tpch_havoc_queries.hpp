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
