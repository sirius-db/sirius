// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "compatibility.hpp"

#include <dlfcn.h>
#include <duckdb.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <sirius/duckdb.hpp>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SIRIUS_UDF_DUCKDB_REVISION
#define SIRIUS_UDF_DUCKDB_REVISION ""
#endif

namespace {
using result_type = duckdb::unique_ptr<duckdb::QueryResult>;

result_type query(duckdb::Connection& connection, std::string const& sql)
{
  auto result = connection.Query(sql);
  if (!result || result->HasError()) {
    throw std::runtime_error(result ? result->GetError() : "query returned no result");
  }
  return result;
}

std::string read(std::string const& path)
{
  std::ifstream stream(path);
  if (!stream) { throw std::runtime_error("cannot read SQL file"); }
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

struct counters {
  uint64_t executions;
  uint64_t plan_fallbacks;
  uint64_t runtime_fallbacks;
};

counters snapshot(duckdb::Connection& connection)
{
  query(connection, "SET gpu_execution=false");
  auto result =
    query(connection,
          "SELECT executions, plan_fallbacks, runtime_fallbacks FROM sirius_execution_stats()");
  query(connection, "SET gpu_execution=true");
  auto values = result->Collection().GetRows();
  return {values.GetValue(0, 0).GetValue<uint64_t>(),
          values.GetValue(1, 0).GetValue<uint64_t>(),
          values.GetValue(2, 0).GetValue<uint64_t>()};
}

counters require_gpu(counters before, counters after, uint64_t min_executions = 1)
{
  if (after.executions <= before.executions ||
      after.executions - before.executions < min_executions ||
      after.plan_fallbacks != before.plan_fallbacks ||
      after.runtime_fallbacks != before.runtime_fallbacks) {
    throw std::runtime_error("GPU execution was absent or fallback occurred");
  }
  return {after.executions - before.executions,
          after.plan_fallbacks - before.plan_fallbacks,
          after.runtime_fallbacks - before.runtime_fallbacks};
}

void compare(duckdb::QueryResult& expected, duckdb::QueryResult& actual)
{
  if (expected.GetTypes() != actual.GetTypes() || expected.RowCount() != actual.RowCount()) {
    throw std::runtime_error("result types or row counts differ");
  }
  auto expected_values = expected.Collection().GetRows();
  auto actual_values   = actual.Collection().GetRows();
  for (duckdb::idx_t row = 0; row < expected.RowCount(); ++row) {
    for (duckdb::idx_t column = 0; column < expected.ColumnCount(); ++column) {
      auto const lhs = expected_values.GetValue(column, row);
      auto const rhs = actual_values.GetValue(column, row);
      bool equal     = lhs.IsNull() == rhs.IsNull();
      if (equal && !lhs.IsNull()) {
        auto const type = expected.GetTypes()[column].id();
        if (type == duckdb::LogicalTypeId::DOUBLE) {
          equal = std::bit_cast<uint64_t>(lhs.GetValue<double>()) ==
                  std::bit_cast<uint64_t>(rhs.GetValue<double>());
        } else if (type == duckdb::LogicalTypeId::FLOAT) {
          equal = std::bit_cast<uint32_t>(lhs.GetValue<float>()) ==
                  std::bit_cast<uint32_t>(rhs.GetValue<float>());
        } else {
          equal = lhs.ToString() == rhs.ToString();
        }
      }
      if (!equal) {
        throw std::runtime_error("result value or NULL mask differs at row " + std::to_string(row));
      }
    }
  }
}

struct measurement {
  std::vector<double> seconds;
  counters gpu_counts{};
};

measurement benchmark(
  duckdb::Connection& connection, std::string const& sql, int warmups, int iterations, bool gpu)
{
  query(connection, gpu ? "SET gpu_execution=true" : "SET gpu_execution=false");
  auto const before = gpu ? snapshot(connection) : counters{};
  measurement result;
  for (int iteration = 0; iteration < warmups + iterations; ++iteration) {
    auto const started = std::chrono::steady_clock::now();
    auto rows          = query(connection, sql);  // Query materializes all rows before returning.
    auto const elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (iteration >= warmups) { result.seconds.push_back(elapsed); }
  }
  if (gpu) {
    result.gpu_counts =
      require_gpu(before, snapshot(connection), static_cast<uint64_t>(warmups) + iterations);
  }
  return result;
}

void print(counters const& value)
{
  std::cout << "{\"executions\":" << value.executions
            << ",\"plan_fallbacks\":" << value.plan_fallbacks
            << ",\"runtime_fallbacks\":" << value.runtime_fallbacks << '}';
}

void print(measurement const& value)
{
  auto sorted = value.seconds;
  std::sort(sorted.begin(), sorted.end());
  auto const size   = sorted.size();
  auto const median = size % 2 ? sorted[size / 2] : (sorted[size / 2 - 1] + sorted[size / 2]) / 2;
  std::cout << "{\"median_seconds\":" << median << ",\"samples_seconds\":[";
  for (std::size_t index = 0; index < value.seconds.size(); ++index) {
    if (index) { std::cout << ','; }
    std::cout << value.seconds[index];
  }
  std::cout << "],\"gpu_counters\":";
  print(value.gpu_counts);
  std::cout << '}';
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    if (argc != 9) { throw std::runtime_error("use run.py to invoke the native host"); }
    auto const warmups    = std::stoi(argv[7]);
    auto const iterations = std::stoi(argv[8]);
    if (warmups < 0 || iterations < 1) { throw std::runtime_error("invalid iteration counts"); }

    // Refuse incompatible C++ layouts before loading code or registering callbacks.
    require_compatible_duckdb(SIRIUS_UDF_DUCKDB_REVISION, duckdb::DuckDB::SourceID());

    void* entry = nullptr;
    if (std::string_view(argv[2]) != "-") {
      // Retain the DSO for the process lifetime: catalogs and prepared plans own callbacks.
      auto* plugin = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
      if (!plugin) { throw std::runtime_error(dlerror()); }
      entry = dlsym(plugin, "register_sirius_udf");
      if (!entry) { throw std::runtime_error("plugin has no register_sirius_udf hook"); }
    }

    duckdb::DuckDB database(argv[1]);  // File-backed native GPU scan.
    duckdb::Connection connection(database);
    {
      duckdb::ExtensionLoader sirius_loader(
        duckdb::DatabaseInstance::GetDatabase(*connection.context), "sirius");
      sirius::register_duckdb_extension(sirius_loader);
    }
    if (entry) {
      duckdb::ExtensionLoader udf_loader(duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                         "sirius_udf_example");
      reinterpret_cast<void (*)(duckdb::ExtensionLoader&)>(entry)(udf_loader);
    }
    query(connection, "SET gpu_execution=false");
    query(connection, read(argv[3]));
    auto const registration = read(argv[4]);
    if (registration.find_first_not_of(" \t\n\r") != std::string::npos) {
      query(connection, registration);
    }
    // Native GPU scans read persisted DuckDB storage blocks.
    query(connection, "CHECKPOINT");
    auto const reference_sql = read(argv[5]);
    auto const candidate_sql = read(argv[6]);
    auto reference           = query(connection, reference_sql);
    auto candidate_cpu       = query(connection, candidate_sql);
    compare(*reference, *candidate_cpu);
    query(connection, "SET enable_duckdb_fallback=false");
    query(connection, "SET gpu_execution=true");
    auto const before  = snapshot(connection);
    auto candidate_gpu = query(connection, candidate_sql);
    compare(*reference, *candidate_gpu);
    auto const evidence = require_gpu(before, snapshot(connection));
    auto cpu            = benchmark(connection, reference_sql, warmups, iterations, false);
    auto gpu            = benchmark(connection, candidate_sql, warmups, iterations, true);

    std::cout << "SIRIUS_UDF_RESULT={\"validation\":\"passed\",\"rows\":" << reference->RowCount()
              << ",\"timing\":\"query wall time including result materialization\""
              << ",\"cpu_baseline\":\"reference.sql\",\"gpu_counters\":";
    print(evidence);
    std::cout << ",\"cpu\":";
    print(cpu);
    std::cout << ",\"gpu\":";
    print(gpu);
    std::cout << "}\n";
    return 0;
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
