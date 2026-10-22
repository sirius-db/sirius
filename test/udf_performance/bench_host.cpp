/*
 * Copyright 2026, Sirius Contributors.
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

#include "compatibility.hpp"
#include "udfs/fast_cpu.hpp"

#include <cuda_runtime_api.h>

#include <dlfcn.h>
#include <duckdb.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/extension/extension_loader.hpp>
#include <duckdb/main/profiler/profiler_print_format.hpp>
#include <sirius/duckdb.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace fs      = std::filesystem;
using result      = duckdb::unique_ptr<duckdb::QueryResult>;
using clock_type  = std::chrono::steady_clock;
using config_type = std::map<std::string, std::string>;

std::string read_file(fs::path const& path)
{
  std::ifstream input(path);
  if (!input) { throw std::runtime_error("Cannot read " + path.string()); }
  return {std::istreambuf_iterator<char>{input}, {}};
}

result query(duckdb::Connection& connection, std::string const& sql)
{
  // Connection::Query materializes the complete result before returning.
  auto output = connection.Query(sql);
  if (!output || output->HasError()) {
    throw std::runtime_error((output ? output->GetError() : "Missing query result") +
                             "\nSQL: " + sql);
  }
  return output;
}

std::string json_quote(std::string const& value)
{
  std::ostringstream output;
  output << '"';
  for (unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 32) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(character)
                 << std::dec;
        } else {
          output << character;
        }
    }
  }
  return output.str() + '"';
}

void compare(duckdb::QueryResult& expected, duckdb::QueryResult& actual)
{
  if (expected.GetTypes() != actual.GetTypes() || expected.RowCount() != actual.RowCount()) {
    throw std::runtime_error("Result logical types or row counts differ");
  }
  auto expected_rows = expected.Collection().GetRows();
  auto actual_rows   = actual.Collection().GetRows();
  for (duckdb::idx_t row = 0; row < expected.RowCount(); ++row) {
    for (duckdb::idx_t column = 0; column < expected.ColumnCount(); ++column) {
      auto const left  = expected_rows.GetValue(column, row);
      auto const right = actual_rows.GetValue(column, row);
      // These workloads return integers and strings. With identical logical types,
      // their string representations retain every bit of the non-NULL value.
      if (left.IsNull() != right.IsNull() ||
          (!left.IsNull() && left.ToString() != right.ToString())) {
        throw std::runtime_error("Result mismatch at row " + std::to_string(row) + ", column " +
                                 std::to_string(column) + ": expected " + left.ToString() +
                                 ", actual " + right.ToString());
      }
    }
  }
}

struct counters {
  uint64_t executions{};
  uint64_t plan{};
  uint64_t runtime{};
};

counters snapshot(duckdb::Connection& observer)
{
  auto output = query(observer,
                      "SELECT executions,plan_fallbacks,runtime_fallbacks "
                      "FROM sirius_execution_stats()");
  auto rows   = output->Collection().GetRows();
  return {rows.GetValue(0, 0).GetValue<uint64_t>(),
          rows.GetValue(1, 0).GetValue<uint64_t>(),
          rows.GetValue(2, 0).GetValue<uint64_t>()};
}

counters checked_delta(counters before, counters after)
{
  if (after.executions < before.executions || after.plan < before.plan ||
      after.runtime < before.runtime) {
    throw std::runtime_error("GPU execution counters moved backwards");
  }
  counters delta{
    after.executions - before.executions, after.plan - before.plan, after.runtime - before.runtime};
  if (delta.executions != 1 || delta.plan != 0 || delta.runtime != 0) {
    throw std::runtime_error(
      "Unexpected GPU execution/fallback counters: " + std::to_string(delta.executions) + "," +
      std::to_string(delta.plan) + "," + std::to_string(delta.runtime));
  }
  return delta;
}

struct mode {
  std::string id;
  std::string label;
  bool gpu{};
  std::string sample_sql;
  std::string dispersed_sql;
  std::string aggregate_sql;
  std::string projection_sql;
  std::vector<double> times;
  std::vector<double> timestamps;
  counters counts;
};

void write_array(std::ostream& output, std::vector<double> const& values)
{
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) { output << ','; }
    output << std::setprecision(17) << values[index];
  }
  output << ']';
}

config_type read_config(fs::path const& job)
{
  config_type config;
  std::istringstream input{read_file(job / "config.txt")};
  std::string line;
  while (std::getline(input, line)) {
    auto const position = line.find('=');
    if (position != std::string::npos) {
      config.emplace(line.substr(0, position), line.substr(position + 1));
    }
  }
  return config;
}

bool flag(config_type const& config, std::string const& name)
{
  auto const& value = config.at(name);
  if (value != "0" && value != "1") {
    throw std::runtime_error("Configuration flag must be 0 or 1: " + name);
  }
  return value == "1";
}

std::string disperse(std::string sql, uint64_t rows)
{
  std::string const needle = "WHERE id<4096";
  auto const position      = sql.find(needle);
  if (position == std::string::npos) {
    throw std::runtime_error("Sample query is missing the prefix predicate");
  }
  auto const stride = std::max<uint64_t>(1, rows / 4096);
  auto const last   = rows > 512 ? rows - 512 : 0;
  auto const predicate =
    "WHERE (id%" + std::to_string(stride) + "=0 OR id<256 OR id>=" + std::to_string(last) + ")";
  sql.replace(position, needle.size(), predicate);
  return sql;
}

mode load_mode(fs::path const& job, uint64_t rows, std::string id, std::string label, bool gpu)
{
  mode output;
  output.id             = std::move(id);
  output.label          = std::move(label);
  output.gpu            = gpu;
  output.sample_sql     = read_file(job / ("sample_" + output.id + ".sql"));
  output.dispersed_sql  = disperse(output.sample_sql, rows);
  output.aggregate_sql  = read_file(job / ("aggregate_" + output.id + ".sql"));
  output.projection_sql = read_file(job / ("projection_" + output.id + ".sql"));
  return output;
}

void write_record(std::ostream& output,
                  std::string const& workload,
                  uint64_t rows,
                  std::string const& kind,
                  std::string const& cache,
                  mode const& cpu,
                  mode const& gpu,
                  double pin_seconds)
{
  output << "{\"workload\":" << json_quote(workload) << ",\"rows\":" << rows
         << ",\"query_kind\":" << json_quote(kind + " / " + cache)
         << ",\"cache_mode\":" << json_quote(cache) << ",\"cpu_baseline\":" << json_quote(cpu.label)
         << ",\"cpu_samples_ms\":";
  write_array(output, cpu.times);
  output << ",\"gpu_samples_ms\":";
  write_array(output, gpu.times);
  output << ",\"cpu_sample_unix_ms\":";
  write_array(output, cpu.timestamps);
  output << ",\"gpu_sample_unix_ms\":";
  write_array(output, gpu.timestamps);
  output << ",\"gpu_counters\":{\"executions\":" << gpu.counts.executions
         << ",\"plan_fallbacks\":0,\"runtime_fallbacks\":0},\"verification\":\"passed\","
            "\"pin_setup_seconds\":"
         << pin_seconds << '}';
}

struct validation_evidence {
  uint64_t gpu_queries{};
  uint64_t prefix_rows{};
  uint64_t dispersed_rows{};
  std::vector<std::string> records;
};

void validate_samples(duckdb::Connection& connection,
                      duckdb::Connection& observer,
                      std::vector<mode> const& cpus,
                      mode const& gpu,
                      std::string const& cache,
                      uint64_t rows,
                      validation_evidence& evidence)
{
  for (bool dispersed : {false, true}) {
    query(connection, "SET gpu_execution=false");
    auto const sql_for = [dispersed](mode const& candidate) -> std::string const& {
      return dispersed ? candidate.dispersed_sql : candidate.sample_sql;
    };
    auto reference = query(connection, sql_for(cpus.front()));
    for (auto const& cpu : cpus) {
      auto actual = query(connection, sql_for(cpu));
      compare(*reference, *actual);
    }
    query(connection, "SET gpu_execution=true");
    auto const before = snapshot(observer);
    auto actual       = query(connection, sql_for(gpu));
    auto const delta  = checked_delta(before, snapshot(observer));
    compare(*reference, *actual);
    query(connection, "SET gpu_execution=false");
    if (reference->RowCount() == 0) { throw std::runtime_error("Correctness sample is empty"); }
    auto values              = reference->Collection().GetRows();
    auto const first         = values.GetValue(0, 0).GetValue<int64_t>();
    auto const last          = values.GetValue(0, reference->RowCount() - 1).GetValue<int64_t>();
    auto const expected_last = dispersed ? rows - 1 : std::min<uint64_t>(rows, 4096) - 1;
    if (first != 0 || last != static_cast<int64_t>(expected_last)) {
      throw std::runtime_error("Correctness sample does not cover its expected endpoints");
    }
    if (dispersed) {
      evidence.dispersed_rows = reference->RowCount();
    } else {
      evidence.prefix_rows = reference->RowCount();
    }
    evidence.gpu_queries += delta.executions;
    std::ostringstream record;
    record << "{\"cache_mode\":" << json_quote(cache)
           << ",\"sample\":" << json_quote(dispersed ? "dispersed" : "prefix")
           << ",\"rows_compared\":" << reference->RowCount() << ",\"min_id\":" << first
           << ",\"max_id\":" << last << ",\"gpu_executions\":" << delta.executions
           << ",\"plan_fallbacks\":0,\"runtime_fallbacks\":0}";
    evidence.records.push_back(record.str());
  }
}

void validate_query_shapes(duckdb::Connection& connection,
                           duckdb::Connection& observer,
                           std::vector<mode> const& cpus,
                           mode const& gpu,
                           std::string const& cache,
                           uint64_t rows,
                           bool projection,
                           validation_evidence& evidence)
{
  for (std::string const kind : {std::string{"aggregate"}, std::string{"projection"}}) {
    if (kind == "projection" && !projection) { continue; }
    auto const sql_for = [&kind](mode const& candidate) -> std::string const& {
      return kind == "aggregate" ? candidate.aggregate_sql : candidate.projection_sql;
    };
    query(connection, "SET gpu_execution=false");
    auto reference           = query(connection, sql_for(cpus.front()));
    auto const expected_rows = kind == "aggregate" ? 1 : rows;
    auto const check         = [&](duckdb::QueryResult& actual) {
      if (actual.RowCount() != expected_rows || actual.GetTypes() != reference->GetTypes()) {
        throw std::runtime_error("Validation query result logical types or row count differ");
      }
      if (kind == "aggregate") { compare(*reference, actual); }
      // Full projections have no ORDER BY and can arrive in different orders.
      // Their cell values are covered by the ordered prefix/dispersed checks;
      // executing this shape additionally validates materialization and schema.
    };
    check(*reference);
    for (auto const& cpu : cpus) {
      auto actual = query(connection, sql_for(cpu));
      check(*actual);
    }
    query(connection, "SET gpu_execution=true");
    auto const before = snapshot(observer);
    auto actual       = query(connection, sql_for(gpu));
    auto const delta  = checked_delta(before, snapshot(observer));
    check(*actual);
    query(connection, "SET gpu_execution=false");
    evidence.gpu_queries += delta.executions;
    std::ostringstream record;
    record << "{\"cache_mode\":" << json_quote(cache) << ",\"sample\":" << json_quote(kind)
           << ",\"rows_checked\":" << expected_rows
           << ",\"rows_compared\":" << (kind == "aggregate" ? 1 : 0) << ",\"verification\":"
           << json_quote(kind == "aggregate" ? "exact_values" : "row_count_and_logical_types")
           << ",\"gpu_executions\":" << delta.executions
           << ",\"plan_fallbacks\":0,\"runtime_fallbacks\":0}";
    evidence.records.push_back(record.str());
  }
}

void validate_q1_oracle(duckdb::Connection& connection,
                        duckdb::Connection& observer,
                        validation_evidence& evidence)
{
  query(connection, "SET gpu_execution=false");
  auto expected = query(connection,
                        "SELECT expected_year,expected_month,expected_day "
                        "FROM q1_oracle ORDER BY seq");
  if (expected->RowCount() != 50000) {
    throw std::runtime_error("Q1 oracle must contain all 50,000 public source rows");
  }
  query(connection, "SET gpu_execution=true");
  auto const before = snapshot(observer);
  auto actual       = query(connection,
                      "SELECT extractyear(value),extractmonth(value),extractday(value) "
                            "FROM q1_oracle ORDER BY seq");
  auto const delta  = checked_delta(before, snapshot(observer));
  compare(*expected, *actual);
  query(connection, "SET gpu_execution=false");
  evidence.gpu_queries += delta.executions;
  evidence.records.emplace_back(
    "{\"cache_mode\":\"warm_storage\",\"sample\":\"q1_original_python_oracle\","
    "\"rows_compared\":50000,\"gpu_executions\":1,\"plan_fallbacks\":0,\"runtime_fallbacks\":0}");
}

int run_stage_profiles(fs::path const& job,
                       duckdb::Connection& connection,
                       duckdb::Connection& observer,
                       config_type const& config,
                       std::vector<mode> const& cpus,
                       mode const& gpu,
                       uint64_t rows,
                       int warmups,
                       int repetitions,
                       bool projection,
                       validation_evidence& evidence)
{
  bool const aggregate = config.contains("stage_aggregate") && flag(config, "stage_aggregate");
  if ((!aggregate && !projection) || flag(config, "pin") || flag(config, "validate_only")) {
    throw std::runtime_error("Stage profiles require a query kind, no pinning and timed execution");
  }
  validate_samples(connection, observer, cpus, gpu, "warm_storage", rows, evidence);
  query(connection, "PRAGMA disable_profiling");
  std::ostringstream profiles;
  profiles << std::setprecision(17);
  bool first           = true;
  uint64_t gpu_queries = 0;
  for (std::string const kind : {std::string{"projection"}, std::string{"aggregate"}}) {
    if ((kind == "projection" && !projection) || (kind == "aggregate" && !aggregate)) { continue; }
    auto const sql_for = [&kind](mode const& candidate) -> std::string const& {
      return kind == "aggregate" ? candidate.aggregate_sql : candidate.projection_sql;
    };
    query(connection, "SET gpu_execution=false");
    auto reference           = query(connection, sql_for(cpus.front()));
    auto const logical_types = reference->GetTypes();
    auto const expected_rows = kind == "aggregate" ? 1 : rows;
    if (reference->RowCount() != expected_rows) {
      throw std::runtime_error("Stage reference query result row count mismatch");
    }
    // Unordered full projections are checked by shape here and exact ordered
    // samples above. Release this full result before running profiled queries.
    if (kind == "projection") { reference.reset(); }
    auto modes = cpus;
    modes.push_back(gpu);
    for (int iteration = 0; iteration < warmups + repetitions; ++iteration) {
      for (std::size_t step = 0; step < modes.size(); ++step) {
        auto const& candidate = modes[(step + iteration) % modes.size()];
        auto const sample_id  = candidate.id + "_" + kind + "_" + std::to_string(iteration);
        auto sql              = sql_for(candidate);
        auto const last       = sql.find_last_not_of(" \t\r\n");
        auto const marker     = " /*sirius_stage:" + sample_id + "*/";
        if (last != std::string::npos && sql[last] == ';') {
          sql.insert(last, marker);
        } else {
          sql += marker;
        }
        query(connection, candidate.gpu ? "SET gpu_execution=true" : "SET gpu_execution=false");
        if (!candidate.gpu) { query(connection, "PRAGMA enable_profiling='no_output'"); }
        auto const before  = candidate.gpu ? snapshot(observer) : counters{};
        auto const started = clock_type::now();
        auto output        = query(connection, sql);
        auto const milliseconds =
          std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
        std::string profile_file;
        if (!candidate.gpu) {
          // Read before any later query on this connection replaces its profile.
          auto const profile =
            connection.GetProfilingInformation(duckdb::ProfilerPrintFormat::JSON());
          query(connection, "PRAGMA disable_profiling");
          if (profile.empty()) { throw std::runtime_error("Missing CPU query profile"); }
          profile_file = "cpu-profile-" + sample_id + ".json";
          std::ofstream saved{job / profile_file};
          saved.exceptions(std::ios::failbit | std::ios::badbit);
          saved << profile;
        }
        if (candidate.gpu) {
          auto const delta = checked_delta(before, snapshot(observer));
          gpu_queries += delta.executions;
        }
        if (output->RowCount() != expected_rows || output->GetTypes() != logical_types) {
          throw std::runtime_error("Profiled query result logical types or row count mismatch");
        }
        if (kind == "aggregate") { compare(*reference, *output); }
        if (!first) { profiles << ','; }
        first = false;
        profiles << "{\"sample_id\":" << json_quote(sample_id)
                 << ",\"engine\":" << json_quote(candidate.gpu ? "gpu" : "cpu")
                 << ",\"cpu_mode\":" << json_quote(candidate.id)
                 << ",\"cpu_baseline\":" << json_quote(candidate.label)
                 << ",\"query_kind\":" << json_quote(kind)
                 << ",\"warmup\":" << (iteration < warmups ? "true" : "false")
                 << ",\"iteration\":" << iteration << ",\"query_wall_ms\":" << milliseconds
                 << ",\"profile_file\":" << json_quote(profile_file)
                 << ",\"sql\":" << json_quote(sql);
        if (candidate.gpu) {
          profiles
            << ",\"gpu_counters\":{\"executions\":1,\"plan_fallbacks\":0,\"runtime_fallbacks\":0}";
        }
        profiles << '}';
      }
    }
    std::cout << config.at("workload") << ' ' << rows << " stage " << kind << " passed\n"
              << std::flush;
  }
  auto const kinds = static_cast<uint64_t>(aggregate) + static_cast<uint64_t>(projection);
  if (gpu_queries != kinds * static_cast<uint64_t>(warmups + repetitions)) {
    throw std::runtime_error("GPU stage evidence does not cover every warmup and measured query");
  }
  std::ostringstream report;
  report << "{\"correctness_verified\":true,\"rows\":" << rows << ",\"input_rows\":" << rows
         << ",\"gpu_validation_queries\":" << evidence.gpu_queries
         << ",\"gpu_counters\":{\"executions\":" << gpu_queries
         << ",\"plan_fallbacks\":0,\"runtime_fallbacks\":0},\"validation_records\":[";
  for (std::size_t index = 0; index < evidence.records.size(); ++index) {
    if (index) { report << ','; }
    report << evidence.records[index];
  }
  report << "],\"profiles\":[" << profiles.str() << "]}\n";
  auto const temporary = job / "stage-measurements.json.tmp";
  {
    std::ofstream output{temporary};
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << report.str();
  }
  fs::rename(temporary, job / "stage-measurements.json");
  return 0;
}

int run(fs::path const& job)
{
  auto const config        = read_config(job);
  auto const rows          = std::stoull(config.at("rows"));
  auto const warmups       = std::stoi(config.at("warmups"));
  auto const repetitions   = std::stoi(config.at("repetitions"));
  auto const validate_only = flag(config, "validate_only");
  auto const pin           = flag(config, "pin");
  auto const projection    = flag(config, "projection");
  if (rows == 0 || warmups < 1 || repetitions < 5 || std::stoi(config.at("threads")) < 1) {
    throw std::runtime_error("Require positive rows/threads, >=1 warmup and >=5 repetitions");
  }

  void* entry        = nullptr;
  auto const& plugin = config.at("plugin");
  if (plugin != "-") {
    // Deliberately retain the mapping until process exit: registered catalog
    // callbacks and bound plans can outlive an individual query or connection.
    void* library = dlopen(plugin.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library) { throw std::runtime_error(dlerror()); }
    entry = dlsym(library, "register_sirius_udf");
    if (!entry) { throw std::runtime_error("Missing UDF registration hook"); }
  }
  duckdb::DuckDB database{(job / "data.duckdb").c_str()};
  duckdb::Connection connection{database};
  {
    duckdb::ExtensionLoader loader{duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                   "sirius"};
    sirius::register_duckdb_extension(loader);
  }
  if (entry) {
    duckdb::ExtensionLoader loader{duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                   "benchmark_udf"};
    reinterpret_cast<void (*)(duckdb::ExtensionLoader&)>(entry)(loader);
  }
  {
    duckdb::ExtensionLoader loader{duckdb::DatabaseInstance::GetDatabase(*connection.context),
                                   "benchmark_fast_cpu"};
    register_fast_cpu(loader);
  }
  // Runtime counters are shared across the database. CPU observation must not
  // introduce another GPU query into the interval being measured.
  duckdb::Connection observer{database};
  query(observer, "SET gpu_execution=false");
  query(connection, "SET gpu_execution=false");
  query(connection, "SET threads=" + config.at("threads"));
  query(connection, "SET enable_duckdb_fallback=false");
  query(connection, "SET expression_evaluator_strategy='ast_interpret'");

  auto const setup_sql        = read_file(job / "setup.sql");
  auto const registration_sql = read_file(job / "registration.sql");
  auto started                = clock_type::now();
  query(connection, setup_sql);
  query(connection, registration_sql);
  query(connection, "CHECKPOINT");
  auto const setup_seconds = std::chrono::duration<double>(clock_type::now() - started).count();
  auto profile = query(connection, "SELECT count(*)::BIGINT,count(value)::BIGINT FROM bench_input");
  auto profile_rows = profile->Collection().GetRows();
  if (profile_rows.GetValue(0, 0).GetValue<uint64_t>() != rows) {
    throw std::runtime_error("Input row count differs from configuration");
  }
  auto const non_null_rows = profile_rows.GetValue(1, 0).GetValue<uint64_t>();
  auto storage             = query(connection, "PRAGMA storage_info('bench_input')");
  std::ofstream{job / "storage-info.txt"} << storage->ToString();

  std::vector<mode> cpus;
  std::istringstream cpu_list{read_file(job / "cpu_modes.tsv")};
  std::string line;
  while (std::getline(cpu_list, line)) {
    auto const position = line.find('\t');
    if (position != std::string::npos) {
      cpus.push_back(
        load_mode(job, rows, line.substr(0, position), line.substr(position + 1), false));
    }
  }
  if (cpus.empty()) { throw std::runtime_error("No CPU baseline modes supplied"); }
  auto const gpu_mode = load_mode(job, rows, "gpu", "GPU", true);

  validation_evidence evidence;
  std::ostringstream records;
  bool first_record = true;
  if (config.at("workload") == "udfbench_q1") {
    validate_q1_oracle(connection, observer, evidence);
  }
  if (config.contains("stage_profile") && flag(config, "stage_profile")) {
    return run_stage_profiles(job,
                              connection,
                              observer,
                              config,
                              cpus,
                              gpu_mode,
                              rows,
                              warmups,
                              repetitions,
                              projection,
                              evidence);
  }
  for (std::string const cache : {std::string{"warm_storage"}, std::string{"pinned_gpu"}}) {
    if (cache == "pinned_gpu" && !pin) { continue; }
    double pin_seconds = 0;
    if (cache == "pinned_gpu") {
      query(connection, "SET gpu_execution=false");
      started = clock_type::now();
      query(connection,
            "CALL pin_table(format='duckdb',name='bench_input',tier='gpu',compression=false)");
      pin_seconds = std::chrono::duration<double>(clock_type::now() - started).count();
    }
    // Both sample sets and every CPU alternative are checked before timing in
    // each cache mode. Dispersed IDs reach later row groups and both endpoints.
    validate_samples(connection, observer, cpus, gpu_mode, cache, rows, evidence);
    if (validate_only) {
      validate_query_shapes(
        connection, observer, cpus, gpu_mode, cache, rows, projection, evidence);
    }
    if (!validate_only) {
      for (std::string const kind : {std::string{"aggregate"}, std::string{"projection"}}) {
        if (kind == "projection" && !projection) { continue; }
        auto modes = cpus;
        modes.push_back(gpu_mode);
        query(connection, "SET gpu_execution=false");
        result aggregate_reference;
        if (kind == "aggregate") {
          aggregate_reference = query(connection, cpus.front().aggregate_sql);
        }
        for (int iteration = 0; iteration < warmups + repetitions; ++iteration) {
          for (std::size_t step = 0; step < modes.size(); ++step) {
            auto& candidate = modes[(step + iteration) % modes.size()];
            query(connection, candidate.gpu ? "SET gpu_execution=true" : "SET gpu_execution=false");
            auto const before = candidate.gpu ? snapshot(observer) : counters{};
            auto const& sql =
              kind == "aggregate" ? candidate.aggregate_sql : candidate.projection_sql;
            auto const timestamp = std::chrono::duration<double, std::milli>(
                                     std::chrono::system_clock::now().time_since_epoch())
                                     .count();
            started     = clock_type::now();
            auto output = query(connection, sql);
            auto const milliseconds =
              std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
            if (output->RowCount() != (kind == "aggregate" ? 1 : rows)) {
              throw std::runtime_error("Timed query result row count mismatch");
            }
            if (candidate.gpu) {
              auto const delta = checked_delta(before, snapshot(observer));
              candidate.counts.executions += delta.executions;
            }
            if (kind == "aggregate") { compare(*aggregate_reference, *output); }
            if (iteration >= warmups) {
              candidate.times.push_back(milliseconds);
              candidate.timestamps.push_back(timestamp);
            }
          }
        }
        auto const& gpu = modes.back();
        if (gpu.counts.executions != static_cast<uint64_t>(warmups + repetitions)) {
          throw std::runtime_error(
            "GPU execution evidence does not cover every timed/warmup query");
        }
        for (std::size_t index = 0; index < cpus.size(); ++index) {
          if (!first_record) { records << ','; }
          first_record = false;
          write_record(
            records, config.at("workload"), rows, kind, cache, modes[index], gpu, pin_seconds);
        }
        std::cout << config.at("workload") << ' ' << rows << ' ' << cache << ' ' << kind
                  << " passed\n"
                  << std::flush;
      }
    }
    if (cache == "pinned_gpu") {
      query(connection, "SET gpu_execution=false");
      query(connection, "CALL unpin_table('bench_input')");
    }
  }

  // Publish JSON only after every validation and benchmark variant succeeds.
  // Failed jobs retain logs and SQL, but cannot look like a passing report.
  std::ostringstream report;
  report << std::setprecision(12)
         << "{\"correctness_verified\":true,\"gpu_validation_queries\":" << evidence.gpu_queries
         << ",\"setup_seconds\":" << setup_seconds << ",\"input_rows\":" << rows
         << ",\"non_null_rows\":" << non_null_rows
         << ",\"correctness_sample_rows\":" << evidence.prefix_rows
         << ",\"dispersed_sample_rows\":" << evidence.dispersed_rows << ",\"validation_records\":[";
  for (std::size_t index = 0; index < evidence.records.size(); ++index) {
    if (index) { report << ','; }
    report << evidence.records[index];
  }
  report << "],\"records\":[" << records.str() << "]}\n";
  auto const temporary = job / "measurements.json.tmp";
  {
    std::ofstream output{temporary};
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << report.str();
  }
  fs::rename(temporary, job / "measurements.json");
  return 0;
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    if (argc == 2 && std::string{argv[1]} == "--device-info") {
      auto const check = [](cudaError_t status) {
        if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
      };
      int count = 0;
      check(cudaGetDeviceCount(&count));
      if (count == 0) { throw std::runtime_error("No visible CUDA devices"); }
      cudaDeviceProp device{};
      std::size_t free = 0, total = 0;
      check(cudaSetDevice(0));
      check(cudaGetDeviceProperties(&device, 0));
      check(cudaMemGetInfo(&free, &total));
      std::cout << "{\"device_count\":" << count << ",\"name\":" << json_quote(device.name)
                << ",\"total_bytes\":" << total << ",\"free_bytes\":" << free << "}\n";
      return 0;
    }
    if (argc != 2) { throw std::runtime_error("Usage: bench_host job-directory"); }
    require_compatible_duckdb(SIRIUS_UDF_DUCKDB_REVISION, duckdb::DuckDB::SourceID());
    return run(std::filesystem::absolute(argv[1]));
  } catch (std::exception const& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
