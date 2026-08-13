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

#include "catch.hpp"
#include "sirius_context.hpp"

#include <cudf/contiguous_split.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream.hpp>

#include <cuda_runtime_api.h>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <data/data_batch_utils.hpp>
#include <data/sirius_converter_registry.hpp>
#include <duckdb.hpp>
#include <duckdb/execution/execution_context.hpp>
#include <duckdb/main/client_config.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <utils/utils.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>  // for setenv/putenv
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <set>
#include <source_location>
#include <string>
#include <vector>

namespace {
// MGPU-06 test helper: enable CUDA driver-level peer access for every GPU
// pair, idempotently, with sticky-error consumption (matches Plan 07-01's
// enable-loop pattern at sirius_context.cpp). Test-scope because this
// TEST_CASE builds a bare memory manager rather than going through
// SiriusContext::initialize() — without this the peer-async convert path
// hits cudaErrorIllegalAddress on the return leg. Returns true if at
// least one pair is bidirectionally P2P-capable.
bool enable_p2p_for_test(int num_gpus)
{
  bool any_enabled = false;
  for (int i = 0; i < num_gpus; ++i) {
    for (int j = 0; j < num_gpus; ++j) {
      if (i == j) { continue; }
      int can_access = 0;
      if (cudaDeviceCanAccessPeer(&can_access, i, j) != cudaSuccess || !can_access) {
        (void)cudaGetLastError();
        continue;
      }
      cudaError_t prev_dev_err = cudaSetDevice(i);
      (void)prev_dev_err;
      cudaError_t enable_err = cudaDeviceEnablePeerAccess(j, 0);
      (void)cudaGetLastError();  // consume sticky state (see 07-01 SUMMARY)
      if (enable_err == cudaSuccess || enable_err == cudaErrorPeerAccessAlreadyEnabled) {
        any_enabled = true;
      }
    }
  }
  cudaSetDevice(0);
  (void)cudaGetLastError();
  return any_enabled;
}

// MGPU-06 data integrity guard (Phase 7 / RESEARCH.md Pitfall 2). FNV-1a
// checksum over a packed batch payload. Duplicated here because the
// equivalent helper in test/cpp/downgrade/test_downgrade_executor.cpp lives
// in an anonymous namespace and is not reachable from this TU.
// Phase 18 / DB-03: const dropped from data_batch& parameter (mirrors
// debug_utils.hpp pattern from plan 18-04). cucascade #117's to_read_only is
// non-const because it acquires the shared lock.
uint64_t compute_batch_checksum_fnv1a64(cucascade::data_batch& batch, rmm::cuda_stream_view stream)
{
  // Phase 18 / DB-03 Recipe R1: scoped read-only accessor for the lifetime
  // of gpu_rep, packed, and host_buf — released at function exit.
  auto ro             = batch.to_read_only();
  auto const& gpu_rep = ro.get_data()->cast<cucascade::gpu_table_representation>();
  auto packed         = cudf::pack(gpu_rep.get_table_view(), stream);
  stream.synchronize();

  auto const bytes = packed.gpu_data->size();
  std::vector<uint8_t> host_buf(bytes);
  cudaMemcpyAsync(
    host_buf.data(), packed.gpu_data->data(), bytes, cudaMemcpyDeviceToHost, stream.value());
  stream.synchronize();

  uint64_t h = 0xcbf29ce484222325ULL;
  for (auto b : host_buf) {
    h ^= static_cast<uint64_t>(b);
    h *= 0x100000001b3ULL;
  }
  return h;
}
}  // namespace

namespace fs = std::filesystem;

struct finally {
  std::function<void()> func;
  ~finally()
  {
    if (func) { func(); }
  }
};

namespace {
struct setting_assignment {
  const char* name;
  const char* value;
};

constexpr std::array<setting_assignment, 10> legacy_only_settings{{
  {"use_pin_memory", "false"},
  {"use_pin_memory_for_caching", "true"},
  {"use_cudf_expr", "false"},
  {"use_custom_top_n", "false"},
  {"use_opt_table_scan", "false"},
  {"opt_table_scan_num_streams", "4"},
  {"opt_table_scan_memcpy_size", "1048576"},
  {"print_gpu_table_max_rows", "42"},
  {"enable_fallback_check", "true"},
  {"modified_pipeline", "true"},
}};

constexpr std::array<const char*, 5> super_sirius_settings{{
  "expression_evaluator_strategy",
  "enable_regex_jit_impl",
  "enable_duckdb_fallback",
  "fuse_merge_pipelines",
  "scan_task_batch_size",
}};
}  // namespace

TEST_CASE("Legacy-only settings follow the build surface",
          "[sirius][config][legacy-settings][isolated_context]")
{
  finally cleanup_env{[]() { setenv("SIRIUS_DISABLE", "1", 1); }};
  setenv("SIRIUS_DISABLE", "1", 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto setting_count = [&con](const char* name) {
    auto result =
      con.Query("SELECT count(*) FROM duckdb_settings() WHERE name = '" + std::string(name) + "'");
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
    return result->GetValue(0, 0).GetValue<int64_t>();
  };

  for (auto const& setting : legacy_only_settings) {
    CAPTURE(setting.name);
#ifdef SIRIUS_ENABLE_LEGACY
    REQUIRE(setting_count(setting.name) == 1);

    auto set_result = con.Query("SET " + std::string(setting.name) + " = " + setting.value);
    REQUIRE(set_result != nullptr);
    REQUIRE_FALSE(set_result->HasError());

    auto reset_result = con.Query("RESET " + std::string(setting.name));
    REQUIRE(reset_result != nullptr);
    REQUIRE_FALSE(reset_result->HasError());
#else
    REQUIRE(setting_count(setting.name) == 0);

    auto set_result = con.Query("SET " + std::string(setting.name) + " = " + setting.value);
    REQUIRE(set_result != nullptr);
    REQUIRE(set_result->HasError());
#endif
  }

  for (auto const* name : super_sirius_settings) {
    CAPTURE(name);
    REQUIRE(setting_count(name) == 1);
  }
}

TEST_CASE("Test-only settings require explicit process opt-in",
          "[sirius][config][test-settings][isolated_context]")
{
  std::optional<std::string> original_test_options;
  if (auto const* value = std::getenv("SIRIUS_ENABLE_TEST_OPTIONS")) {
    original_test_options = value;
  }
  finally restore_env{[original_test_options]() {
    if (original_test_options) {
      setenv("SIRIUS_ENABLE_TEST_OPTIONS", original_test_options->c_str(), 1);
    } else {
      unsetenv("SIRIUS_ENABLE_TEST_OPTIONS");
    }
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  auto setting_count = [](duckdb::Connection& con) {
    auto result = con.Query(
      "SELECT count(*) FROM duckdb_settings() "
      "WHERE name = 'sirius_test_inject_transparent_gpu_error'");
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
    return result->GetValue(0, 0).GetValue<int64_t>();
  };

  setenv("SIRIUS_DISABLE", "1", 1);
  unsetenv("SIRIUS_ENABLE_TEST_OPTIONS");
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    REQUIRE(setting_count(con) == 0);
    auto result = con.Query("SET sirius_test_inject_transparent_gpu_error = 'boom'");
    REQUIRE(result != nullptr);
    REQUIRE(result->HasError());
  }

  setenv("SIRIUS_ENABLE_TEST_OPTIONS", "true", 1);
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    REQUIRE(setting_count(con) == 0);
  }

  setenv("SIRIUS_ENABLE_TEST_OPTIONS", "1", 1);
  {
    duckdb::DuckDB db(nullptr);
    duckdb::Connection con(db);
    REQUIRE(setting_count(con) == 1);
    auto result = con.Query("SET sirius_test_inject_transparent_gpu_error = 'boom'");
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
  }
}

TEST_CASE("Sirius configuration loading from file with configurator",
          "[sirius][context][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "configurator.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  // get client context from con
  auto& client_ctx = *con.context;
  // get registered sirius context
  auto sirius_ctx = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 3);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 1);

  auto const& telemetry = sirius_ctx->get_config().get_telemetry_config();
  REQUIRE_FALSE(telemetry.enable_quent);
  REQUIRE(telemetry.output_directory == "/tmp/sirius_telemetry_config_test");
  REQUIRE(telemetry.engine_name == "sirius_config_test");
}

TEST_CASE("Sirius configuration rejects zero hash partition bytes", "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  fs::path cfg =
    fs::path(loc.file_name()).parent_path() / "data" / "invalid_hash_partition_zero.yaml";

  sirius::sirius_config config;
  REQUIRE_THROWS_WITH(
    config.load_from_file(cfg),
    Catch::Contains("hash_partition_bytes") && Catch::Contains("greater than zero"));
}

namespace {

void require_shared_operator_defaults(const sirius::operator_params& params, uint64_t batch)
{
  REQUIRE(params.scan_task_batch_size == batch);
  REQUIRE(params.hash_partition_bytes == batch);
  REQUIRE(params.concat_batch_bytes == batch);
  REQUIRE(params.sort_sample_bytes == batch);
  REQUIRE(params.max_build_hash_table_bytes == 2 * batch);
}

fs::path config_fixture(std::string const& name)
{
  std::source_location loc = std::source_location::current();
  return fs::path(loc.file_name()).parent_path() / "data" / name;
}

uint64_t expected_effective_batch(const sirius::sirius_config& config)
{
  std::optional<uint64_t> min_capacity;
  for (auto const& space : config.get_memory_space_configs()) {
    auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space);
    if (gpu == nullptr || gpu->memory_capacity == 0) { continue; }
    auto const capacity = static_cast<uint64_t>(gpu->memory_capacity);
    min_capacity        = min_capacity ? std::min(*min_capacity, capacity) : capacity;
  }
  REQUIRE(min_capacity.has_value());
  return std::min(sirius::config::derived_default_batch_size(),
                  std::max<uint64_t>(1, *min_capacity / 40));
}

}  // namespace

TEST_CASE("operator batch defaults use the smallest low-level GPU capacity",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  config.load_from_file(config_fixture("effective_operator_defaults_low_level.yaml"));

  constexpr uint64_t effective_capacity = 256ULL * 1024 * 1024;
  constexpr uint64_t expected_batch     = effective_capacity / 40;
  require_shared_operator_defaults(config.get_operator_params(), expected_batch);
}

TEST_CASE("operator batch defaults use the high-level GPU usage limit",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  config.load_from_file(config_fixture("effective_operator_defaults_high_level.yaml"));

  constexpr uint64_t effective_capacity = 256ULL * 1024 * 1024;
  constexpr uint64_t expected_batch     = effective_capacity / 40;
  require_shared_operator_defaults(config.get_operator_params(), expected_batch);
}

TEST_CASE("operator batch defaults use an explicit high-level GPU usage fraction",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  config.load_from_file(config_fixture("minimal.yaml"));

  require_shared_operator_defaults(config.get_operator_params(), expected_effective_batch(config));
}

TEST_CASE("explicit operator batch values override effective-capacity defaults",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  config.load_from_file(config_fixture("effective_operator_defaults_explicit.yaml"));

  constexpr uint64_t mib = 1024ULL * 1024;
  auto const& params     = config.get_operator_params();
  REQUIRE(params.scan_task_batch_size == 1 * mib);
  REQUIRE(params.hash_partition_bytes == 2 * mib);
  REQUIRE(params.concat_batch_bytes == 3 * mib);
  REQUIRE(params.sort_sample_bytes == 4 * mib);
  REQUIRE(params.max_build_hash_table_bytes == 5 * mib);
}

TEST_CASE("ordinary defaults stay physical-memory-derived without an explicit GPU cap",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  auto const before = config.get_operator_params();
  config.load_from_file(config_fixture("effective_operator_defaults_high_level.yaml"));
  config.apply_defaults();

  require_shared_operator_defaults(config.get_operator_params(), before.scan_task_batch_size);
}

TEST_CASE("null GPU usage limits do not enable effective-capacity defaults",
          "[sirius][config][operator_defaults]")
{
  const char* fixtures[] = {"effective_operator_defaults_null_bytes.yaml",
                            "effective_operator_defaults_null_fraction.yaml"};

  for (auto const* fixture : fixtures) {
    INFO("fixture=" << fixture);
    sirius::sirius_config config;
    auto const physical_default = config.get_operator_params().scan_task_batch_size;

    config.load_from_file(config_fixture(fixture));
    require_shared_operator_defaults(config.get_operator_params(), physical_default);
  }
}

TEST_CASE("repeated loads clear explicit and cap-derived operator values",
          "[sirius][config][operator_defaults]")
{
  sirius::sirius_config config;
  auto const physical_default = config.get_operator_params().scan_task_batch_size;

  config.load_from_file(config_fixture("effective_operator_defaults_explicit.yaml"));
  REQUIRE(config.get_operator_params().scan_task_batch_size == 1ULL * 1024 * 1024);

  config.load_from_file(config_fixture("effective_operator_defaults_high_level.yaml"));
  constexpr uint64_t capped_default = (256ULL * 1024 * 1024) / 40;
  require_shared_operator_defaults(config.get_operator_params(), capped_default);

  config.load_from_file(config_fixture("effective_operator_defaults_uncapped.yaml"));
  require_shared_operator_defaults(config.get_operator_params(), physical_default);
}

TEST_CASE("effective-capacity defaults seed DuckDB SET and RESET",
          "[sirius][context][config][operator_defaults][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  auto const cfg = config_fixture("effective_operator_defaults_high_level.yaml");
  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);
  constexpr uint64_t expected_batch = (256ULL * 1024 * 1024) / 40;

  auto settings = con.Query(R"(
    SELECT current_setting('scan_task_batch_size')::UBIGINT,
           current_setting('hash_partition_bytes')::UBIGINT,
           current_setting('concat_batch_bytes')::UBIGINT,
           current_setting('sort_sample_bytes')::UBIGINT,
           current_setting('max_build_hash_table_bytes')::UBIGINT
  )");
  REQUIRE(settings != nullptr);
  REQUIRE_FALSE(settings->HasError());
  REQUIRE(settings->GetValue(0, 0).GetValue<uint64_t>() == expected_batch);
  REQUIRE(settings->GetValue(1, 0).GetValue<uint64_t>() == expected_batch);
  REQUIRE(settings->GetValue(2, 0).GetValue<uint64_t>() == expected_batch);
  REQUIRE(settings->GetValue(3, 0).GetValue<uint64_t>() == expected_batch);
  REQUIRE(settings->GetValue(4, 0).GetValue<uint64_t>() == 2 * expected_batch);

  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  require_shared_operator_defaults(sirius_ctx->get_config().get_operator_params(), expected_batch);

  auto set = con.Query("SET scan_task_batch_size = 99");
  REQUIRE(set != nullptr);
  REQUIRE_FALSE(set->HasError());
  auto set_readback = con.Query("SELECT current_setting('scan_task_batch_size')::UBIGINT");
  REQUIRE(set_readback != nullptr);
  REQUIRE_FALSE(set_readback->HasError());
  REQUIRE(set_readback->GetValue(0, 0).GetValue<uint64_t>() == 99);
  REQUIRE(sirius_ctx->get_config().get_operator_params().scan_task_batch_size == 99);

  auto reset_setting = con.Query("RESET scan_task_batch_size");
  REQUIRE(reset_setting != nullptr);
  REQUIRE_FALSE(reset_setting->HasError());
  auto reset = con.Query("SELECT current_setting('scan_task_batch_size')::UBIGINT");
  REQUIRE(reset != nullptr);
  REQUIRE_FALSE(reset->HasError());
  REQUIRE(reset->GetValue(0, 0).GetValue<uint64_t>() == expected_batch);
  REQUIRE(sirius_ctx->get_config().get_operator_params().scan_task_batch_size == expected_batch);
}

TEST_CASE("Sirius configuration rejects invalid downgrade hysteresis", "[sirius][config]")
{
  struct invalid_config {
    const char* fixture;
    const char* scope;
    const char* constraint;
  };

  const invalid_config cases[] = {
    {"invalid_memory_gpu_downgrade_zero.yaml", "sirius.memory.gpu", "greater than zero"},
    {"invalid_memory_gpu_downgrade_equal.yaml",
     "sirius.memory.gpu",
     "less than downgrade_trigger_fraction"},
    {"invalid_memory_gpu_downgrade_reversed.yaml",
     "sirius.memory.gpu",
     "less than downgrade_trigger_fraction"},
    {"invalid_memory_host_downgrade_zero.yaml", "sirius.memory.host", "greater than zero"},
    {"invalid_memory_host_downgrade_equal.yaml",
     "sirius.memory.host",
     "less than downgrade_trigger_fraction"},
    {"invalid_memory_host_downgrade_reversed.yaml",
     "sirius.memory.host",
     "less than downgrade_trigger_fraction"},
    {"invalid_space_gpu_downgrade_zero.yaml", "sirius.space.gpu", "greater than zero"},
    {"invalid_space_gpu_downgrade_equal.yaml",
     "sirius.space.gpu",
     "less than downgrade_trigger_fraction"},
    {"invalid_space_gpu_downgrade_reversed.yaml",
     "sirius.space.gpu",
     "less than downgrade_trigger_fraction"},
    {"invalid_space_host_downgrade_zero.yaml", "sirius.space.host", "greater than zero"},
    {"invalid_space_host_downgrade_equal.yaml",
     "sirius.space.host",
     "less than downgrade_trigger_fraction"},
    {"invalid_space_host_downgrade_reversed.yaml",
     "sirius.space.host",
     "less than downgrade_trigger_fraction"},
  };

  for (auto const& invalid : cases) {
    INFO("fixture=" << invalid.fixture << " surface=" << invalid.scope
                    << " constraint=" << invalid.constraint);
    std::source_location loc = std::source_location::current();
    auto const path          = fs::path(loc.file_name()).parent_path() / "data" / invalid.fixture;
    REQUIRE(fs::is_regular_file(path));

    sirius::sirius_config config;
    REQUIRE_THROWS_WITH(config.load_from_file(path),
                        Catch::Contains(invalid.scope) &&
                          Catch::Contains("downgrade_stop_fraction") &&
                          Catch::Contains(invalid.constraint));
  }
}

TEST_CASE("Sirius downgrade hysteresis accepts omitted, null, and one-sided defaults",
          "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  auto const data_dir      = fs::path(loc.file_name()).parent_path() / "data";
  const char* fixtures[]   = {"valid_memory_downgrade_omitted.yaml",
                              "valid_memory_downgrade_partial_defaults.yaml",
                              "valid_space_downgrade_omitted.yaml",
                              "valid_space_downgrade_partial_defaults.yaml"};

  for (auto const* fixture : fixtures) {
    INFO("fixture=" << fixture);
    auto const path = data_dir / fixture;
    REQUIRE(fs::is_regular_file(path));

    sirius::sirius_config config;
    REQUIRE_NOTHROW(config.load_from_file(path));
  }
}

TEST_CASE("Sirius downgrade monitor period rejects negatives and preserves zero disable",
          "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  auto const data_dir      = fs::path(loc.file_name()).parent_path() / "data";

  for (auto const* fixture : {"invalid_downgrade_monitor_period_negative.yaml",
                              "invalid_downgrade_monitor_period_negative_suffix.yaml",
                              "invalid_downgrade_monitor_period_negative_submillisecond.yaml",
                              "invalid_downgrade_monitor_period_negative_fractional.yaml"}) {
    INFO("fixture=" << fixture);
    auto const path = data_dir / fixture;
    REQUIRE(fs::is_regular_file(path));

    sirius::sirius_config config;
    REQUIRE_THROWS_WITH(
      config.load_from_file(path),
      Catch::Contains("downgrade.monitor_period") && Catch::Contains("non-negative"));
  }

  struct valid_config {
    const char* fixture;
    std::chrono::milliseconds expected;
  };
  for (auto const& valid : {
         valid_config{"valid_downgrade_monitor_period_zero.yaml", std::chrono::milliseconds{0}},
         valid_config{"valid_downgrade_monitor_period_positive.yaml",
                      std::chrono::milliseconds{25}},
       }) {
    INFO("fixture=" << valid.fixture);
    auto const path = data_dir / valid.fixture;
    REQUIRE(fs::is_regular_file(path));

    sirius::sirius_config config;
    REQUIRE_NOTHROW(config.load_from_file(path));
    REQUIRE(config.get_downgrade_executor_config().monitor_period == valid.expected);
  }
}

TEST_CASE("Sirius configuration rejects conflicting memory budget forms", "[sirius][config]")
{
  struct invalid_config {
    const char* filename;
    const char* context;
    const char* first;
    const char* second;
  };
  const invalid_config cases[] = {
    {"invalid_memory_gpu_usage_limit_both.yaml",
     "memory.gpu",
     "usage_limit_bytes",
     "usage_limit_fraction"},
    {"invalid_memory_gpu_reservation_limit_both.yaml",
     "memory.gpu",
     "reservation_limit_bytes",
     "reservation_limit_fraction"},
    {"invalid_memory_host_capacity_both.yaml",
     "memory.host",
     "capacity_bytes",
     "capacity_fraction"},
    {"invalid_memory_host_reservation_limit_both.yaml",
     "memory.host",
     "reservation_limit_bytes",
     "reservation_limit_fraction"},
  };

  std::source_location loc = std::source_location::current();
  auto const data_dir      = fs::path(loc.file_name()).parent_path() / "data";
  for (auto const& test : cases) {
    INFO(test.filename);
    sirius::sirius_config config;
    REQUIRE_THROWS_WITH(config.load_from_file(data_dir / test.filename),
                        Catch::Contains(test.context) && Catch::Contains(test.first) &&
                          Catch::Contains(test.second) && Catch::Contains("mutually exclusive"));
  }
}

TEST_CASE("Sirius configuration accepts one form of each memory budget", "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  auto const cfg =
    fs::path(loc.file_name()).parent_path() / "data" / "valid_memory_budget_single_forms.yaml";

  sirius::sirius_config config;
  REQUIRE_NOTHROW(config.load_from_file(cfg));
}

TEST_CASE("Sirius configuration treats null memory budget forms as absent", "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  auto const cfg =
    fs::path(loc.file_name()).parent_path() / "data" / "valid_memory_budget_null_alternates.yaml";

  sirius::sirius_config config;
  REQUIRE_NOTHROW(config.load_from_file(cfg));
  require_shared_operator_defaults(config.get_operator_params(), expected_effective_batch(config));
}

TEST_CASE("DuckDB setting rejects zero hash partition bytes without a Sirius context",
          "[sirius][context][config][isolated_context]")
{
  finally cleanup_env{[]() { setenv("SIRIUS_DISABLE", "1", 1); }};
  setenv("SIRIUS_DISABLE", "1", 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto result = con.Query("SET hash_partition_bytes = 0");
  REQUIRE(result != nullptr);
  REQUIRE(result->HasError());
  REQUIRE_THAT(result->GetError(),
               Catch::Contains("hash_partition_bytes must be greater than zero"));
}

TEST_CASE("YAML-backed operator and compression settings are DuckDB defaults",
          "[sirius][context][config][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg = fs::path(loc.file_name()).parent_path() / "data" / "setting_defaults.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto sirius_ctx = con.context->registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);

  auto settings = con.Query(R"(
    SELECT
      current_setting('scan_task_batch_size')::UBIGINT,
      current_setting('max_sort_partition_bytes')::UBIGINT,
      current_setting('max_sort_partition_memory_fraction')::DOUBLE,
      current_setting('hash_partition_bytes')::UBIGINT,
      current_setting('concat_batch_bytes')::UBIGINT,
      current_setting('sort_sample_bytes')::UBIGINT,
      current_setting('max_build_hash_table_bytes')::UBIGINT,
      current_setting('max_broadcast_join_size')::UBIGINT,
      current_setting('mark_join_build_switch_ratio')::DOUBLE,
      current_setting('enable_dynamic_filter_pushdown')::BOOLEAN,
      current_setting('enable_dynamic_zone_map_filter')::BOOLEAN,
      current_setting('dynamic_filter_domain_coverage_threshold')::DOUBLE,
      current_setting('dynamic_filter_keep_threshold')::DOUBLE,
      current_setting('enable_pinned_zone_map_pruning')::BOOLEAN,
      current_setting('enable_compressed_materialization')::BOOLEAN,
      current_setting('pin_table_compression')::BOOLEAN,
      current_setting('pin_table_input_compression_plan_dir')::VARCHAR,
      current_setting('pin_table_compression_min_batch_size_bytes')::UBIGINT,
      current_setting('pin_table_compression_max_compressed_fraction')::DOUBLE,
      current_setting('enable_runtime_distinct_build_probe')::BOOLEAN
  )");
  REQUIRE(settings != nullptr);
  REQUIRE_FALSE(settings->HasError());

  constexpr uint64_t mib = 1024ULL * 1024;
  REQUIRE(settings->GetValue(0, 0).GetValue<uint64_t>() == 1 * mib);
  REQUIRE(settings->GetValue(1, 0).GetValue<uint64_t>() == 2 * mib);
  REQUIRE(settings->GetValue(2, 0).GetValue<double>() == Approx(0.25));
  REQUIRE(settings->GetValue(3, 0).GetValue<uint64_t>() == 3 * mib);
  REQUIRE(settings->GetValue(4, 0).GetValue<uint64_t>() == 4 * mib);
  REQUIRE(settings->GetValue(5, 0).GetValue<uint64_t>() == 5 * mib);
  REQUIRE(settings->GetValue(6, 0).GetValue<uint64_t>() == 6 * mib);
  REQUIRE(settings->GetValue(7, 0).GetValue<uint64_t>() == 7 * mib);
  REQUIRE(settings->GetValue(8, 0).GetValue<double>() == Approx(3.0));
  REQUIRE_FALSE(settings->GetValue(9, 0).GetValue<bool>());
  REQUIRE(settings->GetValue(10, 0).GetValue<bool>());
  REQUIRE(settings->GetValue(11, 0).GetValue<double>() == Approx(0.8));
  REQUIRE(settings->GetValue(12, 0).GetValue<double>() == Approx(0.7));
  REQUIRE_FALSE(settings->GetValue(13, 0).GetValue<bool>());
  REQUIRE_FALSE(settings->GetValue(14, 0).GetValue<bool>());
  REQUIRE(settings->GetValue(15, 0).GetValue<bool>());
  REQUIRE(settings->GetValue(16, 0).GetValue<std::string>() == "/tmp/sirius-compression-plans");
  REQUIRE(settings->GetValue(17, 0).GetValue<uint64_t>() == 8 * mib);
  REQUIRE(settings->GetValue(18, 0).GetValue<double>() == Approx(0.6));
  REQUIRE_FALSE(settings->GetValue(19, 0).GetValue<bool>());

  auto zero_partition = con.Query("SET hash_partition_bytes = 0");
  REQUIRE(zero_partition != nullptr);
  REQUIRE(zero_partition->HasError());
  REQUIRE(sirius_ctx->get_config().get_operator_params().hash_partition_bytes == 3 * mib);

  auto const require_ok = [&con](std::string const& sql) {
    auto result = con.Query(sql);
    REQUIRE(result != nullptr);
    REQUIRE_FALSE(result->HasError());
  };

  require_ok("SET scan_task_batch_size = 99");
  require_ok("RESET scan_task_batch_size");
  require_ok("SET max_sort_partition_memory_fraction = 0.9");
  require_ok("RESET max_sort_partition_memory_fraction");
  require_ok("SET enable_dynamic_filter_pushdown = true");
  require_ok("RESET enable_dynamic_filter_pushdown");
  require_ok("SET enable_runtime_distinct_build_probe = true");
  require_ok("RESET enable_runtime_distinct_build_probe");
  require_ok("SET pin_table_compression = false");
  require_ok("RESET pin_table_compression");
  require_ok("SET pin_table_compression_max_compressed_fraction = 0.9");
  require_ok("RESET pin_table_compression_max_compressed_fraction");

  auto reset = con.Query(R"(
    SELECT
      current_setting('scan_task_batch_size')::UBIGINT,
      current_setting('max_sort_partition_memory_fraction')::DOUBLE,
      current_setting('enable_dynamic_filter_pushdown')::BOOLEAN,
      current_setting('pin_table_compression')::BOOLEAN,
      current_setting('pin_table_compression_max_compressed_fraction')::DOUBLE,
      current_setting('enable_runtime_distinct_build_probe')::BOOLEAN
  )");
  REQUIRE(reset != nullptr);
  REQUIRE_FALSE(reset->HasError());
  REQUIRE(reset->GetValue(0, 0).GetValue<uint64_t>() == 1 * mib);
  REQUIRE(reset->GetValue(1, 0).GetValue<double>() == Approx(0.25));
  REQUIRE_FALSE(reset->GetValue(2, 0).GetValue<bool>());
  REQUIRE(reset->GetValue(3, 0).GetValue<bool>());
  REQUIRE(reset->GetValue(4, 0).GetValue<double>() == Approx(0.6));
  REQUIRE_FALSE(reset->GetValue(5, 0).GetValue<bool>());

  auto const& params = sirius_ctx->get_config().get_operator_params();
  REQUIRE(params.scan_task_batch_size == 1 * mib);
  REQUIRE(params.max_sort_partition_memory_fraction == Approx(0.25));
  REQUIRE_FALSE(params.enable_dynamic_filter_pushdown);
  REQUIRE_FALSE(params.enable_runtime_distinct_build_probe);
  auto const& compression = sirius_ctx->get_config().get_compression_config();
  REQUIRE(compression.enable_pin_table_compression);
  REQUIRE(compression.max_compressed_fraction == Approx(0.6));
}

TEST_CASE("Sirius configuration loading from file with spaces",
          "[sirius][context][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "spaces.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");

  REQUIRE(sirius_ctx != nullptr);

  auto& manager = sirius_ctx->get_memory_manager();
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::GPU).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).size() == 1);
  REQUIRE(manager.get_memory_spaces_for_tier(cucascade::memory::Tier::DISK).size() == 2);
  REQUIRE(manager.get_all_memory_spaces().size() == 4);
}

TEST_CASE("Sirius configuration rejects competing memory configuration paths", "[sirius][config]")
{
  std::source_location loc = std::source_location::current();
  auto const data_dir      = fs::path(loc.file_name()).parent_path() / "data";

  const char* fixtures[] = {"invalid_memory_and_space.yaml",
                            "invalid_memory_cross_tier_and_space.yaml",
                            "invalid_memory_empty_subblock_and_space.yaml",
                            "invalid_memory_null_leaf_and_space.yaml"};

  for (auto const* fixture : fixtures) {
    INFO("fixture=" << fixture);
    sirius::sirius_config config;
    REQUIRE_THROWS_WITH(config.load_from_file(data_dir / fixture),
                        Catch::Contains("sirius.memory") && Catch::Contains("sirius.space") &&
                          Catch::Contains("mutually exclusive"));
  }
}

TEST_CASE("Sirius configuration keeps absent memory paths out of mutual-exclusion checks",
          "[sirius][config]")
{
  std::source_location loc  = std::source_location::current();
  auto const data_dir       = fs::path(loc.file_name()).parent_path() / "data";
  constexpr std::size_t mib = 1024ULL * 1024;

  for (auto const& [fixture, expected_capacity] : std::vector<std::pair<const char*, std::size_t>>{
         {"valid_space_with_null_memory_subblock.yaml", 128 * mib},
         {"valid_memory_with_empty_space.yaml", 256 * mib}}) {
    INFO("fixture=" << fixture);
    sirius::sirius_config config;
    REQUIRE_NOTHROW(config.load_from_file(data_dir / fixture));

    auto const& spaces = config.get_memory_space_configs();
    auto const gpu     = std::ranges::find_if(spaces, [](auto const& space) {
      return std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space);
    });
    REQUIRE(gpu != spaces.end());
    REQUIRE(std::get<cucascade::memory::gpu_memory_space_config>(*gpu).memory_capacity ==
            expected_capacity);
  }
}

// ============================================================================
// Multi-GPU Foundation Validation Tests
// ============================================================================
//
// Device Guard Audit — GPU thread entry points and their cudaSetDevice usage:
//
// 1. gpu_pipeline_executor::get_per_thread_init()
//    File: src/pipeline/gpu_pipeline_executor.cpp
//    Guard: cudaSetDevice(device_id) — called per worker thread before any GPU work.
//    Also: manager_loop() uses rmm::cuda_set_device_raii for the manager thread.
//    Status: VERIFIED — device_id obtained from memory_space->get_device_id().
//
// 2. downgrade_executor::get_per_thread_init()
//    File: src/downgrade/downgrade_executor.cpp
//    Guard: cudaSetDevice(device_id) — called per worker thread.
//    Note: Returns nullptr if _memory_space is null (host-only downgrade).
//    Status: VERIFIED — device_id obtained from memory_space->get_device_id().
//
// 3. sirius_memory_reservation_manager constructor
//    File: src/memory/sirius_memory_reservation_manager.cpp (via parent class)
//    Guard: rmm::cuda_set_device_raii — used during GPU memory resource creation
//           inside cucascade::memory::memory_reservation_manager constructor.
//    Status: VERIFIED — each GPU space is created with the correct device context.
//
// 4. duckdb_scan_executor
//    File: src/op/scan/duckdb_scan_executor.cpp
//    Role: Host-side DuckDB scanning. GPU upload goes through converters which
//          are dispatched in the pipeline executor's GPU context.
//    Status: N/A — no direct GPU operations; data upload handled by converters.
//
// 5. Legacy CUDA wrappers (src/cuda/cudf/*.cu)
//    Role: Only called from gpu_processing (legacy) path, never from gpu_execution.
//    Status: N/A for Super Sirius (new path).
//
// Summary: All GPU thread entry points in the Super Sirius (gpu_execution) path
// correctly set the CUDA device before performing GPU operations. The device_id
// is derived from the memory_space associated with each executor, ensuring
// multi-GPU correctness when multiple executors target different devices.
// ============================================================================

TEST_CASE("topology_discovery populates GPU info", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());

  auto const& topology = discovery.get_topology();
  REQUIRE(topology.num_gpus >= 1);
  REQUIRE(topology.gpus.size() == topology.num_gpus);

  for (unsigned int i = 0; i < topology.num_gpus; ++i) {
    auto const& gpu = topology.gpus[i];
    REQUIRE(gpu.id == i);
    REQUIRE(gpu.numa_node >= -1);
    REQUIRE_FALSE(gpu.name.empty());
  }
}

TEST_CASE("reservation_manager_configurator builds N GPU spaces", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  cucascade::memory::topology_discovery discovery;
  REQUIRE(discovery.discover());
  auto const& topology = discovery.get_topology();

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(topology.num_gpus).use_numa_id_as_host_id();
  auto configs = builder.build(topology);

  // Count GPU and HOST tier configs (memory_space_config is a variant post-bump f47de0b)
  size_t gpu_count  = 0;
  size_t host_count = 0;
  for (auto const& cfg : configs) {
    if (std::holds_alternative<cucascade::memory::gpu_memory_space_config>(cfg)) { ++gpu_count; }
    if (std::holds_alternative<cucascade::memory::host_memory_space_config>(cfg)) { ++host_count; }
  }

  REQUIRE(gpu_count == topology.num_gpus);
  REQUIRE(host_count >= 1);
}

TEST_CASE("memory_manager creates independent spaces per GPU", "[multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  REQUIRE(device_count >= 1);

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;  // 512 MB for test
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;  // 1 GB

  builder.set_number_of_gpus(static_cast<size_t>(device_count))
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();

  auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == static_cast<size_t>(device_count));

  // Verify each GPU space has a unique device_id
  std::set<int> device_ids;
  for (auto const* space : gpu_spaces) {
    REQUIRE(space != nullptr);
    device_ids.insert(space->get_device_id());
  }
  REQUIRE(device_ids.size() == static_cast<size_t>(device_count));

  auto host_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  REQUIRE(host_spaces.size() >= 1);

  manager->shutdown();
  sirius::converter_registry::shutdown();
}

TEST_CASE("converter_registry has gpu_to_gpu converter (MEM-03)", "[multi_gpu_foundation]")
{
  sirius::converter_registry::reset_for_testing();
  sirius::converter_registry::initialize();

  auto& registry = sirius::converter_registry::get();

  // Validate that GPU-to-GPU converter is registered (prerequisite for MEM-03 cross-GPU transfer)
  bool has_gpu_to_gpu =
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::gpu_table_representation>();
  REQUIRE(has_gpu_to_gpu);

  sirius::converter_registry::shutdown();
}

// MGPU-04 registration gate: cucascade::register_builtin_converters (called
// by sirius::converter_registry::initialize()) registers a GPU->GPU
// peer-async converter at cucascade/src/data/representation_converter.cpp:1464.
// This test is the grep-verifiable gate that proves the registration
// survives SiriusContext init. See 06-RESEARCH.md Finding 2 for why we do
// not register a second converter here.
TEST_CASE("converter_registry exposes gpu_to_gpu converter after initialize() (MGPU-04)",
          "[multi_gpu_foundation][mgpu_04_registration]")
{
  sirius::converter_registry::reset_for_testing();
  sirius::converter_registry::initialize();

  auto& registry = sirius::converter_registry::get();

  bool has_gpu_to_gpu =
    registry
      .has_converter<cucascade::gpu_table_representation, cucascade::gpu_table_representation>();
  REQUIRE(has_gpu_to_gpu);

  sirius::converter_registry::shutdown();
}

TEST_CASE("multi_gpu_config_two_gpus", "[.][multi_gpu_foundation]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs");
    return;
  }

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 512ull << 20;
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;

  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_gpu_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();

  auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);

  REQUIRE(gpu_spaces[0]->get_device_id() == 0);
  REQUIRE(gpu_spaces[1]->get_device_id() == 1);
  REQUIRE(gpu_spaces[0]->get_device_id() != gpu_spaces[1]->get_device_id());

  manager->shutdown();
  sirius::converter_registry::shutdown();
}

// MGPU-04 + MGPU-06 full round-trip: GPU0 -> GPU1 -> GPU0 data conversion
// using the built-in cucascade peer-async converter. Phase 7 Plan 07-01's
// peer-access enable loop at SiriusContext::initialize() closes the
// Phase-4-deferred GPU1 -> GPU0 return-leg bug, so this test now exercises
// both legs and gates correctness with an FNV-1a checksum over the batch
// payload (silent-corruption guard per Pitfall 2 in
// .planning/phases/07-*/07-RESEARCH.md — Ada Lovelace + Sapphire Rapids).
// WARN+return on single-GPU hosts (Catch2 v2 skip idiom).
TEST_CASE("gpu_to_gpu round-trip preserves bytes on N>=2 hosts (MGPU-04 + MGPU-06)",
          "[multi_gpu_foundation][mgpu_04_round_trip]")
{
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  if (device_count < 2) {
    WARN("skipping: requires >=2 GPUs for MGPU-04 round-trip");
    return;
  }

  sirius::converter_registry::reset_for_testing();

  cucascade::memory::reservation_manager_configurator builder;
  const size_t gpu_capacity  = 256ull << 20;
  const double limit_ratio   = 0.75;
  const size_t host_capacity = 1ull << 30;

  builder.set_number_of_gpus(2)
    .set_gpu_usage_limit(gpu_capacity)
    .set_reservation_fraction_per_gpu(limit_ratio)
    .set_per_numa_region_capacity(host_capacity)
    .use_numa_id_as_host_id()
    .set_reservation_fraction_per_numa_region(limit_ratio);

  auto space_configs = builder.build();
  auto manager =
    std::make_unique<sirius::memory::sirius_memory_reservation_manager>(std::move(space_configs));

  sirius::converter_registry::initialize();
  auto& registry = sirius::converter_registry::get();

  auto gpu_spaces = manager->get_memory_spaces_for_tier(cucascade::memory::Tier::GPU);
  REQUIRE(gpu_spaces.size() == 2);
  auto* gpu0 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[0]);
  auto* gpu1 = const_cast<cucascade::memory::memory_space*>(gpu_spaces[1]);
  REQUIRE(gpu0->get_device_id() == 0);
  REQUIRE(gpu1->get_device_id() == 1);

  // Enable CUDA driver-level peer access for every GPU pair — Plan 07-01's
  // enable loop normally runs inside SiriusContext::initialize(), but this
  // TEST_CASE builds a bare memory manager and bypasses that seam. Without
  // the enable call cucascade's peer-async convert_gpu_to_gpu triggers
  // cudaErrorIllegalAddress on the GPU1 -> GPU0 return leg (MGPU-06 bug).
  enable_p2p_for_test(2);

  // Build a minimal GPU-resident batch on gpu0. The make_gpu_batch helper in
  // test_downgrade_executor.cpp lives in an anonymous namespace and is not
  // reachable from this TU; replicate its body inline using the
  // sirius::create_cudf_table_with_random_data + sirius::make_data_batch
  // primitives (same helpers the downgrade test uses).
  auto build_stream                                      = cudf::get_default_stream();
  auto mr                                                = gpu0->get_default_allocator();
  std::vector<cudf::data_type> col_types                 = {cudf::data_type{cudf::type_id::INT32}};
  std::vector<std::optional<std::pair<int, int>>> ranges = {std::make_pair(0, 100000)};
  auto table = sirius::create_cudf_table_with_random_data(
    /*num_rows=*/1024, col_types, ranges, build_stream, mr);
  auto batch = sirius::make_data_batch(
    std::move(table), *gpu0, build_stream, sirius::telemetry::batch_telemetry_info{});
  REQUIRE(batch != nullptr);
  size_t original_bytes = 0;
  {
    auto __ro_1    = batch->to_read_only();
    original_bytes = __ro_1.get_data()->get_size_in_bytes();
  }
  REQUIRE(original_bytes > 0);
  {
    auto __ro_2 = batch->to_read_only();
    REQUIRE(__ro_2.get_memory_space()->get_device_id() == 0);
  }

  rmm::cuda_stream stream;

  // MGPU-06 Pitfall 2 data integrity guard — Ada Lovelace + Sapphire Rapids
  // silent PCIe P2P write-ordering corruption. Capture the FNV-1a checksum
  // over the batch payload BEFORE any cross-GPU transfer.
  auto checksum_pre = compute_batch_checksum_fnv1a64(*batch, stream.view());

  // GPU0 -> GPU1 forward leg.
  // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor.
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu1, stream.view());
  }

  {
    auto __ro_3 = batch->to_read_only();
    REQUIRE(__ro_3.get_memory_space()->get_device_id() == 1);
    REQUIRE(__ro_3.get_data()->get_size_in_bytes() == original_bytes);
  }

  // MGPU-06 return leg: Phase 7 Plan 07-01's peer-access enable loop at
  // SiriusContext::initialize() closes the Phase-4-deferred GPU1 -> GPU0
  // bug. Checksum integrity guard per RESEARCH.md Pitfall 2 (silent data
  // corruption on Ada Lovelace + Sapphire Rapids).
  // Phase 18 / DB-03 Recipe R8 + R3: scoped mutable accessor.
  {
    auto mut = batch->to_mutable();
    mut.convert_to<cucascade::gpu_table_representation>(registry, gpu0, stream.view());
  }

  {
    auto __ro_4 = batch->to_read_only();
    REQUIRE(__ro_4.get_memory_space()->get_tier() == cucascade::memory::Tier::GPU);
    REQUIRE(__ro_4.get_memory_space()->get_device_id() == gpu0->get_device_id());
    REQUIRE(__ro_4.get_data()->get_size_in_bytes() == original_bytes);
  }

  // Final data-integrity gate: post-round-trip checksum must equal
  // pre-round-trip checksum. Failure here on an Ada Lovelace + Intel Xeon
  // Sapphire Rapids (or later) host = silent data corruption; see Pitfall 2
  // in .planning/phases/07-*/07-RESEARCH.md for the NVIDIA-documented
  // mitigation (disable P2P on affected platforms, or use Hopper/Blackwell).
  auto checksum_post = compute_batch_checksum_fnv1a64(*batch, stream.view());
  INFO("MGPU-04 + MGPU-06 round-trip checksum: pre=" << checksum_pre << " post=" << checksum_post);
  REQUIRE(checksum_post == checksum_pre);

  manager->shutdown();
  sirius::converter_registry::shutdown();
}

namespace {
// Execute one statement and report success (used by the guard-isolation test).
bool run_ok(duckdb::Connection& con, std::string const& sql)
{
  auto result = con.Query(sql);
  return result != nullptr && !result->HasError();
}
}  // namespace

// Successor of "Internal query guard preserves transparent execution state":
// the capture lives on per-connection state with a planning generation, so
// the properties to pin are (a) an internal query's lifecycle on ANOTHER
// connection cannot disturb this connection's capture, and (b) a new planning
// attempt on THIS connection structurally invalidates a leftover capture.
// (The old per-query disabled_optimizers save/restore machinery is gone — the
// mask is published once at extension load — so its assertions are retired.)
TEST_CASE("Per-connection state isolates and expires the transparent capture",
          "[sirius][context][isolated_context]")
{
  finally cleanup_env{[]() {
    unsetenv("SIRIUS_CONFIG_FILE");
    setenv("SIRIUS_DISABLE", "1", 1);
  }};

  std::source_location loc = std::source_location::current();
  fs::path cfg             = fs::path(loc.file_name()).parent_path() / "data" / "configurator.yaml";

  unsetenv("SIRIUS_DISABLE");
  setenv("SIRIUS_CONFIG_FILE", cfg.string().c_str(), 1);

  duckdb::DuckDB db(nullptr);
  duckdb::Connection con(db);

  auto& client_ctx = *con.context;
  auto sirius_ctx  = client_ctx.registered_state->Get<duckdb::SiriusContext>("sirius_state");
  REQUIRE(sirius_ctx != nullptr);
  auto conn_state = duckdb::get_sirius_connection_state(client_ctx);
  REQUIRE(conn_state != nullptr);

  // (a) InternalQueryGuard isolation, verified through the REAL entry points
  // and in BOTH directions: while guard(B) is alive, B's queries must not be
  // taken over by transparent execution — but an UNRELATED connection A must
  // proceed completely unaffected (the old SiriusContext-wide depth would
  // suppress A too, so an instance-global regression fails this test). After
  // the guard is gone, B's identical query IS taken over — proving the guard
  // (and not some other condition) caused the suppression.
  {
    duckdb::Connection internal_con(db);  // "B", the guarded connection
    REQUIRE(run_ok(internal_con, "SET gpu_execution=true;"));
    REQUIRE(run_ok(con, "SET gpu_execution=true;"));  // "A", unrelated
    auto const before = sirius_ctx->get_transparent_execution_stats();
    {
      duckdb::SiriusContext::InternalQueryGuard guard(*internal_con.context);
      // B guarded: suppressed.
      REQUIRE(run_ok(internal_con, "SELECT 42;"));
      auto const guarded = sirius_ctx->get_transparent_execution_stats();
      REQUIRE(guarded.successful_rebinds == before.successful_rebinds);
      REQUIRE(guarded.executions == before.executions);

      // A while guard(B) is alive: exactly +1 — the guard does NOT leak to
      // other connections.
      REQUIRE(run_ok(con, "SELECT 42;"));
      auto const cross = sirius_ctx->get_transparent_execution_stats();
      REQUIRE(cross.successful_rebinds == before.successful_rebinds + 1);
      REQUIRE(cross.executions == before.executions + 1);

      // B again, still guarded: still suppressed.
      REQUIRE(run_ok(internal_con, "SELECT 42;"));
      auto const still_guarded = sirius_ctx->get_transparent_execution_stats();
      REQUIRE(still_guarded.successful_rebinds == before.successful_rebinds + 1);
      REQUIRE(still_guarded.executions == before.executions + 1);
    }
    // Guard released: B proceeds normally (+1 control).
    REQUIRE(run_ok(internal_con, "SELECT 42;"));
    auto const unguarded = sirius_ctx->get_transparent_execution_stats();
    REQUIRE(unguarded.successful_rebinds == before.successful_rebinds + 2);
  }

  // (b) Planning-generation expiry, driven by the REAL Prepare path (not by
  // calling the generation helper directly): ExtractPlan leaves a capture; a
  // plain Prepare of the SAME SQL with the optimizer disabled bumps the
  // generation via DuckDB's CanRequestRebind loop WITHOUT producing a new
  // capture (the optimizer hooks never run). Oracle: an implementation that
  // WRONGLY consumes the stale capture records a successful rebind (+1) and
  // also nulls the slot, so "capture is null" alone is ambiguous — the
  // zero-rebind delta across the Prepare is the discriminating assertion.
  auto plan = con.ExtractPlan("SELECT 42;");
  REQUIRE(plan != nullptr);
  conn_state->set_captured_plan(std::move(plan));

  auto const before_prepare      = sirius_ctx->get_transparent_execution_stats();
  auto& client_config            = duckdb::ClientConfig::GetConfig(client_ctx);
  client_config.enable_optimizer = false;
  auto prepared                  = con.Prepare("SELECT 42;");  // SAME SQL as the capture
  client_config.enable_optimizer = true;
  REQUIRE_FALSE(prepared->HasError());
  auto const after_prepare = sirius_ctx->get_transparent_execution_stats();

  REQUIRE(after_prepare.successful_rebinds == before_prepare.successful_rebinds);
  REQUIRE(conn_state->take_captured_plan_if_current() == nullptr);
}
