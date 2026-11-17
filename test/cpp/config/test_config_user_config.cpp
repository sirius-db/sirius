// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/user_config.hpp"
#include "config/yaml.hpp"
#include "sirius_config.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

using namespace sirius::config;
using namespace sirius::config::size_literals;

namespace {

std::filesystem::path data_dir()
{
  return std::filesystem::path(std::source_location::current().file_name()).parent_path() / "data";
}

std::string slurp(const std::filesystem::path& path)
{
  std::ifstream in(path);
  REQUIRE(in);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

std::string read_error(std::string_view text)
{
  try {
    (void)read_user_config(text);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("the user configuration declares every key once", "[config][config_user]")
{
  user_config{}.check_declaration();
}

TEST_CASE("an empty configuration takes every fallback", "[config][config_user]")
{
  auto const config = default_user_config();
  REQUIRE(config.max_concurrent_queries.get() == 1);
  REQUIRE_FALSE(config.memory.configured());
  REQUIRE_FALSE(config.space.configured());
  REQUIRE(config.telemetry.enable_quent.get() == sirius::telemetry_config::default_enable_quent);
  REQUIRE(config.compression.spill_min_batch_bytes.get() == 64_MiB);
  REQUIRE(config.executor.scan_manager.rest.tls.ciphersuites.has_value());
  REQUIRE(read_user_config("sirius: {}\n").max_concurrent_queries.is_fallback());
}

TEST_CASE("sirius.memory and sirius.space lists are exclusive", "[config][config_user]")
{
  REQUIRE(read_error("sirius:\n  memory:\n    gpu: {}\n  space:\n    gpu: [{}]\n") ==
          "'sirius': sirius.memory and non-empty sirius.space lists are mutually exclusive");
  // Empty lists and a null memory block do not count.
  (void)read_user_config("sirius:\n  memory:\n    gpu: {}\n  space:\n    gpu: []\n");
  (void)read_user_config("sirius:\n  memory: ~\n  space:\n    gpu: [{}]\n");
}

TEST_CASE("a full dump reads back, also with space lists", "[config][config_user]")
{
  for (std::string_view text : {"sirius: {}\n",
                                "sirius:\n  space:\n    gpu:\n      - device_id: 0\n",
                                "sirius:\n  memory:\n    host:\n      pool_size: 64\n"}) {
    INFO(text);
    auto const config = read_user_config(text);
    REQUIRE(parse_yaml(write_user_config(config)) == parse_yaml(text));
    auto const all = read_user_config(write_user_config(config, true));
    REQUIRE(all.space.configured() == config.space.configured());
    REQUIRE(write_user_config(all, true) == write_user_config(config, true));
  }
}

TEST_CASE("telemetry and compression options", "[config][config_user]")
{
  REQUIRE(read_error("sirius:\n  telemetry:\n    output_directory: ''\n") ==
          "'sirius.telemetry.output_directory': value out of range: '' (must not be empty)");
  REQUIRE(
    read_error("sirius:\n  compression:\n    max_compressed_fraction: .inf\n") ==
    "'sirius.compression.max_compressed_fraction': value out of range: '.inf' (must be finite "
    "and not negative)");
  REQUIRE(read_error("sirius:\n  compression:\n    spill_encode_reserve_fraction: 1.5\n") ==
          "'sirius.compression.spill_encode_reserve_fraction': value out of range: '1.5' (must be "
          "between 0.0 and 1.0)");
  auto const config = read_user_config(
    "sirius:\n  compression:\n    max_compressed_fraction: 1.5\n    min_batch_size_bytes: 0\n");
  REQUIRE(config.compression.max_compressed_fraction.get() == 1.5);  // above 1 may expand
  REQUIRE(config.compression.min_batch_size_bytes.get() == 0_B);
  REQUIRE(config.compression.enable_pin_table_compression.scope() == option_scope::query);
  REQUIRE(config.compression.enable_spill_compression.scope() == option_scope::session);
}

TEST_CASE("fixture verdicts follow their names", "[config][config_user]")
{
  // invalid_* fixtures are rejected, every other one reads.
  std::size_t rejected = 0, accepted = 0;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    if (entry.path().extension() != ".yaml") { continue; }
    auto const name    = entry.path().filename().string();
    bool const invalid = name.starts_with("invalid_");
    bool ok            = true;
    try {
      (void)read_user_config(slurp(entry.path()));
    } catch (const config_error&) {
      ok = false;
    }
    INFO(name);
    REQUIRE(ok == !invalid);
    ++(ok ? accepted : rejected);
  }
  REQUIRE(rejected >= 40);
  REQUIRE(accepted >= 30);
}

TEST_CASE("max_concurrent_queries", "[config][config_user]")
{
  REQUIRE(read_user_config("sirius:\n  max_concurrent_queries: 4\n").max_concurrent_queries.get() ==
          4);
  REQUIRE(read_error("sirius:\n  max_concurrent_queries: 0\n") ==
          "'sirius.max_concurrent_queries': value out of range: '0' (must be greater than zero)");
  REQUIRE_THAT(read_error("sirius:\n  max_concurrent_queries: -1\n"),
               Catch::Matchers::ContainsSubstring("value out of range"));
  REQUIRE(read_error("sirius:\n  max_concurrent_queries: many\n") ==
          "'sirius.max_concurrent_queries': invalid integer value: 'many'");
  REQUIRE(default_user_config().max_concurrent_queries.scope() == option_scope::session);
}

TEST_CASE("which memory and space combinations conflict", "[config][config_user]")
{
  (void)read_user_config("sirius:\n  memory: {}\n  space:\n    gpu: [{}]\n");
  (void)read_user_config("sirius:\n  memory:\n    gpu: ~\n  space:\n    gpu: [{}]\n");
  for (std::string_view text : {"sirius:\n  memory:\n    host: {}\n  space:\n    host: [{}]\n",
                                "sirius:\n  memory:\n    disk: {}\n  space:\n    disk: [{}]\n",
                                "sirius:\n  memory:\n    gpu: {}\n  space:\n    gpu: [~]\n"}) {
    INFO(text);
    REQUIRE_THAT(read_error(text), Catch::Matchers::EndsWith("are mutually exclusive"));
  }
}

TEST_CASE("memory set in code counts as configured", "[config][config_user]")
{
  auto config = read_user_config("sirius:\n  space:\n    gpu: [{}]\n");
  REQUIRE_FALSE(config.memory.configured());
  config.memory.gpu.usage_limit.assign(0.5);
  REQUIRE(config.memory.configured());
  REQUIRE_THROWS_AS(config.validate(user_config_root), config_error);
  // The user's value is written, not dropped.
  REQUIRE(parse_yaml(write_user_config(config)).find("sirius")->find("memory") != nullptr);
}

TEST_CASE("an empty memory block next to space lists survives a dump", "[config][config_user]")
{
  constexpr std::string_view text =
    "sirius:\n  memory: {}\n  space:\n    gpu:\n      - device_id: 0\n";
  auto const config = read_user_config(text);
  REQUIRE(parse_yaml(write_user_config(config)) == parse_yaml(text));
  auto const all = read_user_config(write_user_config(config, true));
  REQUIRE(all.memory.user_defined());
  REQUIRE_FALSE(all.memory.configured());

  config_block const& erased = config;
  REQUIRE(erased.write(true).find("memory")->entries().empty());
}

TEST_CASE("a failed read leaves the configuration as it was", "[config][config_user]")
{
  auto config = read_user_config("sirius:\n  max_concurrent_queries: 3\n");
  REQUIRE_THROWS_AS(
    read_yaml(
      config, "sirius:\n  max_concurrent_queries: 0\n", user_config_root, environment::prod),
    config_error);
  REQUIRE(config.max_concurrent_queries.get() == 3);
}

TEST_CASE("telemetry and compression rows", "[config][config_user]")
{
  REQUIRE(read_error("sirius:\n  telemetry:\n    engine_name: ''\n") ==
          "'sirius.telemetry.engine_name': value out of range: '' (must not be empty)");
  for (auto const* entry : {"spill_max_compressed_fraction: .nan",
                            "spill_max_compressed_fraction: -0.1",
                            "min_batch_size_bytes: -1"}) {
    INFO(entry);
    (void)read_error(std::string("sirius:\n  compression:\n    ") + entry + "\n");
  }
  auto const defaults = default_user_config();
  REQUIRE_FALSE(defaults.telemetry.enable_nvtx.get());
  REQUIRE(defaults.telemetry.nvtx_injection_lib.get().empty());
  auto const off = read_user_config("sirius:\n  telemetry:\n    enable_quent: false\n");
  REQUIRE_FALSE(read_user_config(write_user_config(off)).telemetry.enable_quent.get());
}

TEST_CASE("a dump outside developer mode leaves developer-only settings out",
          "[config][config_user]")
{
  auto const config = read_user_config(R"(sirius:
  max_concurrent_queries: 2
  space:
    gpu: [{device_id: 0}]
  executor:
    scan_manager:
      uring_n_reactors: 2
      rest:
        queue_depth: 512
      uring:
        slices_per_pass: 8
      preparation:
        lookahead_batches: 3
  operator_params:
    hash_partition_bytes: 64MiB
    scan_task_batch_size: 64MiB
)");
  auto const prod   = parse_yaml(write_user_config(config, write_options{true, false}));
  auto const& root  = *prod.find("sirius");
  REQUIRE(root.find("space") == nullptr);
  auto const& sm = *root.find("executor")->find("scan_manager");
  REQUIRE(sm.find("uring_n_reactors") != nullptr);
  REQUIRE(sm.find("rest") == nullptr);
  REQUIRE(sm.find("uring") == nullptr);
  REQUIRE(sm.find("preparation") == nullptr);
  REQUIRE(sm.find("cache") != nullptr);  // product settings keep their fallbacks
  auto const& op = *root.find("operator_params");
  REQUIRE(op.find("hash_partition_bytes") != nullptr);
  REQUIRE(op.find("scan_task_batch_size") == nullptr);
  REQUIRE(op.find("enable_dynamic_filter") == nullptr);
  REQUIRE(root.find("max_concurrent_queries")->text() == "2");

  // Developer mode, and every in-process use that needs a faithful copy, keeps them.
  auto const all = parse_yaml(write_user_config(config, true));
  REQUIRE(all.find("sirius")->find("space") != nullptr);
  REQUIRE(all.find("sirius")->find("operator_params")->find("scan_task_batch_size") != nullptr);
}
