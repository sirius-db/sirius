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

TEST_CASE("every fixture reads or fails as it did with the old reader", "[config][config_user]")
{
  std::vector<std::filesystem::path> files;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    if (entry.path().extension() == ".yaml") { files.push_back(entry.path()); }
  }
  std::ranges::sort(files);
  REQUIRE(files.size() > 50);

  std::map<std::string, std::string> differences;
  for (auto const& path : files) {
    std::string old_error, new_error;
    bool old_ok = true, new_ok = true;
    try {
      (void)sirius::parsed_sirius_config::from_file(path);
    } catch (const std::exception& e) {
      old_ok    = false;
      old_error = e.what();
    }
    try {
      (void)read_user_config(slurp(path));
    } catch (const config_error& e) {
      new_ok    = false;
      new_error = e.what();
    }
    if (old_ok != new_ok) {
      differences[path.filename().string()] =
        (old_ok ? std::string("old accepted") : "old rejected (" + old_error + ")") + "; " +
        (new_ok ? std::string("new accepted") : "new rejected (" + new_error + ")");
    }
  }
  for (auto const& [file, difference] : differences) {
    UNSCOPED_INFO(file << ": " << difference);
  }
  REQUIRE(differences.empty());
}
