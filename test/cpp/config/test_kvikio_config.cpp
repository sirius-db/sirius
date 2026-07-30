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

#include "catch.hpp"
#include "sirius_config.hpp"

#include <cucascade/io/kvikio/config.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

void write_yaml(std::filesystem::path const& path, std::string const& text)
{
  std::ofstream out(path);
  out << text;
  REQUIRE(out);
}

/// Wraps the kvikio block in the surrounding scan_manager scaffolding.
std::string kvikio_yaml(std::string const& body)
{
  return "sirius:\n"
         "  executor:\n"
         "    scan_manager:\n"
         "      kvikio:\n" +
         body;
}

struct temp_yaml {
  std::filesystem::path path;
  explicit temp_yaml(std::string const& name, std::string const& text)
    : path(std::filesystem::temp_directory_path() / name)
  {
    write_yaml(path, text);
  }
  ~temp_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

}  // namespace

TEST_CASE("kvikio_config defaults leave every knob unset", "[scan_manager][config][kvikio]")
{
  // Unset is meaningful: it means "do not touch kvikIO's env-var-seeded
  // default", which is not the same as writing kvikIO's default value back.
  cucascade::io::kvikio_config const cfg{};
  CHECK_FALSE(cfg.nthreads.has_value());
  CHECK_FALSE(cfg.task_size.has_value());
  CHECK_FALSE(cfg.gds_threshold.has_value());
  CHECK_FALSE(cfg.bounce_buffer_size.has_value());
  CHECK_FALSE(cfg.auto_direct_io_read.has_value());
  CHECK_FALSE(cfg.auto_direct_io_read_overread.has_value());
  CHECK_FALSE(cfg.thread_pool_per_block_device.has_value());
  CHECK_FALSE(cfg.compat_mode.has_value());
}

TEST_CASE("sirius_config leaves the kvikio block unset when absent",
          "[scan_manager][config][kvikio]")
{
  temp_yaml yaml{"sirius_kvikio_absent.yaml",
                 "sirius:\n"
                 "  executor:\n"
                 "    scan_manager:\n"
                 "      uring_n_reactors: 1\n"};

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  auto const& kv = cfg.get_scan_manager_config().kvikio;
  CHECK_FALSE(kv.nthreads.has_value());
  CHECK_FALSE(kv.compat_mode.has_value());
}

TEST_CASE("sirius_config parses every kvikio knob", "[scan_manager][config][kvikio]")
{
  temp_yaml yaml{"sirius_kvikio_full.yaml",
                 kvikio_yaml("        nthreads: 8\n"
                             "        task_size: 8MiB\n"
                             "        gds_threshold: 512KiB\n"
                             "        bounce_buffer_size: 32MiB\n"
                             "        auto_direct_io_read: true\n"
                             "        auto_direct_io_read_overread: true\n"
                             "        thread_pool_per_block_device: true\n"
                             "        compat_mode: off\n")};

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  auto const& kv = cfg.get_scan_manager_config().kvikio;

  REQUIRE(kv.nthreads.has_value());
  CHECK(*kv.nthreads == 8U);
  REQUIRE(kv.task_size.has_value());
  CHECK(*kv.task_size == 8UL * 1024 * 1024);
  REQUIRE(kv.gds_threshold.has_value());
  CHECK(*kv.gds_threshold == 512UL * 1024);
  REQUIRE(kv.bounce_buffer_size.has_value());
  CHECK(*kv.bounce_buffer_size == 32UL * 1024 * 1024);
  REQUIRE(kv.auto_direct_io_read.has_value());
  CHECK(*kv.auto_direct_io_read);
  REQUIRE(kv.auto_direct_io_read_overread.has_value());
  CHECK(*kv.auto_direct_io_read_overread);
  REQUIRE(kv.thread_pool_per_block_device.has_value());
  CHECK(*kv.thread_pool_per_block_device);
  REQUIRE(kv.compat_mode.has_value());
  CHECK(*kv.compat_mode == kvikio::CompatMode::OFF);
}

TEST_CASE("sirius_config maps every kvikio compat_mode spelling", "[scan_manager][config][kvikio]")
{
  auto parse_mode = [](std::string const& spelling) {
    temp_yaml yaml{"sirius_kvikio_mode_" + spelling + ".yaml",
                   kvikio_yaml("        compat_mode: " + spelling + "\n")};
    sirius::sirius_config cfg;
    REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
    return cfg.get_scan_manager_config().kvikio.compat_mode;
  };

  // Note "off"/"on" are YAML 1.1 booleans; they reach the reader as strings
  // only because the field is read as a std::string.
  CHECK(parse_mode("off") == kvikio::CompatMode::OFF);
  CHECK(parse_mode("on") == kvikio::CompatMode::ON);
  CHECK(parse_mode("auto") == kvikio::CompatMode::AUTO);
}

TEST_CASE("sirius_config rejects invalid kvikio values", "[scan_manager][config][kvikio]")
{
  SECTION("unknown compat_mode")
  {
    temp_yaml yaml{"sirius_kvikio_bad_mode.yaml", kvikio_yaml("        compat_mode: sometimes\n")};
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  // kvikIO's apply_kvikio_defaults throws on these; rejecting at parse time
  // names the offending key instead of failing later at ioctx construction.
  SECTION("zero nthreads")
  {
    temp_yaml yaml{"sirius_kvikio_zero_threads.yaml", kvikio_yaml("        nthreads: 0\n")};
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  SECTION("zero task_size")
  {
    temp_yaml yaml{"sirius_kvikio_zero_task.yaml", kvikio_yaml("        task_size: 0\n")};
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  SECTION("zero bounce_buffer_size")
  {
    temp_yaml yaml{"sirius_kvikio_zero_bounce.yaml",
                   kvikio_yaml("        bounce_buffer_size: 0\n")};
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  // gds_threshold=0 is legal (it means "always try GDS").
  SECTION("zero gds_threshold is accepted")
  {
    temp_yaml yaml{"sirius_kvikio_zero_gds.yaml", kvikio_yaml("        gds_threshold: 0\n")};
    sirius::sirius_config cfg;
    REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
    REQUIRE(cfg.get_scan_manager_config().kvikio.gds_threshold.has_value());
    CHECK(*cfg.get_scan_manager_config().kvikio.gds_threshold == 0);
  }
}

TEST_CASE("sirius_config rejects unknown kvikio keys", "[scan_manager][config][kvikio]")
{
  temp_yaml yaml{"sirius_kvikio_unknown.yaml", kvikio_yaml("        nthreads_typo: 4\n")};
  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}
