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
#include "sirius_config.hpp"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

/// Write @p text to @p path and hand back an RAII remover, so a failing REQUIRE inside a
/// test still cleans up the temp file.
struct scoped_yaml {
  explicit scoped_yaml(std::string const& name, std::string const& text)
    : path(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream out(path);
    out << text;
    REQUIRE(out);
  }
  ~scoped_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
  scoped_yaml(scoped_yaml const&)            = delete;
  scoped_yaml& operator=(scoped_yaml const&) = delete;

  std::filesystem::path path;
};

}  // namespace

TEST_CASE("sirius_config parses topology.gpus_per_query", "[topology_config][config]")
{
  scoped_yaml yaml("sirius_gpus_per_query.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: 2\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 2);
}

TEST_CASE("sirius_config defaults gpus_per_query to 0 when absent", "[topology_config][config]")
{
  // 0 means "admit every GPU", so an absent key must not narrow anything.
  scoped_yaml yaml("sirius_gpus_per_query_absent.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    num_gpus: 1\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 0);
}

TEST_CASE("sirius_config accepts an explicit gpus_per_query of 0", "[topology_config][config]")
{
  scoped_yaml yaml("sirius_gpus_per_query_zero.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: 0\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.gpus_per_query() == 0);
}

TEST_CASE("sirius_config rejects a negative gpus_per_query", "[topology_config][config]")
{
  // Admission treats any non-positive value as "no cap", so a negative would silently read
  // as "use every GPU" — the opposite of what someone writing -1 could mean. Reject at load.
  scoped_yaml yaml("sirius_gpus_per_query_negative.yaml",
                   "sirius:\n"
                   "  topology:\n"
                   "    gpus_per_query: -1\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}

TEST_CASE("sirius_config rejects a zero avg_variable_column_bytes", "[topology_config][config]")
{
  // Zero would make variable-width columns contribute nothing to the per-row width, so a
  // mixed schema is under-estimated and the query admitted onto too few GPUs. Unlike
  // admission_bytes_per_gpu, where 0 is the documented off switch, there is no reading of
  // zero here that means anything.
  scoped_yaml yaml("sirius_avg_var_zero.yaml",
                   "sirius:\n"
                   "  operator_params:\n"
                   "    avg_variable_column_bytes: 0\n");

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path));
}

TEST_CASE("sirius_config accepts a zero admission_bytes_per_gpu", "[topology_config][config]")
{
  // 0 is the off switch: it disables the estimate and leaves sizing to gpus_per_query.
  scoped_yaml yaml("sirius_admission_bytes_zero.yaml",
                   "sirius:\n"
                   "  operator_params:\n"
                   "    admission_bytes_per_gpu: 0\n");

  sirius::sirius_config cfg;
  REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
  CHECK(cfg.get_operator_params().admission_bytes_per_gpu == 0);
}

TEST_CASE("sirius_config bounds clustered_bypass_max_overlap_fraction to [0, 1]",
          "[clustered_merge_bypass][config]")
{
  // The knob is a fraction of the smaller batch's key span; values outside [0, 1] have no
  // reading. The endpoints are meaningful: 0 admits only the absolute-floor overlap, 1 admits
  // any adjacent overlap (the disjointedness structure still gates correctness either way).
  auto config_text = [](std::string const& value) {
    return "sirius:\n"
           "  operator_params:\n"
           "    clustered_bypass_max_overlap_fraction: " +
           value + "\n";
  };

  SECTION("rejects a negative fraction")
  {
    scoped_yaml yaml("sirius_bypass_overlap_negative.yaml", config_text("-0.1"));
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  SECTION("rejects a fraction above one")
  {
    scoped_yaml yaml("sirius_bypass_overlap_above_one.yaml", config_text("1.5"));
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path));
  }

  SECTION("accepts the zero endpoint")
  {
    scoped_yaml yaml("sirius_bypass_overlap_zero.yaml", config_text("0"));
    sirius::sirius_config cfg;
    REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
    CHECK(cfg.get_operator_params().clustered_bypass_max_overlap_fraction == 0.0);
  }

  SECTION("accepts the one endpoint")
  {
    scoped_yaml yaml("sirius_bypass_overlap_one.yaml", config_text("1"));
    sirius::sirius_config cfg;
    REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
    CHECK(cfg.get_operator_params().clustered_bypass_max_overlap_fraction == 1.0);
  }
}
