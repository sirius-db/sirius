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
