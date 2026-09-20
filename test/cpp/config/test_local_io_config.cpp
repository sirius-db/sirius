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

#include <filesystem>
#include <fstream>
#include <string>

namespace {

struct scoped_yaml {
  scoped_yaml(std::string const& name, bool use_odirect)
    : path(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream out(path);
    out << "sirius:\n"
           "  executor:\n"
           "    scan_manager:\n"
           "      local:\n"
           "        use_odirect: "
        << (use_odirect ? "true\n" : "false\n");
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

TEST_CASE("sirius_config preserves explicit local use_odirect modes",
          "[scan_manager][config][local]")
{
  for (bool const expected : {false, true}) {
    INFO("use_odirect=" << std::boolalpha << expected);
    scoped_yaml yaml(expected ? "sirius_use_odirect_true.yaml" : "sirius_use_odirect_false.yaml",
                     expected);

    sirius::sirius_config cfg;
    REQUIRE_NOTHROW(cfg.load_from_file(yaml.path));
    CHECK(cfg.get_scan_manager_config().local.use_odirect == expected);
  }
}
