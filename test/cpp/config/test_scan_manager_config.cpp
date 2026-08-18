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
#include "scan_manager/config.hpp"
#include "sirius_config.hpp"

#include <filesystem>
#include <fstream>
#include <string>

using sirius::scan_manager::cache_mode;
using sirius::scan_manager::enum_to_string;
using sirius::scan_manager::io_backend;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::string_to_enum;

namespace {

class scoped_yaml {
 public:
  explicit scoped_yaml(std::string const& name, std::string const& text)
    : path_(std::filesystem::temp_directory_path() / name)
  {
    std::ofstream out(path_);
    out << text;
    REQUIRE(out);
  }

  ~scoped_yaml()
  {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  scoped_yaml(scoped_yaml const&)            = delete;
  scoped_yaml& operator=(scoped_yaml const&) = delete;

  [[nodiscard]] std::filesystem::path const& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

scan_manager_config load_scan_manager(std::string const& name, std::string const& text)
{
  scoped_yaml yaml(name, text);
  sirius::sirius_config cfg;
  cfg.load_from_file(yaml.path());
  return cfg.get_scan_manager_config();
}

std::string scan_manager_yaml(std::string const& body)
{
  return "sirius:\n"
         "  executor:\n"
         "    scan_manager:\n" +
         body;
}

std::string single_gpu_scan_manager_yaml(std::string const& body)
{
  return "sirius:\n"
         "  topology:\n"
         "    num_gpus: 1\n"
         "  executor:\n"
         "    scan_manager:\n" +
         body;
}

}  // namespace

TEST_CASE("cache_mode string_to_enum accepts known modes", "[scan_manager][config][cache_mode]")
{
  cache_mode mode = cache_mode::prefetch;

  REQUIRE(string_to_enum("none", mode));
  CHECK(mode == cache_mode::none);

  REQUIRE(string_to_enum("os", mode));
  CHECK(mode == cache_mode::os);

  REQUIRE(string_to_enum("persistent", mode));
  CHECK(mode == cache_mode::persistent);

  REQUIRE(string_to_enum("prefetch", mode));
  CHECK(mode == cache_mode::prefetch);
}

TEST_CASE("cache_mode string_to_enum rejects unknown modes", "[scan_manager][config][cache_mode]")
{
  cache_mode mode = cache_mode::persistent;

  CHECK_FALSE(string_to_enum("", mode));
  CHECK_FALSE(string_to_enum("PERSISTENT", mode));
  CHECK_FALSE(string_to_enum("odirect", mode));
  CHECK(mode == cache_mode::persistent);
}

TEST_CASE("cache_mode enum_to_string returns canonical names", "[scan_manager][config][cache_mode]")
{
  std::string out;

  REQUIRE(enum_to_string(cache_mode::none, out));
  CHECK(out == "none");
  REQUIRE(enum_to_string(cache_mode::os, out));
  CHECK(out == "os");
  REQUIRE(enum_to_string(cache_mode::persistent, out));
  CHECK(out == "persistent");
  REQUIRE(enum_to_string(cache_mode::prefetch, out));
  CHECK(out == "prefetch");
}

TEST_CASE("scan_manager_config defaults to the none cache mode",
          "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};

  CHECK(cfg.cache == cache_mode::none);
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.enable_prefetch_cache);
  CHECK_FALSE(cfg.prefetch_cache.dispose_on_idle);
}

TEST_CASE("apply_cache_mode derives the knobs for every mode", "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};

  cfg.cache = cache_mode::none;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.enable_prefetch_cache);

  cfg.cache = cache_mode::os;
  cfg.apply_cache_mode();
  CHECK_FALSE(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.enable_prefetch_cache);

  cfg.cache = cache_mode::persistent;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.enable_prefetch_cache);
  CHECK_FALSE(cfg.prefetch_cache.dispose_on_idle);

  cfg.cache = cache_mode::prefetch;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.enable_prefetch_cache);
  CHECK(cfg.prefetch_cache.dispose_on_idle);
}

TEST_CASE("apply_cache_mode overwrites hand-set derived knobs",
          "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};
  cfg.uring.use_odirect              = false;
  cfg.enable_prefetch_cache          = true;
  cfg.prefetch_cache.dispose_on_idle = true;

  cfg.cache = cache_mode::none;
  cfg.apply_cache_mode();

  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.enable_prefetch_cache);
}

TEST_CASE("sirius_config derives scan_manager knobs from the cache mode",
          "[scan_manager][config][cache_mode]")
{
  auto const os =
    load_scan_manager("sirius_cache_mode_os.yaml", scan_manager_yaml("      cache: os\n"));
  CHECK(os.cache == cache_mode::os);
  CHECK_FALSE(os.uring.use_odirect);
  CHECK_FALSE(os.enable_prefetch_cache);

  auto const persistent = load_scan_manager("sirius_cache_mode_persistent.yaml",
                                            scan_manager_yaml("      cache: persistent\n"));
  CHECK(persistent.cache == cache_mode::persistent);
  CHECK(persistent.uring.use_odirect);
  CHECK(persistent.enable_prefetch_cache);
  CHECK_FALSE(persistent.prefetch_cache.dispose_on_idle);

  auto const prefetch = load_scan_manager("sirius_cache_mode_prefetch.yaml",
                                          scan_manager_yaml("      cache: prefetch\n"));
  CHECK(prefetch.cache == cache_mode::prefetch);
  CHECK(prefetch.uring.use_odirect);
  CHECK(prefetch.enable_prefetch_cache);
  CHECK(prefetch.prefetch_cache.dispose_on_idle);
}

TEST_CASE("sirius_config defaults the cache mode to none when YAML omits it",
          "[scan_manager][config][cache_mode]")
{
  auto const cfg = load_scan_manager("sirius_cache_mode_default.yaml",
                                     scan_manager_yaml("      uring_n_reactors: 2\n"));

  CHECK(cfg.cache == cache_mode::none);
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.enable_prefetch_cache);
}

TEST_CASE("sirius_config reads the prefetch_cache tunables", "[scan_manager][config][cache_mode]")
{
  auto const cfg =
    load_scan_manager("sirius_prefetch_cache_tunables.yaml",
                      scan_manager_yaml("      cache: prefetch\n"
                                        "      prefetch_cache:\n"
                                        "        eviction_threshold_fraction: 0.25\n"
                                        "        min_prefetching_budget_fraction: 0.5\n"));

  CHECK(cfg.prefetch_cache.eviction_threshold_fraction == Approx(0.25));
  CHECK(cfg.prefetch_cache.min_prefetching_budget_fraction == Approx(0.5));
  CHECK(cfg.prefetch_cache.dispose_on_idle);
}

TEST_CASE("sirius_config rejects an unknown cache mode", "[scan_manager][config][cache_mode]")
{
  scoped_yaml yaml("sirius_cache_mode_invalid.yaml",
                   scan_manager_yaml("      cache: persistent_host\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config rejects the knobs superseded by the cache mode",
          "[scan_manager][config][cache_mode]")
{
  auto rejects = [](std::string const& name, std::string const& body) {
    scoped_yaml yaml(name, scan_manager_yaml(body));
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path()));
  };

  rejects("sirius_cache_mode_enable_prefetch.yaml", "      enable_prefetch_cache: true\n");
  rejects("sirius_cache_mode_odirect.yaml",
          "      uring:\n"
          "        use_odirect: false\n");
  rejects("sirius_cache_mode_dispose.yaml",
          "      prefetch_cache:\n"
          "        dispose_on_idle: true\n");
  rejects("sirius_cache_mode_old_cache_map.yaml",
          "      cache:\n"
          "        eviction_threshold_fraction: 0.25\n");
}

TEST_CASE("backend string_to_enum accepts known backends", "[scan_manager][config][backend]")
{
  io_backend b = io_backend::kvikio;

  REQUIRE(string_to_enum("sirius", b));
  CHECK(b == io_backend::sirius);

  REQUIRE(string_to_enum("kvikio", b));
  CHECK(b == io_backend::kvikio);
}

TEST_CASE("backend string_to_enum rejects unknown backends", "[scan_manager][config][backend]")
{
  io_backend b = io_backend::kvikio;

  CHECK_FALSE(string_to_enum("", b));
  CHECK_FALSE(string_to_enum("SIRIUS", b));
  CHECK_FALSE(string_to_enum("uring", b));
  CHECK_FALSE(string_to_enum("true", b));
  CHECK(b == io_backend::kvikio);
}

TEST_CASE("backend enum_to_string returns canonical names", "[scan_manager][config][backend]")
{
  std::string out;

  REQUIRE(enum_to_string(io_backend::sirius, out));
  CHECK(out == "sirius");
  REQUIRE(enum_to_string(io_backend::kvikio, out));
  CHECK(out == "kvikio");
}

TEST_CASE("scan_manager_config defaults to the sirius backend", "[scan_manager][config][backend]")
{
  scan_manager_config cfg{};

  CHECK(cfg.backend == io_backend::sirius);
}

TEST_CASE("sirius_config reads the backend from YAML", "[scan_manager][config][backend]")
{
  auto const kvikio = load_scan_manager("sirius_backend_kvikio.yaml",
                                        single_gpu_scan_manager_yaml("      backend: kvikio\n"));
  CHECK(kvikio.backend == io_backend::kvikio);

  auto const sirius = load_scan_manager("sirius_backend_sirius.yaml",
                                        single_gpu_scan_manager_yaml("      backend: sirius\n"));
  CHECK(sirius.backend == io_backend::sirius);

  auto const omitted = load_scan_manager("sirius_backend_default.yaml",
                                         scan_manager_yaml("      uring_n_reactors: 2\n"));
  CHECK(omitted.backend == io_backend::sirius);
}

TEST_CASE("sirius_config rejects an invalid backend", "[scan_manager][config][backend]")
{
  scoped_yaml yaml("sirius_backend_invalid.yaml", scan_manager_yaml("      backend: uring\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config rejects the removed use_sirius_datasource key",
          "[scan_manager][config][backend]")
{
  scoped_yaml yaml("sirius_backend_removed_key.yaml",
                   scan_manager_yaml("      use_sirius_datasource: true\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config reads the uring sub-config", "[scan_manager][config][backend]")
{
  auto const cfg = load_scan_manager("sirius_uring_node.yaml",
                                     scan_manager_yaml("      uring:\n"
                                                       "        max_n_chunks: 4\n"));

  CHECK(cfg.uring.max_n_chunks == 4);
}

TEST_CASE("sirius_config rejects the removed REST max_read_split key",
          "[scan_manager][config][backend]")
{
  scoped_yaml yaml("sirius_rest_removed_key.yaml",
                   scan_manager_yaml("      rest:\n"
                                     "        max_read_split: 16\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config rejects the renamed local sub-config", "[scan_manager][config][backend]")
{
  scoped_yaml yaml("sirius_local_node_removed.yaml",
                   scan_manager_yaml("      local:\n"
                                     "        max_n_chunks: 4\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config forces the sirius backend for multi-GPU",
          "[scan_manager][config][backend]")
{
  auto const cfg = load_scan_manager("sirius_backend_multi_gpu.yaml",
                                     "sirius:\n"
                                     "  topology:\n"
                                     "    num_gpus: 2\n"
                                     "  executor:\n"
                                     "    scan_manager:\n"
                                     "      backend: kvikio\n");

  CHECK(cfg.backend == io_backend::sirius);
}
