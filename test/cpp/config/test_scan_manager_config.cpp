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

using sirius::io::cache::cache_mode;
using sirius::io::cache::eviction_policy;
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

std::string cache_yaml(std::string const& body)
{
  return "sirius:\n"
         "  executor:\n"
         "    scan_manager:\n"
         "      cache:\n" +
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
  cache_mode mode = cache_mode::sirius;

  REQUIRE(sirius::io::cache::string_to_enum("none", mode));
  CHECK(mode == cache_mode::none);

  REQUIRE(sirius::io::cache::string_to_enum("os", mode));
  CHECK(mode == cache_mode::os);

  REQUIRE(sirius::io::cache::string_to_enum("sirius", mode));
  CHECK(mode == cache_mode::sirius);
}

TEST_CASE("cache_mode string_to_enum rejects unknown modes", "[scan_manager][config][cache_mode]")
{
  cache_mode mode = cache_mode::sirius;

  CHECK_FALSE(sirius::io::cache::string_to_enum("", mode));
  CHECK_FALSE(sirius::io::cache::string_to_enum("SIRIUS", mode));
  CHECK_FALSE(sirius::io::cache::string_to_enum("odirect", mode));
  // The pre-consolidation spellings are gone, not silently remapped.
  CHECK_FALSE(sirius::io::cache::string_to_enum("persistent", mode));
  CHECK_FALSE(sirius::io::cache::string_to_enum("prefetch", mode));
  CHECK(mode == cache_mode::sirius);
}

TEST_CASE("cache_mode enum_to_string returns canonical names", "[scan_manager][config][cache_mode]")
{
  std::string out;

  REQUIRE(sirius::io::cache::enum_to_string(cache_mode::none, out));
  CHECK(out == "none");
  REQUIRE(sirius::io::cache::enum_to_string(cache_mode::os, out));
  CHECK(out == "os");
  REQUIRE(sirius::io::cache::enum_to_string(cache_mode::sirius, out));
  CHECK(out == "sirius");
}

TEST_CASE("eviction_policy round-trips through its YAML spelling",
          "[scan_manager][config][cache_mode]")
{
  eviction_policy policy = eviction_policy::lru;

  REQUIRE(sirius::io::cache::string_to_enum("idle", policy));
  CHECK(policy == eviction_policy::idle);
  REQUIRE(sirius::io::cache::string_to_enum("lru", policy));
  CHECK(policy == eviction_policy::lru);
  CHECK_FALSE(sirius::io::cache::string_to_enum("fifo", policy));
  CHECK(policy == eviction_policy::lru);

  std::string out;
  REQUIRE(sirius::io::cache::enum_to_string(eviction_policy::idle, out));
  CHECK(out == "idle");
  REQUIRE(sirius::io::cache::enum_to_string(eviction_policy::lru, out));
  CHECK(out == "lru");
}

TEST_CASE("scan_manager_config defaults to the none cache mode",
          "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};

  CHECK(cfg.cache.mode == cache_mode::none);
  CHECK(cfg.cache.eviction == eviction_policy::lru);
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.enabled());
  CHECK_FALSE(cfg.cache.use_prefetching_cache());
  CHECK_FALSE(cfg.cache.dispose_on_idle);
}

TEST_CASE("apply_cache_mode derives the knobs for every mode", "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};

  cfg.cache.mode = cache_mode::none;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.enabled());
  CHECK_FALSE(cfg.cache.use_prefetching_cache());

  cfg.cache.mode = cache_mode::os;
  cfg.apply_cache_mode();
  CHECK_FALSE(cfg.uring.use_odirect);
  CHECK(cfg.cache.enabled());
  CHECK_FALSE(cfg.cache.use_prefetching_cache());

  cfg.cache.mode     = cache_mode::sirius;
  cfg.cache.eviction = eviction_policy::lru;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.cache.use_prefetching_cache());
  CHECK_FALSE(cfg.cache.dispose_on_idle);

  cfg.cache.eviction = eviction_policy::idle;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.cache.use_prefetching_cache());
  CHECK(cfg.cache.dispose_on_idle);
}

TEST_CASE("apply_cache_mode overwrites hand-set derived knobs",
          "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};
  cfg.uring.use_odirect     = false;
  cfg.cache.dispose_on_idle = true;

  cfg.cache.mode     = cache_mode::none;
  cfg.cache.eviction = eviction_policy::lru;
  cfg.apply_cache_mode();

  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.dispose_on_idle);
}

TEST_CASE("sirius_config derives scan_manager knobs from the cache block",
          "[scan_manager][config][cache_mode]")
{
  auto const os = load_scan_manager("sirius_cache_mode_os.yaml", cache_yaml("        mode: os\n"));
  CHECK(os.cache.mode == cache_mode::os);
  CHECK_FALSE(os.uring.use_odirect);
  CHECK_FALSE(os.cache.use_prefetching_cache());

  auto const lru = load_scan_manager("sirius_cache_mode_lru.yaml",
                                     cache_yaml("        mode: sirius\n"
                                                "        eviction: lru\n"));
  CHECK(lru.cache.mode == cache_mode::sirius);
  CHECK(lru.uring.use_odirect);
  CHECK(lru.cache.use_prefetching_cache());
  CHECK_FALSE(lru.cache.dispose_on_idle);

  auto const idle = load_scan_manager("sirius_cache_mode_idle.yaml",
                                      cache_yaml("        mode: sirius\n"
                                                 "        eviction: idle\n"));
  CHECK(idle.cache.mode == cache_mode::sirius);
  CHECK(idle.uring.use_odirect);
  CHECK(idle.cache.use_prefetching_cache());
  CHECK(idle.cache.dispose_on_idle);
}

TEST_CASE("sirius_config defaults the cache mode to none when YAML omits it",
          "[scan_manager][config][cache_mode]")
{
  auto const cfg = load_scan_manager("sirius_cache_mode_default.yaml",
                                     scan_manager_yaml("      uring_n_reactors: 2\n"));

  CHECK(cfg.cache.mode == cache_mode::none);
  CHECK(cfg.cache.eviction == eviction_policy::lru);
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.use_prefetching_cache());
}

TEST_CASE("sirius_config reads the cache tunables", "[scan_manager][config][cache_mode]")
{
  auto const cfg = load_scan_manager("sirius_cache_tunables.yaml",
                                     cache_yaml("        mode: sirius\n"
                                                "        eviction: idle\n"
                                                "        eviction_threshold_fraction: 0.25\n"
                                                "        min_prefetching_budget_fraction: 0.5\n"));

  CHECK(cfg.cache.eviction_threshold_fraction == Approx(0.25));
  CHECK(cfg.cache.min_prefetching_budget_fraction == Approx(0.5));
  CHECK(cfg.cache.dispose_on_idle);
}

TEST_CASE("sirius_config rejects an unknown cache mode", "[scan_manager][config][cache_mode]")
{
  scoped_yaml yaml("sirius_cache_mode_invalid.yaml", cache_yaml("        mode: persistent_host\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
}

TEST_CASE("sirius_config rejects the knobs superseded by the cache block",
          "[scan_manager][config][cache_mode]")
{
  auto rejects = [](std::string const& name, std::string const& text) {
    scoped_yaml yaml(name, text);
    sirius::sirius_config cfg;
    CHECK_THROWS(cfg.load_from_file(yaml.path()));
  };

  rejects("sirius_cache_mode_dispose.yaml", cache_yaml("        dispose_on_idle: true\n"));
  rejects("sirius_cache_mode_odirect.yaml",
          scan_manager_yaml("      uring:\n"
                            "        use_odirect: false\n"));
  // The pre-consolidation spellings: `cache` as a bare mode scalar, and a
  // sibling `prefetch_cache` block, both now the `cache` block's business.
  rejects("sirius_cache_mode_scan_manager_cache.yaml", scan_manager_yaml("      cache: sirius\n"));
  rejects("sirius_cache_mode_prefetch_cache.yaml",
          scan_manager_yaml("      prefetch_cache:\n"
                            "        eviction_threshold_fraction: 0.25\n"));
}

TEST_CASE("max_readahead_scans overrides the cache-derived readahead budget",
          "[scan_manager][config][readahead]")
{
  constexpr std::size_t backend_budget = 7;
  constexpr auto backend_strategy      = sirius::scan_manager::prefetch_strategy::eager;

  scan_manager_config cfg{};
  // Unset + no cache: nothing to read ahead into.
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 0);

  // Unset + a cache on: the backend reactor's own depth.
  cfg.cache.mode = cache_mode::sirius;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == backend_budget);

  // Explicitly zero: off, cache or no cache.
  cfg.max_readahead_scans = 0;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 0);
  cfg.cache.mode = cache_mode::none;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 0);

  // Explicitly positive: that count, even with the cache off.
  cfg.max_readahead_scans = 3;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 3);
  cfg.cache.mode = cache_mode::sirius;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 3);
}

TEST_CASE("readahead_strategy defers to the backend until it is set",
          "[scan_manager][config][readahead]")
{
  using sirius::scan_manager::prefetch_strategy;
  constexpr std::size_t backend_budget = 7;

  scan_manager_config cfg{};
  cfg.cache.mode = cache_mode::sirius;

  // Unset: whatever the serving backend prefers, either way.
  REQUIRE_FALSE(cfg.readahead_strategy.has_value());
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).strategy ==
        prefetch_strategy::eager);
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::opportunistic).strategy ==
        prefetch_strategy::opportunistic);

  // Set: that strategy, whatever the backend prefers.
  cfg.readahead_strategy = prefetch_strategy::eager;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::opportunistic).strategy ==
        prefetch_strategy::eager);
  cfg.readahead_strategy = prefetch_strategy::opportunistic;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).strategy ==
        prefetch_strategy::opportunistic);
}

TEST_CASE("an opportunistic readahead schedules against the pipeline width",
          "[scan_manager][config][readahead]")
{
  using sirius::scan_manager::prefetch_strategy;
  constexpr std::size_t backend_budget = 7;

  scan_manager_config cfg{};
  cfg.cache.mode     = cache_mode::sirius;
  cfg.pipeline_width = 2;

  // Opportunistic issues one prefetch per non-scan deployment, so the budget
  // that matters is what the executor can run, not what the device can queue.
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::opportunistic).budget == 2);
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == backend_budget);

  // A pinned strategy drives the same substitution as a backend-preferred one.
  cfg.readahead_strategy = prefetch_strategy::opportunistic;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == 2);

  // An explicit budget wins over the pipeline-width substitution.
  cfg.max_readahead_scans = 5;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == 5);
}

TEST_CASE("sirius_config reads max_readahead_scans", "[scan_manager][config][readahead]")
{
  auto const unset = load_scan_manager("sirius_readahead_unset.yaml",
                                       scan_manager_yaml("      rest_n_reactors: 1\n"));
  CHECK_FALSE(unset.max_readahead_scans.has_value());

  auto const off = load_scan_manager("sirius_readahead_off.yaml",
                                     scan_manager_yaml("      max_readahead_scans: 0\n"));
  REQUIRE(off.max_readahead_scans.has_value());
  CHECK(*off.max_readahead_scans == 0);

  auto const budget = load_scan_manager("sirius_readahead_budget.yaml",
                                        scan_manager_yaml("      max_readahead_scans: 12\n"));
  REQUIRE(budget.max_readahead_scans.has_value());
  CHECK(*budget.max_readahead_scans == 12);
}

TEST_CASE("sirius_config reads readahead_strategy", "[scan_manager][config][readahead]")
{
  auto const unset = load_scan_manager("sirius_readahead_strategy_unset.yaml",
                                       scan_manager_yaml("      rest_n_reactors: 1\n"));
  CHECK_FALSE(unset.readahead_strategy.has_value());

  auto const pinned =
    load_scan_manager("sirius_readahead_strategy_set.yaml",
                      scan_manager_yaml("      readahead_strategy: opportunistic\n"));
  REQUIRE(pinned.readahead_strategy.has_value());
  CHECK(*pinned.readahead_strategy == sirius::scan_manager::prefetch_strategy::opportunistic);

  scoped_yaml invalid("sirius_readahead_strategy_invalid.yaml",
                      scan_manager_yaml("      readahead_strategy: lazy\n"));
  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(invalid.path()));
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

TEST_CASE("sirius_config rejects the removed uring max_n_chunks key",
          "[scan_manager][config][backend]")
{
  scoped_yaml yaml("sirius_uring_removed_key.yaml",
                   scan_manager_yaml("      uring:\n"
                                     "        max_n_chunks: 4\n"));

  sirius::sirius_config cfg;
  CHECK_THROWS(cfg.load_from_file(yaml.path()));
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
