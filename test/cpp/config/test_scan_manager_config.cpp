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
#include "utils/log_test_utils.hpp"
#include "utils/sirius_test_env.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/uring/config.hpp>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

using cucascade::io::cache::cache_mode;
using cucascade::io::cache::eviction_policy;
using sirius::scan_manager::enum_to_string;
using sirius::scan_manager::io_backend;
using sirius::scan_manager::scan_manager_config;
using sirius::scan_manager::string_to_enum;

namespace {

class scoped_yaml {
 public:
  explicit scoped_yaml(std::string const& name, std::string const& text)
    : path_(std::filesystem::temp_directory_path() / (std::to_string(::getpid()) + "_" + name))
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

std::string uring_yaml(std::string const& body)
{
  return scan_manager_yaml("      uring:\n" + body);
}

std::string rest_yaml(std::string const& body) { return scan_manager_yaml("      rest:\n" + body); }

std::string tls_yaml(std::string const& body) { return rest_yaml("        tls:\n" + body); }

/// Loading @p text must fail, and the error must end with @p message (the loader
/// prefixes it with the file it was reading).
void require_load_error(std::string const& name,
                        std::string const& text,
                        std::string const& message)
{
  scoped_yaml yaml(name, text);
  sirius::sirius_config cfg;
  REQUIRE_THROWS_WITH(cfg.load_from_file(yaml.path()), Catch::Matchers::EndsWith(": " + message));
}

}  // namespace

TEST_CASE("Preparation timing YAML overrides survive configuration resolution",
          "[scan_manager][config][preparation_config]")
{
  auto [body, fill, min_ms, max_ms, interrupt_ms] =
    GENERATE(table<std::string, int, int, int, int>({
      {"", 50, 50, 2000, 10},
      {"      preparation: {}\n", 50, 50, 2000, 10},
      {"      preparation:\n        max_inflight_metadata_jobs: 0\n", 50, 50, 2000, 10},
      {"      preparation:\n        max_inflight_dv_jobs: 0\n", 50, 50, 2000, 10},
      {"      preparation:\n        interrupt_check_interval: 3\n", 50, 50, 2000, 3},
      {"      preparation:\n        interrupt_check_interval: 1s\n", 50, 50, 2000, 1000},
      {"      preparation:\n        partial_batch: {}\n", 50, 50, 2000, 10},
      {"      preparation:\n        partial_batch:\n          min_fill_percent: 25\n",
       25,
       50,
       2000,
       10},
      {"      preparation:\n        partial_batch:\n          min_residence: 20\n"
       "          max_residence: 500ms\n",
       50,
       20,
       500,
       10},
      {"      preparation:\n        partial_batch:\n          min_residence: 1s\n"
       "          max_residence: 1s\n",
       50,
       1000,
       1000,
       10},
    }));
  INFO(body);
  scoped_yaml yaml("sirius_preparation_timing.yaml", scan_manager_yaml(body));
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 1;
  topology.num_numa_nodes = 1;
  topology.gpus           = {{.id = 0, .numa_node = 0}};
  topology.numa_nodes     = {{.id = 0, .memory_capacity = 1ULL << 30, .has_cpus = true}};
  auto resolved           = sirius::parsed_sirius_config::from_file(yaml.path()).resolve(topology);
  auto const& config      = resolved.get_scan_manager_config();
  auto options            = config.preparation.resolve(config.thread_pool.num_threads);
  REQUIRE(options.partial_batch);
  CHECK(options.partial_batch->min_fill_percent == static_cast<size_t>(fill));
  CHECK(options.partial_batch->min_residence == std::chrono::milliseconds{min_ms});
  CHECK(options.partial_batch->max_residence == std::chrono::milliseconds{max_ms});
  CHECK(options.interrupt_check_interval == std::chrono::milliseconds{interrupt_ms});
  CHECK(options.max_inflight_metadata_jobs ==
        (body.find("max_inflight_metadata_jobs: 0") != std::string::npos ? 0 : 4096));
  CHECK(options.max_inflight_dv_jobs == 0);
}

TEST_CASE("Preparation timing YAML rejects invalid values and unknown keys",
          "[scan_manager][config][preparation_config]")
{
  auto field = GENERATE("interrupt_check_interval: 0",
                        "interrupt_check_interval: -1ms",
                        "interrupt_check_interval: later",
                        "interrupt_check_interval: []",
                        "interrupt_check_interva: 10ms",
                        "max_inflight_job: 2");
  INFO(field);
  scoped_yaml yaml("sirius_preparation_timing_invalid.yaml",
                   scan_manager_yaml(std::string("      preparation:\n        ") + field + "\n"));
  CHECK_THROWS_AS(sirius::parsed_sirius_config::from_file(yaml.path()),
                  sirius::configuration_input_error);
}

TEST_CASE("Partial batch YAML rejects invalid policies and unknown keys",
          "[scan_manager][config][preparation_config][partial_batch]")
{
  auto field = GENERATE("min_fill_percent: 0",
                        "min_fill_percent: -1",
                        "min_fill_percent: 101",
                        "min_fill_percent: 1.5",
                        "min_fill_percent: []",
                        "min_residence: 0ms",
                        "min_residence: -1ms",
                        "min_residence: later",
                        "max_residence: 0",
                        "max_residence: 49ms",
                        "max_residence: []",
                        "min_residence: 3s",
                        "min_fill: 50");
  INFO(field);
  scoped_yaml yaml(
    "sirius_partial_batch_invalid.yaml",
    scan_manager_yaml(std::string("      preparation:\n        partial_batch:\n          ") +
                      field + "\n"));
  CHECK_THROWS_AS(sirius::parsed_sirius_config::from_file(yaml.path()),
                  sirius::configuration_input_error);
}

TEST_CASE("Preparation limit YAML overrides preserve omitted defaults",
          "[scan_manager][config][preparation_config][native_runtime]")
{
  auto selected = GENERATE(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);
  std::array keys{"max_inflight_metadata_jobs",
                  "max_inflight_dv_jobs",
                  "lookahead_batches",
                  "max_control_work",
                  "drain_quantum",
                  "native_reader_workers",
                  "max_native_readers",
                  "coordinator_workers"};
  std::array<size_t, 8> values{1, 2, 7, 4, 5, 5, 8, 3};
  auto const workers = GENERATE(1, 3);
  std::string body   = "      num_threads: " + std::to_string(workers) + "\n      preparation:\n";
  for (size_t i = 0; i < keys.size(); ++i)
    if (selected == 8 || selected == static_cast<int>(i))
      body += std::string("        ") + keys[i] + ": " + std::to_string(values[i]) + "\n";
  if (selected == 9) body += "        coordinator_workers: 0\n";
  INFO(body);
  scoped_yaml yaml("sirius_preparation_limits.yaml", scan_manager_yaml(body));
  cucascade::memory::system_topology_info topology{};
  topology.num_gpus       = 1;
  topology.num_numa_nodes = 1;
  topology.gpus           = {{.id = 0, .numa_node = 0}};
  topology.numa_nodes     = {{.id = 0, .memory_capacity = 1ULL << 30, .has_cpus = true}};
  auto resolved           = sirius::parsed_sirius_config::from_file(yaml.path()).resolve(topology);
  auto const& config      = resolved.get_scan_manager_config();
  REQUIRE(config.thread_pool.num_threads == workers);
  auto options = config.preparation.resolve(config.thread_pool.num_threads);
  std::array actual{options.max_inflight_metadata_jobs,
                    options.max_inflight_dv_jobs,
                    options.lookahead_batches,
                    options.max_control_work,
                    options.drain_quantum,
                    config.preparation.resolved_native_workers(),
                    config.preparation.resolved_native_readers(),
                    config.preparation.resolved_coordinator_workers()};
  std::array<size_t, 8> defaults{4096, 0, 2, size_t(workers), size_t(workers), 2, 64, 1};
  for (size_t i = 0; i < keys.size(); ++i) {
    INFO(keys[i]);
    auto expected = (selected == 8 || selected == static_cast<int>(i)) ? values[i] : defaults[i];
    if (selected == 9 && i == 7) expected = 0;
    CHECK(actual[i] == expected);
  }
  CHECK(options.partial_batch);
  CHECK(options.interrupt_check_interval == std::chrono::milliseconds{10});
}

TEST_CASE("Preparation limit YAML rejects invalid values",
          "[scan_manager][config][preparation_config]")
{
  auto key   = GENERATE("max_inflight_metadata_jobs",
                      "max_inflight_dv_jobs",
                      "lookahead_batches",
                      "max_control_work",
                      "drain_quantum",
                      "native_reader_workers",
                      "max_native_readers",
                      "coordinator_workers");
  auto value = GENERATE("0", "-1", "1.5", "many", "[]", "18446744073709551616");
  if ((std::string_view(key) == "max_inflight_metadata_jobs" ||
       std::string_view(key) == "max_inflight_dv_jobs" ||
       std::string_view(key) == "coordinator_workers") &&
      std::string_view(value) == "0")
    return;
  INFO(key << ": " << value);
  scoped_yaml yaml(
    "sirius_preparation_limit_invalid.yaml",
    scan_manager_yaml(std::string("      preparation:\n        ") + key + ": " + value + "\n"));
  CHECK_THROWS_AS(sirius::parsed_sirius_config::from_file(yaml.path()),
                  sirius::configuration_input_error);
}

TEST_CASE("Preparation YAML rejects retired keys", "[scan_manager][config][preparation_config]")
{
  auto key = GENERATE("max_active_units", "max_pending_results", "max_inflight_jobs");
  scoped_yaml yaml("sirius_preparation_old_window.yaml",
                   scan_manager_yaml(std::string("      preparation:\n        ") + key + ": 8\n"));
  CHECK_THROWS_AS(sirius::parsed_sirius_config::from_file(yaml.path()),
                  sirius::configuration_input_error);
}

TEST_CASE("cache_mode string_to_enum accepts known modes", "[scan_manager][config][cache_mode]")
{
  cache_mode mode = cache_mode::cucs;

  REQUIRE(cucascade::io::cache::string_to_enum("none", mode));
  CHECK(mode == cache_mode::none);

  REQUIRE(cucascade::io::cache::string_to_enum("os", mode));
  CHECK(mode == cache_mode::os);

  REQUIRE(cucascade::io::cache::string_to_enum("cucs", mode));
  CHECK(mode == cache_mode::cucs);
}

TEST_CASE("cache_mode string_to_enum rejects unknown modes", "[scan_manager][config][cache_mode]")
{
  cache_mode mode = cache_mode::cucs;

  CHECK_FALSE(cucascade::io::cache::string_to_enum("", mode));
  CHECK_FALSE(cucascade::io::cache::string_to_enum("CUCS", mode));
  CHECK_FALSE(cucascade::io::cache::string_to_enum("odirect", mode));
  // The pre-consolidation spellings are gone, not silently remapped.
  CHECK_FALSE(cucascade::io::cache::string_to_enum("persistent", mode));
  CHECK_FALSE(cucascade::io::cache::string_to_enum("prefetch", mode));
  CHECK(mode == cache_mode::cucs);
}

TEST_CASE("cache_mode enum_to_string returns canonical names", "[scan_manager][config][cache_mode]")
{
  std::string out;

  REQUIRE(cucascade::io::cache::enum_to_string(cache_mode::none, out));
  CHECK(out == "none");
  REQUIRE(cucascade::io::cache::enum_to_string(cache_mode::os, out));
  CHECK(out == "os");
  REQUIRE(cucascade::io::cache::enum_to_string(cache_mode::cucs, out));
  CHECK(out == "cucs");
}

TEST_CASE("eviction_policy round-trips through its YAML spelling",
          "[scan_manager][config][cache_mode]")
{
  eviction_policy policy = eviction_policy::lru;

  REQUIRE(cucascade::io::cache::string_to_enum("idle", policy));
  CHECK(policy == eviction_policy::idle);
  REQUIRE(cucascade::io::cache::string_to_enum("lru", policy));
  CHECK(policy == eviction_policy::lru);
  CHECK_FALSE(cucascade::io::cache::string_to_enum("fifo", policy));
  CHECK(policy == eviction_policy::lru);

  std::string out;
  REQUIRE(cucascade::io::cache::enum_to_string(eviction_policy::idle, out));
  CHECK(out == "idle");
  REQUIRE(cucascade::io::cache::enum_to_string(eviction_policy::lru, out));
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
  CHECK_FALSE(cfg.cache.use_fs_cache());
  CHECK_FALSE(cfg.cache.dispose_on_idle);
}

TEST_CASE("apply_cache_mode derives the knobs for every mode", "[scan_manager][config][cache_mode]")
{
  scan_manager_config cfg{};

  cfg.cache.mode = cache_mode::none;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.enabled());
  CHECK_FALSE(cfg.cache.use_fs_cache());

  cfg.cache.mode = cache_mode::os;
  cfg.apply_cache_mode();
  CHECK_FALSE(cfg.uring.use_odirect);
  CHECK(cfg.cache.enabled());
  CHECK_FALSE(cfg.cache.use_fs_cache());

  cfg.cache.mode     = cache_mode::cucs;
  cfg.cache.eviction = eviction_policy::lru;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.cache.use_fs_cache());
  CHECK_FALSE(cfg.cache.dispose_on_idle);

  cfg.cache.eviction = eviction_policy::idle;
  cfg.apply_cache_mode();
  CHECK(cfg.uring.use_odirect);
  CHECK(cfg.cache.use_fs_cache());
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
  CHECK_FALSE(os.cache.use_fs_cache());

  auto const lru = load_scan_manager("sirius_cache_mode_lru.yaml",
                                     cache_yaml("        mode: cucs\n"
                                                "        eviction: lru\n"));
  CHECK(lru.cache.mode == cache_mode::cucs);
  CHECK(lru.uring.use_odirect);
  CHECK(lru.cache.use_fs_cache());
  CHECK_FALSE(lru.cache.dispose_on_idle);

  auto const idle = load_scan_manager("sirius_cache_mode_idle.yaml",
                                      cache_yaml("        mode: cucs\n"
                                                 "        eviction: idle\n"));
  CHECK(idle.cache.mode == cache_mode::cucs);
  CHECK(idle.uring.use_odirect);
  CHECK(idle.cache.use_fs_cache());
  CHECK(idle.cache.dispose_on_idle);
}

TEST_CASE("sirius_config accepts the deprecated sirius cache mode as cucs",
          "[scan_manager][config][cache_mode]")
{
  // `sirius` named the prefetching cache before it moved to cuCascade; existing
  // YAML keeps working and selects the same cache.
  auto const cfg =
    load_scan_manager("sirius_cache_mode_sirius_alias.yaml", cache_yaml("        mode: sirius\n"));
  CHECK(cfg.cache.mode == cache_mode::cucs);
  CHECK(cfg.cache.use_fs_cache());
  CHECK(cfg.uring.use_odirect);
}

TEST_CASE("sirius_config defaults the cache mode to none when YAML omits it",
          "[scan_manager][config][cache_mode]")
{
  auto const cfg = load_scan_manager("sirius_cache_mode_default.yaml",
                                     scan_manager_yaml("      uring_n_reactors: 2\n"));

  CHECK(cfg.cache.mode == cache_mode::none);
  CHECK(cfg.cache.eviction == eviction_policy::lru);
  CHECK(cfg.uring.use_odirect);
  CHECK_FALSE(cfg.cache.use_fs_cache());
}

TEST_CASE("sirius_config reads the cache tunables", "[scan_manager][config][cache_mode]")
{
  auto const cfg = load_scan_manager("sirius_cache_tunables.yaml",
                                     cache_yaml("        mode: cucs\n"
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
  cfg.cache.mode = cache_mode::cucs;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == backend_budget);

  // Explicitly zero: off, cache or no cache.
  cfg.max_readahead_scans = 0;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 0);
  cfg.cache.mode = cache_mode::none;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 0);

  // Explicitly positive: that count, even with the cache off.
  cfg.max_readahead_scans = 3;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 3);
  cfg.cache.mode = cache_mode::cucs;
  CHECK(cfg.resolve_readahead(backend_budget, backend_strategy).budget == 3);
}

TEST_CASE("readahead_strategy defers to the backend until it is set",
          "[scan_manager][config][readahead]")
{
  using sirius::scan_manager::prefetch_strategy;
  constexpr std::size_t backend_budget = 7;

  scan_manager_config cfg{};
  cfg.cache.mode = cache_mode::cucs;

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
  cfg.cache.mode     = cache_mode::cucs;
  cfg.pipeline_width = 2;

  // Opportunistic issues one prefetch per non-scan deployment, so the budget
  // that matters is what the executor can run, not what the device can queue.
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::opportunistic).budget == 2);
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == backend_budget);

  // A pinned strategy drives the same substitution as a backend-preferred one.
  cfg.readahead_strategy = prefetch_strategy::opportunistic;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == 2);

  // A backend budget of zero is an opt-out and wins over the opportunistic
  // pipeline-width substitution: the local (uring) backend defaults to 0.
  cfg.readahead_strategy.reset();
  CHECK(cfg.resolve_readahead(0, prefetch_strategy::opportunistic).budget == 0);
  CHECK(cfg.resolve_readahead(0, prefetch_strategy::eager).budget == 0);

  // An explicit budget wins over the pipeline-width substitution.
  cfg.max_readahead_scans = 5;
  CHECK(cfg.resolve_readahead(backend_budget, prefetch_strategy::eager).budget == 5);
}

TEST_CASE("apply_defaults derives the readahead budgets like the file path does",
          "[scan_manager][config][readahead]")
{
  constexpr auto pipeline_threads =
    static_cast<std::size_t>(std::max(1, sirius::exec::default_gpu_pipeline_num_threads));

  sirius::sirius_config cfg;
  cfg.apply_defaults();
  auto const& scan_manager = cfg.get_scan_manager_config();

  // Having no sirius.yaml must size the readahead exactly like an empty one.
  // The local (uring) backend stays at 0 (readahead off); the rest backend
  // defaults to max(8, 2x the pipeline width).
  CHECK(scan_manager.pipeline_width == pipeline_threads);
  CHECK(scan_manager.uring.n_max_concurrent_scans == 0);
  CHECK_FALSE(scan_manager.uring.n_max_concurrent_scans_explicit);
  CHECK(scan_manager.rest.n_max_concurrent_scans == std::max<std::size_t>(8, 2 * pipeline_threads));
}

TEST_CASE("uring scan budget stays 0 unless set", "[scan_manager][config][readahead]")
{
  constexpr std::size_t pipeline_width = 7;
  constexpr std::size_t other_explicit = 5;

  auto yaml = [=](std::optional<std::size_t> uring_budget) {
    auto text =
      std::string{
        "sirius:\n"
        "  executor:\n"
        "    pipeline:\n"
        "      num_threads: "} +
      std::to_string(pipeline_width) +
      "\n"
      "    scan_manager:\n";
    if (uring_budget.has_value()) {
      text +=
        "      uring:\n"
        "        n_max_concurrent_scans: " +
        std::to_string(*uring_budget) + "\n";
    }
    return text;
  };

  // Omitted: the local backend stays off and is not scaled to the pipeline width.
  auto const omitted = load_scan_manager("sirius_uring_budget_omitted.yaml", yaml(std::nullopt));
  CHECK_FALSE(omitted.uring.n_max_concurrent_scans_explicit);
  CHECK(omitted.uring.n_max_concurrent_scans == 0);

  // An explicit 0 is recorded as explicit and keeps the local readahead off.
  auto const explicit_zero = load_scan_manager("sirius_uring_budget_explicit_zero.yaml", yaml(0));
  CHECK(explicit_zero.uring.n_max_concurrent_scans_explicit);
  CHECK(explicit_zero.uring.n_max_concurrent_scans == 0);

  // An explicit positive value opts the local path in and wins.
  auto const explicit_other =
    load_scan_manager("sirius_uring_budget_explicit_other.yaml", yaml(other_explicit));
  CHECK(explicit_other.uring.n_max_concurrent_scans_explicit);
  CHECK(explicit_other.uring.n_max_concurrent_scans == other_explicit);
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
  io_backend b{};

  REQUIRE(string_to_enum("native", b));
  CHECK(b == io_backend::native);

  REQUIRE(string_to_enum("cufile", b));
  CHECK(b == io_backend::cufile);

  // Deprecated spelling of `native`, still accepted.
  REQUIRE(string_to_enum("sirius", b));
  CHECK(b == io_backend::native);
}

TEST_CASE("backend string_to_enum rejects unknown backends", "[scan_manager][config][backend]")
{
  io_backend b{};

  CHECK_FALSE(string_to_enum("", b));
  CHECK_FALSE(string_to_enum("NATIVE", b));
  CHECK_FALSE(string_to_enum("SIRIUS", b));
  CHECK_FALSE(string_to_enum("uring", b));
  CHECK_FALSE(string_to_enum("kvikio", b));
  CHECK_FALSE(string_to_enum("true", b));
  CHECK(b == io_backend::native);
}

TEST_CASE("backend enum_to_string returns canonical names", "[scan_manager][config][backend]")
{
  std::string out;

  REQUIRE(enum_to_string(io_backend::native, out));
  CHECK(out == "native");
  REQUIRE(enum_to_string(io_backend::cufile, out));
  CHECK(out == "cufile");
}

TEST_CASE("scan_manager_config defaults to the native backend", "[scan_manager][config][backend]")
{
  scan_manager_config cfg{};

  CHECK(cfg.backend == io_backend::native);
  CHECK(cfg.cufile_n_reactors == 4);
  CHECK(cfg.cufile.max_inflight == 32);
  CHECK(cfg.cufile.max_op_bytes == (64UL << 20));
  CHECK(cfg.cufile.gds_threshold == (256UL << 10));
  CHECK(cfg.cufile.use_odirect);
  CHECK_FALSE(cfg.cufile.compat_mode.has_value());
}

TEST_CASE("cuFile C++ config is validated before starting the scan manager",
          "[scan_manager][config][backend][cufile]")
{
  scan_manager_config cfg{};
  SECTION("reactor count") { cfg.cufile_n_reactors = 0; }
  SECTION("inflight window") { cfg.cufile.max_inflight = 0; }
  SECTION("operation bytes") { cfg.cufile.max_op_bytes = 4095; }
  SECTION("direct policy")
  {
    cfg.cufile.use_odirect = false;
    cfg.cufile.compat_mode = cucascade::io::cufile::compat::off;
  }
  CHECK_THROWS_AS(sirius::scan_manager::validate_scan_manager_config(cfg), std::invalid_argument);
}

TEST_CASE("to_io_config carries the scan_manager settings into cuCascade's io_config",
          "[scan_manager][config][backend]")
{
  scan_manager_config cfg{};
  cfg.uring_n_reactors      = 3;
  cfg.rest_n_reactors       = 5;
  cfg.cache.mode            = cache_mode::os;
  cfg.cache.eviction        = eviction_policy::idle;
  cfg.object_store.endpoint = "http://object-store.test:9000";
  cfg.rest.max_connections  = 17;

  auto const io = cfg.to_io_config();
  CHECK(io.backend == cucascade::io::io_backend::native);
  CHECK(io.uring_n_reactors == 3);
  CHECK(io.rest_n_reactors == 5);
  // A non-default value in each copied sub-config survives the copy.
  CHECK(io.object_store.endpoint == "http://object-store.test:9000");
  CHECK(io.rest.max_connections == 17);
  CHECK(io.cache.eviction == eviction_policy::idle);
  // Sirius's uring default (readahead off) reaches cuCascade unchanged.
  CHECK(io.uring.n_max_concurrent_scans == 0);
  CHECK_FALSE(io.uring.n_max_concurrent_scans_explicit);
  CHECK(io.rest.n_max_concurrent_scans == cfg.rest.n_max_concurrent_scans);
  CHECK(io.cache.mode == cache_mode::os);
  // The cache-derived knobs are refreshed even though apply_cache_mode() was not called.
  CHECK_FALSE(io.uring.use_odirect);
  CHECK(io.cache.dispose_on_idle);

  // 0 (size the REST pool from the NIC) is handed to cuCascade as is.
  cfg.rest_n_reactors = 0;
  CHECK(cfg.to_io_config().rest_n_reactors == 0);
  cfg.backend = io_backend::cufile;
  CHECK(cfg.to_io_config().backend == cucascade::io::io_backend::cufile);
}

TEST_CASE("sirius_config reads the backend from YAML", "[scan_manager][config][backend]")
{
  auto const native = load_scan_manager("sirius_backend_native.yaml",
                                        single_gpu_scan_manager_yaml("      backend: native\n"));
  CHECK(native.backend == io_backend::native);

  auto const omitted = load_scan_manager("sirius_backend_default.yaml",
                                         scan_manager_yaml("      uring_n_reactors: 2\n"));
  CHECK(omitted.backend == io_backend::native);
}

TEST_CASE("sirius_config rejects the removed kvikio backend and block",
          "[scan_manager][config][backend]")
{
  require_load_error("sirius_backend_kvikio_removed.yaml",
                     single_gpu_scan_manager_yaml("      backend: kvikio\n"),
                     "'scan_manager.backend': backend: kvikio has been removed; "
                     "use backend: native instead");
  for (auto const* block : {"{}", "{task_size: 8 MiB}"}) {
    require_load_error("sirius_kvikio_block_removed.yaml",
                       scan_manager_yaml(std::string{"      kvikio: "} + block + "\n"),
                       "'scan_manager.kvikio': removed; remove this block and use "
                       "backend: native instead");
  }
}

TEST_CASE("sirius_config reads cuFile read tuning", "[scan_manager][config][backend][cufile]")
{
  auto const cfg =
    load_scan_manager("sirius_backend_cufile.yaml",
                      single_gpu_scan_manager_yaml("      backend: cufile\n"
                                                   "      cufile_n_reactors: 3\n"
                                                   "      cache:\n"
                                                   "        mode: cucs\n"
                                                   "      cufile:\n"
                                                   "        max_inflight: 7\n"
                                                   "        max_op_bytes: 8 MiB\n"
                                                   "        gds_threshold: 128 KiB\n"
                                                   "        use_odirect: false\n"
                                                   "        compat_mode: on\n"
                                                   "        poll_interval_us: 50\n"
                                                   "        low_service_interval: 0\n"));
  CHECK(cfg.backend == io_backend::cufile);
  auto const io = cfg.to_io_config();
  CHECK(io.backend == cucascade::io::io_backend::cufile);
  CHECK(cfg.cufile_n_reactors == 3);
  CHECK(cfg.cufile.max_inflight == 7);
  CHECK(cfg.cufile.max_op_bytes == (8UL << 20));
  CHECK(cfg.cufile.gds_threshold == (128UL << 10));
  CHECK_FALSE(cfg.cufile.use_odirect);
  CHECK(cfg.cufile.compat_mode == cucascade::io::cufile::compat::on);
  CHECK(cfg.cufile.poll_interval_us == 50);
  CHECK(cfg.cufile.low_service_interval == 0);
#ifdef CUCASCADE_HAS_CUFILE
  CHECK(io.cufile_n_reactors == cfg.cufile_n_reactors);
  CHECK(io.cufile.max_inflight == cfg.cufile.max_inflight);
  CHECK(io.cufile.max_op_bytes == cfg.cufile.max_op_bytes);
  CHECK(io.cufile.gds_threshold == cfg.cufile.gds_threshold);
  CHECK(io.cufile.use_odirect == cfg.cufile.use_odirect);
  CHECK(io.cufile.compat_mode == cfg.cufile.compat_mode);
  CHECK(io.cufile.poll_interval_us == cfg.cufile.poll_interval_us);
  CHECK(io.cufile.low_service_interval == cfg.cufile.low_service_interval);
#endif

  for (auto const& [spelling, expected] :
       {std::pair{"auto", cucascade::io::cufile::compat::automatic},
        std::pair{"on", cucascade::io::cufile::compat::on},
        std::pair{"off", cucascade::io::cufile::compat::off}}) {
    auto const parsed =
      load_scan_manager("sirius_cufile_compat.yaml",
                        single_gpu_scan_manager_yaml(
                          std::string{"      cufile:\n        compat_mode: "} + spelling + "\n"));
    CHECK(parsed.cufile.compat_mode == expected);
  }
}

TEST_CASE("sirius_config rejects invalid cuFile settings with key context",
          "[scan_manager][config][backend][cufile]")
{
  for (auto const& [body, key] :
       {std::pair{"      cufile_n_reactors: 0\n", "scan_manager.cufile_n_reactors"},
        std::pair{"      cufile_n_reactors: -1\n", "scan_manager.cufile_n_reactors"},
        std::pair{"      cufile:\n        max_inflight: 0\n", "cufile.max_inflight"},
        std::pair{"      cufile:\n        max_inflight: -1\n", "cufile.max_inflight"},
        std::pair{"      cufile:\n        max_op_bytes: 4095\n", "cufile.max_op_bytes"},
        std::pair{"      cufile:\n        n_max_concurrent_scans: -1\n",
                  "cufile.n_max_concurrent_scans"},
        std::pair{"      cufile:\n        n_max_concurrent_scans: 0\n", "not configurable"},
        std::pair{"      cufile:\n        n_max_concurrent_scans: 4\n", "not configurable"},
        std::pair{"      cufile:\n        gds_threshold: -1\n", "cufile.gds_threshold"},
        std::pair{"      cufile:\n        poll_interval_us: -1\n", "cufile.poll_interval_us"},
        std::pair{"      cufile:\n        low_service_interval: -1\n",
                  "cufile.low_service_interval"},
        std::pair{"      cufile:\n        compat_mode: ON\n", "cufile.compat_mode"},
        std::pair{"      cufile:\n        compat_mode: ''\n", "cufile.compat_mode"},
        std::pair{"      cufile:\n        use_odirect: false\n        compat_mode: off\n",
                  "cufile.use_odirect"},
        std::pair{"      cufile:\n        write_split_bytes: 4096\n", "write_split_bytes"},
        std::pair{"      cufile:\n        register_min_bytes: 4096\n", "register_min_bytes"},
        std::pair{"      io_backend: cufile\n", "io_backend"}}) {
    CAPTURE(body);
    scoped_yaml yaml("sirius_cufile_invalid.yaml", single_gpu_scan_manager_yaml(body));
    sirius::sirius_config cfg;
    REQUIRE_THROWS_WITH(cfg.load_from_file(yaml.path()), Catch::Matchers::ContainsSubstring(key));
  }
}

TEST_CASE("the documented cuFile configuration example loads",
          "[scan_manager][config][backend][cufile]")
{
  std::ifstream doc(SIRIUS_PROJECT_ROOT "/docs/super-sirius/configuration.md");
  REQUIRE(doc.is_open());
  bool in_yaml = false;
  std::string block;
  std::string example;
  size_t examples = 0;
  for (std::string line; std::getline(doc, line);) {
    if (!in_yaml) {
      if (line == "```yaml") {
        in_yaml = true;
        block.clear();
      }
    } else if (line == "```") {
      if (block.find("backend: cufile") != std::string::npos) {
        ++examples;
        example = block;
      }
      in_yaml = false;
    } else {
      block += line + '\n';
    }
  }
  REQUIRE(examples == 1);
  scan_manager_config cfg;
  REQUIRE_NOTHROW(cfg = load_scan_manager("sirius_documented_cufile.yaml", example));
  CHECK(cfg.backend == io_backend::cufile);
  CHECK(cfg.cufile_n_reactors == 4);
  CHECK(cfg.cufile.max_inflight == 32);
  CHECK(cfg.cufile.max_op_bytes == (64UL << 20));
  CHECK(cfg.cufile.gds_threshold == (256UL << 10));
  CHECK(cfg.cufile.use_odirect);
  CHECK(cfg.cufile.compat_mode == cucascade::io::cufile::compat::automatic);
  CHECK(cfg.cufile.poll_interval_us == 200);
  CHECK(cfg.cufile.low_service_interval == 4);
}

TEST_CASE("sirius_config accepts the deprecated sirius backend as native",
          "[scan_manager][config][backend]")
{
  auto const cfg = load_scan_manager("sirius_backend_sirius_alias.yaml",
                                     single_gpu_scan_manager_yaml("      backend: sirius\n"));
  CHECK(cfg.backend == io_backend::native);
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

TEST_CASE("REST and TLS empty YAML blocks preserve cuCascade's transport defaults",
          "[scan_manager][config][rest][tls]")
{
  // Sirius sets no REST transport defaults of its own: an empty block must leave
  // every field at cuCascade's struct default (compared field by field, so a
  // cuCascade default change needs no edit here).
  cucascade::io::rest::config const d{};
  for (auto const* body : {"        {}\n", "        tls: {}\n"}) {
    CAPTURE(body);
    auto const cfg = load_scan_manager("sirius_rest_defaults.yaml", rest_yaml(body));
    auto const& r  = cfg.rest;
    CHECK(cfg.rest_n_reactors == 2);
    CHECK(r.request_timeout_s == d.request_timeout_s);
    CHECK(r.stall_speed_limit_bytes == d.stall_speed_limit_bytes);
    CHECK(r.stall_time_s == d.stall_time_s);
    CHECK(r.max_connections == d.max_connections);
    CHECK(r.merge_max_gap == d.merge_max_gap);
    CHECK(r.upkeep_interval == d.upkeep_interval);
    CHECK(r.upkeep_http1 == d.upkeep_http1);
    CHECK(r.conn_max_age == d.conn_max_age);
    CHECK(r.min_segment_bytes == d.min_segment_bytes);
    CHECK(r.max_segment_bytes == d.max_segment_bytes);
    CHECK(r.receive_buffer_bytes == d.receive_buffer_bytes);
    CHECK(r.queue_depth == d.queue_depth);
    CHECK(r.iowq_max_unbound == d.iowq_max_unbound);
    CHECK(r.wait_mode == d.wait_mode);
    CHECK(r.dns_ttl == d.dns_ttl);
    CHECK(r.dns_accumulate == d.dns_accumulate);
    CHECK(r.dns_max_addresses == d.dns_max_addresses);
    CHECK(r.connect_timeout == d.connect_timeout);
    CHECK(r.keepalive_idle_s == d.keepalive_idle_s);
    CHECK(r.keepalive_interval_s == d.keepalive_interval_s);
    CHECK(r.keepalive_count == d.keepalive_count);
    CHECK(r.min_mtu == d.min_mtu);
    CHECK(r.mtu_reject_max_attempts == d.mtu_reject_max_attempts);
    CHECK(r.close_with_rst == d.close_with_rst);
    CHECK(r.fan_out == d.fan_out);
    CHECK(r.interface == d.interface);
    CHECK(r.pin_to_irq_cpus == d.pin_to_irq_cpus);
    CHECK(r.write_split_bytes == d.write_split_bytes);
    CHECK(r.expect_continue == d.expect_continue);
    CHECK(r.continue_timeout == d.continue_timeout);
    CHECK(r.complete_timeout_s == d.complete_timeout_s);
    CHECK(r.max_write_connections == d.max_write_connections);
    CHECK(r.max_retry_attempts == d.max_retry_attempts);
    CHECK(r.max_auth_retry_attempts == d.max_auth_retry_attempts);
    CHECK(r.retry_backoff_base == d.retry_backoff_base);
    CHECK(r.retry_jitter == d.retry_jitter);
    CHECK(r.honor_retry_after == d.honor_retry_after);
    CHECK(r.footer_probe_bytes == d.footer_probe_bytes);
    CHECK(r.list_max_matches == d.list_max_matches);
    CHECK(r.list_max_scanned == d.list_max_scanned);
    CHECK(r.tls.ktls == d.tls.ktls);
    CHECK(r.tls.fallback_rx == d.tls.fallback_rx);
    CHECK(r.tls.placement == d.tls.placement);
    CHECK(r.tls.min_version == d.tls.min_version);
    CHECK(r.tls.max_version == d.tls.max_version);
    CHECK(r.tls.expect_no_pad == d.tls.expect_no_pad);
    CHECK(r.tls.session_cache == d.tls.session_cache);
    CHECK(r.tls.session_cache_per_endpoint == d.tls.session_cache_per_endpoint);
    CHECK(r.tls.max_connection_bytes == d.tls.max_connection_bytes);
    CHECK(r.tls.ciphersuites == d.tls.ciphersuites);
    CHECK(r.tls.cipher_list == d.tls.cipher_list);
    CHECK(r.tls.groups == d.tls.groups);
    CHECK(r.tls.sni_override == d.tls.sni_override);
    CHECK(r.tls.allow_tls13_without_rekey == d.tls.allow_tls13_without_rekey);
    CHECK(r.tls.verify == d.tls.verify);
  }
}

TEST_CASE("sirius_config reads REST transport tuning", "[scan_manager][config][rest]")
{
  using cucascade::io::rest::reactor_wait_mode;
  auto const cfg = load_scan_manager("sirius_rest_transport.yaml",
                                     scan_manager_yaml("      rest_n_reactors: 4\n"
                                                       "      rest:\n"
                                                       "        n_max_concurrent_scans: 5\n"
                                                       "        request_timeout_s: 11\n"
                                                       "        stall_speed_limit_bytes: 4096\n"
                                                       "        stall_time_s: 3\n"
                                                       "        merge_max_gap: 16 KiB\n"
                                                       "        min_segment_bytes: 2 MiB\n"
                                                       "        max_segment_bytes: 8 MiB\n"
                                                       "        footer_probe_bytes: 64 KiB\n"
                                                       "        list_max_matches: 13\n"
                                                       "        list_max_scanned: 31\n"
                                                       "        retry_backoff_base_ms: 7ms\n"
                                                       "        retry_jitter_ms: 9\n"
                                                       "        max_retry_attempts: 5\n"
                                                       "        max_auth_retry_attempts: 2\n"
                                                       "        honor_retry_after: false\n"
                                                       "        max_connections: 13\n"
                                                       "        conn_max_age_s: 2s\n"
                                                       "        upkeep_interval_ms: 1500\n"
                                                       "        upkeep_http1: true\n"
                                                       "        connect_timeout_ms: 1200\n"
                                                       "        dns_ttl_s: 30\n"
                                                       "        dns_accumulate: true\n"
                                                       "        dns_max_addresses: 8\n"
                                                       "        keepalive_idle_s: 20\n"
                                                       "        keepalive_interval_s: 5\n"
                                                       "        keepalive_count: 3\n"
                                                       "        min_mtu: 9001\n"
                                                       "        mtu_reject_max_attempts: 2\n"
                                                       "        close_with_rst: false\n"
                                                       "        receive_buffer_bytes: 1 MiB\n"
                                                       "        queue_depth: 256\n"
                                                       "        wait_mode: uring\n"
                                                       "        iowq_max_unbound: 9\n"
                                                       "        fan_out: 3\n"
                                                       "        interface: eth1\n"
                                                       "        pin_to_irq_cpus: false\n"
                                                       "        write_part_bytes: 10 MiB\n"
                                                       "        expect_continue: true\n"
                                                       "        continue_timeout_ms: 250\n"
                                                       "        complete_timeout_s: 120\n"
                                                       "        max_write_connections: 4\n"));
  CHECK(cfg.rest_n_reactors == 4);
  auto const check = [](cucascade::io::rest::config const& rest) {
    CHECK(rest.n_max_concurrent_scans == 5);
    CHECK(rest.n_max_concurrent_scans_explicit);
    CHECK(rest.request_timeout_s == 11);
    CHECK(rest.stall_speed_limit_bytes == 4096);
    CHECK(rest.stall_time_s == 3);
    CHECK(rest.merge_max_gap == (16UL << 10));
    CHECK(rest.merge_gap_size() == (16UL << 10));
    CHECK(rest.min_segment_bytes == (2UL << 20));
    CHECK(rest.max_segment_bytes == (8UL << 20));
    CHECK(rest.footer_probe_bytes == (64UL << 10));
    CHECK(rest.list_max_matches == 13);
    CHECK(rest.list_max_scanned == 31);
    CHECK(rest.retry_backoff_base == std::chrono::milliseconds{7});
    CHECK(rest.retry_jitter == std::chrono::milliseconds{9});
    CHECK(rest.max_retry_attempts == 5);
    CHECK(rest.max_auth_retry_attempts == 2);
    CHECK_FALSE(rest.honor_retry_after);
    CHECK(rest.max_connections == 13);
    CHECK(rest.conn_max_age == std::chrono::seconds{2});
    CHECK(rest.upkeep_interval == std::chrono::milliseconds{1500});
    CHECK(rest.upkeep_http1);
    CHECK(rest.connect_timeout == std::chrono::milliseconds{1200});
    CHECK(rest.dns_ttl == std::chrono::seconds{30});
    CHECK(rest.dns_accumulate);
    CHECK(rest.dns_max_addresses == 8);
    CHECK(rest.keepalive_idle_s == 20);
    CHECK(rest.keepalive_interval_s == 5);
    CHECK(rest.keepalive_count == 3);
    CHECK(rest.min_mtu == 9001);
    CHECK(rest.mtu_reject_max_attempts == 2);
    CHECK_FALSE(rest.close_with_rst);
    CHECK(rest.receive_buffer_bytes == (1UL << 20));
    CHECK(rest.queue_depth == 256);
    CHECK(rest.wait_mode == reactor_wait_mode::uring);
    CHECK(rest.iowq_max_unbound == 9);
    CHECK(rest.fan_out == 3);
    CHECK(rest.interface == "eth1");
    CHECK_FALSE(rest.pin_to_irq_cpus);
    CHECK(rest.write_split_bytes == (10UL << 20));
    CHECK(rest.expect_continue);
    CHECK(rest.continue_timeout == std::chrono::milliseconds{250});
    CHECK(rest.complete_timeout_s == 120);
    CHECK(rest.max_write_connections == 4);
  };
  check(cfg.rest);
  check(cfg.to_io_config().rest);
  CHECK(cfg.to_io_config().rest_n_reactors == 4);
}

TEST_CASE("REST transport accepts documented zero sentinels and range boundaries",
          "[scan_manager][config][rest]")
{
  auto const cfg = load_scan_manager("sirius_rest_transport_bounds.yaml",
                                     scan_manager_yaml("      rest_n_reactors: 0\n"
                                                       "      rest:\n"
                                                       "        n_max_concurrent_scans: 0\n"
                                                       "        request_timeout_s: 0\n"
                                                       "        stall_speed_limit_bytes: 0\n"
                                                       "        stall_time_s: 0\n"
                                                       "        merge_max_gap: 0\n"
                                                       "        conn_max_age_s: 0\n"
                                                       "        upkeep_interval_ms: 0\n"
                                                       "        retry_backoff_base_ms: 0\n"
                                                       "        retry_jitter_ms: 0\n"
                                                       "        max_retry_attempts: 1\n"
                                                       "        max_auth_retry_attempts: 1\n"
                                                       "        max_connections: 1\n"
                                                       "        connect_timeout_ms: 1\n"
                                                       "        min_segment_bytes: 1\n"
                                                       "        max_segment_bytes: 1\n"
                                                       "        keepalive_count: 0\n"
                                                       "        min_mtu: 0\n"
                                                       "        mtu_reject_max_attempts: 1\n"
                                                       "        receive_buffer_bytes: 16 KiB\n"
                                                       "        queue_depth: 2\n"
                                                       "        iowq_max_unbound: 0\n"
                                                       "        dns_max_addresses: 4096\n"
                                                       "        footer_probe_bytes: 1 GiB\n"
                                                       "        fan_out: 1\n"
                                                       "        write_part_bytes: 5 MiB\n"
                                                       "        continue_timeout_ms: 0\n"
                                                       "        complete_timeout_s: 0\n"
                                                       "        max_write_connections: 0\n"
                                                       "        tls:\n"
                                                       "          session_cache: false\n"
                                                       "          session_cache_per_endpoint: 0\n"
                                                       "          max_connection_bytes: 0\n"));
  // rest_n_reactors 0 sizes the pool from the egress NIC when the REST ioctx is built.
  CHECK(cfg.rest_n_reactors == 0);
  CHECK(cfg.rest.n_max_concurrent_scans == 0);
  CHECK(cfg.rest.n_max_concurrent_scans_explicit);
  CHECK(cfg.rest.request_timeout_s == 0);
  CHECK(cfg.rest.stall_speed_limit_bytes == 0);
  CHECK(cfg.rest.stall_time_s == 0);
  CHECK(cfg.rest.merge_gap_size() == 0);
  CHECK(cfg.rest.conn_max_age.count() == 0);
  CHECK(cfg.rest.upkeep_interval.count() == 0);
  CHECK(cfg.rest.retry_backoff_base.count() == 0);
  CHECK(cfg.rest.retry_jitter.count() == 0);
  CHECK(cfg.rest.max_retry_attempts == 1);
  CHECK(cfg.rest.max_auth_retry_attempts == 1);
  CHECK(cfg.rest.max_connections == 1);
  CHECK(cfg.rest.connect_timeout.count() == 1);
  CHECK(cfg.rest.min_segment_bytes == 1);
  CHECK(cfg.rest.max_segment_bytes == 1);
  CHECK(cfg.rest.keepalive_count == 0);
  CHECK(cfg.rest.min_mtu == 0);
  CHECK(cfg.rest.mtu_reject_max_attempts == 1);
  CHECK(cfg.rest.receive_buffer_bytes == (16UL << 10));
  CHECK(cfg.rest.queue_depth == 2);
  CHECK(cfg.rest.iowq_max_unbound == 0);
  CHECK(cfg.rest.dns_max_addresses == 4096);
  CHECK(cfg.rest.footer_probe_bytes == (1UL << 30));
  CHECK(cfg.rest.fan_out == 1);
  CHECK(cfg.rest.write_split_bytes == (5UL << 20));
  CHECK(cfg.rest.continue_timeout.count() == 0);
  CHECK(cfg.rest.complete_timeout_s == 0);
  CHECK(cfg.rest.max_write_connections == 0);
  CHECK(cfg.rest.tls.session_cache_per_endpoint == 0);
  CHECK(cfg.rest.tls.max_connection_bytes == 0);

  auto const upper = load_scan_manager("sirius_rest_transport_upper.yaml",
                                       rest_yaml("        max_connections: 64\n"
                                                 "        queue_depth: 32768\n"
                                                 "        min_mtu: 65607\n"
                                                 "        receive_buffer_bytes: 4 MiB\n"
                                                 "        write_part_bytes: 5 GiB\n"
                                                 "        max_write_connections: 64\n"));
  CHECK(upper.rest.queue_depth == 32768);
  CHECK(upper.rest.min_mtu == 65607);
  CHECK(upper.rest.receive_buffer_bytes == (4UL << 20));
  CHECK(upper.rest.write_split_bytes == (5UL << 30));
  CHECK(upper.rest.max_write_connections == 64);

  auto const tight =
    load_scan_manager("sirius_rest_queue_tight.yaml",
                      rest_yaml("        max_connections: 64\n        queue_depth: 128\n"));
  CHECK(tight.rest.queue_depth == 128);

  // The default fan-out (2) is clamped to a one-reactor pool rather than rejected,
  // and any fan-out is left to the REST ioctx when the pool is NIC-sized.
  auto const one = load_scan_manager("sirius_rest_fan_out_default.yaml",
                                     scan_manager_yaml("      rest_n_reactors: 1\n"));
  CHECK(one.rest.fan_out == 2);
  auto const automatic = load_scan_manager("sirius_rest_fan_out_auto.yaml",
                                           scan_manager_yaml("      rest_n_reactors: 0\n"
                                                             "      rest:\n"
                                                             "        fan_out: 8\n"));
  CHECK(automatic.rest.fan_out == 8);
}

TEST_CASE("sirius_config rejects invalid REST transport ranges with key context",
          "[scan_manager][config][rest]")
{
  for (auto const& [body, key] :
       {std::pair{"n_max_concurrent_scans: -1", "n_max_concurrent_scans"},
        std::pair{"request_timeout_s: -1", "request_timeout_s"},
        std::pair{"stall_speed_limit_bytes: -1", "stall_speed_limit_bytes"},
        std::pair{"stall_time_s: -1", "stall_time_s"},
        std::pair{"conn_max_age_s: -1", "conn_max_age_s"},
        std::pair{"upkeep_interval_ms: -1", "upkeep_interval_ms"},
        std::pair{"retry_backoff_base_ms: -1", "retry_backoff_base_ms"},
        std::pair{"retry_jitter_ms: -1", "retry_jitter_ms"},
        std::pair{"max_retry_attempts: -1", "max_retry_attempts"},
        std::pair{"max_retry_attempts: 0", "max_retry_attempts"},
        std::pair{"max_auth_retry_attempts: -1", "max_auth_retry_attempts"},
        std::pair{"max_auth_retry_attempts: 0", "max_auth_retry_attempts"},
        std::pair{"list_max_matches: 0", "list_max_matches"},
        std::pair{"list_max_scanned: -1", "list_max_scanned"},
        std::pair{"footer_probe_bytes: 2 GiB", "footer_probe_bytes"},
        std::pair{"connect_timeout_ms: 0", "connect_timeout_ms"},
        std::pair{"connect_timeout_ms: -1", "connect_timeout_ms"},
        std::pair{"connect_timeout_ms: 0.5ms", "connect_timeout_ms"},
        std::pair{"max_connections: 0", "max_connections"},
        std::pair{"max_connections: -1", "max_connections"},
        std::pair{"max_connections: 4097", "max_connections"},
        std::pair{"min_segment_bytes: 0", "min_segment_bytes"},
        std::pair{"min_segment_bytes: -1", "min_segment_bytes"},
        std::pair{"max_segment_bytes: -1", "max_segment_bytes"},
        std::pair{"min_segment_bytes: 64 MiB", "max_segment_bytes"},
        std::pair{"max_segment_bytes: 512 KiB", "max_segment_bytes"},
        std::pair{"dns_ttl_s: 0", "dns_ttl_s"},
        std::pair{"dns_max_addresses: 0", "dns_max_addresses"},
        std::pair{"dns_max_addresses: 4097", "dns_max_addresses"},
        std::pair{"keepalive_idle_s: 0", "keepalive_idle_s"},
        std::pair{"keepalive_interval_s: 0", "keepalive_interval_s"},
        std::pair{"keepalive_count: -1", "keepalive_count"},
        std::pair{"min_mtu: 65608", "min_mtu"},
        std::pair{"min_mtu: -1", "min_mtu"},
        std::pair{"mtu_reject_max_attempts: 0", "mtu_reject_max_attempts"},
        std::pair{"mtu_reject_max_attempts: 4294967296", "mtu_reject_max_attempts"},
        std::pair{"receive_buffer_bytes: 8 KiB", "receive_buffer_bytes"},
        std::pair{"receive_buffer_bytes: 8 MiB", "receive_buffer_bytes"},
        std::pair{"queue_depth: 127", "queue_depth"},
        std::pair{"queue_depth: 32769", "queue_depth"},
        std::pair{"queue_depth: 0", "queue_depth"},
        std::pair{"max_connections: 1024", "queue_depth"},
        std::pair{"wait_mode: poll", "wait_mode"},
        std::pair{"iowq_max_unbound: -1", "iowq_max_unbound"},
        std::pair{"iowq_max_unbound: 4294967296", "iowq_max_unbound"},
        std::pair{"fan_out: 0", "fan_out"},
        std::pair{"write_part_bytes: 0", "write_part_bytes"},
        std::pair{"write_part_bytes: -1", "write_part_bytes"},
        std::pair{"write_part_bytes: 4 MiB", "write_part_bytes"},
        std::pair{"write_part_bytes: 6 GiB", "write_part_bytes"},
        std::pair{"expect_continue: maybe", "expect_continue"},
        std::pair{"continue_timeout_ms: -1", "continue_timeout_ms"},
        std::pair{"complete_timeout_s: -1", "complete_timeout_s"},
        std::pair{"max_write_connections: -1", "max_write_connections"},
        std::pair{"max_write_connections: 65", "max_write_connections"}}) {
    CAPTURE(body);
    scoped_yaml yaml("sirius_rest_transport_invalid.yaml",
                     rest_yaml(std::string{"        "} + body + "\n"));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring(std::string{"rest."} + key));
  }

  // The REST reactor count is a nonnegative total.
  for (auto const* value : {"-1", "1.5", "many"}) {
    CAPTURE(value);
    scoped_yaml yaml("sirius_rest_reactors_invalid.yaml",
                     scan_manager_yaml(std::string{"      rest_n_reactors: "} + value + "\n"));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring("rest_n_reactors"));
  }

  // An explicit fan-out may not exceed an explicit reactor count (rest_ioctx's rule).
  require_load_error("sirius_rest_fan_out_too_wide.yaml",
                     scan_manager_yaml("      rest_n_reactors: 2\n"
                                       "      rest:\n"
                                       "        fan_out: 3\n"),
                     "'rest.fan_out': must be at most rest_n_reactors (2)");
}

TEST_CASE("sirius_config rejects REST keys the transport does not have",
          "[scan_manager][config][rest]")
{
  // Never-released spellings of a different transport design (ring_entries is
  // queue_depth here; the TLS handshake is bounded by connect_timeout_ms).
  for (auto const* key : {"keep_warm_connections: 4",
                          "enable_controller: false",
                          "link_gbps: 100",
                          "tls_handshake_timeout_ms: 3000",
                          "ring_entries: 1024",
                          "sqpoll: true",
                          "defer_taskrun: true"}) {
    CAPTURE(key);
    scoped_yaml yaml("sirius_rest_unknown_key.yaml",
                     rest_yaml(std::string{"        "} + key + "\n"));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring("unknown config key") &&
                          Catch::Matchers::ContainsSubstring("in rest"));
  }
}

TEST_CASE("sirius_config reads TLS transport policy", "[scan_manager][config][rest][tls]")
{
  namespace tls    = cucascade::io::tls;
  auto const cfg   = load_scan_manager("sirius_rest_tls.yaml",
                                     tls_yaml("          ktls: require\n"
                                                "          user_space_rx: library\n"
                                                "          recv_placement: iowq\n"
                                                "          min_version: tls1.3\n"
                                                "          max_version: tls1.3\n"
                                                "          expect_no_pad: false\n"
                                                "          session_cache: true\n"
                                                "          session_cache_per_endpoint: 7\n"
                                                "          max_connection_bytes: 12 GiB\n"
                                                "          ciphersuites: TLS_AES_128_GCM_SHA256\n"
                                                "          cipher_list: ECDHE-RSA-AES128-GCM-SHA256\n"
                                                "          groups: X25519:P-256\n"
                                                "          sni_override: object-store.test\n"
                                                "          allow_tls13_without_rekey: false\n"));
  auto const check = [](tls::config const& policy) {
    CHECK(policy.ktls == tls::ktls_mode::require);
    CHECK(policy.fallback_rx == tls::user_space_rx::library);
    CHECK(policy.placement == tls::recv_placement::iowq);
    CHECK(policy.min_version == tls::version_bound::tls1_3);
    CHECK(policy.max_version == tls::version_bound::tls1_3);
    CHECK_FALSE(policy.expect_no_pad);
    CHECK(policy.session_cache);
    CHECK(policy.session_cache_per_endpoint == 7);
    CHECK(policy.max_connection_bytes == (12ULL << 30));
    CHECK(policy.ciphersuites == "TLS_AES_128_GCM_SHA256");
    CHECK(policy.cipher_list == "ECDHE-RSA-AES128-GCM-SHA256");
    CHECK(policy.groups == "X25519:P-256");
    CHECK(policy.sni_override == "object-store.test");
    CHECK_FALSE(policy.allow_tls13_without_rekey);
    CHECK(policy.verify);
    CHECK(policy.ca_file.empty());
    CHECK(policy.ca_dir.empty());
    CHECK(policy.ca_blob.empty());
  };
  check(cfg.rest.tls);
  check(cfg.to_io_config().rest.tls);
}

TEST_CASE("REST TLS enum spellings select the planned transport modes",
          "[scan_manager][config][rest][tls]")
{
  namespace tls = cucascade::io::tls;
  for (auto const& [spelling, expected] : {std::pair{"prefer", tls::ktls_mode::prefer},
                                           std::pair{"require", tls::ktls_mode::require},
                                           std::pair{"disable", tls::ktls_mode::disable}}) {
    auto const cfg = load_scan_manager("sirius_tls_ktls.yaml",
                                       tls_yaml(std::string{"          ktls: "} + spelling + "\n"));
    CHECK(cfg.rest.tls.ktls == expected);
  }
  for (auto const& [spelling, expected] :
       {std::pair{"record_layer", tls::user_space_rx::record_layer},
        std::pair{"library", tls::user_space_rx::library}}) {
    auto const cfg = load_scan_manager(
      "sirius_tls_rx.yaml", tls_yaml(std::string{"          user_space_rx: "} + spelling + "\n"));
    CHECK(cfg.rest.tls.fallback_rx == expected);
  }
  for (auto const& [spelling, expected] : {std::pair{"automatic", tls::recv_placement::automatic},
                                           std::pair{"inline", tls::recv_placement::inline_},
                                           std::pair{"iowq", tls::recv_placement::iowq}}) {
    auto const cfg =
      load_scan_manager("sirius_tls_placement.yaml",
                        tls_yaml(std::string{"          recv_placement: "} + spelling + "\n"));
    CHECK(cfg.rest.tls.placement == expected);
  }
  for (auto const& [spelling, expected] : {std::pair{"tls1.2", tls::version_bound::tls1_2},
                                           std::pair{"tls1.3", tls::version_bound::tls1_3}}) {
    auto const cfg = load_scan_manager("sirius_tls_version.yaml",
                                       tls_yaml(std::string{"          min_version: "} + spelling +
                                                "\n          max_version: " + spelling + "\n"));
    CHECK(cfg.rest.tls.min_version == expected);
    CHECK(cfg.rest.tls.max_version == expected);
  }
  auto const automatic = load_scan_manager("sirius_tls_max_auto.yaml",
                                           tls_yaml("          max_version: auto\n"
                                                    "          max_connection_bytes: 0\n"
                                                    "          session_cache_per_endpoint: 1\n"));
  CHECK_FALSE(automatic.rest.tls.max_version.has_value());
  CHECK(automatic.rest.tls.max_connection_bytes == 0);
  CHECK(automatic.rest.tls.session_cache_per_endpoint == 1);
}

TEST_CASE("sirius_config rejects invalid TLS policies with key context",
          "[scan_manager][config][rest][tls]")
{
  for (auto const& [body, key] :
       {std::pair{"ktls: auto", "ktls"},
        std::pair{"ktls: user-space", "ktls"},
        std::pair{"ktls: PREFER", "ktls"},
        std::pair{"ktls: ''", "ktls"},
        std::pair{"ktls: []", "ktls"},
        std::pair{"user_space_rx: openssl", "user_space_rx"},
        std::pair{"recv_placement: auto", "recv_placement"},
        std::pair{"recv_placement: inline_", "recv_placement"},
        std::pair{"min_version: auto", "min_version"},
        std::pair{"min_version: tls1.1", "min_version"},
        std::pair{"max_version: TLS1.3", "max_version"},
        std::pair{"max_version: tls1.4", "max_version"},
        std::pair{"min_version: tls1.3\n          max_version: tls1.2", "max_version"},
        std::pair{"session_cache: true\n          session_cache_per_endpoint: 0",
                  "session_cache_per_endpoint"},
        std::pair{"session_cache_per_endpoint: -1", "session_cache_per_endpoint"},
        std::pair{"session_cache_per_endpoint: 4294967296", "session_cache_per_endpoint"},
        std::pair{"max_connection_bytes: -1", "max_connection_bytes"}}) {
    CAPTURE(body);
    scoped_yaml yaml("sirius_rest_tls_invalid.yaml",
                     tls_yaml(std::string{"          "} + body + "\n"));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring(std::string{"rest.tls."} + key));
  }
  for (auto const* body : {"        tls: []\n", "        tls: {unknown_tls_option: true}\n"}) {
    scoped_yaml yaml("sirius_rest_tls_shape.yaml", rest_yaml(body));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring("rest.tls"));
  }
  // Library-internal TLS fields are not configurable.
  for (auto const* key :
       {"alpn_http11: false", "client_cert_file: /c", "client_key_file: /k", "keylog_env: true"}) {
    CAPTURE(key);
    scoped_yaml yaml("sirius_rest_tls_internal.yaml",
                     tls_yaml(std::string{"          "} + key + "\n"));
    REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                        Catch::Matchers::ContainsSubstring("unknown config key"));
  }
}

TEST_CASE("REST TLS trust remains owned by object_store", "[scan_manager][config][rest][tls]")
{
  for (auto const& [key, value] : {std::pair{"verify", "false"},
                                   std::pair{"ca_file", "/tmp/ca.pem"},
                                   std::pair{"ca_dir", "/tmp/certs"},
                                   std::pair{"ca_blob", "PEM"}}) {
    // Even a null value is a forbidden second trust source.
    for (auto const* spelling : {value, "null"}) {
      CAPTURE(key, spelling);
      scoped_yaml yaml("sirius_tls_trust_rejected.yaml",
                       tls_yaml(std::string{"          "} + key + ": " + spelling + "\n"));
      using Catch::Matchers::ContainsSubstring;
      if (std::string_view{key} == "verify") {
        REQUIRE_THROWS_WITH(sirius::parsed_sirius_config::from_file(yaml.path()),
                            ContainsSubstring(std::string{"rest.tls."} + key) &&
                              ContainsSubstring("trust is owned by object_store") &&
                              ContainsSubstring("sirius_config::set_object_store_config()") &&
                              ContainsSubstring("VERIFY_SSL"));
      } else {
        REQUIRE_THROWS_WITH(
          sirius::parsed_sirius_config::from_file(yaml.path()),
          ContainsSubstring(std::string{"rest.tls."} + key) &&
            ContainsSubstring("trust is owned by object_store") &&
            ContainsSubstring("sirius_config::set_object_store_config()") &&
            ContainsSubstring("ca_bundle_path") && ContainsSubstring("NIX_SSL_CERT_FILE") &&
            ContainsSubstring("SSL_CERT_FILE") && !ContainsSubstring("CREATE SECRET"));
      }
    }
  }
}

TEST_CASE("sirius_config forces the native backend for multi-GPU",
          "[scan_manager][config][backend][multi_gpu]")
{
  if (!sirius::test::has_gpus(2)) { return; }

  auto const cufile = load_scan_manager("sirius_cufile_multi_gpu.yaml",
                                        "sirius:\n  topology:\n    num_gpus: 2\n  executor:\n    "
                                        "scan_manager:\n      backend: cufile\n");
  CHECK(cufile.backend == io_backend::native);
}

TEST_CASE("sirius_config keeps the Sirius scan_manager defaults", "[scan_manager][config][uring]")
{
  auto const check_defaults = [&](scan_manager_config const& cfg) {
    CHECK(cfg.uring_n_reactors == 4);
    CHECK(cfg.rest_n_reactors == 2);
    CHECK(cfg.uring.n_max_concurrent_scans == 0);
    CHECK_FALSE(cfg.uring.n_max_concurrent_scans_explicit);
    CHECK(cfg.uring.slices_per_pass == 4);
    CHECK(cfg.uring.range_batch_slices == 8);
  };

  SECTION("default-constructed") { check_defaults(scan_manager_config{}); }

  SECTION("empty uring block")
  {
    check_defaults(load_scan_manager("sirius_uring_empty_block.yaml", uring_yaml("        {}\n")));
  }
}

TEST_CASE("sirius_config reads uring.slices_per_pass", "[scan_manager][config][uring]")
{
  for (std::size_t const value : {std::size_t{0}, std::size_t{8}, std::size_t{64}}) {
    CAPTURE(value);
    auto const cfg =
      load_scan_manager("sirius_uring_slices_per_pass_" + std::to_string(value) + ".yaml",
                        uring_yaml("        slices_per_pass: " + std::to_string(value) + "\n"));
    CHECK(cfg.uring.slices_per_pass == value);
    CHECK(cfg.to_io_config().uring.slices_per_pass == value);
  }
}

TEST_CASE("sirius_config rejects an out-of-range uring.slices_per_pass",
          "[scan_manager][config][uring]")
{
  require_load_error("sirius_uring_slices_per_pass_65.yaml",
                     uring_yaml("        slices_per_pass: 65\n"),
                     "'uring.slices_per_pass': must be between 0 and 64 (0 = no cap: fill every "
                     "free staging slot), got 65");
  require_load_error("sirius_uring_slices_per_pass_negative.yaml",
                     uring_yaml("        slices_per_pass: -1\n"),
                     "'uring.slices_per_pass': must be between 0 and 64 (0 = no cap: fill every "
                     "free staging slot), got -1");
}

TEST_CASE("sirius_config reads uring.range_batch_slices", "[scan_manager][config][uring]")
{
  for (std::size_t const value : {std::size_t{0}, std::size_t{1}, std::size_t{32}}) {
    CAPTURE(value);
    auto const cfg =
      load_scan_manager("sirius_uring_range_batch_slices_" + std::to_string(value) + ".yaml",
                        uring_yaml("        range_batch_slices: " + std::to_string(value) + "\n"));
    CHECK(cfg.uring.range_batch_slices == value);
    CHECK(cfg.to_io_config().uring.range_batch_slices == value);
  }
  require_load_error("sirius_uring_range_batch_slices_negative.yaml",
                     uring_yaml("        range_batch_slices: -1\n"),
                     "'uring.range_batch_slices': must be 0 or more (0 = no split), got -1");
}

TEST_CASE("uring prefetch_reactors is accepted as a deprecated no-op",
          "[scan_manager][config][uring]")
{
  // Reserving reactors for prefetch was replaced by each reactor's priority
  // tiers.  A config that still names the key loads with a deprecation WARN
  // and nothing is partitioned.
  auto const baseline =
    load_scan_manager("sirius_uring_no_prefetch_reactors.yaml", uring_yaml("        {}\n"));

  auto const value = GENERATE(as<std::string>{}, "1", "null", "auto", "[1, 2]", "-3");
  CAPTURE(value);
  sirius::test::scoped_recording_log_sink logs{"warn"};
  auto const cfg = load_scan_manager("sirius_uring_prefetch_reactors.yaml",
                                     uring_yaml("        prefetch_reactors: " + value + "\n"));

  auto const io          = cfg.to_io_config().uring;
  auto const baseline_io = baseline.to_io_config().uring;
  CHECK(io.n_max_concurrent_scans == baseline_io.n_max_concurrent_scans);
  CHECK(io.n_max_concurrent_scans_explicit == baseline_io.n_max_concurrent_scans_explicit);
  CHECK(io.slices_per_pass == baseline_io.slices_per_pass);
  CHECK(io.range_batch_slices == baseline_io.range_batch_slices);
  CHECK(io.use_odirect == baseline_io.use_odirect);
  CHECK(cfg.uring_n_reactors == baseline.uring_n_reactors);

  std::size_t warnings = 0;
  for (auto const& record : logs.records()) {
    if (record.message.find("'uring.prefetch_reactors' is deprecated and ignored") !=
        std::string::npos) {
      ++warnings;
      CHECK(record.level == sirius::log::level::warn);
    }
  }
  CHECK(warnings == 1);
}

TEST_CASE("Native runtime YAML rejects insufficient reader capacity",
          "[scan_manager][config][preparation_config][native_runtime]")
{
  auto fields = GENERATE("native_reader_workers: 3\n        max_native_readers: 2",
                         "native_reader_workers: 65",
                         "max_native_readers: 1");
  scoped_yaml yaml("sirius_native_runtime_invalid.yaml",
                   scan_manager_yaml(std::string("      preparation:\n        ") + fields + "\n"));
  CHECK_THROWS_AS(sirius::parsed_sirius_config::from_file(yaml.path()),
                  sirius::configuration_input_error);
}
