// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/executor.hpp"
#include "config/yaml.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/config.hpp>

#include <chrono>
#include <optional>
#include <string>
#include <string_view>

using namespace sirius::config;
using namespace sirius::config::size_literals;
using namespace std::chrono_literals;
namespace tls  = cucascade::io::tls;
namespace rest = cucascade::io::rest;

namespace {

struct scan_root : block_base<scan_root> {
  executor_options executor;
  auto fields() { return field_list(sub("executor", executor)); }
};

constexpr std::string_view prefix = "sirius:\n  executor:\n    scan_manager:\n";

scan_root read(std::string_view body)
{
  scan_root root;
  read_yaml(root, std::string(prefix) + std::string(body), "sirius", environment::prod);
  return root;
}

std::string read_error(std::string_view body)
{
  scan_root root;
  try {
    read_yaml(root, std::string(prefix) + std::string(body), "sirius", environment::prod);
  } catch (const config_error& e) {
    return e.what();
  }
  FAIL("expected a config_error");
  return {};
}

}  // namespace

TEST_CASE("scan manager blocks declare their keys once", "[config][config_scan_manager]")
{
  scan_root{}.check_declaration();
}

TEST_CASE("absent scan manager blocks take today's defaults", "[config][config_scan_manager]")
{
  scan_root root;
  read_yaml(root, "sirius: {}\n", "sirius", environment::prod);
  auto const& sm = root.executor.scan_manager;
  // Derived when the configuration is resolved.
  REQUIRE(sm.num_threads.is_unset());
  REQUIRE(sm.uring_n_reactors.is_unset());
  REQUIRE(sm.rest_n_reactors.is_unset());
  REQUIRE(sm.uring.n_max_concurrent_scans.is_unset());
  REQUIRE(sm.rest.n_max_concurrent_scans.is_unset());
  REQUIRE(sm.rest.max_connections.is_unset());
  REQUIRE(sm.rest.min_mtu.is_unset());
  // Meaningful when unset.
  REQUIRE(sm.max_readahead_scans.is_unset());
  REQUIRE(sm.readahead_strategy.is_unset());
  REQUIRE(sm.preparation.max_inflight_metadata_jobs.is_unset());

  REQUIRE(sm.backend.get() == sirius::scan_manager::io_backend::native);
  REQUIRE(sm.reserve_rest_reactor_cpus.get());
  REQUIRE(sm.uring.slices_per_pass.get() == 4);
  REQUIRE(sm.uring.range_batch_slices.get() == 8);
  REQUIRE(sm.rest.request_timeout.get() == 30s);
  REQUIRE(sm.rest.stall_speed_limit_bytes.get() == 64_KiB);
  REQUIRE(sm.rest.footer_probe_bytes.get() == 512_KiB);
  REQUIRE(sm.rest.connect_timeout.get() == 3000ms);
  REQUIRE(sm.rest.keepalive_idle.get() == 60s);
  REQUIRE(sm.rest.wait_mode.get() == rest::reactor_wait_mode::epoll);
  REQUIRE(sm.rest.tls.ktls.get() == tls::ktls_mode::prefer);
  REQUIRE(sm.rest.tls.max_version.get() == std::nullopt);
  REQUIRE(sm.rest.tls.max_connection_bytes.get() == 64_GiB);
  REQUIRE(sm.cache.mode.get() == cucascade::io::cache::cache_mode::cucs);  // Sirius's default
  REQUIRE(sm.cache.eviction.get() == cucascade::io::cache::eviction_policy::lru);
  REQUIRE(sm.memory_prefetcher.poll_interval.get() == 2ms);
  REQUIRE(sm.memory_prefetcher.drain_quiet.get() == 100ms);
  REQUIRE(sm.preparation.interrupt_check_interval.get() == 10ms);
  REQUIRE(sm.preparation.partial_batch.min_fill_percent.get() == 50);
  REQUIRE(sm.preparation.partial_batch.max_residence.get() == 2000ms);
}

TEST_CASE("scan manager keys read as upstream reads them", "[config][config_scan_manager]")
{
  auto const root = read(R"(      num_threads: 12
      cpu_affinity: [0, 1]
      backend: sirius
      uring_n_reactors: 2
      rest_n_reactors: 8
      max_readahead_scans: 0
      readahead_strategy: opportunistic
      uring:
        n_max_concurrent_scans: 0
        slices_per_pass: 64
      rest:
        request_timeout_s: 2 min
        stall_speed_limit_bytes: 1 MiB
        keepalive_interval_s: 15
        max_connections: 32
        min_mtu: 9000
        wait_mode: uring
        fan_out: 4
        tls:
          ktls: require
          user_space_rx: library
          recv_placement: inline
          min_version: tls1.3
          max_version: auto
      cache:
        mode: sirius
        eviction: idle
      memory_prefetcher:
        enable: true
        poll_interval_ms: 5
      preparation:
        max_native_readers: 8
        partial_batch:
          min_fill_percent: 75
)");
  auto const& sm  = root.executor.scan_manager;
  REQUIRE(sm.num_threads.get() == 12);
  REQUIRE(sm.backend.get() == sirius::scan_manager::io_backend::native);  // deprecated alias
  REQUIRE(sm.max_readahead_scans.get() == 0);
  REQUIRE(sm.readahead_strategy.get() == sirius::scan_manager::prefetch_strategy::opportunistic);
  REQUIRE(sm.uring.n_max_concurrent_scans.user_defined());  // an explicit 0 counts
  REQUIRE(sm.uring.slices_per_pass.get() == 64);
  REQUIRE(sm.rest.request_timeout.get() == 120s);
  REQUIRE(sm.rest.stall_speed_limit_bytes.get() == 1_MiB);
  REQUIRE(sm.rest.keepalive_interval.get() == 15s);  // bare numbers keep their unit
  REQUIRE(sm.rest.max_connections.get() == 32);
  REQUIRE(sm.rest.min_mtu.get() == bytes(9000));
  REQUIRE(sm.rest.wait_mode.get() == rest::reactor_wait_mode::uring);
  REQUIRE(sm.rest.tls.ktls.get() == tls::ktls_mode::require);
  REQUIRE(sm.rest.tls.user_space_rx.get() == tls::user_space_rx::library);
  REQUIRE(sm.rest.tls.recv_placement.get() == tls::recv_placement::inline_);
  REQUIRE(sm.rest.tls.min_version.get() == tls::version_bound::tls1_3);
  REQUIRE(sm.rest.tls.max_version.user_defined());
  REQUIRE(sm.cache.mode.get() == cucascade::io::cache::cache_mode::cucs);  // deprecated alias
  REQUIRE(sm.cache.eviction.get() == cucascade::io::cache::eviction_policy::idle);
  REQUIRE(sm.memory_prefetcher.poll_interval.get() == 5ms);
  REQUIRE(sm.preparation.max_native_readers.get() == 8);
  REQUIRE(sm.preparation.partial_batch.min_fill_percent.get() == 75);
}

TEST_CASE("scan manager values are checked", "[config][config_scan_manager]")
{
  REQUIRE(read_error("      backend: kvikio\n") ==
          "'sirius.executor.scan_manager.backend': backend: kvikio has been removed; use "
          "backend: native instead");
  REQUIRE(read_error("      backend: s3\n") ==
          "'sirius.executor.scan_manager.backend': invalid enum value 's3'");
  REQUIRE(read_error("      uring_n_reactors: -1\n") ==
          "'sirius.executor.scan_manager.uring_n_reactors': must be a nonnegative integer, got "
          "'-1'");
  REQUIRE(read_error("      uring:\n        slices_per_pass: 65\n") ==
          "'sirius.executor.scan_manager.uring.slices_per_pass': must be between 0 and 64 (0 = no "
          "cap: fill every free staging slot), got 65");
  REQUIRE(read_error("      rest:\n        wait_mode: poll\n") ==
          "'sirius.executor.scan_manager.rest.wait_mode': invalid enum value 'poll' (expected "
          "epoll, uring)");
  REQUIRE(read_error("      rest:\n        max_retry_attempts: 4294967296\n") ==
          "'sirius.executor.scan_manager.rest.max_retry_attempts': value out of range: "
          "'4294967296' (must be at most 4294967295)");
  REQUIRE(read_error("      rest:\n        tls:\n          max_version: tls1.1\n") ==
          "'sirius.executor.scan_manager.rest.tls.max_version': invalid enum value 'tls1.1' "
          "(expected auto, tls1.2, tls1.3)");
  REQUIRE(read_error("      cache:\n        mode: memory\n") ==
          "'sirius.executor.scan_manager.cache.mode': invalid enum value 'memory'");
  REQUIRE(read_error("      memory_prefetcher:\n        poll_interval_ms: 0\n") ==
          "'sirius.executor.scan_manager.memory_prefetcher.poll_interval_ms': value out of range: "
          "'0 ms' (must be greater than 0)");
  REQUIRE(read_error("      preparation:\n        lookahead_batches: 0\n") ==
          "'sirius.executor.scan_manager.preparation.lookahead_batches': value out of range: '0' "
          "(must be greater than 0)");
  REQUIRE(
    read_error("      preparation:\n        partial_batch:\n          min_fill_percent: 0\n") ==
    "'sirius.executor.scan_manager.preparation.partial_batch.min_fill_percent': value out of "
    "range: '0' (must be between 1 and 100)");
}

TEST_CASE("scan manager cross-field rules", "[config][config_scan_manager]")
{
  REQUIRE(read_error("      rest_n_reactors: 2\n      rest:\n        fan_out: 3\n") ==
          "'sirius.executor.scan_manager.rest.fan_out': must be at most rest_n_reactors (2)");
  // The default fan-out is clamped rather than rejected; 0 reactors are sized from the NIC.
  REQUIRE(read("      rest_n_reactors: 1\n").executor.scan_manager.rest.fan_out.get() == 2);
  REQUIRE(read("      rest_n_reactors: 0\n      rest:\n        fan_out: 8\n")
            .executor.scan_manager.rest.fan_out.get() == 8);

  REQUIRE(read_error("      preparation:\n        native_reader_workers: 65\n") ==
          "'sirius.executor.scan_manager.preparation': max_native_readers must be at least "
          "native_reader_workers");
  REQUIRE(read_error("      preparation:\n        partial_batch:\n          min_residence: 3s\n") ==
          "'sirius.executor.scan_manager.preparation.partial_batch': max_residence must be at "
          "least min_residence");
}

TEST_CASE("retired scan manager keys explain themselves", "[config][config_scan_manager]")
{
  REQUIRE_THAT(read_error("      object_store: {}\n"),
               Catch::Matchers::StartsWith(
                 "'sirius.executor.scan_manager.object_store': no longer YAML-loadable"));
  REQUIRE_THAT(read_error("      kvikio: {}\n"),
               Catch::Matchers::StartsWith("'sirius.executor.scan_manager.kvikio': removed"));
  REQUIRE_THAT(
    read_error("      rest:\n        tls_verify: false\n"),
    Catch::Matchers::StartsWith("'sirius.executor.scan_manager.rest.tls_verify': removed"));
  REQUIRE_THAT(
    read_error("      rest:\n        ca_bundle_path: /x\n"),
    Catch::Matchers::StartsWith("'sirius.executor.scan_manager.rest.ca_bundle_path': removed"));
  for (auto const* key : {"verify", "ca_file", "ca_dir", "ca_blob"}) {
    INFO(key);
    REQUIRE_THAT(read_error(std::string("      rest:\n        tls:\n          ") + key + ": x\n"),
                 Catch::Matchers::ContainsSubstring("trust is owned by object_store"));
  }
  REQUIRE(read_error("      preparation:\n        max_inflight_jobs: 4\n") ==
          "unknown config key: 'max_inflight_jobs' in sirius.executor.scan_manager.preparation");
}

TEST_CASE("scan manager blocks write what they read", "[config][config_scan_manager]")
{
  auto const body = R"(      backend: native
      rest:
        request_timeout_s: 2 min
        tls:
          recv_placement: inline
          max_version: auto
      cache:
        mode: cucs
      memory_prefetcher:
        drain_quiet_ms: 1 s
)";
  auto const root = read(body);
  REQUIRE(parse_yaml(write_yaml(root, "sirius")) == parse_yaml(std::string(prefix) + body));

  scan_root all;
  read_yaml(all, write_yaml(root, "sirius", true), "sirius", environment::prod);
  REQUIRE(all.executor.scan_manager.rest.tls.cipher_list.user_defined());
  REQUIRE(all.executor.scan_manager.rest.max_connections.is_unset());  // derived, never written
  REQUIRE(all.executor.scan_manager.memory_prefetcher.drain_quiet.get() == 1000ms);
}
