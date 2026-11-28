// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/block.hpp"
#include "config/options/executor.hpp"
#include "config/yaml.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/config.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
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
  read_yaml(root, std::string(prefix) + std::string(body), "sirius");
  return root;
}

std::string read_error(std::string_view body)
{
  scan_root root;
  try {
    read_yaml(root, std::string(prefix) + std::string(body), "sirius");
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
  read_yaml(root, "sirius: {}\n", "sirius");
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
  REQUIRE_FALSE(sm.httpfs_external_file_cache.get());
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
          "backend: native instead; IO backend for scan reads");
  REQUIRE(read_error("      backend: s3\n") ==
          "'sirius.executor.scan_manager.backend': invalid enum value 's3'"
          "; IO backend for scan reads");
  REQUIRE(read_error("      uring_n_reactors: -1\n") ==
          "'sirius.executor.scan_manager.uring_n_reactors': must be a nonnegative integer, got "
          "'-1'"
          "; io_uring reactors for local-disk reads");
  REQUIRE(read_error("      uring:\n        slices_per_pass: 65\n") ==
          "'sirius.executor.scan_manager.uring.slices_per_pass': must be between 0 and 64 (0 = no "
          "cap: fill every free staging slot), got 65"
          "; slices a uring reactor reads per loop pass");
  REQUIRE(read_error("      rest:\n        wait_mode: poll\n") ==
          "'sirius.executor.scan_manager.rest.wait_mode': invalid enum value 'poll' (expected "
          "epoll, uring)"
          "; how an idle REST reactor blocks");
  REQUIRE(read_error("      rest:\n        max_retry_attempts: 4294967296\n") ==
          "'sirius.executor.scan_manager.rest.max_retry_attempts': value out of range: "
          "'4294967296' (must be between 1 and 4294967295)"
          "; attempt budget for transient request failures");
  REQUIRE(read_error("      rest:\n        tls:\n          max_version: tls1.1\n") ==
          "'sirius.executor.scan_manager.rest.tls.max_version': invalid enum value 'tls1.1' "
          "(expected auto, tls1.2, tls1.3)"
          "; maximum TLS version");
  REQUIRE(read_error("      cache:\n        mode: memory\n") ==
          "'sirius.executor.scan_manager.cache.mode': invalid enum value 'memory'"
          "; which cache the read path goes through");
  REQUIRE(read_error("      memory_prefetcher:\n        poll_interval_ms: 0\n") ==
          "'sirius.executor.scan_manager.memory_prefetcher.poll_interval_ms': value out of range: "
          "'0 ms' (must be greater than zero)"
          "; memory prefetcher sweep interval");
  REQUIRE(read_error("      preparation:\n        lookahead_batches: 0\n") ==
          "'sirius.executor.scan_manager.preparation.lookahead_batches': value out of range: '0' "
          "(must be greater than zero)"
          "; per-scan preparation target in scan batches");
  REQUIRE(
    read_error("      preparation:\n        partial_batch:\n          min_fill_percent: 0\n") ==
    "'sirius.executor.scan_manager.preparation.partial_batch.min_fill_percent': value out of "
    "range: '0' (must be between 1 and 100)"
    "; minimum batch fill for early publication");
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
          "'sirius.executor.scan_manager.preparation.max_native_readers': must be at least "
          "native_reader_workers");
  REQUIRE(read_error("      preparation:\n        partial_batch:\n          min_residence: 3s\n") ==
          "'sirius.executor.scan_manager.preparation.partial_batch.max_residence': must be at "
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
  read_yaml(all, write_yaml(root, "sirius", true), "sirius");
  REQUIRE(all.executor.scan_manager.rest.tls.cipher_list.user_defined());
  REQUIRE(all.executor.scan_manager.rest.max_connections.is_unset());  // derived, never written
  REQUIRE(all.executor.scan_manager.memory_prefetcher.drain_quiet.get() == 1000ms);
}

TEST_CASE("REST ranges are checked when the file is read", "[config][config_scan_manager]")
{
  for (auto const* entry : {"request_timeout_s: -1",        "stall_time_s: -1",
                            "conn_max_age_s: -1s",          "upkeep_interval_ms: -1",
                            "retry_backoff_base_ms: -1",    "retry_jitter_ms: -1",
                            "max_retry_attempts: 0",        "max_auth_retry_attempts: 0",
                            "min_segment_bytes: 0",         "receive_buffer_bytes: 8 KiB",
                            "receive_buffer_bytes: 8 MiB",  "queue_depth: 0",
                            "queue_depth: 40000",           "max_connections: 0",
                            "max_connections: 5000",        "dns_ttl_s: 0",
                            "dns_max_addresses: 0",         "dns_max_addresses: 5000",
                            "connect_timeout_ms: 0",        "keepalive_idle_s: 0",
                            "keepalive_interval_s: 0",      "keepalive_count: -1",
                            "keepalive_idle_s: 4294967297", "min_mtu: 70000",
                            "mtu_reject_max_attempts: 0",   "fan_out: 0",
                            "footer_probe_bytes: 2 GiB",    "list_max_matches: 0",
                            "list_max_scanned: 0"}) {
    std::string_view const text = entry;
    auto const key              = std::string(text.substr(0, text.find(':')));
    INFO(text);
    REQUIRE_THAT(read_error(std::string("      rest:\n        ") + entry + "\n"),
                 Catch::Matchers::StartsWith("'sirius.executor.scan_manager.rest." + key +
                                             "': value out of range"));
  }
  REQUIRE(read("      rest:\n        max_auth_retry_attempts: 4294967295\n")
            .executor.scan_manager.rest.max_auth_retry_attempts.get() == 4294967295ULL);
}

TEST_CASE("REST and TLS rules across keys", "[config][config_scan_manager]")
{
  REQUIRE(read_error("      rest:\n        max_segment_bytes: 1 MiB\n") ==
          "'sirius.executor.scan_manager.rest.max_segment_bytes': must be at least "
          "min_segment_bytes");
  REQUIRE(read_error("      rest:\n        max_connections: 64\n        queue_depth: 100\n") ==
          "'sirius.executor.scan_manager.rest.queue_depth': must be at least 2 * max_connections");
  // With max_connections derived, queue_depth is checked when the configuration is resolved.
  REQUIRE(
    read("      rest:\n        queue_depth: 100\n").executor.scan_manager.rest.queue_depth.get() ==
    100);
  REQUIRE(read_error("      rest:\n        tls:\n          min_version: tls1.3\n"
                     "          max_version: tls1.2\n") ==
          "'sirius.executor.scan_manager.rest.tls.max_version': must not be below min_version");
  REQUIRE(read_error("      rest:\n        tls:\n          session_cache: true\n"
                     "          session_cache_per_endpoint: 0\n") ==
          "'sirius.executor.scan_manager.rest.tls.session_cache_per_endpoint': must be greater "
          "than 0 with the session cache on");
}

TEST_CASE("explicit zeros count as user-defined, nulls do not", "[config][config_scan_manager]")
{
  auto const zeros = read(
                       "      rest_n_reactors: 0\n      rest:\n        min_mtu: 0\n"
                       "        n_max_concurrent_scans: 0\n")
                       .executor.scan_manager;
  REQUIRE(zeros.rest_n_reactors.user_defined());
  REQUIRE(zeros.rest.min_mtu.user_defined());
  REQUIRE(zeros.rest.n_max_concurrent_scans.user_defined());
  auto const nulls = read(
                       "      rest_n_reactors: ~\n      rest:\n        min_mtu: ~\n"
                       "        max_connections: ~\n")
                       .executor.scan_manager;
  REQUIRE(nulls.rest_n_reactors.is_unset());
  REQUIRE(nulls.rest.min_mtu.is_unset());
  REQUIRE(nulls.rest.max_connections.is_unset());
}

TEST_CASE("more scan manager rows", "[config][config_scan_manager]")
{
  // max_native_readers against the default worker count of 2, and against explicit counts.
  REQUIRE_THAT(
    read_error("      preparation:\n        max_native_readers: 1\n"),
    Catch::Matchers::EndsWith("max_native_readers': must be at least native_reader_workers"));
  REQUIRE_THAT(
    read_error("      preparation:\n        native_reader_workers: 3\n"
               "        max_native_readers: 2\n"),
    Catch::Matchers::EndsWith("max_native_readers': must be at least native_reader_workers"));
  // rest.fan_out against rest_n_reactors, at the boundaries.
  (void)read("      rest_n_reactors: 1\n      rest:\n        fan_out: 2\n");  // default, by value
  (void)read("      rest_n_reactors: 4\n      rest:\n        fan_out: 4\n");
  (void)read("      rest_n_reactors: ~\n      rest:\n        fan_out: 8\n");
  // partial_batch.
  auto const pb = [](std::string_view body) {
    return read_error(std::string("      preparation:\n        partial_batch:\n          ") +
                      std::string(body) + "\n");
  };
  REQUIRE_THAT(pb("max_residence: 0"),
               Catch::Matchers::EndsWith("max_residence': must be at least min_residence"));
  REQUIRE_THAT(pb("max_residence: 49ms"),
               Catch::Matchers::EndsWith("max_residence': must be at least min_residence"));
  REQUIRE_THAT(pb("min_residence: -1ms"),
               Catch::Matchers::ContainsSubstring("must be greater than zero"));
  REQUIRE_THAT(pb("min_fill_percent: 101"),
               Catch::Matchers::ContainsSubstring("must be between 1 and 100"));
  REQUIRE_THAT(read_error("      preparation:\n        interrupt_check_interval: 0\n"),
               Catch::Matchers::ContainsSubstring("must be greater than zero"));
  // Prefetcher, cache, strategy and counts.
  for (auto const* body : {"memory_prefetcher:\n        num_threads: 0",
                           "memory_prefetcher:\n        min_free_fraction: 1.5",
                           "memory_prefetcher:\n        drain_quiet_ms: -1",
                           "cache:\n        min_prefetching_budget_fraction: .nan",
                           "num_threads: 0",
                           "uring_n_reactors: 0"}) {
    INFO(body);
    REQUIRE_THAT(read_error(std::string("      ") + body + "\n"),
                 Catch::Matchers::ContainsSubstring("value out of range"));
  }
  REQUIRE_THAT(read_error("      cache:\n        eviction: x\n"),
               Catch::Matchers::ContainsSubstring("invalid enum value 'x'"));
  REQUIRE_THAT(read_error("      readahead_strategy: lazy\n"),
               Catch::Matchers::ContainsSubstring("invalid enum value 'lazy'"));
  REQUIRE_THAT(read_error("      max_readahead_scans: -1\n"),
               Catch::Matchers::ContainsSubstring("must be a nonnegative integer"));
  // Counts cuCascade keeps in an unsigned int.
  for (auto const* body : {"rest:\n        mtu_reject_max_attempts: 4294967296",
                           "rest:\n        iowq_max_unbound: 4294967296"}) {
    INFO(body);
    REQUIRE_THAT(read_error(std::string("      ") + body + "\n"),
                 Catch::Matchers::ContainsSubstring("integer out of range"));
  }
  REQUIRE_THAT(read_error("      rest:\n        tls:\n          session_cache_per_endpoint: -1\n"),
               Catch::Matchers::ContainsSubstring("must be a nonnegative integer"));
}

TEST_CASE("retired scan manager keys fire on null and keep their own advice",
          "[config][config_scan_manager]")
{
  REQUIRE_THAT(read_error("      object_store: ~\n"),
               Catch::Matchers::ContainsSubstring("no longer"));
  REQUIRE_THAT(read_error("      kvikio: ~\n"),
               Catch::Matchers::ContainsSubstring("backend: native"));
  REQUIRE_THAT(read_error("      rest:\n        tls:\n          verify: ~\n"),
               Catch::Matchers::ContainsSubstring("VERIFY_SSL on the selected CREATE SECRET"));
  REQUIRE_THAT(
    read_error("      rest:\n        tls:\n          ca_file: x\n"),
    Catch::Matchers::ContainsSubstring("NIX_SSL_CERT_FILE / SSL_CERT_FILE / SSL_CERT_DIR"));
}

TEST_CASE("deprecated spellings are written canonically", "[config][config_scan_manager]")
{
  auto const root    = read("      backend: sirius\n      cache:\n        mode: sirius\n");
  auto const written = parse_yaml(write_yaml(root, "sirius"));
  auto const& sm     = *written.find("sirius")->find("executor")->find("scan_manager");
  REQUIRE(sm.find("backend")->text() == "native");
  REQUIRE(sm.find("cache")->find("mode")->text() == "cucs");
}

TEST_CASE("scan manager fallbacks equal the runtime defaults", "[config][config_scan_manager]")
{
  scan_root root;
  read_yaml(root, "sirius: {}\n", "sirius");
  auto const& sm = root.executor.scan_manager;
  sirius::scan_manager::scan_manager_config const runtime{};
  auto const& r = runtime.rest;
  auto const& o = sm.rest;
  REQUIRE(o.request_timeout.get().count() == r.request_timeout_s);
  REQUIRE(o.stall_speed_limit_bytes.get().count() == r.stall_speed_limit_bytes);
  REQUIRE(o.stall_time.get().count() == r.stall_time_s);
  REQUIRE(o.merge_max_gap.get().count() == static_cast<std::uint64_t>(r.merge_max_gap));
  REQUIRE(o.min_segment_bytes.get().count() == static_cast<std::uint64_t>(r.min_segment_bytes));
  REQUIRE(o.max_segment_bytes.get().count() == static_cast<std::uint64_t>(r.max_segment_bytes));
  REQUIRE(o.footer_probe_bytes.get().count() == static_cast<std::uint64_t>(r.footer_probe_bytes));
  REQUIRE(o.list_max_matches.get() == r.list_max_matches);
  REQUIRE(o.list_max_scanned.get() == r.list_max_scanned);
  REQUIRE(o.retry_backoff_base.get() == r.retry_backoff_base);
  REQUIRE(o.retry_jitter.get() == r.retry_jitter);
  REQUIRE(o.max_retry_attempts.get() == r.max_retry_attempts);
  REQUIRE(o.max_auth_retry_attempts.get() == r.max_auth_retry_attempts);
  REQUIRE(o.honor_retry_after.get() == r.honor_retry_after);
  REQUIRE(o.conn_max_age.get() == r.conn_max_age);
  REQUIRE(o.upkeep_interval.get() == r.upkeep_interval);
  REQUIRE(o.upkeep_http1.get() == r.upkeep_http1);
  REQUIRE(o.connect_timeout.get() == r.connect_timeout);
  REQUIRE(o.dns_ttl.get() == r.dns_ttl);
  REQUIRE(o.dns_accumulate.get() == r.dns_accumulate);
  REQUIRE(o.dns_max_addresses.get() == r.dns_max_addresses);
  REQUIRE(o.keepalive_idle.get().count() == r.keepalive_idle_s);
  REQUIRE(o.keepalive_interval.get().count() == r.keepalive_interval_s);
  REQUIRE(o.keepalive_count.get() == r.keepalive_count);
  REQUIRE(o.mtu_reject_max_attempts.get() == r.mtu_reject_max_attempts);
  REQUIRE(o.close_with_rst.get() == r.close_with_rst);
  REQUIRE(o.receive_buffer_bytes.get().count() ==
          static_cast<std::uint64_t>(r.receive_buffer_bytes));
  REQUIRE(o.queue_depth.get() == r.queue_depth);
  REQUIRE(o.wait_mode.get() == r.wait_mode);
  REQUIRE(o.iowq_max_unbound.get() == r.iowq_max_unbound);
  REQUIRE(o.fan_out.get() == r.fan_out);
  REQUIRE(o.interface.get() == r.interface);
  REQUIRE(o.pin_to_irq_cpus.get() == r.pin_to_irq_cpus);
  REQUIRE(o.raise_reactor_priority.get() == r.raise_reactor_priority);

  auto const& t = r.tls;
  REQUIRE(o.tls.ktls.get() == t.ktls);
  REQUIRE(o.tls.user_space_rx.get() == t.fallback_rx);
  REQUIRE(o.tls.recv_placement.get() == t.placement);
  REQUIRE(o.tls.min_version.get() == t.min_version);
  REQUIRE(o.tls.max_version.get() == t.max_version);
  REQUIRE(o.tls.expect_no_pad.get() == t.expect_no_pad);
  REQUIRE(o.tls.session_cache.get() == t.session_cache);
  REQUIRE(o.tls.session_cache_per_endpoint.get() == t.session_cache_per_endpoint);
  REQUIRE(o.tls.max_connection_bytes.get().count() ==
          static_cast<std::uint64_t>(t.max_connection_bytes));
  REQUIRE(o.tls.ciphersuites.get() == t.ciphersuites);
  REQUIRE(o.tls.cipher_list.get() == t.cipher_list);
  REQUIRE(o.tls.groups.get() == t.groups);
  REQUIRE(o.tls.sni_override.get() == t.sni_override);
  REQUIRE(o.tls.allow_tls13_without_rekey.get() == t.allow_tls13_without_rekey);

  REQUIRE(sm.uring.slices_per_pass.get() == runtime.uring.slices_per_pass);
  REQUIRE(sm.uring.range_batch_slices.get() == runtime.uring.range_batch_slices);
  REQUIRE(sm.cache.mode.get() == runtime.cache.mode);
  REQUIRE(sm.cache.eviction.get() == runtime.cache.eviction);
  REQUIRE(sm.cache.min_prefetching_budget_fraction.get() ==
          runtime.cache.min_prefetching_budget_fraction);
  REQUIRE(sm.cache.eviction_threshold_fraction.get() == runtime.cache.eviction_threshold_fraction);
  REQUIRE(sm.memory_prefetcher.enable.get() == runtime.memory_prefetcher.enable);
  REQUIRE(sm.memory_prefetcher.num_threads.get() == runtime.memory_prefetcher.num_threads);
  REQUIRE(sm.memory_prefetcher.min_free_fraction.get() ==
          runtime.memory_prefetcher.min_free_fraction);
  REQUIRE(static_cast<std::size_t>(sm.memory_prefetcher.poll_interval.get().count()) ==
          runtime.memory_prefetcher.poll_interval_ms);
  REQUIRE(static_cast<std::size_t>(sm.memory_prefetcher.drain_quiet.get().count()) ==
          runtime.memory_prefetcher.drain_quiet_ms);
  REQUIRE(sm.preparation.interrupt_check_interval.get() ==
          runtime.preparation.interrupt_check_interval);
  REQUIRE(sm.backend.get() == runtime.backend);
  REQUIRE(sm.reserve_rest_reactor_cpus.get() == runtime.reserve_rest_reactor_cpus);
  REQUIRE(sm.httpfs_external_file_cache.get() == runtime.httpfs_external_file_cache);
}
