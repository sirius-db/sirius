// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/option.hpp"
#include "config/size.hpp"
#include "config/validators.hpp"
#include "scan_manager/config.hpp"
#include "scan_manager/preparation_options.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/uring/config.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::config {

namespace detail {

inline const scan_manager::scan_manager_config& scan_manager_defaults()
{
  static const scan_manager::scan_manager_config value{};
  return value;
}

inline const scan_manager::partial_batch_options& partial_batch_defaults()
{
  static const scan_manager::partial_batch_options value{};
  return value;
}

/// The largest MTU an IPv4 TCP packet can declare (65535 + IP and TCP options).
inline constexpr std::int64_t max_min_mtu = 65535 + 60 + 12;

/// keepalive seconds: positive, and within the int cuCascade keeps them in.
inline between<std::chrono::seconds> keepalive_seconds()
{
  return {std::chrono::seconds{1}, std::chrono::seconds{std::numeric_limits<int>::max()}};
}

}  // namespace detail

/// `sirius.executor.scan_manager.uring`: the local-disk IO reactors.
struct uring_options : block_base<uring_options> {
  /// Unset: one scan per pipeline thread, derived when the configuration is resolved; an explicit
  /// 0 keeps local readahead off.
  config_option<std::size_t> n_max_concurrent_scans =
    option<std::size_t>("n_max_concurrent_scans")
      .describe("readahead budget published by the local uring backend")
      .build();
  /// 0 = no cap: fill every free staging slot.
  config_option<std::size_t> slices_per_pass =
    option<std::size_t>("slices_per_pass")
      .validate([](std::size_t v) -> parse_status {
        if (v <= cucascade::io::uring::max_slices_per_pass) { return {}; }
        return parse_status::failure(
          "must be between 0 and " + std::to_string(cucascade::io::uring::max_slices_per_pass) +
          " (0 = no cap: fill every free staging slot), got " + std::to_string(v));
      })
      .fallback(detail::scan_manager_defaults().uring.slices_per_pass)
      .describe("slices a uring reactor reads per loop pass")
      .build();
  /// 0 = no split.
  config_option<std::size_t> range_batch_slices =
    option<std::size_t>("range_batch_slices")
      .fallback(detail::scan_manager_defaults().uring.range_batch_slices)
      .describe("most slices per queue entry of a multi-range read")
      .build();

  auto fields() { return field_list(n_max_concurrent_scans, slices_per_pass, range_batch_slices); }
};

/// `sirius.executor.scan_manager.rest.tls`: TLS for object-store connections. Trust settings belong
/// to the object store configuration, not to this file.
struct tls_options : block_base<tls_options> {
  using ktls_mode     = cucascade::io::tls::ktls_mode;
  using rx_mode       = cucascade::io::tls::user_space_rx;
  using placement     = cucascade::io::tls::recv_placement;
  using version_bound = cucascade::io::tls::version_bound;

  static const cucascade::io::tls::config& defaults()
  {
    return detail::scan_manager_defaults().rest.tls;
  }

  config_option<ktls_mode> ktls = option<ktls_mode>("ktls")
                                    .choices({{"prefer", ktls_mode::prefer},
                                              {"require", ktls_mode::require},
                                              {"disable", ktls_mode::disable}})
                                    .fallback(defaults().ktls)
                                    .describe("kernel TLS offload policy")
                                    .build();
  config_option<rx_mode> user_space_rx =
    option<rx_mode>("user_space_rx")
      .choices({{"record_layer", rx_mode::record_layer}, {"library", rx_mode::library}})
      .fallback(defaults().fallback_rx)
      .describe("user-space TLS receive path")
      .build();
  config_option<placement> recv_placement = option<placement>("recv_placement")
                                              .choices({{"automatic", placement::automatic},
                                                        {"inline", placement::inline_},
                                                        {"iowq", placement::iowq}})
                                              .fallback(defaults().placement)
                                              .describe("where kTLS body decryption runs")
                                              .build();
  config_option<version_bound> min_version =
    option<version_bound>("min_version")
      .choices({{"tls1.2", version_bound::tls1_2}, {"tls1.3", version_bound::tls1_3}})
      .fallback(defaults().min_version)
      .describe("minimum TLS version")
      .build();
  /// auto: the library's highest.
  config_option<std::optional<version_bound>> max_version =
    option<std::optional<version_bound>>("max_version")
      .choices({{"auto", std::nullopt},
                {"tls1.2", version_bound::tls1_2},
                {"tls1.3", version_bound::tls1_3}})
      .fallback(defaults().max_version)
      .describe("maximum TLS version")
      .build();
  config_option<bool> expect_no_pad =
    option<bool>("expect_no_pad")
      .fallback(defaults().expect_no_pad)
      .describe("request the TLS 1.3 kTLS no-padding receive optimization")
      .build();
  config_option<bool> session_cache = option<bool>("session_cache")
                                        .fallback(defaults().session_cache)
                                        .describe("client TLS session cache")
                                        .build();
  config_option<unsigned> session_cache_per_endpoint =
    option<unsigned>("session_cache_per_endpoint")
      .fallback(defaults().session_cache_per_endpoint)
      .describe("cached TLS sessions per endpoint")
      .build();
  /// Bytes after which a connection is replaced; 0 = never.
  config_option<ubytes> max_connection_bytes =
    option<ubytes>("max_connection_bytes")
      .fallback(ubytes(defaults().max_connection_bytes))
      .describe("bytes after which a TLS connection is retired")
      .build();
  config_option<std::string> ciphersuites = option<std::string>("ciphersuites")
                                              .fallback(defaults().ciphersuites)
                                              .describe("TLS 1.3 cipher suites in OpenSSL syntax")
                                              .build();
  config_option<std::string> cipher_list = option<std::string>("cipher_list")
                                             .fallback(defaults().cipher_list)
                                             .describe("TLS 1.2 ciphers in OpenSSL syntax")
                                             .build();
  config_option<std::string> groups = option<std::string>("groups")
                                        .fallback(defaults().groups)
                                        .describe("TLS key-exchange groups")
                                        .build();
  config_option<std::string> sni_override =
    option<std::string>("sni_override")
      .fallback(defaults().sni_override)
      .describe("SNI name sent and verified instead of the endpoint host")
      .build();
  config_option<bool> allow_tls13_without_rekey =
    option<bool>("allow_tls13_without_rekey")
      .fallback(defaults().allow_tls13_without_rekey)
      .describe("allow TLS 1.3 on kernels without kTLS re-key support")
      .build();

  auto fields()
  {
    constexpr auto trust =
      "trust is owned by object_store; set ca_bundle_path through "
      "sirius_config::set_object_store_config() (kept when a SQL secret is selected), or "
      "NIX_SSL_CERT_FILE / SSL_CERT_FILE / SSL_CERT_DIR in the process environment "
      "(a usable NIX_SSL_CERT_FILE takes precedence over SSL_CERT_FILE); SQL secrets carry no CA";
    return field_list(
      removed("verify",
              "trust is owned by object_store; set VERIFY_SSL on the selected CREATE SECRET (TYPE "
              "SIRIUS_S3, ...) or tls_verify through sirius_config::set_object_store_config() "
              "instead"),
      removed("ca_file", trust),
      removed("ca_dir", trust),
      removed("ca_blob", trust),
      ktls,
      user_space_rx,
      recv_placement,
      min_version,
      max_version,
      expect_no_pad,
      session_cache,
      session_cache_per_endpoint,
      max_connection_bytes,
      ciphersuites,
      cipher_list,
      groups,
      sni_override,
      allow_tls13_without_rekey);
  }

  void validate(std::string_view path) const
  {
    if (auto const& max = max_version.get(); max && *max < min_version.get()) {
      throw config_error("'" + join_path(path, max_version.key()) +
                         "': must not be below min_version");
    }
    if (session_cache.get() && session_cache_per_endpoint.get() == 0) {
      throw config_error("'" + join_path(path, session_cache_per_endpoint.key()) +
                         "': must be greater than 0 with the session cache on");
    }
  }
};

/// `sirius.executor.scan_manager.rest`: the object-store (REST) IO reactors. Each key's range is
/// cuCascade's (rest::config::validate()); resolving the configuration runs that check again with
/// the derived values in place.
struct rest_options : block_base<rest_options> {
  static const cucascade::io::rest::config& defaults()
  {
    return detail::scan_manager_defaults().rest;
  }

  // -- scans and requests
  /// Unset: twice the pipeline threads (at least 8), derived when the configuration is resolved.
  config_option<std::size_t> n_max_concurrent_scans =
    option<std::size_t>("n_max_concurrent_scans")
      .describe("readahead budget published by the REST backend")
      .build();
  config_option<std::chrono::seconds> request_timeout =
    option<std::chrono::seconds>("request_timeout_s")
      .validate(non_negative<std::chrono::seconds>{})
      .fallback(std::chrono::seconds(defaults().request_timeout_s))
      .describe("timeout of one control-plane request")
      .build();
  /// Bytes per second; a transfer slower than this for stall_time_s is abandoned.
  config_option<bytes> stall_speed_limit_bytes =
    option<bytes>("stall_speed_limit_bytes")
      .validate(non_negative<bytes>{})
      .fallback(bytes(defaults().stall_speed_limit_bytes))
      .describe("minimum transfer speed before a GET counts as stalled")
      .build();
  config_option<std::chrono::seconds> stall_time =
    option<std::chrono::seconds>("stall_time_s")
      .validate(non_negative<std::chrono::seconds>{})
      .fallback(std::chrono::seconds(defaults().stall_time_s))
      .describe("window of the GET stall detector")
      .build();
  config_option<ubytes> merge_max_gap =
    option<ubytes>("merge_max_gap")
      .fallback(ubytes(defaults().merge_max_gap))
      .describe("largest gap between ranges planned as one read")
      .build();
  config_option<ubytes> min_segment_bytes = option<ubytes>("min_segment_bytes")
                                              .validate(positive<ubytes>{})
                                              .fallback(ubytes(defaults().min_segment_bytes))
                                              .describe("lower bound of the adaptive GET size")
                                              .build();
  config_option<ubytes> max_segment_bytes = option<ubytes>("max_segment_bytes")
                                              .fallback(ubytes(defaults().max_segment_bytes))
                                              .describe("upper bound of the adaptive GET size")
                                              .build();
  config_option<ubytes> footer_probe_bytes =
    option<ubytes>("footer_probe_bytes")
      .validate(between<ubytes>{ubytes{}, ubytes(gibibytes(1))})
      .fallback(ubytes(defaults().footer_probe_bytes))
      .describe("suffix-range window of the Parquet footer probe")
      .build();
  config_option<std::size_t> list_max_matches =
    option<std::size_t>("list_max_matches")
      .validate(positive<std::size_t>{})
      .fallback(defaults().list_max_matches)
      .describe("cap on files a glob or listing may match")
      .build();
  config_option<std::size_t> list_max_scanned = option<std::size_t>("list_max_scanned")
                                                  .validate(positive<std::size_t>{})
                                                  .fallback(defaults().list_max_scanned)
                                                  .describe("cap on objects a LIST sweep may scan")
                                                  .build();

  // -- retry policy
  config_option<std::chrono::milliseconds> retry_backoff_base =
    option<std::chrono::milliseconds>("retry_backoff_base_ms")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(defaults().retry_backoff_base)
      .describe("base of the exponential retry backoff")
      .build();
  config_option<std::chrono::milliseconds> retry_jitter =
    option<std::chrono::milliseconds>("retry_jitter_ms")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(defaults().retry_jitter)
      .describe("random jitter added to each retry backoff")
      .build();
  config_option<std::size_t> max_retry_attempts =
    option<std::size_t>("max_retry_attempts")
      .validate(between<std::size_t>{1, std::size_t{std::numeric_limits<unsigned>::max()}})
      .fallback(defaults().max_retry_attempts)
      .describe("attempt budget for transient request failures")
      .build();
  config_option<std::size_t> max_auth_retry_attempts =
    option<std::size_t>("max_auth_retry_attempts")
      .validate(between<std::size_t>{1, std::size_t{std::numeric_limits<unsigned>::max()}})
      .fallback(defaults().max_auth_retry_attempts)
      .describe("attempt budget for HTTP 403 re-authorization")
      .build();
  config_option<bool> honor_retry_after =
    option<bool>("honor_retry_after")
      .fallback(defaults().honor_retry_after)
      .describe("wait a numeric Retry-After instead of the backoff")
      .build();

  // -- connection pool
  /// Unset: half the queue depth, up to 128 with kernel TLS or 64 without, derived when the
  /// configuration is resolved.
  config_option<std::size_t> max_connections =
    option<std::size_t>("max_connections")
      .validate(between<std::size_t>{1, 4096})
      .mark_production()
      .describe("keep-alive connections per REST reactor")
      .build();
  config_option<std::chrono::seconds> conn_max_age =
    option<std::chrono::seconds>("conn_max_age_s")
      .validate(non_negative<std::chrono::seconds>{})
      .fallback(defaults().conn_max_age)
      .mark_production()
      .describe("idle cutoff of a pooled connection")
      .build();
  config_option<std::chrono::milliseconds> upkeep_interval =
    option<std::chrono::milliseconds>("upkeep_interval_ms")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(defaults().upkeep_interval)
      .describe("period of idle connection upkeep")
      .build();
  config_option<bool> upkeep_http1 = option<bool>("upkeep_http1")
                                       .fallback(defaults().upkeep_http1)
                                       .describe("send upkeep requests on idle pooled connections")
                                       .build();
  config_option<std::chrono::milliseconds> connect_timeout =
    option<std::chrono::milliseconds>("connect_timeout_ms")
      .validate(positive<std::chrono::milliseconds>{})
      .fallback(defaults().connect_timeout)
      .describe("timeout of TCP connect plus TLS handshake")
      .build();
  config_option<std::chrono::seconds> dns_ttl = option<std::chrono::seconds>("dns_ttl_s")
                                                  .validate(positive<std::chrono::seconds>{})
                                                  .fallback(defaults().dns_ttl)
                                                  .describe("lifetime of a DNS answer")
                                                  .build();
  config_option<bool> dns_accumulate = option<bool>("dns_accumulate")
                                         .fallback(defaults().dns_accumulate)
                                         .describe("union fresh DNS answers with previous ones")
                                         .build();
  config_option<std::size_t> dns_max_addresses = option<std::size_t>("dns_max_addresses")
                                                   .validate(between<std::size_t>{1, 4096})
                                                   .fallback(defaults().dns_max_addresses)
                                                   .describe("DNS addresses kept per host")
                                                   .build();
  config_option<std::chrono::seconds> keepalive_idle =
    option<std::chrono::seconds>("keepalive_idle_s")
      .validate(detail::keepalive_seconds())
      .fallback(std::chrono::seconds(defaults().keepalive_idle_s))
      .describe("TCP keepalive idle time before the first probe")
      .build();
  config_option<std::chrono::seconds> keepalive_interval =
    option<std::chrono::seconds>("keepalive_interval_s")
      .validate(detail::keepalive_seconds())
      .fallback(std::chrono::seconds(defaults().keepalive_interval_s))
      .describe("time between TCP keepalive probes")
      .build();
  /// 0 = the kernel's default.
  config_option<int> keepalive_count =
    option<int>("keepalive_count")
      .validate(non_negative<int>{})
      .fallback(defaults().keepalive_count)
      .describe("TCP keepalive probes before a connection is dropped")
      .build();
  /// Unset: the egress NIC's MTU when it uses jumbo frames, else 0 (no check), derived when the
  /// configuration is resolved.
  config_option<ubytes> min_mtu =
    option<ubytes>("min_mtu")
      .validate(between<ubytes>{ubytes{}, ubytes(detail::max_min_mtu)})
      .mark_production()
      .describe("smallest effective MTU a connection may have")
      .build();
  config_option<unsigned> mtu_reject_max_attempts =
    option<unsigned>("mtu_reject_max_attempts")
      .validate(positive<unsigned>{})
      .fallback(defaults().mtu_reject_max_attempts)
      .describe("MTU rejections before a connection is accepted anyway")
      .build();
  config_option<bool> close_with_rst = option<bool>("close_with_rst")
                                         .fallback(defaults().close_with_rst)
                                         .describe("close sockets with an RST instead of a FIN")
                                         .build();

  // -- transport
  config_option<ubytes> receive_buffer_bytes =
    option<ubytes>("receive_buffer_bytes")
      .validate(between<ubytes>{ubytes(kibibytes(16)), ubytes(mebibytes(4))})
      .fallback(ubytes(defaults().receive_buffer_bytes))
      .describe("per-connection receive buffer size")
      .build();
  config_option<std::size_t> queue_depth =
    option<std::size_t>("queue_depth")
      .validate(between<std::size_t>{2, 32768})
      .fallback(defaults().queue_depth)
      .describe("io_uring submission entries per REST reactor")
      .build();
  config_option<cucascade::io::rest::reactor_wait_mode> wait_mode =
    option<cucascade::io::rest::reactor_wait_mode>("wait_mode")
      .choices({{"epoll", cucascade::io::rest::reactor_wait_mode::epoll},
                {"uring", cucascade::io::rest::reactor_wait_mode::uring}})
      .fallback(defaults().wait_mode)
      .describe("how an idle REST reactor blocks")
      .build();
  /// 0 = the kernel's default.
  config_option<unsigned> iowq_max_unbound =
    option<unsigned>("iowq_max_unbound")
      .fallback(defaults().iowq_max_unbound)
      .describe("io-wq unbound-worker cap per reactor ring")
      .build();

  // -- placement
  config_option<std::size_t> fan_out = option<std::size_t>("fan_out")
                                         .validate(positive<std::size_t>{})
                                         .fallback(defaults().fan_out)
                                         .describe("REST reactors one read is split across")
                                         .build();
  /// Empty: the default route's interface.
  config_option<std::string> interface = option<std::string>("interface")
                                           .fallback(defaults().interface)
                                           .describe("egress network interface for REST reactors")
                                           .build();
  config_option<bool> pin_to_irq_cpus = option<bool>("pin_to_irq_cpus")
                                          .fallback(defaults().pin_to_irq_cpus)
                                          .mark_production()
                                          .describe("pin REST reactors to the NIC's IRQ CPUs")
                                          .build();
  config_option<bool> raise_reactor_priority =
    option<bool>("raise_reactor_priority")
      .fallback(defaults().raise_reactor_priority)
      .describe("raise the scheduling priority of pinned REST reactors")
      .build();

  tls_options tls;

  auto fields()
  {
    constexpr auto moved = "removed; use sirius_config::set_object_store_config() instead";
    return field_list(removed("ca_bundle_path", moved),
                      removed("tls_verify", moved),
                      n_max_concurrent_scans,
                      request_timeout,
                      stall_speed_limit_bytes,
                      stall_time,
                      merge_max_gap,
                      min_segment_bytes,
                      max_segment_bytes,
                      footer_probe_bytes,
                      list_max_matches,
                      list_max_scanned,
                      retry_backoff_base,
                      retry_jitter,
                      max_retry_attempts,
                      max_auth_retry_attempts,
                      honor_retry_after,
                      max_connections,
                      conn_max_age,
                      upkeep_interval,
                      upkeep_http1,
                      connect_timeout,
                      dns_ttl,
                      dns_accumulate,
                      dns_max_addresses,
                      keepalive_idle,
                      keepalive_interval,
                      keepalive_count,
                      min_mtu,
                      mtu_reject_max_attempts,
                      close_with_rst,
                      receive_buffer_bytes,
                      queue_depth,
                      wait_mode,
                      iowq_max_unbound,
                      fan_out,
                      interface,
                      pin_to_irq_cpus,
                      raise_reactor_priority,
                      sub("tls", tls));
  }

  /// The rules that do not need derived values; with an unset max_connections, queue_depth is
  /// checked against the derived one when the configuration is resolved.
  void validate(std::string_view path) const
  {
    if (max_segment_bytes.get() < min_segment_bytes.get()) {
      throw config_error("'" + join_path(path, max_segment_bytes.key()) +
                         "': must be at least min_segment_bytes");
    }
    if (auto const* connections = max_connections.get_if();
        connections != nullptr && queue_depth.get() < 2 * *connections) {
      throw config_error("'" + join_path(path, queue_depth.key()) +
                         "': must be at least 2 * max_connections");
    }
  }
};

/// `sirius.executor.scan_manager.cache`: the read cache in front of the IO backends.
struct cache_options : block_base<cache_options> {
  using cache_mode      = cucascade::io::cache::cache_mode;
  using eviction_policy = cucascade::io::cache::eviction_policy;

  /// `sirius` is the deprecated spelling of `cucs`.
  config_option<cache_mode> mode =
    option<cache_mode>("mode")
      .parser([](std::string_view text, cache_mode& out) -> parse_status {
        if (text == "sirius") {
          out = cache_mode::cucs;
          return {};
        }
        return parser<cache_mode>{}(text, out);
      })
      .fallback(detail::scan_manager_defaults().cache.mode)
      .mark_production()
      .describe("which cache the read path goes through")
      .build();
  config_option<eviction_policy> eviction =
    option<eviction_policy>("eviction")
      .fallback(detail::scan_manager_defaults().cache.eviction)
      .mark_production()
      .describe("what retires idle chunks from the prefetching cache")
      .build();
  config_option<double> min_prefetching_budget_fraction =
    option<double>("min_prefetching_budget_fraction")
      .validate(fraction{})
      .fallback(detail::scan_manager_defaults().cache.min_prefetching_budget_fraction)
      .mark_production()
      .describe("floor of the pool reserved for prefetching")
      .build();
  config_option<double> eviction_threshold_fraction =
    option<double>("eviction_threshold_fraction")
      .validate(fraction{})
      .fallback(detail::scan_manager_defaults().cache.eviction_threshold_fraction)
      .mark_production()
      .describe("pinned host memory usage that starts cache eviction")
      .build();

  auto fields()
  {
    return field_list(mode, eviction, min_prefetching_budget_fraction, eviction_threshold_fraction);
  }
};

/// `sirius.executor.scan_manager.memory_prefetcher`: copies cached input toward the GPU while
/// memory is free.
struct memory_prefetcher_options : block_base<memory_prefetcher_options> {
  static const scan_manager::memory_prefetcher_config& defaults()
  {
    return detail::scan_manager_defaults().memory_prefetcher;
  }

  config_option<bool> enable = option<bool>("enable")
                                 .fallback(defaults().enable)
                                 .mark_production()
                                 .describe("background host-to-GPU upload of cached scan splits")
                                 .build();
  config_option<std::size_t> num_threads = option<std::size_t>("num_threads")
                                             .validate(positive<std::size_t>{})
                                             .fallback(defaults().num_threads)
                                             .describe("memory prefetcher worker threads")
                                             .build();
  config_option<double> min_free_fraction =
    option<double>("min_free_fraction")
      .validate(fraction{})
      .fallback(defaults().min_free_fraction)
      .describe("GPU memory share kept free after each prefetch")
      .build();
  config_option<std::chrono::milliseconds> poll_interval =
    option<std::chrono::milliseconds>("poll_interval_ms")
      .validate(positive<std::chrono::milliseconds>{})
      .fallback(std::chrono::milliseconds(defaults().poll_interval_ms))
      .describe("memory prefetcher sweep interval")
      .build();
  config_option<std::chrono::milliseconds> drain_quiet =
    option<std::chrono::milliseconds>("drain_quiet_ms")
      .validate(non_negative<std::chrono::milliseconds>{})
      .fallback(std::chrono::milliseconds(defaults().drain_quiet_ms))
      .describe("quiet time before a connector stops counting as draining")
      .build();

  auto fields()
  {
    return field_list(enable, num_threads, min_free_fraction, poll_interval, drain_quiet);
  }
};

/// `sirius.executor.scan_manager.preparation.partial_batch`: when a partly filled batch is
/// published anyway. Always on.
struct partial_batch_options : block_base<partial_batch_options> {
  config_option<std::size_t> min_fill_percent =
    option<std::size_t>("min_fill_percent")
      .validate(between<std::size_t>{1, 100})
      .fallback(detail::partial_batch_defaults().min_fill_percent)
      .describe("minimum batch fill for early publication")
      .build();
  config_option<std::chrono::milliseconds> min_residence =
    option<std::chrono::milliseconds>("min_residence")
      .validate(positive<std::chrono::milliseconds>{})
      .fallback(detail::partial_batch_defaults().min_residence)
      .describe("minimum age before early batch publication")
      .build();
  config_option<std::chrono::milliseconds> max_residence =
    option<std::chrono::milliseconds>("max_residence")
      .fallback(detail::partial_batch_defaults().max_residence)
      .describe("longest wait before a partial batch is published")
      .build();

  auto fields() { return field_list(min_fill_percent, min_residence, max_residence); }

  void validate(std::string_view path) const
  {
    if (max_residence.get() < min_residence.get()) {
      throw config_error("'" + join_path(path, max_residence.key()) +
                         "': must be at least min_residence");
    }
  }
};

/// `sirius.executor.scan_manager.preparation`: how scans prepare their inputs. Unset limits are
/// resolved from the scan worker count when the scan manager starts.
struct preparation_options : block_base<preparation_options> {
  /// 0 = unlimited.
  config_option<std::size_t> max_inflight_metadata_jobs =
    option<std::size_t>("max_inflight_metadata_jobs")
      .describe("cap on in-flight metadata preparation jobs")
      .build();
  /// 0 = unlimited.
  config_option<std::size_t> max_inflight_dv_jobs =
    option<std::size_t>("max_inflight_dv_jobs")
      .describe("cap on in-flight DV preparation jobs")
      .build();
  config_option<std::size_t> lookahead_batches =
    option<std::size_t>("lookahead_batches")
      .validate(positive<std::size_t>{})
      .describe("per-scan preparation target in scan batches")
      .build();
  config_option<std::size_t> max_control_work =
    option<std::size_t>("max_control_work")
      .validate(positive<std::size_t>{})
      .describe("maximum coordinator actions per iteration")
      .build();
  config_option<std::size_t> drain_quantum =
    option<std::size_t>("drain_quantum")
      .validate(positive<std::size_t>{})
      .describe("maximum coalescing work per result advancement")
      .build();
  /// 0 = a dedicated coordinator per attempt.
  config_option<std::size_t> coordinator_workers =
    option<std::size_t>("coordinator_workers")
      .describe("shared scan coordinator workers per scan manager")
      .build();
  config_option<std::size_t> native_reader_workers =
    option<std::size_t>("native_reader_workers")
      .validate(positive<std::size_t>{})
      .describe("native-reader worker threads per scan manager")
      .build();
  config_option<std::size_t> max_native_readers =
    option<std::size_t>("max_native_readers")
      .validate(positive<std::size_t>{})
      .describe("native-reader slots per scan manager")
      .build();
  config_option<std::chrono::milliseconds> interrupt_check_interval =
    option<std::chrono::milliseconds>("interrupt_check_interval")
      .validate(positive<std::chrono::milliseconds>{})
      .fallback(scan_manager::k_interrupt_check_interval)
      .describe("longest wait between interruption checks")
      .build();
  partial_batch_options partial_batch;

  auto fields()
  {
    return field_list(max_inflight_metadata_jobs,
                      max_inflight_dv_jobs,
                      lookahead_batches,
                      max_control_work,
                      drain_quantum,
                      coordinator_workers,
                      native_reader_workers,
                      max_native_readers,
                      sub("partial_batch", partial_batch),
                      interrupt_check_interval);
  }

  void validate(std::string_view path) const
  {
    // Unset counts take the defaults scan_manager::preparation_config resolves them to.
    scan_manager::preparation_config resolved;
    if (auto const* workers = native_reader_workers.get_if()) {
      resolved.native_reader_workers = *workers;
    }
    if (auto const* readers = max_native_readers.get_if()) {
      resolved.max_native_readers = *readers;
    }
    if (resolved.resolved_native_readers() < resolved.resolved_native_workers()) {
      throw config_error("'" + join_path(path, max_native_readers.key()) +
                         "': must be at least native_reader_workers");
    }
  }
};

/// `sirius.executor.scan_manager.gds`: GPUDirect Storage reads (`backend: gds`).
struct gds_options : block_base<gds_options> {
  static const sirius::io::gds::gds_config& defaults()
  {
    return detail::scan_manager_defaults().gds;
  }

  /// Measured on GPFS: 32 and 64 reads in flight per GPU are fastest, 128 is slower, and at 256 a
  /// cuFile read stopped returning.
  config_option<std::size_t> num_threads =
    option<std::size_t>("num_threads")
      .validate(between<std::size_t>{1, sirius::io::gds::k_max_num_threads})
      .fallback(defaults().num_threads)
      .describe("threads that issue cuFile reads")
      .build();
  config_option<ubytes> task_size = option<ubytes>("task_size")
                                      .validate(positive<ubytes>{})
                                      .fallback(ubytes(defaults().task_size))
                                      .describe("bytes per cuFile read")
                                      .build();
  config_option<bool> require =
    option<bool>("require")
      .fallback(defaults().require)
      .describe("refuse to start, and fail reads, when cuFile would read through POSIX")
      .build();
  config_option<std::chrono::seconds> read_timeout =
    option<std::chrono::seconds>("read_timeout_s")
      .validate(positive<std::chrono::seconds>{})
      .fallback(defaults().read_timeout)
      .describe("seconds a cuFile read may be in flight before the process ends")
      .build();
  /// 0 turns the check off.
  config_option<double> slow_read_ms_per_mib =
    option<double>("slow_read_ms_per_mib")
      .validate(non_negative_finite{})
      .fallback(defaults().slow_read_ms_per_mib)
      .describe("milliseconds per MiB above which reads fail")
      .build();
  /// 0 = no cap.
  config_option<double> max_gb_per_s =
    option<double>("max_gb_per_s")
      .validate(non_negative_finite{})
      .fallback(defaults().max_gb_per_s)
      .describe("most gigabytes per second a process starts reading through cuFile")
      .build();

  auto fields()
  {
    return field_list(
      num_threads, task_size, require, read_timeout, slow_read_ms_per_mib, max_gb_per_s);
  }
};

/// `sirius.executor.scan_manager`: the scan workers and the IO backends behind them.
struct scan_manager_options : block_base<scan_manager_options> {
  /// Unset: the fixed scan-worker default, derived when the configuration is resolved.
  config_option<int> num_threads =
    option<int>("num_threads")
      .validate(positive<int>{})
      .mark_production()
      .describe("scan metadata and Parquet coalescing worker threads")
      .build();
  config_option<std::vector<int>> cpu_affinity =
    option<std::vector<int>>("cpu_affinity")
      .fallback(std::vector<int>{})
      .describe("CPU cores scan manager threads are pinned to")
      .build();
  /// `sirius` is the deprecated spelling of `native`.
  config_option<scan_manager::io_backend> backend =
    option<scan_manager::io_backend>("backend")
      .parser([](std::string_view text, scan_manager::io_backend& out) -> parse_status {
        if (text == "kvikio") {
          return parse_status::failure(
            "backend: kvikio has been removed; use backend: native instead");
        }
        return parser<scan_manager::io_backend>{}(text, out);
      })
      .fallback(detail::scan_manager_defaults().backend)
      .mark_production()
      .describe("IO backend for scan reads")
      .build();
  /// Unset: derived from the CPU count when the configuration is resolved.
  config_option<std::size_t> uring_n_reactors =
    option<std::size_t>("uring_n_reactors")
      .validate(positive<std::size_t>{})
      .describe("io_uring reactors for local-disk reads")
      .build();
  /// Unset: derived from the egress NIC and the CPU count when the configuration is resolved;
  /// 0 lets cuCascade size each pool from the NIC.
  config_option<std::size_t> rest_n_reactors = option<std::size_t>("rest_n_reactors")
                                                 .mark_production()
                                                 .describe("REST reactors for object-store reads")
                                                 .build();
  config_option<bool> reserve_rest_reactor_cpus =
    option<bool>("reserve_rest_reactor_cpus")
      .fallback(detail::scan_manager_defaults().reserve_rest_reactor_cpus)
      .describe("keep other thread pools off REST reactor CPUs")
      .build();
  config_option<bool> httpfs_external_file_cache =
    option<bool>("httpfs_external_file_cache")
      .fallback(detail::scan_manager_defaults().httpfs_external_file_cache)
      .describe("let DuckDB's external file cache keep bytes read through sirius_httpfs")
      .build();
  /// Unset: the cache and the backend decide; 0 turns readahead off.
  config_option<std::size_t> max_readahead_scans =
    option<std::size_t>("max_readahead_scans")
      .mark_production()
      .describe("scans the readahead may keep in flight")
      .build();
  /// Unset: the backend's preference.
  config_option<scan_manager::prefetch_strategy> readahead_strategy =
    option<scan_manager::prefetch_strategy>("readahead_strategy")
      .describe("when the scan readahead issues reads")
      .build();

  uring_options uring;
  rest_options rest;
  cache_options cache;
  memory_prefetcher_options memory_prefetcher;
  preparation_options preparation;
  gds_options gds;

  auto fields()
  {
    return field_list(
      removed("object_store",
              "no longer YAML-loadable; use CREATE SECRET (TYPE SIRIUS_S3, ...) or "
              "sirius_config::set_object_store_config()"),
      removed("kvikio", "removed; remove this block and use backend: native instead"),
      num_threads,
      cpu_affinity,
      backend,
      uring_n_reactors,
      rest_n_reactors,
      reserve_rest_reactor_cpus,
      httpfs_external_file_cache,
      max_readahead_scans,
      readahead_strategy,
      sub("uring", uring),
      sub("rest", rest),
      sub("cache", cache),
      sub("memory_prefetcher", memory_prefetcher),
      sub("preparation", preparation),
      sub("gds", gds));
  }

  void validate(std::string_view path) const
  {
    // An explicit reactor count bounds the fan-out; the default fan-out is clamped instead. An
    // unnamed count is only known once the configuration is resolved, which checks it again.
    auto const* reactors = rest_n_reactors.get_if();
    auto const fan_out   = rest.fan_out.get();
    if (reactors != nullptr && *reactors > 0 && fan_out > *reactors &&
        fan_out != rest_options::defaults().fan_out) {
      throw config_error("'" + join_path(join_path(path, "rest"), rest.fan_out.key()) +
                         "': must be at most rest_n_reactors (" + std::to_string(*reactors) + ")");
    }
  }
};

}  // namespace sirius::config
