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

#include "sirius_config.hpp"

#include "config/user_config.hpp"
#include "exec/config.hpp"
#include "exec/thread_defaults.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "yaml_reader.hpp"

#include <cuda_runtime_api.h>

#include <cucascade/exec/thread_util.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/rest/rest_ioctx.hpp>
#include <cucascade/io/tls/config.hpp>
#include <cucascade/io/uring/config.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/cpu_constraints.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace sirius {

namespace config {

uint64_t derived_default_batch_size()
{
  // cudaGetDeviceCount/Properties honor CUDA_VISIBLE_DEVICES and do not create a context.
  static uint64_t const value = [] {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
      return DEFAULT_BATCH_SIZE;
    }
    uint64_t min_total = 0;
    for (int id = 0; id < device_count; ++id) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, id) != cudaSuccess) { continue; }
      auto const total = static_cast<uint64_t>(prop.totalGlobalMem);
      min_total        = min_total == 0 ? total : std::min(min_total, total);
    }
    if (min_total == 0) { return DEFAULT_BATCH_SIZE; }
    constexpr uint64_t min_batch = 512ULL * 1024 * 1024;       // 512 MiB floor
    constexpr uint64_t max_batch = 5ULL * 1024 * 1024 * 1024;  // 5 GiB ceiling
    return std::clamp(min_total / 40, min_batch, max_batch);   // 2.5%
  }();
  return value;
}

}  // namespace config

#ifdef SIRIUS_DISTRIBUTION_DEFAULTS
bool const telemetry_config::default_enable_quent = false;
#else
bool const telemetry_config::default_enable_quent = true;
#endif

namespace {

struct batch_setting {
  const char* name;
  uint64_t operator_params::* member;
  uint64_t multiplier;
};

constexpr batch_setting batch_settings[] = {
  {"scan_task_batch_size", &operator_params::scan_task_batch_size, 1},
  {"hash_partition_bytes", &operator_params::hash_partition_bytes, 1},
  {"concat_batch_bytes", &operator_params::concat_batch_bytes, 1},
  {"cross_join_task_bytes", &operator_params::cross_join_task_bytes, 1},
  {"sort_sample_bytes", &operator_params::sort_sample_bytes, 1},
  {"max_build_hash_table_bytes", &operator_params::max_build_hash_table_bytes, 8},
  {"max_build_probe_resident_bytes", &operator_params::max_build_probe_resident_bytes, 8}};

}  // namespace

operator_params operator_params::with_batch_size(uint64_t batch_size)
{
  operator_params result{};
  for (const auto& setting : batch_settings) {
    result.*setting.member = batch_size * setting.multiplier;
  }
  return result;
}

static void reject_mutually_exclusive(yaml::reader& reader,
                                      const char* context,
                                      const char* first,
                                      const char* second)
{
  auto const first_value  = reader.optional_node(first);
  auto const second_value = reader.optional_node(second);
  if (first_value && second_value) {
    throw std::runtime_error(std::string(context) + ": '" + first + "' and '" + second +
                             "' are mutually exclusive");
  }
}

// ================ from_yaml for external types ================= //

static void validate_downgrade_fractions(std::string_view scope, double trigger, double stop)
{
  if (stop <= 0.0) {
    throw std::runtime_error(std::string(scope) +
                             ": downgrade_stop_fraction must be greater than zero");
  }
  if (stop >= trigger) {
    throw std::runtime_error(std::string(scope) +
                             ": downgrade_stop_fraction must be less than "
                             "downgrade_trigger_fraction");
  }
}

static void from_yaml(const YAML::Node& node, cucascade::memory::gpu_memory_space_config& opt)
{
  opt.per_stream_reservation = false;  // default to false for sirius
  yaml::reader r(node, "gpu_memory_space");
  r.optional("device_id", opt.device_id);
  r.optional("per_stream_reservation", opt.per_stream_reservation);
  r.optional(
    "reservation_limit_fraction", opt.reservation_limit_fraction, yaml::fraction<double>{});
  r.optional(
    "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, yaml::fraction<double>{});
  r.optional("downgrade_stop_fraction", opt.downgrade_stop_fraction, yaml::fraction<double>{});
  r.optional("memory_capacity", yaml::bytes(opt.memory_capacity));
  r.reject_unknown();
  validate_downgrade_fractions(
    "sirius.space.gpu", opt.downgrade_trigger_fraction, opt.downgrade_stop_fraction);
}

static void from_yaml(const YAML::Node& node, cucascade::memory::host_memory_space_config& opt)
{
  yaml::reader r(node, "host_memory_space");
  r.optional("numa_id", opt.numa_id);
  r.optional(
    "reservation_limit_fraction", opt.reservation_limit_fraction, yaml::fraction<double>{});
  r.optional(
    "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, yaml::fraction<double>{});
  r.optional("downgrade_stop_fraction", opt.downgrade_stop_fraction, yaml::fraction<double>{});
  r.optional("memory_capacity", yaml::bytes(opt.memory_capacity));
  r.optional("block_size", yaml::bytes(opt.block_size));
  r.optional("pool_size", opt.pool_size);
  r.optional("initial_number_pools", opt.initial_number_pools);
  r.reject_unknown();
  validate_downgrade_fractions(
    "sirius.space.host", opt.downgrade_trigger_fraction, opt.downgrade_stop_fraction);
}

static void from_yaml(const YAML::Node& node, cucascade::memory::disk_memory_space_config& opt)
{
  yaml::reader r(node, "disk_memory_space");
  r.optional("disk_id", opt.disk_id);
  r.optional("mount_path", opt.mount_paths);
  r.optional("memory_capacity", yaml::bytes(opt.memory_capacity));
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, exec::thread_pool_config& opt)
{
  yaml::reader r(node, "thread_pool");
  r.optional("num_threads", opt.num_threads, yaml::greater_than<int>{0});
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, creator::task_creator_config& opt)
{
  yaml::reader r(node, "task_creator");
  if (r.has("strategy")) {
    throw std::runtime_error(
      "'sirius.executor.task_creator.strategy': removed; task creation policy is internal and "
      "currently demand-driven; remove this key");
  }
  if (r.has("priority_order")) {
    throw std::runtime_error(
      "'sirius.executor.task_creator.priority_order': removed; scheduling priority is internal "
      "and currently source-first; remove this key");
  }
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{0});
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  r.reject_unknown();
}

// Read counts signed first, so negative YAML values cannot wrap into an unsigned field.
template <typename T>
static void read_count(yaml::reader& r,
                       char const* key,
                       T& out,
                       bool positive     = false,
                       std::uint64_t max = std::numeric_limits<T>::max())
{
  std::int64_t value{};
  r.optional(key, value, [=](auto v) {
    if (v < 0 || (positive && v == 0) || static_cast<std::uint64_t>(v) > max) {
      std::string text = positive ? "must be a positive integer" : "must be a nonnegative integer";
      if (max < static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
        text += " <= " + std::to_string(max);
      }
      throw std::runtime_error(text);
    }
    return true;
  });
  if (r.has_value(key)) { out = static_cast<T>(value); }
}

// Read one of a closed set of spellings; anything else names the accepted ones.
template <typename T>
static void read_choice(yaml::reader& r,
                        char const* key,
                        T& out,
                        std::initializer_list<std::pair<std::string_view, T>> choices)
{
  std::string value;
  r.optional(key, value, [&](auto const& spelling) {
    std::string expected;
    for (auto const& [name, choice] : choices) {
      if (spelling == name) {
        out = choice;
        return true;
      }
      expected += (expected.empty() ? "" : ", ") + std::string{name};
    }
    throw std::runtime_error("invalid enum value '" + spelling + "' (expected " + expected + ")");
  });
}

static void from_yaml(const YAML::Node& node, cucascade::io::tls::config& opt)
{
  namespace tls = cucascade::io::tls;
  yaml::reader r(node, "rest.tls");
  if (r.has("verify")) {
    throw std::runtime_error(
      "'sirius.executor.scan_manager.rest.tls.verify': trust is owned by object_store; set "
      "VERIFY_SSL on the selected CREATE SECRET (TYPE SIRIUS_S3, ...) or tls_verify through "
      "sirius_config::set_object_store_config() instead");
  }
  for (auto const* key : {"ca_file", "ca_dir", "ca_blob"}) {
    if (r.has(key)) {
      throw std::runtime_error(
        "'sirius.executor.scan_manager.rest.tls." + std::string(key) +
        "': trust is owned by object_store; set ca_bundle_path through "
        "sirius_config::set_object_store_config() (kept when a SQL secret is selected), or "
        "NIX_SSL_CERT_FILE / SSL_CERT_FILE / SSL_CERT_DIR in the process environment "
        "(a usable NIX_SSL_CERT_FILE takes precedence over SSL_CERT_FILE); SQL secrets carry no "
        "CA");
    }
  }
  read_choice(r,
              "ktls",
              opt.ktls,
              {{"prefer", tls::ktls_mode::prefer},
               {"require", tls::ktls_mode::require},
               {"disable", tls::ktls_mode::disable}});
  read_choice(
    r,
    "user_space_rx",
    opt.fallback_rx,
    {{"record_layer", tls::user_space_rx::record_layer}, {"library", tls::user_space_rx::library}});
  read_choice(r,
              "recv_placement",
              opt.placement,
              {{"automatic", tls::recv_placement::automatic},
               {"inline", tls::recv_placement::inline_},
               {"iowq", tls::recv_placement::iowq}});
  read_choice(r,
              "min_version",
              opt.min_version,
              {{"tls1.2", tls::version_bound::tls1_2}, {"tls1.3", tls::version_bound::tls1_3}});
  std::optional<tls::version_bound> max_version = opt.max_version;
  read_choice(r,
              "max_version",
              max_version,
              {{"auto", std::optional<tls::version_bound>{}},
               {"tls1.2", std::optional{tls::version_bound::tls1_2}},
               {"tls1.3", std::optional{tls::version_bound::tls1_3}}});
  opt.max_version = max_version;
  r.optional("expect_no_pad", opt.expect_no_pad);
  r.optional("session_cache", opt.session_cache);
  // Ranges and cross-field rules (0 only with the cache off, version ordering) are checked by
  // the rest reader through rest::config::validate().
  read_count(r, "session_cache_per_endpoint", opt.session_cache_per_endpoint);
  r.optional("max_connection_bytes", yaml::bytes(opt.max_connection_bytes));
  r.optional("ciphersuites", opt.ciphersuites);
  r.optional("cipher_list", opt.cipher_list);
  r.optional("groups", opt.groups);
  r.optional("sni_override", opt.sni_override);
  r.optional("allow_tls13_without_rekey", opt.allow_tls13_without_rekey);
  r.reject_unknown();
}

// cuCascade's rest::config::validate() names struct fields; report them under their YAML keys.
static std::string rest_yaml_key(std::string_view field)
{
  static constexpr std::pair<std::string_view, std::string_view> renamed[] = {
    {"upkeep_interval", "upkeep_interval_ms"},
    {"conn_max_age", "conn_max_age_s"},
    {"retry_backoff_base", "retry_backoff_base_ms"},
    {"retry_jitter", "retry_jitter_ms"},
    {"connect_timeout", "connect_timeout_ms"},
    {"dns_ttl", "dns_ttl_s"},
    {"tls.fallback_rx", "tls.user_space_rx"},
    {"tls.placement", "tls.recv_placement"},
  };
  for (auto const& [from, to] : renamed) {
    if (field == from) { return std::string{to}; }
  }
  return std::string{field};
}

// cuCascade's rest::config::validate() on @p opt, its rejection named by YAML key below @p path.
static std::optional<std::string> rest_config_error(cucascade::io::rest::config const& opt,
                                                    std::string_view path)
{
  try {
    opt.validate();
  } catch (std::invalid_argument const& e) {
    std::string_view what = e.what();
    constexpr std::string_view prefix{"rest::config::"};
    if (what.starts_with(prefix)) { what.remove_prefix(prefix.size()); }
    auto const colon = what.find(": ");
    if (colon == std::string_view::npos) {
      return "'" + std::string(path) + "': " + std::string{what};
    }
    return "'" + std::string(path) + "." + rest_yaml_key(what.substr(0, colon)) +
           "': " + std::string{what.substr(colon + 2)};
  }
  return std::nullopt;
}

static void from_yaml(const YAML::Node& node, cucascade::io::rest::config& opt)
{
  namespace rest = cucascade::io::rest;
  yaml::reader r(node, "rest");
  for (auto const* key : {"ca_bundle_path", "tls_verify"}) {
    if (r.has(key)) {
      throw std::runtime_error("'sirius.executor.scan_manager.rest." + std::string(key) +
                               "': removed; use sirius_config::set_object_store_config() instead");
    }
  }
  // This reader owns the YAML surface: names, types (signed reads, so a negative value cannot
  // wrap into an unsigned field), enum spellings and the trust rejections. Value ranges and
  // cross-field rules belong to cuCascade and are enforced once, by validate() below.

  // -- scans and requests ------------------------------------------------------
  read_count(r, "n_max_concurrent_scans", opt.n_max_concurrent_scans);
  // Naming even the struct default must prevent pipeline-width default derivation.
  if (r.has_value("n_max_concurrent_scans")) { opt.n_max_concurrent_scans_explicit = true; }
  r.optional("request_timeout_s", opt.request_timeout_s);
  r.optional("stall_speed_limit_bytes", opt.stall_speed_limit_bytes);
  r.optional("stall_time_s", opt.stall_time_s);
  // Consumed by the logical range planner, independently of physical GET sizing.
  r.optional("merge_max_gap", yaml::bytes(opt.merge_max_gap));
  r.optional("min_segment_bytes", yaml::bytes(opt.min_segment_bytes));
  r.optional("max_segment_bytes", yaml::bytes(opt.max_segment_bytes));
  r.optional("footer_probe_bytes", yaml::bytes(opt.footer_probe_bytes));
  read_count(r, "list_max_matches", opt.list_max_matches);
  read_count(r, "list_max_scanned", opt.list_max_scanned);

  // -- retry policy --------------------------------------------------------------
  r.optional("retry_backoff_base_ms", opt.retry_backoff_base);
  r.optional("retry_jitter_ms", opt.retry_jitter);
  read_count(
    r, "max_retry_attempts", opt.max_retry_attempts, false, std::numeric_limits<unsigned>::max());
  read_count(r,
             "max_auth_retry_attempts",
             opt.max_auth_retry_attempts,
             false,
             std::numeric_limits<unsigned>::max());
  r.optional("honor_retry_after", opt.honor_retry_after);

  // -- connection pool -------------------------------------------------------------
  read_count(r, "max_connections", opt.max_connections);
  r.optional("conn_max_age_s", opt.conn_max_age);
  r.optional("upkeep_interval_ms", opt.upkeep_interval);
  r.optional("upkeep_http1", opt.upkeep_http1);
  r.optional("connect_timeout_ms", opt.connect_timeout);
  r.optional("dns_ttl_s", opt.dns_ttl);
  r.optional("dns_accumulate", opt.dns_accumulate);
  read_count(r, "dns_max_addresses", opt.dns_max_addresses);
  r.optional("keepalive_idle_s", opt.keepalive_idle_s);
  r.optional("keepalive_interval_s", opt.keepalive_interval_s);
  r.optional("keepalive_count", opt.keepalive_count);
  r.optional("min_mtu", yaml::bytes(opt.min_mtu));
  read_count(r, "mtu_reject_max_attempts", opt.mtu_reject_max_attempts);
  r.optional("close_with_rst", opt.close_with_rst);

  // -- transport -----------------------------------------------------------------
  r.optional("receive_buffer_bytes", yaml::bytes(opt.receive_buffer_bytes));
  read_count(r, "queue_depth", opt.queue_depth);
  read_choice(
    r,
    "wait_mode",
    opt.wait_mode,
    {{"epoll", rest::reactor_wait_mode::epoll}, {"uring", rest::reactor_wait_mode::uring}});
  read_count(r, "iowq_max_unbound", opt.iowq_max_unbound);

  // -- placement -------------------------------------------------------------------
  // fan_out vs. rest_n_reactors is checked by the scan_manager reader, which owns the count.
  read_count(r, "fan_out", opt.fan_out);
  r.optional("interface", opt.interface);
  r.optional("pin_to_irq_cpus", opt.pin_to_irq_cpus);
  r.optional("raise_reactor_priority", opt.raise_reactor_priority);

  if (auto n = r.optional_node("tls")) sirius::from_yaml(*n, opt.tls);
  r.reject_unknown();

  // cuCascade's validate() enforces every range and cross-field rule exactly as the reactors
  // will; its errors name struct fields, reported here under their YAML keys.
  if (auto error = rest_config_error(opt, "rest")) { throw std::runtime_error(*error); }
}

// cuCascade does not validate slices_per_pass (a value above the slot count merely acts
// as no cap) and its factories report other config errors only through compiled-out
// logging, so Sirius validates the uring keys here and names the key.
static void from_yaml(const YAML::Node& node, cucascade::io::uring::config& opt)
{
  yaml::reader r(node, "uring");
  {
    // Preserve whether the value was named: an explicit value may equal the
    // struct default, and cuCascade's readahead resolution tells the two apart.
    std::optional<std::size_t> n_max;
    r.optional("n_max_concurrent_scans", n_max);
    if (n_max.has_value()) {
      opt.n_max_concurrent_scans          = *n_max;
      opt.n_max_concurrent_scans_explicit = true;
    }
  }
  {
    // Signed, so a negative value is reported as itself rather than wrapped.
    std::optional<long long> slices;
    r.optional("slices_per_pass", slices);
    if (slices.has_value()) {
      auto const max_slices = static_cast<long long>(cucascade::io::uring::max_slices_per_pass);
      if (*slices < 0 || *slices > max_slices) {
        throw std::runtime_error(
          "'uring.slices_per_pass': must be between 0 and " + std::to_string(max_slices) +
          " (0 = no cap: fill every free staging slot), got " + std::to_string(*slices));
      }
      opt.slices_per_pass = static_cast<std::size_t>(*slices);
    }
  }
  {
    // Signed for the same reason.
    std::optional<long long> batch;
    r.optional("range_batch_slices", batch);
    if (batch.has_value()) {
      if (*batch < 0) {
        throw std::runtime_error(
          "'uring.range_batch_slices': must be 0 or more (0 = no split), got " +
          std::to_string(*batch));
      }
      opt.range_batch_slices = static_cast<std::size_t>(*batch);
    }
  }
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, cucascade::io::cache::config& opt)
{
  yaml::reader r(node, "cache");
  {
    // Read as a string so the deprecated spelling `sirius` (the cache's name before it
    // moved to cuCascade) still selects `cucs`; everything else goes through
    // cuCascade's own parser, with the reader's usual error format.
    std::optional<std::string> mode;
    r.optional("mode", mode);
    if (mode.has_value()) {
      if (*mode == "sirius") {
        opt.mode = cucascade::io::cache::cache_mode::cucs;
      } else if (!cucascade::io::cache::string_to_enum(std::string_view{*mode}, opt.mode)) {
        throw std::runtime_error("'cache.mode': invalid enum value '" + *mode + "'");
      }
    }
  }
  r.optional("eviction", opt.eviction);
  r.optional("min_prefetching_budget_fraction",
             opt.min_prefetching_budget_fraction,
             yaml::fraction<double>{});
  r.optional(
    "eviction_threshold_fraction", opt.eviction_threshold_fraction, yaml::fraction<double>{});
  r.reject_unknown();
  opt.apply_mode();
}

static void from_yaml(const YAML::Node& node, scan_manager::memory_prefetcher_config& opt)
{
  yaml::reader r(node, "memory_prefetcher");
  r.optional("enable", opt.enable);
  r.optional("num_threads", opt.num_threads, yaml::greater_than<std::size_t>{0});
  r.optional("min_free_fraction", opt.min_free_fraction, yaml::fraction<double>{});
  r.optional("poll_interval_ms", opt.poll_interval_ms, yaml::greater_than<std::size_t>{0});
  r.optional("drain_quiet_ms", opt.drain_quiet_ms);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, scan_manager::preparation_config& opt)
{
  yaml::reader r(node, "preparation");
  auto limit = [&](char const* key, std::optional<size_t>& out, bool allow_zero = false) {
    // Parse signed first so negative values cannot wrap into a large size_t.
    std::optional<int64_t> value;
    r.optional(key, value);
    if (!value) return;
    if (*value < 0 || (!allow_zero && *value == 0) ||
        static_cast<uint64_t>(*value) > std::numeric_limits<size_t>::max())
      throw std::runtime_error(std::string("preparation.") + key +
                               (allow_zero ? ": must be a nonnegative integer that fits size_t"
                                           : ": must be a positive integer that fits size_t"));
    out = static_cast<size_t>(*value);
  };
  limit("max_inflight_metadata_jobs", opt.max_inflight_metadata_jobs, true);
  limit("max_inflight_dv_jobs", opt.max_inflight_dv_jobs, true);
  limit("lookahead_batches", opt.lookahead_batches);
  limit("max_control_work", opt.max_control_work);
  limit("drain_quantum", opt.drain_quantum);
  limit("coordinator_workers", opt.coordinator_workers, true);
  limit("native_reader_workers", opt.native_reader_workers);
  limit("max_native_readers", opt.max_native_readers);
  if (opt.resolved_native_readers() < opt.resolved_native_workers())
    throw std::runtime_error(
      "preparation.max_native_readers must be at least native_reader_workers");
  if (auto node = r.optional_node("partial_batch")) {
    auto& policy = opt.partial_batch.emplace();
    yaml::reader partial(*node, "preparation.partial_batch");
    int64_t fill = policy.min_fill_percent;
    partial.optional(
      "min_fill_percent", fill, [](auto value) { return value >= 1 && value <= 100; });
    policy.min_fill_percent = static_cast<size_t>(fill);
    partial.optional("min_residence", policy.min_residence);
    partial.optional("max_residence", policy.max_residence);
    policy.validate();
    partial.reject_unknown();
  }
  r.optional("interrupt_check_interval", opt.interrupt_check_interval, [](auto value) {
    return value.count() > 0;
  });
  r.reject_unknown();
}

static void validate_rest_fan_out(scan_manager::scan_manager_config const& opt)
{
  if (opt.rest_n_reactors > 0 && opt.rest.fan_out > opt.rest_n_reactors &&
      opt.rest.fan_out != cucascade::io::rest::config{}.fan_out) {
    throw std::runtime_error("'rest.fan_out': must be at most rest_n_reactors (" +
                             std::to_string(opt.rest_n_reactors) + ")");
  }
}

static void from_yaml(const YAML::Node& node, scan_manager::scan_manager_config& opt)
{
  yaml::reader r(node, "scan_manager");
  if (r.has("object_store")) {
    throw std::runtime_error(
      "'sirius.executor.scan_manager.object_store': no longer YAML-loadable; use "
      "CREATE SECRET (TYPE SIRIUS_S3, ...) or sirius_config::set_object_store_config()");
  }
  opt.num_threads_explicit = r.has_value("num_threads");
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{0});
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  std::optional<std::string> backend;
  r.optional("backend", backend);
  if (backend) {
    if (*backend == "kvikio") {
      throw std::runtime_error(
        "'scan_manager.backend': backend: kvikio has been removed; "
        "use backend: native instead");
    }
    if (!scan_manager::string_to_enum(*backend, opt.backend)) {
      throw std::runtime_error("'scan_manager.backend': invalid enum value '" + *backend + "'");
    }
  }
  if (r.has("kvikio")) {
    throw std::runtime_error(
      "'scan_manager.kvikio': removed; remove this block and use "
      "backend: native instead");
  }
  r.optional("uring_n_reactors", opt.uring_n_reactors, yaml::greater_than<std::size_t>{0});
  // Per routed REST context, shared by all GPUs; 0 sizes each pool from the egress NIC's IRQs.
  read_count(r, "rest_n_reactors", opt.rest_n_reactors);
  r.optional("reserve_rest_reactor_cpus", opt.reserve_rest_reactor_cpus);
  r.optional("httpfs_external_file_cache", opt.httpfs_external_file_cache);
  r.optional("max_readahead_scans", opt.max_readahead_scans);
  r.optional("readahead_strategy", opt.readahead_strategy);
  if (auto n = r.optional_node("uring")) sirius::from_yaml(*n, opt.uring);
  if (auto n = r.optional_node("rest")) sirius::from_yaml(*n, opt.rest);
  // Same rule as rest_ioctx: the default fan-out is clamped to a smaller pool, any other value
  // must fit it.  An auto-sized pool (0) is only known when the REST ioctx is built.
  if (r.has_value("rest_n_reactors")) { validate_rest_fan_out(opt); }
  if (auto n = r.optional_node("cache")) sirius::from_yaml(*n, opt.cache);
  if (auto n = r.optional_node("memory_prefetcher")) from_yaml(*n, opt.memory_prefetcher);
  if (auto n = r.optional_node("preparation")) from_yaml(*n, opt.preparation);
  r.reject_unknown();
  // Stamped here rather than by the caller: `cache` and the `uring` sub-config
  // it overrides have both been read by now.
  opt.apply_cache_mode();
}

static void from_yaml(const YAML::Node& node, operator_params& opt)
{
  yaml::reader r(node, "operator_params");
  r.optional("enable_late_materialization", opt.enable_late_materialization);
  r.optional("enable_fused_scan_filter", opt.enable_fused_scan_filter);
  r.optional("enable_low_cardinality_groupby", opt.enable_low_cardinality_groupby);
  r.optional("late_materialization_min_boundaries",
             opt.late_materialization_min_boundaries,
             [](int value) { return value >= 0; });
  r.optional("late_materialization_min_value_x_boundaries",
             opt.late_materialization_min_value_x_boundaries,
             [](std::int64_t value) { return value >= 0; });
  r.optional("late_materialization_count_defer", opt.late_materialization_count_defer);
  r.optional("late_materialization_min_value_compressed",
             opt.late_materialization_min_value_compressed,
             [](std::int64_t value) { return value >= 0; });
  r.optional("fused_scan_diagnostics", opt.fused_scan.diagnostics);
  auto const positive_finite = [](double value) { return std::isfinite(value) && value > 0.0; };
  r.optional("fused_scan_max_selectivity", opt.fused_scan.max_selectivity, positive_finite);
  r.optional("fused_scan_full_route_max_selectivity",
             opt.fused_scan.full_route_max_selectivity,
             positive_finite);
  r.optional("fused_scan_index_walk_max_selectivity",
             opt.fused_scan.index_walk_max_selectivity,
             positive_finite);
  r.optional("fused_scan_max_membership_sources",
             opt.fused_scan.max_membership_sources,
             [](std::size_t value) {
               return value <= static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
             });
  r.optional("scan_task_batch_size", yaml::bytes(opt.scan_task_batch_size));
  if (opt.scan_task_batch_size == 0) {
    throw std::runtime_error("'operator_params.scan_task_batch_size': must be greater than zero");
  }
  r.optional("max_sort_partition_bytes", yaml::bytes(opt.max_sort_partition_bytes));
  r.optional("max_sort_partition_memory_fraction",
             opt.max_sort_partition_memory_fraction,
             yaml::fraction<double>{});
  r.optional("hash_partition_bytes", yaml::bytes(opt.hash_partition_bytes));
  if (opt.hash_partition_bytes == 0) {
    throw std::runtime_error("'operator_params.hash_partition_bytes': must be greater than zero");
  }
  r.optional("concat_batch_bytes", yaml::bytes(opt.concat_batch_bytes));
  r.optional("cross_join_task_bytes", yaml::bytes(opt.cross_join_task_bytes));
  r.optional("sort_sample_bytes", yaml::bytes(opt.sort_sample_bytes));
  r.optional("max_build_hash_table_bytes", yaml::bytes(opt.max_build_hash_table_bytes));
  r.optional("max_build_probe_resident_bytes", yaml::bytes(opt.max_build_probe_resident_bytes));
  r.optional("max_broadcast_join_size", yaml::bytes(opt.max_broadcast_join_size));
  r.optional("mark_join_build_switch_ratio",
             opt.mark_join_build_switch_ratio,
             yaml::between<double>{0.0, std::numeric_limits<double>::infinity()});
  if (r.has("enable_runtime_distinct_build_probe")) {
    throw std::runtime_error(
      "'sirius.operator_params.enable_runtime_distinct_build_probe': removed; runtime distinct "
      "build probing is an internal join policy (temporarily disabled pending issue #1600); "
      "remove this key");
  }
  r.optional("enable_dynamic_filter", opt.enable_dynamic_filter);
  r.optional("enable_dynamic_filter_multi_partition", opt.enable_dynamic_filter_multi_partition);
  if (r.has("max_dynamic_filter_bloom_bytes_per_gpu")) {
    throw std::runtime_error(
      "'operator_params.max_dynamic_filter_bloom_bytes_per_gpu': removed; the Bloom budget is "
      "always 4x scan_task_batch_size; remove this key. Set enable_dynamic_filter_multi_partition "
      "to false to disable Bloom accumulation");
  }
  r.optional("enable_dynamic_zone_map_filter", opt.enable_dynamic_zone_map_filter);
  r.optional("dynamic_filter_domain_coverage_threshold",
             opt.dynamic_filter_domain_coverage_threshold,
             config::valid_domain_coverage_threshold{});
  r.optional("dynamic_filter_inlist_max_l2_fraction",
             opt.dynamic_filter_inlist_max_l2_fraction,
             yaml::fraction<double>{});
  r.optional(
    "dynamic_filter_keep_threshold", opt.dynamic_filter_keep_threshold, yaml::fraction<double>{});
  r.optional("enable_pinned_zone_map_pruning", opt.enable_pinned_zone_map_pruning);
  r.optional("enable_decimal_sum_stats_bound", opt.enable_decimal_sum_stats_bound);
  r.optional("enable_compressed_materialization", opt.enable_compressed_materialization);
  r.optional("enable_dictionary_carrier", opt.enable_dictionary_carrier);
  r.optional("enable_dense_count_join", opt.enable_dense_count_join);
  r.optional("enable_delim_aggregate_reuse", opt.enable_delim_aggregate_reuse);
  r.optional("pin_table_prove_unique_keys", opt.prove_unique_keys);
  // 0 is meaningful here: it selects the share of GPU memory space capacity.
  r.optional("pin_table_unique_exact_max_bytes", yaml::bytes(opt.unique_exact_max_bytes));
  r.optional("enable_exists_minmax_rewrite", opt.enable_exists_minmax_rewrite);
  r.optional("enable_partial_aggregate_below_join", opt.enable_partial_aggregate_below_join);
  r.optional("enable_subquery_key_carry", opt.enable_subquery_key_carry);
  r.optional("enable_exists_from_aggregate", opt.enable_exists_from_aggregate);
  r.optional("parquet_footer_statistics_max_remote_reads",
             opt.parquet_footer_statistics_max_remote_reads,
             yaml::greater_than<std::int64_t>{-2});
  r.optional("parquet_footer_statistics_max_local_reads",
             opt.parquet_footer_statistics_max_local_reads,
             yaml::greater_than<std::int64_t>{-2});
  if (r.has("dense_count_join_max_bytes")) {
    throw std::runtime_error(
      "'sirius.operator_params.dense_count_join_max_bytes': removed; dense count-join histogram "
      "sizing is an internal engine policy; remove this key");
  }
  // 0 is meaningful here: it turns the estimate off and leaves sizing to gpus_per_query.
  r.optional("admission_bytes_per_gpu", yaml::bytes(opt.admission_bytes_per_gpu));
  r.optional("avg_variable_column_bytes", yaml::bytes(opt.avg_variable_column_bytes));
  // 0 is not meaningful here: variable-width columns would contribute nothing to the per-row
  // width, so a mixed schema is under-estimated and the query admitted onto too few GPUs.
  if (opt.avg_variable_column_bytes == 0) {
    throw std::runtime_error(
      "'operator_params.avg_variable_column_bytes': must be greater than zero");
  }
  r.optional("enable_runtime_size_estimation", opt.enable_runtime_size_estimation);
  r.optional("use_hw_decompression", opt.use_hw_decompression);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, telemetry_config& opt)
{
  yaml::reader r(node, "telemetry");
  r.optional("enable_quent", opt.enable_quent);
  r.optional("enable_nvtx", opt.enable_nvtx);
  r.optional("output_directory", opt.output_directory, [](std::string const& value) {
    if (!value.empty()) return true;
    throw std::runtime_error("must not be empty");
  });
  r.optional("engine_name", opt.engine_name, [](std::string const& value) {
    if (!value.empty()) return true;
    throw std::runtime_error("must not be empty");
  });
  r.optional("nvtx_injection_lib", opt.nvtx_injection_lib);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, compression_config& opt)
{
  yaml::reader r(node, "compression");
  r.optional("enable_pin_table_compression", opt.enable_pin_table_compression);
  r.optional("min_batch_size_bytes", yaml::bytes(opt.min_batch_size_bytes));
  r.optional("max_compressed_fraction", opt.max_compressed_fraction, [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
  r.optional("min_table_device_fraction", opt.min_table_device_fraction, [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
  r.optional("input_plan_dir", opt.input_plan_dir);
  r.optional("enable_spill_compression", opt.enable_spill_compression);
  r.optional("spill_min_batch_bytes", yaml::bytes(opt.spill_min_batch_bytes));
  r.optional("spill_max_compressed_fraction", opt.spill_max_compressed_fraction, [](double value) {
    return std::isfinite(value) && value >= 0.0;
  });
  r.optional("spill_encode_reserve_fraction", opt.spill_encode_reserve_fraction, [](double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
  });
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, exec::downgrade_executor_config& opt)
{
  yaml::reader r(node, "downgrade");
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{0});
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  r.optional("monitor_period", opt.monitor_period);
  r.optional("copy_chunk_bytes", yaml::bytes(opt.copy_chunk_bytes));
  r.reject_unknown();
}

namespace {

struct topology {
  /// 0 = auto: use every GPU visible to topology discovery (CUDA_VISIBLE_DEVICES-aware).
  std::variant<size_t, std::vector<int>> num_gpus_or_gpu_ids{size_t{0}};
  /// 0 = all active GPUs per query (default). Positive value limits each query to the first
  /// N entries of the sorted active-GPU list, reserving the rest for future concurrent queries.
  int gpus_per_query{0};

  static void from_yaml(const YAML::Node& node, topology& opt)
  {
    yaml::reader r(node, "topology");
    reject_mutually_exclusive(r, "topology", "gpu_ids", "num_gpus");

    if (r.has_value("gpu_ids")) {
      std::vector<int> ids;
      r.optional("gpu_ids", ids);
      if (ids.empty()) {
        throw std::runtime_error("topology.gpu_ids must contain at least one device id");
      }
      if (std::ranges::any_of(ids, [](int id) { return id < 0; })) {
        throw std::runtime_error("topology.gpu_ids must contain only non-negative device ids");
      }
      auto sorted_ids = ids;
      std::ranges::sort(sorted_ids);
      if (std::ranges::adjacent_find(sorted_ids) != sorted_ids.end()) {
        throw std::runtime_error("topology.gpu_ids must not contain duplicate device ids");
      }
      opt.num_gpus_or_gpu_ids = std::move(ids);
    } else {
      long long n = 0;
      r.optional("num_gpus", n);
      if (n < 0) { throw std::runtime_error("topology.num_gpus must be non-negative"); }
      opt.num_gpus_or_gpu_ids = static_cast<size_t>(n);
    }
    // greater_than{-1} is >= 0: a negative count would otherwise silently read as "use all",
    // since the admission path treats any non-positive value as unset.
    r.optional("gpus_per_query", opt.gpus_per_query, yaml::greater_than<int>{-1});
    r.reject_unknown();
  }
};

/// Resolve the configured GPU count: explicit values pass through; 0 (auto) means every
/// discovered GPU, or 1 when discovery found none (it leaves the ctor default in place).
size_t resolve_num_gpus(size_t requested, const cucascade::memory::system_topology_info& hw)
{
  if (requested > 0) { return requested; }
  return hw.num_gpus > 0 ? static_cast<size_t>(hw.num_gpus) : size_t{1};
}

struct gpu_mem_config {
  std::variant<double, std::uint64_t> usage_limit{0.95};
  std::variant<double, std::uint64_t> reservation_limit{1.0};
  double downgrade_trigger_fraction{0.8};
  double downgrade_stop_fraction{0.6};

  static void from_yaml(const YAML::Node& node, gpu_mem_config& opt)
  {
    yaml::reader r(node, "memory.gpu");
    // usage_limit: fraction (double) or absolute bytes — mutually exclusive keys
    std::optional<std::uint64_t> usage_bytes;
    double usage_frac = 0.95;
    reject_mutually_exclusive(r, "memory.gpu", "usage_limit_bytes", "usage_limit_fraction");
    r.optional("usage_limit_bytes", yaml::bytes(usage_bytes));
    r.optional("usage_limit_fraction", usage_frac, yaml::fraction<double>{});
    opt.usage_limit = usage_bytes ? std::variant<double, std::uint64_t>{*usage_bytes}
                                  : std::variant<double, std::uint64_t>{usage_frac};
    // reservation_limit: fraction or absolute bytes
    std::optional<std::uint64_t> res_bytes;
    double res_frac = 1.0;
    reject_mutually_exclusive(
      r, "memory.gpu", "reservation_limit_bytes", "reservation_limit_fraction");
    r.optional("reservation_limit_bytes", yaml::bytes(res_bytes));
    r.optional("reservation_limit_fraction", res_frac, yaml::fraction<double>{});
    opt.reservation_limit = res_bytes ? std::variant<double, std::uint64_t>{*res_bytes}
                                      : std::variant<double, std::uint64_t>{res_frac};
    r.optional(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, yaml::fraction<double>{});
    r.optional("downgrade_stop_fraction", opt.downgrade_stop_fraction, yaml::fraction<double>{});
    r.reject_unknown();
    validate_downgrade_fractions(
      "sirius.memory.gpu", opt.downgrade_trigger_fraction, opt.downgrade_stop_fraction);
  }

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    if (std::holds_alternative<double>(usage_limit)) {
      auto const fraction = std::get<double>(usage_limit);
      // YAML permits zero; the configurator's fraction setter requires a positive value.
      if (fraction == 0.0) {
        builder.set_gpu_usage_limit(0);
      } else {
        builder.set_usage_limit_ratio_per_gpu(fraction);
      }
    } else {
      builder.set_gpu_usage_limit(std::get<std::uint64_t>(usage_limit));
    }
    if (std::holds_alternative<double>(reservation_limit)) {
      builder.set_reservation_fraction_per_gpu(std::get<double>(reservation_limit));
    } else {
      builder.set_reservation_limit_per_gpu(std::get<std::uint64_t>(reservation_limit));
    }
    builder.set_downgrade_fractions_per_gpu(downgrade_trigger_fraction, downgrade_stop_fraction);
    // Keep the high-level path on Sirius's default. The low-level
    // space.gpu[] replacement surface retains the diagnostic per-stream control.
    builder.track_reservation_per_stream(false);
  }
};

struct host_mem_config {
  // Default to 90% of each backing NUMA node's RAM, or explicit bytes per NUMA node.
  std::variant<double, std::uint64_t> capacity{0.9};
  std::variant<double, std::uint64_t> reservation_limit{1.0};
  // stop < trigger < reservation_limit, like the GPU tier. Host->disk eviction still
  // needs a configured downgrade_root_dirs; without one the executor warns and skips.
  double downgrade_trigger_fraction{0.9};
  double downgrade_stop_fraction{0.8};
  std::size_t block_size{cucascade::memory::default_block_size};
  std::size_t pool_size{cucascade::memory::default_pool_size};
  // Omitted: 10% of NUMA RAM in whole pools, capped by the host capacity.
  std::optional<std::size_t> initial_number_pools;

  static void from_yaml(const YAML::Node& node, host_mem_config& opt)
  {
    yaml::reader r(node, "memory.host");
    // capacity: fraction of node RAM (double) or absolute bytes per node — mutually exclusive keys
    std::optional<std::uint64_t> cap_bytes;
    std::optional<double> cap_frac;
    reject_mutually_exclusive(r, "memory.host", "capacity_bytes", "capacity_fraction");
    r.optional("capacity_bytes", yaml::bytes(cap_bytes));
    r.optional("capacity_fraction", cap_frac);
    if (cap_frac && !(*cap_frac > 0.0 && *cap_frac <= 1.0)) {
      throw std::runtime_error("memory.host.capacity_fraction: must be in (0.0, 1.0], got " +
                               std::to_string(*cap_frac));
    }
    if (cap_bytes) {
      opt.capacity = *cap_bytes;
    } else if (cap_frac) {
      opt.capacity = *cap_frac;
    }
    std::optional<std::uint64_t> res_bytes;
    double res_frac = 1.0;
    reject_mutually_exclusive(
      r, "memory.host", "reservation_limit_bytes", "reservation_limit_fraction");
    r.optional("reservation_limit_bytes", yaml::bytes(res_bytes));
    r.optional("reservation_limit_fraction", res_frac, yaml::fraction<double>{});
    opt.reservation_limit = res_bytes ? std::variant<double, std::uint64_t>{*res_bytes}
                                      : std::variant<double, std::uint64_t>{res_frac};
    r.optional(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, yaml::fraction<double>{});
    r.optional("downgrade_stop_fraction", opt.downgrade_stop_fraction, yaml::fraction<double>{});
    r.optional("block_size", yaml::bytes(opt.block_size));
    r.optional("pool_size", opt.pool_size);
    r.optional("initial_number_pools", opt.initial_number_pools);
    r.reject_unknown();
    validate_downgrade_fractions(
      "sirius.memory.host", opt.downgrade_trigger_fraction, opt.downgrade_stop_fraction);
  }

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    // cucascade builds one numa_region_pinned_host_memory_resource per
    // distinct NUMA node when the configurator sees this call. Relied upon by
    // SiriusContext::initialize() which asserts host_spaces.size() ==
    // topology.num_numa_nodes on the default path. YAML configs may override
    // by explicitly setting per-space numa_id.
    builder.use_numa_id_as_host_id();
    if (std::holds_alternative<double>(reservation_limit)) {
      builder.set_reservation_fraction_per_numa_region(std::get<double>(reservation_limit));
    } else {
      builder.set_reservation_limit_per_numa_region(std::get<std::uint64_t>(reservation_limit));
    }
    builder.set_downgrade_fractions_per_numa_region(downgrade_trigger_fraction,
                                                    downgrade_stop_fraction);
    if (std::holds_alternative<double>(capacity)) {
      builder.set_usage_limit_ratio_per_numa_region(std::get<double>(capacity));
    } else {
      builder.set_per_numa_region_capacity(std::get<std::uint64_t>(capacity));
    }
    // cuCascade maps (chunk_size, block_size, initial_block_count) onto the host config's
    // (block_size, pool_size, initial_number_pools). Its setter requires a positive count.
    // build() only creates configs; resolve() replaces this temporary count before allocation.
    builder.set_host_pool_features(block_size, pool_size, 1);
  }
};

struct disk_mem_config {
  int id{0};
  std::size_t capacity_bytes{1024UL << 30};  // 1TB
  std::string downgrade_root_dirs;

  static void from_yaml(const YAML::Node& node, disk_mem_config& opt)
  {
    yaml::reader r(node, "memory.disk");
    r.optional("disk_id", opt.id);
    r.optional("capacity_bytes", yaml::bytes(opt.capacity_bytes));
    r.optional("downgrade_root_dirs", opt.downgrade_root_dirs);
    r.reject_unknown();
  }

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    if (downgrade_root_dirs.empty() || capacity_bytes == 0) { return; }
    builder.set_disk_mounting_point(id, capacity_bytes, downgrade_root_dirs);
  }
};

// Helper: read a vector using file-local from_yaml overloads
template <typename T>
void read_yaml_vec(const YAML::Node& node, std::vector<T>& out)
{
  if (!node.IsSequence()) { throw std::runtime_error("expected a sequence"); }
  for (const auto& item : node) {
    T val{};
    from_yaml(item, val);
    out.push_back(std::move(val));
  }
}

uint64_t effective_default_batch_size(
  const std::vector<cucascade::memory::memory_space_config>& memory_space_configs)
{
  std::optional<uint64_t> min_gpu_capacity;
  for (auto const& space : memory_space_configs) {
    auto const* gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space);
    if (gpu == nullptr || gpu->memory_capacity == 0) { continue; }
    auto const capacity = static_cast<uint64_t>(gpu->memory_capacity);
    min_gpu_capacity    = min_gpu_capacity ? std::min(*min_gpu_capacity, capacity) : capacity;
  }
  if (!min_gpu_capacity) { return config::derived_default_batch_size(); }

  // Keep the existing 2.5% policy, but apply it to an explicitly configured
  // effective capacity. The one-byte floor keeps hash_partition_bytes valid
  // for even a pathological test configuration; the physical default retains
  // the existing 5 GiB ceiling.
  auto const effective_relative = std::max<uint64_t>(1, *min_gpu_capacity / 40);
  return std::min(config::derived_default_batch_size(), effective_relative);
}

operator_params operator_defaults_for(
  const std::vector<cucascade::memory::memory_space_config>& memory_space_configs,
  bool use_effective_capacity)
{
  return operator_params::with_batch_size(use_effective_capacity
                                            ? effective_default_batch_size(memory_space_configs)
                                            : config::derived_default_batch_size());
}

}  // namespace

// ================ sirius_config ================= //

struct parsed_sirius_config::Impl {
  sirius_config settings;
  topology topo;
  gpu_mem_config gpu_cfg;
  host_mem_config host_cfg;
  disk_mem_config disk_cfg;
  std::filesystem::path source_path;
  bool use_effective_gpu_capacity = false;
  struct {
    bool pipeline{}, task_creator{}, downgrade{}, scan_manager{}, uring{}, rest{};
  } explicit_threads;
  /// Whether the YAML named scan_manager.rest.min_mtu, an explicit 0 included.
  bool explicit_rest_min_mtu = false;
  /// Whether the YAML named scan_manager.rest.max_connections.
  bool explicit_rest_max_connections = false;
  std::vector<uint64_t operator_params::*> explicit_batch_settings;
};

parsed_sirius_config::parsed_sirius_config() : impl_(std::make_shared<Impl>()) {}
parsed_sirius_config::parsed_sirius_config(std::shared_ptr<const Impl> impl)
  : impl_(std::move(impl))
{
}

const operator_params& parsed_sirius_config::get_operator_params() const noexcept
{
  return impl_->settings.get_operator_params();
}
output::parquet_compression parsed_sirius_config::parquet_output_compression() const noexcept
{
  return impl_->settings.parquet_output_compression();
}
const compression_config& parsed_sirius_config::get_compression_config() const noexcept
{
  return impl_->settings.get_compression_config();
}
const telemetry_config& parsed_sirius_config::get_telemetry_config() const noexcept
{
  return impl_->settings.get_telemetry_config();
}
const scan_manager::scan_manager_config& parsed_sirius_config::get_scan_manager_config()
  const noexcept
{
  return impl_->settings.get_scan_manager_config();
}

namespace {
// Compatibility for internal callers explicitly requesting a resolved configuration.
// Runtime entry points instead let SiriusContext discover hardware after telemetry starts.
cucascade::memory::system_topology_info discover_configuration_topology()
{
  cucascade::memory::topology_discovery discovery;
  if (!discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                          /*with_runtime_attributes=*/true)) {
    throw std::runtime_error("Failed to discover hardware topology");
  }
  return discovery.get_topology();
}
}  // namespace

void sirius_config::apply_defaults()
{
  *this = parsed_sirius_config{}.resolve(discover_configuration_topology());
}

parsed_sirius_config parsed_sirius_config::from_file(const std::filesystem::path& path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path.string());
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception& e) {
    throw configuration_input_error(path.string() + ": " + e.what());
  }
  return from_node(root, path);
}

void sirius_config::load_from_file(const std::filesystem::path& config_path)
{
  auto resolved =
    parsed_sirius_config::from_file(config_path).resolve(discover_configuration_topology());
  // Object-store settings are supplied programmatically, never through YAML.
  resolved.set_object_store_config(_scan_manager_config.object_store);
  *this = std::move(resolved);
}

parsed_sirius_config parsed_sirius_config::from_node(const YAML::Node& root,
                                                     const std::filesystem::path& source_path)
try {
  auto impl                             = std::make_shared<Impl>();
  impl->source_path                     = source_path;
  auto& settings                        = impl->settings;
  auto& topo                            = impl->topo;
  auto& gpu_cfg                         = impl->gpu_cfg;
  auto& host_cfg                        = impl->host_cfg;
  auto& disk_cfg                        = impl->disk_cfg;
  bool high_level_memory_configured     = false;
  bool explicit_high_level_gpu_capacity = false;
  std::optional<YAML::Node> operator_node;
  std::vector<cucascade::memory::gpu_memory_space_config> gpu_space_configs;
  std::vector<cucascade::memory::host_memory_space_config> host_space_configs;
  std::vector<cucascade::memory::disk_memory_space_config> disk_space_configs;

  yaml::reader top(root);
  auto sirius_node = top.optional_node("sirius");
  top.reject_unknown();

  if (!sirius_node) { throw std::runtime_error("missing top-level 'sirius' key"); }

  yaml::reader r(*sirius_node, "sirius");

  // Topology
  r.optional("topology", topo);
  settings._gpus_per_query = topo.gpus_per_query;

  // High-level memory config (mutually exclusive with space config)
  if (auto mem_node = r.optional_node("memory")) {
    yaml::reader mr(*mem_node, "sirius.memory");
    if (auto n = mr.optional_node("gpu")) {
      high_level_memory_configured = true;
      yaml::reader gpu_reader(*n, "sirius.memory.gpu");
      explicit_high_level_gpu_capacity =
        gpu_reader.has_value("usage_limit_bytes") || gpu_reader.has_value("usage_limit_fraction");
      gpu_mem_config::from_yaml(*n, gpu_cfg);
    }
    if (auto n = mr.optional_node("host")) {
      high_level_memory_configured = true;
      host_mem_config::from_yaml(*n, host_cfg);
    }
    if (auto n = mr.optional_node("disk")) {
      high_level_memory_configured = true;
      disk_mem_config::from_yaml(*n, disk_cfg);
    }
    mr.reject_unknown();
  }

  r.optional(
    "max_concurrent_queries", settings._max_concurrent_queries, yaml::greater_than<int>{0});
  // Executors
  if (auto exec_node = r.optional_node("executor")) {
    yaml::reader er(*exec_node, "sirius.executor");
    if (auto n = er.optional_node("task_creator")) {
      impl->explicit_threads.task_creator = yaml::reader(*n).has_value("num_threads");
      from_yaml(*n, settings._task_creator_config);
    }
    if (auto n = er.optional_node("scan_manager")) {
      yaml::reader counts(*n);
      impl->explicit_threads.scan_manager = counts.has_value("num_threads");
      impl->explicit_threads.uring        = counts.has_value("uring_n_reactors");
      impl->explicit_threads.rest         = counts.has_value("rest_n_reactors");
      if (auto rest = counts.optional_node("rest")) {
        yaml::reader rest_keys(*rest);
        impl->explicit_rest_min_mtu         = rest_keys.has_value("min_mtu");
        impl->explicit_rest_max_connections = rest_keys.has_value("max_connections");
      }
      from_yaml(*n, settings._scan_manager_config);
    }
    if (auto n = er.optional_node("pipeline")) {
      impl->explicit_threads.pipeline = yaml::reader(*n).has_value("num_threads");
      from_yaml(*n, settings._gpu_pipeline_executor_config);
    }
    if (auto n = er.optional_node("downgrade")) {
      impl->explicit_threads.downgrade = yaml::reader(*n).has_value("num_threads");
      from_yaml(*n, settings._downgrade_executor_config);
    }
    er.reject_unknown();
  }

  // Parse operator settings below and record which batch sizes are explicit,
  // so hardware resolution only fills in omitted values.
  operator_node = r.optional_node("operator_params");

  // Telemetry
  if (auto n = r.optional_node("telemetry")) { sirius::from_yaml(*n, settings._telemetry_config); }

  // Compression
  if (auto output_node = r.optional_node("output")) {
    yaml::reader output_reader(*output_node, "sirius.output");
    if (auto parquet_node = output_reader.optional_node("parquet")) {
      yaml::reader parquet_reader(*parquet_node, "sirius.output.parquet");
      std::string codec = "snappy";
      parquet_reader.optional("compression", codec);
      settings._parquet_output_compression = output::parse_compression(codec);
      parquet_reader.reject_unknown();
    }
    output_reader.reject_unknown();
  }
  if (auto n = r.optional_node("compression")) {
    sirius::from_yaml(*n, settings._compression_config);
  }

  // Explicit space configs (low-level API)
  if (auto space_node = r.optional_node("space")) {
    yaml::reader sr(*space_node, "sirius.space");
    if (auto n = sr.optional_node("gpu")) read_yaml_vec(*n, gpu_space_configs);
    if (auto n = sr.optional_node("host")) read_yaml_vec(*n, host_space_configs);
    if (auto n = sr.optional_node("disk")) read_yaml_vec(*n, disk_space_configs);
    sr.reject_unknown();
  }

  bool const explicit_space_configured =
    !gpu_space_configs.empty() || !host_space_configs.empty() || !disk_space_configs.empty();
  if (high_level_memory_configured && explicit_space_configured) {
    throw std::runtime_error(
      "sirius.memory and non-empty sirius.space lists are mutually exclusive");
  }

  r.reject_unknown();

  // Build memory space configs
  std::copy(gpu_space_configs.begin(),
            gpu_space_configs.end(),
            std::back_inserter(settings._memory_space_configs));
  std::copy(host_space_configs.begin(),
            host_space_configs.end(),
            std::back_inserter(settings._memory_space_configs));
  std::copy(disk_space_configs.begin(),
            disk_space_configs.end(),
            std::back_inserter(settings._memory_space_configs));

  bool const explicit_low_level_gpu_capacity =
    !gpu_space_configs.empty() &&
    std::ranges::all_of(gpu_space_configs, [](auto const& gpu) { return gpu.memory_capacity > 0; });
  impl->use_effective_gpu_capacity = settings._memory_space_configs.empty()
                                       ? explicit_high_level_gpu_capacity
                                       : explicit_low_level_gpu_capacity;
  if (operator_node) {
    sirius::from_yaml(*operator_node, settings._operator_params);
    yaml::reader r(*operator_node, "operator_params");
    for (const auto& setting : batch_settings) {
      if (r.has_value(setting.name)) { impl->explicit_batch_settings.push_back(setting.member); }
    }
  }
  return parsed_sirius_config(std::move(impl));
} catch (const std::bad_alloc&) {
  throw;
} catch (const std::exception& e) {
  throw configuration_input_error(source_path.empty() ? e.what()
                                                      : source_path.string() + ": " + e.what());
}

parsed_sirius_config parsed_sirius_config::from_user_config(const config::user_config& user)
try {
  // The rules across blocks, also for a configuration built in code rather than read.
  user.validate(config::user_config_root);

  auto impl      = std::make_shared<Impl>();
  auto& settings = impl->settings;
  auto& named    = impl->explicit_threads;

  // A bytes | fraction pair as the runtime keeps it.
  auto const bytes_or_fraction = [](auto const& option) -> std::variant<double, std::uint64_t> {
    if (auto const* bytes = option.template get_if<config::ubytes>()) { return bytes->count(); }
    return option.template get<double>();
  };
  auto const assign_if_set = [](auto const& option, auto& target) {
    if (auto const* value = option.get_if()) { target = *value; }
  };

  // -- topology
  if (auto const* ids = user.topology.gpus.get_if<std::vector<int>>()) {
    impl->topo.num_gpus_or_gpu_ids = *ids;
  } else {
    impl->topo.num_gpus_or_gpu_ids = user.topology.gpus.get<std::size_t>();
  }
  impl->topo.gpus_per_query = user.topology.gpus_per_query.get();
  settings._gpus_per_query  = impl->topo.gpus_per_query;

  // -- memory tiers
  auto const& gpu                           = user.memory.gpu;
  impl->gpu_cfg.usage_limit                 = bytes_or_fraction(gpu.usage_limit);
  impl->gpu_cfg.reservation_limit           = bytes_or_fraction(gpu.reservation_limit);
  impl->gpu_cfg.downgrade_trigger_fraction  = gpu.downgrade_trigger_fraction.get();
  impl->gpu_cfg.downgrade_stop_fraction     = gpu.downgrade_stop_fraction.get();
  auto const& host                          = user.memory.host;
  impl->host_cfg.capacity                   = bytes_or_fraction(host.capacity);
  impl->host_cfg.reservation_limit          = bytes_or_fraction(host.reservation_limit);
  impl->host_cfg.downgrade_trigger_fraction = host.downgrade_trigger_fraction.get();
  impl->host_cfg.downgrade_stop_fraction    = host.downgrade_stop_fraction.get();
  impl->host_cfg.block_size                 = host.block_size.get().count();
  impl->host_cfg.pool_size                  = host.pool_size.get();
  if (auto const* pools = host.initial_number_pools.get_if()) {
    impl->host_cfg.initial_number_pools = *pools;
  }
  auto const& disk                   = user.memory.disk;
  impl->disk_cfg.id                  = disk.disk_id.get();
  impl->disk_cfg.capacity_bytes      = disk.capacity_bytes.get().count();
  impl->disk_cfg.downgrade_root_dirs = disk.downgrade_root_dirs.get();
  settings._max_concurrent_queries   = user.max_concurrent_queries.get();

  // -- executors
  auto const& creator = user.executor.task_creator;
  assign_if_set(creator.num_threads, settings._task_creator_config.thread_pool.num_threads);
  settings._task_creator_config.thread_pool.cpu_affinity_list = creator.cpu_affinity.get();
  named.task_creator                                          = creator.num_threads.user_defined();

  auto const& pipeline = user.executor.pipeline;
  assign_if_set(pipeline.num_threads, settings._gpu_pipeline_executor_config.num_threads);
  named.pipeline = pipeline.num_threads.user_defined();

  auto const& downgrade = user.executor.downgrade;
  auto& downgrade_cfg   = settings._downgrade_executor_config;
  assign_if_set(downgrade.num_threads, downgrade_cfg.thread_pool.num_threads);
  downgrade_cfg.thread_pool.cpu_affinity_list = downgrade.cpu_affinity.get();
  downgrade_cfg.monitor_period                = downgrade.monitor_period.get();
  downgrade_cfg.copy_chunk_bytes              = downgrade.copy_chunk_bytes.get().count();
  named.downgrade                             = downgrade.num_threads.user_defined();

  // -- scan manager
  auto const& sm = user.executor.scan_manager;
  auto& sm_cfg   = settings._scan_manager_config;
  assign_if_set(sm.num_threads, sm_cfg.thread_pool.num_threads);
  sm_cfg.thread_pool.cpu_affinity_list = sm.cpu_affinity.get();
  sm_cfg.backend                       = sm.backend.get();
  assign_if_set(sm.uring_n_reactors, sm_cfg.uring_n_reactors);
  assign_if_set(sm.rest_n_reactors, sm_cfg.rest_n_reactors);
  sm_cfg.reserve_rest_reactor_cpus = sm.reserve_rest_reactor_cpus.get();
  if (auto const* scans = sm.max_readahead_scans.get_if()) { sm_cfg.max_readahead_scans = *scans; }
  if (auto const* strategy = sm.readahead_strategy.get_if()) {
    sm_cfg.readahead_strategy = *strategy;
  }
  named.scan_manager = sm.num_threads.user_defined();
  named.uring        = sm.uring_n_reactors.user_defined();
  named.rest         = sm.rest_n_reactors.user_defined();

  auto& uring = sm_cfg.uring;
  if (auto const* scans = sm.uring.n_max_concurrent_scans.get_if()) {
    uring.n_max_concurrent_scans          = *scans;
    uring.n_max_concurrent_scans_explicit = true;
  }
  uring.slices_per_pass    = sm.uring.slices_per_pass.get();
  uring.range_batch_slices = sm.uring.range_batch_slices.get();

  auto const& r = sm.rest;
  auto& rest    = sm_cfg.rest;
  if (auto const* scans = r.n_max_concurrent_scans.get_if()) {
    rest.n_max_concurrent_scans          = *scans;
    rest.n_max_concurrent_scans_explicit = true;
  }
  rest.request_timeout_s       = static_cast<long>(r.request_timeout.get().count());
  rest.stall_speed_limit_bytes = static_cast<long>(r.stall_speed_limit_bytes.get().count());
  rest.stall_time_s            = static_cast<long>(r.stall_time.get().count());
  rest.merge_max_gap           = r.merge_max_gap.get().count();
  rest.min_segment_bytes       = r.min_segment_bytes.get().count();
  rest.max_segment_bytes       = r.max_segment_bytes.get().count();
  rest.footer_probe_bytes      = r.footer_probe_bytes.get().count();
  rest.list_max_matches        = r.list_max_matches.get();
  rest.list_max_scanned        = r.list_max_scanned.get();
  rest.retry_backoff_base      = r.retry_backoff_base.get();
  rest.retry_jitter            = r.retry_jitter.get();
  rest.max_retry_attempts      = r.max_retry_attempts.get();
  rest.max_auth_retry_attempts = r.max_auth_retry_attempts.get();
  rest.honor_retry_after       = r.honor_retry_after.get();
  assign_if_set(r.max_connections, rest.max_connections);
  rest.conn_max_age      = r.conn_max_age.get();
  rest.upkeep_interval   = r.upkeep_interval.get();
  rest.upkeep_http1      = r.upkeep_http1.get();
  rest.connect_timeout   = r.connect_timeout.get();
  rest.dns_ttl           = r.dns_ttl.get();
  rest.dns_accumulate    = r.dns_accumulate.get();
  rest.dns_max_addresses = r.dns_max_addresses.get();
  // Validated to fit an int when read.
  rest.keepalive_idle_s     = static_cast<int>(r.keepalive_idle.get().count());
  rest.keepalive_interval_s = static_cast<int>(r.keepalive_interval.get().count());
  rest.keepalive_count      = r.keepalive_count.get();
  if (auto const* mtu = r.min_mtu.get_if()) { rest.min_mtu = mtu->count(); }
  rest.mtu_reject_max_attempts        = r.mtu_reject_max_attempts.get();
  rest.close_with_rst                 = r.close_with_rst.get();
  rest.receive_buffer_bytes           = r.receive_buffer_bytes.get().count();
  rest.queue_depth                    = r.queue_depth.get();
  rest.wait_mode                      = r.wait_mode.get();
  rest.iowq_max_unbound               = r.iowq_max_unbound.get();
  rest.fan_out                        = r.fan_out.get();
  rest.interface                      = r.interface.get();
  rest.pin_to_irq_cpus                = r.pin_to_irq_cpus.get();
  rest.raise_reactor_priority         = r.raise_reactor_priority.get();
  impl->explicit_rest_min_mtu         = r.min_mtu.user_defined();
  impl->explicit_rest_max_connections = r.max_connections.user_defined();

  auto const& t                       = r.tls;
  rest.tls.ktls                       = t.ktls.get();
  rest.tls.fallback_rx                = t.user_space_rx.get();
  rest.tls.placement                  = t.recv_placement.get();
  rest.tls.min_version                = t.min_version.get();
  rest.tls.max_version                = t.max_version.get();
  rest.tls.expect_no_pad              = t.expect_no_pad.get();
  rest.tls.session_cache              = t.session_cache.get();
  rest.tls.session_cache_per_endpoint = t.session_cache_per_endpoint.get();
  rest.tls.max_connection_bytes       = t.max_connection_bytes.get().count();
  rest.tls.ciphersuites               = t.ciphersuites.get();
  rest.tls.cipher_list                = t.cipher_list.get();
  rest.tls.groups                     = t.groups.get();
  rest.tls.sni_override               = t.sni_override.get();
  rest.tls.allow_tls13_without_rekey  = t.allow_tls13_without_rekey.get();

  auto& cache                           = sm_cfg.cache;
  cache.mode                            = sm.cache.mode.get();
  cache.eviction                        = sm.cache.eviction.get();
  cache.min_prefetching_budget_fraction = sm.cache.min_prefetching_budget_fraction.get();
  cache.eviction_threshold_fraction     = sm.cache.eviction_threshold_fraction.get();
  cache.apply_mode();

  auto& prefetcher             = sm_cfg.memory_prefetcher;
  prefetcher.enable            = sm.memory_prefetcher.enable.get();
  prefetcher.num_threads       = sm.memory_prefetcher.num_threads.get();
  prefetcher.min_free_fraction = sm.memory_prefetcher.min_free_fraction.get();
  prefetcher.poll_interval_ms =
    static_cast<std::size_t>(sm.memory_prefetcher.poll_interval.get().count());
  prefetcher.drain_quiet_ms =
    static_cast<std::size_t>(sm.memory_prefetcher.drain_quiet.get().count());

  auto const& prep          = sm.preparation;
  auto& prep_cfg            = sm_cfg.preparation;
  auto const optional_count = [](auto const& option) -> std::optional<std::size_t> {
    if (auto const* value = option.get_if()) { return *value; }
    return std::nullopt;
  };
  prep_cfg.max_inflight_metadata_jobs = optional_count(prep.max_inflight_metadata_jobs);
  prep_cfg.max_inflight_dv_jobs       = optional_count(prep.max_inflight_dv_jobs);
  prep_cfg.lookahead_batches          = optional_count(prep.lookahead_batches);
  prep_cfg.max_control_work           = optional_count(prep.max_control_work);
  prep_cfg.drain_quantum              = optional_count(prep.drain_quantum);
  prep_cfg.coordinator_workers        = optional_count(prep.coordinator_workers);
  prep_cfg.native_reader_workers      = optional_count(prep.native_reader_workers);
  prep_cfg.max_native_readers         = optional_count(prep.max_native_readers);
  prep_cfg.interrupt_check_interval   = prep.interrupt_check_interval.get();
  prep_cfg.partial_batch              = scan_manager::partial_batch_options{
                 .min_fill_percent = prep.partial_batch.min_fill_percent.get(),
                 .min_residence    = prep.partial_batch.min_residence.get(),
                 .max_residence    = prep.partial_batch.max_residence.get()};
  sm_cfg.apply_cache_mode();

  // -- operators
  auto const& op                          = user.operator_params;
  auto& ops                               = settings._operator_params;
  ops.enable_late_materialization         = op.enable_late_materialization.get();
  ops.enable_fused_scan_filter            = op.enable_fused_scan_filter.get();
  ops.enable_low_cardinality_groupby      = op.enable_low_cardinality_groupby.get();
  ops.late_materialization_min_boundaries = op.late_materialization_min_boundaries.get();
  ops.late_materialization_min_value_x_boundaries =
    op.late_materialization_min_value_x_boundaries.get();
  ops.late_materialization_count_defer = op.late_materialization_count_defer.get();
  ops.late_materialization_min_value_compressed =
    op.late_materialization_min_value_compressed.get();
  ops.fused_scan.diagnostics                   = op.fused_scan_diagnostics.get();
  ops.fused_scan.max_selectivity               = op.fused_scan_max_selectivity.get();
  ops.fused_scan.full_route_max_selectivity    = op.fused_scan_full_route_max_selectivity.get();
  ops.fused_scan.index_walk_max_selectivity    = op.fused_scan_index_walk_max_selectivity.get();
  ops.fused_scan.max_membership_sources        = op.fused_scan_max_membership_sources.get();
  ops.max_sort_partition_bytes                 = op.max_sort_partition_bytes.get().count();
  ops.max_sort_partition_memory_fraction       = op.max_sort_partition_memory_fraction.get();
  ops.max_broadcast_join_size                  = op.max_broadcast_join_size.get().count();
  ops.mark_join_build_switch_ratio             = op.mark_join_build_switch_ratio.get();
  ops.enable_dynamic_filter                    = op.enable_dynamic_filter.get();
  ops.enable_dynamic_filter_multi_partition    = op.enable_dynamic_filter_multi_partition.get();
  ops.enable_dynamic_zone_map_filter           = op.enable_dynamic_zone_map_filter.get();
  ops.dynamic_filter_domain_coverage_threshold = op.dynamic_filter_domain_coverage_threshold.get();
  ops.dynamic_filter_inlist_max_l2_fraction    = op.dynamic_filter_inlist_max_l2_fraction.get();
  ops.dynamic_filter_keep_threshold            = op.dynamic_filter_keep_threshold.get();
  ops.enable_pinned_zone_map_pruning           = op.enable_pinned_zone_map_pruning.get();
  ops.enable_decimal_sum_stats_bound           = op.enable_decimal_sum_stats_bound.get();
  ops.enable_compressed_materialization        = op.enable_compressed_materialization.get();
  ops.enable_dense_count_join                  = op.enable_dense_count_join.get();
  ops.enable_delim_aggregate_reuse             = op.enable_delim_aggregate_reuse.get();
  ops.prove_unique_keys                        = op.pin_table_prove_unique_keys.get();
  ops.unique_exact_max_bytes                   = op.pin_table_unique_exact_max_bytes.get().count();
  ops.enable_exists_minmax_rewrite             = op.enable_exists_minmax_rewrite.get();
  ops.enable_partial_aggregate_below_join      = op.enable_partial_aggregate_below_join.get();
  ops.enable_subquery_key_carry                = op.enable_subquery_key_carry.get();
  ops.enable_exists_from_aggregate             = op.enable_exists_from_aggregate.get();
  ops.admission_bytes_per_gpu                  = op.admission_bytes_per_gpu.get().count();
  ops.avg_variable_column_bytes                = op.avg_variable_column_bytes.get().count();
  ops.enable_runtime_size_estimation           = op.enable_runtime_size_estimation.get();
  ops.use_hw_decompression                     = op.use_hw_decompression.get();
  // Unset batch sizes are derived from GPU memory when the configuration is resolved.
  std::pair<config::config_option<config::ubytes> const*, uint64_t operator_params::*> const
    batch_options[] = {
      {&op.scan_task_batch_size, &operator_params::scan_task_batch_size},
      {&op.hash_partition_bytes, &operator_params::hash_partition_bytes},
      {&op.concat_batch_bytes, &operator_params::concat_batch_bytes},
      {&op.cross_join_task_bytes, &operator_params::cross_join_task_bytes},
      {&op.sort_sample_bytes, &operator_params::sort_sample_bytes},
      {&op.max_build_hash_table_bytes, &operator_params::max_build_hash_table_bytes},
      {&op.max_build_probe_resident_bytes, &operator_params::max_build_probe_resident_bytes}};
  for (auto const& [option, member] : batch_options) {
    if (auto const* value = option->get_if()) {
      ops.*member = value->count();
      impl->explicit_batch_settings.push_back(member);
    }
  }

  // -- telemetry and compression
  auto& telemetry              = settings._telemetry_config;
  telemetry.enable_quent       = user.telemetry.enable_quent.get();
  telemetry.enable_nvtx        = user.telemetry.enable_nvtx.get();
  telemetry.output_directory   = user.telemetry.output_directory.get();
  telemetry.engine_name        = user.telemetry.engine_name.get();
  telemetry.nvtx_injection_lib = user.telemetry.nvtx_injection_lib.get();

  auto const& c                             = user.compression;
  auto& compression                         = settings._compression_config;
  compression.enable_pin_table_compression  = c.enable_pin_table_compression.get();
  compression.min_batch_size_bytes          = c.min_batch_size_bytes.get().count();
  compression.max_compressed_fraction       = c.max_compressed_fraction.get();
  compression.input_plan_dir                = c.input_plan_dir.get();
  compression.enable_spill_compression      = c.enable_spill_compression.get();
  compression.spill_min_batch_bytes         = c.spill_min_batch_bytes.get().count();
  compression.spill_max_compressed_fraction = c.spill_max_compressed_fraction.get();
  compression.spill_encode_reserve_fraction = c.spill_encode_reserve_fraction.get();

  // -- memory spaces listed one by one: all GPU, then host, then disk spaces
  for (auto const& space : user.space.gpu) {
    cucascade::memory::gpu_memory_space_config cfg;
    cfg.device_id                  = space.device_id.get();
    cfg.per_stream_reservation     = space.per_stream_reservation.get();
    cfg.reservation_limit_fraction = space.reservation_limit_fraction.get();
    cfg.downgrade_trigger_fraction = space.downgrade_trigger_fraction.get();
    cfg.downgrade_stop_fraction    = space.downgrade_stop_fraction.get();
    cfg.memory_capacity            = space.memory_capacity.get().count();
    settings._memory_space_configs.emplace_back(std::move(cfg));
  }
  for (auto const& space : user.space.host) {
    cucascade::memory::host_memory_space_config cfg;
    cfg.numa_id                    = space.numa_id.get();
    cfg.reservation_limit_fraction = space.reservation_limit_fraction.get();
    cfg.downgrade_trigger_fraction = space.downgrade_trigger_fraction.get();
    cfg.downgrade_stop_fraction    = space.downgrade_stop_fraction.get();
    cfg.memory_capacity            = space.memory_capacity.get().count();
    cfg.block_size                 = space.block_size.get().count();
    cfg.pool_size                  = space.pool_size.get();
    cfg.initial_number_pools       = space.initial_number_pools.get();
    settings._memory_space_configs.emplace_back(std::move(cfg));
  }
  for (auto const& space : user.space.disk) {
    cucascade::memory::disk_memory_space_config cfg;
    cfg.disk_id         = space.disk_id.get();
    cfg.mount_paths     = space.mount_path.get();
    cfg.memory_capacity = space.memory_capacity.get().count();
    settings._memory_space_configs.emplace_back(std::move(cfg));
  }
  impl->use_effective_gpu_capacity = settings._memory_space_configs.empty()
                                       ? gpu.usage_limit.user_defined()
                                       : user.space.gpu_capacities_configured();
  return parsed_sirius_config(std::move(impl));
} catch (const config::config_error& e) {
  throw configuration_input_error(e.what());
}

sirius_config create_sirius_config(const config::user_config& user,
                                   const cucascade::memory::system_topology_info& topology,
                                   const egress_nic_probe& probe)
{
  return parsed_sirius_config::from_user_config(user).resolve(topology, probe);
}

std::optional<cucascade::memory::nic_topology> discover_egress_nic(
  std::string_view configured_interface)
{
  try {
    if (auto const iface = cucascade::memory::select_egress_interface(configured_interface);
        iface.has_value()) {
      return cucascade::memory::discover_nic(*iface);
    }
  } catch (...) {
    // No NIC information: the caller falls back to the CPU-count default.
  }
  return std::nullopt;
}

bool rest_reactors_sized_from_nic(
  cucascade::memory::kernel_tls_info const& kernel_tls,
  std::optional<cucascade::memory::nic_topology> const& egress_nic) noexcept
{
  return kernel_tls.loaded && egress_nic.has_value() && egress_nic->irqs_steered();
}

std::size_t rest_reactor_irq_cpus(
  cucascade::memory::kernel_tls_info const& kernel_tls,
  std::optional<cucascade::memory::nic_topology> const& egress_nic) noexcept
{
  return rest_reactors_sized_from_nic(kernel_tls, egress_nic) ? egress_nic->irq_cpus.size()
                                                              : std::size_t{0};
}

std::size_t default_rest_max_connections(cucascade::memory::kernel_tls_info const& kernel_tls,
                                         std::size_t queue_depth) noexcept
{
  constexpr std::size_t kernel_tls_connections = 128;
  constexpr std::size_t user_tls_connections   = 64;
  auto const wanted = kernel_tls.loaded ? kernel_tls_connections : user_tls_connections;
  return std::clamp<std::size_t>(queue_depth / 2, 1, wanted);
}

std::size_t default_rest_min_mtu(
  std::optional<cucascade::memory::nic_topology> const& egress_nic) noexcept
{
  constexpr int standard_ethernet_mtu = 1500;
  return egress_nic.has_value() && egress_nic->mtu > standard_ethernet_mtu
           ? static_cast<std::size_t>(egress_nic->mtu)
           : std::size_t{0};
}

sirius_config parsed_sirius_config::resolve(
  const cucascade::memory::system_topology_info& hw_topology, const egress_nic_probe& probe) const
try {
  auto settings         = impl_->settings;
  settings._hw_topology = hw_topology;
  auto constraints      = cucascade::memory::discover_cpu_constraints();
  if (!constraints.diagnostic.empty()) SIRIUS_LOG_WARN("CPU placement: {}", constraints.diagnostic);
  SIRIUS_LOG_INFO("CPU startup boundary: process={} allowed={}",
                  cucascade::memory::format_cpu_list(constraints.process_cpus),
                  cucascade::memory::format_cpu_list(constraints.allowed_cpus));
  settings._scan_manager_config.startup_cpus = std::move(constraints.allowed_cpus);
  const auto& topo                           = impl_->topo;
  const auto& gpu_cfg                        = impl_->gpu_cfg;
  const auto& host_cfg                       = impl_->host_cfg;
  const auto& disk_cfg                       = impl_->disk_cfg;
  if (settings._memory_space_configs.empty()) {
    cucascade::memory::reservation_manager_configurator builder;
    if (std::holds_alternative<size_t>(topo.num_gpus_or_gpu_ids)) {
      builder.set_number_of_gpus(
        resolve_num_gpus(std::get<size_t>(topo.num_gpus_or_gpu_ids), hw_topology));
    } else {
      const auto& gpu_ids = std::get<std::vector<int>>(topo.num_gpus_or_gpu_ids);
      builder.set_gpu_ids(gpu_ids);
    }
    gpu_cfg.setup_configurator(builder);
    host_cfg.setup_configurator(builder);
    disk_cfg.setup_configurator(builder);
    if (auto bytes = std::get_if<std::uint64_t>(&gpu_cfg.usage_limit)) {
      // Resolve selected devices through the same configurator before its byte-limit assertion.
      auto capacity_builder = builder;
      capacity_builder.set_usage_limit_ratio_per_gpu(1.0);
      for (auto const& space : capacity_builder.build(hw_topology)) {
        if (auto gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space);
            gpu && *bytes > gpu->memory_capacity) {
          throw configuration_input_error(
            "sirius.memory.gpu.usage_limit_bytes exceeds the capacity of GPU " +
            std::to_string(gpu->device_id));
        }
      }
    }
    settings._memory_space_configs = builder.build(hw_topology);
    for (auto& space : settings._memory_space_configs) {
      auto* host = std::get_if<cucascade::memory::host_memory_space_config>(&space);
      if (!host) { continue; }
      if (host_cfg.initial_number_pools.has_value()) {
        host->initial_number_pools = *host_cfg.initial_number_pools;
        continue;
      }
      auto const* node = hw_topology.find_numa_node(host->numa_id);
      if (!node || node->memory_capacity == 0 || node->is_device_memory) {
        throw configuration_input_error(
          "memory.host: default initial pool sizing requires host RAM capacity for NUMA node " +
          std::to_string(host->numa_id) + "; set initial_number_pools explicitly");
      }
      // Match the allocator's per-block alignment, then round the 10% target down to
      // whole pools without exceeding the configured host capacity.
      constexpr auto alignment = alignof(std::max_align_t);
      constexpr auto max_size  = std::numeric_limits<std::size_t>::max();
      if (host->block_size == 0 || host->pool_size == 0 ||
          host->block_size > max_size - (alignment - 1)) {
        throw configuration_input_error("memory.host: invalid block_size or pool_size");
      }
      auto const aligned_block_size = (host->block_size + alignment - 1) / alignment * alignment;
      if (host->pool_size > max_size / aligned_block_size) {
        throw configuration_input_error("memory.host: pool size in bytes overflows");
      }
      host->initial_number_pools = std::min(node->memory_capacity / 10, host->memory_capacity) /
                                   (aligned_block_size * host->pool_size);
    }
  }

  const auto defaults =
    operator_defaults_for(settings._memory_space_configs, impl_->use_effective_gpu_capacity);
  auto const is_explicit = [&](uint64_t operator_params::* member) {
    return std::find(impl_->explicit_batch_settings.begin(),
                     impl_->explicit_batch_settings.end(),
                     member) != impl_->explicit_batch_settings.end();
  };
  for (const auto& setting : batch_settings) {
    if (!is_explicit(setting.member)) {
      settings._operator_params.*setting.member = defaults.*setting.member;
    }
  }
  // Derive the index budget from the startup partition size, preserving YAML tuning.
  // The former payload limit was 2x partitions; 4-byte rows need a 4x larger index estimate
  // (16 bytes/row). The independent resident budget remains batch-derived.
  // SET hash_partition_bytes does not re-derive either budget.
  if (!is_explicit(&operator_params::max_build_hash_table_bytes)) {
    settings._operator_params.max_build_hash_table_bytes =
      sirius::memory::saturating_mul(8, settings._operator_params.hash_partition_bytes);
  }
  // Pool widths depend on configured GPU executors, not the per-query GPU subset.
  auto const gpu_count = static_cast<std::size_t>(std::count_if(
    settings._memory_space_configs.begin(),
    settings._memory_space_configs.end(),
    [](auto const& space) {
      return std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space);
    }));
  auto const& named    = impl_->explicit_threads;
  // The egress NIC is looked up once, and only when a value derived from it is unnamed: the REST
  // reactor count (which loaded kernel TLS must also allow) or rest.min_mtu.
  std::optional<cucascade::memory::nic_topology> egress_nic;
  if ((!named.rest && hw_topology.kernel_tls.loaded) || !impl_->explicit_rest_min_mtu) {
    egress_nic = probe(settings._scan_manager_config.rest.interface);
  }
  auto const nic_irq_cpus =
    named.rest ? std::size_t{0} : rest_reactor_irq_cpus(hw_topology.kernel_tls, egress_nic);
  auto const threads = exec::derive_thread_defaults(
    std::thread::hardware_concurrency(), gpu_count, hw_topology.kernel_tls.loaded, nic_irq_cpus);
  if (!named.pipeline) { settings._gpu_pipeline_executor_config.num_threads = threads.pipeline; }
  if (!named.task_creator) {
    settings._task_creator_config.thread_pool.num_threads = threads.task_creator;
  }
  if (!named.downgrade) {
    settings._downgrade_executor_config.thread_pool.num_threads = threads.downgrade;
  }
  if (!named.scan_manager) {
    settings._scan_manager_config.thread_pool.num_threads = threads.scan_manager;
  }
  if (!named.uring) { settings._scan_manager_config.uring_n_reactors = threads.uring_reactors; }
  if (!named.rest) { settings._scan_manager_config.rest_n_reactors = threads.rest_reactors; }
  if (!impl_->explicit_rest_min_mtu) {
    settings._scan_manager_config.rest.min_mtu = default_rest_min_mtu(egress_nic);
  }
  if (!impl_->explicit_rest_max_connections) {
    auto& rest           = settings._scan_manager_config.rest;
    rest.max_connections = default_rest_max_connections(hw_topology.kernel_tls, rest.queue_depth);
  }
  // With the derived values in place, the REST settings pass cuCascade's own check.
  if (auto error = rest_config_error(settings._scan_manager_config.rest,
                                     "sirius.executor.scan_manager.rest")) {
    throw configuration_input_error(*error);
  }
  validate_rest_fan_out(settings._scan_manager_config);
  settings.finalize_derived_config();
  return settings;
} catch (const std::bad_alloc&) {
  throw;
} catch (const std::exception& e) {
  if (impl_->source_path.empty()) { throw; }
  throw std::runtime_error(impl_->source_path.string() + ": " + e.what());
}

void sirius_config::finalize_derived_config()
{
  // The opportunistic strategy schedules against what the executor can run,
  // not what the device can queue, so it needs the pipeline pool's width.
  _scan_manager_config.pipeline_width =
    static_cast<std::size_t>(std::max(1, _gpu_pipeline_executor_config.num_threads));
  derive_rest_scan_budget();
  derive_uring_scan_budget();
  enforce_native_backend_for_multi_gpu();
  reserve_rest_reactor_cpus();
  _scan_manager_config.worker_startup_cpus = get_worker_startup_cpus();
}

std::vector<int> reserve_cpus_for_rest_reactors(
  scan_manager::scan_manager_config const& scan_manager,
  rest_reactor_cpu_planner const& plan,
  std::vector<int> const& allowed,
  std::span<exec::thread_pool_config* const> pools)
{
  if (!scan_manager.reserve_rest_reactor_cpus) { return {}; }
  std::vector<int> reactor_cpus;
  try {
    reactor_cpus = plan(scan_manager.rest_n_reactors, scan_manager.rest);
  } catch (const std::exception& e) {
    SIRIUS_LOG_WARN(
      "sirius_config: cannot plan the REST reactor CPUs ({}); thread pools keep "
      "their affinity",
      e.what());
    return {};
  }
  if (reactor_cpus.empty()) { return {}; }
  std::ranges::sort(reactor_cpus);

  auto const without_reactors = [&](std::vector<int> const& cpus) {
    std::vector<int> kept;
    for (int const cpu : cpus) {
      if (!std::binary_search(reactor_cpus.begin(), reactor_cpus.end(), cpu)) {
        kept.push_back(cpu);
      }
    }
    return kept;
  };
  auto const pool_cpus = without_reactors(allowed);
  if (pool_cpus.empty()) {
    SIRIUS_LOG_WARN(
      "sirius_config: the REST reactors would take every allowed CPU ({}); thread "
      "pools keep their affinity",
      cucascade::memory::format_cpu_list(allowed));
    return {};
  }
  for (auto* pool : pools) {
    auto kept =
      without_reactors(pool->cpu_affinity_list.empty() ? allowed : pool->cpu_affinity_list);
    // A pool configured onto reactor CPUs only keeps its configuration rather than losing all
    // its CPUs.
    if (!kept.empty()) { pool->cpu_affinity_list = std::move(kept); }
  }
  SIRIUS_LOG_INFO(
    "sirius_config: REST reactors pin to CPUs {}; Sirius thread pools restricted to CPUs {}",
    cucascade::memory::format_cpu_list(reactor_cpus),
    cucascade::memory::format_cpu_list(pool_cpus));
  return reactor_cpus;
}

std::vector<int> sirius_config::get_worker_startup_cpus() const
{
  auto cpus = get_startup_cpus();
  std::erase_if(cpus, [&](int cpu) {
    return std::binary_search(_rest_reactor_cpus.begin(), _rest_reactor_cpus.end(), cpu);
  });
  return cpus.empty() ? get_startup_cpus() : cpus;
}

void sirius_config::reserve_rest_reactor_cpus()
{
  // Pinned REST reactors share their CPUs with the NIC's interrupt handling (they are pinned to
  // the steered IRQ CPUs).  Every other Sirius thread placed there competes with both, so the
  // pools are restricted to the remaining CPUs.  The REST pool is built lazily, after the pools,
  // so the CPUs come from cuCascade's planning rather than from the pool.
  exec::thread_pool_config* const pools[] = {&_gpu_pipeline_executor_config,
                                             &_scan_manager_config.thread_pool,
                                             &_task_creator_config.thread_pool,
                                             &_downgrade_executor_config.thread_pool};
  if (_scan_manager_config.startup_cpus.empty())
    _scan_manager_config.startup_cpus = cucascade::memory::discover_cpu_constraints().allowed_cpus;
  auto const& allowed = _scan_manager_config.startup_cpus;
  // Explicit pool affinity may narrow the rank boundary, never expand it.
  for (auto* pool : pools) {
    auto cpus = pool->cpu_affinity_list.empty()
                  ? allowed
                  : cucascade::memory::intersect_cpu_affinity(pool->cpu_affinity_list, allowed);
    if (cpus.empty())
      throw configuration_input_error("thread pool affinity has no CPUs within the rank boundary");
    pool->cpu_affinity_list = std::move(cpus);
  }
  if (!_scan_manager_config.reserve_rest_reactor_cpus) {
    _rest_reactor_cpus.clear();
    return;
  }
  exec::thread_util::scoped_thread_affinity affinity(allowed);
  _rest_reactor_cpus = reserve_cpus_for_rest_reactors(
    _scan_manager_config, cucascade::io::rest::planned_reactor_cpus, allowed, pools);
}

void sirius_config::derive_rest_scan_budget()
{
  // An object-store read is latency-bound, not bandwidth-bound: a ranged GET
  // spends most of its life waiting for the first byte.  At one outstanding scan
  // per pipeline thread the readahead has no lead — every split a thread picks
  // up is the split whose prefetch only just started, so the read blocks on it.
  // Several splits per thread give the prefetch time to land.  Local disk has no
  // such round trip to hide, so its budget is derived separately
  // (derive_uring_scan_budget).
  //
  // Only the untouched default is replaced, so an explicit
  // rest.n_max_concurrent_scans in the config still wins.  The derived budget is
  // 2x the configured pipeline pool size, floored at 8 so a small pool still
  // keeps enough ranged GETs in flight to cover the link's round-trip latency.
  constexpr std::size_t scans_per_thread = 2;
  constexpr std::size_t min_rest_scans   = 8;
  if (_scan_manager_config.rest.n_max_concurrent_scans_explicit) { return; }

  auto const derived =
    std::max(min_rest_scans,
             static_cast<std::size_t>(std::max(1, _gpu_pipeline_executor_config.num_threads)) *
               scans_per_thread);
  if (derived == _scan_manager_config.rest.n_max_concurrent_scans) { return; }

  SIRIUS_LOG_INFO(
    "sirius_config: rest.n_max_concurrent_scans defaulted to max({}, {}x the configured pipeline "
    "pool size) = {}, replacing the built-in default of {}",
    min_rest_scans,
    scans_per_thread,
    derived,
    _scan_manager_config.rest.n_max_concurrent_scans);
  _scan_manager_config.rest.n_max_concurrent_scans = derived;
}

void sirius_config::derive_uring_scan_budget()
{
  // One scan per pipeline thread.  The opportunistic readahead issues only while
  // a pipeline thread is off computing rather than reading, so the scans it can
  // usefully keep in flight are bounded by the pool's width; publishing that
  // width keeps the backend's budget in step with what the strategy schedules
  // against (and is the budget an `eager` override would run with).
  //
  // Only the untouched default is replaced, so an explicit
  // uring.n_max_concurrent_scans in the config still wins, an explicit 0 (which
  // keeps the local readahead off) included.
  if (_scan_manager_config.uring.n_max_concurrent_scans_explicit) { return; }

  auto const derived =
    static_cast<std::size_t>(std::max(1, _gpu_pipeline_executor_config.num_threads));
  if (derived == _scan_manager_config.uring.n_max_concurrent_scans) { return; }

  SIRIUS_LOG_INFO(
    "sirius_config: uring.n_max_concurrent_scans defaulted to the configured pipeline pool size = "
    "{}, replacing the built-in default of {}",
    derived,
    _scan_manager_config.uring.n_max_concurrent_scans);
  _scan_manager_config.uring.n_max_concurrent_scans = derived;
}

void sirius_config::enforce_native_backend_for_multi_gpu()
{
  size_t num_gpus = std::ranges::count_if(_memory_space_configs, [](auto const& space) {
    return std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space);
  });
  if (num_gpus > 1 && _scan_manager_config.backend != scan_manager::io_backend::native) {
    SIRIUS_LOG_WARN(
      "sirius_config: backend was not 'native' but {} GPUs are configured; "
      "the native backend is required for multi-GPU IO routing. Overriding "
      "backend to 'native'.",
      num_gpus);
    _scan_manager_config.backend = scan_manager::io_backend::native;
  }
}

const std::vector<cucascade::memory::memory_space_config>& sirius_config::get_memory_space_configs()
  const noexcept
{
  return _memory_space_configs;
}

const exec::thread_pool_config& sirius_config::get_gpu_pipeline_executor_config() const noexcept
{
  return _gpu_pipeline_executor_config;
}

const exec::downgrade_executor_config& sirius_config::get_downgrade_executor_config() const noexcept
{
  return _downgrade_executor_config;
}

const creator::task_creator_config& sirius_config::get_task_creator_config() const noexcept
{
  return _task_creator_config;
}

const scan_manager::scan_manager_config& sirius_config::get_scan_manager_config() const noexcept
{
  return _scan_manager_config;
}

void sirius_config::set_scan_manager_config(scan_manager::scan_manager_config config) noexcept
{
  _scan_manager_config = std::move(config);
}

void sirius_config::set_object_store_config(cucascade::io::object_store_config config) noexcept
{
  _scan_manager_config.object_store = std::move(config);
}

}  // namespace sirius
