// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/user_config.hpp"
#include "config/yaml.hpp"
#include "config/yaml_cpp.hpp"
#include "sirius_config.hpp"

#include <cucascade/io/cache/config.hpp>
#include <cucascade/io/rest/config.hpp>
#include <cucascade/io/tls/config.hpp>
#include <cucascade/memory/config.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
#include <typeinfo>
#include <variant>
#include <vector>

namespace {

std::filesystem::path data_dir()
{
  return std::filesystem::path(std::source_location::current().file_name()).parent_path() / "data";
}

std::string slurp(const std::filesystem::path& path)
{
  std::ifstream in(path);
  REQUIRE(in);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

const cucascade::memory::system_topology_info& this_machine()
{
  static const auto topology = [] {
    cucascade::memory::topology_discovery discovery;
    REQUIRE(discovery.discover(cucascade::memory::NetworkDeviceVerification::EXISTS_ACTIVE_IP,
                               /*with_runtime_attributes=*/true));
    return discovery.get_topology();
  }();
  return topology;
}

/// No egress NIC: the CPU-count and no-MTU-check defaults, the same on every run.
std::optional<cucascade::memory::nic_topology> no_nic(std::string_view) { return std::nullopt; }

/// Every runtime value a configuration reaches, derives, or must leave alone, as text.
class describer {
 public:
  describer() { _out << std::setprecision(17); }

  template <class T>
  describer& operator()(std::string_view key, const T& value)
  {
    _out << key << '=';
    print(value);
    _out << '\n';
    return *this;
  }
  [[nodiscard]] std::string str() const { return _out.str(); }

 private:
  template <class T>
  void print(const T& value)
  {
    if constexpr (std::is_enum_v<T>) {
      _out << static_cast<long long>(value);
    } else if constexpr (requires { value.count(); }) {
      _out << value.count();
    } else {
      _out << value;
    }
  }
  template <class T>
  void print(const std::optional<T>& value)
  {
    if (value) {
      print(*value);
    } else {
      _out << "unset";
    }
  }
  template <class T>
  void print(const std::vector<T>& values)
  {
    _out << '[';
    for (auto const& value : values) {
      print(value);
      _out << ',';
    }
    _out << ']';
  }

  std::ostringstream _out;
};

void describe_pool(describer& d, const std::string& name, const sirius::exec::thread_pool_config& p)
{
  d(name + ".num_threads", p.num_threads)(name + ".prefix", p.thread_name_prefix)(
    name + ".cpus", p.cpu_affinity_list);
}

/// With @p in_effect, what the engine runs with rather than how it is stored: preparation limits
/// as the scan manager resolves them when it starts, and no record of which scan budgets were
/// named.
std::string describe(const sirius::sirius_config& c, bool in_effect = false)
{
  describer d;
  for (auto const& space : c.get_memory_space_configs()) {
    if (auto const* g = std::get_if<cucascade::memory::gpu_memory_space_config>(&space)) {
      d("gpu.device", g->device_id)("gpu.reservation", g->reservation_limit_fraction)(
        "gpu.trigger", g->downgrade_trigger_fraction)("gpu.stop", g->downgrade_stop_fraction)(
        "gpu.capacity", g->memory_capacity)("gpu.per_stream", g->per_stream_reservation);
    } else if (auto const* h = std::get_if<cucascade::memory::host_memory_space_config>(&space)) {
      d("host.numa", h->numa_id)("host.reservation", h->reservation_limit_fraction)(
        "host.trigger", h->downgrade_trigger_fraction)("host.stop", h->downgrade_stop_fraction)(
        "host.capacity", h->memory_capacity)("host.block", h->block_size)(
        "host.pool", h->pool_size)("host.pools", h->initial_number_pools)("host.threads",
                                                                          h->initial_pool_threads);
    } else if (auto const* k = std::get_if<cucascade::memory::disk_memory_space_config>(&space)) {
      d("disk.id", k->disk_id)("disk.capacity", k->memory_capacity)("disk.paths", k->mount_paths);
    } else {
      d("space", "other");
    }
  }
  d("gpus_per_query", c.gpus_per_query())("max_concurrent_queries", c.max_concurrent_queries());
  auto const& creator = c.get_task_creator_config();
  describe_pool(d, "task_creator", creator.thread_pool);
  d("task_creator.strategy", creator.strategy)("task_creator.priority", creator.priority);
  describe_pool(d, "pipeline", c.get_gpu_pipeline_executor_config());
  auto const& dg = c.get_downgrade_executor_config();
  describe_pool(d, "downgrade", dg.thread_pool);
  d("downgrade.monitor", dg.monitor_period)("downgrade.chunk", dg.copy_chunk_bytes)(
    "downgrade.numa", dg.preferred_numa_node);
  d("rest_reactor_cpus", c.get_rest_reactor_cpus());

  auto const& sm = c.get_scan_manager_config();
  describe_pool(d, "scan_manager", sm.thread_pool);
  d("sm.backend", sm.backend)("sm.uring_reactors", sm.uring_n_reactors)(
    "sm.rest_reactors", sm.rest_n_reactors)("sm.reserve", sm.reserve_rest_reactor_cpus)(
    "sm.efc", sm.httpfs_external_file_cache)("sm.startup", sm.startup_cpus)(
    "sm.worker_startup", sm.worker_startup_cpus)("sm.readahead", sm.max_readahead_scans)(
    "sm.strategy", sm.readahead_strategy)("sm.width", sm.pipeline_width);
  auto const& u = sm.uring;
  d("uring.scans", u.n_max_concurrent_scans);
  if (!in_effect) { d("uring.explicit", u.n_max_concurrent_scans_explicit); }
  d("uring.odirect", u.use_odirect)("uring.slices", u.slices_per_pass)(
    "uring.batch", u.range_batch_slices)("uring.service", u.low_service_interval);
  auto const& r = sm.rest;
  d("rest.scans", r.n_max_concurrent_scans);
  if (!in_effect) { d("rest.explicit", r.n_max_concurrent_scans_explicit); }
  d("rest.timeout", r.request_timeout_s)("rest.stall_speed", r.stall_speed_limit_bytes)(
    "rest.stall_time", r.stall_time_s)("rest.ca_bundle", r.ca_bundle_path)(
    "rest.tls_verify", r.tls_verify)("rest.connections", r.max_connections)(
    "rest.gap", r.merge_max_gap)("rest.upkeep", r.upkeep_interval)(
    "rest.upkeep_http1", r.upkeep_http1)("rest.age", r.conn_max_age)(
    "rest.min_segment", r.min_segment_bytes)("rest.max_segment", r.max_segment_bytes)(
    "rest.recv_buffer", r.receive_buffer_bytes)("rest.iovecs", r.max_recv_iovecs)(
    "rest.waitall", r.recv_waitall)("rest.queue", r.queue_depth)("rest.sqpoll", r.sqpoll)(
    "rest.single_issuer", r.single_issuer)("rest.defer", r.defer_taskrun)(
    "rest.ring_fd", r.register_ring_fd)("rest.iowq", r.iowq_max_unbound)(
    "rest.iowq_bound", r.iowq_max_bound)("rest.wait", r.wait_mode)(
    "rest.copy_poll", r.copy_poll_interval)("rest.hold_slot", r.hold_slot_during_copy)(
    "rest.dns_ttl", r.dns_ttl)("rest.dns_acc", r.dns_accumulate)(
    "rest.dns_max", r.dns_max_addresses)("rest.connect", r.connect_timeout)(
    "rest.ka_idle", r.keepalive_idle_s)("rest.ka_interval", r.keepalive_interval_s)(
    "rest.ka_count", r.keepalive_count)("rest.rcvbuf", r.so_rcvbuf)("rest.sndbuf", r.so_sndbuf)(
    "rest.busy_poll", r.busy_poll_us)("rest.user_timeout", r.tcp_user_timeout_ms)(
    "rest.min_mtu", r.min_mtu)("rest.mtu_attempts", r.mtu_reject_max_attempts)(
    "rest.rst", r.close_with_rst)("rest.fan_out", r.fan_out)("rest.interface", r.interface)(
    "rest.irq", r.pin_to_irq_cpus)("rest.priority", r.raise_reactor_priority)(
    "rest.retries", r.max_retry_attempts)("rest.auth_retries", r.max_auth_retry_attempts)(
    "rest.backoff", r.retry_backoff_base)("rest.jitter", r.retry_jitter)(
    "rest.retry_after", r.honor_retry_after)("rest.footer", r.footer_probe_bytes)(
    "rest.list_matches", r.list_max_matches)("rest.list_scanned", r.list_max_scanned)(
    "rest.footer_inflight", r.footer_resolve_max_inflight)("rest.footer_stash",
                                                           r.footer_resolve_stash_budget);
  auto const& t = r.tls;
  d("tls.ktls", t.ktls)("tls.rx", t.fallback_rx)("tls.placement", t.placement)(
    "tls.min", t.min_version)("tls.max", t.max_version)("tls.no_pad", t.expect_no_pad)(
    "tls.verify", t.verify)("tls.ca_file", t.ca_file)("tls.ca_dir", t.ca_dir)(
    "tls.ca_blob", t.ca_blob)("tls.cache", t.session_cache)(
    "tls.per_endpoint", t.session_cache_per_endpoint)("tls.conn_bytes", t.max_connection_bytes)(
    "tls.suites", t.ciphersuites)("tls.ciphers", t.cipher_list)("tls.groups", t.groups)(
    "tls.sni", t.sni_override)("tls.alpn", t.alpn_http11)("tls.rekey", t.allow_tls13_without_rekey)(
    "tls.cert", t.client_cert_file)("tls.key", t.client_key_file)("tls.keylog", t.keylog_env);
  auto const& k = sm.cache;
  d("cache.mode", k.mode)("cache.eviction", k.eviction)("cache.prefetch",
                                                        k.min_prefetching_budget_fraction)(
    "cache.threshold", k.eviction_threshold_fraction)("cache.dispose", k.dispose_on_idle);
  auto const& m = sm.memory_prefetcher;
  d("prefetch.enable", m.enable)("prefetch.threads", m.num_threads)(
    "prefetch.free", m.min_free_fraction)("prefetch.poll", m.poll_interval_ms)("prefetch.quiet",
                                                                               m.drain_quiet_ms);
  auto const& p = sm.preparation;
  if (in_effect) {
    auto const r = p.resolve(sm.thread_pool.num_threads);
    d("prep.meta", r.max_inflight_metadata_jobs)("prep.dv", r.max_inflight_dv_jobs)(
      "prep.lookahead", r.lookahead_batches)("prep.control", r.max_control_work)(
      "prep.drain", r.drain_quantum)("prep.coordinators", p.resolved_coordinator_workers())(
      "prep.workers", p.resolved_native_workers())("prep.readers", p.resolved_native_readers());
  } else {
    d("prep.meta", p.max_inflight_metadata_jobs)("prep.dv", p.max_inflight_dv_jobs)(
      "prep.lookahead", p.lookahead_batches)("prep.control", p.max_control_work)(
      "prep.drain", p.drain_quantum)("prep.coordinators", p.coordinator_workers)(
      "prep.workers", p.native_reader_workers)("prep.readers", p.max_native_readers);
  }
  d("prep.interrupt", p.interrupt_check_interval)("prep.partial", p.partial_batch.has_value());
  if (p.partial_batch) {
    d("prep.fill", p.partial_batch->min_fill_percent)(
      "prep.min_res", p.partial_batch->min_residence)("prep.max_res",
                                                      p.partial_batch->max_residence);
  }

  auto const& o = c.get_operator_params();
  d("op.late", o.enable_late_materialization)("op.fused", o.enable_fused_scan_filter)(
    "op.low_card", o.enable_low_cardinality_groupby)("op.lm_bounds",
                                                     o.late_materialization_min_boundaries)(
    "op.lm_x", o.late_materialization_min_value_x_boundaries)(
    "op.lm_defer", o.late_materialization_count_defer)("op.lm_compressed",
                                                       o.late_materialization_min_value_compressed)(
    "op.fs_diag", o.fused_scan.diagnostics)("op.fs_sel", o.fused_scan.max_selectivity)(
    "op.fs_full", o.fused_scan.full_route_max_selectivity)("op.fs_walk",
                                                           o.fused_scan.index_walk_max_selectivity)(
    "op.fs_sources", o.fused_scan.max_membership_sources)("op.swar", o.like_swar_fastpath)(
    "op.expr", o.expression_strategy)("op.regex", o.enable_regex_jit)(
    "op.scan_batch", o.scan_task_batch_size)("op.sort_bytes", o.max_sort_partition_bytes)(
    "op.sort_fraction", o.max_sort_partition_memory_fraction)("op.hash", o.hash_partition_bytes)(
    "op.concat", o.concat_batch_bytes)("op.cross", o.cross_join_task_bytes)(
    "op.sample", o.sort_sample_bytes)("op.build_table", o.max_build_hash_table_bytes)(
    "op.resident", o.max_build_probe_resident_bytes)("op.broadcast", o.max_broadcast_join_size)(
    "op.mark", o.mark_join_build_switch_ratio)(
    "op.distinct", o.enable_runtime_distinct_build_probe)("op.df", o.enable_dynamic_filter)(
    "op.df_multi", o.enable_dynamic_filter_multi_partition)("op.peers", o.exchange_peers)(
    "op.rank", o.exchange_rank)("op.staging", o.exchange_host_staging)(
    "op.df_zone", o.enable_dynamic_zone_map_filter)("op.df_cover",
                                                    o.dynamic_filter_domain_coverage_threshold)(
    "op.df_l2", o.dynamic_filter_inlist_max_l2_fraction)(
    "op.df_keep", o.dynamic_filter_keep_threshold)("op.size_est", o.enable_runtime_size_estimation)(
    "op.sum_bound", o.enable_decimal_sum_stats_bound)(
    "op.zone_prune", o.enable_pinned_zone_map_pruning)("op.hw", o.use_hw_decompression)(
    "op.cm", o.enable_compressed_materialization)("op.dict", o.enable_dictionary_carrier)(
    "op.unique", o.prove_unique_keys)("op.unique_bytes", o.unique_exact_max_bytes)(
    "op.dense", o.enable_dense_count_join)("op.delim", o.enable_delim_aggregate_reuse)(
    "op.minmax", o.enable_exists_minmax_rewrite)(
    "op.partial", o.enable_partial_aggregate_below_join)("op.carry", o.enable_subquery_key_carry)(
    "op.exists", o.enable_exists_from_aggregate)("op.having", o.enable_complete_group_having)(
    "op.late_join", o.enable_late_dimension_join)("op.footer_remote",
                                                  o.parquet_footer_statistics_max_remote_reads)(
    "op.footer_local", o.parquet_footer_statistics_max_local_reads)(
    "op.dense_bytes", o.dense_count_join_max_bytes)("op.dense_fraction",
                                                    o.dense_count_join_memory_fraction)(
    "op.admission", o.admission_bytes_per_gpu)("op.var_bytes", o.avg_variable_column_bytes);

  auto const& tel = c.get_telemetry_config();
  d("tel.quent", tel.enable_quent)("tel.nvtx", tel.enable_nvtx)("tel.dir", tel.output_directory)(
    "tel.engine", tel.engine_name)("tel.lib", tel.nvtx_injection_lib);
  auto const& z = c.get_compression_config();
  d("z.pin", z.enable_pin_table_compression)("z.min", z.min_batch_size_bytes)(
    "z.max", z.max_compressed_fraction)("z.plans", z.input_plan_dir)(
    "z.spill", z.enable_spill_compression)("z.spill_min", z.spill_min_batch_bytes)(
    "z.spill_max", z.spill_max_compressed_fraction)("z.spill_reserve",
                                                    z.spill_encode_reserve_fraction);
  return d.str();
}

/// What resolving produced: the description, or the exception's type and message.
struct outcome {
  std::string description;
  std::string error_type;
  std::string error;
};

template <class Resolve>
outcome resolved(Resolve&& resolve)
{
  try {
    return {describe(resolve()), {}, {}};
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const sirius::configuration_input_error& e) {
    return {{}, "configuration_input_error", e.what()};
  } catch (const std::exception& e) {
    return {{}, typeid(e).name(), e.what()};
  }
}

/// Both resolve alike: the same description, or the same failure. Reading a file prefixes its
/// messages with the file name.
void require_same(const outcome& from_file, const outcome& from_user, std::string_view source)
{
  INFO("file error: " << from_file.error);
  INFO("user configuration error: " << from_user.error_type << ": " << from_user.error);
  REQUIRE(from_user.description == from_file.description);
  if (from_file.error.empty()) {
    REQUIRE(from_user.error.empty());
    return;
  }
  REQUIRE(from_user.error_type == "configuration_input_error");
  auto expected = from_file.error;
  if (!source.empty() && expected.starts_with(std::string(source) + ": ")) {
    expected.erase(0, source.size() + 2);
  }
  REQUIRE(from_user.error == expected);
}

/// A machine with kernel TLS loaded and a jumbo-frame egress NIC: the derived REST defaults the
/// explicit-value flags gate.
cucascade::memory::system_topology_info with_ktls()
{
  auto topology              = this_machine();
  topology.kernel_tls.loaded = true;
  return topology;
}

struct counting_nic {
  int* probes;
  std::optional<cucascade::memory::nic_topology> operator()(std::string_view interface) const
  {
    ++*probes;
    cucascade::memory::nic_topology nic;
    nic.interface = std::string(interface.empty() ? "eth9" : interface);
    nic.mtu       = 9000;
    return nic;
  }
};

}  // namespace

TEST_CASE("a fixture file and its user configuration resolve alike", "[config][config_create]")
{
  auto const& topology = this_machine();
  std::size_t compared = 0;
  bool every_key_seen  = false;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    auto const name = entry.path().filename().string();
    if (entry.path().extension() != ".yaml" || name.starts_with("invalid_")) { continue; }
    INFO(name);
    auto const from_file = resolved([&] {
      return sirius::parsed_sirius_config::from_file(entry.path()).resolve(topology, no_nic);
    });
    auto const from_user = resolved([&] {
      return sirius::create_sirius_config(
        sirius::config::read_user_config(slurp(entry.path())), topology, no_nic);
    });
    require_same(from_file, from_user, entry.path().string());
    ++compared;
    if (name == "valid_every_key.yaml") {
      every_key_seen = true;
      REQUIRE_FALSE(from_user.description.empty());  // every key, resolved
    }
  }
  REQUIRE(compared >= 30);
  REQUIRE(every_key_seen);
}

TEST_CASE("an invalid fixture file fails with its path named once", "[config][config_create]")
{
  std::size_t checked = 0;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    auto const name = entry.path().filename().string();
    if (entry.path().extension() != ".yaml" || !name.starts_with("invalid_")) { continue; }
    INFO(name);
    auto const path    = entry.path().string();
    auto const failure = resolved([&] {
      return sirius::parsed_sirius_config::from_file(entry.path()).resolve(this_machine(), no_nic);
    });
    // Whether reading or resolving fails, the error keeps its type and names the file once.
    REQUIRE(failure.error_type == "configuration_input_error");
    REQUIRE(failure.error.starts_with(path + ": "));
    REQUIRE(failure.error.find(path, path.size()) == std::string::npos);
    ++checked;
  }
  REQUIRE(checked >= 50);
}

TEST_CASE("a document loaded by yaml-cpp converts as parse_yaml reads its text",
          "[config][config_create]")
{
  std::size_t compared = 0;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    if (entry.path().extension() != ".yaml") { continue; }
    INFO(entry.path().filename().string());
    auto const text = slurp(entry.path());
    std::optional<sirius::config::node> parsed;
    try {
      parsed = sirius::config::parse_yaml(text);
    } catch (const sirius::config::config_error&) {
      // A fixture parse_yaml rejects (duplicate keys, a second document) is not compared.
      continue;
    }
    REQUIRE(sirius::config::from_yaml_cpp(YAML::Load(text)) == *parsed);
    ++compared;
  }
  REQUIRE(compared >= 80);
}

TEST_CASE("every key of a configuration reaches its runtime field", "[config][config_create]")
{
  using namespace std::chrono_literals;
  constexpr std::uint64_t KiB = 1ULL << 10;
  constexpr std::uint64_t MiB = 1ULL << 20;
  constexpr std::uint64_t GiB = 1ULL << 30;
  constexpr std::uint64_t TiB = 1ULL << 40;
  // valid_every_key.yaml sets every key away from its default; resolving it must land each value
  // in its own runtime field.
  auto const c = sirius::parsed_sirius_config::from_file(data_dir() / "valid_every_key.yaml")
                   .resolve(this_machine(), no_nic);

  std::size_t gpus = 0, hosts = 0, disks = 0;
  for (auto const& space : c.get_memory_space_configs()) {
    if (auto const* g = std::get_if<cucascade::memory::gpu_memory_space_config>(&space)) {
      ++gpus;
      CHECK(g->device_id == 0);
      CHECK(g->reservation_limit_fraction == 0.95);
      CHECK(g->downgrade_trigger_fraction == 0.85);
      CHECK(g->downgrade_stop_fraction == 0.5);
    } else if (auto const* h = std::get_if<cucascade::memory::host_memory_space_config>(&space)) {
      ++hosts;
      CHECK(h->memory_capacity == 4 * GiB);
      CHECK(h->reservation_limit_fraction == 0.9);
      CHECK(h->downgrade_trigger_fraction == 0.85);
      CHECK(h->downgrade_stop_fraction == 0.7);
      CHECK(h->block_size == 2 * MiB);
      CHECK(h->pool_size == 64);
      CHECK(h->initial_number_pools == 2);
    } else if (auto const* k = std::get_if<cucascade::memory::disk_memory_space_config>(&space)) {
      ++disks;
      CHECK(k->disk_id == 1);
      CHECK(k->memory_capacity == 2 * TiB);
      CHECK(k->mount_paths == "/tmp");
    }
  }
  CHECK(gpus == 1);
  CHECK(hosts >= 1);
  CHECK(disks == 1);
  CHECK(c.gpus_per_query() == 1);
  CHECK(c.max_concurrent_queries() == 2);

  auto const& creator = c.get_task_creator_config().thread_pool;
  CHECK(creator.num_threads == 3);
  CHECK(creator.cpu_affinity_list == std::vector<int>{0, 1});
  CHECK(c.get_gpu_pipeline_executor_config().num_threads == 5);
  auto const& downgrade = c.get_downgrade_executor_config();
  CHECK(downgrade.thread_pool.num_threads == 2);
  CHECK(downgrade.thread_pool.cpu_affinity_list == std::vector<int>{4});
  CHECK(downgrade.monitor_period == 20ms);
  CHECK(downgrade.copy_chunk_bytes == 512 * MiB);

  auto const& sm = c.get_scan_manager_config();
  CHECK(sm.thread_pool.num_threads == 6);
  CHECK(sm.thread_pool.cpu_affinity_list == std::vector<int>{2, 3});
  CHECK(sm.backend == sirius::scan_manager::io_backend::native);
  CHECK(sm.uring_n_reactors == 3);
  CHECK(sm.rest_n_reactors == 6);
  CHECK_FALSE(sm.reserve_rest_reactor_cpus);
  CHECK(sm.httpfs_external_file_cache);
  CHECK(sm.max_readahead_scans == 5);
  CHECK(sm.readahead_strategy == sirius::scan_manager::prefetch_strategy::opportunistic);

  CHECK(sm.uring.n_max_concurrent_scans == 7);
  CHECK(sm.uring.n_max_concurrent_scans_explicit);
  CHECK(sm.uring.slices_per_pass == 8);
  CHECK(sm.uring.range_batch_slices == 4);

  auto const& r = sm.rest;
  CHECK(r.n_max_concurrent_scans == 12);
  CHECK(r.n_max_concurrent_scans_explicit);
  CHECK(r.request_timeout_s == 45);
  CHECK(r.stall_speed_limit_bytes == 32768);
  CHECK(r.stall_time_s == 20);
  CHECK(r.merge_max_gap == 1 * MiB);
  CHECK(r.min_segment_bytes == 2 * MiB);
  CHECK(r.max_segment_bytes == 8 * MiB);
  CHECK(r.footer_probe_bytes == 256 * KiB);
  CHECK(r.list_max_matches == 5000);
  CHECK(r.list_max_scanned == 50000);
  CHECK(r.retry_backoff_base == 75ms);
  CHECK(r.retry_jitter == 25ms);
  CHECK(r.max_retry_attempts == 4);
  CHECK(r.max_auth_retry_attempts == 2);
  CHECK_FALSE(r.honor_retry_after);
  CHECK(r.max_connections == 96);
  CHECK(r.conn_max_age == 30s);
  CHECK(r.upkeep_interval == 5000ms);
  CHECK(r.upkeep_http1);
  CHECK(r.connect_timeout == 2500ms);
  CHECK(r.dns_ttl == 120s);
  CHECK(r.dns_accumulate);
  CHECK(r.dns_max_addresses == 32);
  CHECK(r.keepalive_idle_s == 30);
  CHECK(r.keepalive_interval_s == 10);
  CHECK(r.keepalive_count == 3);
  CHECK(r.min_mtu == 1400);
  CHECK(r.mtu_reject_max_attempts == 4);
  CHECK_FALSE(r.close_with_rst);
  CHECK(r.receive_buffer_bytes == 512 * KiB);
  CHECK(r.queue_depth == 512);
  CHECK(r.wait_mode == cucascade::io::rest::reactor_wait_mode::uring);
  CHECK(r.iowq_max_unbound == 4);
  CHECK(r.fan_out == 3);
  CHECK(r.interface == "lo");
  CHECK_FALSE(r.pin_to_irq_cpus);
  CHECK(r.raise_reactor_priority);

  auto const& t = r.tls;
  CHECK(t.ktls == cucascade::io::tls::ktls_mode::disable);
  CHECK(t.fallback_rx == cucascade::io::tls::user_space_rx::library);
  CHECK(t.placement == cucascade::io::tls::recv_placement::iowq);
  CHECK(t.min_version == cucascade::io::tls::version_bound::tls1_2);
  CHECK(t.max_version == cucascade::io::tls::version_bound::tls1_3);
  CHECK_FALSE(t.expect_no_pad);
  CHECK(t.session_cache);
  CHECK(t.session_cache_per_endpoint == 16);
  CHECK(t.max_connection_bytes == 1 * GiB);
  CHECK(t.ciphersuites == "TLS_AES_256_GCM_SHA384");
  CHECK(t.cipher_list == "ECDHE-RSA-AES256-GCM-SHA384");
  CHECK(t.groups == "X25519");
  CHECK(t.sni_override == "example.internal");
  CHECK_FALSE(t.allow_tls13_without_rekey);

  CHECK(sm.cache.mode == cucascade::io::cache::cache_mode::os);
  CHECK(sm.cache.eviction == cucascade::io::cache::eviction_policy::idle);
  CHECK(sm.cache.min_prefetching_budget_fraction == 0.1);
  CHECK(sm.cache.eviction_threshold_fraction == 0.7);

  auto const& m = sm.memory_prefetcher;
  CHECK(m.enable);
  CHECK(m.num_threads == 3);
  CHECK(m.min_free_fraction == 0.3);
  CHECK(m.poll_interval_ms == 4);
  CHECK(m.drain_quiet_ms == 50);

  auto const& p = sm.preparation;
  CHECK(p.max_inflight_metadata_jobs == 128);
  CHECK(p.max_inflight_dv_jobs == 8);
  CHECK(p.lookahead_batches == 3);
  CHECK(p.max_control_work == 5);
  CHECK(p.drain_quantum == 6);
  CHECK(p.coordinator_workers == 2);
  CHECK(p.native_reader_workers == 3);
  CHECK(p.max_native_readers == 9);
  CHECK(p.interrupt_check_interval == 20ms);
  REQUIRE(p.partial_batch.has_value());
  CHECK(p.partial_batch->min_fill_percent == 60);
  CHECK(p.partial_batch->min_residence == 40ms);
  CHECK(p.partial_batch->max_residence == 1500ms);

  auto const& tel = c.get_telemetry_config();
  CHECK_FALSE(tel.enable_quent);
  CHECK(tel.enable_nvtx);
  CHECK(tel.output_directory == "every_key_telemetry");
  CHECK(tel.engine_name == "everyKey");
  CHECK(tel.nvtx_injection_lib == "/nonexistent/libnvtx_inject.so");

  auto const& z = c.get_compression_config();
  CHECK_FALSE(z.enable_pin_table_compression);
  CHECK(z.min_batch_size_bytes == 2 * MiB);
  CHECK(z.max_compressed_fraction == 0.6);
  CHECK(z.input_plan_dir == "/nonexistent/plans");
  CHECK(z.enable_spill_compression);
  CHECK(z.spill_min_batch_bytes == 32 * MiB);
  CHECK(z.spill_max_compressed_fraction == 0.5);
  CHECK(z.spill_encode_reserve_fraction == 0.25);

  auto const& o = c.get_operator_params();
  CHECK_FALSE(o.enable_late_materialization);
  CHECK_FALSE(o.enable_fused_scan_filter);
  CHECK_FALSE(o.enable_low_cardinality_groupby);
  CHECK(o.late_materialization_min_boundaries == 6);
  CHECK(o.late_materialization_min_value_x_boundaries == 64);
  CHECK(o.late_materialization_count_defer);
  CHECK(o.late_materialization_min_value_compressed == 2);
  CHECK(o.fused_scan.diagnostics);
  CHECK(o.fused_scan.max_selectivity == 0.3);
  CHECK(o.fused_scan.full_route_max_selectivity == 0.2);
  CHECK(o.fused_scan.index_walk_max_selectivity == 0.25);
  CHECK(o.fused_scan.max_membership_sources == 2);
  CHECK(o.scan_task_batch_size == 64 * MiB);
  CHECK(o.max_sort_partition_bytes == 128 * MiB);
  CHECK(o.max_sort_partition_memory_fraction == 0.25);
  CHECK(o.hash_partition_bytes == 96 * MiB);
  CHECK(o.concat_batch_bytes == 48 * MiB);
  CHECK(o.cross_join_task_bytes == 40 * MiB);
  CHECK(o.sort_sample_bytes == 24 * MiB);
  CHECK(o.max_build_hash_table_bytes == 512 * MiB);
  CHECK(o.max_build_probe_resident_bytes == 384 * MiB);
  CHECK(o.max_broadcast_join_size == 128 * MiB);
  CHECK(o.mark_join_build_switch_ratio == 4.0);
  CHECK_FALSE(o.enable_dynamic_filter);
  CHECK_FALSE(o.enable_dynamic_filter_multi_partition);
  CHECK(o.enable_dynamic_zone_map_filter);
  CHECK(o.dynamic_filter_domain_coverage_threshold == 0.8);
  CHECK(o.dynamic_filter_inlist_max_l2_fraction == 0.25);
  CHECK(o.dynamic_filter_keep_threshold == 0.8);
  CHECK_FALSE(o.enable_pinned_zone_map_pruning);
  CHECK_FALSE(o.enable_decimal_sum_stats_bound);
  CHECK_FALSE(o.enable_compressed_materialization);
  CHECK(o.enable_dictionary_carrier);
  CHECK_FALSE(o.enable_dense_count_join);
  CHECK_FALSE(o.enable_delim_aggregate_reuse);
  CHECK_FALSE(o.prove_unique_keys);
  CHECK(o.unique_exact_max_bytes == 256 * MiB);
  CHECK_FALSE(o.enable_exists_minmax_rewrite);
  CHECK_FALSE(o.enable_partial_aggregate_below_join);
  CHECK(o.enable_subquery_key_carry);
  CHECK(o.enable_exists_from_aggregate);
  CHECK_FALSE(o.enable_complete_group_having);
  CHECK_FALSE(o.enable_late_dimension_join);
  CHECK(o.parquet_footer_statistics_max_remote_reads == 64);
  CHECK(o.parquet_footer_statistics_max_local_reads == -1);
  CHECK(o.admission_bytes_per_gpu == 4 * GiB);
  CHECK(o.avg_variable_column_bytes == 48);
  CHECK(o.enable_runtime_size_estimation);
  CHECK(o.use_hw_decompression);
}

TEST_CASE("explicit REST values gate what is derived from the NIC and kernel TLS",
          "[config][config_create]")
{
  auto const topology = with_ktls();
  auto const resolve  = [&](std::string_view text, int& probes) {
    return sirius::create_sirius_config(
             sirius::config::read_user_config(text), topology, counting_nic{&probes})
      .get_scan_manager_config();
  };
  int probes   = 0;
  auto derived = resolve("sirius: {}\n", probes);
  REQUIRE(derived.rest.min_mtu == 9000);         // the jumbo-frame NIC's MTU
  REQUIRE(derived.rest.max_connections == 128);  // kernel TLS: half the queue, up to 128
  REQUIRE(probes == 1);

  probes     = 0;
  auto named = resolve(
    "sirius:\n  executor:\n    scan_manager:\n      rest_n_reactors: 4\n"
    "      rest:\n        min_mtu: 0\n        max_connections: 64\n"
    "        n_max_concurrent_scans: 8\n      uring:\n"
    "        n_max_concurrent_scans: 0\n",
    probes);
  REQUIRE(probes == 0);  // nothing left to derive from the NIC
  REQUIRE(named.rest.min_mtu == 0);
  REQUIRE(named.rest.max_connections == 64);
  REQUIRE(named.rest_n_reactors == 4);
  REQUIRE(named.rest.n_max_concurrent_scans == 8);
  REQUIRE(named.rest.n_max_concurrent_scans_explicit);
  REQUIRE(named.uring.n_max_concurrent_scans == 0);
  REQUIRE(named.uring.n_max_concurrent_scans_explicit);

  auto const operators =
    sirius::create_sirius_config(sirius::config::read_user_config(
                                   "sirius:\n  operator_params:\n    hash_partition_bytes: 1MiB\n"),
                                 topology,
                                 no_nic)
      .get_operator_params();
  REQUIRE(operators.max_build_hash_table_bytes == 8ULL << 20);  // derived from the named size
}

TEST_CASE("an empty or never-read configuration resolves as the default one",
          "[config][config_create]")
{
  auto const& topology = this_machine();
  auto const expected  = describe(sirius::parsed_sirius_config{}.resolve(topology, no_nic));
  REQUIRE(describe(sirius::create_sirius_config(
            sirius::config::default_user_config(), topology, no_nic)) == expected);
  REQUIRE(describe(sirius::create_sirius_config(sirius::config::user_config{}, topology, no_nic)) ==
          expected);
}

TEST_CASE("a written configuration resolves as the one it was written from",
          "[config][config_create]")
{
  // Every key set: writing the user's values and reading them back changes nothing.
  auto const user   = sirius::config::read_user_config(slurp(data_dir() / "valid_every_key.yaml"));
  auto const reread = sirius::config::read_user_config(sirius::config::write_user_config(user));
  REQUIRE(describe(sirius::create_sirius_config(reread, this_machine(), no_nic)) ==
          describe(sirius::create_sirius_config(user, this_machine(), no_nic)));
}

TEST_CASE("a configuration built in code obeys every block's rules", "[config][config_create]")
{
  sirius::config::user_config user;
  user.max_concurrent_queries.assign(2);  // never read: fallbacks fill in the rest
  REQUIRE(sirius::create_sirius_config(user, this_machine(), no_nic).max_concurrent_queries() == 2);

  user.memory.gpu.downgrade_trigger_fraction.assign(0.5);  // below the default stop of 0.6
  REQUIRE_THROWS_WITH(
    sirius::create_sirius_config(user, this_machine(), no_nic),
    "'sirius.memory.gpu.downgrade_stop_fraction': must be less than downgrade_trigger_fraction");

  auto spaces = sirius::config::read_user_config("sirius:\n  space:\n    gpu: [{}]\n");
  spaces.memory.gpu.usage_limit.assign(0.5);
  REQUIRE_THROWS_AS(sirius::create_sirius_config(spaces, this_machine(), no_nic),
                    sirius::configuration_input_error);
}

TEST_CASE("resolution failures are configuration errors", "[config][config_create]")
{
  auto const fails = [](std::string_view text) {
    return resolved([&] {
      return sirius::create_sirius_config(
        sirius::config::read_user_config(text), this_machine(), no_nic);
    });
  };
  auto const too_many = fails("sirius:\n  topology:\n    num_gpus: 64\n");
  REQUIRE(too_many.error_type == "configuration_input_error");
  auto const fan_out =
    fails("sirius:\n  executor:\n    scan_manager:\n      rest:\n        fan_out: 17\n");
  REQUIRE(fan_out.error_type == "configuration_input_error");
  REQUIRE_THAT(fan_out.error,
               Catch::Matchers::StartsWith(
                 "'sirius.executor.scan_manager.rest.fan_out': must be at most rest_n_reactors"));
}

TEST_CASE("REST limits that depend on a derived value are checked after derivation",
          "[config][config_create]")
{
  // An unnamed max_connections derives from queue_depth (half of it), so a small queue no longer
  // fails against the struct default of 64.
  constexpr std::string_view text =
    "sirius:\n  executor:\n    scan_manager:\n      rest:\n        queue_depth: 100\n";
  auto const config =
    sirius::create_sirius_config(sirius::config::read_user_config(text), this_machine(), no_nic);
  REQUIRE(config.get_scan_manager_config().rest.max_connections == 50);
}

TEST_CASE("a NIC MTU above what min_mtu may name derives no check", "[config][config_create]")
{
  auto const huge_mtu = [](std::string_view) {
    cucascade::memory::nic_topology nic;
    nic.interface = "lo";
    nic.mtu       = 70000;
    return std::optional(nic);
  };
  auto const config =
    sirius::create_sirius_config(sirius::config::default_user_config(), this_machine(), huge_mtu);
  REQUIRE(config.get_scan_manager_config().rest.min_mtu == 0);
}

TEST_CASE("the effective configuration resolves to what it was derived from",
          "[config][config_create]")
{
  for (auto const& user :
       {sirius::config::default_user_config(),
        sirius::config::read_user_config(slurp(data_dir() / "valid_every_key.yaml"))}) {
    auto const resolved  = sirius::create_sirius_config(user, this_machine(), no_nic);
    auto const effective = sirius::effective_user_config(user, resolved);
    // Hardware-derived values are spelled out...
    REQUIRE(effective.executor.pipeline.num_threads.user_defined());
    REQUIRE(effective.executor.scan_manager.rest.max_connections.user_defined());
    REQUIRE(effective.operator_params.hash_partition_bytes.user_defined());
    // ...and resolving them changes nothing the engine runs with.
    REQUIRE(describe(sirius::create_sirius_config(effective, this_machine(), no_nic),
                     /*in_effect=*/true) == describe(resolved, /*in_effect=*/true));
  }
}

TEST_CASE("the effective configuration file resolves alike in every build",
          "[config][config_create]")
{
  int probes = 0;
  struct machine {
    const char* name;
    cucascade::memory::system_topology_info topology;
    sirius::egress_nic_probe probe;
  };
  machine const machines[] = {
    {"this machine", this_machine(), no_nic},
    {"kernel TLS and a jumbo-frame NIC", with_ktls(), counting_nic{&probes}}};
  for (auto const& m : machines) {
    for (bool const developer : {false, true}) {
      INFO(m.name << (developer ? ", developer mode" : ""));
      auto const resolved =
        sirius::create_sirius_config(sirius::config::default_user_config(), m.topology, m.probe);
      auto const text =
        sirius::effective_config_text(resolved, sirius::config::default_user_config(), developer);
      REQUIRE(text.starts_with("# "));
      // Settings left out outside developer mode are re-derived as they were the first time.
      auto const reread = sirius::config::read_user_config(text);
      REQUIRE(describe(sirius::create_sirius_config(reread, m.topology, m.probe),
                       /*in_effect=*/true) == describe(resolved, /*in_effect=*/true));
      auto const document = sirius::config::parse_yaml(text);
      auto const* root    = document.find("sirius");
      REQUIRE(root != nullptr);
      auto const* executor = root->find("executor");
      REQUIRE(executor != nullptr);
      auto const* sm = executor->find("scan_manager");
      REQUIRE(sm != nullptr);
      REQUIRE(sm->find("rest") != nullptr);
      REQUIRE((sm->find("rest")->find("queue_depth") != nullptr) == developer);
      REQUIRE((sm->find("uring") != nullptr) == developer);
    }
  }
}

TEST_CASE("the effective configuration keeps what the config file named", "[config][config_create]")
{
  // A production setting, developer-only ones, and a developer-only block.
  auto const file  = sirius::parsed_sirius_config::from_document(sirius::config::parse_yaml(R"(
sirius:
  memory:
    gpu:
      usage_limit_fraction: 0.5
  executor:
    scan_manager:
      uring:
        n_max_concurrent_scans: 7
  operator_params:
    enable_late_materialization: false
)"));
  auto const& user = file.get_user_config();
  REQUIRE(user.operator_params.enable_late_materialization.user_defined());
  REQUIRE_FALSE(user.operator_params.enable_fused_scan_filter.user_defined());
  auto const resolved = sirius::create_sirius_config(user, this_machine(), no_nic);
  for (bool const developer : {false, true}) {
    INFO((developer ? "developer mode" : "production"));
    auto const text    = sirius::effective_config_text(resolved, user, developer);
    auto const written = sirius::config::read_user_config(text);
    REQUIRE(written.memory.gpu.usage_limit.get<double>() == 0.5);
    REQUIRE(written.executor.scan_manager.uring.n_max_concurrent_scans.get() == 7);
    REQUIRE_FALSE(written.operator_params.enable_late_materialization.get());
    // Developer-only settings the file did not name are written in developer mode only.
    REQUIRE(written.operator_params.enable_fused_scan_filter.user_defined() == developer);
    REQUIRE(describe(sirius::create_sirius_config(written, this_machine(), no_nic),
                     /*in_effect=*/true) == describe(resolved, /*in_effect=*/true));
  }
}

TEST_CASE("the effective configuration file", "[config][config_create]")
{
  auto const dir = std::filesystem::temp_directory_path() /
                   ("sirius_effective_config_test_" + std::to_string(getpid()));
  auto const path = dir / "nested" / "effective_config.yaml";
  std::filesystem::remove_all(dir);
  auto const user     = sirius::config::default_user_config();
  auto const resolved = sirius::create_sirius_config(user, this_machine(), no_nic);
  sirius::write_effective_config(path, resolved, user);
  REQUIRE(std::filesystem::exists(path));
  auto const files = [&] {
    return std::distance(std::filesystem::directory_iterator(path.parent_path()),
                         std::filesystem::directory_iterator{});
  };
  REQUIRE(files() == 1);  // no temporary left behind
  REQUIRE(slurp(path) ==
          sirius::effective_config_text(resolved, user, sirius::config::developer_mode()));

  auto const written = sirius::config::read_user_config(slurp(path));
  REQUIRE(written.executor.pipeline.num_threads.get() ==
          resolved.get_gpu_pipeline_executor_config().num_threads);
  REQUIRE(written.memory.gpu.usage_limit.user_defined());  // fallbacks are spelled out too

  sirius::write_effective_config(path, resolved, user);  // a second start replaces it
  REQUIRE(files() == 1);
  REQUIRE(sirius::config::read_user_config(slurp(path)).memory.gpu.usage_limit.user_defined());
  std::filesystem::remove_all(dir);
}
