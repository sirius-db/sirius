// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/user_config.hpp"
#include "sirius_config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/topology_discovery.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <source_location>
#include <sstream>
#include <string>
#include <string_view>
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

/// Every runtime value a configuration file reaches, or that is derived from one, as text.
class describer {
 public:
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

void describe_pool(describer& d, std::string_view name, const sirius::exec::thread_pool_config& p)
{
  d(std::string(name) + ".num_threads", p.num_threads)(
    std::string(name) + ".prefix", p.thread_name_prefix)(std::string(name) + ".cpus",
                                                         p.cpu_affinity_list);
}

std::string describe(const sirius::sirius_config& c)
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
  describe_pool(d, "task_creator", c.get_task_creator_config().thread_pool);
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
    "sm.placement", sm.rest_placement_cpus)("sm.readahead", sm.max_readahead_scans)(
    "sm.strategy", sm.readahead_strategy)("sm.width", sm.pipeline_width);
  auto const& u = sm.uring;
  d("uring.scans", u.n_max_concurrent_scans)("uring.explicit", u.n_max_concurrent_scans_explicit)(
    "uring.odirect", u.use_odirect)("uring.slices", u.slices_per_pass)(
    "uring.batch", u.range_batch_slices)("uring.service", u.low_service_interval);
  auto const& r = sm.rest;
  d("rest.scans", r.n_max_concurrent_scans)("rest.explicit", r.n_max_concurrent_scans_explicit)(
    "rest.timeout", r.request_timeout_s)("rest.stall_speed", r.stall_speed_limit_bytes)(
    "rest.stall_time", r.stall_time_s)("rest.connections", r.max_connections)(
    "rest.gap", r.merge_max_gap)("rest.upkeep", r.upkeep_interval)(
    "rest.upkeep_http1", r.upkeep_http1)("rest.age", r.conn_max_age)(
    "rest.min_segment", r.min_segment_bytes)("rest.max_segment", r.max_segment_bytes)(
    "rest.recv_buffer", r.receive_buffer_bytes)("rest.iovecs", r.max_recv_iovecs)(
    "rest.queue", r.queue_depth)("rest.wait", r.wait_mode)("rest.iowq", r.iowq_max_unbound)(
    "rest.dns_ttl", r.dns_ttl)("rest.dns_acc", r.dns_accumulate)(
    "rest.dns_max", r.dns_max_addresses)("rest.connect", r.connect_timeout)(
    "rest.ka_idle", r.keepalive_idle_s)("rest.ka_interval", r.keepalive_interval_s)(
    "rest.ka_count", r.keepalive_count)("rest.min_mtu", r.min_mtu)(
    "rest.mtu_attempts", r.mtu_reject_max_attempts)("rest.rst", r.close_with_rst)(
    "rest.fan_out", r.fan_out)("rest.interface", r.interface)("rest.irq", r.pin_to_irq_cpus)(
    "rest.priority", r.raise_reactor_priority)("rest.retries", r.max_retry_attempts)(
    "rest.auth_retries", r.max_auth_retry_attempts)("rest.backoff", r.retry_backoff_base)(
    "rest.jitter", r.retry_jitter)("rest.retry_after", r.honor_retry_after)(
    "rest.footer", r.footer_probe_bytes)("rest.list_matches", r.list_max_matches)(
    "rest.list_scanned", r.list_max_scanned);
  auto const& t = r.tls;
  d("tls.ktls", t.ktls)("tls.rx", t.fallback_rx)("tls.placement", t.placement)(
    "tls.min", t.min_version)("tls.max", t.max_version)("tls.no_pad", t.expect_no_pad)(
    "tls.cache", t.session_cache)("tls.per_endpoint", t.session_cache_per_endpoint)(
    "tls.conn_bytes", t.max_connection_bytes)("tls.suites", t.ciphersuites)(
    "tls.ciphers", t.cipher_list)("tls.groups", t.groups)("tls.sni", t.sni_override)(
    "tls.rekey", t.allow_tls13_without_rekey);
  auto const& k = sm.cache;
  d("cache.mode", k.mode)("cache.eviction", k.eviction)("cache.prefetch",
                                                        k.min_prefetching_budget_fraction)(
    "cache.threshold", k.eviction_threshold_fraction)("cache.dispose", k.dispose_on_idle);
  auto const& m = sm.memory_prefetcher;
  d("prefetch.enable", m.enable)("prefetch.threads", m.num_threads)(
    "prefetch.free", m.min_free_fraction)("prefetch.poll", m.poll_interval_ms)("prefetch.quiet",
                                                                               m.drain_quiet_ms);
  auto const& p = sm.preparation;
  d("prep.meta", p.max_inflight_metadata_jobs)("prep.dv", p.max_inflight_dv_jobs)(
    "prep.lookahead", p.lookahead_batches)("prep.control", p.max_control_work)(
    "prep.drain", p.drain_quantum)("prep.coordinators", p.coordinator_workers)(
    "prep.workers", p.native_reader_workers)("prep.readers", p.max_native_readers)(
    "prep.interrupt", p.interrupt_check_interval)("prep.partial", p.partial_batch.has_value());
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
    "op.fs_sources", o.fused_scan.max_membership_sources)("op.scan_batch", o.scan_task_batch_size)(
    "op.sort_bytes", o.max_sort_partition_bytes)(
    "op.sort_fraction", o.max_sort_partition_memory_fraction)("op.hash", o.hash_partition_bytes)(
    "op.concat", o.concat_batch_bytes)("op.cross", o.cross_join_task_bytes)(
    "op.sample", o.sort_sample_bytes)("op.build_table", o.max_build_hash_table_bytes)(
    "op.resident", o.max_build_probe_resident_bytes)("op.broadcast", o.max_broadcast_join_size)(
    "op.mark", o.mark_join_build_switch_ratio)(
    "op.distinct", o.enable_runtime_distinct_build_probe)("op.df", o.enable_dynamic_filter)(
    "op.df_multi", o.enable_dynamic_filter_multi_partition)(
    "op.df_zone", o.enable_dynamic_zone_map_filter)("op.df_cover",
                                                    o.dynamic_filter_domain_coverage_threshold)(
    "op.df_l2", o.dynamic_filter_inlist_max_l2_fraction)(
    "op.df_keep", o.dynamic_filter_keep_threshold)("op.size_est", o.enable_runtime_size_estimation)(
    "op.sum_bound", o.enable_decimal_sum_stats_bound)(
    "op.zone_prune", o.enable_pinned_zone_map_pruning)("op.hw", o.use_hw_decompression)(
    "op.cm", o.enable_compressed_materialization)("op.unique", o.prove_unique_keys)(
    "op.unique_bytes", o.unique_exact_max_bytes)("op.dense", o.enable_dense_count_join)(
    "op.delim", o.enable_delim_aggregate_reuse)("op.minmax", o.enable_exists_minmax_rewrite)(
    "op.partial", o.enable_partial_aggregate_below_join)("op.carry", o.enable_subquery_key_carry)(
    "op.exists", o.enable_exists_from_aggregate)("op.dense_bytes", o.dense_count_join_max_bytes)(
    "op.dense_fraction", o.dense_count_join_memory_fraction)(
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

/// The resolved configuration's description, or "error" when resolving throws.
template <class Resolve>
std::string resolved(Resolve&& resolve)
{
  try {
    return describe(resolve());
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const std::exception&) {
    return "error";
  }
}

}  // namespace

TEST_CASE("create_sirius_config resolves every fixture as resolve() does",
          "[config][config_create]")
{
  auto const& topology = this_machine();
  std::size_t compared = 0, resolved_ok = 0;
  for (auto const& entry : std::filesystem::directory_iterator(data_dir())) {
    if (entry.path().extension() != ".yaml" ||
        entry.path().filename().string().starts_with("invalid_")) {
      continue;
    }
    INFO(entry.path().filename().string());
    auto const old_config = resolved([&] {
      return sirius::parsed_sirius_config::from_file(entry.path()).resolve(topology, no_nic);
    });
    auto const new_config = resolved([&] {
      return sirius::create_sirius_config(
        sirius::config::read_user_config(slurp(entry.path())), topology, no_nic);
    });
    REQUIRE(new_config == old_config);
    ++compared;
    resolved_ok += old_config != "error" ? 1 : 0;
  }
  REQUIRE(compared >= 30);
  REQUIRE(resolved_ok >= 20);
}

TEST_CASE("an empty configuration resolves as the default one", "[config][config_create]")
{
  auto const& topology = this_machine();
  REQUIRE(describe(sirius::create_sirius_config(
            sirius::config::default_user_config(), topology, no_nic)) ==
          describe(sirius::parsed_sirius_config{}.resolve(topology, no_nic)));
}

TEST_CASE("REST limits that depend on a derived value are checked after derivation",
          "[config][config_create]")
{
  // An unnamed max_connections derives from queue_depth (half of it), so a small queue no longer
  // fails against the struct default of 64.
  auto const user = sirius::config::read_user_config(
    "sirius:\n  executor:\n    scan_manager:\n      rest:\n        queue_depth: 100\n");
  auto const config = sirius::create_sirius_config(user, this_machine(), no_nic);
  REQUIRE(config.get_scan_manager_config().rest.max_connections == 50);
}

TEST_CASE("rules across blocks hold for a configuration built in code", "[config][config_create]")
{
  auto user = sirius::config::read_user_config("sirius:\n  space:\n    gpu: [{}]\n");
  user.memory.gpu.usage_limit.assign(0.5);
  REQUIRE_THROWS_AS(sirius::create_sirius_config(user, this_machine(), no_nic),
                    sirius::configuration_input_error);
}
