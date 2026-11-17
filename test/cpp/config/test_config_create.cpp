// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0

#include "catch.hpp"
#include "config/user_config.hpp"
#include "sirius_config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/net_topology.hpp>
#include <cucascade/memory/topology_discovery.hpp>

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
    "sm.placement", sm.rest_placement_cpus)("sm.readahead", sm.max_readahead_scans)(
    "sm.strategy", sm.readahead_strategy)("sm.width", sm.pipeline_width);
  auto const& u = sm.uring;
  d("uring.scans", u.n_max_concurrent_scans)("uring.explicit", u.n_max_concurrent_scans_explicit)(
    "uring.odirect", u.use_odirect)("uring.slices", u.slices_per_pass)(
    "uring.batch", u.range_batch_slices)("uring.service", u.low_service_interval);
  auto const& r = sm.rest;
  d("rest.scans", r.n_max_concurrent_scans)("rest.explicit", r.n_max_concurrent_scans_explicit)(
    "rest.timeout", r.request_timeout_s)("rest.stall_speed", r.stall_speed_limit_bytes)(
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
void require_same(const outcome& old_path, const outcome& new_path, std::string_view source)
{
  INFO("old error: " << old_path.error);
  INFO("new error: " << new_path.error_type << ": " << new_path.error);
  REQUIRE(new_path.description == old_path.description);
  if (old_path.error.empty()) {
    REQUIRE(new_path.error.empty());
    return;
  }
  REQUIRE(new_path.error_type == "configuration_input_error");
  auto expected = old_path.error;
  if (!source.empty() && expected.starts_with(std::string(source) + ": ")) {
    expected.erase(0, source.size() + 2);
  }
  REQUIRE(new_path.error == expected);
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
    auto const old_path = resolved([&] {
      return sirius::parsed_sirius_config::from_file(entry.path()).resolve(topology, no_nic);
    });
    auto const new_path = resolved([&] {
      return sirius::create_sirius_config(
        sirius::config::read_user_config(slurp(entry.path())), topology, no_nic);
    });
    require_same(old_path, new_path, entry.path().string());
    ++compared;
    if (name == "valid_every_key.yaml") {
      every_key_seen = true;
      REQUIRE_FALSE(new_path.description.empty());  // every key, resolved
    }
  }
  REQUIRE(compared >= 30);
  REQUIRE(every_key_seen);
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
    "'sirius.memory.gpu': downgrade_stop_fraction must be less than downgrade_trigger_fraction");

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
