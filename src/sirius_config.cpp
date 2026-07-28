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

#include "exec/config.hpp"
#include "log/logging.hpp"
#include "yaml_reader.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <yaml-cpp/yaml.h>

#include <exception>
#include <variant>
#include <vector>

namespace sirius {

// ================ from_yaml for external types ================= //

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
  r.optional("thread_name_prefix", opt.thread_name_prefix);
  r.optional("cpu_affinity", opt.cpu_affinity_list);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, creator::task_creator_config& opt)
{
  yaml::reader r(node, "task_creator");
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{0});
  r.optional("thread_name_prefix", opt.thread_pool.thread_name_prefix);
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  r.optional("strategy", opt.strategy);
  r.optional("priority_order", opt.priority);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, sirius::io::object_store_config& opt)
{
  yaml::reader r(node, "object_store");
  r.optional("endpoint", opt.endpoint);
  r.optional("region", opt.region);
  r.optional("access_key", opt.access_key);
  r.optional("secret_key", opt.secret_key);
  r.optional("session_token", opt.session_token);
  r.optional("s3_transport", opt.s3_transport);
  r.optional("signing_mode", opt.s3_signing_mode);
  r.optional("ca_bundle_path", opt.ca_bundle_path);
  r.optional("tls_verify", opt.tls_verify);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, sirius::io::rest::config& opt)
{
  yaml::reader r(node, "rest");
  r.optional("request_timeout_s", opt.request_timeout_s);
  r.optional("ca_bundle_path", opt.ca_bundle_path);
  r.optional("tls_verify", opt.tls_verify);
  r.optional("max_connections", opt.max_connections);
  r.optional("chunk_size", yaml::bytes(opt.chunk_size));
  r.optional("max_n_chunks", opt.max_n_chunks);
  r.optional("max_read_split", opt.max_read_split);
  r.optional("bounce_block_size", yaml::bytes(opt.bounce_block_size));
  r.optional("upkeep_interval_ms", opt.upkeep_interval);
  r.optional("conn_max_age_s", opt.conn_max_age);
  r.optional("retry_backoff_base_ms", opt.retry_backoff_base);
  r.optional("retry_jitter_ms", opt.retry_jitter);
  r.optional("max_retry_attempts", opt.max_retry_attempts);
  r.optional("max_auth_retry_attempts", opt.max_auth_retry_attempts);
  r.optional("honor_retry_after", opt.honor_retry_after);
  r.optional("perf_instrumentation", opt.perf_instrumentation);
  r.optional("footer_probe_bytes", yaml::bytes(opt.footer_probe_bytes));
  r.optional("list_max_matches", opt.list_max_matches);
  r.optional("list_max_scanned", opt.list_max_scanned);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, sirius::io::uring::config& opt)
{
  yaml::reader r(node, "local");
  r.optional("use_odirect", opt.use_odirect);
  r.optional("max_n_chunks", opt.max_n_chunks);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, sirius::io::cache::config& opt)
{
  yaml::reader r(node, "cache");
  r.optional(
    "inflight_io_chunk_budget", opt.inflight_io_chunk_budget, yaml::greater_than<std::size_t>{0});
  r.optional("dispose_after_use", opt.dispose_after_use);
  r.optional("min_prefetching_budget_fraction",
             opt.min_prefetching_budget_fraction,
             yaml::fraction<double>{});
  r.optional(
    "eviction_threshold_fraction", opt.eviction_threshold_fraction, yaml::fraction<double>{});
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, scan_manager::scan_manager_config& opt)
{
  yaml::reader r(node, "scan_manager");
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{2});
  r.optional("thread_name_prefix", opt.thread_pool.thread_name_prefix);
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  r.optional("use_sirius_datasource", opt.use_sirius_datasource);
  r.optional("uring_n_reactors", opt.uring_n_reactors, yaml::greater_than<std::size_t>{0});
  r.optional("rest_n_reactors", opt.rest_n_reactors, yaml::greater_than<std::size_t>{0});
  r.optional("enable_prefetch_cache", opt.enable_prefetch_cache);
  if (auto n = r.optional_node("local")) sirius::from_yaml(*n, opt.local);
  if (auto n = r.optional_node("rest")) sirius::from_yaml(*n, opt.rest);
  if (auto n = r.optional_node("cache")) sirius::from_yaml(*n, opt.cache);
  if (auto n = r.optional_node("object_store")) sirius::from_yaml(*n, opt.object_store);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, operator_params& opt)
{
  yaml::reader r(node, "operator_params");
  r.optional("scan_task_batch_size", yaml::bytes(opt.scan_task_batch_size));
  r.optional("max_sort_partition_bytes", yaml::bytes(opt.max_sort_partition_bytes));
  r.optional("max_sort_partition_memory_fraction",
             opt.max_sort_partition_memory_fraction,
             yaml::fraction<double>{});
  r.optional("hash_partition_bytes", yaml::bytes(opt.hash_partition_bytes));
  r.optional("concat_batch_bytes", yaml::bytes(opt.concat_batch_bytes));
  r.optional("sort_sample_bytes", yaml::bytes(opt.sort_sample_bytes));
  r.optional("max_build_hash_table_bytes", yaml::bytes(opt.max_build_hash_table_bytes));
  r.optional("max_broadcast_join_size", yaml::bytes(opt.max_broadcast_join_size));
  r.optional("mark_join_build_switch_ratio", opt.mark_join_build_switch_ratio);
  r.optional("enable_dynamic_filter_pushdown", opt.enable_dynamic_filter_pushdown);
  r.optional("enable_dynamic_zone_map_filter", opt.enable_dynamic_zone_map_filter);
  r.optional("dynamic_filter_domain_coverage_threshold",
             opt.dynamic_filter_domain_coverage_threshold);
  r.optional("dynamic_filter_keep_threshold", opt.dynamic_filter_keep_threshold);
  r.optional("enable_pinned_zone_map_pruning", opt.enable_pinned_zone_map_pruning);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, telemetry_config& opt)
{
  yaml::reader r(node, "telemetry");
  r.optional("enable_quent", opt.enable_quent);
  r.optional("enable_batch_events", opt.enable_batch_events);
  r.optional("output_directory", opt.output_directory);
  r.optional("engine_name", opt.engine_name);
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, compression_config& opt)
{
  yaml::reader r(node, "compression");
  r.optional("enable_pin_table_compression", opt.enable_pin_table_compression);
  r.optional("min_batch_size_bytes", yaml::bytes(opt.min_batch_size_bytes));
  r.optional("max_compressed_fraction", opt.max_compressed_fraction);
  r.optional("input_plan_dir", opt.input_plan_dir);
  r.optional("column_threads", opt.column_threads, yaml::greater_than<int>{0});
  r.optional("enable_spill_compression", opt.enable_spill_compression);
  r.optional(
    "spill_explore_beam_width", opt.spill_explore_beam_width, yaml::greater_than<uint32_t>{0});
  r.optional("spill_explore_max_bytes", yaml::bytes(opt.spill_explore_max_bytes));
  r.optional("spill_replan_after_uses", opt.spill_replan_after_uses);
  r.optional("spill_error_tolerance", opt.spill_error_tolerance, yaml::greater_than<uint32_t>{0});
  r.optional("spill_replan_change_threshold", opt.spill_replan_change_threshold);
  r.optional("spill_explore_sample_rows", yaml::bytes(opt.spill_explore_sample_rows));
  r.reject_unknown();
}

static void from_yaml(const YAML::Node& node, exec::downgrade_executor_config& opt)
{
  yaml::reader r(node, "downgrade");
  r.optional("num_threads", opt.thread_pool.num_threads, yaml::greater_than<int>{0});
  r.optional("thread_name_prefix", opt.thread_pool.thread_name_prefix);
  r.optional("cpu_affinity", opt.thread_pool.cpu_affinity_list);
  r.optional("monitor_period", opt.monitor_period);
  r.reject_unknown();
}

namespace {

struct topology {
  std::variant<size_t, std::vector<int>> num_gpus_or_gpu_ids{size_t{1}};

  static void from_yaml(const YAML::Node& node, topology& opt)
  {
    yaml::reader r(node, "topology");
    // gpu_ids and num_gpus are mutually exclusive; try gpu_ids first
    std::vector<int> ids;
    r.optional("gpu_ids", ids);
    if (!ids.empty()) {
      opt.num_gpus_or_gpu_ids = std::move(ids);
    } else {
      size_t n = 1;
      r.optional("num_gpus", n);
      opt.num_gpus_or_gpu_ids = n;
    }
    r.reject_unknown();
  }
};

struct gpu_mem_config {
  std::variant<double, std::uint64_t> usage_limit{0.95};
  std::variant<double, std::uint64_t> reservation_limit{0.9};
  double downgrade_trigger_fraction{1.0};
  double downgrade_stop_fraction{0.7};
  bool track_per_stream_reservation{false};

  static void from_yaml(const YAML::Node& node, gpu_mem_config& opt)
  {
    opt.track_per_stream_reservation = false;
    yaml::reader r(node, "memory.gpu");
    // usage_limit: fraction (double) or absolute bytes — mutually exclusive keys
    std::optional<std::uint64_t> usage_bytes;
    double usage_frac = 0.95;
    r.optional("usage_limit_bytes", yaml::bytes(usage_bytes));
    r.optional("usage_limit_fraction", usage_frac, yaml::fraction<double>{});
    opt.usage_limit = usage_bytes ? std::variant<double, std::uint64_t>{*usage_bytes}
                                  : std::variant<double, std::uint64_t>{usage_frac};
    // reservation_limit: fraction or absolute bytes
    std::optional<std::uint64_t> res_bytes;
    double res_frac = 0.9;
    r.optional("reservation_limit_bytes", yaml::bytes(res_bytes));
    r.optional("reservation_limit_fraction", res_frac, yaml::fraction<double>{});
    opt.reservation_limit = res_bytes ? std::variant<double, std::uint64_t>{*res_bytes}
                                      : std::variant<double, std::uint64_t>{res_frac};
    r.optional(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, yaml::fraction<double>{});
    r.optional("downgrade_stop_fraction", opt.downgrade_stop_fraction, yaml::fraction<double>{});
    r.optional("track_per_stream_reservation", opt.track_per_stream_reservation);
    r.reject_unknown();
  }

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    if (std::holds_alternative<double>(usage_limit)) {
      builder.set_usage_limit_ratio_per_gpu(std::get<double>(usage_limit));
    } else {
      builder.set_gpu_usage_limit(std::get<std::uint64_t>(usage_limit));
    }
    if (std::holds_alternative<double>(reservation_limit)) {
      builder.set_reservation_fraction_per_gpu(std::get<double>(reservation_limit));
    } else {
      builder.set_reservation_limit_per_gpu(std::get<std::uint64_t>(reservation_limit));
    }
    builder.set_downgrade_fractions_per_gpu(downgrade_trigger_fraction, downgrade_stop_fraction);
    builder.track_reservation_per_stream(track_per_stream_reservation);
  }
};

struct host_mem_config {
  std::uint64_t numa_region_capacity_bytes = 8UL << 30;  // 8GB per NUMA node
  std::variant<double, std::uint64_t> reservation_limit{0.9};
  double downgrade_trigger_fraction{0.8};
  double downgrade_stop_fraction{0.7};
  std::size_t block_size{cucascade::memory::default_block_size};
  std::size_t pool_size{cucascade::memory::default_pool_size};
  std::size_t initial_number_pools{cucascade::memory::default_initial_number_pools};

  static void from_yaml(const YAML::Node& node, host_mem_config& opt)
  {
    yaml::reader r(node, "memory.host");
    r.optional("capacity_bytes", yaml::bytes(opt.numa_region_capacity_bytes));
    std::optional<std::uint64_t> res_bytes;
    double res_frac = 0.9;
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
  }

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    // cucascade builds one numa_region_pinned_host_memory_resource per
    // distinct NUMA node when the configurator sees this call. Relied upon by
    // SiriusContext::initialize() which asserts host_spaces.size() ==
    // topology.num_numa_nodes on the default path. YAML configs may override
    // by explicitly setting per-space numa_id.
    builder.use_host_per_numa();
    if (std::holds_alternative<double>(reservation_limit)) {
      builder.set_reservation_fraction_per_host(std::get<double>(reservation_limit));
    } else {
      builder.set_reservation_limit_per_host(std::get<std::uint64_t>(reservation_limit));
    }
    builder.set_downgrade_fractions_per_host(downgrade_trigger_fraction, downgrade_stop_fraction);
    builder.set_per_host_capacity(numa_region_capacity_bytes);
    // NOTE on argument order: cucascade's set_host_pool_features has confusingly-named
    // parameters (chunk_size, block_size, initial_block_count) that it internally remaps onto
    // host_memory_space_config::{block_size, pool_size, initial_number_pools} (see
    // reservation_manager_configurator::build()). Passing our {block_size, pool_size,
    // initial_number_pools} positionally therefore lands each value in the correctly-named
    // cucascade config field — the names line up with the resulting config struct, not with the
    // setter's parameter names.
    builder.set_host_pool_features(block_size, pool_size, initial_number_pools);
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

}  // namespace

// ================ sirius_config ================= //

sirius_config::sirius_config()
{
  cucascade::memory::topology_discovery discovery;
  if (discovery.discover()) { _hw_topology = discovery.get_topology(); }
}

void sirius_config::apply_defaults()
{
  // Run the configurator with default values to populate memory space configs
  topology topo;
  gpu_mem_config gpu_cfg;
  host_mem_config host_cfg;
  disk_mem_config disk_cfg;

  cucascade::memory::reservation_manager_configurator builder;
  builder.set_number_of_gpus(std::get<size_t>(topo.num_gpus_or_gpu_ids));
  gpu_cfg.setup_configurator(builder);
  host_cfg.setup_configurator(builder);
  disk_cfg.setup_configurator(builder);
  _memory_space_configs = builder.build(_hw_topology);
}

void sirius_config::load_from_file(const std::filesystem::path& config_path)
{
  try {
    YAML::Node root;
    try {
      root = YAML::LoadFile(config_path.string());
    } catch (const YAML::Exception& e) {
      throw std::runtime_error("failed to parse YAML: " + std::string(e.what()));
    }

    yaml::reader top(root);
    auto sirius_node = top.optional_node("sirius");
    top.reject_unknown();

    if (!sirius_node) { throw std::runtime_error("missing top-level 'sirius' key"); }

    yaml::reader r(*sirius_node, "sirius");

    // Topology
    topology topo;
    r.optional("topology", topo);

    // High-level memory config (mutually exclusive with space config)
    gpu_mem_config gpu_cfg;
    host_mem_config host_cfg;
    disk_mem_config disk_cfg;

    if (auto mem_node = r.optional_node("memory")) {
      yaml::reader mr(*mem_node, "sirius.memory");
      if (auto n = mr.optional_node("gpu")) gpu_mem_config::from_yaml(*n, gpu_cfg);
      if (auto n = mr.optional_node("host")) host_mem_config::from_yaml(*n, host_cfg);
      if (auto n = mr.optional_node("disk")) disk_mem_config::from_yaml(*n, disk_cfg);
      mr.reject_unknown();
    }

    // Executors
    if (auto exec_node = r.optional_node("executor")) {
      yaml::reader er(*exec_node, "sirius.executor");
      if (auto n = er.optional_node("task_creator")) from_yaml(*n, _task_creator_config);
      if (auto n = er.optional_node("scan_manager")) from_yaml(*n, _scan_manager_config);
      if (auto n = er.optional_node("pipeline")) from_yaml(*n, _gpu_pipeline_executor_config);
      if (auto n = er.optional_node("downgrade")) from_yaml(*n, _downgrade_executor_config);
      er.reject_unknown();
    }

    // Operator params
    if (auto n = r.optional_node("operator_params")) { sirius::from_yaml(*n, _operator_params); }

    // Telemetry
    if (auto n = r.optional_node("telemetry")) { sirius::from_yaml(*n, _telemetry_config); }

    // Compression
    if (auto n = r.optional_node("compression")) { sirius::from_yaml(*n, _compression_config); }

    // Explicit space configs (low-level API)
    std::vector<cucascade::memory::gpu_memory_space_config> gpu_space_configs;
    std::vector<cucascade::memory::host_memory_space_config> host_space_configs;
    std::vector<cucascade::memory::disk_memory_space_config> disk_space_configs;

    if (auto space_node = r.optional_node("space")) {
      yaml::reader sr(*space_node, "sirius.space");
      if (auto n = sr.optional_node("gpu")) read_yaml_vec(*n, gpu_space_configs);
      if (auto n = sr.optional_node("host")) read_yaml_vec(*n, host_space_configs);
      if (auto n = sr.optional_node("disk")) read_yaml_vec(*n, disk_space_configs);
      sr.reject_unknown();
    }

    r.reject_unknown();

    // Build memory space configs
    _memory_space_configs.clear();

    std::copy(gpu_space_configs.begin(),
              gpu_space_configs.end(),
              std::back_inserter(_memory_space_configs));
    std::copy(host_space_configs.begin(),
              host_space_configs.end(),
              std::back_inserter(_memory_space_configs));
    std::copy(disk_space_configs.begin(),
              disk_space_configs.end(),
              std::back_inserter(_memory_space_configs));

    bool using_configurator = _memory_space_configs.empty();
    if (using_configurator) {
      cucascade::memory::reservation_manager_configurator builder;
      if (std::holds_alternative<size_t>(topo.num_gpus_or_gpu_ids)) {
        builder.set_number_of_gpus(std::get<size_t>(topo.num_gpus_or_gpu_ids));
      } else {
        const auto& gpu_ids = std::get<std::vector<int>>(topo.num_gpus_or_gpu_ids);
        builder.set_gpu_ids(gpu_ids);
      }
      gpu_cfg.setup_configurator(builder);
      host_cfg.setup_configurator(builder);
      disk_cfg.setup_configurator(builder);
      _memory_space_configs = builder.build(_hw_topology);
    }

    enforce_sirius_datasource_for_multi_gpu();

  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to load config from " + config_path.string() + ": " +
                             e.what());
  }
}

void sirius_config::enforce_sirius_datasource_for_multi_gpu()
{
  size_t num_gpus = std::ranges::count_if(_memory_space_configs, [](auto const& space) {
    return std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space);
  });
  if (num_gpus > 1 && !_scan_manager_config.use_sirius_datasource) {
    SIRIUS_LOG_WARN(
      "sirius_config: use_sirius_datasource was false but {} GPUs are configured; "
      "the sirius datasource is required for multi-GPU IO routing. Overriding "
      "use_sirius_datasource to true.",
      num_gpus);
    _scan_manager_config.use_sirius_datasource = true;
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

}  // namespace sirius
