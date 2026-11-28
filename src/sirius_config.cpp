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
#include "config/yaml.hpp"
#include "config/yaml_cpp.hpp"
#include "exec/config.hpp"
#include "exec/thread_defaults.hpp"
#include "exec/thread_util.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"

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
#include <unistd.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
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

/// The operator option @p setting is configured by, found by key so the two cannot disagree.
template <class Options>
auto& batch_option(Options& op, const batch_setting& setting)
{
  std::array const options{&op.scan_task_batch_size,
                           &op.hash_partition_bytes,
                           &op.concat_batch_bytes,
                           &op.cross_join_task_bytes,
                           &op.sort_sample_bytes,
                           &op.max_build_hash_table_bytes,
                           &op.max_build_probe_resident_bytes};
  static_assert(std::tuple_size_v<decltype(options)> == std::size(batch_settings));
  for (auto* option : options) {
    if (option->key() == setting.name) { return *option; }
  }
  throw std::logic_error(std::string("no operator option for batch setting ") + setting.name);
}

}  // namespace

operator_params operator_params::with_batch_size(uint64_t batch_size)
{
  operator_params result{};
  for (const auto& setting : batch_settings) {
    result.*setting.member = batch_size * setting.multiplier;
  }
  return result;
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

// The REST fan-out against the reactor count, named under @p path ("rest" or a full path).
static std::optional<std::string> rest_fan_out_error(scan_manager::scan_manager_config const& opt,
                                                     std::string_view path)
{
  if (opt.rest_n_reactors > 0 && opt.rest.fan_out > opt.rest_n_reactors &&
      opt.rest.fan_out != cucascade::io::rest::config{}.fan_out) {
    return "'" + std::string(path) + ".fan_out': must be at most rest_n_reactors (" +
           std::to_string(opt.rest_n_reactors) + ")";
  }
  return std::nullopt;
}

namespace {

struct topology {
  /// 0 = auto: use every GPU visible to topology discovery (CUDA_VISIBLE_DEVICES-aware).
  std::variant<size_t, std::vector<int>> num_gpus_or_gpu_ids{size_t{0}};
  /// 0 = all active GPUs per query (default). Positive value limits each query to the first
  /// N entries of the sorted active-GPU list, reserving the rest for future concurrent queries.
  int gpus_per_query{0};
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

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    if (downgrade_root_dirs.empty() || capacity_bytes == 0) { return; }
    builder.set_disk_mounting_point(id, capacity_bytes, downgrade_root_dirs);
  }
};

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
  /// What the user configured, read back: user-defined values, fallbacks for the rest.
  config::user_config user;
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

parsed_sirius_config::parsed_sirius_config()
  : impl_([] {
      auto impl = std::make_shared<Impl>();
      impl->user.read(config::node{}, config::user_config_root);
      return impl;
    }())
{
}
parsed_sirius_config::parsed_sirius_config(std::shared_ptr<const Impl> impl)
  : impl_(std::move(impl))
{
}

const config::user_config& parsed_sirius_config::get_user_config() const noexcept
{
  return impl_->user;
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
  std::string text;
  try {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw configuration_input_error(path.string() + ": cannot open the file"); }
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } catch (const std::ios_base::failure&) {
    throw configuration_input_error(path.string() + ": cannot read the file");
  }
  config::node document;
  try {
    document = config::parse_yaml(text);
  } catch (const config::config_error& e) {
    throw configuration_input_error(path.string() + ": " + e.what());
  }
  return from_document(document, path);
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
{
  config::node document;
  try {
    document = config::from_yaml_cpp(root);
  } catch (const config::config_error& e) {
    throw configuration_input_error(source_path.empty() ? e.what()
                                                        : source_path.string() + ": " + e.what());
  }
  return from_document(document, source_path);
}

parsed_sirius_config parsed_sirius_config::from_document(const config::node& document,
                                                         const std::filesystem::path& source_path)
try {
  return from_user_config(config::read_user_config(document), source_path);
} catch (const config::config_error& e) {
  throw configuration_input_error(source_path.empty() ? e.what()
                                                      : source_path.string() + ": " + e.what());
} catch (const configuration_input_error& e) {
  if (source_path.empty()) { throw; }
  throw configuration_input_error(source_path.string() + ": " + e.what());
}

parsed_sirius_config parsed_sirius_config::from_user_config(const config::user_config& input)
{
  return from_user_config(input, {});
}

parsed_sirius_config parsed_sirius_config::from_user_config(
  const config::user_config& input, const std::filesystem::path& source_path)
try {
  // Read back what the user defined: fallbacks fill in the rest and every block checks its rules,
  // for a configuration built in code (or never read) as much as for one read from a file.
  auto impl = std::make_shared<Impl>();
  impl->user.read(input.write(false), config::user_config_root);
  auto const& user  = impl->user;
  impl->source_path = source_path;
  auto& settings    = impl->settings;
  auto& named       = impl->explicit_threads;

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
  sm_cfg.reserve_rest_reactor_cpus  = sm.reserve_rest_reactor_cpus.get();
  sm_cfg.httpfs_external_file_cache = sm.httpfs_external_file_cache.get();
  if (auto const* scans = sm.max_readahead_scans.get_if()) { sm_cfg.max_readahead_scans = *scans; }
  if (auto const* strategy = sm.readahead_strategy.get_if()) {
    sm_cfg.readahead_strategy = *strategy;
  }
  named.scan_manager          = sm.num_threads.user_defined();
  sm_cfg.num_threads_explicit = named.scan_manager;
  named.uring                 = sm.uring_n_reactors.user_defined();
  named.rest                  = sm.rest_n_reactors.user_defined();

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
  ops.enable_dictionary_carrier                = op.enable_dictionary_carrier.get();
  ops.enable_dense_count_join                  = op.enable_dense_count_join.get();
  ops.enable_delim_aggregate_reuse             = op.enable_delim_aggregate_reuse.get();
  ops.prove_unique_keys                        = op.pin_table_prove_unique_keys.get();
  ops.unique_exact_max_bytes                   = op.pin_table_unique_exact_max_bytes.get().count();
  ops.enable_exists_minmax_rewrite             = op.enable_exists_minmax_rewrite.get();
  ops.enable_partial_aggregate_below_join      = op.enable_partial_aggregate_below_join.get();
  ops.enable_subquery_key_carry                = op.enable_subquery_key_carry.get();
  ops.enable_exists_from_aggregate             = op.enable_exists_from_aggregate.get();
  ops.enable_late_dimension_join               = op.enable_late_dimension_join.get();
  ops.parquet_footer_statistics_max_remote_reads =
    op.parquet_footer_statistics_max_remote_reads.get();
  ops.parquet_footer_statistics_max_local_reads =
    op.parquet_footer_statistics_max_local_reads.get();
  ops.admission_bytes_per_gpu        = op.admission_bytes_per_gpu.get().count();
  ops.avg_variable_column_bytes      = op.avg_variable_column_bytes.get().count();
  ops.enable_runtime_size_estimation = op.enable_runtime_size_estimation.get();
  ops.use_hw_decompression           = op.use_hw_decompression.get();
  // Unset batch sizes are derived from GPU memory when the configuration is resolved.
  for (auto const& setting : batch_settings) {
    if (auto const* value = batch_option(op, setting).get_if()) {
      ops.*setting.member = value->count();
      impl->explicit_batch_settings.push_back(setting.member);
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
  compression.min_table_device_fraction     = c.min_table_device_fraction.get();
  compression.enable_spill_compression      = c.enable_spill_compression.get();
  compression.spill_min_batch_bytes         = c.spill_min_batch_bytes.get().count();
  compression.spill_max_compressed_fraction = c.spill_max_compressed_fraction.get();
  compression.spill_encode_reserve_fraction = c.spill_encode_reserve_fraction.get();

  settings._parquet_output_compression = user.output.parquet.compression.get();

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

config::user_config effective_user_config(const config::user_config& user,
                                          const sirius_config& resolved)
{
  config::user_config out;
  out.read(user.write(false), config::user_config_root);
  // What resolution derived for an option the user left to it, as the user would have written it.
  auto const derived = [](auto& option, auto value) {
    if (!option.user_defined()) { option.assign(std::move(value)); }
  };
  // Affinities lose the CPUs the REST reactors pin to, so the effective lists always apply.
  auto const in_effect = [](auto& option, auto value) {
    if (!option.get_if() || *option.get_if() != value) { option.assign(std::move(value)); }
  };

  std::size_t gpu_spaces = 0;
  std::vector<std::size_t> host_pools;
  for (auto const& space : resolved.get_memory_space_configs()) {
    if (std::holds_alternative<cucascade::memory::gpu_memory_space_config>(space)) { ++gpu_spaces; }
    if (auto const* host = std::get_if<cucascade::memory::host_memory_space_config>(&space)) {
      host_pools.push_back(host->initial_number_pools);
    }
  }
  if (!out.space.configured()) {
    derived(out.topology.gpus, gpu_spaces);
    // One count for every NUMA node, when the derived counts agree.
    if (!host_pools.empty() &&
        std::ranges::adjacent_find(host_pools, std::ranges::not_equal_to{}) == host_pools.end()) {
      derived(out.memory.host.initial_number_pools, host_pools.front());
    }
  }

  auto& executor = out.executor;
  derived(executor.task_creator.num_threads,
          resolved.get_task_creator_config().thread_pool.num_threads);
  in_effect(executor.task_creator.cpu_affinity,
            resolved.get_task_creator_config().thread_pool.cpu_affinity_list);
  derived(executor.pipeline.num_threads, resolved.get_gpu_pipeline_executor_config().num_threads);
  auto const& downgrade = resolved.get_downgrade_executor_config();
  derived(executor.downgrade.num_threads, downgrade.thread_pool.num_threads);
  in_effect(executor.downgrade.cpu_affinity, downgrade.thread_pool.cpu_affinity_list);

  auto const& sm_cfg = resolved.get_scan_manager_config();
  auto& sm           = executor.scan_manager;
  derived(sm.num_threads, sm_cfg.thread_pool.num_threads);
  in_effect(sm.cpu_affinity, sm_cfg.thread_pool.cpu_affinity_list);
  derived(sm.uring_n_reactors, sm_cfg.uring_n_reactors);
  derived(sm.rest_n_reactors, sm_cfg.rest_n_reactors);
  derived(sm.uring.n_max_concurrent_scans, sm_cfg.uring.n_max_concurrent_scans);
  derived(sm.rest.n_max_concurrent_scans, sm_cfg.rest.n_max_concurrent_scans);
  derived(sm.rest.max_connections, sm_cfg.rest.max_connections);
  derived(sm.rest.min_mtu, config::ubytes(sm_cfg.rest.min_mtu));

  // The preparation limits as the scan manager resolves them from its worker count.
  auto const& prep_cfg = sm_cfg.preparation;
  auto const prep      = prep_cfg.resolve(sm_cfg.thread_pool.num_threads);
  auto& prep_options   = sm.preparation;
  derived(prep_options.max_inflight_metadata_jobs, prep.max_inflight_metadata_jobs);
  derived(prep_options.max_inflight_dv_jobs, prep.max_inflight_dv_jobs);
  derived(prep_options.lookahead_batches, prep.lookahead_batches);
  derived(prep_options.max_control_work, prep.max_control_work);
  derived(prep_options.drain_quantum, prep.drain_quantum);
  derived(prep_options.coordinator_workers, prep_cfg.resolved_coordinator_workers());
  derived(prep_options.native_reader_workers, prep_cfg.resolved_native_workers());
  derived(prep_options.max_native_readers, prep_cfg.resolved_native_readers());

  for (auto const& setting : batch_settings) {
    derived(batch_option(out.operator_params, setting),
            config::ubytes(resolved.get_operator_params().*setting.member));
  }
  return out;
}

namespace {

/// Add to @p into what only @p from has: keys of mappings, and the same inside elements of
/// sequences of equal length. A value @p into holds is kept.
void add_missing(config::node& into, const config::node& from)
{
  if (into.is_mapping() && from.is_mapping()) {
    for (auto& [key, value] : into.entries()) {
      if (auto const* other = from.find(key)) { add_missing(value, *other); }
    }
    for (auto const& [key, value] : from.entries()) {
      if (into.find(key) == nullptr) { into.set(key, value); }
    }
  } else if (into.is_sequence() && from.is_sequence() &&
             into.elements().size() == from.elements().size()) {
    for (std::size_t i = 0; i < into.elements().size(); ++i) {
      add_missing(into.elements()[i], from.elements()[i]);
    }
  }
}

}  // namespace

std::string effective_config_text(const sirius_config& resolved,
                                  const config::user_config& user,
                                  bool developer)
{
  auto document = effective_user_config(user, resolved)
                    .write(config::write_options{/*fallbacks=*/true, developer});
  // What the user wrote stays in the file, developer-only settings included.
  add_missing(document, user.write(false));
  auto root = config::node::mapping();
  root.set(std::string(config::user_config_root), std::move(document));
  return "# The configuration Sirius ran with on this machine, written when it started: the\n"
         "# values the config file named, and every other setting's default or the value derived\n"
         "# from this machine's hardware. To use it, copy it to ~/.sirius/sirius.yaml.\n" +
         config::emit_yaml(root);
}

void write_effective_config(const std::filesystem::path& path,
                            const sirius_config& resolved,
                            const config::user_config& user)
{
  auto const text = effective_config_text(resolved, user, config::developer_mode());
  if (path.has_parent_path()) { std::filesystem::create_directories(path.parent_path()); }
  // Replace the file whole, so a reader never sees half of it. mkstemp gives this writer a
  // temporary no other process (on this host or one sharing the directory) can open.
  auto pattern = path.string() + ".XXXXXX";
  int const fd = ::mkstemp(pattern.data());
  if (fd < 0) {
    throw std::system_error(
      errno, std::generic_category(), "cannot create a file next to " + path.string());
  }
  bool open = true;
  try {
    for (std::size_t written = 0; written < text.size();) {
      auto const n = ::write(fd, text.data() + written, text.size() - written);
      if (n < 0 && errno == EINTR) { continue; }
      if (n < 0) {
        throw std::system_error(errno, std::generic_category(), "cannot write " + pattern);
      }
      written += static_cast<std::size_t>(n);
    }
    // Released even when close() fails (Linux), so it is never closed twice.
    open = false;
    if (::close(fd) != 0) {
      throw std::system_error(errno, std::generic_category(), "cannot write " + pattern);
    }
    std::filesystem::rename(pattern, path);
  } catch (...) {
    if (open) { ::close(fd); }
    std::error_code ignored;
    std::filesystem::remove(pattern, ignored);
    throw;
  }
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
  // The largest min_mtu cuCascade accepts (an IPv4 TCP packet's 65535 bytes plus options); a NIC
  // reporting more gets no check rather than a configuration that fails to start.
  constexpr int largest_min_mtu = 65535 + 60 + 12;
  return egress_nic.has_value() && egress_nic->mtu > standard_ethernet_mtu &&
             egress_nic->mtu <= largest_min_mtu
           ? static_cast<std::size_t>(egress_nic->mtu)
           : std::size_t{0};
}

namespace {
// cuCascade's configurator rejects selections the machine cannot serve (more GPUs than it has, an
// unknown device id, a host fraction without known NUMA memory): a configuration error.
std::vector<cucascade::memory::memory_space_config> build_memory_spaces(
  cucascade::memory::reservation_manager_configurator const& builder,
  cucascade::memory::system_topology_info const& topology)
{
  try {
    return builder.build(topology);
  } catch (const std::bad_alloc&) {
    throw;
  } catch (const configuration_input_error&) {
    throw;
  } catch (const std::exception& e) {
    throw configuration_input_error(e.what());
  }
}
}  // namespace

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
      for (auto const& space : build_memory_spaces(capacity_builder, hw_topology)) {
        if (auto gpu = std::get_if<cucascade::memory::gpu_memory_space_config>(&space);
            gpu && *bytes > gpu->memory_capacity) {
          throw configuration_input_error(
            "sirius.memory.gpu.usage_limit_bytes exceeds the capacity of GPU " +
            std::to_string(gpu->device_id));
        }
      }
    }
    settings._memory_space_configs = build_memory_spaces(builder, hw_topology);
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
  if (auto error =
        rest_fan_out_error(settings._scan_manager_config, "sirius.executor.scan_manager.rest")) {
    throw configuration_input_error(*error);
  }
  settings.finalize_derived_config();
  return settings;
} catch (const std::bad_alloc&) {
  throw;
} catch (const configuration_input_error& e) {
  if (impl_->source_path.empty()) { throw; }
  throw configuration_input_error(impl_->source_path.string() + ": " + e.what());
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
