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

#include "config_option.hpp"
#include "exec/config.hpp"

#include <cucascade/memory/config.hpp>
#include <cucascade/memory/reservation_manager_configurator.hpp>
#include <libconfig.h++>

#include <exception>
#include <variant>
#include <vector>

namespace sirius {
template <>
struct sirius::config::custom_config_registrar<cucascade::memory::gpu_memory_space_config> {
  static void config(sirius::config::configuration_setter& setter,
                     cucascade::memory::gpu_memory_space_config& opt)
  {
    opt.per_stream_reservation = false;  // default to false for sirius
    setter.add_config("device_id", opt.device_id);
    setter.add_config("per_stream_reservation", opt.per_stream_reservation);
    setter.add_config(
      "reservation_limit_fraction", opt.reservation_limit_fraction, fraction<double>{});
    setter.add_config(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, fraction<double>{});
    setter.add_config("downgrade_stop_fraction", opt.downgrade_stop_fraction, fraction<double>{});
    setter.add_config("memory_capacity", opt.memory_capacity);
  }
};

template <>
struct sirius::config::custom_config_registrar<cucascade::memory::host_memory_space_config> {
  static void config(sirius::config::configuration_setter& setter,
                     cucascade::memory::host_memory_space_config& opt)
  {
    setter.add_config("numa_id", opt.numa_id);
    setter.add_config(
      "reservation_limit_fraction", opt.reservation_limit_fraction, fraction<double>{});
    setter.add_config(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction, fraction<double>{});
    setter.add_config("downgrade_stop_fraction", opt.downgrade_stop_fraction, fraction<double>{});
    setter.add_config("memory_capacity", opt.memory_capacity);
    setter.add_config("block_size", opt.block_size);
    setter.add_config("pool_size", opt.pool_size);
    setter.add_config("initial_number_pools", opt.initial_number_pools);
  }
};

template <>
struct sirius::config::custom_config_registrar<cucascade::memory::disk_memory_space_config> {
  static void config(sirius::config::configuration_setter& setter,
                     cucascade::memory::disk_memory_space_config& opt)
  {
    setter.add_config("disk_id", opt.disk_id);
    setter.add_config("mount_path", opt.mount_paths);
    setter.add_config("memory_capacity", opt.memory_capacity);
  }
};

template <>
struct sirius::config::custom_config_registrar<sirius::exec::thread_pool_config> {
  static void config(sirius::config::configuration_setter& setter,
                     sirius::exec::thread_pool_config& opt)
  {
    setter.add_config("num_threads", opt.num_threads, sirius::config::greater_than<size_t>{0});
    setter.add_config("thread_name_prefix", opt.thread_name_prefix);
    setter.add_config("cpu_affinity", opt.cpu_affinity_list);
  }
};

namespace {
struct topology {
  std::variant<size_t, std::vector<int>> num_gpus_or_gpu_ids{size_t{1}};
};

struct gpu_mem_config {
  std::variant<double, std::uint64_t> usage_limit_fraction_or_bytes{0.95};
  std::variant<double, std::uint64_t> reservation_limit_fraction_or_bytes{0.9};
  double downgrade_trigger_fraction_{0.8};
  double downgrade_stop_fraction_{0.7};
  bool track_per_stream_reservation{false};

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    if (std::holds_alternative<double>(usage_limit_fraction_or_bytes)) {
      builder.set_usage_limit_ratio_per_gpu(std::get<double>(usage_limit_fraction_or_bytes));
    } else {
      builder.set_gpu_usage_limit(std::get<std::uint64_t>(usage_limit_fraction_or_bytes));
    }

    if (std::holds_alternative<double>(reservation_limit_fraction_or_bytes)) {
      builder.set_reservation_fraction_per_gpu(
        std::get<double>(reservation_limit_fraction_or_bytes));
    } else {
      builder.set_reservation_fraction_per_gpu(
        std::get<std::uint64_t>(reservation_limit_fraction_or_bytes));
    }
    builder.set_downgrade_fractions_per_gpu(downgrade_trigger_fraction_, downgrade_stop_fraction_);
    builder.track_reservation_per_stream(track_per_stream_reservation);
  }
};

struct host_mem_config {
  std::uint64_t numa_region_capacity_bytes = 8UL << 30;  // 8GB per NUMA node
  std::variant<double, std::uint64_t> reservation_limit_fraction_or_bytes{0.9};
  double downgrade_trigger_fraction_{0.8};
  double downgrade_stop_fraction_{0.7};
  std::size_t block_size{cucascade::memory::default_block_size};  // 64MB blocks
  std::size_t pool_size{cucascade::memory::default_pool_size};    // 1024 blocks per pool
  std::size_t initial_number_pools{cucascade::memory::default_initial_number_pools};

  void setup_configurator(cucascade::memory::reservation_manager_configurator& builder) const
  {
    builder.use_host_per_numa();
    if (std::holds_alternative<double>(reservation_limit_fraction_or_bytes)) {
      builder.set_reservation_fraction_per_host(
        std::get<double>(reservation_limit_fraction_or_bytes));
    } else {
      builder.set_reservation_fraction_per_host(
        std::get<std::uint64_t>(reservation_limit_fraction_or_bytes));
    }
    builder.set_downgrade_fractions_per_host(downgrade_trigger_fraction_, downgrade_stop_fraction_);
    builder.set_per_host_capacity(numa_region_capacity_bytes);
    builder.set_host_pool_features(block_size, pool_size, initial_number_pools);
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

}  // namespace

template <>
struct sirius::config::custom_config_registrar<sirius::topology> {
  static void config(sirius::config::configuration_setter& setter, sirius::topology& opt)
  {
    setter.add_variant_config<size_t>("num_gpus", opt.num_gpus_or_gpu_ids);
    setter.add_variant_config<std::vector<int>>("gpu_ids", opt.num_gpus_or_gpu_ids);
  }
};

template <>
struct sirius::config::custom_config_registrar<sirius::gpu_mem_config> {
  static void config(sirius::config::configuration_setter& setter, sirius::gpu_mem_config& opt)
  {
    opt.track_per_stream_reservation = false;
    setter.add_variant_config<double>(
      "usage_limit_fraction", opt.usage_limit_fraction_or_bytes, fraction<double>{});
    setter.add_variant_config<size_t>("usage_limit_bytes", opt.usage_limit_fraction_or_bytes);
    setter.add_variant_config<double>(
      "reservation_limit_fraction", opt.reservation_limit_fraction_or_bytes, fraction<double>{});
    setter.add_variant_config<size_t>("reservation_limit_bytes",
                                      opt.reservation_limit_fraction_or_bytes);
    setter.add_config(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction_, fraction<double>{});
    setter.add_config("downgrade_stop_fraction", opt.downgrade_stop_fraction_, fraction<double>{});
    setter.add_config("track_per_stream_reservation", opt.track_per_stream_reservation);
  }
};

template <>
struct sirius::config::custom_config_registrar<sirius::host_mem_config> {
  static void config(sirius::config::configuration_setter& setter, sirius::host_mem_config& opt)
  {
    setter.add_config("capacity_bytes", opt.numa_region_capacity_bytes);
    setter.add_variant_config<double>(
      "reservation_limit_fraction", opt.reservation_limit_fraction_or_bytes, fraction<double>{});
    setter.add_variant_config<size_t>("reservation_limit_bytes",
                                      opt.reservation_limit_fraction_or_bytes);
    setter.add_config(
      "downgrade_trigger_fraction", opt.downgrade_trigger_fraction_, fraction<double>{});
    setter.add_config("downgrade_stop_fraction", opt.downgrade_stop_fraction_, fraction<double>{});
    setter.add_config("block_size", opt.block_size);
    setter.add_config("pool_size", opt.pool_size);
    setter.add_config("initial_number_pools", opt.initial_number_pools);
  }
};

template <>
struct sirius::config::custom_config_registrar<sirius::disk_mem_config> {
  static void config(sirius::config::configuration_setter& setter, sirius::disk_mem_config& opt)
  {
    setter.add_config("disk_id", opt.id);
    setter.add_config("capacity_bytes", opt.capacity_bytes);
    setter.add_config("downgrade_root_dirs", opt.downgrade_root_dirs);
  }
};

template <>
struct sirius::config::custom_config_registrar<sirius::operator_params> {
  static void config(sirius::config::configuration_setter& setter, sirius::operator_params& opt)
  {
    setter.add_config("scan_task_batch_size", opt.scan_task_batch_size);
    setter.add_config("default_scan_task_varchar_size", opt.default_scan_task_varchar_size);
    setter.add_config("max_sort_partition_bytes", opt.max_sort_partition_bytes);
    setter.add_config("hash_partition_bytes", opt.hash_partition_bytes);
    setter.add_config("concat_batch_bytes", opt.concat_batch_bytes);
  }
};

sirius_config::sirius_config()
{
  cucascade::memory::topology_discovery discovery;
  if (discovery.discover()) { _hw_topology = discovery.get_topology(); }
}

void sirius_config::load_from_file(const std::filesystem::path& config_path)
{
  libconfig::Config config;
  config.readFile(config_path.string().c_str());

  config::configuration_setter config_setter;

  topology topology_instance;
  gpu_mem_config gpu_memory_config_instance;
  host_mem_config host_memory_config_instance;
  disk_mem_config disk_memory_config_instance;

  std::vector<cucascade::memory::gpu_memory_space_config> gpu_memory_space_configs;
  std::vector<cucascade::memory::host_memory_space_config> host_memory_space_configs;
  std::vector<cucascade::memory::disk_memory_space_config> disk_memory_space_configs;

  config_setter.add_config("sirius.topology", topology_instance);
  config_setter.add_config("sirius.memory.gpu", gpu_memory_config_instance);
  config_setter.add_config("sirius.memory.host", host_memory_config_instance);
  config_setter.add_config("sirius.memory.disk", disk_memory_config_instance);
  config_setter.add_config("sirius.executor.task_creator", _task_creator_config);
  config_setter.add_config("sirius.executor.pipeline", _gpu_pipeline_executor_config);
  config_setter.add_config("sirius.executor.downgrade", _downgrade_executor_config);
  config_setter.add_config("sirius.executor.duckdb_scan", _duckdb_scan_executor_config);
  config_setter.add_config("sirius.executor.duckdb_scan.cache", _enable_scan_caching);
  config_setter.add_config("sirius.operator_params", _operator_params);

  config_setter.add_config("sirius.space.gpu", gpu_memory_space_configs);
  config_setter.add_config("sirius.space.host", host_memory_space_configs);
  config_setter.add_config("sirius.space.disk", disk_memory_space_configs);

  _memory_space_configs.clear();

  try {
    config_setter.apply(config.getRoot());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Failed to load Sirius configuration from file: " + config_path.string() + " " + e.what());
  }

  // std::cerr << "Loaded Sirius configuration from file: " << config_path << std::endl;

  std::copy(gpu_memory_space_configs.begin(),
            gpu_memory_space_configs.end(),
            std::back_inserter(_memory_space_configs));
  std::copy(host_memory_space_configs.begin(),
            host_memory_space_configs.end(),
            std::back_inserter(_memory_space_configs));
  std::copy(disk_memory_space_configs.begin(),
            disk_memory_space_configs.end(),
            std::back_inserter(_memory_space_configs));

  bool using_configurator = _memory_space_configs.empty();
  if (using_configurator) {
    cucascade::memory::reservation_manager_configurator builder;
    if (std::holds_alternative<size_t>(topology_instance.num_gpus_or_gpu_ids)) {
      builder.set_number_of_gpus(std::get<size_t>(topology_instance.num_gpus_or_gpu_ids));
    } else {
      const auto& gpu_ids = std::get<std::vector<int>>(topology_instance.num_gpus_or_gpu_ids);
      builder.set_gpu_ids(gpu_ids);
    }
    gpu_memory_config_instance.setup_configurator(builder);
    host_memory_config_instance.setup_configurator(builder);
    disk_memory_config_instance.setup_configurator(builder);
    _memory_space_configs = builder.build(_hw_topology);
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

const exec::thread_pool_config& sirius_config::get_downgrade_executor_config() const noexcept
{
  return _downgrade_executor_config;
}

const exec::thread_pool_config& sirius_config::get_task_creator_config() const noexcept
{
  return _task_creator_config;
}

const exec::thread_pool_config& sirius_config::get_duckdb_scan_executor_config() const noexcept
{
  return _duckdb_scan_executor_config;
}

}  // namespace sirius
