/**
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory/topology_discovery.hpp"

#include <dlfcn.h>
#include <nvml.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace fs = std::filesystem;

namespace cucascade::memory {

namespace {

// Runtime NVML loader to avoid link-time dependency on CUDA::nvml
struct NvmlLoader {
  void* handle = nullptr;

  // Function pointers
  nvmlReturn_t (*p_nvmlInit_v2)()                                              = nullptr;
  nvmlReturn_t (*p_nvmlShutdown)()                                             = nullptr;
  nvmlReturn_t (*p_nvmlDeviceGetCount_v2)(unsigned int*)                       = nullptr;
  nvmlReturn_t (*p_nvmlDeviceGetHandleByIndex_v2)(unsigned int, nvmlDevice_t*) = nullptr;
  nvmlReturn_t (*p_nvmlDeviceGetName)(nvmlDevice_t, char*, unsigned int)       = nullptr;
  nvmlReturn_t (*p_nvmlDeviceGetPciInfo_v3)(nvmlDevice_t, nvmlPciInfo_t*)      = nullptr;
  nvmlReturn_t (*p_nvmlDeviceGetUUID)(nvmlDevice_t, char*, unsigned int)       = nullptr;
  const char* (*p_nvmlErrorString)(nvmlReturn_t)                               = nullptr;

  NvmlLoader() { load(); }

  ~NvmlLoader()
  {
    if (handle) { dlclose(handle); }
  }

  void load()
  {
    // Try the SONAME first, then the unversioned name as a fallback
    handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!handle) { handle = dlopen("libnvidia-ml.so", RTLD_LAZY | RTLD_LOCAL); }
    if (!handle) { return; }

    p_nvmlInit_v2  = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(handle, "nvmlInit_v2"));
    p_nvmlShutdown = reinterpret_cast<nvmlReturn_t (*)()>(dlsym(handle, "nvmlShutdown"));
    p_nvmlDeviceGetCount_v2 =
      reinterpret_cast<nvmlReturn_t (*)(unsigned int*)>(dlsym(handle, "nvmlDeviceGetCount_v2"));
    p_nvmlDeviceGetHandleByIndex_v2 =
      reinterpret_cast<nvmlReturn_t (*)(unsigned int, nvmlDevice_t*)>(
        dlsym(handle, "nvmlDeviceGetHandleByIndex_v2"));
    p_nvmlDeviceGetName = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int)>(
      dlsym(handle, "nvmlDeviceGetName"));
    p_nvmlDeviceGetPciInfo_v3 = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlPciInfo_t*)>(
      dlsym(handle, "nvmlDeviceGetPciInfo_v3"));
    p_nvmlDeviceGetUUID = reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, char*, unsigned int)>(
      dlsym(handle, "nvmlDeviceGetUUID"));
    p_nvmlErrorString =
      reinterpret_cast<const char* (*)(nvmlReturn_t)>(dlsym(handle, "nvmlErrorString"));
    // If any required symbol is missing, treat NVML as unavailable
    if (!p_nvmlInit_v2 || !p_nvmlShutdown || !p_nvmlDeviceGetCount_v2 ||
        !p_nvmlDeviceGetHandleByIndex_v2 || !p_nvmlDeviceGetName || !p_nvmlDeviceGetPciInfo_v3 ||
        !p_nvmlDeviceGetUUID || !p_nvmlErrorString) {
      dlclose(handle);
      handle                          = nullptr;
      p_nvmlInit_v2                   = nullptr;
      p_nvmlShutdown                  = nullptr;
      p_nvmlDeviceGetCount_v2         = nullptr;
      p_nvmlDeviceGetHandleByIndex_v2 = nullptr;
      p_nvmlDeviceGetName             = nullptr;
      p_nvmlDeviceGetPciInfo_v3       = nullptr;
      p_nvmlDeviceGetUUID             = nullptr;
      p_nvmlErrorString               = nullptr;
    }
  }

  [[nodiscard]] bool available() const
  {
    return handle && p_nvmlInit_v2 && p_nvmlShutdown && p_nvmlDeviceGetCount_v2 &&
           p_nvmlDeviceGetHandleByIndex_v2 && p_nvmlDeviceGetName && p_nvmlDeviceGetPciInfo_v3 &&
           p_nvmlDeviceGetUUID && p_nvmlErrorString;
  }
};

NvmlLoader& get_nvml_loader()
{
  static NvmlLoader loader;
  return loader;
}

/**
 * @brief Network device with topology information.
 */
struct NetworkDeviceWithTopology {
  std::string name;
  int numa_node;
  std::string pci_bus_id;
};

/**
 * @brief Read a file and return its content.
 *
 * Attempts to open and read the file at @p path. If the file cannot be opened or
 * read, returns an empty string. On success, returns the full contents with a
 * single trailing '\n' removed if present.
 *
 * @param path File to read.
 * @return File content on success; empty string on failure.
 */
std::string read_file_content(std::string const& path)
{
  std::ifstream file(path);
  if (!file.is_open()) { return ""; }
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();
  // Trim trailing newline
  if (!content.empty() && content.back() == '\n') { content.pop_back(); }
  return content;
}

/**
 * @brief Parse a CPU list string into a vector of core ids.
 *
 * Accepts formats like "0-31,128-159" or comma-separated single cores. If @p cpulist
 * is empty, returns an empty vector.
 *
 * @param cpulist CPU list string (e.g., "0-31,128-159").
 * @return Vector of CPU core ids; empty if @p cpulist is empty.
 * @throw std::invalid_argument or std::out_of_range if tokens are malformed and cannot be
 * parsed with std::stoi.
 */
std::vector<int> parse_cpu_list(std::string const& cpulist)
{
  std::vector<int> cores;
  if (cpulist.empty()) { return cores; }

  std::istringstream iss(cpulist);
  std::string token;
  while (std::getline(iss, token, ',')) {
    size_t dash_pos = token.find('-');
    if (dash_pos != std::string::npos) {
      // Range, e.g., "0-31"
      int start = std::stoi(token.substr(0, dash_pos));
      int end   = std::stoi(token.substr(dash_pos + 1));
      for (int i = start; i <= end; ++i) {
        cores.push_back(i);
      }
    } else {
      // Single core, e.g., "5"
      cores.push_back(std::stoi(token));
    }
  }
  return cores;
}

/**
 * @brief Normalize PCI bus ID to standard format.
 *
 * Converts the domain to 4 hex digits and lowercases the entire string. If the input
 * does not contain a colon (unexpected format), the input is returned unchanged.
 *
 * @param pci_bus_id PCI bus ID to normalize.
 * @return Normalized PCI bus ID in format (0000:06:00.0); unchanged on unrecognized
 * input.
 */
std::string normalize_pci_bus_id(std::string const& pci_bus_id)
{
  // NVML may return format like "00000000:0A:00.0" but /sys uses "0000:0a:00.0"
  // Find the first colon and take the last 4 hex digits before it as domain
  size_t colon_pos = pci_bus_id.find(':');
  if (colon_pos == std::string::npos) { return pci_bus_id; }
  std::string domain = pci_bus_id.substr(0, colon_pos);
  if (domain.length() > 4) { domain = domain.substr(domain.length() - 4); }

  // Convert to lowercase
  std::string normalized_id = domain + pci_bus_id.substr(colon_pos);
  std::ranges::transform(normalized_id, normalized_id.begin(), ::tolower);

  return normalized_id;
}

/**
 * @brief Get NUMA node from /sys for a PCI device.
 *
 * Reads /sys/bus/pci/devices/<pci>/numa_node. If the file is missing, empty, or
 * cannot be parsed as an integer, returns -1.
 *
 * @param pci_bus_id PCI bus ID of the device.
 * @return NUMA node number on success; -1 if unavailable.
 */
int get_numa_node_from_sys(std::string const& pci_bus_id)
{
  std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
  std::string path          = "/sys/bus/pci/devices/" + normalized_id + "/numa_node";
  std::string content       = read_file_content(path);
  if (content.empty()) { return -1; }
  try {
    return std::stoi(content);
  } catch (...) {
    return -1;
  }
}

/**
 * @brief Get CPU affinity list from /sys for a PCI device.
 *
 * Reads /sys/bus/pci/devices/<pci>/local_cpulist and returns the file content as-is
 * (with any trailing newline trimmed). If the file is missing or unreadable, returns
 * an empty string.
 *
 * @param pci_bus_id PCI bus ID of the device.
 * @return CPU affinity list string; empty on failure.
 */
std::string get_cpu_affinity_from_sys(std::string const& pci_bus_id)
{
  std::string normalized_id = normalize_pci_bus_id(pci_bus_id);
  std::string path          = "/sys/bus/pci/devices/" + normalized_id + "/local_cpulist";
  return read_file_content(path);
}

/**
 * @brief Get PCI bus ID from a device directory in /sys.
 *
 * Resolves <device_path>/device symlink and returns its basename (e.g., "0000:06:00.0").
 * Returns an empty string if the symlink does not exist or cannot be resolved.
 *
 * @param device_path Path to the device in /sys.
 * @return PCI bus ID of the device; empty string if not available.
 */
std::string get_pci_bus_id_from_device(std::string const& device_path)
{
  fs::path device_link = fs::path(device_path) / "device";
  if (!fs::exists(device_link)) { return ""; }

  try {
    fs::path real_path = fs::canonical(device_link);
    return real_path.filename().string();
  } catch (...) {
    return "";
  }
}

/**
 * @brief Parse PCI bus number from a PCI ID string.
 *
 * Expects format domain:bus:device.function (e.g., "0000:06:00.0"). If parsing fails
 * or the expected separators are missing, returns -1.
 *
 * @param pci_id PCI ID string.
 * @return PCI bus number (integer) on success; -1 on failure.
 */
int get_pci_bus_number(std::string const& pci_id)
{
  size_t first_colon = pci_id.find(':');
  if (first_colon == std::string::npos) { return -1; }

  size_t second_colon = pci_id.find(':', first_colon + 1);
  if (second_colon == std::string::npos) { return -1; }

  std::string bus_str = pci_id.substr(first_colon + 1, second_colon - first_colon - 1);
  try {
    return std::stoi(bus_str, nullptr, 16);
  } catch (...) {
    return -1;
  }
}

/**
 * @brief Get PCIe path type between two devices by analyzing /sys topology.
 *
 * Uses NUMA node comparison and PCI bus number proximity as a heuristic. If either
 * device's bus cannot be parsed, falls back to PHB. If NUMA nodes differ (and are
 * known), returns SYS.
 *
 * @param gpu_pci_id PCI bus ID of the GPU device.
 * @param nic_pci_id PCI bus ID of the NIC device.
 * @return Path type (PIX, PXB, PHB, NODE, or SYS); PHB when indeterminate.
 */
PciePathType get_pcie_path_type(std::string const& gpu_pci_id, std::string const& nic_pci_id)
{
  std::string gpu_norm = normalize_pci_bus_id(gpu_pci_id);
  std::string nic_norm = normalize_pci_bus_id(nic_pci_id);

  // Read NUMA nodes
  int gpu_numa = -1, nic_numa = -1;
  std::string gpu_numa_str = read_file_content("/sys/bus/pci/devices/" + gpu_norm + "/numa_node");
  std::string nic_numa_str = read_file_content("/sys/bus/pci/devices/" + nic_norm + "/numa_node");

  if (!gpu_numa_str.empty()) { gpu_numa = std::stoi(gpu_numa_str); }
  if (!nic_numa_str.empty()) { nic_numa = std::stoi(nic_numa_str); }

  // If different NUMA nodes, it's a SYS connection
  if (gpu_numa != nic_numa && gpu_numa >= 0 && nic_numa >= 0) { return PciePathType::SYS; }

  // Use PCI bus number proximity as a heuristic for connection quality
  // Devices on nearby PCI buses are typically on the same PCIe root complex
  int gpu_bus = get_pci_bus_number(gpu_pci_id);
  int nic_bus = get_pci_bus_number(nic_pci_id);

  if (gpu_bus < 0 || nic_bus < 0) {
    // Can't determine, assume PHB
    return PciePathType::PHB;
  }

  int bus_distance = std::abs(gpu_bus - nic_bus);

  // Heuristic based on PCI bus proximity:
  // - Very close buses (distance <= 2): likely PIX (single bridge)
  // - Moderate distance (3-10): likely PHB (host bridge)
  // - Large distance (>10): likely NODE or worse

  if (bus_distance <= 2) {
    return PciePathType::PIX;
  } else if (bus_distance <= 10) {
    return PciePathType::PHB;
  } else {
    return PciePathType::NODE;
  }
}

/**
 * @brief Discover network devices (InfiniBand/RoCE).
 *
 * Scans /sys/class/infiniband and collects device name, NUMA node, and PCI bus ID.
 * If the directory does not exist or an error occurs during iteration, returns an
 * empty vector (a warning is logged for iteration errors).
 *
 * @return Vector of discovered network devices; empty if none or on failure.
 */
std::vector<NetworkDeviceWithTopology> discover_network_devices_with_topology()
{
  std::vector<NetworkDeviceWithTopology> devices;
  std::string ib_path = "/sys/class/infiniband";

  if (!fs::exists(ib_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(ib_path)) {
      if (!entry.is_directory()) { continue; }

      NetworkDeviceWithTopology dev;
      dev.name = entry.path().filename().string();

      // Get device's NUMA node and PCI bus ID
      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering network devices: " << e.what() << std::endl;
  }

  return devices;
}

std::vector<storage_device_info> discover_storage_devices_with_topology()
{
  std::vector<storage_device_info> devices;
  std::string nvme_path = "/sys/class/nvme";

  if (!fs::exists(nvme_path)) { return devices; }

  try {
    for (auto const& entry : fs::directory_iterator(nvme_path)) {
      if (!entry.is_directory()) { continue; }

      storage_device_info dev;
      dev.name = entry.path().filename().string();
      dev.type = StorageDriveType::NVME;

      // Get device's NUMA node and PCI bus ID
      std::string numa_path = entry.path().string() + "/device/numa_node";
      std::string numa_str  = read_file_content(numa_path);
      dev.numa_node         = numa_str.empty() ? -1 : std::stoi(numa_str);
      dev.pci_bus_id        = get_pci_bus_id_from_device(entry.path().string());

      devices.push_back(dev);
    }
  } catch (std::exception const& e) {
    std::cerr << "Warning: Error discovering NVMe devices: " << e.what() << std::endl;
  }

  return devices;
}

/**
 * @brief Map network devices to a GPU based on PCIe topology and NUMA proximity.
 *
 * Prefers NICs with the best (lowest) PCIe path type to the GPU. If no PCI topology
 * information is available, falls back to matching by NUMA node. If still ambiguous,
 * returns all devices. May return an empty vector when @p network_devices is empty
 * or none have PCI info.
 *
 * @param gpu_numa_node NUMA node to query.
 * @param network_devices Network devices on the system.
 * @return Vector of device names selected for the GPU; possibly empty.
 */
std::vector<std::string> map_network_devices_to_gpu(
  std::string const& gpu_pci_id,
  int gpu_numa_node,
  std::vector<network_device_info> const& network_devices)
{
  std::vector<std::string> mapped_devices;

  // Structure to hold NIC with its topology path type
  struct NicWithPath {
    std::string name;
    PciePathType path_type;
  };

  std::vector<NicWithPath> nics_with_paths;

  // Query topology distance for each NIC
  for (auto const& dev : network_devices) {
    if (dev.pci_bus_id.empty()) {
      continue;  // Skip devices without PCI info
    }

    NicWithPath nic;
    nic.name      = dev.name;
    nic.path_type = get_pcie_path_type(gpu_pci_id, dev.pci_bus_id);

    nics_with_paths.push_back(nic);
  }

  // Find the best (lowest) path type
  if (nics_with_paths.empty()) { return mapped_devices; }

  PciePathType best_path_type = PciePathType::SYS;
  for (auto const& nic : nics_with_paths) {
    if (nic.path_type < best_path_type) { best_path_type = nic.path_type; }
  }

  // Return all NICs with the best path type
  for (auto const& nic : nics_with_paths) {
    if (nic.path_type == best_path_type) { mapped_devices.push_back(nic.name); }
  }

  // If no devices found, fall back to NUMA-based mapping
  if (mapped_devices.empty()) {
    for (auto const& dev : network_devices) {
      if (dev.numa_node == gpu_numa_node) { mapped_devices.push_back(dev.name); }
    }
  }

  // Last resort: return all devices
  if (mapped_devices.empty() && !network_devices.empty()) {
    for (auto const& dev : network_devices) {
      mapped_devices.push_back(dev.name);
    }
  }

  return mapped_devices;
}

/**
 * @brief Get system hostname.
 *
 * Returns the system hostname on success; returns an empty string if the
 * hostname cannot be determined.
 *
 * @return Hostname string, or an empty string if unavailable.
 */
std::string get_hostname()
{
  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) == 0) { return std::string(hostname.data()); }
  return "";
}

/**
 * @brief Count NUMA nodes on the system.
 *
 * Counts subdirectories named "node*" under /sys/devices/system/node. Returns 0 if
 * the directory does not exist or cannot be iterated.
 *
 * @return Number of NUMA nodes; 0 if unavailable.
 */
int count_numa_nodes()
{
  std::string numa_path = "/sys/devices/system/node";
  int count             = 0;

  if (!fs::exists(numa_path)) { return 0; }

  try {
    for (auto const& entry : fs::directory_iterator(numa_path)) {
      std::string name = entry.path().filename().string();
      if (name.starts_with("node")) {  // starts with "node"
        count++;
      }
    }
  } catch (...) {
    return 0;
  }

  return count;
}

}  // namespace

bool topology_discovery::discover()
{
  system_topology_info topology;
  NvmlLoader& nvml    = get_nvml_loader();
  nvmlReturn_t result = NVML_SUCCESS;
  if (!nvml.available()) {
    std::cerr << "NVML not available: failed to load libnvidia-ml.so.1" << std::endl;
    // Continue anyway to report system info even without GPUs
  } else {
    result = nvml.p_nvmlInit_v2();
    if (result != NVML_SUCCESS) {
      std::cerr << "Failed to initialize NVML: " << nvml.p_nvmlErrorString(result) << std::endl;
      // Continue anyway to report system info even without GPUs
    }
  }

  // Get GPU count
  unsigned int device_count = 0;
  bool nvml_available       = false;
  if (nvml.available() && result == NVML_SUCCESS) {
    result = nvml.p_nvmlDeviceGetCount_v2(&device_count);
    if (result != NVML_SUCCESS) {
      std::cerr << "Warning: Failed to get device count: " << nvml.p_nvmlErrorString(result)
                << std::endl;
      device_count = 0;
    } else {
      nvml_available = true;
    }
  }

  // Discover network devices
  std::vector<NetworkDeviceWithTopology> network_devices_with_topology =
    discover_network_devices_with_topology();

  // Get system information
  topology.hostname            = get_hostname();
  topology.num_numa_nodes      = count_numa_nodes();
  topology.num_gpus            = device_count;
  topology.num_network_devices = static_cast<int>(network_devices_with_topology.size());

  // Convert network devices to public format
  topology.network_devices.clear();
  for (auto const& dev : network_devices_with_topology) {
    network_device_info info;
    info.name       = dev.name;
    info.numa_node  = dev.numa_node;
    info.pci_bus_id = dev.pci_bus_id;
    topology.network_devices.push_back(info);
  }

  topology.storage_devices = discover_storage_devices_with_topology();

  // Collect GPU information
  topology.gpus.clear();

  for (unsigned int i = 0; i < device_count; ++i) {
    nvmlDevice_t device;
    result = nvml.p_nvmlDeviceGetHandleByIndex_v2(i, &device);
    if (result != NVML_SUCCESS) {
      std::cerr << "Warning: Failed to get handle for GPU " << i << ": "
                << nvml.p_nvmlErrorString(result) << std::endl;
      continue;
    }

    gpu_topology_info gpu;
    gpu.id = i;

    // Get device name, PCI bus ID and UUID
    std::array<char, NVML_DEVICE_NAME_BUFFER_SIZE> name{};
    result = nvml.p_nvmlDeviceGetName(device, name.data(), NVML_DEVICE_NAME_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
      gpu.name = std::string(name.data());
    } else {
      gpu.name = "Unknown";
    }

    nvmlPciInfo_t pci_info;
    result = nvml.p_nvmlDeviceGetPciInfo_v3(device, &pci_info);
    if (result == NVML_SUCCESS) {
      gpu.pci_bus_id = std::string(pci_info.busId);
    } else {
      std::cerr << "Warning: Failed to get PCI info for GPU " << i << std::endl;
      continue;
    }

    std::array<char, NVML_DEVICE_UUID_BUFFER_SIZE> uuid{};
    result = nvml.p_nvmlDeviceGetUUID(device, uuid.data(), NVML_DEVICE_UUID_BUFFER_SIZE);
    if (result == NVML_SUCCESS) {
      gpu.uuid = std::string(uuid.data());
    } else {
      gpu.uuid = "Unknown";
    }

    // Get NUMA node and CPU affinity from /sys
    gpu.numa_node         = get_numa_node_from_sys(gpu.pci_bus_id);
    gpu.cpu_affinity_list = get_cpu_affinity_from_sys(gpu.pci_bus_id);
    gpu.cpu_cores         = parse_cpu_list(gpu.cpu_affinity_list);

    // Memory binding is typically the same as NUMA node
    if (gpu.numa_node >= 0) { gpu.memory_binding.push_back(gpu.numa_node); }

    // Map network devices to this GPU using PCIe topology
    gpu.network_devices =
      map_network_devices_to_gpu(gpu.pci_bus_id, gpu.numa_node, topology.network_devices);

    topology.gpus.push_back(gpu);
  }

  if (nvml_available) { nvml.p_nvmlShutdown(); }

  topology_ = std::move(topology);
  return true;
}

}  // namespace cucascade::memory
