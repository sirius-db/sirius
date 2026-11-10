/*
 * Copyright 2026, Sirius Contributors.
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

#include "ducklake/provider_api.hpp"

#include <absl/cleanup/cleanup.h>
#include <dlfcn.h>
#include <duckdb/main/config.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <bit>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <type_traits>
#include <utility>

namespace sirius::ducklake {
namespace {
using namespace duckdb;

std::optional<ExtensionInstallInfo> install_snapshot(DatabaseInstance& db,
                                                     bool in_ducklake_load_callback)
{
  auto info = ExtensionManager::Get(db).GetExtensionInfo("ducklake");
  if (!info || !info->is_loaded) return {};
  // DuckDB writes install_info under this lock, after publishing is_loaded. Its
  // own FinishLoad callback already holds the lock and sees the completed write.
  auto* lock = in_ducklake_load_callback ? nullptr : info->lock.native_handle();
  // LOADs can nest. POSIX trylock also returns busy for this thread's own
  // non-recursive lock. A later scan lookup retries after FinishLoad returns.
  if (lock && pthread_mutex_trylock(lock) != 0) return {};
  absl::Cleanup unlock = [&] {
    if (lock) pthread_mutex_unlock(lock);
  };
  if (!info->install_info) return {};
  return *info->install_info;
}

using file_id = std::pair<dev_t, ino_t>;

std::optional<file_id> mapped_file_id(void const* pointer)
{
  auto address = reinterpret_cast<std::uintptr_t>(pointer);
  std::ifstream maps("/proc/self/maps");
  for (std::string line; std::getline(maps, line);) {
    std::uintptr_t first, last;
    unsigned int major_device, minor_device;
    std::uintmax_t inode;
    if (std::sscanf(line.c_str(),
                    "%" SCNxPTR "-%" SCNxPTR " %*s %*s %x:%x %" SCNuMAX,
                    &first,
                    &last,
                    &major_device,
                    &minor_device,
                    &inode) == 5 &&
        first <= address && address < last && inode != 0) {
      return file_id{makedev(major_device, minor_device), inode};
    }
  }
  return {};
}

bool mapped_file_matches(link_map const& map, int fd, struct stat const& file)
{
  auto loaded = mapped_file_id(map.l_ld);
  if (!loaded) return false;
  if (*loaded == file_id{file.st_dev, file.st_ino}) return true;
  // Overlayfs may expose a different device/inode through stat than through a VMA.
  // Map the same open file so both identities come from the kernel's VMA view.
  void* probe = mmap(nullptr, 1, PROT_READ, MAP_PRIVATE, fd, 0);
  if (probe == MAP_FAILED) return false;
  absl::Cleanup unmap = [&] { munmap(probe, 1); };
  return mapped_file_id(probe) == loaded;
}

struct symbols {
  void* module;
  link_map* image;

  void* find(char const* name) const
  {
    auto* address = dlsym(module, name);
    Dl_info origin{};
    // dlsym also searches this object's dependencies. Only this object's exports qualify.
    if (!address || !dladdr(address, &origin) ||
        origin.dli_fbase != reinterpret_cast<void*>(image->l_addr))
      return nullptr;
    return address;
  }

  template <typename T>
  T member(char const* name) const
  {
    // On Linux x86-64 and AArch64, a non-virtual Itanium member pointer holds the
    // entry address and a zero this-adjustment. The compiler supplies the call ABI.
    static_assert(std::is_member_function_pointer_v<T>);
    auto address = find(name);
    if (!address) return nullptr;
    struct {
      void* entry;
      std::ptrdiff_t adjustment;
    } value{address, 0};
    return std::bit_cast<T>(value);
  }
};

template <typename Derived, typename Base>
Derived const* checked_cast(Base const* value, __cxxabiv1::__class_type_info const* target)
{
  if (!value) return nullptr;
  // The destination RTTI belongs to the already loaded provider. Referencing typeid(Derived)
  // here would require the provider at wrapper load time.
  return static_cast<Derived const*>(__cxxabiv1::__dynamic_cast(
    value, static_cast<__cxxabiv1::__class_type_info const*>(&typeid(Base)), target, -1));
}
}  // namespace

std::shared_ptr<const provider_api> resolve_provider(duckdb::DatabaseInstance& db,
                                                     bool in_ducklake_load_callback)
{
  auto install = install_snapshot(db, in_ducklake_load_callback);
  if (!install || install->version != "417a9117f5" ||
      install->mode == ExtensionInstallMode::STATICALLY_LINKED)
    return {};
  // The pinned provider registers this setter in this database. DuckDB copies the
  // option under its settings lock. Its address only locates the candidate image.
  ExtensionOption option;
  Dl_info origin{};
  if (!DBConfig::GetConfig(db).TryGetExtensionOption("ducklake_target_file_size", option) ||
      !option.set_function || !dladdr(reinterpret_cast<void*>(option.set_function), &origin) ||
      !origin.dli_fname || !*origin.dli_fname)
    return {};
  std::string path =
    install->mode == ExtensionInstallMode::NOT_INSTALLED ? install->full_path : origin.dli_fname;
  if (path.empty()) return {};
  path   = std::filesystem::absolute(path).string();
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) return {};
  absl::Cleanup close_fd = [&] { ::close(fd); };
  struct stat loaded{};
  if (fstat(fd, &loaded) != 0 || !S_ISREG(loaded.st_mode) || loaded.st_size == 0) return {};

  std::shared_ptr<void> module(dlopen(origin.dli_fname, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD),
                               [](void* h) {
                                 if (h) dlclose(h);
                               });
  link_map* map = nullptr;
  if (!module || dlinfo(module.get(), RTLD_DI_LINKMAP, &map) != 0 || !map ||
      reinterpret_cast<void*>(map->l_addr) != origin.dli_fbase ||
      !mapped_file_matches(*map, fd, loaded))
    return {};

  auto api    = std::make_shared<provider_api>();
  api->module = std::move(module);
  api->device = loaded.st_dev;
  api->inode  = loaded.st_ino;
  symbols lookup{api->module.get(), map};
  api->scan_function = reinterpret_cast<decltype(api->scan_function)>(
    lookup.find("_ZN6duckdb17DuckLakeFunctions23GetDuckLakeScanFunctionERNS_16DatabaseInstanceE"));
  api->create_reader = reinterpret_cast<decltype(api->create_reader)>(
    lookup.find("_ZN6duckdb23DuckLakeMultiFileReader14CreateInstanceERKNS_18BoundTableFunctionE"));
  api->get_files =
    lookup.member<decltype(api->get_files)>("_ZNK6duckdb21DuckLakeMultiFileList8GetFilesEv");
  api->get_delete_scan_entry = lookup.member<decltype(api->get_delete_scan_entry)>(
    "_ZN6duckdb21DuckLakeMultiFileList18GetDeleteScanEntryEm");
  api->set_scan_columns = lookup.member<decltype(api->set_scan_columns)>(
    "_ZNK6duckdb21DuckLakeMultiFileList14SetScanColumnsERKNS_6vectorINS_"
    "25MultiFileColumnDefinitionELb1ESaIS2_EEERKNS1_INS_11ColumnIndexELb1ESaIS7_EEE");
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI
  api->has_local_delete = lookup.member<decltype(api->has_local_delete)>(
    "_ZNK6duckdb19DuckLakeTransaction21HasLocalDeleteForFileENS_10TableIndexERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEE");
#else
  // This export takes libstdc++'s C++11 std::string by reference.
  api->has_local_delete = nullptr;
#endif
  api->info_type =
    static_cast<decltype(api->info_type)>(lookup.find("_ZTIN6duckdb20DuckLakeFunctionInfoE"));
  api->reader_type =
    static_cast<decltype(api->reader_type)>(lookup.find("_ZTIN6duckdb23DuckLakeMultiFileReaderE"));
  api->list_type =
    static_cast<decltype(api->list_type)>(lookup.find("_ZTIN6duckdb21DuckLakeMultiFileListE"));
  if (!api->scan_function || !api->create_reader || !api->get_files ||
      !api->get_delete_scan_entry || !api->set_scan_columns || !api->has_local_delete ||
      !api->info_type || !api->reader_type || !api->list_type)
    return {};
  api->path = std::move(path);
  return api;
}

bool provider_api::matches_loaded_file() const
{
  struct stat current{};
  return ::stat(path.c_str(), &current) == 0 && current.st_dev == device && current.st_ino == inode;
}

duckdb::DuckLakeFunctionInfo const* provider_api::function_info(
  duckdb::TableFunctionInfo const* p) const
{
  return checked_cast<duckdb::DuckLakeFunctionInfo>(p, info_type);
}

duckdb::DuckLakeMultiFileReader const* provider_api::reader(duckdb::MultiFileReader const* p) const
{
  return checked_cast<duckdb::DuckLakeMultiFileReader>(p, reader_type);
}

duckdb::DuckLakeMultiFileList const* provider_api::file_list(duckdb::MultiFileList const* p) const
{
  return checked_cast<duckdb::DuckLakeMultiFileList>(p, list_type);
}
}  // namespace sirius::ducklake
