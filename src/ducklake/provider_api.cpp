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

#include <dlfcn.h>
#include <duckdb/common/file_system.hpp>
#include <duckdb/main/extension_helper.hpp>
#include <duckdb/main/extension_manager.hpp>
#include <link.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include <bit>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace sirius::ducklake {
namespace {
using namespace duckdb;

std::string provider_load_path(DatabaseInstance& db, ExtensionInstallInfo const& install)
{
  if (install.mode == ExtensionInstallMode::NOT_INSTALLED) return install.full_path;
  if (install.mode == ExtensionInstallMode::STATICALLY_LINKED) return {};
  // Named LOAD searches the extension directories in order. install.full_path
  // records the download URL or source file, which can differ from the loaded copy.
  auto& fs = db.GetFileSystem();
  for (auto const& dir : ExtensionHelper::GetExtensionDirectoryPath(db, fs)) {
    auto path = fs.JoinPath(dir, "ducklake.duckdb_extension");
    if (fs.FileExists(path)) return path;
  }
  return {};
}

// link_map::l_name can be renamed after LOAD. The kernel records the backing
// device and inode of each mapping independently of changes to that pathname.
bool mapped_file_matches(link_map const& map, struct stat const& expected)
{
  auto address = reinterpret_cast<std::uintptr_t>(map.l_ld);
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
        first <= address && address < last) {
      return inode != 0 && inode == expected.st_ino &&
             makedev(major_device, minor_device) == expected.st_dev;
    }
  }
  return false;
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

std::shared_ptr<const provider_api> provider_api::resolve(duckdb::DatabaseInstance& db)
{
  auto& manager = duckdb::ExtensionManager::Get(db);
  if (!manager.ExtensionIsLoaded("ducklake")) return {};
  auto info = manager.GetExtensionInfo("ducklake");
  if (!info || !info->install_info || info->install_info->version != "417a9117f5") return {};
  auto path = provider_load_path(db, *info->install_info);
  if (path.empty()) return {};
  path = std::filesystem::absolute(path).string();
  struct stat loaded{};
  if (::stat(path.c_str(), &loaded) != 0 || !S_ISREG(loaded.st_mode)) return {};

  // Retain the object DuckDB already loaded. Never load a candidate to inspect its exports.
  std::shared_ptr<void> module(dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD),
                               [](void* h) {
                                 if (h) dlclose(h);
                               });
  link_map* map = nullptr;
  if (!module || dlinfo(module.get(), RTLD_DI_LINKMAP, &map) != 0 || !map) return {};
  if (!mapped_file_matches(*map, loaded)) return {};

  auto api    = std::make_shared<provider_api>();
  api->module = std::move(module);
  api->device = loaded.st_dev;
  api->inode  = loaded.st_ino;
  symbols lookup{api->module.get(), map};
  api->scan_function = reinterpret_cast<decltype(scan_function)>(
    lookup.find("_ZN6duckdb17DuckLakeFunctions23GetDuckLakeScanFunctionERNS_16DatabaseInstanceE"));
  api->create_reader = reinterpret_cast<decltype(create_reader)>(
    lookup.find("_ZN6duckdb23DuckLakeMultiFileReader14CreateInstanceERKNS_18BoundTableFunctionE"));
  api->get_files =
    lookup.member<decltype(get_files)>("_ZNK6duckdb21DuckLakeMultiFileList8GetFilesEv");
  api->set_scan_columns = lookup.member<decltype(set_scan_columns)>(
    "_ZNK6duckdb21DuckLakeMultiFileList14SetScanColumnsERKNS_6vectorINS_"
    "25MultiFileColumnDefinitionELb1ESaIS2_EEERKNS1_INS_11ColumnIndexELb1ESaIS7_EEE");
#if defined(_GLIBCXX_USE_CXX11_ABI) && _GLIBCXX_USE_CXX11_ABI
  api->has_local_delete = lookup.member<decltype(has_local_delete)>(
    "_ZNK6duckdb19DuckLakeTransaction21HasLocalDeleteForFileENS_10TableIndexERKNSt7__cxx1112basic_"
    "stringIcSt11char_traitsIcESaIcEEE");
#else
  // This export takes libstdc++'s C++11 std::string by reference.
  api->has_local_delete = nullptr;
#endif
  api->info_type =
    static_cast<decltype(info_type)>(lookup.find("_ZTIN6duckdb20DuckLakeFunctionInfoE"));
  api->reader_type =
    static_cast<decltype(reader_type)>(lookup.find("_ZTIN6duckdb23DuckLakeMultiFileReaderE"));
  api->list_type =
    static_cast<decltype(list_type)>(lookup.find("_ZTIN6duckdb21DuckLakeMultiFileListE"));
  if (!api->scan_function || !api->create_reader || !api->get_files || !api->set_scan_columns ||
      !api->has_local_delete || !api->info_type || !api->reader_type || !api->list_type)
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
