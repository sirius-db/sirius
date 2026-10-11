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

#pragma once

#include <dlfcn.h>
#include <duckdb/catalog/catalog.hpp>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <memory>

namespace sirius::planner::detail {
// Relies on the Linux Itanium C++ ABI, Catalog declaring its destructor as its
// first virtual member, and DuckCatalog defining it out of line. Recheck when
// upgrading DuckDB. Unlike RTTI and vtables, code is not subject to copy relocations.
inline void const* host_code_address(duckdb::Catalog const& catalog)
{
  void const* vtable{};
  std::memcpy(&vtable, static_cast<void const*>(&catalog), sizeof(vtable));
  void const* destructor{};
  std::memcpy(&destructor, vtable, sizeof(destructor));
  return destructor;
}

inline link_map const* host_module(duckdb::DatabaseInstance& db)
{
  Dl_info owner{};
  void* module{};
  if (!::dladdr1(
        host_code_address(duckdb::Catalog::GetSystemCatalog(db)), &owner, &module, RTLD_DL_LINKMAP))
    return nullptr;
  return static_cast<link_map const*>(module);
}

/// Resolves a global function from the host module's static symbol table (`.symtab`).
/// The official DuckDB CLI exports no symbols, so `dlsym` finds nothing there, but the
/// definitions are still in its `.symtab`. The image file is accepted only if its program
/// header table matches the one the loader mapped for this module (same load bias).
/// Returns a defined global/weak `STT_FUNC` address or nullptr; `host_factory` still checks
/// that the address belongs to the module.
inline void* host_static_symbol(link_map const* module, char const* symbol)
{
  if (!module || !symbol) return nullptr;
  // The main program has an empty name in the link map; its image is reachable via procfs.
  char const* path = module->l_name && module->l_name[0] ? module->l_name : "/proc/self/exe";
  int fd           = ::open(path, O_RDONLY | O_CLOEXEC);
  if (fd < 0) return nullptr;
  struct stat status{};
  if (::fstat(fd, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(Elf64_Ehdr))) {
    ::close(fd);
    return nullptr;
  }
  auto const size = static_cast<std::size_t>(status.st_size);
  void* mapping   = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapping == MAP_FAILED) return nullptr;
  auto unmap = [size](void* value) { ::munmap(value, size); };
  std::unique_ptr<void, decltype(unmap)> guard(mapping, unmap);
  auto const* image = static_cast<unsigned char const*>(mapping);
  auto in_file      = [size](std::uint64_t offset, std::uint64_t bytes) {
    return offset <= size && bytes <= size - offset;
  };

  Elf64_Ehdr header{};
  std::memcpy(&header, image, sizeof(header));
  if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
      header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_phentsize != sizeof(Elf64_Phdr) ||
      header.e_shentsize != sizeof(Elf64_Shdr))
    return nullptr;

  // The file must be the image the loader mapped: same program header table, same load bias.
  if (!in_file(header.e_phoff, std::uint64_t{header.e_phnum} * sizeof(Elf64_Phdr))) return nullptr;
  struct phdr_probe {
    ElfW(Addr) base;
    unsigned char const* file_phdr;
    std::size_t phnum;
    bool matched;
  } probe{module->l_addr, image + header.e_phoff, header.e_phnum, false};
  ::dl_iterate_phdr(
    [](dl_phdr_info* info, std::size_t, void* data) {
      auto* probe = static_cast<phdr_probe*>(data);
      if (info->dlpi_addr != probe->base) return 0;
      probe->matched =
        info->dlpi_phnum == probe->phnum &&
        std::memcmp(info->dlpi_phdr, probe->file_phdr, probe->phnum * sizeof(Elf64_Phdr)) == 0;
      return 1;
    },
    &probe);
  if (!probe.matched) return nullptr;

  if (!in_file(header.e_shoff, std::uint64_t{header.e_shnum} * sizeof(Elf64_Shdr))) return nullptr;
  auto section = [&](std::size_t index) {
    Elf64_Shdr value{};
    std::memcpy(&value, image + header.e_shoff + index * sizeof(Elf64_Shdr), sizeof(value));
    return value;
  };
  std::size_t const want = std::strlen(symbol);
  for (std::size_t i = 0; i < header.e_shnum; ++i) {
    auto const symtab = section(i);
    if (symtab.sh_type != SHT_SYMTAB || symtab.sh_entsize != sizeof(Elf64_Sym)) continue;
    if (symtab.sh_link >= header.e_shnum || !in_file(symtab.sh_offset, symtab.sh_size)) continue;
    auto const strtab = section(symtab.sh_link);
    if (strtab.sh_type != SHT_STRTAB || !in_file(strtab.sh_offset, strtab.sh_size)) continue;
    auto const* names = reinterpret_cast<char const*>(image + strtab.sh_offset);
    for (std::uint64_t offset = 0; offset + sizeof(Elf64_Sym) <= symtab.sh_size;
         offset += sizeof(Elf64_Sym)) {
      Elf64_Sym entry{};
      std::memcpy(&entry, image + symtab.sh_offset + offset, sizeof(entry));
      if (ELF64_ST_TYPE(entry.st_info) != STT_FUNC || entry.st_shndx == SHN_UNDEF ||
          entry.st_value == 0)
        continue;
      auto const bind = ELF64_ST_BIND(entry.st_info);
      if (bind != STB_GLOBAL && bind != STB_WEAK) continue;
      if (entry.st_name >= strtab.sh_size || strtab.sh_size - entry.st_name <= want) continue;
      if (std::memcmp(names + entry.st_name, symbol, want + 1) != 0) continue;
      return reinterpret_cast<void*>(static_cast<std::uintptr_t>(module->l_addr + entry.st_value));
    }
  }
  return nullptr;
}

template <class Factory>
Factory host_factory(duckdb::DatabaseInstance& db, Factory local_factory, char const* symbol)
{
  auto const* module = host_module(db);
  if (!module) return nullptr;
  Dl_info implementation{};
  void* implementation_map{};
  // Embedded DuckDB may hide its symbols. Its directly linked factory is trusted
  // only when it belongs to the same module as the host's system catalog.
  if (::dladdr1(reinterpret_cast<void*>(local_factory),
                &implementation,
                &implementation_map,
                RTLD_DL_LINKMAP) &&
      implementation_map == module)
    return local_factory;

  // Use the loader's existing record, without probing a file or changing symbol scope.
  // The main executable has an empty name and must be opened through its main handle.
  auto const* name = module->l_name && module->l_name[0] ? module->l_name : nullptr;
  auto* handle     = ::dlopen(name, RTLD_NOW | RTLD_NOLOAD);
  if (!handle) return nullptr;
  auto close = [](void* value) { ::dlclose(value); };
  std::unique_ptr<void, decltype(close)> guard(handle, close);
  auto* factory = ::dlsym(handle, symbol);
  // A handle can also resolve symbols from dependencies. Only this host may grant trust.
  if (factory && (!::dladdr1(factory, &implementation, &implementation_map, RTLD_DL_LINKMAP) ||
                  implementation_map != module))
    factory = nullptr;
  // No dynamic export (official DuckDB CLI): fall back to the static symbol table.
  if (!factory) {
    factory = host_static_symbol(module, symbol);
    if (factory && (!::dladdr1(factory, &implementation, &implementation_map, RTLD_DL_LINKMAP) ||
                    implementation_map != module))
      factory = nullptr;
  }
  return reinterpret_cast<Factory>(factory);
}

}  // namespace sirius::planner::detail
