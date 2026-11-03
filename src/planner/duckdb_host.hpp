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
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

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

/// Opens the file the loader mapped for `module`, not the file currently at its path.
///
/// Reads /proc/self/maps for an executable mapping of the module, then opens it through
/// /proc/self/map_files/<start>-<end>, which refers to the mapped inode even if the path
/// has since been replaced. If map_files is not permitted, opens the path and accepts it
/// only if its device and inode match the mapping. Returns -1 when neither is possible.
inline int open_mapped_image(link_map const* module)
{
  if (!module) return -1;
  std::string path;
  if (module->l_name && module->l_name[0]) {
    path = module->l_name;
  } else {
    char buf[PATH_MAX];
    ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return -1;
    path.assign(buf, static_cast<std::size_t>(n));
  }
  std::FILE* maps = std::fopen("/proc/self/maps", "r");
  if (!maps) return -1;
  char line[PATH_MAX + 128];
  unsigned long start = 0, end = 0, inode = 0;
  unsigned dev_major = 0, dev_minor = 0;
  bool found = false;
  while (std::fgets(line, sizeof(line), maps)) {
    unsigned long s, e, offset, ino;
    char perms[8];
    unsigned dmaj, dmin;
    int consumed = 0;
    if (std::sscanf(line,
                    "%lx-%lx %7s %lx %x:%x %lu %n",
                    &s,
                    &e,
                    perms,
                    &offset,
                    &dmaj,
                    &dmin,
                    &ino,
                    &consumed) != 7)
      continue;
    if (perms[2] != 'x' || ino == 0) continue;
    char const* name = line + consumed;
    while (*name == ' ')
      ++name;
    std::size_t len = std::strlen(name);
    while (len && (name[len - 1] == '\n' || name[len - 1] == ' '))
      --len;
    if (std::string(name, len) != path) continue;
    start = s, end = e, inode = ino, dev_major = dmaj, dev_minor = dmin, found = true;
    break;
  }
  std::fclose(maps);
  if (!found) return -1;

  char link[64];
  std::snprintf(link, sizeof(link), "/proc/self/map_files/%lx-%lx", start, end);
  int fd = ::open(link, O_RDONLY | O_CLOEXEC);
  if (fd >= 0) return fd;

  fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return -1;
  struct stat status{};
  if (::fstat(fd, &status) != 0 || status.st_ino != inode || major(status.st_dev) != dev_major ||
      minor(status.st_dev) != dev_minor) {
    ::close(fd);
    return -1;
  }
  return fd;
}

/// Resolves `symbol` from the static symbol table of the ELF image open on `fd`, which must
/// be the image loaded as `module`. Checks, in order: ELF64 little-endian; the file's program
/// header table equals the one dl_iterate_phdr reports for the module at l_addr; exactly one
/// defined STT_FUNC carries the name (GLOBAL/WEAK preferred, a single LOCAL accepted); the
/// address lies in the module (dladdr1); the function's first bytes in memory equal the file's.
/// Returns nullptr on any failure. Takes ownership of nothing; the caller closes `fd`.
inline void* host_static_symbol_in_image(link_map const* module, char const* symbol, int fd)
{
  if (!module || !symbol || fd < 0) return nullptr;
  struct stat status{};
  if (::fstat(fd, &status) != 0 || status.st_size < static_cast<off_t>(sizeof(Elf64_Ehdr)))
    return nullptr;
  auto const size = static_cast<std::size_t>(status.st_size);
  void* mapping   = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
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
  if (!in_file(header.e_phoff, std::uint64_t{header.e_phnum} * sizeof(Elf64_Phdr))) return nullptr;

  // Same program header table at the same load bias as the mapped module.
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
  auto phdr = [&](std::size_t index) {
    Elf64_Phdr value{};
    std::memcpy(&value, image + header.e_phoff + index * sizeof(Elf64_Phdr), sizeof(value));
    return value;
  };

  // One definition by name. A single LOCAL is unambiguous; several are not.
  std::size_t const want = std::strlen(symbol);
  Elf64_Sym chosen{};
  bool have_global = false;
  int locals       = 0;
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
      if (entry.st_name >= strtab.sh_size || strtab.sh_size - entry.st_name <= want) continue;
      if (std::memcmp(names + entry.st_name, symbol, want + 1) != 0) continue;
      auto const bind = ELF64_ST_BIND(entry.st_info);
      if (bind == STB_GLOBAL || bind == STB_WEAK) {
        if (have_global) return nullptr;  // duplicate global definition: not trusted
        chosen = entry, have_global = true;
      } else if (bind == STB_LOCAL) {
        if (!have_global) chosen = entry;
        ++locals;
      }
    }
  }
  if (!have_global && locals != 1) return nullptr;

  // The address must belong to the module, and the code there must be the file's code.
  auto const address = static_cast<std::uintptr_t>(module->l_addr + chosen.st_value);
  Dl_info info{};
  void* owner{};
  if (!::dladdr1(reinterpret_cast<void*>(address), &info, &owner, RTLD_DL_LINKMAP) ||
      owner != module)
    return nullptr;
  std::uint64_t file_offset = 0;
  bool located              = false;
  for (std::size_t i = 0; i < header.e_phnum; ++i) {
    auto const segment = phdr(i);
    if (segment.p_type != PT_LOAD || !(segment.p_flags & PF_X)) continue;
    if (chosen.st_value < segment.p_vaddr || chosen.st_value >= segment.p_vaddr + segment.p_filesz)
      continue;
    file_offset = chosen.st_value - segment.p_vaddr + segment.p_offset;
    located     = true;
    break;
  }
  if (!located) return nullptr;
  std::size_t const compare = std::clamp<std::size_t>(chosen.st_size, 16, 64);
  if (!in_file(file_offset, compare)) return nullptr;
  if (std::memcmp(reinterpret_cast<void const*>(address), image + file_offset, compare) != 0)
    return nullptr;
  return reinterpret_cast<void*>(address);
}

/// `host_static_symbol_in_image` on the file the loader mapped for `module`.
inline void* host_static_symbol(link_map const* module, char const* symbol)
{
  int fd = open_mapped_image(module);
  if (fd < 0) return nullptr;
  void* result = host_static_symbol_in_image(module, symbol, fd);
  ::close(fd);
  return result;
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

  // Dynamic exports first: the loader's record, no file access, no symbol scope change.
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
  // No export (official DuckDB CLI): read the mapped image's static symbol table. This is the
  // one path that opens a file; it opens the mapped inode and compares the function's bytes.
  if (!factory) factory = host_static_symbol(module, symbol);
  return reinterpret_cast<Factory>(factory);
}

}  // namespace sirius::planner::detail
