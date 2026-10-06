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
#include <link.h>

#include <cstring>
#include <memory>

namespace sirius::planner::detail {
// On the Linux Itanium C++ ABI, Catalog's first virtual slot is its destructor.
// DuckCatalog defines that destructor out of line in DuckDB. Unlike RTTI and
// vtables, its code is not subject to executable copy relocations.
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
  if (!factory || !::dladdr1(factory, &implementation, &implementation_map, RTLD_DL_LINKMAP) ||
      implementation_map != module)
    return nullptr;
  return reinterpret_cast<Factory>(factory);
}

}  // namespace sirius::planner::detail
