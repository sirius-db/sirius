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

#include "io/datasource_factory.hpp"

#include <cudf/io/datasource.hpp>

#include <cctype>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {

// RFC 3986 §3.1: schemes are case-insensitive. uri_parser lowercases the
// parsed scheme on the read side; registry must normalize the same way on
// the write side so callers don't need to remember which side does it. Used
// by both register_ioctx and lookup so a register("S3", ...) is found by a
// lookup("s3"), and vice versa.
std::string to_lower_scheme(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

}  // namespace

using scheme_checker_type = io_context_registry::scheme_checker_type;
using factory_type        = io_context_registry::factory_type;

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

io_context_registry::io_context_registry(config_type config) : _config(std::move(config)) {}

void io_context_registry::register_ioctx(io_context_type type,
                                         scheme_checker_type checker,
                                         factory_type factory)
{
  if (!checker) throw std::invalid_argument("datasource_registry: null scheme checker");
  if (!factory) throw std::invalid_argument("datasource_registry: null factory");
  std::lock_guard lk{_mtx};
  _entries[type] = {std::move(checker), std::move(factory), type};
}

std::optional<io_context_type> io_context_registry::lookup(std::string_view scheme) const noexcept
{
  std::shared_lock lk{_mtx};
  for (const auto& [type, entry] : _entries) {
    if (entry.checker(scheme)) return type;
  }
  return std::nullopt;
}

std::shared_ptr<sirius_ioctx> io_context_registry::make_ioctx(io_context_type type) const noexcept
{
  std::shared_lock lk{_mtx};
  auto it = _entries.find(type);
  if (it == _entries.end()) return nullptr;
  return it->second.factory(_config);
}

void io_context_registry::clear()
{
  std::unique_lock lk{_mtx};
  _entries.clear();
}

}  // namespace sirius::io
