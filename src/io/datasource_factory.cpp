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

#include "io/io_context.hpp"
#include "io/kvikio/kvikio_context.hpp"
#include "io/object_store_config.hpp"
#include "io/reactor_placement.hpp"
#include "io/rest/rest_ioctx.hpp"
#include "io/rest/s3/sigv4_authorizer.hpp"
#include "io/rest/s3/static_credentials.hpp"
#include "io/uring/uring_ioctx.hpp"
#include "log/logging.hpp"
#include "memory/topology_index.hpp"
#include "scan_manager/config.hpp"

#include <cudf/io/datasource.hpp>

#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>

#include <algorithm>
#include <cctype>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

namespace sirius::io {

namespace {

bool same_object_store_config(object_store_config const& lhs, object_store_config const& rhs)
{
  return lhs.endpoint == rhs.endpoint && lhs.region == rhs.region &&
         lhs.access_key == rhs.access_key && lhs.secret_key == rhs.secret_key &&
         lhs.session_token == rhs.session_token && lhs.s3_transport == rhs.s3_transport &&
         lhs.s3_signing_mode == rhs.s3_signing_mode && lhs.ca_bundle_path == rhs.ca_bundle_path &&
         lhs.tls_verify == rhs.tls_verify;
}

std::size_t hash_object_store_config(object_store_config const& config)
{
  std::size_t hash = 0;
  auto combine = [&](std::size_t value) { hash ^= value + 0x9e3779b9 + (hash << 6) + (hash >> 2); };
  auto add_string = [&](std::string const& value) { combine(std::hash<std::string>{}(value)); };
  add_string(config.endpoint);
  add_string(config.region);
  add_string(config.access_key);
  add_string(config.secret_key);
  add_string(config.session_token);
  combine(std::hash<int>{}(static_cast<int>(config.s3_transport)));
  combine(std::hash<int>{}(static_cast<int>(config.s3_signing_mode)));
  add_string(config.ca_bundle_path);
  combine(std::hash<bool>{}(config.tls_verify));
  return hash;
}

std::string canonicalize_scheme(std::string_view path)
{
  std::string canonical{path};
  auto const delimiter = canonical.find("://");
  if (delimiter == std::string::npos) { return canonical; }
  for (std::size_t i = 0; i < delimiter; ++i) {
    canonical[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(canonical[i])));
  }
  return canonical;
}

}  // namespace

std::optional<std::uint64_t> scoped_object_store_configs::install(std::string_view path,
                                                                  object_store_config config)
{
  std::string scope   = canonicalize_scheme(path);
  auto const wildcard = scope.find_first_of("*?[");
  if (wildcard != std::string::npos) {
    auto const slash = scope.rfind('/', wildcard);
    scope = slash == std::string::npos ? scope.substr(0, wildcard) : scope.substr(0, slash + 1);
  }
  if (scope.empty()) return std::nullopt;
  auto snapshot_config   = std::make_shared<const object_store_config>(std::move(config));
  auto const config_hash = hash_object_store_config(*snapshot_config);
  std::lock_guard lk{_mtx};
  auto const scope_it = _scope_ids.find(scope);
  auto const old_id =
    scope_it == _scope_ids.end() ? std::nullopt : std::optional<std::uint64_t>{scope_it->second};
  if (old_id && same_object_store_config(*_snapshots.at(*old_id).value.config, *snapshot_config)) {
    return std::nullopt;
  }

  auto new_id        = std::uint64_t{0};
  auto config_ids_it = _config_ids_by_hash.find(config_hash);
  if (config_ids_it != _config_ids_by_hash.end()) {
    for (auto candidate_id : config_ids_it->second) {
      auto const& candidate = _snapshots.at(candidate_id).value;
      if (same_object_store_config(*candidate.config, *snapshot_config)) {
        new_id = candidate_id;
        break;
      }
    }
  }
  if (new_id == 0) {
    new_id = _next_id++;
    _snapshots.emplace(
      new_id, stored_snapshot{snapshot{new_id, std::move(snapshot_config)}, 0, config_hash});
    _config_ids_by_hash[config_hash].push_back(new_id);
  }
  if (old_id && *old_id == new_id) return std::nullopt;
  _scope_ids[std::move(scope)] = new_id;
  ++_snapshots.at(new_id).scope_refs;

  if (!old_id) return std::nullopt;
  auto& old_snapshot = _snapshots.at(*old_id);
  if (--old_snapshot.scope_refs != 0) return std::nullopt;
  auto const old_hash = old_snapshot.config_hash;
  _snapshots.erase(*old_id);
  auto hash_it = _config_ids_by_hash.find(old_hash);
  if (hash_it != _config_ids_by_hash.end()) {
    auto& ids = hash_it->second;
    ids.erase(std::remove(ids.begin(), ids.end(), *old_id), ids.end());
    if (ids.empty()) _config_ids_by_hash.erase(hash_it);
  }
  return old_id;
}

std::optional<scoped_object_store_configs::snapshot> scoped_object_store_configs::resolve(
  std::string_view path) const
{
  std::lock_guard lk{_mtx};
  auto canonical = canonicalize_scheme(path);
  auto candidate = std::string_view{canonical};
  while (!candidate.empty()) {
    auto it = _scope_ids.find(candidate);
    if (it != _scope_ids.end()) return _snapshots.at(it->second).value;
    if (candidate.back() == '/') {
      candidate.remove_suffix(1);
      continue;
    }
    auto const slash = candidate.rfind('/');
    if (slash == std::string_view::npos) break;
    candidate = candidate.substr(0, slash + 1);
  }
  return std::nullopt;
}

namespace {

// RFC 3986 §3.1: schemes are case-insensitive. uri_parser lowercases the
// parsed scheme on the read side; registry must normalize the same way on
// the write side so callers don't need to remember which side does it. Used
// by both register_ioctx and lookup so a register("S3", ...) is found by a
// lookup("s3"), and vice versa.
[[maybe_unused]] std::string to_lower_scheme(std::string_view s)
{
  std::string out;
  out.reserve(s.size());
  for (char c : s)
    out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
  return out;
}

// Build one context per participating host pool, keeping its staging resource
// alive through the reservation manager. Reactor count remains a total budget,
// raised only when necessary to give every participating NUMA domain a worker.
template <class Context, class MakeContext>
std::shared_ptr<Context> make_placed_ioctx(cucascade::memory::memory_reservation_manager& rm,
                                           memory::topology_index const* topology,
                                           std::size_t count,
                                           MakeContext make_context)
{
  using Reactor     = typename Context::reactor_type;
  auto const spaces = rm.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST);
  std::vector<int> nodes;
  for (auto* space : spaces)
    nodes.push_back(space->get_device_id());
  std::vector<std::pair<int, int>> gpu_nodes;
  if (topology) {
    for (int gpu : topology->gpu_ids())
      gpu_nodes.emplace_back(gpu, topology->numa_node_of(gpu));
  }
  // REST supports metadata-only contexts without a host pool. Its existing
  // device-read validation remains responsible for rejecting unstaged reads.
  auto placement = nodes.empty() ? reactor_placement{std::vector<std::size_t>(count, 0), {}}
                                 : place_reactors(count, nodes, gpu_nodes);
  if (placement.host_indices.size() > count) {
    SIRIUS_LOG_INFO("I/O reactor count raised from {} to {} for NUMA-local staging",
                    count,
                    placement.host_indices.size());
  }
  std::vector<std::shared_ptr<typename Reactor::reactor_context>> contexts(
    std::max<std::size_t>(1, spaces.size()));
  std::vector<std::unique_ptr<Reactor>> reactors;
  for (auto host : placement.host_indices) {
    if (!contexts[host]) {
      auto* mr       = spaces.empty()
                         ? nullptr
                         : spaces[host]->template get_memory_resource_of<cucascade::memory::Tier::HOST>();
      contexts[host] = make_context(mr);
    }
    reactors.push_back(std::make_unique<Reactor>(
      contexts[host], std::format("io-{}-{}", nodes.empty() ? -1 : nodes[host], reactors.size())));
  }
  for (auto const& [gpu, indices] : placement.devices) {
    SIRIUS_LOG_INFO("I/O staging GPU {} -> HOST:{} ({} reactors)",
                    gpu,
                    nodes[placement.host_indices[indices.front()]],
                    indices.size());
  }
  return std::make_shared<Context>(std::move(reactors), std::move(placement.devices));
}

/// Build a SigV4 authorizer from the object-store credentials, or nullptr when
/// the store is not configured (empty endpoint / credentials / region — which
/// disables the REST backend).  The signing form follows @c s3_signing_mode.
std::shared_ptr<rest::request_authorizer> make_s3_authorizer(const object_store_config& os)
{
  if (os.endpoint.empty() || os.region.empty() || os.access_key.empty() || os.secret_key.empty()) {
    return nullptr;
  }
  auto creds = rest::s3::static_credentials_from(os);
  switch (os.s3_signing_mode) {
    case object_store_config::signing_mode::header:
      return std::make_shared<rest::s3::sigv4_header_authorizer>(
        std::move(creds), os.region, os.endpoint);
    case object_store_config::signing_mode::presigned:
      return std::make_shared<rest::s3::sigv4_presigned_authorizer>(
        std::move(creds), os.region, os.endpoint);
  }
  return nullptr;
}

}  // namespace

using scheme_checker_type = io_context_registry::scheme_checker_type;
using factory_type        = io_context_registry::factory_type;

factory_type make_kvikio_ioctx_factory()
{
  return [](const scan_manager::scan_manager_config& config) -> std::shared_ptr<ioctx> {
    try {
      return std::make_shared<kvikio_context>(config.kvikio, config.object_store);
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("make_kvikio_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}

factory_type make_uring_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const memory::topology_index> topology)
{
  return [&reservation_manager, topology = std::move(topology)](
           const scan_manager::scan_manager_config& config) -> std::shared_ptr<ioctx> {
    try {
      if (reservation_manager.get_memory_spaces_for_tier(cucascade::memory::Tier::HOST).empty()) {
        SIRIUS_LOG_ERROR(
          "make_uring_ioctx_factory: no HOST-tier memory resource for the reactor staging");
        return nullptr;
      }
      return make_placed_ioctx<uring::uring_ioctx>(
        reservation_manager, topology.get(), config.uring_n_reactors, [&](auto* mr) {
          return std::make_shared<uring::uring_reactor::reactor_context>(config.uring, mr);
        });
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("make_uring_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}

factory_type make_rest_ioctx_factory(
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const memory::topology_index> topology)
{
  return [&reservation_manager, topology = std::move(topology)](
           const scan_manager::scan_manager_config& config) -> std::shared_ptr<ioctx> {
    try {
      auto authorizer = make_s3_authorizer(config.object_store);
      if (!authorizer) {
        SIRIUS_LOG_WARN(
          "make_rest_ioctx_factory: object store not configured (endpoint / credentials / "
          "region missing); REST backend disabled");
        return nullptr;
      }
      auto rest_cfg = config.rest;
      // The object store owns the endpoint and its TLS trust; the reactor's
      // curl GETs must verify against the same CA bundle / policy the authorizer
      // presigns for, so source these from object_store rather than rest config.
      rest_cfg.ca_bundle_path = config.object_store.ca_bundle_path;
      rest_cfg.tls_verify     = config.object_store.tls_verify;
      return make_placed_ioctx<rest::rest_ioctx>(
        reservation_manager, topology.get(), config.rest_n_reactors, [&](auto* mr) {
          return std::make_shared<rest::rest_reactor::reactor_context>(rest_cfg, authorizer, mr);
        });
    } catch (const std::exception& e) {
      SIRIUS_LOG_ERROR("make_rest_ioctx_factory: construction failed: {}", e.what());
      return nullptr;
    }
  };
}

// ---------------------------------------------------------------------------
// datasource_registry
// ---------------------------------------------------------------------------

io_context_registry::io_context_registry(
  config_type config,
  cucascade::memory::memory_reservation_manager& reservation_manager,
  std::shared_ptr<const memory::topology_index> topology)
  : _config(std::move(config)),
    _reservation_manager(reservation_manager),
    _prefer_kvikio(_config.backend == scan_manager::io_backend::kvikio)
{
  // uring / rest claim paths via their reactor's static supports() (local
  // files and s3:// URLs respectively).  kvikio is the universal fallback —
  // cudf's default datasource handles any path — so it matches everything.
  _entries.emplace(
    io_context_type::kvikio,
    entry{
      io_context_type::kvikio, [](std::string_view) { return true; }, make_kvikio_ioctx_factory()});
  _entries.emplace(io_context_type::uring,
                   entry{io_context_type::uring,
                         &uring::uring_reactor::supports,
                         make_uring_ioctx_factory(_reservation_manager, topology)});
  _entries.emplace(io_context_type::restful,
                   entry{io_context_type::restful,
                         &rest::rest_reactor::supports,
                         make_rest_ioctx_factory(_reservation_manager, topology)});
}

void io_context_registry::register_ioctx(io_context_type type,
                                         scheme_checker_type checker,
                                         factory_type factory)
{
  if (!checker) throw std::invalid_argument("datasource_registry: null scheme checker");
  if (!factory) throw std::invalid_argument("datasource_registry: null factory");
  std::lock_guard lk{_mtx};
  _entries[type] = {type, std::move(checker), std::move(factory)};
}

std::optional<io_context_type> io_context_registry::lookup_path(
  std::string_view path) const noexcept
{
  std::shared_lock lk{_mtx};
  // kvikio's checker matches everything; _entries iterates in unspecified order,
  // so defer the catch-all and let an explicit backend (uring/restful) win.
  std::optional<io_context_type> fallback;
  for (const auto& [type, entry] : _entries) {
    if (!entry.checker(path)) continue;
    if (type == io_context_type::kvikio) {
      fallback = type;
      continue;
    }
    // backend=kvikio takes over reads from BOTH explicit backends: local files
    // from uring, s3:// objects from rest (kvikIO's RemoteHandle serves them).
    // LIST still needs the rest ioctx, which callers fetch by type instead.
    if (_prefer_kvikio && (type == io_context_type::uring || type == io_context_type::restful)) {
      continue;
    }
    return type;
  }
  return fallback;
}

std::shared_ptr<ioctx> io_context_registry::make_ioctx(io_context_type type) const noexcept
{
  return make_ioctx(type, _config);
}

std::shared_ptr<ioctx> io_context_registry::make_ioctx(io_context_type type,
                                                       const config_type& config) const noexcept
{
  std::shared_lock lk{_mtx};
  auto it = _entries.find(type);
  if (it == _entries.end()) return nullptr;
  return it->second.factory(config);
}

void io_context_registry::clear()
{
  std::unique_lock lk{_mtx};
  _entries.clear();
}

}  // namespace sirius::io
