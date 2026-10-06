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

#include "io/scoped_object_store_configs.hpp"

#include <algorithm>
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

}  // namespace

std::optional<std::uint64_t> scoped_object_store_configs::install(std::string_view path,
                                                                  object_store_config config)
{
  std::string scope{path};
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
  auto candidate = path;
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

}  // namespace sirius::io
