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

#include "query_id.hpp"

#include <cucascade/data/data_repository.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <vector>

namespace sirius::data {
class data_repository_manager_registry;
}

namespace sirius::exec {

/**
 * @brief Makes exchange data spillable: parked fragment output, stream inputs and received
 * batches.
 *
 * The downgrade executor only sweeps repositories registered in the data repository registry,
 * and exchange repositories belong to fragments, outside any query's pipeline wiring. So a GPU
 * under pressure could not move exchange data to host, and a query whose shuffle exceeded the
 * pool failed outright. One forwarding repository, registered for the context's lifetime under
 * a reserved query id, views every tracked repository; the executor converts the batches it
 * finds there in place, so the owning repository sees them on host and they move back to the
 * GPU when an operator or an export next touches them.
 *
 * Tracking holds a weak reference: it never extends a repository's life, and a repository that
 * is gone drops out of the view. The reserved id is the newest possible, and the executor spills
 * newest-first, so exchange data spills before any running query's pipeline data.
 */
class exchange_staging {
 public:
  /// The reserved registry slot. Query ids count up from 0 and never reach it.
  static constexpr sirius::query_id_t query_id =
    sirius::make_query_id(std::numeric_limits<std::uint32_t>::max());

  explicit exchange_staging(sirius::data::data_repository_manager_registry& registry);

  /// Exposes @p repository to the downgrade executor until it is destroyed.
  void track(std::shared_ptr<cucascade::shared_data_repository> const& repository);

  /// Repositories currently tracked and still alive.
  [[nodiscard]] std::size_t tracked() const;

 private:
  struct targets {
    mutable std::mutex mutex;
    std::vector<std::weak_ptr<cucascade::shared_data_repository>> repositories;
  };
  std::shared_ptr<targets> _targets;
};

}  // namespace sirius::exec
