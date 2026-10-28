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

#pragma once

#include "io/ioctx_resolver.hpp"
#include "op/scan/gpu_ingestible.hpp"
#include "op/scan/gpu_ingestible_types.hpp"

#include <memory>

namespace sirius::scan_manager {

// Binds an ingestible to its file resolver. The scan operator owns its lifetime.
class split_provider {
 public:
  /// Concrete construction path — borrows the given ingestible non-owningly.
  /// The ingestible must outlive the provider; in practice the operator owns
  /// the ingestible (single shared_ptr) and the provider is created after
  /// install and torn down by scan_manager reset() before the operator goes
  /// away. Callers that need a shared_ptr can promote via
  /// `provider.get_ingestible().shared_from_this()` (enabled by
  /// @c gpu_ingestible inheriting @c std::enable_shared_from_this).
  explicit split_provider(op::scan::gpu_ingestible& ingestible, sirius::io::ioctx_resolver resolve);

  virtual ~split_provider() = default;

  /// Build this scan's preparation lanes; retain the ingestible through drain.
  std::vector<op::scan::gpu_ingestible::preparation_source> make_preparation_sources();

  /// Accessor for the borrowed ingestible. Callers that need a
  /// @c shared_ptr<gpu_ingestible> can promote via
  /// `provider.get_ingestible().shared_from_this()`.
  [[nodiscard]] op::scan::gpu_ingestible& get_ingestible() const noexcept { return *_ingestible; }

 private:
  /// Non-owning pointer to the borrowed ingestible. The operator owns the
  /// lifetime; the provider is always destroyed first via
  /// @c sirius_scan_manager::reset.
  op::scan::gpu_ingestible* _ingestible{nullptr};

  /// Resolves each file's ioctx by path (s3:// -> rest, local -> uring),
  /// forwarded into the ingestible so a mixed-scheme scan routes per file.
  sirius::io::ioctx_resolver _resolve;
};

}  // namespace sirius::scan_manager
