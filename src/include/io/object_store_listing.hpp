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

#include "io/s3/s3_list_parser.hpp"

#include <functional>
#include <optional>
#include <string_view>

namespace sirius::io {

/// Listing capability of an object-store backend.  The scan manager's glob
/// and LIST paths depend on this interface, not a concrete ioctx type.
/// Listing is prefix resolution on the control plane; it is independent of
/// how the data plane reads the resolved keys.
class object_store_listing {
 public:
  virtual ~object_store_listing() = default;

  /// Stream ListObjectsV2 pages under @p prefix to @p sink, one call per
  /// page.  @p sink returns false to stop early.  @p page_size is clamped
  /// to [1,1000] (0 and >1000 mean 1000).  Throws (never truncates) on a
  /// truncated page without a continuation token, and once more than
  /// @p max_scanned entries have been scanned across pages.
  virtual void list_objects_paged(std::string_view bucket,
                                  std::string_view prefix,
                                  std::size_t page_size,
                                  std::function<bool(s3::list_objects_v2_page const&)> const& sink,
                                  std::optional<std::size_t> max_scanned = std::nullopt) = 0;

  /// The backend's configured matched cap for glob resolution.
  [[nodiscard]] virtual std::size_t list_max_matches() const = 0;
};

}  // namespace sirius::io
