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

#include "scan_manager/split_connector.hpp"

#include <memory>
#include <utility>

namespace sirius::scan_manager {

/**
 * @brief Grants tests the producer side of @ref split_connector without widening it for production.
 *
 * The only production producer is @c load_balancing_scan_batch_coalescer, and the only public route
 * to a populated connector (@c drain_cached_provider) produces *resident* splits and needs a GPU
 * batch — it cannot reach the metadata-split path at all. This seam is how the metadata path gets
 * tested, and it grants nothing to production code.
 *
 * @note Must stay in @c sirius::scan_manager. @c split_connector befriends this struct with an
 *       unqualified declaration, which names @c sirius::scan_manager::split_connector_test_access;
 *       a struct of the same name at global scope is a different class and is not a friend.
 */
struct split_connector_test_access {
  /// Enqueue a ready split, exactly as the coalescer's sequencer task does.
  static void push(split_connector& connector, std::unique_ptr<op::operator_data> split)
  {
    connector.push_split(std::move(split));
  }
};

}  // namespace sirius::scan_manager
