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

#include "io/gpu_ingestible.hpp"

#include <stdexcept>
#include <utility>

namespace sirius::io {

std::shared_ptr<gpu_ingestible> make_gpu_ingestible(std::unique_ptr<ingestible_table_info> info,
                                                    scan_manager::sirius_scan_manager const& mgr)
{
  if (!info) {
    throw std::runtime_error("[sirius::io::make_gpu_ingestible] table_info must not be null.");
  }
  auto* raw = info.get();
  return raw->make_ingestible(std::move(info), mgr);
}

}  // namespace sirius::io
