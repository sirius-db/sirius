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

#include "scan_manager/split_provider.hpp"

#include "op/scan/gpu_ingestible.hpp"

#include <utility>

namespace sirius::scan_manager {

split_provider::split_provider(op::scan::gpu_ingestible& ingestible, io::ioctx_resolver resolve)
  : _ingestible(&ingestible), _resolve(std::move(resolve))
{
}

std::vector<op::scan::gpu_ingestible::preparation_source> split_provider::make_preparation_sources()
{
  return _ingestible->make_preparation_sources(_resolve);
}

}  // namespace sirius::scan_manager
