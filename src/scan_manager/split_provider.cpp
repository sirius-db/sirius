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

#include "op/sirius_physical_operator.hpp"
#include "scan_manager/split_connector.hpp"

#include <utility>

namespace sirius::scan_manager {

void split_provider::push_to_connector(split_connector& connector,
                                       std::unique_ptr<op::operator_data> split)
{
  connector.push_split(std::move(split));
}

}  // namespace sirius::scan_manager
