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

#include "op/sirius_physical_partition_consumer_operator.hpp"

#include "sirius/exception.hpp"

namespace sirius {
namespace op {

sirius_physical_partition_consumer_operator::~sirius_physical_partition_consumer_operator() {}

void sirius_physical_partition_consumer_operator::set_placement(
  std::shared_ptr<const partition_placement> placement)
{
  if (placement == nullptr) {
    throw sirius::internal_exception("set_placement called with a null placement on " + get_name());
  }
  auto expected = _placement.load(std::memory_order_acquire);
  while (expected == nullptr) {
    if (_placement.compare_exchange_weak(expected, placement, std::memory_order_acq_rel)) {
      return;
    }
  }
  if (*expected != *placement) {
    throw sirius::internal_exception("set_placement: " + get_name() + " already has placement " +
                                     expected->to_string() + ", refusing " +
                                     placement->to_string());
  }
}

std::shared_ptr<const partition_placement> sirius_physical_partition_consumer_operator::placement()
  const
{
  auto installed = _placement.load(std::memory_order_acquire);
  if (installed == nullptr) {
    throw sirius::internal_exception("placement: " + get_name() +
                                     " has no installed partition placement");
  }
  return installed;
}

}  // namespace op
}  // namespace sirius
