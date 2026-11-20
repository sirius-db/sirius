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

#include "op/dynamic_filter/complete_build_inventory.hpp"
#include "op/partition/partition_input_observer.hpp"

#include <atomic>
#include <mutex>

namespace sirius::op {
class sirius_physical_hash_join;

/// Hash-join-owned observation of its explicitly bound build input.
class hash_join_partition_observer final : public partition_input_observer {
 public:
  hash_join_partition_observer(sirius_physical_hash_join& join, sirius_physical_partition& input)
    : _join(join), _input(input)
  {
  }
  void on_arrival(std::string_view, cucascade::data_batch&) override;
  bool before_claim() override;
  std::optional<uint64_t> before_transform(pipelineable_operator_data const&,
                                           cucascade::read_only_data_batch const&,
                                           ::cuda::stream_ref) override;
  void after_transform(uint64_t, cucascade::memory::memory_space&, ::cuda::stream_ref) override;
  void finalize_input() override;

 private:
  bool eligible() const;
  sirius_physical_hash_join& _join;
  sirius_physical_partition& _input;
  build_arrival_ledger _arrival_ledger;
  std::mutex _mutex;
  bool _decided = false;
  std::atomic<bool> _accumulating{false};
};
}  // namespace sirius::op
