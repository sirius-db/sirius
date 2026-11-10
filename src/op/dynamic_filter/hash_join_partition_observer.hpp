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

#include "op/dynamic_filter/build_input_certifier.hpp"
#include "op/partition/partition_input_observer.hpp"

namespace sirius::op {
class sirius_physical_hash_join;

/// Hash-join-owned observation of its explicitly bound build input: the join's dynamic-filter
/// construction site on every layout.
class hash_join_partition_observer final : public partition_input_observer {
 public:
  hash_join_partition_observer(sirius_physical_hash_join& join, sirius_physical_partition& input);
  void on_arrival(std::string_view, cucascade::data_batch&) override;
  bool before_claim() override;
  void on_hash_input(cucascade::read_only_data_batch const&) override;
  std::optional<uint64_t> before_transform(pipelineable_operator_data const&,
                                           cucascade::read_only_data_batch const&,
                                           ::cuda::stream_ref) override;
  void after_transform(uint64_t, cucascade::memory::memory_space&, ::cuda::stream_ref) override;
  void finalize_input() override;
  [[nodiscard]] std::size_t task_scratch_bytes() const noexcept override;

 private:
  sirius_physical_hash_join& _join;
  build_input_certifier _certifier;
};
}  // namespace sirius::op
