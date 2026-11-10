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

#include "op/sirius_physical_operator.hpp"

namespace sirius::op {

class sirius_physical_partition;

/// Input lifecycle observation. `before_transform` runs on every execute path before the output is
/// built; `on_hash_input` runs only on the hash path, before scatter. Transform completion runs
/// after the output is built, before sink.
class partition_input_observer {
 public:
  virtual ~partition_input_observer()                                             = default;
  virtual void on_arrival(std::string_view port_id, cucascade::data_batch& batch) = 0;
  virtual bool before_claim()                                                     = 0;
  virtual void on_hash_input(cucascade::read_only_data_batch const& batch)        = 0;
  virtual std::optional<uint64_t> before_transform(pipelineable_operator_data const& input,
                                                   cucascade::read_only_data_batch const& batch,
                                                   ::cuda::stream_ref stream)     = 0;
  virtual void after_transform(uint64_t id,
                               cucascade::memory::memory_space& space,
                               ::cuda::stream_ref stream)                         = 0;
  virtual void finalize_input()                                                   = 0;
  /// Bytes a task of the observed input may take from its allocator beyond its input and output; 0
  /// when the observer builds nothing.
  [[nodiscard]] virtual std::size_t task_scratch_bytes() const noexcept = 0;
};

}  // namespace sirius::op
