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

#include "op/dynamic_filter/hash_join_partition_observer.hpp"

#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_partition.hpp"

namespace sirius::op {

hash_join_partition_observer::hash_join_partition_observer(sirius_physical_hash_join& join,
                                                           sirius_physical_partition& input)
  : _join(join), _certifier(input, join.dynamic_filter_session())
{
}

void hash_join_partition_observer::on_arrival(std::string_view port_id,
                                              cucascade::data_batch& batch)
{
  _certifier.on_arrival(port_id, batch);
}

bool hash_join_partition_observer::before_claim() { return _certifier.decide(); }

void hash_join_partition_observer::on_hash_input(cucascade::read_only_data_batch const& batch)
{
  // Join-wide MARK build facts (any row, any NULL key) for a hash-partitioned build.
  _join.observe_partitioned_build_batch(batch);
}

std::optional<uint64_t> hash_join_partition_observer::before_transform(
  pipelineable_operator_data const& input,
  cucascade::read_only_data_batch const& batch,
  ::cuda::stream_ref stream)
{
  return _certifier.contribute(input, batch, stream);
}

void hash_join_partition_observer::after_transform(uint64_t id,
                                                   cucascade::memory::memory_space& space,
                                                   ::cuda::stream_ref stream)
{
  _join.dynamic_filter_session().publish_if_final(id, space, stream);
}

void hash_join_partition_observer::finalize_input()
{
  auto& session = _join.dynamic_filter_session();
  session.finalize_input();
  D_ASSERT(!session.plan().enabled() || session.terminal());
}

std::size_t hash_join_partition_observer::task_scratch_bytes() const noexcept
{
  return _join.dynamic_filter_session().task_scratch_bound();
}

}  // namespace sirius::op
