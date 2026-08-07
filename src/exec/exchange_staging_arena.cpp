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

#include "exec/exchange_staging_arena.hpp"

#include "sirius/exception.hpp"
#include "yaml_reader.hpp"  // sirius::yaml::parse_bytes

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>

namespace sirius::exec {

namespace {
constexpr std::uint64_t align_up(std::uint64_t len)
{
  return (len + exchange_staging_arena::kAlignment - 1) & ~(exchange_staging_arena::kAlignment - 1);
}
}  // namespace

exchange_staging_arena::exchange_staging_arena(std::uint64_t capacity_bytes)
  : capacity_(capacity_bytes)
{
  if (capacity_bytes == 0) {
    throw sirius::invalid_input_exception("exchange staging arena: capacity must be nonzero");
  }
  // Plain cudaMalloc, by contract (see the class comment): pool memory silently loses the
  // transport's GPU-to-GPU fast path.
  if (auto err = cudaMalloc(&base_, capacity_bytes); err != cudaSuccess) {
    throw sirius::internal_exception("exchange staging arena: cudaMalloc of {} bytes failed: {}",
                                     capacity_bytes,
                                     cudaGetErrorString(err));
  }
}

exchange_staging_arena::~exchange_staging_arena() { cudaFree(base_); }

std::unique_ptr<exchange_staging_arena> exchange_staging_arena::from_env()
{
  const char* value = std::getenv(kCapacityEnvVar);
  if (value == nullptr) { return nullptr; }
  std::uint64_t bytes = 0;
  try {
    bytes = sirius::yaml::parse_bytes(value);
  } catch (const std::exception& e) {
    throw sirius::invalid_input_exception(std::string(kCapacityEnvVar) + ": " + e.what());
  }
  return std::make_unique<exchange_staging_arena>(bytes);
}

exchange_staging_arena& exchange_staging_arena::require(exchange_staging_arena* arena)
{
  if (arena == nullptr) {
    throw sirius::invalid_input_exception(
      "exchange staging arena not configured (set SIRIUS_EXCHANGE_STAGING_BYTES)");
  }
  return *arena;
}

std::uint64_t exchange_staging_arena::lease(std::uint64_t len)
{
  if (len == 0) {
    // A zero-length lease would alias the next lease's offset and break release-by-offset.
    throw sirius::invalid_input_exception("exchange staging arena: zero-length lease");
  }
  std::lock_guard lock(mutex_);
  const auto aligned = align_up(len);
  const auto free    = capacity_ - head_;
  if (aligned > free) {
    throw sirius::invalid_input_exception(
      "exchange staging arena exhausted: requested {} bytes ({} aligned), {} free of {} capacity "
      "with {} leases outstanding (raise SIRIUS_EXCHANGE_STAGING_BYTES)",
      len,
      aligned,
      free,
      capacity_,
      leases_.size());
  }
  const auto offset = head_;
  head_ += aligned;
  high_water_ = std::max(high_water_, head_);
  leases_.emplace(offset, aligned);
  return offset;
}

void exchange_staging_arena::release(std::uint64_t offset)
{
  std::lock_guard lock(mutex_);
  auto it = leases_.find(offset);
  if (it == leases_.end()) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: release of offset {} which is not an outstanding lease "
      "(double release?)",
      offset);
  }
  leases_.erase(it);
  // Trailing reclamation, no free list: the bump head drops back to the end of the highest
  // lease still outstanding (to the base when none remain). Gaps below the head are only
  // reusable once everything above them goes back — but a lease that outlives its neighbours
  // now pins at most the region up to its own end, never the whole arena, so steady-state
  // traffic keeps reusing the same space above it instead of burning arena for the process
  // lifetime.
  if (leases_.empty()) {
    head_ = 0;
  } else {
    const auto& highest = *leases_.rbegin();
    head_               = highest.first + highest.second;
  }
}

std::size_t exchange_staging_arena::outstanding() const
{
  std::lock_guard lock(mutex_);
  return leases_.size();
}

std::uint64_t exchange_staging_arena::high_water() const
{
  std::lock_guard lock(mutex_);
  return high_water_;
}

}  // namespace sirius::exec
