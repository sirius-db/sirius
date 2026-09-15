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

#include "log/logging.hpp"
#include "sirius/exception.hpp"
#include "yaml_reader.hpp"  // sirius::yaml::parse_bytes

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdlib>
#include <iterator>

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
  // The size the operator actually got, on the operator's own terms. This slab sits OUTSIDE
  // the RMM pool, so it appears in no config dump and in no pool accounting -- without this
  // line the only way to learn it is to read the launcher's environment.
  SIRIUS_LOG_INFO("exchange staging arena: {} bytes (cudaMalloc)", capacity_);
  free_.emplace(0, capacity_);
}

exchange_staging_arena::~exchange_staging_arena()
{
  // The one number that says how much arena a workload ACTUALLY needed. Sizing this slab is the
  // hardest knob to set (it depends on how many exports are in flight at once, not on data
  // volume) and the arena fails hard rather than degrading, so without this line the only
  // feedback an operator gets is "exhausted" or silence -- there is no way to learn a passing
  // run had 90% headroom. Nonzero `outstanding` at teardown means a leaked lease.
  {
    std::lock_guard lock(mutex_);
    SIRIUS_LOG_INFO(
      "exchange staging arena: peak live {} of {} bytes ({} leases outstanding, {} free blocks, "
      "largest {})",
      peak_live_bytes_,
      capacity_,
      leases_.size(),
      free_.size(),
      largest_free_locked());
  }

  cudaFree(base_);
}

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
  // align_up wraps for len within kAlignment-1 of UINT64_MAX; a wrapped 0 would slip past the
  // fit scan and register a zero-length lease aliasing a live one. `len` will be wire-supplied
  // by a peer's lease request once a transport sits on top, so this guard is load-bearing, not
  // theoretical.
  if (aligned < len || aligned > capacity_) {
    throw sirius::invalid_input_exception(
      "exchange staging arena: lease of {} bytes exceeds the {} byte capacity", len, capacity_);
  }

  // Address-ordered first fit: keeps low addresses dense and needs no second index. At the tens
  // of blocks this arena holds, the linear scan is cheaper than maintaining a size index.
  for (auto it = free_.begin(); it != free_.end(); ++it) {
    if (it->second < aligned) { continue; }
    const auto offset    = it->first;
    const auto block_len = it->second;
    free_.erase(it);
    if (block_len > aligned) { free_.emplace(offset + aligned, block_len - aligned); }
    leases_.emplace(offset, aligned);
    live_bytes_ += aligned;
    peak_live_bytes_ = std::max(peak_live_bytes_, live_bytes_);
    return offset;
  }

  // Both numbers, because they mean different things: total free short of the request means
  // raise capacity (or fix retention); total free ample but largest block short means external
  // fragmentation, which a bigger arena does not necessarily fix.
  throw sirius::invalid_input_exception(
    "exchange staging arena exhausted: requested {} bytes ({} aligned), {} free of {} capacity "
    "in {} blocks (largest {}), {} leases outstanding holding {} bytes "
    "(raise SIRIUS_EXCHANGE_STAGING_BYTES)",
    len,
    aligned,
    total_free_locked(),
    capacity_,
    free_.size(),
    largest_free_locked(),
    leases_.size(),
    live_bytes_);
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
  const auto len = it->second;
  leases_.erase(it);
  live_bytes_ -= len;

  // Insert and coalesce with both neighbours, so the free list never holds two adjacent blocks
  // and released space is reusable regardless of the order releases arrive in. Merge forward
  // first (this block absorbs its successor), then backward (the predecessor absorbs the
  // result) -- doing it in the other order would leave `ins` dangling before the forward merge.
  auto [ins, ok] = free_.emplace(offset, len);
  (void)ok;  // offset came out of leases_, so it cannot already be in free_

  auto next = std::next(ins);
  if (next != free_.end() && ins->first + ins->second == next->first) {
    ins->second += next->second;
    free_.erase(next);
  }
  if (ins != free_.begin()) {
    auto prev = std::prev(ins);
    if (prev->first + prev->second == ins->first) {
      prev->second += ins->second;
      free_.erase(ins);
    }
  }
}

std::uint64_t exchange_staging_arena::total_free_locked() const
{
  std::uint64_t sum = 0;
  for (const auto& [offset, len] : free_) {
    sum += len;
  }
  return sum;
}

std::uint64_t exchange_staging_arena::largest_free_locked() const
{
  std::uint64_t best = 0;
  for (const auto& [offset, len] : free_) {
    best = std::max(best, len);
  }
  return best;
}

std::size_t exchange_staging_arena::outstanding() const
{
  std::lock_guard lock(mutex_);
  return leases_.size();
}

std::uint64_t exchange_staging_arena::total_free() const
{
  std::lock_guard lock(mutex_);
  return total_free_locked();
}

std::uint64_t exchange_staging_arena::largest_free() const
{
  std::lock_guard lock(mutex_);
  return largest_free_locked();
}

std::uint64_t exchange_staging_arena::live_bytes() const
{
  std::lock_guard lock(mutex_);
  return live_bytes_;
}

std::uint64_t exchange_staging_arena::peak_live_bytes() const
{
  std::lock_guard lock(mutex_);
  return peak_live_bytes_;
}

}  // namespace sirius::exec
