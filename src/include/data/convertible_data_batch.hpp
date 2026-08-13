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

#pragma once

#include "compression/compressed_disk_representation.hpp"
#include "compression/compressed_representation.hpp"
#include "compression/compression_converters.hpp"
#include "compression/output_compression.hpp"
#include "compression/plan_register.hpp"
#include "compression/compression_device_pool.hpp"
#include "compression/spill_context.hpp"
#include "data/convertible_data.hpp"
#include "data/sirius_converter_registry.hpp"
#include "log/logging.hpp"
#include "telemetry/batch_telemetry.hpp"

#include <rmm/cuda_stream_view.hpp>
#include <rmm/error.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/cudf/host_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/disk_data_representation.hpp>
#include <cucascade/memory/common.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_reservation_manager.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Concrete convertible_data wrapping a cucascade::data_batch.
 *
 * Implements the RAII mutable lock / convert / auto-release pattern for a single
 * data_batch. Acquires mutable_data_batch via to_mutable() (blocking) or
 * try_to_mutable() (non-blocking), calls convert_to on the accessor, and relies on
 * the mutable_data_batch destructor to restore the batch to idle on all exit paths
 * (success, failure, exception). No manual state save/restore is needed.
 */
class convertible_data_batch : public convertible_data {
 public:
  /**
   * @brief Construct from a shared_ptr to a cucascade data_batch.
   *
   * @param batch       The data batch to wrap (shared ownership retained).
   * @param source_repo The repository this batch came from. Used to key the
   *                    spill-plan register so the compressor can learn a plan per
   *                    operator output edge. May be null (e.g. in tests).
   */
  explicit convertible_data_batch(std::shared_ptr<cucascade::data_batch> batch,
                                  const cucascade::shared_data_repository* source_repo = nullptr)
    : _batch(std::move(batch)), _source_repo(source_repo)
  {
  }

  /**
   * @brief Convert this batch to reside in one of the target memory spaces.
   *
   * Acquires an exclusive (mutable) lock on the batch -- blocking if blocking=true,
   * non-blocking if blocking=false. Iterates target_spaces in order, requesting a
   * reservation in each via the reservation manager. On the first successful reservation
   * the batch is converted via convert_to on the mutable accessor. The mutable_data_batch
   * RAII destructor automatically restores the batch to idle on all exit paths.
   *
   * @param target_spaces  Candidate memory spaces to convert into (tried in order).
   * @param stream         CUDA stream for asynchronous memory operations.
   * @param res_mgr        Reservation manager for acquiring memory in the target space.
   * @param blocking       When true, uses to_mutable() (blocks until exclusive lock acquired).
   *                       When false, uses try_to_mutable() (returns nullopt immediately if
   *                       the lock is unavailable).
   * @return A vector of bytes converted per target space index on success, or nullopt if
   *         no conversion occurred.
   */
  std::optional<std::vector<std::size_t>> convert(
    const std::vector<const cucascade::memory::memory_space*>& target_spaces,
    rmm::cuda_stream_view stream,
    sirius::memory::sirius_memory_reservation_manager& res_mgr,
    bool blocking) override
  {
    std::optional<cucascade::mutable_data_batch> mut_opt;
    if (blocking) {
      mut_opt.emplace(_batch->to_mutable());
    } else {
      mut_opt = _batch->try_to_mutable();
      if (!mut_opt) { return std::nullopt; }
    }
    auto& mut = *mut_opt;

    // Check if the batch is already in the target space
    auto cur_space = mut.get_memory_space();
    for (std::size_t idx = 0; idx < target_spaces.size(); ++idx) {
      const auto* space = target_spaces[idx];
      if (cur_space == space) { return std::nullopt; }
    }

    auto data_size = mut.get_data()->get_size_in_bytes();

    for (std::size_t idx = 0; idx < target_spaces.size(); ++idx) {
      const auto* space = target_spaces[idx];
      auto* mem_space   = res_mgr.get_memory_space(space->get_tier(), space->get_id().device_id);
      if (!mem_space) { continue; }

      // Non-blocking reservation
      auto reservation = mem_space->make_reservation_or_null(data_size);
      if (!reservation) { continue; }

      // When downgrading off the GPU, rebind the source buffers' deallocation stream to this
      // downgrade stream so that when the conversion below frees the GPU representation, the
      // free lands on the active (downgrade) stream rather than the stream the data was
      // originally produced on. We hold the exclusive (mutable) lock here, and convert_to()
      // synchronizes `stream` after the D2H copy and before destroying the source
      // representation, so the free is correctly ordered. No-op for non-GPU-table sources.
      if (cur_space != nullptr && cur_space->get_tier() == cucascade::memory::Tier::GPU) {
        mut.rebind_stream(stream);
      }

      auto& converter_registry = sirius::converter_registry::get();

      // A batch already held compressed on the GPU (eager task-output
      // compression) re-stages rather than compresses: the bytes are finished,
      // only their location changes.
      //
      // It must NOT go through try_convert_compressed, which is gated on
      // enable_spill_compression — that setting governs whether to *spend* time
      // compressing on the spill path, a decision already taken and paid for
      // here. And it must not fall through to the uncompressed path either: no
      // converter exists from compressed_device_representation to
      // host_data_representation, so the batch would be un-evictable for the
      // rest of the query and the downgrade executor would spin against it.
      const bool already_compressed =
        dynamic_cast<const compressed_device_representation*>(mut.get_data()) != nullptr;

      switch (space->get_tier()) {
        case cucascade::memory::Tier::GPU:
          mut.convert_to<cucascade::gpu_table_representation>(
            converter_registry, *reservation, stream);
          break;
        case cucascade::memory::Tier::HOST:
          if (already_compressed) {
            mut.convert_to<compressed_host_representation>(
              converter_registry, *reservation, stream);
          } else if (!try_convert_compressed<compressed_host_representation>(
                       mut, converter_registry, *reservation, stream)) {
            mut.convert_to<cucascade::host_data_representation>(
              converter_registry, *reservation, stream);
          }
          break;
        case cucascade::memory::Tier::DISK:
          if (already_compressed) {
            mut.convert_to<compressed_disk_representation>(
              converter_registry, *reservation, stream);
          } else if (!try_convert_compressed<compressed_disk_representation>(
                       mut, converter_registry, *reservation, stream)) {
            mut.convert_to<cucascade::disk_data_representation>(
              converter_registry, *reservation, stream);
          }
          break;
        default: continue;
      }

      sirius::telemetry::batch_telemetry_registry::instance().on_tier_change(
        mut.get_batch_id(),
        space->get_tier(),
        space->get_id().device_id,
        mut.get_data()->get_size_in_bytes());

      // RAII: mutable_data_batch destructor releases the mutable lock and transitions back to idle
      // automatically.
      std::vector<std::size_t> bytes_per_target(target_spaces.size(), 0);
      bytes_per_target[idx] = data_size;
      return bytes_per_target;
    }
    return std::nullopt;
  }

  /**
   * @brief Get the size in bytes of this batch in the specified memory space.
   *
   * Acquires a shared (read-only) lock to access the memory space pointer, then
   * compares with the provided space. Returns the batch size in bytes if the batch
   * resides in the given space, 0 otherwise.
   *
   * @param space The memory space to query.
   * @return The batch size in bytes if the batch resides in the given space, 0 otherwise.
   */
  std::size_t bytes_in_space(cucascade::memory::memory_space* space) const override
  {
    auto ro = _batch->to_read_only();
    if (ro.get_memory_space() == space) { return ro.get_data()->get_size_in_bytes(); }
    return 0;
  }

  [[nodiscard]] std::size_t predicted_compression_saving() const override
  {
    return compression::estimate_device_compression(*_batch, _source_repo).predicted_freed;
  }

  [[nodiscard]] std::size_t compress_in_place(rmm::cuda_stream_view stream) override
  {
    return compression::compress_in_place_for_downgrade(*_batch, _source_repo, stream);
  }

  [[nodiscard]] bool is_device_compressed() const override
  {
    // Non-blocking. This runs while ordering every candidate, so a blocking
    // to_read_only() here deadlocks the downgrade loop against any batch another
    // thread holds exclusively. An unavailable batch is reported as
    // uncompressed: that only affects eviction *order*, and convert() re-checks
    // the real representation under its own lock before doing anything.
    auto ro = _batch->try_to_read_only();
    if (!ro) { return false; }
    return dynamic_cast<const compressed_device_representation*>(ro->get_data()) != nullptr;
  }

 private:
  /**
   * @brief Attempt a Simpatico-compressed spill into @p CompressedRep.
   *
   * Installs the spill context (so the converter can resolve — or, on the first
   * spill from this edge, explore and cache — a plan keyed by the source
   * repository), then converts. Returns false when compression is disabled, the
   * batch has no known source edge, or the conversion throws; the caller then
   * spills uncompressed.
   *
   * A throwing conversion leaves the batch untouched: convert_to() only installs
   * the new representation after the converter returns, so falling through to the
   * uncompressed path is safe.
   */
  template <typename CompressedRep>
  bool try_convert_compressed(cucascade::mutable_data_batch& mut,
                              cucascade::representation_converter_registry& converter_registry,
                              cucascade::memory::reservation& reservation,
                              rmm::cuda_stream_view stream)
  {
    // Each early return below is a *silent* skip that no other log line records,
    // so a run where compression barely fires looks identical to one where it was
    // never asked. On q3/SF1000 that hid the real limiter: 39 of 42 spilled
    // batches bypassed the encoder with only one decline logged. Tagged one line
    // per skip so the reasons can be counted from the log.
    if (!compression::spill_compression_enabled()) {
      SIRIUS_LOG_DEBUG("[convertible_data_batch] spill compression skip: suppressed-or-disabled");
      return false;
    }
    if (_source_repo == nullptr) {
      SIRIUS_LOG_DEBUG("[convertible_data_batch] spill compression skip: no source edge");
      return false;
    }

    auto& reg      = compression::plan_register::global();
    const auto ctx = compression::make_spill_context(_source_repo);

    // Too small to repay the encode. The cost per batch is roughly fixed — a
    // per-column sync to read back data-dependent output sizes, plus staging —
    // so beneath some size compressing is slower than spilling raw whatever the
    // ratio. This matters most when operator batch limits are lowered to relieve
    // GPU pressure, since that shrinks spill batches too: at SF1000 with 500 MB
    // operator batches, spill batches arrived at ~500 KB and a downgrade request
    // moved 1.06 GB across 79 of them at 71.7 MB/s, against 9,056 MB/s raw.
    //
    // Checked before decide_spill_plan so an undersized batch neither consumes a
    // plan-register use nor perturbs an edge's verdict: the batch says nothing
    // about whether that edge's *data* is compressible.
    if (const auto* data = mut.get_data();
        data != nullptr && data->get_size_in_bytes() < ctx.min_batch_bytes) {
      SIRIUS_LOG_DEBUG("[convertible_data_batch] spill compression skip: batch {}B < min {}B",
                       data->get_size_in_bytes(),
                       ctx.min_batch_bytes);
      return false;
    }

    // Count this attempt however it turns out. A skipped edge must still
    // accumulate uses, otherwise its entry never expires and the "not worth
    // compressing" verdict would stick for the rest of the query. Runs after
    // convert_to, so an explore inside the converter (which resets the count)
    // is followed by this increment rather than clobbered by it.
    struct use_noter {
      compression::plan_register& reg;
      const cucascade::shared_data_repository* repo;
      ~use_noter() { reg.note_spill_plan_use(repo); }
    } noter{reg, _source_repo};

    const auto decision = reg.decide_spill_plan(_source_repo, ctx.replan_after_uses);
    if (decision.verdict == compression::plan_register::spill_plan_verdict::skip) {
      // Compression already proved not worth it for this edge: skip the
      // conversion entirely rather than compressing and discarding it again.
      SIRIUS_LOG_DEBUG("[convertible_data_batch] spill compression skip: repo={} written off",
                       static_cast<const void*>(_source_repo));
      return false;
    }

    try {
      compression::scoped_spill_context guard(ctx);
      mut.convert_to<CompressedRep>(converter_registry, reservation, stream);
      return true;
    } catch (const sirius::spill_source_consumed&) {
      // The converter already owns (and has partly freed) this batch's columns,
      // so there is no intact source to spill uncompressed. Falling back would
      // convert an empty representation and hand downstream a zero-column batch.
      // Fail the downgrade instead; the executor logs it and moves on.
      throw;
    } catch (const std::exception& e) {
      // An OOM here is compression's own doing: the encode wanted device memory
      // during a spill, which is exactly when there is none. Suppress further
      // attempts for the rest of this pressure episode so the remaining batches
      // spill raw instead of each paying a failed encode first. The downgrade
      // monitor lifts it once the space is back under its trigger.
      //
      // This is the latch point rather than the OOM policy because
      // oom_handling_policy::handle_oom is the ordinary allocation-retry path --
      // it fires on every failed allocation under pressure, not just on the ones
      // compression caused, so latching there toggles far too often to hold for
      // an episode. Only out-of-memory declines latch; a plan that merely
      // compressed too little is a verdict about the data, not memory pressure.
      // Only when the encoder shares the query's pool. With a dedicated arena
      // the premise is gone: an OOM there means the arena's other concurrent
      // encodes have it full right now, which says nothing about query memory
      // and clears as they finish. Latching on it would disable compression for
      // the whole episode on the first burst of concurrent spills.
      if (dynamic_cast<const rmm::out_of_memory*>(&e) != nullptr &&
          !compression::compression_device_pool_enabled()) {
        compression::set_spill_compression_suppressed(true);
      }
      SIRIUS_LOG_DEBUG(
        "[convertible_data_batch] compressed spill declined ({}); "
        "falling back to uncompressed",
        e.what());
      return false;
    }
  }

  std::shared_ptr<cucascade::data_batch> _batch;
  const cucascade::shared_data_repository* _source_repo{nullptr};
};

/**
 * @brief Concrete convertible_data_provider wrapping a cucascade::shared_data_repository.
 *
 * Iterates partitions and batches within a shared_data_repository, filtering by idle
 * state and matching memory_space. The default iteration order is last-to-first
 * (back-to-front) for both partitions and batches, matching the downgrade eviction
 * pattern of preferring the most recently added data.
 * Only batches in batch_state::idle are considered.
 */
class convertible_data_batch_provider : public convertible_data_provider {
 public:
  /**
   * @brief Construct from a raw pointer to a shared_data_repository.
   * @param repo The repository to iterate (non-owning; caller ensures lifetime).
   */
  explicit convertible_data_batch_provider(cucascade::shared_data_repository* repo) : _repo(repo) {}

  /**
   * @brief Get the next convertible batch matching the given memory space.
   *
   * Iterates partitions and batches. When front_to_back is false (the typical
   * downgrade use case), iterates from last partition to first, and within each
   * partition from last batch to first. Returns the first batch that is idle and
   * resides in the requested memory space.
   *
   * @param space           The memory space to filter by.
   * @param front_to_back   Iteration direction.
   * @return A convertible_data_batch wrapping the matching batch, or nullptr.
   */
  std::unique_ptr<convertible_data> get_next_convertible(cucascade::memory::memory_space* space,
                                                         bool front_to_back) override
  {
    auto num_parts = _repo->num_partitions();
    if (num_parts == 0) { return nullptr; }

    if (front_to_back) {
      for (std::size_t p = 0; p < num_parts; ++p) {
        auto batch_ids = _repo->get_batch_ids(p);
        for (std::size_t i = 0; i < batch_ids.size(); ++i) {
          auto result = try_get_batch(batch_ids[i], p, space);
          if (result) { return result; }
        }
      }
    } else {
      for (std::size_t p = num_parts; p > 0; --p) {
        auto batch_ids = _repo->get_batch_ids(p - 1);
        for (std::size_t i = batch_ids.size(); i > 0; --i) {
          auto result = try_get_batch(batch_ids[i - 1], p - 1, space);
          if (result) { return result; }
        }
      }
    }

    return nullptr;
  }

  /**
   * @brief Get all convertible batches matching the given memory space.
   *
   * Same iteration order as get_next_convertible but collects all matching batches.
   *
   * @param space             The memory space to filter by.
   * @param front_to_back     Iteration direction.
   * @param ignore_subscribed When true (default), skip batches that have been subscribed to by a
   * task
   * @return A vector of convertible_data_batch instances (may be empty).
   */
  std::vector<std::unique_ptr<convertible_data>> get_all_convertible(
    cucascade::memory::memory_space* space,
    bool front_to_back,
    bool ignore_subscribed = true) override
  {
    std::vector<std::unique_ptr<convertible_data>> results;
    auto num_parts = _repo->num_partitions();
    if (num_parts == 0) { return results; }

    if (front_to_back) {
      for (std::size_t p = 0; p < num_parts; ++p) {
        auto batch_ids = _repo->get_batch_ids(p);
        for (std::size_t i = 0; i < batch_ids.size(); ++i) {
          auto result = try_get_batch(batch_ids[i], p, space, ignore_subscribed);
          if (result) { results.push_back(std::move(result)); }
        }
      }
    } else {
      for (std::size_t p = num_parts; p > 0; --p) {
        auto batch_ids = _repo->get_batch_ids(p - 1);
        for (std::size_t i = batch_ids.size(); i > 0; --i) {
          auto result = try_get_batch(batch_ids[i - 1], p - 1, space, ignore_subscribed);
          if (result) { results.push_back(std::move(result)); }
        }
      }
    }

    return results;
  }

  /**
   * @brief Get the total byte size of all batches in the given memory space.
   *
   * Iterates all partitions front-to-back, summing bytes for batches residing
   * in the specified space. Accesses batch data via to_read_only() as required
   * by the new cucascade API.
   *
   * @param space The memory space to query.
   * @return Total size in bytes.
   */
  std::size_t get_bytes_in_space(cucascade::memory::memory_space* space) const
  {
    std::size_t total = 0;
    auto num_parts    = _repo->num_partitions();

    for (std::size_t p = 0; p < num_parts; ++p) {
      auto batch_ids = _repo->get_batch_ids(p);
      for (auto batch_id : batch_ids) {
        auto batch = _repo->get_data_batch_by_id(batch_id, p);
        if (!batch) { continue; }
        auto ro = batch->to_read_only();
        if (ro.get_memory_space() == space) { total += ro.get_data()->get_size_in_bytes(); }
      }
    }

    return total;
  }

 private:
  /**
   * @brief Try to get a matching batch by id, checking idle state and memory space.
   *
   * Only considers batches in batch_state::idle. Accesses the memory space via
   * to_read_only() as required by the new cucascade API (get_memory_space() is
   * private on idle data_batch).
   *
   * @param batch_id          The batch ID to retrieve.
   * @param partition_idx     The partition containing the batch.
   * @param space             The target memory space to match.
   * @param ignore_subscribed When true (default), skip batches that have been subscribed to by a
   * task.
   * @return A convertible_data_batch if the batch matches, nullptr otherwise.
   */
  std::unique_ptr<convertible_data> try_get_batch(uint64_t batch_id,
                                                  std::size_t partition_idx,
                                                  cucascade::memory::memory_space* space,
                                                  bool ignore_subscribed = true) const
  {
    auto batch = _repo->get_data_batch_by_id(batch_id, partition_idx);
    if (!batch) { return nullptr; }

    // A subscribed batch is being held by a task (queued or preparing); skip it so we don't
    // downgrade data a task is about to use.
    if (ignore_subscribed && batch->get_subscriber_count() > 0) { return nullptr; }

    if (batch->get_state() != cucascade::batch_state::idle) { return nullptr; }

    auto ro = batch->try_to_read_only();
    if (!ro) { return nullptr; }
    if (ro->get_memory_space() == space) {
      return std::make_unique<convertible_data_batch>(std::move(batch), _repo);
    }

    return nullptr;
  }

  cucascade::shared_data_repository* _repo;
};

}  // namespace sirius
