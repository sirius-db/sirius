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

#include "spill_compression.hpp"

#include "compressed_disk_representation.hpp"
#include "compressed_representation.hpp"
#include "sirius_config.hpp"
#include "spill_io.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/copying.hpp>
#include <cudf/reduction/distinct_count.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/resource_ref.hpp>

#include <absl/cleanup/cleanup.h>
#include <api/compressed_table_io.hpp>
#include <api/simpatico_codegen.hpp>
#include <codegen/util/stream_pool.hpp>
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/error.hpp>
#include <cucascade/memory/disk_table.hpp>
#include <cucascade/memory/fixed_size_host_memory_resource.hpp>
#include <cucascade/memory/memory_reservation.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <cucascade/memory/reservation_aware_resource_adaptor.hpp>
#include <log/logging.hpp>
#include <memory/sirius_memory_reservation_manager.hpp>
#include <util/host_observation.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::compression {

namespace {

std::mutex g_settings_mutex;
spill_compression_settings g_settings;
// Read on every spill; the struct behind it changes only on (re)configuration.
std::atomic<bool> g_enabled{false};

/// The plans the columns of a batch were last sampled to, kept for the next batches of the
/// same column types (see spill_compression_settings::plan_reuse_batches).
struct sampled_plans {
  /// One plan per column of the batch.
  std::vector<std::string> plans;
  /// Compressed over device size of the batch they were sampled on.
  double fraction{0.0};
  /// Batches that may still reuse them.
  std::uint64_t uses_left{0};
};

/// A batch whose compressed fraction exceeds that of the batch its plans were sampled on by
/// this factor ends their reuse: its values no longer look like the sample.
constexpr double kPlanDriftFactor = 1.25;
/// Schemas remembered; the map is emptied when it would grow past this.
constexpr std::size_t kMaxRememberedSchemas = 256;

std::mutex g_plans_mutex;
/// Keyed by the column type ids of a batch.
std::unordered_map<std::string, sampled_plans> g_plans;

}  // namespace

void set_spill_compression_settings(spill_compression_settings const& settings)
{
  {
    std::lock_guard lock(g_settings_mutex);
    g_settings = settings;
    g_enabled.store(settings.enabled, std::memory_order_release);
  }
  std::lock_guard lock(g_plans_mutex);
  g_plans.clear();
}

void apply_spill_compression_config(compression_config const& config)
{
  set_spill_compression_settings({
    .enabled                 = config.enable_spill_compression,
    .min_batch_bytes         = config.spill_min_batch_bytes,
    .max_compressed_fraction = config.spill_max_compressed_fraction,
    .encode_reserve_fraction = config.spill_encode_reserve_fraction,
    .read_cache_max_bytes    = config.spill_read_cache_max_bytes,
    .whole_decode_max_bytes  = config.spill_whole_decode_max_bytes,
    .encode_wait_ms          = config.spill_encode_wait_ms,
    .plan_reuse_batches      = config.spill_plan_reuse_batches,
  });
}

spill_compression_settings get_spill_compression_settings()
{
  std::lock_guard lock(g_settings_mutex);
  return g_settings;
}

bool spill_compression_applies(std::size_t bytes)
{
  if (!g_enabled.load(std::memory_order_acquire)) { return false; }
  return bytes >= get_spill_compression_settings().min_batch_bytes;
}

namespace {

constexpr char const* kBitpackPlan =
  "input -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDeltaRleBitpackPlan =
  "input -> delta -> differences\n"
  "delta.differences -> rle -> runs, values\n"
  "delta.differences.runs -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n"
  "delta.differences.values -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kDictionaryPlan =
  "input -> dictionary -> keys_offsets, keys_chars, indices\n"
  "dictionary.indices -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStringSplitPlan =
  "input -> str_split -> offsets, chars\n"
  "str_split.chars -> ans\n"
  "str_split.offsets -> delta -> differences\n"
  "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStringSplitRawCharsPlan =
  "input -> str_split -> offsets, chars\n"
  "str_split.offsets -> delta -> differences\n"
  "str_split.offsets.differences -> bitpack -> chunk_min, chunk_count, chunk_bits, packed\n";
constexpr char const* kStoredPlan = "input -> identity\n";

/// Rows a string column needs before a dictionary can pay for its key table.
constexpr cudf::size_type kMinDictionaryRows = 4096;
/// Most rows sampled to estimate a string column's cardinality (see plan_sample_rows).
constexpr cudf::size_type kDictionarySampleRows = 64 * 1024;
/// Dictionary-encode when at most this fraction of the sampled values are distinct.
constexpr double kMaxDictionaryDistinctFraction = 0.1;
/// Largest fixed-width column bitpack accepts. Its encode scratch holds the column at
/// full width as 32-bit words in a cudf column, which caps it at 2^31 words.
constexpr std::size_t kMaxBitpackColumnBytes = std::size_t{1} << 33;  // 8 GiB
/// Most rows sampled to choose an integer column's plan (see plan_sample_rows).
constexpr cudf::size_type kIntegerSampleRows = 64 * 1024;
/// Values per bitpack chunk; each chunk is packed at the bit width of its own range.
constexpr std::size_t kBitpackChunk = 1024;
/// A delta plan replaces plain bitpack only when it is estimated this much smaller.
constexpr double kMinDeltaSaving = 0.9;
/// A pinned column takes delta -> rle -> bitpack over the best plan a filtered scan can decode
/// in place only when rle saves at least this fraction of the column's uncompressed size more.
/// rle decodes in full on every scan, so a gain that is large as a ratio but small in bytes (a
/// sorted key whose differences already pack to almost nothing) does not pay for it.
constexpr double kMinPinRleSavingOfColumn = 0.25;

enum class plan_use { spill, pin };

/// Rows of a column sampled to choose its plan: the whole column up to @p cap rows, and for a
/// longer one 1/64 of it, between cap / 16 and cap. The sample is read back to the host (or
/// counted on the device) once per column of every spilled batch, so for a batch of a few
/// megabytes a full-size sample costs more than the encode it chooses a plan for; a batch of
/// 4 million rows or more is sampled at @p cap as before. A pinned column is encoded once and
/// always sampled at @p cap.
cudf::size_type plan_sample_rows(cudf::size_type rows, cudf::size_type cap, plan_use use)
{
  if (rows <= cap) { return rows; }
  if (use == plan_use::pin) { return cap; }
  return std::clamp(rows / 64, cap / 16, cap);
}

/// Estimated bitpack size of a stream of values pushed one at a time: each chunk at the
/// bit width of its range, plus its min/count/width header.
class bitpack_size_estimate {
 public:
  void push(__int128 value)
  {
    lo_ = count_ == 0 ? value : std::min(lo_, value);
    hi_ = count_ == 0 ? value : std::max(hi_, value);
    if (++count_ == kBitpackChunk) { close_chunk(); }
  }

  /// The estimate, counting a partial last chunk.
  double finish()
  {
    close_chunk();
    return bytes_;
  }

 private:
  void close_chunk()
  {
    if (count_ == 0) { return; }
    auto range = static_cast<unsigned __int128>(hi_ - lo_);
    int bits   = 0;
    for (; range != 0; range >>= 1) {
      ++bits;
    }
    bytes_ += static_cast<double>(count_) * bits / 8 + 16;
    count_ = 0;
  }

  __int128 lo_{0};
  __int128 hi_{0};
  std::size_t count_{0};
  double bytes_{0};
};

/// Whether @p column is integer-valued (an integer, a decimal up to 64 bits, a date or a
/// time) and so chooses its plan with integer_plan().
bool samples_as_integer(cudf::column_view const& column)
{
  auto const type = column.type();
  bool const integer_valued =
    cudf::is_integral_not_bool(type) || cudf::is_chrono(type) || cudf::is_fixed_point(type);
  return integer_valued && cudf::size_of(type) <= sizeof(std::int64_t) && column.size() > 0;
}

/// Bytes of the prefix of @p column that integer_plan() estimates on.
std::size_t integer_sample_bytes(cudf::column_view const& column, plan_use use)
{
  return static_cast<std::size_t>(plan_sample_rows(column.size(), kIntegerSampleRows, use)) *
         cudf::size_of(column.type());
}

/// Integer-valued columns (integers, decimals up to 64 bits, dates and times) choose
/// between bitpack, delta -> bitpack and delta -> rle -> bitpack by estimating each one's
/// size on a prefix sample: @p raw, the first integer_sample_bytes() of the column's values.
/// Sorted keys, which often reach a spill still in order, pack far smaller as differences,
/// and runs of equal keys smaller still as runs. The sample is bounded (kIntegerSampleRows),
/// so estimating it on the host costs little next to the GPU encode that follows.
std::string integer_plan(cudf::column_view const& column,
                         std::span<std::uint8_t const> raw,
                         plan_use use)
{
  auto const width = cudf::size_of(column.type());
  auto const rows  = raw.size() / width;

  bool const is_unsigned = cudf::is_unsigned(column.type());
  auto const shift       = 64 - 8 * static_cast<int>(width);
  bitpack_size_estimate values;
  bitpack_size_estimate deltas;
  bitpack_size_estimate run_lengths;
  bitpack_size_estimate run_values;
  __int128 previous      = 0;
  __int128 run_value     = 0;
  std::size_t run_length = 0;
  for (std::size_t i = 0; i < rows; ++i) {
    // Widened to 128 bits so differences cannot overflow.
    std::uint64_t bits = 0;
    std::memcpy(&bits, raw.data() + i * width, width);
    __int128 const value =
      is_unsigned ? static_cast<__int128>(bits) : static_cast<std::int64_t>(bits << shift) >> shift;
    // The first value is its own difference, as delta stores it.
    __int128 const delta = i == 0 ? value : value - previous;
    previous             = value;
    values.push(value);
    deltas.push(delta);
    if (run_length > 0 && delta == run_value) {
      ++run_length;
      continue;
    }
    if (run_length > 0) {
      run_lengths.push(static_cast<__int128>(run_length));
      run_values.push(run_value);
    }
    run_value  = delta;
    run_length = 1;
  }
  run_lengths.push(static_cast<__int128>(run_length));
  run_values.push(run_value);

  auto const bitpack = values.finish();
  auto const delta   = deltas.finish();
  auto const rle     = run_lengths.finish() + run_values.finish();
  if (use == plan_use::pin) {
    bool const take_delta = delta < kMinDeltaSaving * bitpack;
    auto const filterable = take_delta ? delta : bitpack;
    auto const sample     = static_cast<double>(rows * width);
    if (filterable - rle >= kMinPinRleSavingOfColumn * sample) { return kDeltaRleBitpackPlan; }
    return take_delta ? kDeltaBitpackPlan : kBitpackPlan;
  }
  if (rle < kMinDeltaSaving * std::min(bitpack, delta)) { return kDeltaRleBitpackPlan; }
  if (delta < kMinDeltaSaving * bitpack) { return kDeltaBitpackPlan; }
  return kBitpackPlan;
}

}  // namespace

namespace {

/// The plan of @p column when its type and size settle it, or null when it takes a sample.
char const* unsampled_spill_plan(cudf::column_view const& column)
{
  if (column.type().id() == cudf::type_id::STRING) {
    return column.size() < kMinDictionaryRows ? kStoredPlan : nullptr;
  }
  if (!cudf::is_fixed_width(column.type())) { return kStoredPlan; }
  auto const bytes = static_cast<std::size_t>(column.size()) * cudf::size_of(column.type());
  if (bytes > kMaxBitpackColumnBytes) { return kStoredPlan; }
  return samples_as_integer(column) ? nullptr : kBitpackPlan;
}

/// The plan of a string column of at least kMinDictionaryRows rows.
std::string string_plan(cudf::column_view const& column, ::cuda::stream_ref stream, plan_use use)
{
  // A prefix is a cheap and, for batches built from scans and joins, representative
  // sample; a wrong guess costs ratio, never correctness.
  auto const sample_rows = plan_sample_rows(column.size(), kDictionarySampleRows, use);
  auto const sample      = cudf::slice(column, {0, sample_rows}, stream).front();
  auto const distinct    = cudf::distinct_count(
    sample, cudf::null_policy::INCLUDE, cudf::nan_policy::NAN_IS_VALID, stream);
  return static_cast<double>(distinct) <= kMaxDictionaryDistinctFraction * sample_rows
           ? kDictionaryPlan
         : use == plan_use::pin ? kStringSplitRawCharsPlan
                                : kStringSplitPlan;
}

/// One plan per column of @p view, sampling every column whose type and size do not settle
/// it. The integer samples of all columns are read back together, with one wait for
/// @p stream.
std::vector<std::string> sample_plans(cudf::table_view const& view,
                                      ::cuda::stream_ref stream,
                                      plan_use use)
{
  std::vector<std::string> plans(static_cast<std::size_t>(view.num_columns()));
  std::vector<std::vector<std::uint8_t>> samples(plans.size());
  std::vector<simpatico::device_read> reads;
  for (std::size_t i = 0; i < plans.size(); ++i) {
    auto const& column = view.column(static_cast<cudf::size_type>(i));
    if (auto const* plan = unsampled_spill_plan(column)) {
      plans[i] = plan;
    } else if (samples_as_integer(column)) {
      samples[i].resize(integer_sample_bytes(column, use));
      reads.push_back({samples[i].data(),
                       column.head<std::uint8_t>() + column.offset() * cudf::size_of(column.type()),
                       samples[i].size()});
    }
  }
  if (!reads.empty()) { simpatico::read_device_bytes_completed(reads, stream); }
  for (std::size_t i = 0; i < plans.size(); ++i) {
    if (!plans[i].empty()) { continue; }
    auto const& column = view.column(static_cast<cudf::size_type>(i));
    plans[i]           = samples_as_integer(column) ? integer_plan(column, samples[i], use)
                                                    : string_plan(column, stream, use);
  }
  return plans;
}

/// The key of @p view in the remembered plans: its column type ids.
std::string schema_key(cudf::table_view const& view)
{
  std::string key;
  key.reserve(static_cast<std::size_t>(view.num_columns()) * sizeof(std::int32_t));
  for (auto const& column : view) {
    auto const id = static_cast<std::int32_t>(column.type().id());
    key.append(reinterpret_cast<char const*>(&id), sizeof id);
  }
  return key;
}

/// The plans of a batch and where they came from.
struct batch_plans {
  std::vector<std::string> plans;
  /// Taken from an earlier batch of the same column types instead of sampled.
  bool reused{false};
  /// The compressed fraction of the batch the reused plans were sampled on.
  double sampled_fraction{0.0};
};

/// The plans to encode @p view with: those remembered for its column types while they may be
/// reused, else sampled. A column whose type and size settle its plan gets that plan either
/// way, so a remembered plan is never applied to a column it does not fit.
batch_plans choose_spill_plans(cudf::table_view const& view,
                               std::string const& key,
                               bool may_reuse,
                               ::cuda::stream_ref stream)
{
  batch_plans out;
  if (may_reuse) {
    std::lock_guard lock(g_plans_mutex);
    auto const it = g_plans.find(key);
    if (it != g_plans.end() && it->second.uses_left > 0 &&
        it->second.plans.size() == static_cast<std::size_t>(view.num_columns())) {
      --it->second.uses_left;
      out.plans            = it->second.plans;
      out.reused           = true;
      out.sampled_fraction = it->second.fraction;
    }
  }
  if (!out.reused) {
    out.plans = sample_plans(view, stream, plan_use::spill);
    return out;
  }
  for (std::size_t i = 0; i < out.plans.size(); ++i) {
    if (auto const* plan = unsampled_spill_plan(view.column(static_cast<cudf::size_type>(i)))) {
      out.plans[i] = plan;
    }
  }
  return out;
}

/// Remember the sampled @p plans of a batch that compressed to @p fraction of its size, for
/// the next @p reuse_batches - 1 batches of the same column types.
void remember_spill_plans(std::string const& key,
                          std::vector<std::string> plans,
                          double fraction,
                          std::uint64_t reuse_batches)
{
  std::lock_guard lock(g_plans_mutex);
  if (g_plans.size() >= kMaxRememberedSchemas && !g_plans.contains(key)) { g_plans.clear(); }
  g_plans[key] = {std::move(plans), fraction, reuse_batches - 1};
}

/// Have the next batch of these column types sampled again.
void forget_spill_plans(std::string const& key)
{
  std::lock_guard lock(g_plans_mutex);
  g_plans.erase(key);
}

}  // namespace

std::string default_spill_plan(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  return sample_plans(cudf::table_view{{column}}, stream, plan_use::spill).front();
}

std::string default_pin_plan(cudf::column_view const& column, ::cuda::stream_ref stream)
{
  return sample_plans(cudf::table_view{{column}}, stream, plan_use::pin).front();
}

namespace {

// Encodes that hold a headroom reservation right now, per GPU. A downgrade request hands
// several batches to the downgrade threads at once, and each encode reserves a share of its
// batch's size from a device that is being relieved because it is full: the first ones get
// their headroom and the next ones do not, although it comes back within the time of one
// encode. Those wait for an encode of their GPU to end instead of spilling uncompressed.
std::mutex g_encode_mutex;
std::condition_variable g_encode_finished;
std::unordered_map<int, std::size_t> g_encodes_in_flight;

}  // namespace

void detail::note_encode_started(int device)
{
  std::lock_guard lock(g_encode_mutex);
  ++g_encodes_in_flight[device];
}

void detail::note_encode_finished(int device) noexcept
{
  {
    std::lock_guard lock(g_encode_mutex);
    auto const it = g_encodes_in_flight.find(device);
    if (it != g_encodes_in_flight.end() && it->second > 0) { --it->second; }
  }
  g_encode_finished.notify_all();
}

bool detail::wait_for_encode_end(int device, std::chrono::steady_clock::time_point deadline)
{
  std::unique_lock lock(g_encode_mutex);
  auto const it = g_encodes_in_flight.find(device);
  if (it == g_encodes_in_flight.end() || it->second == 0) { return false; }
  return g_encode_finished.wait_until(lock, deadline) != std::cv_status::timeout;
}

}  // namespace sirius::compression

namespace sirius {

namespace {

using compression::spill_compression_declined;

/// Streams the column-parallel encode runs on.
constexpr std::size_t kEncodeStreams = 4;

/// A batch encoded on the device, ready to be staged to host memory or a file. The
/// payload buffers belong to @c table and stay valid while it lives.
struct encoded_batch {
  simpatico::compressed_table table;
  std::vector<std::uint8_t> header;
  std::vector<simpatico::payload_buffer_ref> buffers;
  std::uint64_t payload_bytes{0};
  std::size_t uncompressed_bytes{0};
  std::int64_t num_rows{0};
};

std::vector<std::string> column_names(std::size_t num_columns)
{
  std::vector<std::string> names;
  names.reserve(num_columns);
  for (std::size_t i = 0; i < num_columns; ++i) {
    names.push_back("c" + std::to_string(i));
  }
  return names;
}

/// Retries the headroom reservation of @p bytes each time an encode on the GPU of @p space
/// ends, for @p wait_ms at most. Gives up at once when no encode is in flight there: nothing
/// is then about to give headroom back. Returns the reservation, or null.
std::unique_ptr<cucascade::memory::reservation> wait_for_encode_headroom(
  cucascade::memory::memory_space& space, std::size_t bytes, std::uint64_t wait_ms)
{
  auto const started  = std::chrono::steady_clock::now();
  auto const deadline = started + std::chrono::milliseconds{wait_ms};
  auto const device   = space.get_device_id();
  std::unique_ptr<cucascade::memory::reservation> reservation;
  bool waited = false;
  while (!reservation) {
    // False with no encode in flight on the GPU, or at the deadline.
    auto const encode_ended = compression::detail::wait_for_encode_end(device, deadline);
    waited = waited || encode_ended || std::chrono::steady_clock::now() >= deadline;
    if (!encode_ended) { break; }
    reservation = space.make_reservation_or_null(bytes);
  }
  if (waited) {
    spill::io_stats::get().record_encode_wait(spill::micros_since(started), reservation != nullptr);
  }
  return reservation;
}

/// Encode @p source with its default plans. Throws spill_compression_declined when the
/// encode cannot get its headroom or does not save enough; any other exception is a
/// failure the caller treats the same way.
encoded_batch encode_for_spill(const cucascade::gpu_table_representation& source,
                               ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::spill_encode"};
  source.wait_for_writer(stream);
  auto const settings = compression::get_spill_compression_settings();
  auto const view     = source.get_table_view();

  encoded_batch out;
  out.uncompressed_bytes = source.get_size_in_bytes();
  out.num_rows           = view.num_rows();

  auto& space        = source.get_memory_space();
  auto* const gpu_mr = space.get_memory_resource_of<cucascade::memory::Tier::GPU>();
  if (gpu_mr == nullptr) { throw std::logic_error("spill encode: GPU space has no allocator"); }
  // A reservation covers the allocations of the stream it is attached to, or of the
  // calling thread under per-thread tracking. Only in the latter case does it also
  // cover the stream pool the column-parallel encode runs on, so per-stream tracking
  // encodes on `stream` alone.
  auto const per_thread =
    memory::sirius_memory_reservation_manager::gpu_allocation_tracking_scope(space) ==
    cucascade::memory::reservation_aware_resource_adaptor::AllocationTrackingScope::PER_THREAD;

  // The encoder allocates from the same GPU pool the batch is being evicted from, so
  // it runs only under a reservation of its headroom, which keeps it from taking
  // memory that other tasks are about to reserve.
  bool reserved              = false;
  std::size_t reserved_bytes = 0;
  if (settings.encode_reserve_fraction > 0.0) {
    auto const bytes = static_cast<std::size_t>(settings.encode_reserve_fraction *
                                                static_cast<double>(out.uncompressed_bytes));
    auto reservation = space.make_reservation_or_null(bytes);
    if (!reservation && settings.encode_wait_ms > 0) {
      reservation = wait_for_encode_headroom(space, bytes, settings.encode_wait_ms);
    }
    if (!reservation) {
      throw spill_compression_declined(
        "no GPU headroom for a " + std::to_string(bytes) + " byte encode",
        spill::encode_outcome::no_headroom);
    }
    if (!gpu_mr->attach_reservation_to_tracker(stream, std::move(reservation))) {
      throw spill_compression_declined("the allocation tracker already holds a reservation",
                                       spill::encode_outcome::no_headroom);
    }
    reserved       = true;
    reserved_bytes = bytes;
    compression::detail::note_encode_started(space.get_device_id());
  }
  absl::Cleanup detach = [&] {
    if (!reserved) { return; }
    // What the encode allocated at most under its reservation, also when it failed.
    spill::io_stats::get().record_encode(reserved_bytes, gpu_mr->get_peak_allocated_bytes(stream));
    gpu_mr->reset_stream_reservation(stream);
    compression::detail::note_encode_finished(space.get_device_id());
  };

  auto const mr          = rmm::device_async_resource_ref{*gpu_mr};
  auto const key         = compression::schema_key(view);
  bool const remembering = settings.plan_reuse_batches > 1;
  // A batch that saves too little with plans sampled on an earlier batch is encoded once
  // more with its own.
  for (bool may_reuse = remembering;; may_reuse = false) {
    auto const plan_started = std::chrono::steady_clock::now();
    auto chosen             = compression::choose_spill_plans(view, key, may_reuse, stream);
    spill::io_stats::get().record_encode_plans(chosen.reused, spill::micros_since(plan_started));
    std::string plan;
    bool all_stored = true;
    for (std::size_t i = 0; i < chosen.plans.size(); ++i) {
      if (i > 0) { plan += "---\n"; }
      all_stored = all_stored && chosen.plans[i] == compression::kStoredPlan;
      plan += chosen.plans[i];
    }
    // Storing every column as is saves nothing, and the encode would still copy the batch.
    if (all_stored) { throw spill_compression_declined("no column of the batch compresses"); }

    if (per_thread) {
      // The pool streams are not ordered after `stream`, which the batch and the plan
      // sampling ran on.
      stream.sync();
      out.table = simpatico::compress_with_plan(
        view, plan, simpatico::thread_device_stream_pool(kEncodeStreams), mr, {});
    } else {
      out.table = simpatico::compress_with_plan(view, plan, stream, mr, {});
    }

    out.header.clear();
    out.buffers.clear();
    out.payload_bytes = 0;
    auto const error  = simpatico::build_compressed_table_header(
      out.table, out.header, out.buffers, out.payload_bytes, stream);
    if (!error.empty()) { throw std::runtime_error("build_compressed_table_header: " + error); }

    auto const compressed_bytes = out.header.size() + out.payload_bytes;
    auto const fraction =
      static_cast<double>(compressed_bytes) / static_cast<double>(out.uncompressed_bytes);
    if (fraction > settings.max_compressed_fraction) {
      if (chosen.reused) {
        compression::forget_spill_plans(key);
        out.buffers.clear();
        out.table = simpatico::compressed_table{};
        continue;
      }
      throw spill_compression_declined("compressed " + std::to_string(compressed_bytes) +
                                       " bytes of " + std::to_string(out.uncompressed_bytes));
    }
    if (chosen.reused) {
      if (fraction > compression::kPlanDriftFactor * chosen.sampled_fraction) {
        compression::forget_spill_plans(key);
      }
    } else if (remembering) {
      compression::remember_spill_plans(
        key, std::move(chosen.plans), fraction, settings.plan_reuse_batches);
    }
    break;
  }
  auto const compressed_bytes = out.header.size() + out.payload_bytes;
  SIRIUS_LOG_DEBUG("[spill_compression] compressed {} bytes to {} ({} rows, {} columns)",
                   out.uncompressed_bytes,
                   compressed_bytes,
                   out.num_rows,
                   view.num_columns());
  return out;
}

/// A file under the disk space's mount directory, deleted unless released.
class spill_file {
 public:
  explicit spill_file(cucascade::memory::memory_space const& disk_space)
    : _path(cucascade::memory::generate_disk_file_path(disk_space.get_disk_mount_path()))
  {
  }
  ~spill_file()
  {
    if (!_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(_path, ec);
    }
  }
  spill_file(const spill_file&)            = delete;
  spill_file& operator=(const spill_file&) = delete;

  [[nodiscard]] std::string const& path() const noexcept { return _path; }
  [[nodiscard]] std::string release() noexcept { return std::exchange(_path, {}); }

 private:
  std::string _path;
};

/// gpu_table to compressed_host: encode on the device, stage the payload in pinned
/// blocks of the target host space.
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_host(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::gpu_to_compressed_host"};
  auto const& gpu  = source.cast<cucascade::gpu_table_representation>();
  auto encoded     = encode_for_spill(gpu, stream);
  auto* host_space = const_cast<cucascade::memory::memory_space*>(target_memory_space);
  auto* host_mr    = host_space->get_memory_resource_of<cucascade::memory::Tier::HOST>();
  if (host_mr == nullptr) {
    throw std::runtime_error("host space has no fixed_size_host_memory_resource");
  }

  auto blob           = std::make_shared<pinned_compressed_blob>();
  blob->payload       = host_mr->allocate_multiple_blocks(encoded.payload_bytes, reservation);
  blob->payload_bytes = encoded.payload_bytes;
  for (auto const& buffer : encoded.buffers) {
    if (buffer.size_bytes > 0 && buffer.device_ptr != nullptr) {
      copy_device_to_pinned_blocks(buffer.device_ptr,
                                   *blob->payload,
                                   buffer.offset,
                                   static_cast<std::size_t>(buffer.size_bytes),
                                   stream);
    }
  }
  // The device buffers go with `encoded` at return.
  stream.sync();
  blob->header = std::move(encoded.header);

  auto const num_columns = encoded.table.num_columns();
  return std::make_unique<compressed_host_representation>(
    *host_space,
    std::move(blob),
    column_names(num_columns),
    static_cast<std::size_t>(encoded.payload_bytes),
    encoded.uncompressed_bytes,
    encoded.num_rows);
}

/// gpu_table to compressed_disk: encode on the device, write the .hpln file through the
/// disk space's I/O context.
std::unique_ptr<cucascade::idata_representation> compress_gpu_to_disk(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::gpu_to_compressed_disk"};
  auto const& gpu          = source.cast<cucascade::gpu_table_representation>();
  auto const source_device = gpu.get_memory_space().get_device_id();
  rmm::cuda_set_device_raii const device_guard{rmm::cuda_device_id{source_device}};
  int stream_device = -1;
  CUCASCADE_CUDA_TRY(cudaStreamGetDevice(stream.get(), &stream_device));
  if (stream_device != source_device) {
    throw std::invalid_argument("compressed spill: stream belongs to a different GPU");
  }
  auto encoded = encode_for_spill(gpu, stream);
  auto ctx     = compression::detail::spill_context(*target_memory_space);
  if (!ctx->supports_write() || !ctx->supports_device_write()) {
    throw std::logic_error("compressed spill: DISK context cannot write host and device memory");
  }
  auto const header = encoded.header.size();
  auto const bytes  = header + static_cast<std::size_t>(encoded.payload_bytes);
  std::vector<cucascade::io::write_slice> host_slices{{0, header, encoded.header.data()}};
  std::vector<cucascade::io::write_slice> device_slices;
  device_slices.reserve(encoded.buffers.size());
  for (auto const& buffer : encoded.buffers) {
    if (buffer.size_bytes > 0 && buffer.device_ptr != nullptr) {
      device_slices.emplace_back(header + static_cast<std::size_t>(buffer.offset),
                                 static_cast<std::size_t>(buffer.size_bytes),
                                 static_cast<std::uint8_t const*>(buffer.device_ptr));
    }
  }
  spill_file file(*target_memory_space);
  auto object =
    ctx->open_io_object_for_write(file.path(), compression::detail::spill_open_options(bytes));
  compression::detail::spill_request request{
    ctx->mixed_writev_async(*object, host_slices, device_slices, stream)};
  compression::detail::check_spill_bytes(request.get(), bytes);
  // Completion is host-observed and drains device transfers before encoded buffers die.

  auto result = std::make_unique<compressed_disk_representation>(
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    file.path(),
    header,
    bytes,
    encoded.uncompressed_bytes,
    encoded.num_rows);
  (void)file.release();
  return result;
}

/// compressed_host to compressed_disk: write the header and pinned payload as an .hpln
/// file through the disk space's I/O context, with no decode and no device work.
std::unique_ptr<cucascade::idata_representation> flush_host_to_disk(
  const cucascade::idata_representation& source,
  const cucascade::memory::memory_space* target_memory_space,
  [[maybe_unused]] ::cuda::stream_ref stream,
  [[maybe_unused]] cucascade::memory::reservation* reservation)
{
  nvtx_scoped_range nvtx_range{"sirius::compression::compressed_host_to_disk"};
  auto const& host = source.cast<compressed_host_representation>();
  if (host.selected_indices().has_value()) {
    // A projection is a view of a pinned table, never a spilled batch.
    throw std::runtime_error("cannot flush a column projection to disk");
  }

  auto const header        = host.header();
  auto const& payload      = host.payload();
  auto const payload_bytes = host.get_size_in_bytes();
  auto const block_size    = payload.block_size();
  auto ctx                 = compression::detail::spill_context(*target_memory_space);
  if (!ctx->supports_write()) {
    throw std::logic_error("compressed spill: DISK context cannot write host memory");
  }
  std::vector<cucascade::io::write_slice> slices{{0, header.size(), header.data()}};
  std::size_t written = 0;
  for (std::size_t block = 0; written < payload_bytes; ++block) {
    auto const n = std::min(block_size, payload_bytes - written);
    slices.emplace_back(
      header.size() + written, n, reinterpret_cast<std::uint8_t const*>(payload.at(block).data()));
    written += n;
  }
  spill_file file(*target_memory_space);
  auto object = ctx->open_io_object_for_write(
    file.path(), compression::detail::spill_open_options(header.size() + payload_bytes));
  compression::detail::spill_request request{ctx->host_writev_async(*object, slices)};
  compression::detail::check_spill_bytes(request.get(), header.size() + payload_bytes);

  auto result = std::make_unique<compressed_disk_representation>(
    *const_cast<cucascade::memory::memory_space*>(target_memory_space),
    file.path(),
    header.size(),
    header.size() + payload_bytes,
    host.get_uncompressed_data_size_in_bytes(),
    host.num_rows());
  (void)file.release();
  return result;
}

}  // namespace

void register_spill_compression_converters(cucascade::representation_converter_registry& registry)
{
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_host_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_host_representation>(
        compress_gpu_to_host);
  }
  if (!registry
         .has_converter<cucascade::gpu_table_representation, compressed_disk_representation>()) {
    registry
      .register_converter<cucascade::gpu_table_representation, compressed_disk_representation>(
        compress_gpu_to_disk);
  }
  if (!registry.has_converter<compressed_host_representation, compressed_disk_representation>()) {
    registry.register_converter<compressed_host_representation, compressed_disk_representation>(
      flush_host_to_disk);
  }
}

}  // namespace sirius
