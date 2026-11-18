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

#include "downgrade/spill_io_stats.hpp"

#include <format>

namespace sirius::spill {

namespace {

void add(std::atomic<std::uint64_t>& counter, std::uint64_t value) noexcept
{
  counter.fetch_add(value, std::memory_order_relaxed);
}

}  // namespace

io_stats& io_stats::get() noexcept
{
  static io_stats instance;
  return instance;
}

void io_stats::record_store(cucascade::memory::Tier target,
                            std::size_t source_bytes,
                            std::size_t stored_bytes,
                            bool compressed_from_gpu,
                            std::uint64_t micros) noexcept
{
  if (target == cucascade::memory::Tier::HOST) {
    add(_c.to_host_batches, 1);
    add(_c.to_host_source_bytes, source_bytes);
    add(_c.to_host_stored_bytes, stored_bytes);
  } else if (target == cucascade::memory::Tier::DISK) {
    add(_c.to_disk_batches, 1);
    add(_c.to_disk_source_bytes, source_bytes);
    add(_c.to_disk_stored_bytes, stored_bytes);
  } else {
    return;
  }
  if (compressed_from_gpu) {
    add(_c.compressed_batches, 1);
    add(_c.compressed_input_bytes, source_bytes);
    add(_c.compressed_output_bytes, stored_bytes);
    add(_c.compress_micros, micros);
  }
}

void io_stats::record_uncompressed_fallback(encode_outcome outcome,
                                            std::size_t bytes,
                                            std::uint64_t micros) noexcept
{
  switch (outcome) {
    case encode_outcome::no_headroom: add(_c.declined_no_headroom, 1); break;
    case encode_outcome::small_saving: add(_c.declined_small_saving, 1); break;
    case encode_outcome::failed: add(_c.encode_failed, 1); break;
  }
  add(_c.uncompressed_fallback_bytes, bytes);
  add(_c.fallback_micros, micros);
}

void io_stats::record_encode(std::size_t reserved_bytes, std::size_t peak_bytes) noexcept
{
  add(_c.encodes, 1);
  add(_c.encode_reserved_bytes, reserved_bytes);
  add(_c.encode_peak_bytes, peak_bytes);
  if (peak_bytes > reserved_bytes) { add(_c.encodes_over_reservation, 1); }
}

void io_stats::record_encode_wait(std::uint64_t micros, bool granted) noexcept
{
  add(_c.encode_waits, 1);
  if (granted) { add(_c.encode_waits_granted, 1); }
  add(_c.encode_wait_micros, micros);
}

void io_stats::record_reload(cucascade::memory::Tier source,
                             std::size_t stored_bytes,
                             std::size_t device_bytes,
                             std::uint64_t micros,
                             bool compressed) noexcept
{
  if (source == cucascade::memory::Tier::HOST) {
    add(_c.from_host_batches, 1);
    add(_c.from_host_stored_bytes, stored_bytes);
    add(_c.from_host_device_bytes, device_bytes);
    add(_c.from_host_micros, micros);
    if (compressed) {
      add(_c.from_host_compressed_batches, 1);
      add(_c.from_host_compressed_stored_bytes, stored_bytes);
      add(_c.from_host_compressed_micros, micros);
    }
  } else if (source == cucascade::memory::Tier::DISK) {
    add(_c.from_disk_batches, 1);
    add(_c.from_disk_stored_bytes, stored_bytes);
    add(_c.from_disk_device_bytes, device_bytes);
    add(_c.from_disk_micros, micros);
    if (compressed) {
      add(_c.from_disk_compressed_batches, 1);
      add(_c.from_disk_compressed_stored_bytes, stored_bytes);
      add(_c.from_disk_compressed_micros, micros);
    }
  }
}

void io_stats::record_payload_load(cucascade::memory::Tier source, std::size_t bytes) noexcept
{
  if (source == cucascade::memory::Tier::HOST) {
    add(_c.payload_host_loads, 1);
    add(_c.payload_host_bytes, bytes);
  } else if (source == cucascade::memory::Tier::DISK) {
    add(_c.payload_disk_loads, 1);
    add(_c.payload_disk_bytes, bytes);
  }
}

template <bool Reset>
io_totals io_stats::read() noexcept
{
  auto const value = [](std::atomic<std::uint64_t>& counter) {
    return Reset ? counter.exchange(0, std::memory_order_relaxed)
                 : counter.load(std::memory_order_relaxed);
  };
  io_totals t;
  t.to_host_batches                   = value(_c.to_host_batches);
  t.to_host_source_bytes              = value(_c.to_host_source_bytes);
  t.to_host_stored_bytes              = value(_c.to_host_stored_bytes);
  t.to_disk_batches                   = value(_c.to_disk_batches);
  t.to_disk_source_bytes              = value(_c.to_disk_source_bytes);
  t.to_disk_stored_bytes              = value(_c.to_disk_stored_bytes);
  t.compressed_batches                = value(_c.compressed_batches);
  t.compressed_input_bytes            = value(_c.compressed_input_bytes);
  t.compressed_output_bytes           = value(_c.compressed_output_bytes);
  t.compress_micros                   = value(_c.compress_micros);
  t.declined_no_headroom              = value(_c.declined_no_headroom);
  t.declined_small_saving             = value(_c.declined_small_saving);
  t.encode_failed                     = value(_c.encode_failed);
  t.uncompressed_fallback_bytes       = value(_c.uncompressed_fallback_bytes);
  t.fallback_micros                   = value(_c.fallback_micros);
  t.encodes                           = value(_c.encodes);
  t.encode_reserved_bytes             = value(_c.encode_reserved_bytes);
  t.encode_peak_bytes                 = value(_c.encode_peak_bytes);
  t.encodes_over_reservation          = value(_c.encodes_over_reservation);
  t.encode_waits                      = value(_c.encode_waits);
  t.encode_waits_granted              = value(_c.encode_waits_granted);
  t.encode_wait_micros                = value(_c.encode_wait_micros);
  t.from_host_batches                 = value(_c.from_host_batches);
  t.from_host_stored_bytes            = value(_c.from_host_stored_bytes);
  t.from_host_device_bytes            = value(_c.from_host_device_bytes);
  t.from_host_micros                  = value(_c.from_host_micros);
  t.from_disk_batches                 = value(_c.from_disk_batches);
  t.from_disk_stored_bytes            = value(_c.from_disk_stored_bytes);
  t.from_disk_device_bytes            = value(_c.from_disk_device_bytes);
  t.from_disk_micros                  = value(_c.from_disk_micros);
  t.from_host_compressed_batches      = value(_c.from_host_compressed_batches);
  t.from_host_compressed_stored_bytes = value(_c.from_host_compressed_stored_bytes);
  t.from_host_compressed_micros       = value(_c.from_host_compressed_micros);
  t.from_disk_compressed_batches      = value(_c.from_disk_compressed_batches);
  t.from_disk_compressed_stored_bytes = value(_c.from_disk_compressed_stored_bytes);
  t.from_disk_compressed_micros       = value(_c.from_disk_compressed_micros);
  t.payload_host_loads                = value(_c.payload_host_loads);
  t.payload_host_bytes                = value(_c.payload_host_bytes);
  t.payload_disk_loads                = value(_c.payload_disk_loads);
  t.payload_disk_bytes                = value(_c.payload_disk_bytes);
  return t;
}

io_totals io_stats::snapshot() const noexcept { return const_cast<io_stats*>(this)->read<false>(); }

io_totals io_stats::take() noexcept { return read<true>(); }

std::string format_io_totals(const io_totals& t)
{
  return std::format(
    "to_host_batches={} to_host_source_bytes={} to_host_stored_bytes={} to_disk_batches={} "
    "to_disk_source_bytes={} to_disk_stored_bytes={} compressed_batches={} "
    "compressed_input_bytes={} compressed_output_bytes={} compress_micros={} "
    "declined_no_headroom={} declined_small_saving={} encode_failed={} "
    "uncompressed_fallback_bytes={} fallback_micros={} encodes={} encode_reserved_bytes={} "
    "encode_peak_bytes={} encodes_over_reservation={} encode_waits={} encode_waits_granted={} "
    "encode_wait_micros={} from_host_batches={} "
    "from_host_stored_bytes={} from_host_device_bytes={} from_host_micros={} "
    "from_disk_batches={} from_disk_stored_bytes={} from_disk_device_bytes={} "
    "from_disk_micros={} from_host_compressed_batches={} from_host_compressed_stored_bytes={} "
    "from_host_compressed_micros={} from_disk_compressed_batches={} "
    "from_disk_compressed_stored_bytes={} from_disk_compressed_micros={} payload_host_loads={} "
    "payload_host_bytes={} payload_disk_loads={} payload_disk_bytes={}",
    t.to_host_batches,
    t.to_host_source_bytes,
    t.to_host_stored_bytes,
    t.to_disk_batches,
    t.to_disk_source_bytes,
    t.to_disk_stored_bytes,
    t.compressed_batches,
    t.compressed_input_bytes,
    t.compressed_output_bytes,
    t.compress_micros,
    t.declined_no_headroom,
    t.declined_small_saving,
    t.encode_failed,
    t.uncompressed_fallback_bytes,
    t.fallback_micros,
    t.encodes,
    t.encode_reserved_bytes,
    t.encode_peak_bytes,
    t.encodes_over_reservation,
    t.encode_waits,
    t.encode_waits_granted,
    t.encode_wait_micros,
    t.from_host_batches,
    t.from_host_stored_bytes,
    t.from_host_device_bytes,
    t.from_host_micros,
    t.from_disk_batches,
    t.from_disk_stored_bytes,
    t.from_disk_device_bytes,
    t.from_disk_micros,
    t.from_host_compressed_batches,
    t.from_host_compressed_stored_bytes,
    t.from_host_compressed_micros,
    t.from_disk_compressed_batches,
    t.from_disk_compressed_stored_bytes,
    t.from_disk_compressed_micros,
    t.payload_host_loads,
    t.payload_host_bytes,
    t.payload_disk_loads,
    t.payload_disk_bytes);
}

}  // namespace sirius::spill
