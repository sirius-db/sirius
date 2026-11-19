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

#include "catch.hpp"
#include "downgrade/spill_io_stats.hpp"

#include <string>

using sirius::spill::encode_outcome;
using sirius::spill::io_stats;
using Tier = cucascade::memory::Tier;

// The counters are process-wide, so every check is on the difference a call makes.
TEST_CASE("spill io stats count stores by target tier in stored bytes", "[spill_io_stats]")
{
  auto& stats       = io_stats::get();
  auto const before = stats.snapshot();
  stats.record_store(Tier::HOST, 1000, 250, true);
  stats.record_store(Tier::DISK, 250, 300, false);  // a compressed host batch written out
  stats.record_store(Tier::DISK, 4000, 4000, false);
  stats.record_store(Tier::GPU, 77, 77, true);  // not a spill: ignored
  auto const after = stats.snapshot();

  CHECK(after.to_host_batches - before.to_host_batches == 1);
  CHECK(after.to_host_source_bytes - before.to_host_source_bytes == 1000);
  CHECK(after.to_host_stored_bytes - before.to_host_stored_bytes == 250);
  CHECK(after.to_disk_batches - before.to_disk_batches == 2);
  CHECK(after.to_disk_source_bytes - before.to_disk_source_bytes == 4250);
  CHECK(after.to_disk_stored_bytes - before.to_disk_stored_bytes == 4300);
  CHECK(after.compressed_batches - before.compressed_batches == 1);
  CHECK(after.compressed_input_bytes - before.compressed_input_bytes == 1000);
  CHECK(after.compressed_output_bytes - before.compressed_output_bytes == 250);
}

TEST_CASE("spill io stats count encodes, fallbacks, reloads and payload copies", "[spill_io_stats]")
{
  auto& stats       = io_stats::get();
  auto const before = stats.snapshot();
  stats.record_uncompressed_fallback(encode_outcome::no_headroom, 10);
  stats.record_uncompressed_fallback(encode_outcome::no_headroom, 20);
  stats.record_uncompressed_fallback(encode_outcome::small_saving, 30);
  stats.record_uncompressed_fallback(encode_outcome::failed, 40);
  stats.record_encode(100, 60);
  stats.record_encode(100, 130);
  stats.record_encode_plans(false, 40);
  stats.record_encode_plans(true, 2);
  stats.record_encode_plans(true, 3);
  stats.record_reload(Tier::HOST, 5, 50);
  stats.record_reload(Tier::DISK, 7, 70);
  stats.record_reload(Tier::DISK, 9, 90, 300, true);
  stats.record_reload(Tier::GPU, 1, 1);  // a move between GPUs reads nothing back
  stats.record_payload_load(Tier::HOST, 11);
  stats.record_payload_load(Tier::DISK, 13);
  auto const after = stats.snapshot();

  CHECK(after.declined_no_headroom - before.declined_no_headroom == 2);
  CHECK(after.declined_small_saving - before.declined_small_saving == 1);
  CHECK(after.encode_failed - before.encode_failed == 1);
  CHECK(after.uncompressed_fallback_bytes - before.uncompressed_fallback_bytes == 100);
  CHECK(after.encodes - before.encodes == 2);
  CHECK(after.encode_reserved_bytes - before.encode_reserved_bytes == 200);
  CHECK(after.encode_peak_bytes - before.encode_peak_bytes == 190);
  CHECK(after.encodes_over_reservation - before.encodes_over_reservation == 1);
  CHECK(after.encode_plans_sampled - before.encode_plans_sampled == 1);
  CHECK(after.encode_plans_reused - before.encode_plans_reused == 2);
  CHECK(after.encode_plan_micros - before.encode_plan_micros == 45);
  CHECK(after.from_host_batches - before.from_host_batches == 1);
  CHECK(after.from_host_stored_bytes - before.from_host_stored_bytes == 5);
  CHECK(after.from_host_device_bytes - before.from_host_device_bytes == 50);
  CHECK(after.from_disk_batches - before.from_disk_batches == 2);
  CHECK(after.from_disk_stored_bytes - before.from_disk_stored_bytes == 16);
  CHECK(after.from_disk_device_bytes - before.from_disk_device_bytes == 160);
  CHECK(after.from_disk_micros - before.from_disk_micros == 300);
  CHECK(after.from_disk_compressed_batches - before.from_disk_compressed_batches == 1);
  CHECK(after.from_disk_compressed_stored_bytes - before.from_disk_compressed_stored_bytes == 9);
  CHECK(after.from_disk_compressed_micros - before.from_disk_compressed_micros == 300);
  CHECK(after.from_host_compressed_batches == before.from_host_compressed_batches);
  CHECK(after.payload_host_loads - before.payload_host_loads == 1);
  CHECK(after.payload_host_bytes - before.payload_host_bytes == 11);
  CHECK(after.payload_disk_loads - before.payload_disk_loads == 1);
  CHECK(after.payload_disk_bytes - before.payload_disk_bytes == 13);
}

TEST_CASE("spill io stats format every counter as key=value", "[spill_io_stats]")
{
  sirius::spill::io_totals totals;
  totals.to_disk_stored_bytes = 123456;
  totals.from_disk_batches    = 7;
  totals.payload_disk_bytes   = 99;
  auto const line             = sirius::spill::format_io_totals(totals);
  CHECK(line.find("to_disk_stored_bytes=123456 ") != std::string::npos);
  CHECK(line.find("from_disk_batches=7 ") != std::string::npos);
  CHECK(line.find("to_host_batches=0 ") == 0);
  CHECK(line.ends_with("payload_disk_bytes=99"));
}
