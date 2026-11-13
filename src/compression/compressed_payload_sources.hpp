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

#include "compression/compressed_disk_representation.hpp"
#include "compression/compressed_representation.hpp"
#include "compression/simpatico_compressed_representation.hpp"

#include <api/chunk_decode.hpp>

#include <memory>

namespace sirius::compression {

/**
 * @brief Payload source over a device-resident compressed batch.
 *
 * Zero copy: device_base() is the batch's payload and read_to_device is a device-to-device copy.
 * The source refers to the representation's memory; the caller keeps the batch alive (and
 * read-locked) for as long as the source is used.
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_device_representation& representation);

/**
 * @brief Payload source over a compressed batch held in pinned host blocks.
 *
 * read_to_device copies the requested byte range host-to-device, block by block. Same lifetime
 * rule as the device overload.
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_host_representation& representation);

/**
 * @brief Payload source over a compressed batch spilled to a .hpln file.
 *
 * read_to_device reads the requested byte range of the file (payload offsets start at
 * header_bytes()) and copies it to the device. The source shares the file, so it stays readable
 * for the source's lifetime.
 */
std::shared_ptr<simpatico::payload_source> make_payload_source(
  const compressed_disk_representation& representation);

/**
 * @brief The parsed .hpln header of a compressed representation of any tier.
 *
 * Parses on every call; the caller caches the result per call site (the row-range readers keep
 * it in their reader state).
 *
 * @throws std::runtime_error if the header is malformed.
 * @throws std::invalid_argument if @p representation is not one of the three compressed tiers.
 */
std::shared_ptr<const simpatico::hpln_description> describe(
  const simpatico_compressed_representation& representation);

}  // namespace sirius::compression
