// SPDX-License-Identifier: Apache-2.0
#pragma once

// Internal: the header-only .hpln parse behind simpatico::describe_hpln_header. Defined in
// api/compressed_table_io.cpp, next to the parser it shares with
// read_compressed_table_from_memory. Not part of the public API.

#include "api/chunk_decode.hpp"

#include <cstdint>
#include <span>
#include <string>

namespace simpatico::detail {

/// Parses the .hpln structural header `header` into `out`. Returns an error message, or empty
/// on success. Touches neither the payload nor the device. May throw on a corrupt type tag.
std::string parse_hpln_description(std::span<const std::uint8_t> header, hpln_description& out);

}  // namespace simpatico::detail
