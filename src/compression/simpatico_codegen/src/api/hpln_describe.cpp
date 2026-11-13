// SPDX-License-Identifier: Apache-2.0
//
// describe_hpln_header: header-only parse of a .hpln stream (see api/chunk_decode.hpp).

#include "api/chunk_decode.hpp"
#include "api/hpln_header_parse.hpp"

#include <cstdint>
#include <exception>
#include <span>
#include <stdexcept>
#include <string>

namespace simpatico {

hpln_description describe_hpln_header(std::span<const std::byte> header)
{
  hpln_description out;
  std::string err;
  try {
    err = detail::parse_hpln_description(
      {reinterpret_cast<const std::uint8_t*>(header.data()), header.size()}, out);
  } catch (std::exception const& e) {
    // The parser sizes buffers with cudf::size_of, which throws on a corrupt type tag.
    err = e.what();
  }
  if (!err.empty()) throw std::runtime_error("describe_hpln_header: " + err);
  return out;
}

}  // namespace simpatico
