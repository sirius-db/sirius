// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nvcomp/version.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace simpatico::detail {

// Payload generations, not .hpln layout versions. nvCOMP 6 changed ANS and
// Cascaded's payloads; unrelated codec payloads do not need this discriminator.
// A future nvCOMP upgrade must explicitly establish which generation it supports.
static_assert(NVCOMP_VER_MAJOR == 5 || NVCOMP_VER_MAJOR == 6,
              "Review ANS and Cascaded payload compatibility for this nvCOMP version");
inline constexpr std::uint8_t kNvcompPayloadFormat = NVCOMP_VER_MAJOR;

inline std::string nvcomp_payload_compatibility_error(std::string_view codec,
                                                      std::uint8_t payload_format)
{
  if (payload_format != kNvcompPayloadFormat) {
    return std::string(codec) + ": incompatible payload format " + std::to_string(payload_format) +
           " (expected " + std::to_string(kNvcompPayloadFormat) + ")";
  }
#if NVCOMP_VER_MAJOR >= 6
  if (codec == "nvcomp_cascaded") return "nvcomp_cascaded: unavailable with nvCOMP 6";
#endif
  return {};
}

inline void require_nvcomp_payload_format(std::string_view codec, std::uint8_t payload_format)
{
  auto const error = nvcomp_payload_compatibility_error(codec, payload_format);
  if (!error.empty()) throw std::runtime_error(error);
}

}  // namespace simpatico::detail
