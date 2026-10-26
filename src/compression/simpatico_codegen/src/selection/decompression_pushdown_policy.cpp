// SPDX-License-Identifier: Apache-2.0

#include "codegen/selection/decompression_pushdown_policy.hpp"

#include <cstdlib>
#include <string_view>

namespace sirius::codegen {

namespace {

/// "Set and not exactly 0" — the contract every boolean knob here uses.
bool env_flag(char const* name)
{
  char const* value = std::getenv(name);
  return value != nullptr && std::string_view{value} != "0";
}

/// A positive double, or @p fallback when unset, unparsable or <= 0. Rejecting
/// 0 is what makes "set it tiny" a kill switch rather than a silent default.
double env_fraction(char const* name, double fallback)
{
  char const* value = std::getenv(name);
  if (value == nullptr || *value == '\0') { return fallback; }
  char* end      = nullptr;
  double const d = std::strtod(value, &end);
  return (end != value && d > 0.0) ? d : fallback;
}

}  // namespace

bool decompression_pushdown_enabled()
{
  static bool const enabled = env_flag("SIRIUS_EXP_FUSED_SCAN_FILTER");
  return enabled;
}

bool decompression_pushdown_diag_enabled()
{
  static bool const enabled = env_flag("SIRIUS_EXP_FUSED_SCAN_DIAG");
  return enabled;
}

double decompression_pushdown_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_MAX_SEL", 0.35);
  return value;
}

double decompression_pushdown_full_route_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_TIERB_MAX_SEL", 0.10);
  return value;
}

double decompression_pushdown_index_walk_max_selectivity()
{
  static double const value = env_fraction("SIRIUS_EXP_FUSED_SCAN_K4_MAX_SEL", 0.15);
  return value;
}

std::size_t decompression_pushdown_max_membership_sources()
{
  static std::size_t const cap = [] {
    char const* value = std::getenv("SIRIUS_EXP_FUSED_SCAN_MAX_MEMBER");
    if (value == nullptr || *value == '\0') { return static_cast<std::size_t>(1); }
    char* end              = nullptr;
    long long const parsed = std::strtoll(value, &end, 10);
    return (end != value && parsed >= 0) ? static_cast<std::size_t>(parsed)
                                         : static_cast<std::size_t>(1);
  }();
  return cap;
}

}  // namespace sirius::codegen
