/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/batch_coalescer.hpp"

#include <absl/functional/function_ref.h>

namespace sirius::op::scan {
// A file tail may retain the last memory permit. Only defer it across an input
// boundary if a successor is already prepared and can run without another permit.
class ready_input_coalescer : public batch_coalescer {
 public:
  virtual bool can_share(scan_info const&) const = 0;
  // Consult the coordinator only at an input boundary, after bounded advance.
  virtual cursor_step advance_ready(scan_info&,
                                    size_t&,
                                    size_t,
                                    absl::FunctionRef<bool()> successor_ready) = 0;
  cursor_step advance(scan_info& input, size_t& cursor, size_t quantum) final
  {
    return advance_ready(input, cursor, quantum, [] { return false; });
  }
};
}  // namespace sirius::op::scan
