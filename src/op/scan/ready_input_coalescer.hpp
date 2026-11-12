/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "op/scan/batch_coalescer.hpp"

namespace sirius::op::scan {
// A file tail may retain the last memory permit. Only defer it across an input
// boundary if a successor is already prepared and can run without another permit.
class ready_input_coalescer : public batch_coalescer {
 public:
  virtual bool can_share(scan_info const&) const                 = 0;
  virtual cursor_step advance_input(scan_info&, size_t&, size_t) = 0;
  // Complete a shareable input boundary with the coordinator's readiness snapshot.
  virtual void finish_input(cursor_step&, bool successor_ready) = 0;
  cursor_step advance(scan_info& input, size_t& cursor, size_t quantum) final
  {
    auto step = advance_input(input, cursor, quantum);
    if (step.finished && can_share(input)) finish_input(step, false);
    return step;
  }
};
}  // namespace sirius::op::scan
