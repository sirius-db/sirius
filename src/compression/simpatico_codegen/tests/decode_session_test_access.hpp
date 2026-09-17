// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/decode_session.hpp"

namespace simpatico {

/** Raw runtime tests borrow a session-owned frame; finish or destruction still owns completion. */
struct decode_session_test_access {
  static decode_frame& frame(decode_session& session) { return session.register_test_frame(); }
};

}  // namespace simpatico
