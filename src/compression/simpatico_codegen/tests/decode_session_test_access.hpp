// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "decode/decode_session.hpp"

namespace simpatico {

/** Raw runtime tests borrow a session-owned frame; finish or destruction still owns completion. */
struct decode_session_test_access {
  static decode_frame& frame(decode_session& session) { return session.register_test_frame(); }
  /** Count the host upload arrays that the session's frames still hold. */
  static std::size_t host_uploads(decode_session const& session)
  {
    return session.retained_host_uploads();
  }
};

}  // namespace simpatico
