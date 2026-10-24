// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

namespace sirius::test::paimon_fixture {
enum mode {
  valid,
  bad_magic,
  bad_version,
  bad_size,
  missing_callback,
  foreign_callback,
  null_output,
  partial_error,
  describe_error,
  invalid_view,
  valid_v2,
  missing_accepts,
  missing_describe,
  missing_release,
  mode_count
};
}
