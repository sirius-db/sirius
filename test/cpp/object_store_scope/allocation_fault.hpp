// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>

namespace sirius::test::allocation_fault {

struct observation {
  std::size_t requests;
  bool hit;
};

bool supported() noexcept;

// When supported(), ordinal zero counts without failing. A reached nonzero ordinal fails
// once, disabling interception before unwinding or a nothrow allocation's fallback can run.
void arm(std::size_t ordinal = 0) noexcept;
observation disarm() noexcept;

}  // namespace sirius::test::allocation_fault
