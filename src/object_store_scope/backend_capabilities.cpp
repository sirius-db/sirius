// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/backend_capabilities.hpp"

namespace sirius::object_store_scope {

// Options the configured IO backend cannot honor are refused rather than ignored. At this tree
// the only registered backend is the Sirius stack (uring + REST), whatever the configuration
// selects: no S3-over-RDMA backend exists.
capabilities capabilities_of(const sirius::scan_manager::scan_manager_config& /*scan_manager*/)
{
  return {};
}

}  // namespace sirius::object_store_scope
