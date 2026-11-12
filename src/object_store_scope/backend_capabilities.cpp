// Copyright 2026 Sirius Contributors
// SPDX-License-Identifier: Apache-2.0
#include "object_store_scope/backend_capabilities.hpp"

namespace sirius::object_store_scope {

// Options the configured IO backend cannot honor are refused rather than ignored. At this tree
// the registered backends are the Sirius stack (uring + REST) and kvikIO: no S3-over-RDMA
// backend exists, and kvikIO's remote handle consumes endpoint/region/credentials only.
capabilities capabilities_of(const sirius::scan_manager::scan_manager_config& scan_manager)
{
  if (scan_manager.backend == sirius::scan_manager::io_backend::kvikio) {
    return {.rdma_transport      = false,
            .signing_mode_choice = false,
            .custom_ca_bundle    = false,
            .backend_name        = "kvikio"};
  }
  return {};
}

}  // namespace sirius::object_store_scope
