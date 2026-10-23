// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <sirius/udf.hpp>

namespace sirius_udf_example {
std::shared_ptr<sirius::gpu_scalar_function const> make_gpu_function();
}  // namespace sirius_udf_example
