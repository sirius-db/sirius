/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// cucascade
#include <cucascade/data/representation_converter.hpp>

namespace sirius {

/**
 * @brief Register Sirius's tier converters into @p registry.
 *
 * Replaces cuCascade's removed register_builtin_converters() (NVIDIA/cuCascade#142). Registers the
 * GPU/HOST/DISK representation converters (the cuDF-specific conversion logic that moved into
 * Sirius). Disk converters resolve the I/O backend from the disk memory_space at conversion time,
 * so each disk memory_space can use a different backend. Call once at startup after the memory
 * spaces are constructed.
 *
 * @param registry The converter registry to register converters with.
 */
void register_converters(cucascade::representation_converter_registry& registry);

}  // namespace sirius
