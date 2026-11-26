/*
 * Copyright 2026, Sirius Contributors.
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
#include "op/scan/arrow/arrow_batch_producer.hpp"
namespace sirius::op::scan {
struct arrow_copy_limits {
  int64_t max_rows;
  uint64_t max_copy_bytes;
  uint64_t staging_bytes;
};
namespace detail {
// The caller keeps complete source roots readable/immutable through validation and copy.
// These helpers account only for detached output/staging, never for native input allocation.
// Native producers retain resource errors. Arrow push translates an out-of-bounds offer
// to invalid_argument at the offer boundary.
void validate_arrow_copy_input(ArrowSchema const&,
                               ArrowArray const&,
                               materialization_program const&,
                               arrow_copy_limits const&,
                               arrow_metadata_lengths const& = {});
// Input must have passed validation with these exact program/bounds.
std::shared_ptr<owned_arrow_batch> copy_admitted_arrow_input(ArrowSchema const&,
                                                             ArrowArray const&,
                                                             materialization_program const&,
                                                             arrow_copy_limits const&,
                                                             admitted_arrow_input);
}  // namespace detail
}  // namespace sirius::op::scan
