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
#include "op/scan/arrow_materializer.hpp"
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/table_scan/split_validation.hpp"

namespace sirius::op::scan {
// Provider-neutral input; owns the complete detached Arrow tree and its capacity.
// It supplies data/evidence, never executable validation callbacks.
class owned_arrow_scan_info final : public scan_info {
 public:
  owned_arrow_scan_info(std::shared_ptr<owned_arrow_batch>,
                        scan_contract_id,
                        uint64_t generation,
                        uint64_t sequence,
                        std::shared_ptr<physical_profile_table>,
                        std::size_t admitted_host_bytes,
                        std::shared_ptr<void const> read_identity = {},
                        std::shared_ptr<void> input_lifetime      = {});
  owned_arrow_scan_info(materialization_program const&,
                        uint64_t sequence,
                        std::shared_ptr<physical_profile_table>,
                        std::shared_ptr<void const> read_identity);
  bool rootless() const noexcept { return !batch_; }
  std::shared_ptr<void const> const& read_identity() const noexcept { return read_identity_; }
  std::vector<cudf::data_type> input_types() const;
  std::shared_ptr<owned_arrow_batch> const& batch() const noexcept { return batch_; }
  uint64_t generation() const noexcept { return generation_; }
  uint64_t sequence() const noexcept { return sequence_; }
  std::size_t estimated_bytes() const noexcept override { return bytes_; }
  std::size_t preparation_bytes() const noexcept override { return host_bytes_; }
  std::unique_ptr<owned_arrow_scan_info> coalesced_copy() const;

 private:
  owned_arrow_scan_info(owned_arrow_scan_info const&);
  std::shared_ptr<void const> read_identity_;
  std::shared_ptr<void> input_lifetime_;
  std::shared_ptr<owned_arrow_batch> batch_;
  std::vector<cudf::data_type> empty_types_;
  void set_profile(scan_contract_id, std::shared_ptr<physical_profile_table>);
  uint64_t generation_, sequence_;
  std::size_t bytes_, host_bytes_;
};
// Bind this immutable program during trusted planning, alongside Parquet if the
// scan has mixed inputs. A later provider using the same Arrow representation
// reuses this validator without changing the executor.
std::shared_ptr<split_validator const> owned_arrow_split_validator(materialization_program);
}  // namespace sirius::op::scan
