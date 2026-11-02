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

#include "op/scan/table_scan/owned_arrow_scan.hpp"

namespace sirius::op::scan {
namespace {
std::string identity(uint64_t sequence)
{
  return "owned_arrow_host_v1:unit=" + std::to_string(sequence);
}
class arrow_validator final : public split_validator {
  materialization_program program;

 public:
  explicit arrow_validator(materialization_program p) : program(std::move(p))
  {
    if (!program.contract || !program.generation || !program.input_types)
      throw std::invalid_argument("Arrow validator requires a bound program");
  }
  bool accepts(scan_info const& input) const noexcept override
  {
    return dynamic_cast<owned_arrow_scan_info const*>(&input) != nullptr;
  }
  split_validation_evidence inspect(split_validation_context const& context,
                                    scan_info const& input) const override
  {
    auto fail = [&](char const* text) { throw certificate_incomplete(context.contract, {}, text); };
    auto* arrow = dynamic_cast<owned_arrow_scan_info const*>(&input);
    if (!arrow) fail("Arrow validator/input mismatch");
    if (program.contract != context.contract || arrow->generation() != program.generation)
      fail("Arrow bound program mismatch");
    auto applicable = check_bit(later_check::footer_per_file) |
                      check_bit(later_check::profile_per_file) |
                      check_bit(later_check::schema_per_file) | check_bit(later_check::host_staged);
    if ((context.required & ~applicable).any())
      fail("Arrow input cannot satisfy bound requirements");
    if (!arrow->rootless() && !arrow->batch()->has_capacity()) fail("Arrow input owner missing");
    if (arrow->rootless() && (!arrow->read_identity() || arrow->preparation_bytes()))
      fail("Arrow empty source binding missing");
    try {
      if (arrow->batch())
        arrow->batch()->validate_for(program);
      else if (arrow->input_types() != *program.input_types)
        fail("Arrow empty source schema mismatch");
    } catch (std::invalid_argument const& error) {
      throw certificate_incomplete(context.contract, {}, error.what());
    }
    auto certificates = input.certificates();
    auto dependencies = input.dependencies();
    if (certificates.size() != 1) fail("Arrow input requires one certificate");
    auto const& certificate = certificates[0];
    auto const& dependency  = dependencies[0];
    if (certificate.split_id != arrow->sequence() ||
        certificate.input_identity != identity(arrow->sequence()))
      fail("Arrow unit identity mismatch");
    if (dependency.footer || dependency.datasource || dependency.parquet_approval ||
        dependency.checkpoint_iteration)
      fail("Arrow input carries foreign storage evidence");
    if (!dependency.profiles || !dependency.profiles->contains(certificate.profile))
      fail("Arrow profile missing");
    auto profile = dependency.profiles->get(certificate.profile);
    auto types   = arrow->input_types();
    if (profile.storage_version != 1 || profile.columns.size() != types.size())
      fail("Arrow profile mismatch");
    for (std::size_t i = 0; i < types.size(); ++i)
      if (profile.columns[i].type != static_cast<uint32_t>(types[i].id()) ||
          profile.columns[i].scale != types[i].scale())
        fail("Arrow profile type mismatch");
    return {{check_bit(later_check::host_staged)}, arrow->generation()};
  }
};
}  // namespace
owned_arrow_scan_info::owned_arrow_scan_info(std::shared_ptr<owned_arrow_batch> batch,
                                             scan_contract_id contract,
                                             uint64_t generation,
                                             uint64_t sequence,
                                             std::shared_ptr<physical_profile_table> profiles,
                                             std::size_t admitted_host_bytes,
                                             std::shared_ptr<void const> read_identity,
                                             std::shared_ptr<void> input_lifetime)
  : read_identity_(std::move(read_identity)),
    input_lifetime_(std::move(input_lifetime)),
    batch_(std::move(batch)),
    generation_(generation),
    sequence_(sequence),
    bytes_(0),
    host_bytes_(admitted_host_bytes)
{
  if (!batch_ || !batch_->has_capacity() || !contract || !generation || !profiles ||
      (batch_->row_count() && !host_bytes_))
    throw std::invalid_argument("Arrow scan requires owned input and a bound contract");
  bytes_ = batch_->estimated_device_bytes();
  set_profile(contract, std::move(profiles));
}
owned_arrow_scan_info::owned_arrow_scan_info(materialization_program const& program,
                                             uint64_t sequence,
                                             std::shared_ptr<physical_profile_table> profiles,
                                             std::shared_ptr<void const> read_identity)
  : read_identity_(std::move(read_identity)),
    generation_(program.generation),
    sequence_(sequence),
    bytes_(0),
    host_bytes_(0)
{
  if (!program.contract || !generation_ || !program.input_types || !profiles || !read_identity_)
    throw std::invalid_argument("Arrow empty source requires a bound schema and read identity");
  empty_types_ = *program.input_types;
  set_profile(program.contract, std::move(profiles));
}
std::vector<cudf::data_type> owned_arrow_scan_info::input_types() const
{
  return batch_ ? batch_->input_types() : empty_types_;
}
void owned_arrow_scan_info::set_profile(scan_contract_id contract,
                                        std::shared_ptr<physical_profile_table> profiles)
{
  auto types = input_types();
  physical_profile profile;
  profile.storage_version = 1;
  for (auto type : types) {
    physical_column_profile column;
    column.type  = static_cast<uint32_t>(type.id());
    column.scale = type.scale();
    profile.columns.push_back(column);
  }
  split_materializer_certificate certificate;
  certificate.contract_id    = contract;
  certificate.split_id       = sequence_;
  certificate.input_identity = identity(sequence_);
  certificate.profile        = profiles->add(std::move(profile));
  certificate.validation     = check_bit(later_check::host_staged);
  split_dependencies dependency;
  dependency.profiles = std::move(profiles);
  set_contract_payload(contract, {std::move(certificate)}, {std::move(dependency)});
}
owned_arrow_scan_info::owned_arrow_scan_info(owned_arrow_scan_info const& other)
  : read_identity_(other.read_identity_),
    input_lifetime_(other.input_lifetime_),
    batch_(other.batch_),
    empty_types_(other.empty_types_),
    generation_(other.generation_),
    sequence_(other.sequence_),
    bytes_(other.bytes_),
    host_bytes_(other.host_bytes_)
{
  set_contract_payload(other.contract_id(),
                       {other.certificates().begin(), other.certificates().end()},
                       {other.dependencies().begin(), other.dependencies().end()});
}
std::unique_ptr<owned_arrow_scan_info> owned_arrow_scan_info::coalesced_copy() const
{
  return std::unique_ptr<owned_arrow_scan_info>(new owned_arrow_scan_info(*this));
}
std::shared_ptr<split_validator const> owned_arrow_split_validator(materialization_program program)
{
  return std::make_shared<arrow_validator>(std::move(program));
}
}  // namespace sirius::op::scan
