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

#include "op/dynamic_filter/build_input_certifier.hpp"

#include "log/logging.hpp"
#include "op/dynamic_filter/build_input_certification.hpp"
#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "op/sirius_physical_operator.hpp"

#include <utility>
#include <variant>

namespace sirius::op {

build_input_certifier::build_input_certifier(sirius_physical_operator const& site,
                                             dynamic_filter_publication_session& session) noexcept
  : _site(site), _session(session)
{
}

void build_input_certifier::on_arrival(std::string_view port_id, cucascade::data_batch& batch)
{
  if (port_id != "default" || !_session.plan().enabled()) { return; }
  auto const* port = _site.try_get_port(port_id);
  if (port != nullptr && port->type == MemoryBarrierType::FULL) { _arrival_ledger.record(batch); }
}

bool build_input_certifier::decide()
{
  std::scoped_lock guard(_decision_mutex);
  if (_decided) { return true; }
  if (_session.plan().enabled()) {
    auto const facts    = read_build_port_facts(_site);
    auto const decision = certify_decision(facts);
    if (std::holds_alternative<certification_pending>(decision)) { return false; }
    auto certified = std::holds_alternative<certification_proceed>(decision)
                       ? _arrival_ledger.certify({.repository_batches = facts.repository_batches})
                       : certified_build_input{std::get<construction_decline>(decision)};
    if (auto const* decline = std::get_if<construction_decline>(&certified)) {
      SIRIUS_LOG_INFO("[build_input_certifier] {} (id={}) constructs no dynamic filter: {}.",
                      _site.get_name(),
                      _site.has_operator_id() ? _site.get_operator_id()
                                              : sirius_physical_operator::invalid_operator_id,
                      to_string(*decline));
      _session.decline(*decline);
    } else if (_session.begin(std::move(std::get<complete_build_inventory>(certified)))) {
      _began.store(true, std::memory_order_release);
    }
  }
  if (!_began.load(std::memory_order_relaxed)) { _arrival_ledger.abandon(); }
  _decided = true;
  return true;
}

std::optional<std::uint64_t> build_input_certifier::contribute(
  pipelineable_operator_data const& input,
  cucascade::read_only_data_batch const& batch,
  ::cuda::stream_ref stream)
{
  if (!_began.load(std::memory_order_acquire)) { return std::nullopt; }
  auto const ids = input.original_batch_ids();
  if (!_site.port_directive().empty() || input.get_data_batches().size() != 1 || ids.size() != 1) {
    _session.decline_task_input();
    return std::nullopt;
  }
  _session.contribute(ids.front(), batch, stream);
  return ids.front();
}

}  // namespace sirius::op
