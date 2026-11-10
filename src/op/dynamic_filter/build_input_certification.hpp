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

/**
 * @file build_input_certification.hpp
 * @brief The decision `build_input_certifier` makes for a dynamic-filter construction site, the
 * build PARTITION (`hash_join_partition_observer`) or `sirius_physical_build_filter`, before it
 * certifies the site's complete build input and begins a `dynamic_filter_publication_session`
 * attempt.
 *
 * `read_build_port_facts` captures an operator's input ports as a plain `build_port_facts` value,
 * and the pure `certify_decision` maps it to pending, a `construction_decline`, or proceed. On
 * proceed `build_input_certifier` certifies its `build_arrival_ledger`, which yields the
 * `complete_build_inventory` or one of the ledger's own declines.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <variant>

namespace sirius::op {

enum class MemoryBarrierType;
class sirius_physical_operator;

/**
 * @brief Why a build input cannot construct dynamic filters
 *
 * `dynamic_filter_publication_session` counts each reason (`input_shape` and `inventory_mismatch`
 * in `accumulations_skipped_inventory`, the others in `accumulations_declined_<reason>`) and names
 * it in the declined keys' `[dynamic_filter_key]` lines.
 */
enum class construction_decline : std::uint8_t {
  input_shape,         ///< Not one single-partition `default` FULL data port with a source
  pipeline_barrier,    ///< A PIPELINE barrier (an ORDER BY producer) never holds the complete input
  port_directive,      ///< A late-materialization directive defers input columns
  inventory_mismatch,  ///< The ledger disagrees with the repository or cannot form an inventory
  unreadable_arrival,  ///< An arrival whose rows or types the ledger cannot accept
  compressed_arrival   ///< A compressed arrival, which exposes only its row count
};

[[nodiscard]] constexpr std::string_view to_string(construction_decline reason) noexcept
{
  switch (reason) {
    case construction_decline::input_shape: return "input_shape";
    case construction_decline::pipeline_barrier: return "pipeline_barrier";
    case construction_decline::port_directive: return "port_directive";
    case construction_decline::inventory_mismatch: return "inventory_mismatch";
    case construction_decline::unreadable_arrival: return "unreadable_arrival";
    case construction_decline::compressed_arrival: return "compressed_arrival";
  }
  return "invalid";
}

/**
 * @brief What `certify_decision` reads from the input ports of one build-filter construction site
 */
struct build_port_facts {
  std::size_t data_ports;             ///< Ports with a repository
  bool single_port_is_default;        ///< Whether the one data port is `default`
  std::size_t repository_partitions;  ///< Partitions of that port's repository
  MemoryBarrierType barrier;          ///< That port's barrier
  bool has_source_pipeline;           ///< Whether that port has a source pipeline
  bool source_finished;               ///< Whether that source pipeline has finished
  std::size_t repository_batches;  ///< Batches queued in partition 0, read after `source_finished`
  bool has_port_directive;         ///< Whether the site has a late-materialization directive
};

/// The source has not finished; decide again later.
struct certification_pending {
  [[nodiscard]] bool operator==(certification_pending const&) const = default;
};

/// Certify the ledger against `build_port_facts::repository_batches`.
struct certification_proceed {
  [[nodiscard]] bool operator==(certification_proceed const&) const = default;
};

using certification_decision =
  std::variant<certification_pending, construction_decline, certification_proceed>;

/**
 * @brief Decides whether a construction site's input can be certified
 *
 * Declines are checked before completion, so an input that can never certify declines at once and
 * never waits for its source: a wrong shape (including a PARTIAL barrier or no source pipeline),
 * then a PIPELINE barrier, then a port directive. Only a FULL input of the right shape pends until
 * its source finishes.
 */
[[nodiscard]] certification_decision certify_decision(build_port_facts const& facts) noexcept;

/**
 * @brief Reads @p site's input ports as `build_port_facts`
 *
 * Reads whether the source finished before the repository's batch count, so a count read from a
 * finished source is final.
 */
[[nodiscard]] build_port_facts read_build_port_facts(sirius_physical_operator const& site);

}  // namespace sirius::op
