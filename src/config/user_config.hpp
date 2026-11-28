// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/node.hpp"
#include "config/option.hpp"
#include "config/options/compression.hpp"
#include "config/options/executor.hpp"
#include "config/options/memory.hpp"
#include "config/options/operator.hpp"
#include "config/options/output.hpp"
#include "config/options/space.hpp"
#include "config/options/telemetry.hpp"
#include "config/options/topology.hpp"
#include "config/validators.hpp"

#include <string>
#include <string_view>

namespace sirius::config {

/// The document key the configuration lives under.
inline constexpr std::string_view user_config_root = "sirius";

/**
 * @brief Everything a user can configure: the `sirius:` document, as typed option blocks.
 *
 * This is the input. Reading a file fills it (read_user_config), writing dumps it
 * (write_user_config), and resolving it against the machine's topology produces the runtime
 * configuration the engine runs with.
 */
struct user_config : block_base<user_config> {
  topology_options topology;
  memory_options memory;
  config_option<int> max_concurrent_queries = option<int>("max_concurrent_queries")
                                                .validate(positive<int>{})
                                                .fallback(1)
                                                .describe("queries that may execute at once")
                                                .build();
  executor_options executor;
  telemetry_options telemetry;
  compression_options compression;
  space_options space;
  operator_options operator_params;
  output_options output;
  /// The launch scripts read sirius.distributed (scripts/sirius_distributed_config.py); Sirius
  /// keeps it as written.
  config_option<node> distributed = option<node>("distributed")
                                      .parser([](const node& n, node& out) -> parse_status {
                                        out = n;
                                        return {};
                                      })
                                      .writer([](const node& n) { return n; })
                                      .mark_production()
                                      .describe("settings of the launch scripts")
                                      .build();

  auto fields()
  {
    return field_list(sub("topology", topology),
                      sub("memory", memory),
                      max_concurrent_queries,
                      sub("executor", executor),
                      sub("telemetry", telemetry),
                      sub("compression", compression),
                      sub("space", space),
                      sub("operator_params", operator_params),
                      sub("output", output),
                      distributed);
  }

  /// sirius.memory and non-empty sirius.space lists are two ways to describe the same spaces.
  void validate(std::string_view path) const;

  /// As block_base::write, except that a memory block holding only fallbacks is left out when
  /// space lists describe the memory, so the dump reads back.
  [[nodiscard]] node write(const write_options& options) const override;
};

/// Read the YAML text of a configuration file. @throws yaml_syntax_error / config_error.
[[nodiscard]] user_config read_user_config(std::string_view yaml);

/// Read a configuration document already parsed into a node tree.
[[nodiscard]] user_config read_user_config(const node& document);

/// The configuration of an empty file: every option at its fallback.
[[nodiscard]] user_config default_user_config();

/// The YAML text read_user_config reads back to @p config: the user's values, plus what @p options
/// adds (every fallback, developer-only settings).
[[nodiscard]] std::string write_user_config(const user_config& config,
                                            const write_options& options = {});

/// Whether this build runs in developer mode (CMake SIRIUS_DEVELOPER_MODE): configuration dumps
/// then include developer-only settings.
[[nodiscard]] bool developer_mode() noexcept;

}  // namespace sirius::config
