// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/block.hpp"
#include "config/node.hpp"
#include "config/option.hpp"

#include <string>
#include <string_view>

namespace sirius::config {

/// The text is not well-formed YAML (as opposed to well-formed YAML with bad settings, which is a
/// plain config_error).
class yaml_syntax_error : public config_error {
 public:
  using config_error::config_error;
};

/**
 * @brief Parse a YAML document into a node tree.
 *
 * Scalars keep their text (quoted or not); `~`, `null` and an empty value become null nodes;
 * anchors and aliases are resolved. An empty document is null.
 *
 * @throws yaml_syntax_error for malformed YAML.
 * @throws config_error for a duplicate key or a non-scalar key in a mapping.
 */
[[nodiscard]] node parse_yaml(std::string_view text);

/// Emit @p n as a block-style YAML document that parse_yaml reads back to an equal tree.
[[nodiscard]] std::string emit_yaml(const node& n);

/**
 * @brief Read the YAML document @p text into @p block: the document must be a mapping holding
 * exactly @p root_key (e.g. "sirius"), whose value is the block's mapping.
 *
 * @throws yaml_syntax_error / config_error as parse_yaml and config_block::read.
 */
void read_yaml(config_block& block,
               std::string_view text,
               std::string_view root_key,
               environment env);

/// The YAML document read_yaml reads back to @p block's values: `<root_key>:` holding the block's
/// file values, plus its fallbacks when @p include_fallbacks.
[[nodiscard]] std::string write_yaml(const config_block& block,
                                     std::string_view root_key,
                                     bool include_fallbacks = false);

}  // namespace sirius::config
