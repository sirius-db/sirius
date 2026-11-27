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
 * - Scalars keep their text, quoted or not; tags are ignored (`!!str 1` is the text "1").
 * - Plain `~`, `null`, `Null`, `NULL` and an empty value are null; quoted ones stay text.
 * - Aliases are resolved into copies, up to 100000 values in all; merge keys (`<<: *a`) are not
 *   merged, so `<<` is an ordinary (and unknown) key.
 * - Exactly one document; trailing empty documents are ignored. An empty text is null.
 *
 * @throws yaml_syntax_error for malformed YAML, a second document, nesting deeper than 64 levels
 * (also an alias inside the value it names), or too many values.
 * @throws config_error for a duplicate key or a non-scalar key in a mapping (with its position).
 */
[[nodiscard]] node parse_yaml(std::string_view text);

/// Emit @p n as a block-style YAML document that parse_yaml reads back to an equal tree. Text
/// must be valid UTF-8: the emitter replaces invalid sequences in quoted text.
[[nodiscard]] std::string emit_yaml(const node& n);

/**
 * @brief Read the YAML document @p text into @p block: the document must be a mapping holding
 * only @p root_key (e.g. "sirius"), whose value is the block's mapping. A missing or null root
 * value is an error (write `sirius: {}` for an empty configuration).
 *
 * @throws yaml_syntax_error / config_error as parse_yaml and config_block::read; @p block is
 * unchanged on failure.
 */
void read_yaml(config_block& block, std::string_view text, std::string_view root_key);

/// read_yaml for a document already parsed into a node tree.
void read_yaml(config_block& block, const node& document, std::string_view root_key);

/// The YAML document read_yaml reads back to @p block's values: `<root_key>:` holding the block's
/// user-defined values, plus what @p options adds.
[[nodiscard]] std::string write_yaml(const config_block& block,
                                     std::string_view root_key,
                                     const write_options& options = {});

}  // namespace sirius::config
