// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/node.hpp"

#include <yaml-cpp/yaml.h>

namespace sirius::config {

/// A document yaml-cpp has already loaded, as a node tree, with parse_yaml's rules (bounded
/// aliases, duplicate and non-scalar keys rejected).
/// @throws yaml_syntax_error / config_error as parse_yaml.
[[nodiscard]] node from_yaml_cpp(const YAML::Node& document);

}  // namespace sirius::config
