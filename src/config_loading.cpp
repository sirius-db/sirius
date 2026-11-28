// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "config_loading.hpp"

#include "config/yaml.hpp"

#include <fstream>

namespace sirius {
parsed_sirius_config load_configuration(const std::filesystem::path& path)
{
  try {
    std::ifstream file;
    file.exceptions(std::ios::badbit);
    file.open(path, std::ios::binary);
    if (!file) {
      throw configuration_load_error(
        configuration_load_error_code::io, "Cannot open configuration file", path);
    }
    std::string contents;
    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
      contents.append(buffer, static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof()) {
      throw configuration_load_error(
        configuration_load_error_code::io, "Cannot read configuration file", path);
    }
    config::node document;
    try {
      document = config::parse_yaml(contents);
    } catch (const config::yaml_syntax_error& e) {
      throw configuration_load_error(configuration_load_error_code::malformed_yaml, e.what(), path);
    }
    return parsed_sirius_config::from_document(document, path);
  } catch (const std::ios_base::failure&) {
    throw configuration_load_error(
      configuration_load_error_code::io, "Cannot read configuration file", path);
  } catch (const config::config_error& e) {
    // Well-formed YAML with a duplicate or non-scalar key.
    throw configuration_load_error(
      configuration_load_error_code::invalid_configuration, e.what(), path);
  } catch (const configuration_input_error& e) {
    throw configuration_load_error(configuration_load_error_code::invalid_configuration, e.what());
  }
}
}  // namespace sirius
