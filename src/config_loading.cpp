// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "config_loading.hpp"

#include <yaml-cpp/yaml.h>

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
        SIRIUS_CONFIGURATION_IO, "Cannot open configuration file", path);
    }
    std::string contents;
    char buffer[8192];
    while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
      contents.append(buffer, static_cast<std::size_t>(file.gcount()));
    }
    if (!file.eof()) {
      throw configuration_load_error(
        SIRIUS_CONFIGURATION_IO, "Cannot read configuration file", path);
    }
    return parsed_sirius_config::from_node(YAML::Load(contents), path);
  } catch (const std::ios_base::failure& e) {
    throw configuration_load_error(SIRIUS_CONFIGURATION_IO, e.what(), path);
  } catch (const YAML::Exception& e) {
    throw configuration_load_error(SIRIUS_MALFORMED_YAML, e.what(), path);
  } catch (const configuration_input_error& e) {
    throw configuration_load_error(SIRIUS_INVALID_CONFIGURATION, e.what());
  }
}
}  // namespace sirius
