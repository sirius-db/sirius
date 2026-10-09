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

#include "sirius_config.hpp"

#include <sirius/context/config_builder.hpp>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <utility>

namespace sirius {

struct ContextConfig::Impl {
  parsed_sirius_config config;
};

struct ContextConfigBuilder::Impl {
  parsed_sirius_config config;
};

ContextConfig::ContextConfig(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
ContextConfig::ContextConfig(const ContextConfig&) noexcept            = default;
ContextConfig& ContextConfig::operator=(const ContextConfig&) noexcept = default;
ContextConfig::~ContextConfig() noexcept                               = default;

ContextConfigBuilder::ContextConfigBuilder() : impl_(std::make_shared<Impl>()) {}
ContextConfigBuilder::ContextConfigBuilder(std::shared_ptr<const Impl> impl)
  : impl_(std::move(impl))
{
}
ContextConfigBuilder::ContextConfigBuilder(const ContextConfigBuilder&) noexcept = default;
ContextConfigBuilder& ContextConfigBuilder::operator=(const ContextConfigBuilder&) noexcept =
  default;
ContextConfigBuilder::~ContextConfigBuilder() noexcept = default;

std::expected<ContextConfigBuilder, Error> ContextConfigBuilder::from_yaml(
  const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return std::unexpected(
      Error{ErrorCode::configuration_io, "Cannot open configuration file: " + path.string()});
  }
  std::string contents;
  char buffer[8192];
  while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
    contents.append(buffer, static_cast<std::size_t>(file.gcount()));
  }
  if (!file.eof()) {
    return std::unexpected(
      Error{ErrorCode::configuration_io, "Cannot read configuration file: " + path.string()});
  }

  auto impl = std::make_shared<Impl>();
  try {
    auto root    = YAML::Load(contents);
    impl->config = parsed_sirius_config::from_node(root, path);
  } catch (const YAML::Exception& e) {
    return std::unexpected(Error{ErrorCode::malformed_yaml, path.string() + ": " + e.what()});
  } catch (const configuration_input_error& e) {
    return std::unexpected(Error{ErrorCode::invalid_configuration, e.what()});
  }
  return ContextConfigBuilder(std::move(impl));
}

std::expected<ContextConfig, Error> ContextConfigBuilder::build() const
{
  return ContextConfig(std::make_shared<ContextConfig::Impl>(ContextConfig::Impl{impl_->config}));
}

}  // namespace sirius
