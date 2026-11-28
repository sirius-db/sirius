// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include "config/user_config.hpp"

#include "config/yaml.hpp"

#include <string>
#include <string_view>

namespace sirius::config {

void user_config::validate(std::string_view path) const
{
  if (memory.configured() && space.configured()) {
    throw config_error(detail::framed(
      path, "sirius.memory and non-empty sirius.space lists are mutually exclusive"));
  }
}

node user_config::write(const write_options& options) const
{
  auto out = block_base<user_config>::write(options);
  if (space.configured() && !memory.configured()) {
    // Unused memory tiers would read back as configured; keep only the block the user wrote.
    if (memory.user_defined()) {
      out.set("memory", node::mapping());
    } else {
      out.erase("memory");
    }
  }
  return out;
}

user_config read_user_config(std::string_view yaml)
{
  user_config config;
  read_yaml(config, yaml, user_config_root);
  return config;
}

user_config read_user_config(const node& document)
{
  user_config config;
  read_yaml(config, document, user_config_root);
  return config;
}

user_config default_user_config()
{
  user_config config;
  config.read(node{}, user_config_root);
  return config;
}

std::string write_user_config(const user_config& config, const write_options& options)
{
  return write_yaml(config, user_config_root, options);
}

bool developer_mode() noexcept
{
#ifdef SIRIUS_DEVELOPER_MODE
  return true;
#else
  return false;
#endif
}

}  // namespace sirius::config
