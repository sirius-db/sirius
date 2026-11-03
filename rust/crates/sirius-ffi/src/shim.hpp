// C++ adapters for the cxx bridge: named-field snapshot construction, the object-store factory
// overload, and borrowed std::string arguments forwarded to the drop and
// lookup members that take std::string_view. Everything else binds `include/sirius/ffi.hpp` as is.
#pragma once

#include "sirius/ffi.hpp"

#include <memory>
#include <string>

namespace sirius::ffi {

struct ObjectStoreConfigArgs;

std::shared_ptr<ObjectStoreConfig> make_object_store_config(const ObjectStoreConfigArgs& args);

std::unique_ptr<Context> make_context_from_config_with_object_store(
  const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg);

// cxx cannot pass std::string_view; the caller owns these strings, the adapters only borrow them.
void drop_object_store_prefix(const Context& ctx, const std::string& uri_prefix);
std::shared_ptr<ObjectStoreConfig> object_store_config_for_uri(const Context& ctx,
                                                               const std::string& uri);

}  // namespace sirius::ffi
