// C++ side of the `sirius-sys` bridge for the object-store API: the two entry points that cxx
// cannot bind directly — aggregate construction of `ObjectStoreConfig` from named fields, and
// the `make_context_from_config` overload (cxx has no overloads). Everything else in the bridge
// binds `include/sirius/ffi.hpp` as is.
#pragma once

#include "sirius/ffi.hpp"

#include <memory>
#include <string>

namespace sirius::ffi {

struct ObjectStoreConfigArgs;

std::shared_ptr<ObjectStoreConfig> make_object_store_config(const ObjectStoreConfigArgs& args);

std::unique_ptr<Context> make_context_from_config_with_object_store(
  const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg);

}  // namespace sirius::ffi
