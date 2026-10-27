#include "sirius-sys/src/shim.hpp"

#include "sirius-sys/src/lib.rs.h"  // ObjectStoreConfigArgs (cxx-generated shared struct)

#include <utility>

namespace sirius::ffi {

namespace {

// Empty `transport` / `signing_mode` take the C++ defaults, so Rust callers need not spell them.
std::string or_default(const rust::String& value, const char* fallback)
{
  return value.empty() ? std::string(fallback) : std::string(value);
}

}  // namespace

std::shared_ptr<ObjectStoreConfig> make_object_store_config(const ObjectStoreConfigArgs& args)
{
  return std::make_shared<ObjectStoreConfig>(
    ObjectStoreConfig{.endpoint       = std::string(args.endpoint),
                      .region         = std::string(args.region),
                      .access_key     = std::string(args.access_key),
                      .secret_key     = std::string(args.secret_key),
                      .session_token  = std::string(args.session_token),
                      .transport      = or_default(args.transport, "auto"),
                      .signing_mode   = or_default(args.signing_mode, "presigned"),
                      .ca_bundle_path = std::string(args.ca_bundle_path),
                      .uri_prefix     = std::string(args.uri_prefix)});
}

std::unique_ptr<Context> make_context_from_config_with_object_store(
  const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg)
{
  return make_context_from_config(config_path, std::move(cfg));
}

}  // namespace sirius::ffi
