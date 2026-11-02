#include "sirius-sys/src/shim.hpp"

#include "sirius-sys/src/lib.rs.h"  // ObjectStoreConfigArgs (cxx-generated shared struct)

#include <utility>

namespace sirius::ffi {

std::shared_ptr<ObjectStoreConfig> make_object_store_config(const ObjectStoreConfigArgs& args)
{
  return std::make_shared<ObjectStoreConfig>(
    ObjectStoreConfig{.endpoint       = std::string(args.endpoint),
                      .region         = std::string(args.region),
                      .access_key     = std::string(args.access_key),
                      .secret_key     = std::string(args.secret_key),
                      .session_token  = std::string(args.session_token),
                      .transport      = args.transport,
                      .signing_mode   = args.signing_mode,
                      .ca_bundle_path = std::string(args.ca_bundle_path),
                      .uri_prefix     = std::string(args.uri_prefix)});
}

std::unique_ptr<Context> make_context_from_config_with_object_store(
  const std::string& config_path, std::shared_ptr<ObjectStoreConfig> cfg)
{
  return make_context_from_config(config_path, std::move(cfg));
}

void drop_object_store_prefix(const Context& ctx, const std::string& uri_prefix)
{
  ctx.drop_object_store(uri_prefix);
}

std::shared_ptr<ObjectStoreConfig> object_store_config_for_uri(const Context& ctx,
                                                               const std::string& uri)
{
  return ctx.object_store_config_for(uri);
}

}  // namespace sirius::ffi
