/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#pragma once

#include <cucascade/io/object_store_config.hpp>

#include <duckdb/main/client_context.hpp>

#include <string_view>

namespace duckdb {
class SecretManager;
}

namespace sirius::io::s3 {

using cucascade::io::object_store_config;

/// True when @p path begins with the S3 scheme (case-insensitive).
bool is_s3_path(std::string_view path) noexcept;

/// Register Sirius's own `SIRIUS_S3` config secret type. This does not require httpfs.
void register_sirius_s3_secret(duckdb::SecretManager& manager);

/// Resolve a matching `SIRIUS_S3` secret first, then an httpfs `S3` secret if
/// available. Without either match, return @p defaults. A matching secret is a
/// complete credential source and does not inherit credential or endpoint fields
/// from defaults. Invalid matching secrets are errors, not fallback triggers.
object_store_config resolve_duckdb_s3_secret(duckdb::ClientContext& context,
                                             std::string_view path,
                                             object_store_config defaults);

}  // namespace sirius::io::s3
