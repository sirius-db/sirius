/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include "io/s3/duckdb_secret_config.hpp"

#include <duckdb/catalog/catalog_transaction.hpp>
#include <duckdb/common/exception.hpp>
#include <duckdb/common/string_util.hpp>
#include <duckdb/main/client_context.hpp>
#include <duckdb/main/database.hpp>
#include <duckdb/main/secret/secret.hpp>
#include <duckdb/main/secret/secret_manager.hpp>

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace sirius::io::s3 {
namespace {

duckdb::unique_ptr<duckdb::BaseSecret> create_sirius_s3_secret(duckdb::ClientContext&,
                                                               duckdb::CreateSecretInput& input)
{
  auto secret =
    duckdb::make_uniq<duckdb::KeyValueSecret>(input.scope, input.type, input.provider, input.name);
  for (auto const& [key, value] : input.options) {
    secret->secret_map[key] = value;
  }
  secret->redact_keys.insert("key_id");
  secret->redact_keys.insert("secret");
  secret->redact_keys.insert("session_token");
  return duckdb::unique_ptr_cast<duckdb::KeyValueSecret, duckdb::BaseSecret>(std::move(secret));
}

bool get_string(duckdb::KeyValueSecret const& secret, char const* name, std::string& output)
{
  duckdb::Value value;
  if (!secret.TryGetValue(name, value) || value.IsNull()) { return false; }
  try {
    output = value.GetValue<std::string>();
  } catch (...) {
    throw duckdb::InvalidInputException("Sirius S3 secret option has an invalid value");
  }
  return true;
}

bool get_bool(duckdb::KeyValueSecret const& secret, char const* name, bool& output)
{
  duckdb::Value value;
  if (!secret.TryGetValue(name, value) || value.IsNull()) { return false; }
  try {
    output = value.GetValue<bool>();
  } catch (...) {
    throw duckdb::InvalidInputException("Sirius S3 secret option has an invalid value");
  }
  return true;
}

void reject_present_option(duckdb::KeyValueSecret const& secret, char const* name)
{
  duckdb::Value value;
  if (secret.TryGetValue(name, value) && !value.IsNull()) {
    throw duckdb::NotImplementedException(std::string("Sirius S3 secrets do not support option '") +
                                          name + "'");
  }
}

void reject_nonempty_string_option(duckdb::KeyValueSecret const& secret, char const* name)
{
  std::string value;
  if (get_string(secret, name, value) && !value.empty()) { reject_present_option(secret, name); }
}

void reject_true_bool_option(duckdb::KeyValueSecret const& secret, char const* name)
{
  bool value = false;
  if (get_bool(secret, name, value) && value) { reject_present_option(secret, name); }
}

std::string canonicalize_s3_scheme(std::string_view path)
{
  auto canonical = std::string{path};
  canonical.replace(0, std::string_view{"s3://"}.size(), "s3://");
  return canonical;
}

duckdb::CatalogTransaction secret_catalog_transaction(duckdb::ClientContext& context)
{
  if (context.transaction.HasActiveTransaction()) {
    return duckdb::CatalogTransaction::GetSystemCatalogTransaction(context);
  }
  return duckdb::CatalogTransaction::GetSystemTransaction(
    duckdb::DatabaseInstance::GetDatabase(context));
}

}  // namespace

void register_sirius_s3_secret(duckdb::SecretManager& manager)
{
  duckdb::SecretType type;
  type.name             = "sirius_s3";
  type.deserializer     = duckdb::KeyValueSecret::Deserialize<duckdb::KeyValueSecret>;
  type.default_provider = "config";
  type.extension        = "sirius";
  manager.RegisterSecretType(type);

  duckdb::CreateSecretFunction function;
  function.secret_type                       = "sirius_s3";
  function.provider                          = "config";
  function.function                          = create_sirius_s3_secret;
  function.named_parameters["key_id"]        = duckdb::LogicalType::VARCHAR;
  function.named_parameters["secret"]        = duckdb::LogicalType::VARCHAR;
  function.named_parameters["region"]        = duckdb::LogicalType::VARCHAR;
  function.named_parameters["session_token"] = duckdb::LogicalType::VARCHAR;
  function.named_parameters["endpoint"]      = duckdb::LogicalType::VARCHAR;
  function.named_parameters["url_style"]     = duckdb::LogicalType::VARCHAR;
  function.named_parameters["use_ssl"]       = duckdb::LogicalType::BOOLEAN;
  function.named_parameters["verify_ssl"]    = duckdb::LogicalType::BOOLEAN;
  manager.RegisterSecretFunction(std::move(function), duckdb::OnCreateConflict::ERROR_ON_CONFLICT);
}

bool is_s3_path(std::string_view path) noexcept
{
  constexpr std::string_view scheme = "s3://";
  if (path.size() < scheme.size()) { return false; }
  for (std::size_t i = 0; i < scheme.size(); ++i) {
    if (static_cast<char>(std::tolower(static_cast<unsigned char>(path[i]))) != scheme[i]) {
      return false;
    }
  }
  return true;
}

object_store_config resolve_duckdb_s3_secret(duckdb::ClientContext& context,
                                             std::string_view path,
                                             object_store_config defaults)
{
  if (!is_s3_path(path)) { return defaults; }
  // Look up the scoped secret directly so an absent secret can be distinguished
  // from a selected secret with missing fields. Credentials are resolved for
  // every bind/open, so replacement affects future work. Sirius keeps the
  // resolved config snapshot needed by its REST signer, but does not retain the
  // DuckDB secret object or its catalog name in bind data.
  // File-system callbacks can also run outside query execution (for example a
  // direct FileSystem::OpenFile in an embedding). In that case there is no
  // ClientContext transaction to borrow, so use DuckDB's committed system view.
  auto transaction = secret_catalog_transaction(context);
  auto& manager    = duckdb::SecretManager::Get(context);
  // URI schemes are case-insensitive, but DuckDB secret scopes use case-sensitive prefix
  // matching. Canonicalize only the scheme so SCOPE 's3://bucket/' also covers
  // 'S3://bucket/object' without changing case-sensitive bucket or object-key bytes.
  auto const secret_path = canonicalize_s3_scheme(path);
  auto match             = manager.LookupSecret(transaction, secret_path, "sirius_s3");
  if (!match.HasMatch()) { match = manager.LookupSecret(transaction, secret_path, "s3"); }
  if (!match.HasMatch()) { return defaults; }
  if (!duckdb::StringUtil::CIEquals(match.GetSecret().GetProvider(), "config")) {
    throw duckdb::NotImplementedException(
      "Sirius S3 currently supports only static TYPE SIRIUS_S3 or TYPE S3 PROVIDER CONFIG secrets");
  }

  auto const* secret_ptr = dynamic_cast<duckdb::KeyValueSecret const*>(&match.GetSecret());
  if (secret_ptr == nullptr) {
    throw duckdb::InvalidInputException("The selected Sirius S3 secret has an unsupported format");
  }
  auto const& secret = *secret_ptr;
  reject_present_option(secret, "refresh_info");
  reject_nonempty_string_option(secret, "refresh");
  reject_true_bool_option(secret, "requester_pays");
  reject_true_bool_option(secret, "url_compatibility_mode");
  reject_nonempty_string_option(secret, "http_proxy");
  reject_nonempty_string_option(secret, "http_proxy_username");
  reject_nonempty_string_option(secret, "http_proxy_password");
  reject_present_option(secret, "extra_http_headers");
  reject_nonempty_string_option(secret, "sse_c_key");
  reject_nonempty_string_option(secret, "kms_key_id");

  std::string access_key;
  std::string secret_key;
  bool const has_secret_key_id = get_string(secret, "key_id", access_key);
  bool const has_secret_key    = get_string(secret, "secret", secret_key);
  if (!has_secret_key_id || !has_secret_key || access_key.empty() || secret_key.empty()) {
    throw duckdb::InvalidInputException(
      "The selected Sirius S3 secret must provide non-empty key_id and secret credentials");
  }

  // Once a secret is selected, do not combine its fields with fallback credentials
  // or endpoint settings. The selected secret is a self-contained credential
  // source; non-authentication transport tuning remains from the default config.
  defaults.access_key = std::move(access_key);
  defaults.secret_key = std::move(secret_key);
  defaults.region.clear();
  defaults.session_token.clear();
  defaults.endpoint.clear();
  get_string(secret, "region", defaults.region);
  get_string(secret, "session_token", defaults.session_token);
  bool use_ssl = true;
  get_bool(secret, "use_ssl", use_ssl);
  get_bool(secret, "verify_ssl", defaults.tls_verify);
  get_string(secret, "endpoint", defaults.endpoint);

  std::string url_style;
  if (get_string(secret, "url_style", url_style)) {
    std::transform(url_style.begin(), url_style.end(), url_style.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
  }
  if (!url_style.empty() && url_style != "path") {
    throw duckdb::NotImplementedException(
      "Sirius S3 secrets currently support only url_style='path'");
  }

  // The Sirius REST backend currently signs path-style requests. Its endpoint
  // parser accepts an explicit scheme; DuckDB's httpfs secret stores SSL as a
  // separate boolean, so preserve an explicit endpoint scheme and otherwise
  // synthesize the matching scheme.
  if (!defaults.endpoint.empty() && defaults.endpoint.find("://") == std::string::npos) {
    defaults.endpoint = std::string(use_ssl ? "https://" : "http://") + defaults.endpoint;
  }
  // httpfs treats a config secret without ENDPOINT as the AWS service. Mirror
  // its region default and endpoint selection without inheriting YAML values.
  if (defaults.region.empty()) { defaults.region = "us-east-1"; }
  if (defaults.endpoint.empty()) {
    auto const suffix = defaults.region.rfind("cn-", 0) == 0 ? "amazonaws.com.cn" : "amazonaws.com";
    defaults.endpoint =
      std::string(use_ssl ? "https://" : "http://") + "s3." + defaults.region + "." + suffix;
  }
  return defaults;
}

}  // namespace sirius::io::s3
