// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#include <sirius-c/object_store/config.h>
#include <sirius-c/object_store/config_builder.h>
#include <sirius-c/object_store/registry.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expr)                                                      \
  do {                                                                   \
    if (!(expr)) {                                                       \
      fprintf(stderr, "Failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      goto cleanup;                                                      \
    }                                                                    \
  } while (0)

#define DERIVE_TEXT(field, bytes)                                             \
  do {                                                                        \
    CHECK(sirius_object_store_config_builder_with_##field(                    \
            builder, bytes, strlen(bytes), &next, &error) == SIRIUS_SUCCESS); \
    CHECK(next != NULL && error == NULL);                                     \
    sirius_object_store_config_builder_release(builder);                      \
    builder = next;                                                           \
    next    = NULL;                                                           \
  } while (0)

int main(void)
{
  int result                                  = 1;
  sirius_object_store_config_builder* builder = NULL;
  sirius_object_store_config_builder* next    = NULL;
  sirius_object_store_config* config          = NULL;
  sirius_object_store_registry* empty         = NULL;
  sirius_object_store_registry* fallback      = NULL;
  sirius_object_store_registry* scoped        = NULL;
  sirius_error* error                         = NULL;

  CHECK(sirius_object_store_config_builder_create(&builder, &error) == SIRIUS_SUCCESS);
  CHECK(builder != NULL && error == NULL);
  CHECK(sirius_object_store_config_builder_build(builder, &config, &error) ==
        SIRIUS_INVALID_CONFIGURATION);
  CHECK(config == NULL && error != NULL);
  sirius_error_destroy(error);
  error = NULL;

  sirius_object_store_config_builder_retain(builder);
  sirius_object_store_config_builder_release(builder);
  DERIVE_TEXT(endpoint, "https://object-store.example.invalid:443");
  DERIVE_TEXT(region, "consumer-region");
  DERIVE_TEXT(access_key, "consumer-access");
  DERIVE_TEXT(secret_key, "consumer-secret");
  DERIVE_TEXT(session_token, "consumer-session");
  DERIVE_TEXT(ca_bundle_path, "/missing/consumer-ca.pem");
  CHECK(sirius_object_store_config_builder_with_transport(
          builder, SIRIUS_OBJECT_STORE_TRANSPORT_RDMA, &next, &error) == SIRIUS_SUCCESS);
  CHECK(next != NULL && error == NULL);
  sirius_object_store_config_builder_release(builder);
  builder = next;
  next    = NULL;
  CHECK(sirius_object_store_config_builder_with_signing_mode(
          builder, SIRIUS_OBJECT_STORE_SIGNING_HEADER, &next, &error) == SIRIUS_SUCCESS);
  CHECK(next != NULL && error == NULL);
  sirius_object_store_config_builder_release(builder);
  builder = next;
  next    = NULL;
  CHECK(sirius_object_store_config_builder_build(builder, &config, &error) == SIRIUS_SUCCESS);
  CHECK(config != NULL && error == NULL);
  sirius_object_store_config_builder_release(builder);
  builder = NULL;
  sirius_object_store_config_retain(config);
  sirius_object_store_config_release(config);
  CHECK(sirius_object_store_registry_create(&empty, &error) == SIRIUS_SUCCESS);
  CHECK(empty != NULL && error == NULL);
  CHECK(sirius_object_store_registry_with_store(empty, NULL, 0, config, &fallback, &error) ==
        SIRIUS_SUCCESS);
  CHECK(fallback != NULL && error == NULL);
  CHECK(
    sirius_object_store_registry_with_store(
      fallback, "s3://bucket/data/", sizeof("s3://bucket/data/") - 1, config, &scoped, &error) ==
    SIRIUS_SUCCESS);
  CHECK(scoped != NULL && error == NULL);
  sirius_object_store_registry_retain(scoped);
  sirius_object_store_registry_release(scoped);
  result = 0;

cleanup:
  sirius_object_store_registry_release(scoped);
  sirius_object_store_registry_release(fallback);
  sirius_object_store_registry_release(empty);
  sirius_error_destroy(error);
  sirius_object_store_config_builder_release(next);
  sirius_object_store_config_builder_release(builder);
  sirius_object_store_config_release(config);
  return result;
}
