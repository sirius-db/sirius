// Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
#define _POSIX_C_SOURCE 200809L
#include <sirius/c/context/config_builder.h>
#include <sirius/c/version.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(expr)                                                      \
  do {                                                                   \
    if (!(expr)) {                                                       \
      fprintf(stderr, "Failed: %s:%d: %s\n", __FILE__, __LINE__, #expr); \
      return 1;                                                          \
    }                                                                    \
  } while (0)

int main(void)
{
  sirius_context_config_builder* builder = NULL;
  sirius_context_config* config          = NULL;
  sirius_error* error                    = NULL;
  CHECK(sirius_abi_version() == SIRIUS_ABI_VERSION);
  CHECK(sirius_context_config_builder_create(&builder, &error) == SIRIUS_SUCCESS);
  CHECK(builder != NULL && error == NULL);
  sirius_context_config_builder_retain(builder);
  sirius_context_config_builder_release(builder);
  CHECK(sirius_context_config_builder_build(builder, &config, &error) == SIRIUS_SUCCESS);
  sirius_context_config_builder_release(builder);
  sirius_context_config_retain(config);
  sirius_context_config_release(config);
  sirius_context_config_release(config);
  CHECK(sirius_context_config_builder_create(NULL, &error) == SIRIUS_INVALID_ARGUMENT);
  CHECK(sirius_error_message_size(error) == strlen(sirius_error_message(error)));
  sirius_error_destroy(error);
  CHECK(sirius_context_config_builder_from_yaml("a\0b", 3, &builder, &error) ==
        SIRIUS_INVALID_ARGUMENT);
  CHECK(builder == NULL);
  sirius_error_destroy(error);
  CHECK(sirius_context_config_builder_build(NULL, &config, NULL) == SIRIUS_INVALID_ARGUMENT);
  CHECK(config == NULL);
  CHECK(sirius_context_config_builder_from_yaml("", 0, &builder, &error) ==
        SIRIUS_CONFIGURATION_IO);
  sirius_error_destroy(error);
  sirius_context_config_builder_release(NULL);
  sirius_context_config_release(NULL);
  sirius_error_destroy(NULL);
  CHECK(sirius_error_message_size(NULL) == 0);
  CHECK(strcmp(sirius_error_message(NULL), "") == 0);
  char path[] = "/tmp/sirius-c-\xff-XXXXXX";
  int fd      = mkstemp(path);
  CHECK(fd >= 0);
  CHECK(close(fd) == 0);
  FILE* file = fopen(path, "w");
  CHECK(file != NULL);
  CHECK(fputs("sirius: {}\n", file) >= 0);
  CHECK(fclose(file) == 0);
  CHECK(sirius_context_config_builder_from_yaml(path, strlen(path), &builder, &error) ==
        SIRIUS_SUCCESS);
  CHECK(unlink(path) == 0);
  CHECK(sirius_context_config_builder_build(builder, &config, &error) == SIRIUS_SUCCESS);
  sirius_context_config_builder_release(builder);
  sirius_context_config_release(config);
  file = fopen(path, "w");
  CHECK(file != NULL);
  CHECK(fputs("sirius: [\n", file) >= 0);
  CHECK(fclose(file) == 0);
  CHECK(sirius_context_config_builder_from_yaml(path, strlen(path), &builder, &error) ==
        SIRIUS_MALFORMED_YAML);
  CHECK(builder == NULL);
  sirius_error_destroy(error);
  file = fopen(path, "w");
  CHECK(file != NULL);
  CHECK(fputs("sirius: {unknown_setting: true}\n", file) >= 0);
  CHECK(fclose(file) == 0);
  CHECK(sirius_context_config_builder_from_yaml(path, strlen(path), &builder, &error) ==
        SIRIUS_INVALID_CONFIGURATION);
  sirius_error_destroy(error);
  CHECK(unlink(path) == 0);
  return 0;
}
