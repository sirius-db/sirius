PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# CPU-only: builds a stock DuckDB with quack and sirius_gateway, against the same
# DuckDB checkout as sirius-duckdb so the gateway and the ranks speak one version.
DISABLE_SANITIZER ?= 1
DUCKDB_SRCDIR := $(PROJ_DIR)../sirius-duckdb/duckdb/
SIRIUS_DUCKDB_EXPLICIT_VERSION := $(if $(filter d54cbbe3df17e8e5ce68312f483b4155836112aa,$(shell git -C '$(DUCKDB_SRCDIR)' rev-parse HEAD 2>/dev/null)),v2.0.0,)
override EXT_FLAGS += -DDUCKDB_EXPLICIT_VERSION=$(SIRIUS_DUCKDB_EXPLICIT_VERSION)

GEN ?= ninja

EXT_NAME=sirius_gateway
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

include $(PROJ_DIR)../sirius-duckdb/extension-ci-tools/makefiles/duckdb_extension.Makefile
