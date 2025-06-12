PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=sirius
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
# EXT_FLAGS=-DBUILD_PYTHON=1
GEN=ninja

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile