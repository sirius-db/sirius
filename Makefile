PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=komodo
EXT_CONFIG=${PROJ_DIR}extension_config.cmake
GEN=ninja
BUILD_PYTHON=1

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile