include("${DUCKDB_MODULE_BASE_DIR}/.github/config/extensions/avro.cmake")
include("${DUCKDB_MODULE_BASE_DIR}/.github/config/extensions/iceberg.cmake")

# Pinned avro-c uses pre-C23 function declarations; GCC 15 defaults to C23.
cmake_language(
  DEFER
  DIRECTORY
  "${CMAKE_SOURCE_DIR}"
  CALL
  set_target_properties
  avro-static
  PROPERTIES
  C_STANDARD
  99
  C_STANDARD_REQUIRED
  ON)
