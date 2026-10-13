target_sources(
  sirius_objects PRIVATE numeric_narrowing.cpp timestamp_semantics.cpp
                         transaction_utils.cpp type_conversions.cpp)

# DuckDB's database-manager header requires its private fmt namespace. Keep it
# in a translation unit without RAPIDS/spdlog headers.
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/transaction_utils.cpp" TARGET_DIRECTORY
  sirius_objects
  PROPERTIES INCLUDE_DIRECTORIES
             "${SIRIUS_DUCKDB_SOURCE_DIR}/third_party/fmt/include")
