duckdb_extension_load(
  sirius
  SOURCE_DIR
  ${CMAKE_CURRENT_LIST_DIR}
  INCLUDE_DIR
  ${CMAKE_CURRENT_LIST_DIR}/src/include
  TEST_DIR
  ${CMAKE_CURRENT_LIST_DIR}/test/sql
  LOAD_TESTS)

# Ranks behind the distributed gateway serve their session over Quack.
include(${CMAKE_CURRENT_LIST_DIR}/duckdb/.github/config/extensions/quack.cmake)
duckdb_extension_statically_link(quack)
