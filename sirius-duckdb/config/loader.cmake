if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT MUSL_ENABLED)
  duckdb_extension_load(sirius SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../loader
                        INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../loader/src)
endif()
