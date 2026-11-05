target_sources(sirius_objects PRIVATE duckdb_sink.cpp level.cpp logging.cpp
                                      noop_sink.cpp)

# Keep spdlog's external fmt headers separate from DuckDB's bundled fmt headers.
add_library(sirius_spdlog_sink OBJECT spdlog_owning_sink.cpp)
set_target_properties(
  sirius_spdlog_sink
  PROPERTIES POSITION_INDEPENDENT_CODE ON
             CXX_VISIBILITY_PRESET hidden
             CXX_SCAN_FOR_MODULES OFF)
target_compile_features(sirius_spdlog_sink PRIVATE cxx_std_20)
target_include_directories(sirius_spdlog_sink PRIVATE "${CMAKE_SOURCE_DIR}/src")
target_link_libraries(sirius_spdlog_sink PRIVATE spdlog::spdlog)
