add_executable(paimon_delete_writer
               "${CMAKE_CURRENT_LIST_DIR}/delete_writer.cpp")
target_compile_features(paimon_delete_writer PRIVATE cxx_std_17)
target_include_directories(paimon_delete_writer
                           PRIVATE "${PROJECT_SOURCE_DIR}/include")
# Factories are registered by static initializers; retain their complete
# archives.
target_link_libraries(
  paimon_delete_writer
  PRIVATE -Wl,--start-group
          -Wl,--whole-archive
          paimon_local_file_system_static
          paimon_parquet_file_format_static
          paimon_avro_file_format_static
          paimon_file_index_static
          paimon_global_index_static
          -Wl,--no-whole-archive
          paimon_static
          arrow
          -Wl,--end-group)
