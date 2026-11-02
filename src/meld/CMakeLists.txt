if(SIRIUS_ENABLE_MELD_UNITS)
  target_sources(sirius_objects
                 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/meld_unit.cpp)
  # Only the bridge sees Meld's and MLIR's headers.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/meld_unit.cpp TARGET_DIRECTORY sirius_objects
    PROPERTIES INCLUDE_DIRECTORIES "${SIRIUS_MELD_INCLUDE_DIRS}")
endif()
