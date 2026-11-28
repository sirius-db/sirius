# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 This
# builds the native read boundary, not an authenticated/admitted SQL source.
find_package(Paimon CONFIG REQUIRED)
if(NOT TARGET Paimon::paimon_shared)
  message(
    FATAL_ERROR
      "Paimon SDK components require the shared Paimon::paimon_shared target")
endif()

find_path(
  SIRIUS_PAIMON_ARROW_INCLUDE_DIR
  NAMES arrow/c/abi.h
  DOC "Include root for the Arrow C Data Interface used by Paimon")
if(NOT EXISTS "${SIRIUS_PAIMON_ARROW_INCLUDE_DIR}/arrow/c/abi.h")
  message(
    FATAL_ERROR
      "Paimon SDK components require arrow/c/abi.h. Set SIRIUS_PAIMON_ARROW_INCLUDE_DIR to the SDK's matching Arrow include root."
  )
endif()

target_sources(
  sirius_objects
  PRIVATE src/op/scan/paimon/paimon_batch_cursor.cpp
          src/op/scan/paimon/paimon_native_read.cpp
          src/op/scan/paimon/paimon_arrow_producer.cpp)
target_include_directories(sirius_objects SYSTEM
                           PRIVATE "${SIRIUS_PAIMON_ARROW_INCLUDE_DIR}")
target_link_libraries(sirius_objects PRIVATE Paimon::paimon_shared)
target_link_libraries(sirius_core PRIVATE Paimon::paimon_shared)
target_link_libraries(sirius_shared PRIVATE Paimon::paimon_shared)
