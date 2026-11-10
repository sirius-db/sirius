cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

project(sirius-duckdb)

set(extension_name ${EXTENSION_NAME})
if(NOT extension_name MATCHES "^sirius(_cuda(12|13))?$")
  message(
    FATAL_ERROR "EXTENSION_NAME must be sirius, sirius_cuda12, or sirius_cuda13"
  )
endif()
string(TOUPPER "${extension_name}" extension_name_upper)
set_extension_version_sources(${extension_name})

set(SIRIUS_DUCKDB_LINKAGE
    shared
    CACHE STRING "Sirius library linkage")
set_property(CACHE SIRIUS_DUCKDB_LINKAGE PROPERTY STRINGS static shared)
if(SIRIUS_DUCKDB_LINKAGE STREQUAL "static")
  find_package(sirius CONFIG REQUIRED COMPONENTS static)
  set(sirius_target sirius::sirius_static)
elseif(SIRIUS_DUCKDB_LINKAGE STREQUAL "shared")
  find_package(sirius CONFIG REQUIRED)
  if(NOT TARGET sirius::sirius)
    message(FATAL_ERROR "Shared linkage requires the Sirius shared library")
  endif()
  set(sirius_target sirius::sirius)
else()
  message(FATAL_ERROR "SIRIUS_DUCKDB_LINKAGE must be static or shared")
endif()

include(cmake/third-party-notices.cmake)
set(notice_components "${CMAKE_CURRENT_BINARY_DIR}/third_party_components")
file(REMOVE_RECURSE "${notice_components}")
sirius_notices_add_duckdb("${notice_components}/duckdb" "${DuckDB_SOURCE_DIR}"
                          "${DUCKDB_VERSION}")
set(notice_directories "${notice_components}" "${SIRIUS_THIRD_PARTY_DIR}")
if(SIRIUS_DUCKDB_LINKAGE STREQUAL "static")
  if(NOT VCPKG_INSTALLED_DIR OR NOT VCPKG_TARGET_TRIPLET)
    message(FATAL_ERROR "Static linkage requires the vcpkg installed tree")
  endif()
  set(vcpkg_installed "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
  sirius_notices_add_vcpkg("${notice_components}/vcpkg" "${vcpkg_installed}")
  # Ports install the components they vendor under share/<port>/third_party.
  file(
    GLOB port_components CONFIGURE_DEPENDS
    LIST_DIRECTORIES true
    "${vcpkg_installed}/share/*/third_party")
  list(APPEND notice_directories ${port_components})
  # The backend links its C++ runtime statically (see the link options below).
  set(gcc_runtime_version "")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(gcc_runtime_version "${CMAKE_CXX_COMPILER_VERSION}")
  endif()
  sirius_third_party_component(
    "${notice_components}/gcc-runtime"
    NAME
    "GCC runtime libraries (libstdc++, libgcc)"
    VERSION
    "${gcc_runtime_version}"
    LICENSE
    "GPL-3.0-or-later WITH GCC-exception-3.1"
    HOMEPAGE
    "https://gcc.gnu.org/onlinedocs/libstdc++/manual/license.html"
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/../licenses/gcc-runtime/COPYING.RUNTIME"
    "${CMAKE_CURRENT_SOURCE_DIR}/../licenses/gcc-runtime/COPYING3")
endif()
sirius_notices_generate(
  notices_source
  SYMBOL
  "${extension_name}"
  TITLE
  "Third-party software notices for the ${extension_name} DuckDB extension"
  VERSION
  "${DUCKDB_EXTENSION_${extension_name_upper}_EXT_VERSION}"
  SIRIUS_LICENSE
  "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE"
  DIRECTORIES
  ${notice_directories})

set(EXTENSION_SOURCES src/sirius/sirius_extension.cpp
                      src/sirius/sirius_licenses.cpp ${notices_source})

# DuckDB's linker options are directory-wide; this backend uses BFD. Keep the
# override local so the host and sibling extensions retain their linker.
foreach(flags IN ITEMS CMAKE_SHARED_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS)
  if(DUCKDB_LINKER_FLAG)
    string(REPLACE "${DUCKDB_LINKER_FLAG}" "" ${flags} "${${flags}}")
  endif()
  if(DUCKDB_MOLD_FLAGS)
    string(REPLACE "${DUCKDB_MOLD_FLAGS}" "" ${flags} "${${flags}}")
  endif()
  string(REGEX REPLACE "-fuse-ld=bfd([ \t]|$)" " " remaining_flags
                       "${${flags}}")
  if(remaining_flags MATCHES "-fuse-ld=")
    message(
      FATAL_ERROR
        "Sirius requires BFD: remove -fuse-ld from ${flags} and select the host linker with DUCKDB_LINKER"
    )
  endif()
endforeach()

build_static_extension(${extension_name} ${EXTENSION_SOURCES})
build_loadable_extension(${extension_name} " " ${EXTENSION_SOURCES})
# Preserve RMM's shared device-resource registry across library boundaries.
set_target_properties(${extension_name}_loadable_extension
                      PROPERTIES LINKER_TYPE BFD)

if(SIRIUS_DUCKDB_LINKAGE STREQUAL "static")
  configure_file(exports.map.in exports.map @ONLY)
  get_target_property(sirius_link_libraries
                      ${extension_name}_loadable_extension LINK_LIBRARIES)
  list(REMOVE_ITEM sirius_link_libraries "-Wl,--exclude-libs,ALL" duckdb_static)
  set_target_properties(
    ${extension_name}_loadable_extension
    PROPERTIES LINK_LIBRARIES "${sirius_link_libraries}" SKIP_BUILD_RPATH ON)
  target_link_options(
    ${extension_name}_loadable_extension PRIVATE
    "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/exports.map"
    -static-libgcc -static-libstdc++)
  set_property(
    TARGET ${extension_name}_loadable_extension
    APPEND
    PROPERTY LINK_DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/exports.map")
  add_custom_command(
    TARGET ${extension_name}_loadable_extension
    POST_BUILD
    COMMAND
      "${CMAKE_COMMAND}" -DREMOVE_RPATH=ON
      "-DEXTENSION=$<TARGET_FILE:${extension_name}_loadable_extension>" -P
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check-static-extension.cmake"
    VERBATIM)
endif()

foreach(target IN ITEMS ${extension_name}_extension
                        ${extension_name}_loadable_extension)
  target_link_libraries(${target} ${sirius_target})
  target_compile_definitions(
    ${target}
    PRIVATE
      SIRIUS_EXTENSION_NAME=${extension_name}
      SIRIUS_EXTENSION_VERSION="${DUCKDB_EXTENSION_${extension_name_upper}_EXT_VERSION}"
      SIRIUS_NOTICES_NAMESPACE=${notices_source_NAMESPACE})
  set_target_properties(
    ${target}
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CXX_SCAN_FOR_MODULES OFF)
  target_include_directories(${target}
                             PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
endforeach()

install(
  TARGETS ${extension_name}_extension
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
