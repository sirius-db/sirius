# =============================================================================
# Copyright 2026, Sirius Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# Third-party code compiled into the Sirius library from this source tree.
# Dependencies found as packages (vcpkg or Conda) carry their own notices, and
# sirius-duckdb adds DuckDB's from the host DuckDB tree.
include("${CMAKE_CURRENT_LIST_DIR}/sirius-third-party-component.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sirius-rust-third-party.cmake")
if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()

set(SIRIUS_THIRD_PARTY_BINARY_DIR "${CMAKE_BINARY_DIR}/third_party")
set(SIRIUS_THIRD_PARTY_INSTALL_DIR
    "${CMAKE_INSTALL_DATADIR}/sirius/third_party")

function(sirius_git_revision out_var dir)
  set(revision "")
  find_package(Git QUIET)
  if(GIT_FOUND)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -C "${dir}" rev-parse --short=12 HEAD
      OUTPUT_VARIABLE revision
      OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  endif()
  set(${out_var}
      "${revision}"
      PARENT_SCOPE)
endfunction()

file(REMOVE_RECURSE "${SIRIUS_THIRD_PARTY_BINARY_DIR}")

sirius_git_revision(cucascade_revision "${PROJECT_SOURCE_DIR}/cucascade")
sirius_third_party_component(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/cucascade"
  NAME
  "cuCascade"
  VERSION
  "${cucascade_revision}"
  LICENSE
  "Apache-2.0"
  HOMEPAGE
  "https://github.com/NVIDIA/cuCascade"
  FILES
  "${PROJECT_SOURCE_DIR}/cucascade/LICENSE.md")

sirius_git_revision(substrait_revision "${PROJECT_SOURCE_DIR}/substrait")
sirius_third_party_component(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/duckdb-substrait"
  NAME
  "DuckDB Substrait extension"
  VERSION
  "${substrait_revision}"
  LICENSE
  "MIT"
  HOMEPAGE
  "https://github.com/sirius-db/duckdb-substrait-extension"
  FILES
  "${PROJECT_SOURCE_DIR}/substrait/LICENSE")
sirius_third_party_component(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/substrait"
  NAME
  "Substrait"
  LICENSE
  "Apache-2.0"
  HOMEPAGE
  "https://substrait.io"
  FILES
  "${PROJECT_SOURCE_DIR}/substrait/third_party/substrait/LICENSE")

file(STRINGS
     "${PROJECT_SOURCE_DIR}/substrait/third_party/google/protobuf/port_def.inc"
     protobuf_version_line REGEX "^#define PROTOBUF_VERSION [0-9]+$")
string(REGEX REPLACE ".* ([0-9]+)$" "\\1" protobuf_version
                     "${protobuf_version_line}")
math(EXPR protobuf_major "${protobuf_version} / 1000000")
math(EXPR protobuf_minor "${protobuf_version} / 1000 % 1000")
math(EXPR protobuf_patch "${protobuf_version} % 1000")
sirius_third_party_component(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/protobuf"
  NAME
  "Protocol Buffers"
  VERSION
  "${protobuf_major}.${protobuf_minor}.${protobuf_patch}"
  LICENSE
  "BSD-3-Clause"
  HOMEPAGE
  "https://github.com/protocolbuffers/protobuf"
  FILES
  "${PROJECT_SOURCE_DIR}/substrait/third_party/google/protobuf/LICENSE")

execute_process(
  COMMAND "${Rust_COMPILER_CACHED}" --version
  OUTPUT_VARIABLE rustc_version
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REGEX REPLACE "^rustc ([^ ]+).*" "\\1" rustc_version "${rustc_version}")
# Rust ships the notices of most crates its standard library bundles as
# COPYRIGHT-library.html; it omits compiler_builtins, addr2line, miniz_oxide and
# adler2, whose licenses are committed.
execute_process(
  COMMAND "${Rust_COMPILER_CACHED}" --print sysroot
  OUTPUT_VARIABLE rust_sysroot
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
set(rust_std_files
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-MIT"
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-APACHE"
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-compiler-builtins.txt"
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-MIT-addr2line"
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-MIT-adler2"
    "${PROJECT_SOURCE_DIR}/licenses/rust-std/LICENSE-MIT-miniz_oxide.md")
set(rust_std_note "")
set(rust_copyright "${rust_sysroot}/share/doc/rust/COPYRIGHT-library.html")
if(EXISTS "${rust_copyright}")
  file(READ "${rust_copyright}" rust_notices)
  string(REGEX REPLACE "<[^>]*>" "" rust_notices "${rust_notices}")
  # Match without the terminating semicolon, a CMake list separator.
  string(REGEX MATCHALL "&#[0-9]+" numeric_entities "${rust_notices}")
  list(REMOVE_DUPLICATES numeric_entities)
  foreach(entity IN LISTS numeric_entities)
    string(SUBSTRING "${entity}" 2 -1 code)
    if(code GREATER_EQUAL 32 AND code LESS 127)
      string(ASCII ${code} character)
      string(REPLACE "${entity};" "${character}" rust_notices "${rust_notices}")
    endif()
  endforeach()
  foreach(entity IN ITEMS "&lt;=<" "&gt;=>" "&quot;=\"" "&apos;='" "&nbsp;= "
                          "&amp;=&")
    string(REGEX MATCH "^([^=]+)=(.*)$" _ "${entity}")
    string(REPLACE "${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" rust_notices
                   "${rust_notices}")
  endforeach()
  string(REGEX REPLACE "[ \t]+\n" "\n" rust_notices "${rust_notices}")
  string(REGEX REPLACE "\n\n\n+" "\n\n" rust_notices "${rust_notices}")
  file(WRITE "${CMAKE_BINARY_DIR}/rust-std/COPYRIGHT-library.txt"
       "${rust_notices}")
  list(APPEND rust_std_files
       "${CMAKE_BINARY_DIR}/rust-std/COPYRIGHT-library.txt")
  set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS "${rust_copyright}")
else()
  message(WARNING "Rust ${rustc_version} has no ${rust_copyright}; the notices "
                  "of crates bundled in its standard library are not embedded")
  set(rust_std_note
      "The Rust standard library also bundles crates such as hashbrown, gimli and rustc-demangle; COPYRIGHT-library.html of Rust ${rustc_version} lists their licenses."
  )
endif()
sirius_third_party_component(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/rust-std"
  NAME
  "Rust standard library"
  VERSION
  "${rustc_version}"
  LICENSE
  "(MIT OR Apache-2.0) AND MIT AND Apache-2.0 WITH LLVM-exception"
  HOMEPAGE
  "https://www.rust-lang.org"
  FILES
  ${rust_std_files}
  NOTE
  "${rust_std_note}")
sirius_rust_third_party_components(
  "${SIRIUS_THIRD_PARTY_BINARY_DIR}/rust"
  CARGO
  "${Rust_CARGO_CACHED}"
  MANIFEST_PATH
  "${PROJECT_SOURCE_DIR}/rust/Cargo.toml"
  PACKAGE
  telemetry-bridge
  TARGET
  "${Rust_CARGO_TARGET_CACHED}")
set_property(
  DIRECTORY
  APPEND
  PROPERTY CMAKE_CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/rust/Cargo.lock")

# Replace, not merge, so components dropped since the last install disappear.
cmake_path(ABSOLUTE_PATH SIRIUS_THIRD_PARTY_INSTALL_DIR BASE_DIRECTORY
           "\${CMAKE_INSTALL_PREFIX}" OUTPUT_VARIABLE third_party_install_path)
install(
  CODE "file(REMOVE_RECURSE \"\$ENV{DESTDIR}${third_party_install_path}\")"
  COMPONENT sirius_library)
install(
  DIRECTORY "${SIRIUS_THIRD_PARTY_BINARY_DIR}/"
  DESTINATION "${SIRIUS_THIRD_PARTY_INSTALL_DIR}"
  COMPONENT sirius_library)
