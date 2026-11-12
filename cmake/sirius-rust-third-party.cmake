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

include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/sirius-third-party-component.cmake")
if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()

# sirius_rust_third_party_components(<dir> CARGO <cargo> MANIFEST_PATH <toml>
# PACKAGE <crate> TARGET <triple>)
#
# Writes one component per crate linked into PACKAGE for TARGET: normal
# dependencies only, without proc-macros or build scripts, and without the path
# crates of this repository. Crate sources must already be fetched.
function(sirius_rust_third_party_components dir)
  cmake_parse_arguments(PARSE_ARGV 1 arg ""
                        "CARGO;MANIFEST_PATH;PACKAGE;TARGET" "")
  # License expressions for git dependencies whose manifests declare none, keyed
  # by repository URL without the query or fragment.
  set(license_clarifications "https://github.com/rapidsai/quent.git=Apache-2.0")
  # License files for crates whose packages ship none.
  set(license_files
      "nvtx-sys=${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../licenses/nvtx/LICENSE.txt"
  )
  # Fetching also unpacks the sources whose license files are read below.
  execute_process(
    COMMAND "${arg_CARGO}" fetch --locked --manifest-path "${arg_MANIFEST_PATH}"
            --target "${arg_TARGET}"
    ERROR_VARIABLE fetch_error
    RESULT_VARIABLE fetch_result)
  if(NOT fetch_result EQUAL 0)
    message(
      FATAL_ERROR "cargo fetch failed for ${arg_PACKAGE}:\n${fetch_error}")
  endif()
  execute_process(
    COMMAND
      "${arg_CARGO}" tree --locked --offline --manifest-path
      "${arg_MANIFEST_PATH}" --package "${arg_PACKAGE}" --target "${arg_TARGET}"
      --edges normal,no-proc-macro --prefix none --format "{p}|{l}|{r}"
    OUTPUT_VARIABLE tree
    ERROR_VARIABLE tree_error
    RESULT_VARIABLE tree_result)
  if(NOT tree_result EQUAL 0)
    message(FATAL_ERROR "cargo tree failed for ${arg_PACKAGE}:\n${tree_error}")
  endif()
  if(DEFINED ENV{CARGO_HOME})
    set(cargo_home "$ENV{CARGO_HOME}")
  else()
    set(cargo_home "$ENV{HOME}/.cargo")
  endif()

  file(REMOVE_RECURSE "${dir}")
  string(REPLACE "\n" ";" lines "${tree}")
  set(seen)
  foreach(line IN LISTS lines)
    string(REGEX REPLACE " \\(\\*\\)$" "" line "${line}")
    if(line STREQUAL "" OR line IN_LIST seen)
      continue()
    endif()
    list(APPEND seen "${line}")
    if(NOT line MATCHES
       "^([^ |]+) v([^ |]+)( \\(([^)]*)\\))?\\|([^|]*)\\|(.*)$")
      message(FATAL_ERROR "Unexpected cargo tree line: ${line}")
    endif()
    set(name "${CMAKE_MATCH_1}")
    set(version "${CMAKE_MATCH_2}")
    set(component "rust-${CMAKE_MATCH_1}-${CMAKE_MATCH_2}")
    set(source "${CMAKE_MATCH_4}")
    set(license "${CMAKE_MATCH_5}")
    set(homepage "${CMAKE_MATCH_6}")
    if(source MATCHES "^/")
      # Path crates belong to this repository.
      continue()
    elseif(source MATCHES "^([^?#]+)[^#]*#([0-9a-f]+)$")
      set(repository "${CMAKE_MATCH_1}")
      set(revision "${CMAKE_MATCH_2}")
      # Checkouts are named by an abbreviated revision.
      file(
        GLOB checkouts
        LIST_DIRECTORIES true
        "${cargo_home}/git/checkouts/*/*")
      set(crate_dirs)
      foreach(checkout IN LISTS checkouts)
        cmake_path(GET checkout FILENAME short_revision)
        string(LENGTH "${short_revision}" short_length)
        string(SUBSTRING "${revision}" 0 ${short_length} prefix)
        if(IS_DIRECTORY "${checkout}"
           AND short_length GREATER_EQUAL 7
           AND prefix STREQUAL short_revision)
          list(APPEND crate_dirs "${checkout}")
        endif()
      endforeach()
      if(NOT homepage)
        string(REGEX REPLACE "\\.git$" "" homepage "${repository}")
      endif()
      if(NOT license)
        foreach(clarification IN LISTS license_clarifications)
          if(clarification MATCHES "^${repository}=(.+)$")
            set(license "${CMAKE_MATCH_1}")
          endif()
        endforeach()
      endif()
      set(version "${version} (${revision})")
    elseif(source STREQUAL "")
      file(
        GLOB crate_dirs
        LIST_DIRECTORIES true
        "${cargo_home}/registry/src/*/${name}-${version}")
    else()
      message(FATAL_ERROR "Unsupported source for crate ${name}: ${source}")
    endif()
    # Several registry caches or checkout URLs may hold the crate; cargo
    # verified each copy, so any of them has the same files.
    if(NOT crate_dirs)
      message(
        FATAL_ERROR
          "No source directory for crate ${name} ${version} under ${cargo_home}"
      )
    endif()
    list(SORT crate_dirs)
    list(GET crate_dirs 0 crate_dir)
    if(NOT license)
      message(FATAL_ERROR "Crate ${name} declares no license")
    endif()
    file(
      GLOB candidates
      LIST_DIRECTORIES false
      "${crate_dir}/*")
    set(files)
    foreach(candidate IN LISTS candidates)
      cmake_path(GET candidate FILENAME file_name)
      string(TOUPPER "${file_name}" upper)
      if(upper MATCHES "^(LICEN[CS]E|COPYING|COPYRIGHT|NOTICE|UNLICENSE)")
        list(APPEND files "${candidate}")
      endif()
    endforeach()
    list(SORT files)
    if(NOT files)
      foreach(entry IN LISTS license_files)
        if(entry MATCHES "^${name}=(.+)$")
          set(files "${CMAKE_MATCH_1}")
        endif()
      endforeach()
    endif()
    set(note)
    if(NOT files)
      set(note
          "The ${name} crate is distributed under ${license} but its package includes no license file; see ${homepage}."
      )
    endif()
    sirius_third_party_component(
      "${dir}/${component}"
      NAME
      "${name} (Rust crate)"
      VERSION
      "${version}"
      LICENSE
      "${license}"
      HOMEPAGE
      "${homepage}"
      FILES
      ${files}
      NOTE
      "${note}")
  endforeach()
endfunction()
