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

# A third-party component is a directory holding component.cmake (name, version,
# SPDX license expression, homepage, license file names) next to its license and
# notice files. The engine and the vcpkg overlay ports install these under
# share/<package>/third_party; sirius-duckdb embeds them in the extension.
include_guard(GLOBAL)
# Empty keyword values mean "not set" either way; NEW silences the warning.
if(POLICY CMP0174)
  cmake_policy(SET CMP0174 NEW)
endif()

# sirius_third_party_component(<dir> NAME <name> LICENSE <spdx> [VERSION
# <version>] [HOMEPAGE <url>] [FILES <file>...] [NOTE <text>])
#
# Writes the component into <dir>. FILES are copied under their file names. NOTE
# replaces the license text when a component ships no license file.
function(sirius_third_party_component dir)
  cmake_parse_arguments(PARSE_ARGV 1 arg ""
                        "NAME;VERSION;LICENSE;HOMEPAGE;NOTE" "FILES")
  if(arg_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unexpected third-party arguments: ${arg_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT arg_NAME OR NOT arg_LICENSE)
    message(
      FATAL_ERROR "Third-party component in ${dir} needs NAME and LICENSE")
  endif()
  if(NOT arg_FILES AND NOT arg_NOTE)
    message(FATAL_ERROR "Third-party component ${arg_NAME} needs FILES or NOTE")
  endif()
  file(REMOVE_RECURSE "${dir}")
  file(MAKE_DIRECTORY "${dir}")
  set(file_names)
  foreach(file IN LISTS arg_FILES)
    if(NOT EXISTS "${file}")
      message(FATAL_ERROR "License file of ${arg_NAME} not found: ${file}")
    endif()
    cmake_path(GET file FILENAME file_name)
    if(file_name IN_LIST file_names OR file_name STREQUAL "component.cmake")
      message(
        FATAL_ERROR "Duplicate license file name for ${arg_NAME}: ${file}")
    endif()
    list(APPEND file_names "${file_name}")
    file(COPY_FILE "${file}" "${dir}/${file_name}")
    if(NOT CMAKE_SCRIPT_MODE_FILE)
      set_property(
        DIRECTORY
        APPEND
        PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
    endif()
  endforeach()
  set(content "")
  foreach(field IN ITEMS NAME VERSION LICENSE HOMEPAGE NOTE)
    string(TOLOWER "${field}" key)
    string(FIND "${arg_${field}}" "]==]" bracket)
    if(NOT bracket EQUAL -1)
      message(FATAL_ERROR "Third-party ${key} of ${arg_NAME} contains ]==]")
    endif()
    string(APPEND content "set(component_${key} [==[${arg_${field}}]==])\n")
  endforeach()
  string(APPEND content "set(component_files [==[${file_names}]==])\n")
  file(WRITE "${dir}/component.cmake" "${content}")
endfunction()

# sirius_third_party_source_notices(<output> ROOT <dir> LICENSE <spdx> SOURCES
# <file>...)
#
# Writes the copyright and license comment blocks of SOURCES whose SPDX license
# identifiers name anything besides Apache-2.0: third-party snippets embedded in
# otherwise Apache-2.0 sources. Fails unless every license found appears in
# LICENSE, the expression declared for the snippets, and every such file yields
# a notice.
function(sirius_third_party_source_notices output)
  cmake_parse_arguments(PARSE_ARGV 1 arg "" "ROOT;LICENSE" "SOURCES")
  string(REGEX MATCHALL "[A-Za-z0-9.+-]+" declared "${arg_LICENSE}")
  set(notices "")
  foreach(source IN LISTS arg_SOURCES)
    file(STRINGS "${source}" identifiers REGEX "SPDX-License-Identifier:")
    set(third_party FALSE)
    foreach(identifier IN LISTS identifiers)
      string(REGEX REPLACE ".*SPDX-License-Identifier:[ \t]*" "" expression
                           "${identifier}")
      string(REGEX REPLACE "[ \t]*(\\*/)?[ \t]*$" "" expression "${expression}")
      string(REGEX MATCHALL "[A-Za-z0-9.+-]+" terms "${expression}")
      foreach(term IN LISTS terms)
        if(term MATCHES "^(AND|OR|WITH|Apache-2\\.0)$")
          continue()
        endif()
        set(third_party TRUE)
        if(NOT term IN_LIST declared)
          file(RELATIVE_PATH name "${arg_ROOT}" "${source}")
          message(FATAL_ERROR "${name} has ${term} code; add it to the "
                              "declared snippet license ${arg_LICENSE}")
        endif()
      endforeach()
    endforeach()
    if(NOT third_party)
      continue()
    endif()
    file(READ "${source}" rest)
    file(RELATIVE_PATH name "${arg_ROOT}" "${source}")
    set(blocks "")
    while(TRUE)
      string(FIND "${rest}" "/*" start)
      if(start EQUAL -1)
        break()
      endif()
      string(SUBSTRING "${rest}" ${start} -1 rest)
      string(FIND "${rest}" "*/" end)
      if(end EQUAL -1)
        break()
      endif()
      math(EXPR length "${end} + 2")
      string(SUBSTRING "${rest}" 0 ${length} block)
      string(SUBSTRING "${rest}" ${length} -1 rest)
      string(TOLOWER "${block}" lower)
      if(lower MATCHES "copyright|licen")
        string(APPEND blocks "${block}\n\n")
      endif()
    endwhile()
    if(blocks STREQUAL "")
      message(FATAL_ERROR "No notice comment found in ${name}")
    endif()
    string(APPEND notices "---- ${name} ----\n\n${blocks}")
  endforeach()
  if(notices STREQUAL "")
    message(FATAL_ERROR "No third-party snippets found under ${arg_ROOT}")
  endif()
  file(WRITE "${output}" "${notices}")
endfunction()
