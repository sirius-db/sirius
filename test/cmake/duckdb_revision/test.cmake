# Checks when the DuckDB extension accepts a Sirius package, from the root of a
# Sirius checkout:
#
# cmake -P test/cmake/duckdb_revision/test.cmake
cmake_minimum_required(VERSION 3.30.4)

set(sirius "${CMAKE_CURRENT_LIST_DIR}/../../..")

# Each case runs this script again, so a rejection does not end the test.
if(DEFINED CASE_LINKAGE)
  include("${sirius}/cmake/sirius-duckdb-revision.cmake")
  sirius_require_duckdb_revision("${CASE_LIBRARY_REVISION}"
                                 "${CASE_SOURCE_DIR}" "${CASE_LINKAGE}")
  return()
endif()

find_program(git NAMES git REQUIRED)
string(RANDOM LENGTH 8 run)
set(work "${sirius}/build/duckdb-revision-test-${run}")

# Variables that would make git act on the caller's repository.
execute_process(
  COMMAND "${git}" rev-parse --local-env-vars
  OUTPUT_VARIABLE local_variables
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REGEX REPLACE "[ \n]+" ";" local_variables "${local_variables}")
list(TRANSFORM local_variables PREPEND "--unset=")

# Creates a repository with one commit and sets <out> to that commit. The
# directory names the commit, so two repositories get different revisions.
function(create_repository dir out)
  file(MAKE_DIRECTORY "${dir}")
  set(git_in_dir
      "${CMAKE_COMMAND}"
      -E
      env
      ${local_variables}
      "${git}"
      -C
      "${dir}")
  execute_process(COMMAND ${git_in_dir} init --quiet COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND
      ${git_in_dir} -c user.name=test -c user.email=test@example.invalid -c
      core.hooksPath=/dev/null commit --quiet --no-gpg-sign --allow-empty -m
      "${dir}" COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND ${git_in_dir} rev-parse HEAD
    OUTPUT_VARIABLE revision
    OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  set(${out}
      "${revision}"
      PARENT_SCOPE)
endfunction()

# Runs <ARGN>. <expected> is "accepted", or a regex its failure must match.
function(expect name expected)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE result
    ERROR_VARIABLE error
    OUTPUT_QUIET)
  # CMake wraps long messages.
  string(REGEX REPLACE "[ \t\r\n]+" " " error "${error}")
  if(expected STREQUAL "accepted")
    if(NOT result EQUAL 0)
      message(SEND_ERROR "${name}: expected acceptance, got:\n${error}")
    endif()
  elseif(result EQUAL 0 OR NOT error MATCHES "${expected}")
    message(
      SEND_ERROR
        "${name}: expected a rejection matching '${expected}', got:\n${error}")
  endif()
  message(STATUS "${name}")
endfunction()

# Checks the comparison itself.
function(check name expected library_revision source_dir linkage)
  expect(
    "${name}"
    "${expected}"
    "${CMAKE_COMMAND}"
    "-DCASE_LIBRARY_REVISION=${library_revision}"
    "-DCASE_SOURCE_DIR=${source_dir}"
    "-DCASE_LINKAGE=${linkage}"
    -P
    "${CMAKE_CURRENT_LIST_FILE}")
endfunction()

create_repository("${work}/duckdb" revision)
create_repository("${work}/other" other)
# A directory inside a repository is not a DuckDB checkout of its own.
file(MAKE_DIRECTORY "${work}/duckdb/nested")
set(mismatch
    "was built from DuckDB ${other}, but this extension is built from DuckDB ${revision}"
)
set(unknown "Cannot verify")

foreach(linkage IN ITEMS shared static)
  check("${linkage}: same revision" accepted "${revision}" "${work}/duckdb"
        ${linkage})
  check("${linkage}: other revision" "${mismatch}" "${other}" "${work}/duckdb"
        ${linkage})
endforeach()
check("shared: unknown library revision" accepted "" "${work}/duckdb" shared)
check("static: unknown library revision" "${unknown}" "" "${work}/duckdb"
      static)
check("shared: source is not a checkout" accepted "${revision}"
      "${work}/duckdb/nested" shared)
check("static: source is not a checkout" "${unknown}" "${revision}"
      "${work}/duckdb/nested" static)

# An inherited GIT_DIR must not make git answer for another repository.
set(ENV{GIT_DIR} "${work}/other/.git")
check("static: inherited GIT_DIR, same revision" accepted "${revision}"
      "${work}/duckdb" static)
check("static: inherited GIT_DIR, other revision" "${mismatch}" "${other}"
      "${work}/duckdb" static)
unset(ENV{GIT_DIR})

# The wrapper applies the check to the package it finds. Configure it in a
# stand-in for a DuckDB tree embedded in a project at the other revision, so
# only DuckDB's own directory gives the revision the package is compared with.
function(write_package dir library_revision)
  set(SIRIUS_DUCKDB_SOURCE_REVISION "${library_revision}")
  set(PACKAGE_INIT "macro(check_required_components)\nendmacro()")
  configure_file("${sirius}/cmake/sirius-config.cmake.in"
                 "${dir}/sirius-config.cmake" @ONLY)
  file(COPY "${sirius}/cmake/sirius-duckdb-revision.cmake" DESTINATION "${dir}")
  file(WRITE "${dir}/sirius-targets.cmake"
       "add_library(sirius::sirius INTERFACE IMPORTED)\n")
  file(WRITE "${dir}/sirius-static-targets.cmake"
       "add_library(sirius::sirius_static INTERFACE IMPORTED)\n")
  file(WRITE "${dir}/sirius-static-dependencies.cmake" "")
  file(WRITE "${dir}/sirius-static-targetsCorrosion.cmake" "")
endfunction()
write_package("${work}/package-other" "${other}")
write_package("${work}/package-unknown" "")
set(passed "reached the extension targets")
file(
  WRITE "${work}/other/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.30.4)
project(host LANGUAGES NONE)
add_subdirectory(\"${work}/duckdb\" duckdb)
")
file(
  WRITE "${work}/duckdb/CMakeLists.txt"
  "project(DuckDB LANGUAGES NONE)
function(set_extension_version_sources)
endfunction()
function(build_static_extension)
  message(FATAL_ERROR \"${passed}\")
endfunction()
set(EXTENSION_NAME sirius)
add_subdirectory(\"${sirius}/sirius-duckdb\" wrapper)
")

# Configures the wrapper with <linkage> against the package in <package>.
function(check_wrapper name expected package linkage)
  expect(
    "${name}"
    "${expected}"
    "${CMAKE_COMMAND}"
    -S
    "${work}/other"
    -B
    "${work}/wrapper-${package}-${linkage}"
    "-Dsirius_DIR=${work}/${package}"
    "-DSIRIUS_DUCKDB_LINKAGE=${linkage}")
endfunction()
check_wrapper("shared wrapper: package from another revision" "${mismatch}"
              package-other shared)
check_wrapper("static wrapper: package from another revision" "${mismatch}"
              package-other static)
check_wrapper("shared wrapper: package from an unknown revision" "${passed}"
              package-unknown shared)
check_wrapper("static wrapper: package from an unknown revision" "${unknown}"
              package-unknown static)

file(REMOVE_RECURSE "${work}")
