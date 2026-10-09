# Only Sirius depends on the local checkout; other ports retain their cache
# keys.
if(NOT PORT STREQUAL "sirius")
  return()
endif()

get_filename_component(sirius_source "${CMAKE_CURRENT_LIST_DIR}/../../.."
                       ABSOLUTE)
find_program(sirius_git NAMES git REQUIRED)
set(sirius_files)
foreach(repository . duckdb cucascade substrait)
  if(NOT EXISTS "${sirius_source}/${repository}/CMakeLists.txt")
    message(
      FATAL_ERROR "Initialize the Sirius engine submodules before building")
  endif()
  set(paths .)
  if(repository STREQUAL ".")
    set(paths CMakeLists.txt LICENSE cmake include src rust)
  endif()
  execute_process(
    COMMAND
      "${sirius_git}" -c core.quotePath=false -C
      "${sirius_source}/${repository}" ls-files --cached --others
      --exclude-standard -- ${paths}
    OUTPUT_VARIABLE files COMMAND_ERROR_IS_FATAL ANY)
  if(files MATCHES ";")
    message(FATAL_ERROR "Sirius source paths cannot contain semicolons")
  endif()
  string(REGEX REPLACE "\n$" "" files "${files}")
  string(REPLACE "\n" ";" files "${files}")
  foreach(file IN LISTS files)
    if(file MATCHES "^\"")
      string(JSON file GET "[${file}]" 0)
    endif()
    cmake_path(SET path NORMALIZE "${sirius_source}/${repository}/${file}")
    # Deleted files and unused nested submodules are not build inputs.
    if(EXISTS "${path}" AND NOT IS_DIRECTORY "${path}")
      list(APPEND sirius_files "${path}")
    endif()
  endforeach()
endforeach()
list(REMOVE_DUPLICATES sirius_files)
list(SORT sirius_files)

# Hash relative names too, so renames invalidate the package without making
# cache keys depend on the checkout location.
set(sirius_inventory
    "${CMAKE_CURRENT_LIST_DIR}/../../build/vcpkg/sirius-source-files.txt")
set(sirius_hashes "")
foreach(path IN LISTS sirius_files)
  file(RELATIVE_PATH name "${sirius_source}" "${path}")
  file(SHA256 "${path}" hash)
  string(APPEND sirius_hashes "${hash}  ${name}\n")
endforeach()
file(WRITE "${sirius_inventory}" "${sirius_hashes}")
list(APPEND VCPKG_HASH_ADDITIONAL_FILES "${sirius_inventory}")
