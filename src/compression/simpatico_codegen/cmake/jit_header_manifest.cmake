# Header bundle v2: XXH3-128 of "simpatico-headers-v2\n" followed by sorted
# records: decimal-byte-length(name) : name decimal-byte-length(contents) :
# contents. Lengths delimit fields even when contents contain colons/newlines.
# Normalize before BOTH embedding and hashing. No per-header digests.
find_program(
  XXHSUM_EXECUTABLE xxhsum
  HINTS "${VCPKG_INSTALLED_DIR}/${VCPKG_HOST_TRIPLET}/tools/xxhash"
  NO_CMAKE_FIND_ROOT_PATH REQUIRED)

function(jit_read_header path output)
  file(READ "${path}" contents)
  # CMake's text read removes a trailing CR, even without a following LF.
  # Restore that newline before normalizing, including legacy CR-only files.
  file(SIZE "${path}" size)
  if(size GREATER 0)
    math(EXPR last "${size}-1")
    file(
      READ "${path}" last_byte
      OFFSET ${last}
      LIMIT 1
      HEX)
    if(last_byte STREQUAL "0d")
      string(APPEND contents "\r")
    endif()
  endif()
  string(REPLACE "\r\n" "\n" normalized "${contents}")
  string(REPLACE "\r" "\n" normalized "${normalized}")
  set(${output}
      "${normalized}"
      PARENT_SCOPE)
endfunction()

function(jit_manifest_record name contents output)
  string(LENGTH "${name}" name_length)
  string(LENGTH "${contents}" content_length)
  set(${output}
      "${name_length}:${name}${content_length}:${contents}"
      PARENT_SCOPE)
endfunction()

function(jit_file_identity path output)
  execute_process(COMMAND "${XXHSUM_EXECUTABLE}" -H128 "${path}"
                  OUTPUT_VARIABLE checksum COMMAND_ERROR_IS_FATAL ANY)
  string(REGEX MATCH "^[0-9a-fA-F]+" digest "${checksum}")
  string(LENGTH "${digest}" digest_length)
  if(NOT digest_length EQUAL 32)
    message(FATAL_ERROR "xxhsum -H128 returned an invalid digest: ${checksum}")
  endif()
  string(TOLOWER "${digest}" digest)
  set(${output}
      "${digest}"
      PARENT_SCOPE)
endfunction()

function(jit_bundle_identity manifest output)
  # Each generated output owns its scratch file, including parallel builds.
  set(input "${OUT}.manifest")
  file(WRITE "${input}" "${manifest}")
  jit_file_identity("${input}" digest)
  file(REMOVE "${input}")
  set(${output}
      "${digest}"
      PARENT_SCOPE)
endfunction()
