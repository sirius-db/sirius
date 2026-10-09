cmake_minimum_required(VERSION 3.30.4)
if(REMOVE_RPATH)
  # Conda compiler drivers can inject paths independently of CMake.
  file(RPATH_REMOVE FILE "${EXTENSION}")
endif()
find_program(READELF NAMES readelf llvm-readelf REQUIRED)
execute_process(COMMAND "${READELF}" -d "${EXTENSION}"
                OUTPUT_VARIABLE dynamic COMMAND_ERROR_IS_FATAL ANY)
string(REGEX MATCHALL "\\(NEEDED\\)[^\n]*" needed "${dynamic}")
foreach(entry IN LISTS needed)
  string(REGEX REPLACE ".*\\[([^]]+)\\].*" "\\1" library "${entry}")
  if(NOT
     library
     MATCHES
     "^(lib(c|m|dl|rt|pthread|util|resolv|cuda|nvidia-ml)\\.so(\\.[0-9]+)*|ld-linux[^/]*\\.so(\\.[0-9]+)*)$"
  )
    message(
      FATAL_ERROR
        "Distribution extension has an unbundled dependency: ${library}")
  endif()
endforeach()
if(dynamic MATCHES "\\((RPATH|RUNPATH)\\)")
  message(FATAL_ERROR "Distribution extension contains a runtime search path")
endif()
