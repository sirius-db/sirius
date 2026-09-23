# Build private fixture header providers through the production generators.
# Source, options, and cache/compiler implementation remain identical.
file(MAKE_DIRECTORY "${OUT_DIR}/codegen/jit" "${OUT_DIR}/inputs")
set(stdint_input "${SOURCE}/src/codegen/stdint_shim.hpp")
set(rle_input "${SOURCE}/src/codegen/decode/rle_block.cuh")
set(cccl_inputs "${INCLUDE_DIRS_FILE}")
if(VARIANT STREQUAL "project")
  file(READ "${stdint_input}" content)
  set(stdint_input "${OUT_DIR}/inputs/stdint_shim.hpp")
  file(WRITE "${stdint_input}"
       "${content}\n#define SIMPATICO_CACHE_TEST_VALUE 2\n")
elseif(VARIANT STREQUAL "cccl")
  file(STRINGS "${INCLUDE_DIRS_FILE}" roots)
  foreach(root IN LISTS roots)
    if(EXISTS "${root}/cuda/std/type_traits")
      file(READ "${root}/cuda/std/type_traits" content)
      break()
    endif()
  endforeach()
  if(NOT DEFINED content)
    message(FATAL_ERROR "CCCL fixture input not found")
  endif()
  file(MAKE_DIRECTORY "${OUT_DIR}/inputs/cccl/cuda/std")
  file(WRITE "${OUT_DIR}/inputs/cccl/cuda/std/type_traits"
       "${content}\n#define SIMPATICO_CACHE_TEST_VALUE 3\n")
  file(READ "${INCLUDE_DIRS_FILE}" includes)
  set(cccl_inputs "${OUT_DIR}/inputs/include_dirs.txt")
  file(WRITE "${cccl_inputs}" "${OUT_DIR}/inputs/cccl\n${includes}")
endif()
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" "-DIN_STDINT=${stdint_input}" "-DIN_RLE=${rle_input}"
    "-DOUT=${OUT_DIR}/codegen/jit/embedded_headers.h" -P
    "${SOURCE}/cmake/embed_jit_headers.cmake" COMMAND_ERROR_IS_FATAL ANY)
execute_process(
  COMMAND
    "${CMAKE_COMMAND}" "-DINCLUDE_DIRS_FILE=${cccl_inputs}"
    "-DOUT=${OUT_DIR}/codegen/jit/cccl_embedded_headers.cpp" -P
    "${SOURCE}/cmake/embed_cccl_headers.cmake" COMMAND_ERROR_IS_FATAL ANY)
