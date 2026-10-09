# Host-only test of cmake/jit_epoch.cmake: the epoch depends on object contents
# and their order, not on object paths. Invoked via `cmake -P` with
# JIT_EPOCH_SCRIPT and WORK_DIR.
cmake_minimum_required(VERSION 3.24)

file(REMOVE_RECURSE "${WORK_DIR}")

function(write_objects dir)
  set(paths "")
  set(index 0)
  foreach(content IN LISTS ARGN)
    file(WRITE "${dir}/object${index}.o" "${content}")
    list(APPEND paths "${dir}/object${index}.o")
    math(EXPR index "${index} + 1")
  endforeach()
  string(JOIN "\n" listing ${paths})
  file(WRITE "${dir}/objects.txt" "${listing}\n")
endfunction()

function(epoch_of dir output)
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}" "-DOBJECTS_FILE=${dir}/objects.txt"
      "-DOUT=${dir}/jit_epoch.cpp" "-DSTAMP=${dir}/jit_epoch.stamp" -P
      "${JIT_EPOCH_SCRIPT}" COMMAND_ERROR_IS_FATAL ANY
    OUTPUT_QUIET)
  if(NOT EXISTS "${dir}/jit_epoch.stamp")
    message(FATAL_ERROR "jit_epoch.cmake did not touch its stamp")
  endif()
  file(READ "${dir}/jit_epoch.cpp" generated)
  if(NOT generated MATCHES "kJitEpoch\\[\\] = \"([0-9a-f]+)\"")
    message(FATAL_ERROR "no epoch in generated source:\n${generated}")
  endif()
  string(LENGTH "${CMAKE_MATCH_1}" length)
  if(NOT length EQUAL 32)
    message(FATAL_ERROR "epoch is not 32 hex digits: ${CMAKE_MATCH_1}")
  endif()
  set(${output}
      "${CMAKE_MATCH_1}"
      PARENT_SCOPE)
endfunction()

write_objects("${WORK_DIR}/base" "renderer" "nvrtc")
epoch_of("${WORK_DIR}/base" base)
epoch_of("${WORK_DIR}/base" rerun)
if(NOT base STREQUAL rerun)
  message(FATAL_ERROR "epoch changed without input changes")
endif()

write_objects("${WORK_DIR}/moved" "renderer" "nvrtc")
epoch_of("${WORK_DIR}/moved" moved)
if(NOT base STREQUAL moved)
  message(FATAL_ERROR "epoch depends on object paths")
endif()

write_objects("${WORK_DIR}/edited" "renderer" "nvrtC")
epoch_of("${WORK_DIR}/edited" edited)
if(base STREQUAL edited)
  message(FATAL_ERROR "epoch ignores object contents")
endif()

write_objects("${WORK_DIR}/added" "renderer" "nvrtc" "headers")
epoch_of("${WORK_DIR}/added" added)
if(base STREQUAL added)
  message(FATAL_ERROR "epoch ignores an added object")
endif()

write_objects("${WORK_DIR}/split" "render" "ernvrtc")
epoch_of("${WORK_DIR}/split" split)
if(base STREQUAL split)
  message(FATAL_ERROR "epoch ignores object boundaries")
endif()

message(STATUS "jit epoch: OK")
