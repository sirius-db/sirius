# The installed and source layouts must embed the same logical headers, even
# when CUB, Thrust, and libcudacxx have separate include directories.
cmake_minimum_required(VERSION 3.24)

set(headers cub/version.cuh thrust/version.h cuda/std/cstdint
            cub/block/block_reduce.cuh cub/detail/helper.cuh)
foreach(layout installed source)
  set(include_dirs "")
  foreach(header IN LISTS headers)
    if(layout STREQUAL "installed")
      set(root "${TEST_DIR}/installed/include")
    elseif(header MATCHES "^cuda/")
      set(root "${TEST_DIR}/source/libcudacxx/include")
    else()
      string(REGEX REPLACE "/.*" "" component "${header}")
      set(root "${TEST_DIR}/source/${component}")
    endif()
    list(APPEND include_dirs "${root}")
    if(header STREQUAL "cub/block/block_reduce.cuh")
      set(content
          "#include <cuda/std/cstdint>\n#include <thrust/version.h>\n#include \"../detail/helper.cuh\"\n"
      )
    else()
      set(content "// ${header}\n")
    endif()
    file(WRITE "${root}/${header}" "${content}")
  endforeach()
  list(REMOVE_DUPLICATES include_dirs)
  list(JOIN include_dirs "\n" include_dirs_text)
  file(WRITE "${TEST_DIR}/${layout}-includes.txt" "${include_dirs_text}\n")
  execute_process(
    COMMAND
      "${CMAKE_COMMAND}"
      "-DINCLUDE_DIRS_FILE=${TEST_DIR}/${layout}-includes.txt"
      "-DOUT=${TEST_DIR}/${layout}.cpp" "-DDEPFILE=${TEST_DIR}/${layout}.d" -P
      "${EMBED_SCRIPT}"
    RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Embedding ${layout} layout failed")
  endif()
endforeach()

file(READ "${TEST_DIR}/installed.cpp" installed)
file(READ "${TEST_DIR}/source.cpp" source)
if(NOT installed STREQUAL source)
  message(FATAL_ERROR "Header layout changed the embedded contents/fingerprint")
endif()
foreach(header cuda/std/cstdint thrust/version.h cub/detail/helper.cuh)
  string(FIND "${source}" "\"${header}\"" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Missing transitive header: ${header}")
  endif()
endforeach()

# Dependencies must refer to the physical files in each component root.
file(READ "${TEST_DIR}/source.d" dependencies)
foreach(path cub/cub/detail/helper.cuh thrust/thrust/version.h
             libcudacxx/include/cuda/std/cstdint)
  string(FIND "${dependencies}" "/source/${path}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR "Missing source-layout dependency: ${path}")
  endif()
endforeach()
