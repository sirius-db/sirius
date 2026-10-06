cmake_minimum_required(VERSION 3.24)
project(static_ucx_probe LANGUAGES CXX)

find_package(ucx CONFIG REQUIRED)
add_executable(static_ucx_probe static_ucx_probe.cpp)
target_compile_features(static_ucx_probe PRIVATE cxx_std_17)
target_link_libraries(static_ucx_probe PRIVATE ucx::ucx)

enable_testing()
add_test(NAME static_linkage
         COMMAND "${CMAKE_COMMAND}" -DPROBE=$<TARGET_FILE:static_ucx_probe> -P
                 "${CMAKE_CURRENT_LIST_DIR}/check_linkage.cmake")
add_test(
  NAME relocatable_metadata
  COMMAND "${CMAKE_COMMAND}" -DPCFILE=${ucx_DIR}/../../lib/pkgconfig/ucx.pc -P
          "${CMAKE_CURRENT_LIST_DIR}/check_metadata.cmake")
add_test(NAME cuda_tcp COMMAND static_ucx_probe)
set_tests_properties(cuda_tcp PROPERTIES ENVIRONMENT "UCX_TLS=tcp,cuda_copy")
add_test(NAME cuda_shared_memory COMMAND static_ucx_probe)
set_tests_properties(cuda_shared_memory PROPERTIES ENVIRONMENT
                                                   "UCX_TLS=sm,cuda_copy")
