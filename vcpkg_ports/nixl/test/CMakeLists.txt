cmake_minimum_required(VERSION 3.24)
project(nixl_static_probe LANGUAGES CXX)

find_package(nixl 1.5 CONFIG REQUIRED)
add_executable(nixl_static_probe static_nixl_probe.cpp)
target_link_libraries(nixl_static_probe PRIVATE nixl::nixl)

add_library(nixl_shared_link_probe SHARED shared_link_probe.cpp)
target_link_libraries(nixl_shared_link_probe PRIVATE nixl::nixl)
target_link_options(nixl_shared_link_probe PRIVATE -fuse-ld=bfd
                    -Wl,--no-undefined -Wl,--exclude-libs,ALL)

enable_testing()
foreach(probe IN ITEMS nixl_static_probe nixl_shared_link_probe)
  add_test(NAME ${probe}_linkage
           COMMAND "${CMAKE_COMMAND}" -DPROBE=$<TARGET_FILE:${probe}> -P
                   "${CMAKE_CURRENT_LIST_DIR}/check_linkage.cmake")
endforeach()
add_test(NAME cuda_tcp COMMAND nixl_static_probe)
set_tests_properties(cuda_tcp PROPERTIES ENVIRONMENT "UCX_TLS=tcp,cuda_copy"
                                         TIMEOUT 20)
add_test(NAME cuda_shared_memory_with_tcp COMMAND nixl_static_probe)
set_tests_properties(
  cuda_shared_memory_with_tcp PROPERTIES ENVIRONMENT "UCX_TLS=sm,tcp,cuda_copy"
                                         TIMEOUT 20)
