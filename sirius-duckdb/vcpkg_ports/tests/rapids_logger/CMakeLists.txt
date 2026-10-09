cmake_minimum_required(VERSION 3.30.4)
project(rapids_logger_probe LANGUAGES CXX)

set(CPM_DOWNLOAD_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/reject-cpm.cmake")
file(WRITE "${CPM_DOWNLOAD_LOCATION}"
     "message(FATAL_ERROR \"Installed packages must not load CPM\")\n")

add_subdirectory(import)
add_executable(rapids_logger_probe link_probe.cpp)
target_link_libraries(rapids_logger_probe PRIVATE rapids_logger::rapids_logger)

enable_testing()
add_test(NAME rapids_logger_probe COMMAND rapids_logger_probe)
