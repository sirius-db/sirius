option(SIRIUS_ENABLE_NIXL "Build the native NIXL exchange transport" OFF)
if(NOT SIRIUS_ENABLE_NIXL)
  return()
endif()

if(VCPKG_BUILD)
  find_package(nixl 1.5.0 EXACT CONFIG REQUIRED)
else()
  include("${CMAKE_CURRENT_LIST_DIR}/sirius-nixl-build.cmake")
endif()

get_target_property(SIRIUS_NIXL_INCLUDE_DIR nixl::nixl
                    INTERFACE_INCLUDE_DIRECTORIES)
