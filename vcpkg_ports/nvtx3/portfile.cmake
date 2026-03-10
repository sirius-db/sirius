vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/NVTX
  REF
  69c9949150ac1c310758a304082228a36d5e4758
  SHA512
  356b855b785d2b18a0a671e6582439dd8f9e49dfbda9e81053cb33cc3fbe70699f044fa78107d83f3f2e2a9ced6605c6fcfd8ed1d4a4e76da9295b721acbaf3c
  HEAD_REF
  release-v3)

# nvtx3 is header-only, just install the headers
file(
  INSTALL "${SOURCE_PATH}/c/include/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include"
  FILES_MATCHING
  PATTERN "*.h")
file(
  INSTALL "${SOURCE_PATH}/c/include/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include"
  FILES_MATCHING
  PATTERN "*.hpp")

# Create CMake config files for find_package support
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvtx3/nvtx3-config.cmake"
  [[
include(CMakeFindDependencyMacro)

if(NOT TARGET nvtx3::nvtx3)
    add_library(nvtx3::nvtx3 INTERFACE IMPORTED)
    set_target_properties(nvtx3::nvtx3 PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include"
    )
endif()

# Create the nvtx3-cpp target that rmm expects
if(NOT TARGET nvtx3::nvtx3-cpp)
    add_library(nvtx3::nvtx3-cpp INTERFACE IMPORTED)
    set_target_properties(nvtx3::nvtx3-cpp PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../include"
    )
endif()

# Set version information
set(nvtx3_VERSION "3.2.0")
set(nvtx3_FOUND TRUE)
]])

# Create version file for find_package version checking
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvtx3/nvtx3-config-version.cmake"
  [[
set(PACKAGE_VERSION "3.2.0")

if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
]])

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
