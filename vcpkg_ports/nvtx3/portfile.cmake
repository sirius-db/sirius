vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/NVTX
  REF
  v${VERSION}
  SHA512
  b65eb392f2fcf4a96fef84932cf61ceb85ae8a424e1128e77adde1e0452291d5e3eacd8bc0354f8878551ede0070dc0881406a4ffac1f3b13d2f7e56f4e0c41a
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
set(nvtx3_VERSION "${VERSION}")
set(nvtx3_FOUND TRUE)
]])

# Create version file for find_package version checking
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvtx3/nvtx3-config-version.cmake"
  [[
set(PACKAGE_VERSION "${VERSION}")

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
