vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

set(NVCOMP_VERSION "${VERSION}")

# CUDA version from triplet (set via VCPKG_CUDA_VERSION env var)
if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(
    FATAL_ERROR
      "VCPKG_CUDA_VERSION not set. Set the VCPKG_CUDA_VERSION environment variable to 12 or 13."
  )
endif()
set(CUDA_VERSION "${VCPKG_CUDA_VERSION}")

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(NVCOMP_PLATFORM "linux-x86_64")
  if(CUDA_VERSION STREQUAL "12")
    set(NVCOMP_SHA512
        "376ecb4e17ab1e345f1f42168691f5471bab9bca95bebd35f0c6fb91e10172d42317f073e9f6a914ed3a9d01c0318755e3c1e36c920802f8047dad9c72ea3092"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "329f773003e2413b21bce65236bed8fbbeb62db3da4480c732af6769ea8a1901c893e150f7fc03dbce3a502ec290ccb8073cb379d50dabffd1caa0863ed21a8f"
    )
  else()
    message(
      FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13"
    )
  endif()
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NVCOMP_PLATFORM "linux-sbsa")
  if(CUDA_VERSION STREQUAL "12")
    set(NVCOMP_SHA512
        "8512fa10efb3eedf614ab68f176d08ff6cd167b2769812c042246c7e4b7699bdf82d4f4d701c983e326635aed4bc6e93f0202b5c4a215d1207656b30cc801f31"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "fbbaacf598a8051cbb9285369fd96056cf7971ebf53698b98e6d3baa8a7d8340ab5c6fe39b02ecd5c941d77dd12dafe889ef28f0ae7198014b282d4f54d5d26b"
    )
  else()
    message(
      FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13"
    )
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/nvcomp/redist/nvcomp/${NVCOMP_PLATFORM}/nvcomp-${NVCOMP_PLATFORM}-${NVCOMP_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"
  FILENAME
  "nvcomp-${NVCOMP_PLATFORM}-${NVCOMP_VERSION}_cuda${CUDA_VERSION}-archive.tar.xz"
  SHA512
  ${NVCOMP_SHA512})

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

# Install headers
file(GLOB HEADER_FILES "${SOURCE_PATH}/include/*")
file(INSTALL ${HEADER_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Install libraries
file(GLOB LIB_FILES "${SOURCE_PATH}/lib/*.a")
file(INSTALL ${LIB_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")

# Install CMake config files (targets only, we'll write a custom config.cmake)
file(INSTALL "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-config-version.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")
file(INSTALL "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-targets-static.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")
file(INSTALL
     "${SOURCE_PATH}/lib/cmake/nvcomp/nvcomp-targets-static-release.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvcomp")

# Write a custom config file that works with vcpkg layout
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/nvcomp/nvcomp-config.cmake"
  "
get_filename_component(PACKAGE_PREFIX_DIR \"\${CMAKE_CURRENT_LIST_DIR}/../../\" ABSOLUTE)

set(nvcomp_VERSION ${VERSION})
set(nvcomp_INCLUDE_DIR \"\${PACKAGE_PREFIX_DIR}/include\")
set(nvcomp_LIBRARY_DIR \"\${PACKAGE_PREFIX_DIR}/lib\")

# Check headers and library directories exist
if(NOT EXISTS \"\${nvcomp_INCLUDE_DIR}/nvcomp.h\")
    message(FATAL_ERROR \"nvcomp headers not found at \${nvcomp_INCLUDE_DIR}\")
endif()

# Load the target definitions
include(\"\${CMAKE_CURRENT_LIST_DIR}/nvcomp-targets-static.cmake\")

# Create alias for compatibility with downstream projects
if(TARGET nvcomp::nvcomp_static AND NOT TARGET nvcomp::nvcomp)
    add_library(nvcomp::nvcomp ALIAS nvcomp::nvcomp_static)
endif()

set(nvcomp_FOUND TRUE)
")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
