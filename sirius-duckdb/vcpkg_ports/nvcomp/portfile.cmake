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
        "fcfc3702723255a541c9e253df0f4321ed6c82a7455d6df278abba3275f19c05226b779556c9021e6f873c40fb6a869a23fd2d5465eb572772490e745640e059"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "2f15a892bdc75b2fe7a9fe7b2b739b2968fb769928837d5bce879d2c8bef65c58895e5c4219777d78f08083f15b7d743c06b68fc33cf2d059fd0d773fba32614"
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
        "1373059689917b44146811c76b66941e9057d19affa80d81f9488eccb526da8e1712cd3a1874215c66817b2125e51e7f546a5838c9d330ce1d57fa1a9670ec7c"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "02727d9fc502bea346c2e70a98769e92fa70ad9e4dcd685e0bafcea6b1901142a2b08ae4be957f3078df779e4cc1be16450e392873b11baafb2a2818c7eb9c96"
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

# Keep nvCOMP's bundled logger ABI private to its static archives.
vcpkg_cmake_get_vars(NVCOMP_CMAKE_VARS)
include("${NVCOMP_CMAKE_VARS}")
include("${CMAKE_CURRENT_LIST_DIR}/isolate-logging.cmake")
file(GLOB NVCOMP_ARCHIVES "${CURRENT_PACKAGES_DIR}/lib/*.a")
nvcomp_isolate_logging(
  "${VCPKG_DETECTED_CMAKE_NM}" "${VCPKG_DETECTED_CMAKE_OBJCOPY}"
  "${CURRENT_BUILDTREES_DIR}/logging-symbols.txt" ${NVCOMP_ARCHIVES})

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
