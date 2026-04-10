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
        "dcc5ce865aae7027c98508aaf70bb0b1fc07a1b6e574088ca117b35c0e537d6a9b11e42e538a53d1c725e6c91aa0d9e0f88fb72a0162d2a284fdcb948bee3449"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "78ff0e0fc3844bbd2ac90f47cedebe2286106f129c788acf971f81d35ab4703c76b058fb4153515a826e2891a2305f03c46679760249be29a9cbe84c2223fb9f"
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
        "a733f7b8e79275e962c5e13e8b5e389ca2c6f0657b59db88d37db824bebb7b3715806ba10d7054ee74c02d0c36b3ebebaba90a225aa6b4152c002f6d1b266770"
    )
  elseif(CUDA_VERSION STREQUAL "13")
    set(NVCOMP_SHA512
        "c69c9556bfeefae2fbe31ef820dfcecf7543ff05e17c01034ce31bfb361080102674a6c30bc11556cc5663fedf98a8a5bf5e447f2dff8ac7ada98b64aa5d64cc"
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
