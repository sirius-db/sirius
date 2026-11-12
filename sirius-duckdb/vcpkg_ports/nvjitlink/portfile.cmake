vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(
    FATAL_ERROR
      "VCPKG_CUDA_VERSION not set. Set the VCPKG_CUDA_VERSION environment variable to 12 or 13."
  )
endif()
set(CUDA_VERSION "${VCPKG_CUDA_VERSION}")

# The libnvjitlink redist version differs per CUDA major.
if(CUDA_VERSION STREQUAL "13")
  set(NVJITLINK_VERSION "13.5.37")
elseif(CUDA_VERSION STREQUAL "12")
  set(NVJITLINK_VERSION "12.9.86")
else()
  message(
    FATAL_ERROR "Unsupported CUDA version: ${CUDA_VERSION}. Supported: 12, 13")
endif()

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(NVJITLINK_PLATFORM "linux-x86_64")
  if(CUDA_VERSION STREQUAL "13")
    set(NVJITLINK_SHA512
        "061df3c6f57f4aab3f6066e7ab28b818115caff406e4e346931d241bd0dd690d66fed99c29bb20df7a8080538948635db3525032a07a96c6aac83e54bbabbb56"
    )
  else()
    set(NVJITLINK_SHA512
        "84ad400c98c52dc8d4370ae4c2617886dc651f5608864b5ca36adb5a5a3d2d05eb9d63871c8da668ab3d3434df906b1d93431a896dc72b5e340aecedf58bc48e"
    )
  endif()
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(NVJITLINK_PLATFORM "linux-sbsa")
  if(CUDA_VERSION STREQUAL "13")
    set(NVJITLINK_SHA512
        "5d0c54c435f6275c60d744fba0f8520e2845f1a07065f40327194996f77de9bd0c4740121eb3ac80fe9cdd70b74b5ee764850b0dbd0757c1cb8d789a595e9d4d"
    )
  else()
    set(NVJITLINK_SHA512
        "18fcbbb60edaa77e4ad02b510a70177f506a1775ccb22822c1501c349736fdd12ac5d268671c14429c1925724165a72c4198bf5bacd7231da80400ea1726388a"
    )
  endif()
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

vcpkg_download_distfile(
  ARCHIVE
  URLS
  "https://developer.download.nvidia.com/compute/cuda/redist/libnvjitlink/${NVJITLINK_PLATFORM}/libnvjitlink-${NVJITLINK_PLATFORM}-${NVJITLINK_VERSION}-archive.tar.xz"
  FILENAME
  "libnvjitlink-${NVJITLINK_PLATFORM}-${NVJITLINK_VERSION}-archive.tar.xz"
  SHA512
  ${NVJITLINK_SHA512})

vcpkg_extract_source_archive(SOURCE_PATH ARCHIVE "${ARCHIVE}")

file(INSTALL "${SOURCE_PATH}/include/nvJitLink.h"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(INSTALL "${SOURCE_PATH}/lib/libnvJitLink_static.a"
     DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/nvjitlink-config.cmake"
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/nvjitlink")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
