vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

# cuVS is consumed as a prebuilt static library from the RAPIDS conda channel
# instead of being compiled from source. Building libcuvs from source in the
# distribution CI took roughly 50 to 75 minutes per matrix leg, because it
# compiles the full cuVS kernel set across every GPU arch, even though Sirius
# only uses brute_force and distance. The prebuilt libcuvs-static package
# already carries the same all-arch archive, so here we just download and
# install it.
#
# Two conda packages together provide what we need: 1. libcuvs-static   gives
# lib/libcuvs_static.a and the static-target cmake files 2. libcuvs-headers
# gives include/cuvs/** and the header-target cmake file They don't provide
# cuvs-config.cmake (RAPIDS keeps that in the shared libcuvs package), so we
# write one down below.

set(CUVS_PKG_VERSION "26.06.00")
# Suffix of the conda build string for this exact version, shared by every
# arch/cuda variant: libcuvs-<comp>-26.06.00-cuda<major>_<CUVS_BUILD_TAG>.conda
set(CUVS_BUILD_TAG "260604_2bd7cd71")

# CUDA major comes from the triplet via VCPKG_CUDA_VERSION
if(NOT DEFINED VCPKG_CUDA_VERSION)
  message(FATAL_ERROR "VCPKG_CUDA_VERSION not set. Set it to 12 or 13.")
endif()
if(NOT VCPKG_CUDA_VERSION STREQUAL "12" AND NOT VCPKG_CUDA_VERSION STREQUAL
                                            "13")
  message(
    FATAL_ERROR
      "Unsupported CUDA version: ${VCPKG_CUDA_VERSION}. Supported: 12, 13.")
endif()
set(CUDA_MAJOR "${VCPKG_CUDA_VERSION}")

if(VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(CONDA_SUBDIR "linux-64")
elseif(VCPKG_TARGET_ARCHITECTURE STREQUAL "arm64")
  set(CONDA_SUBDIR "linux-aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${VCPKG_TARGET_ARCHITECTURE}")
endif()

if(CONDA_SUBDIR STREQUAL "linux-64" AND CUDA_MAJOR STREQUAL "12")
  set(CUVS_STATIC_SHA512
      "ab23e644b9d3767fc597907ba1bfd8a2bae98da94e3fae8bfaf60d7ec842dfbb057b03ecc71ea17d802c63c7e17d5091bb77a1ff07bdf26e0db836fdaf02285d"
  )
  set(CUVS_HEADERS_SHA512
      "ee21f1608253892213c029df0dc50ee68fb29ddee828bc83b2374349719ae396e8f1e95d159f8acd221157c3179cf6cb4f8f5900595c0187588c59c51ca59474"
  )
elseif(CONDA_SUBDIR STREQUAL "linux-64" AND CUDA_MAJOR STREQUAL "13")
  set(CUVS_STATIC_SHA512
      "c05738f5bc9ca7dfa4fcfaa02cf2c56ed6421dfa4d79ba4d8fd2a879cb322422eed29127d6b07bf85383c1fc775a0c93a45043158490c632911e79dc54cd4dc5"
  )
  set(CUVS_HEADERS_SHA512
      "7ba4ec90c312ff4b70311b900d018449802ed1960280a03b295c748efec0ff7ea66c6d81bc0116a20d415d0cb92b7adbaa219ab6bb91e7a9d71551a019e5dd89"
  )
elseif(CONDA_SUBDIR STREQUAL "linux-aarch64" AND CUDA_MAJOR STREQUAL "12")
  set(CUVS_STATIC_SHA512
      "e118b5b49749edc1d64de272ad7ae52723164cb84b4dfeed9528d8fd50f26a040bed2d233bece2dd826d789aaacff969bc0521a4b384aac0ccc0d1579db617fd"
  )
  set(CUVS_HEADERS_SHA512
      "74d269c04e0f11632306191c076fcf7fe19b17f6ef08ebfec964e40ab7c52eff65adb1b8f13e9ce81a50a5709a27db04d9a02f6f2ff2a0c7dbb2acd853411ed2"
  )
elseif(CONDA_SUBDIR STREQUAL "linux-aarch64" AND CUDA_MAJOR STREQUAL "13")
  set(CUVS_STATIC_SHA512
      "54a62eb1a02e3f397cc16de64b6f3c75268c22de41e437341f178bab9f0d361bfc30bf9bcdd8baf2ff62dac0bee7693a2f583e3c58448f98d3c6cde02a181b48"
  )
  set(CUVS_HEADERS_SHA512
      "84569dc5314c4bf0f1d14cfdba9b905d0eb7d25d354e3b1a93196d812f5b7c6a60aa882e1aca74e224c3582c2449de3479c9ea83ad505cf9831c5c4d25c821e4"
  )
endif()

set(CUVS_STAGE "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-stage")
file(REMOVE_RECURSE "${CUVS_STAGE}")
file(MAKE_DIRECTORY "${CUVS_STAGE}")

function(cuvs_download_conda component sha)
  set(filename
      "libcuvs-${component}-${CUVS_PKG_VERSION}-cuda${CUDA_MAJOR}_${CUVS_BUILD_TAG}.conda"
  )
  vcpkg_download_distfile(
    archive
    URLS
    "https://conda.anaconda.org/rapidsai/${CONDA_SUBDIR}/${filename}"
    FILENAME
    "${filename}"
    SHA512
    "${sha}")
  set(unzip_dir
      "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-${component}-conda")
  file(REMOVE_RECURSE "${unzip_dir}")
  file(MAKE_DIRECTORY "${unzip_dir}")
  file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${unzip_dir}")
  file(GLOB payload "${unzip_dir}/pkg-*.tar.zst")
  file(ARCHIVE_EXTRACT INPUT "${payload}" DESTINATION "${CUVS_STAGE}")
endfunction()

cuvs_download_conda("static" "${CUVS_STATIC_SHA512}")
cuvs_download_conda("headers" "${CUVS_HEADERS_SHA512}")

file(INSTALL "${CUVS_STAGE}/lib/libcuvs_static.a"
     DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
file(INSTALL "${CUVS_STAGE}/include/"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")
file(
  INSTALL "${CUVS_STAGE}/lib/cmake/cuvs/cuvs-cuvs_static-static-targets.cmake"
  "${CUVS_STAGE}/lib/cmake/cuvs/cuvs-cuvs_static-static-targets-release.cmake"
  "${CUVS_STAGE}/lib/cmake/cuvs/cuvs-cuvs_cpp_headers-cpp-headers-targets.cmake"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/cuvs")

vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-cuvs_static-static-targets.cmake"
  [[get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)]]
  [[get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)]])
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-cuvs_cpp_headers-cpp-headers-targets.cmake"
  [[get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)]]
  [[get_filename_component(_IMPORT_PREFIX "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)
get_filename_component(_IMPORT_PREFIX "${_IMPORT_PREFIX}" PATH)]])

# cuvs-config.cmake
file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-config.cmake"
  [[include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit)
find_dependency(raft)
find_dependency(rmm)

include("${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_cpp_headers-cpp-headers-targets.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/cuvs-cuvs_static-static-targets.cmake")

set(cuvs_FOUND TRUE)
]])

file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/cuvs/cuvs-config-version.cmake"
  [[set(PACKAGE_VERSION "26.06.00")
if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
  set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
  if(PACKAGE_VERSION VERSION_EQUAL PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_EXACT TRUE)
  endif()
endif()
]])

file(
  WRITE "${CURRENT_PACKAGES_DIR}/share/cuvs/copyright"
  "cuVS is distributed under the Apache License 2.0.\nSee https://github.com/rapidsai/cuvs/blob/v26.06.00/LICENSE\n"
)
