vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  NVIDIA/cccl
  REF
  # Match the CCCL commit pinned by rapids-cmake 26.10.
  b98d4bd6ff0715be42b7cfc78dfa9c4e212c5c5a
  SHA512
  c1f194125145a9ae76e1cd8468eacdce3968b58c4f9bcad612637725eadd5cd6173716986d81bcee191d6ddae6d4f8ad7b3e4e9e514b4860cd0874b7a775d3f3
  HEAD_REF
  main)

# CCCL is header-only, install all headers (including extension-less C++
# standard headers)
file(
  INSTALL "${SOURCE_PATH}/thrust/thrust/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/thrust"
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.inl")
file(
  INSTALL "${SOURCE_PATH}/cub/cub/"
  DESTINATION "${CURRENT_PACKAGES_DIR}/include/cub"
  FILES_MATCHING
  PATTERN "*.cuh")
# libcudacxx has both .h/.hpp headers AND extension-less C++ standard headers
# (like climits, cstdint, etc.)
file(COPY "${SOURCE_PATH}/libcudacxx/include/"
     DESTINATION "${CURRENT_PACKAGES_DIR}/include")

# Install CMake config files
file(GLOB CCCL_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/cccl/*")
file(INSTALL ${CCCL_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/cccl")
file(GLOB THRUST_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/thrust/*")
file(INSTALL ${THRUST_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/thrust")
file(GLOB CUB_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/cub/*")
file(INSTALL ${CUB_CMAKE_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/share/cub")
file(GLOB LIBCUDACXX_CMAKE_FILES "${SOURCE_PATH}/lib/cmake/libcudacxx/*")
file(INSTALL ${LIBCUDACXX_CMAKE_FILES}
     DESTINATION "${CURRENT_PACKAGES_DIR}/share/libcudacxx")

# Fix header-search.cmake files: upstream configs expect headers relative to the
# CCCL source tree (e.g. ../../../libcudacxx/include). In vcpkg, all headers are
# installed to ${prefix}/include, so fix the paths.
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/libcudacxx/libcudacxx-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../libcudacxx/include]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/thrust/thrust-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../thrust]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cub/cub-header-search.cmake"
  [[${CMAKE_CURRENT_LIST_DIR}/../../../cub]]
  [[${CMAKE_CURRENT_LIST_DIR}/../../include]])

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
