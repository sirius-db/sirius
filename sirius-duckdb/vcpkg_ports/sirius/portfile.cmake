vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

get_filename_component(SOURCE_PATH "${CURRENT_PORT_DIR}/../../.." ABSOLUTE)

# vcpkg disables FetchContent downloads during configuration.
vcpkg_from_git(
  OUT_SOURCE_PATH CORROSION_SOURCE_PATH URL
  https://github.com/corrosion-rs/corrosion.git REF
  1499b14e4906a2890f5cee1547c8848db261753d)
# Same commit as the onpair_cpp FetchContent in
# src/compression/simpatico_codegen.
vcpkg_from_github(
  OUT_SOURCE_PATH
  ONPAIR_CPP_SOURCE_PATH
  REPO
  gargiulofrancesco/onpair_cpp
  REF
  ae590713515c7bb7893e14a757b484545e5339c3
  SHA512
  0481833df8e48785d8a1036432d14771714b097725c0b115c6a58a219d91154495d2383a92b4064824160b53d3a2bb62a0edb40d6080b46ece4c48f6f44d2259
)

find_program(SIRIUS_COMPILER_CACHE NAMES sccache ccache)
set(sirius_launchers)
if(SIRIUS_COMPILER_CACHE)
  foreach(language C CXX CUDA)
    list(APPEND sirius_launchers
         "-DCMAKE_${language}_COMPILER_LAUNCHER=${SIRIUS_COMPILER_CACHE}")
  endforeach()
endif()

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DVCPKG_BUILD=ON
  -DCPM_LOCAL_PACKAGES_ONLY=ON
  -DSIRIUS_BUILD_SHARED=OFF
  -DSIRIUS_BUILD_STATIC=ON
  -DSIRIUS_BUILD_TESTS=OFF
  "-DVCPKG_HOST_TRIPLET=${HOST_TRIPLET}"
  "-DFETCHCONTENT_SOURCE_DIR_CORROSION=${CORROSION_SOURCE_PATH}"
  "-DFETCHCONTENT_SOURCE_DIR_ONPAIR_CPP_SRC=${ONPAIR_CPP_SOURCE_PATH}"
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
  ${sirius_launchers})
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME sirius CONFIG_PATH lib/cmake/sirius)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
     "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
