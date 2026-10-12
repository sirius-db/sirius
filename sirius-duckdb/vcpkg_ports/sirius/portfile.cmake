vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

get_filename_component(SOURCE_PATH "${CURRENT_PORT_DIR}/../../.." ABSOLUTE)

# vcpkg disables FetchContent downloads during configuration.
vcpkg_from_git(
  OUT_SOURCE_PATH CORROSION_SOURCE_PATH URL
  https://github.com/corrosion-rs/corrosion.git REF
  1499b14e4906a2890f5cee1547c8848db261753d)

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
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
  ${sirius_launchers})
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME sirius CONFIG_PATH lib/cmake/sirius)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
     "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
