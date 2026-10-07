vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

set(sirius_url https://github.com/mbrobbel/sirius.git)
set(sirius_ref 72c326f0e2c813fe5a10f014564fbc6d36b5fb7f)
vcpkg_from_git(OUT_SOURCE_PATH SOURCE_PATH URL "${sirius_url}" REF
               "${sirius_ref}")
vcpkg_from_git(
  OUT_SOURCE_PATH DUCKDB_SOURCE_PATH URL
  https://github.com/sirius-db/duckdb.git REF
  069cc9f9b5be802405797faecc284961b07c70ef)
vcpkg_from_git(
  OUT_SOURCE_PATH CUCASCADE_SOURCE_PATH URL
  https://github.com/NVIDIA/cuCascade.git REF
  bd4aeedbee1201df68bd042374fd9ab6afb5b46d)
vcpkg_from_git(
  OUT_SOURCE_PATH SUBSTRAIT_SOURCE_PATH URL
  https://github.com/sirius-db/duckdb-substrait-extension.git REF
  a7e045befd5479569f0a84b241868253cdfde0b3)
vcpkg_from_git(
  OUT_SOURCE_PATH CORROSION_SOURCE_PATH URL
  https://github.com/corrosion-rs/corrosion.git REF
  1499b14e4906a2890f5cee1547c8848db261753d)
# Release archives do not contain Git submodule contents.
file(COPY "${CUCASCADE_SOURCE_PATH}/" DESTINATION "${SOURCE_PATH}/cucascade")
file(COPY "${SUBSTRAIT_SOURCE_PATH}/" DESTINATION "${SOURCE_PATH}/substrait")

find_program(SIRIUS_SCCACHE sccache)
set(sirius_launchers)
if(SIRIUS_SCCACHE)
  foreach(language C CXX CUDA)
    list(APPEND sirius_launchers
         "-DCMAKE_${language}_COMPILER_LAUNCHER=${SIRIUS_SCCACHE}")
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
  "-DSIRIUS_DUCKDB_SOURCE_DIR=${DUCKDB_SOURCE_PATH}"
  "-DFETCHCONTENT_SOURCE_DIR_CORROSION=${CORROSION_SOURCE_PATH}"
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
  ${sirius_launchers})
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME sirius CONFIG_PATH lib/cmake/sirius)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
     "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
