vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  apache/arrow-nanoarrow
  REF
  "apache-arrow-nanoarrow-${VERSION}"
  SHA512
  85a8fad09bfc9dfadaac1d085a900376bf37b216a243ecf3e5b0c13939ef3cbcdb510cccedc87bdca6ef8cebbc7ac60cb220ae160a4abf478227c19155991b56
  HEAD_REF
  main)

file(REMOVE_RECURSE "${SOURCE_PATH}/thirdparty")

string(COMPARE EQUAL ${VCPKG_LIBRARY_LINKAGE} "dynamic"
               NANOARROW_INSTALL_SHARED)

vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}"
  OPTIONS
  -DNANOARROW_INSTALL_SHARED=${NANOARROW_INSTALL_SHARED}
  -DNANOARROW_DEBUG_EXTRA_WARNINGS=OFF
  -DNANOARROW_DEVICE=ON
  -DNANOARROW_DEVICE_WITH_CUDA=OFF
  -DNANOARROW_DEVICE_WITH_METAL=OFF)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(PACKAGE_NAME nanoarrow CONFIG_PATH lib/cmake/nanoarrow)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/lib/cmake"
     "${CURRENT_PACKAGES_DIR}/lib/cmake")

# Fix nanoarrow-config.cmake to guard against duplicate target creation when the
# config file is included multiple times (e.g., from both sirius and cudf)
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/nanoarrow/nanoarrow-config.cmake"
  [[foreach(target nanoarrow nanoarrow_ipc nanoarrow_device nanoarrow_testing)
  if(TARGET nanoarrow::${target}_static)
    if(BUILD_SHARED_LIBS)
      add_library(nanoarrow::${target} ALIAS nanoarrow::${target}_shared)
    else()
      add_library(nanoarrow::${target} ALIAS nanoarrow::${target}_static)
    endif()
  endif()
endforeach()]]
  [[foreach(target nanoarrow nanoarrow_ipc nanoarrow_device nanoarrow_testing)
  if(TARGET nanoarrow::${target}_static AND NOT TARGET nanoarrow::${target})
    if(BUILD_SHARED_LIBS)
      add_library(nanoarrow::${target} ALIAS nanoarrow::${target}_shared)
    else()
      add_library(nanoarrow::${target} ALIAS nanoarrow::${target}_static)
    endif()
  endif()
endforeach()]])

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
