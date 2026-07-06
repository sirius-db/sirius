vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cudf
  REF
  v${VERSION}
  SHA512
  630c7ec1195b0aaefd738715379981951fb43575e22ef50d35ec3645a3f7e2f66fa0feb2ad72e6164dfbfce8515205616a4da9d633fb8b7864acb727e4d60240
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v${VERSION}
  SHA512
  d3d7a1f807a9b71ed15c972742a4dbee0746cc65b1bfa7eef9a8e036a992a37fcfdfbff79fc27cf053dc5a37978abf86b93b56bc6f605f04244e8f6776595bdd
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.2.3
  SHA512
  eb7b5ebf6289d10307b8a34d9d1469ffcb63e9371e9dd5ccbda0351923b920ebae8220ceaa8d1d52c9bed57200f35921a6365a5f9a25a209a98314f75195310c
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  JITIFY_PATH
  REPO
  NVIDIA/jitify
  REF
  44e978b21fc8bdb6b2d7d8d179523c8350db72e5
  SHA512
  c6a175ae6ebae066285f1d662f8a7f73ea595fa17cf1ae7c66261899f5458e0c674eb5d546c404b8840cd1a2e760d72903b7bf6f5a48d32b13ebb5325256a2c4
  HEAD_REF
  master)

vcpkg_from_github(
  OUT_SOURCE_PATH
  BS_THREAD_POOL_PATH
  REPO
  bshoshany/thread-pool
  REF
  v4.1.0
  SHA512
  4908f00def23082e7ddc0b24a710e53b3fde51b02188e79cfcd9dabb22627ebd1b6e5b3c4bf1b366eae79660c26878cc034c171747c3d0b7ef8a98c85a77033b
  HEAD_REF
  master)

# vcpkg sets FETCHCONTENT_FULLY_DISCONNECTED=ON during port builds, which
# prevents CPM from downloading any sources. We must provide all CPM
# dependencies as local sources. zstd: cudf's get_zstd.cmake forces download via
# CPM_DOWNLOAD_zstd=ON
vcpkg_from_github(
  OUT_SOURCE_PATH
  ZSTD_PATH
  REPO
  facebook/zstd
  REF
  v1.5.7
  SHA512
  26e441267305f6e58080460f96ab98645219a90d290a533410b1b0b1d2f870721c95f8384e342ee647c5e968385a5b7e30c2d04340c37f59b3e6d86762c3260c
  HEAD_REF
  dev)

# cuco: rapids-cmake always_download=true forces download
vcpkg_from_github(
  OUT_SOURCE_PATH
  CUCO_PATH
  REPO
  NVIDIA/cuCollections
  REF
  f517bbb1277753b1852dfd388993383e401eaa38
  SHA512
  53f6185db57eba7391fd9b93b342d9da3c4cdf906dc47fad02cc1eb34867bb11ce798f0e94fee71a3cfb6423e6cad907ebe21e46997b4f4ef2b6e1e3b0a45d82
  HEAD_REF
  dev)

# Patch get_zstd.cmake to remove forced download - we provide zstd source via
# CPM variable
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_zstd.cmake"
  "set(CPM_DOWNLOAD_zstd ON)"
  "# CPM_DOWNLOAD_zstd removed - using CPM_zstd_SOURCE instead")

# Patch rapids_logger to use vcpkg's spdlog::spdlog target instead of spdlog
vcpkg_replace_string(
  "${RAPIDS_LOGGER_PATH}/CMakeLists.txt"
  "set_target_properties(spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "set_target_properties(spdlog::spdlog PROPERTIES POSITION_INDEPENDENT_CODE ON)"
)

# Patch nanoarrow - vcpkg's nanoarrow_static is an ALIAS target, can't set
# properties on it
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_nanoarrow.cmake"
  "set_target_properties(nanoarrow_static PROPERTIES POSITION_INDEPENDENT_CODE ON)"
  "# set_target_properties disabled for vcpkg ALIAS target")

# Use -I for vcpkg include dir to ensure vcpkg CCCL headers beat pixi's older
# CCCL. CUDF_BUILD_TESTUTIL=OFF disables the test-utility targets (the only
# thing that pulls GTest), so no GTest source needs to be provided.
vcpkg_cmake_configure(
  SOURCE_PATH
  "${SOURCE_PATH}/cpp"
  OPTIONS
  -DFETCHCONTENT_SOURCE_DIR_RAPIDS-CMAKE=${RAPIDS_CMAKE_PATH}
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DCPM_jitify_SOURCE=${JITIFY_PATH}
  -DCPM_bs_thread_pool_SOURCE=${BS_THREAD_POOL_PATH}
  -DCPM_zstd_SOURCE=${ZSTD_PATH}
  -DCPM_cuco_SOURCE=${CUCO_PATH}
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  -DBUILD_SHARED_LIBS=OFF
  -DCUDA_STATIC_RUNTIME=ON
  -DBUILD_TESTS=OFF
  -DCUDF_BUILD_TESTUTIL=OFF
  -DBUILD_BENCHMARKS=OFF
  -DCUDF_KVIKIO_REMOTE_IO=OFF
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -Wno-error=sign-compare -Wno-error=parentheses"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -Xcompiler=-Wno-error=sign-compare -Xcompiler=-Wno-error=parentheses"
)

vcpkg_cmake_install()

# Remove CPM-installed zstd files that conflict with standalone zstd:x64-linux
# port
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zdict.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zstd.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/include/zstd_errors.h")
file(REMOVE "${CURRENT_PACKAGES_DIR}/lib/libzstd.a")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")

# cudf 26.06 fully vendors cuco: it no longer installs lib/cmake/cuco, and
# cudf-dependencies.cmake no longer calls find_dependency(cuco), so the old
# move-to-share step (needed for 26.04) is obsolete and would fail here.

vcpkg_cmake_config_fixup(PACKAGE_NAME cudf CONFIG_PATH lib/cmake/cudf)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# Fix cudf-dependencies.cmake to skip ALIAS targets when setting
# IMPORTED_GLOBAL. ALIAS targets like CCCL::CUB, CCCL::libcudacxx don't support
# set_target_properties.
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cudf/cudf-dependencies.cmake"
  [[foreach(target IN LISTS rapids_global_targets)
  if(TARGET ${target})
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()]]
  [[foreach(target IN LISTS rapids_global_targets)
  if(TARGET ${target})
    get_target_property(_aliased ${target} ALIASED_TARGET)
    if(_aliased)
      # Skip ALIAS targets - can't set properties on them
      continue()
    endif()
    get_target_property(_is_imported ${target} IMPORTED)
    get_target_property(_already_global ${target} IMPORTED_GLOBAL)
    if(_is_imported AND NOT _already_global)
        set_target_properties(${target} PROPERTIES IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()]])

# Remove rapids_logger files that conflict with rmm (rmm already provides them)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/include/rapids_logger")
file(REMOVE "${CURRENT_PACKAGES_DIR}/lib/librapids_logger.a")
file(REMOVE "${CURRENT_PACKAGES_DIR}/debug/lib/librapids_logger.a")

# Fix cudf-targets.cmake: remove conda_env from link libraries and its target
# definition. rapids-cmake creates a conda_env target with hardcoded pixi/conda
# paths that aren't portable.
vcpkg_replace_string("${CURRENT_PACKAGES_DIR}/share/cudf/cudf-targets.cmake"
                     ";\$<LINK_ONLY:cudf::conda_env>" "")

# cudf 26.06 exports the bare target name `nanoarrow_static` in its link
# interface, but the vcpkg nanoarrow port provides the namespaced imported
# target `nanoarrow::nanoarrow_static`. Without this, consumers get an
# unresolved `-lnanoarrow_static` and the link fails. Point it at the namespaced
# target (carries IMPORTED_LOCATION for libnanoarrow_static.a).
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cudf/cudf-targets.cmake"
  "\$<LINK_ONLY:nanoarrow_static>" "\$<LINK_ONLY:nanoarrow::nanoarrow_static>")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
