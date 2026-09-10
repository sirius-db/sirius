vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cudf
  REF
  v${VERSION}
  SHA512
  52572d301bdd10b218c3c9275d219f37c791a3a4fd913dc6533ddfd7df0e0fa180363793c28870ea9b5f7e1db22aedd7669a701ca07d2daa1e054bf88bdc0c85
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.08.00
  SHA512
  472e3bbc0aeedce6632c339f5a383a25524df8462a58891474523cd558b7d8d8bc09b24e8f75da6f2fbfcb83b302caa911fb5a38bc6c04871a570390e7b4a5b8
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
  RTCX_PATH
  REPO
  rapidsai/librtcx
  REF
  efad266c1fd9de6d8486c6ba71bfa74df063eb1f
  SHA512
  fb5aad3fa23d98c6ecb7308411e1b24032fd784c37c26121f32a79fcf94af086a3768c7d9b0283c21d5c6fe7658c410b8e5c775ffdaa5d1b38a59abf96e4ba51
  HEAD_REF
  main)

# librtcx's embed helper uses std::variant without including its header.
vcpkg_replace_string("${RTCX_PATH}/embed.hpp" "#include <vector>"
                     "#include <variant>\n#include <vector>")

# Use the static CUDA redistributables supplied by the overlay ports.
vcpkg_replace_string(
  "${RTCX_PATH}/CMakeLists.txt"
  "CUDA::nvrtc_static CUDA::nvrtc_builtins_static" "nvrtc::nvrtc_static")
vcpkg_replace_string("${RTCX_PATH}/CMakeLists.txt" "CUDA::nvJitLink_static"
                     "nvjitlink::nvjitlink_static")
vcpkg_replace_string(
  "${RTCX_PATH}/CMakeLists.txt"
  "add_library(rtcx STATIC hash.cpp rtcx.cpp)"
  [[rapids_find_package(nvrtc REQUIRED CONFIG BUILD_EXPORT_SET rtcx-exports INSTALL_EXPORT_SET rtcx-exports)
rapids_find_package(nvjitlink REQUIRED CONFIG BUILD_EXPORT_SET rtcx-exports INSTALL_EXPORT_SET rtcx-exports)
add_library(rtcx STATIC hash.cpp rtcx.cpp)]])

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
  0883368d39296f3bef3a058033141bcc642c5c54
  SHA512
  be8040f9ad46a2f8cc9d9c53a2bdbd0c9ff1e83c3f93788ae5508bc9f4610028e2d326e6b357a41a65a93064960db8d18963a540f25609fe2f254eb231484aee
  HEAD_REF
  dev)

# Patch get_zstd.cmake to remove forced download - we provide zstd source via
# CPM variable
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_zstd.cmake"
  "set(CPM_DOWNLOAD_zstd ON)"
  "# CPM_DOWNLOAD_zstd removed - using CPM_zstd_SOURCE instead")

# Honor explicit static CUDA options without forcing downloads of vcpkg deps.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/CMakeLists.txt"
  [[else()
  set(CUDA_STATIC_RUNTIME
      OFF
      CACHE BOOL "Statically link the CUDA runtime" FORCE
  )
  set(RTCX_STATIC_LINK_NVRTC
      OFF
      CACHE BOOL "Use static linking for NVRTC" FORCE
  )
  set(RTCX_STATIC_LINK_NVJITLINK
      OFF
      CACHE BOOL "Use static linking for nvJitLink" FORCE
  )
endif()]]
  [[endif()]])

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
  -DCPM_rtcx_SOURCE=${RTCX_PATH}
  -DRTCX_STATIC_LINK_NVRTC=ON
  -DRTCX_STATIC_LINK_NVJITLINK=ON
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
  "-DCMAKE_CXX_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER -Wno-error=sign-compare -Wno-error=parentheses"
  "-DCMAKE_CUDA_FLAGS=-I${CURRENT_INSTALLED_DIR}/include -DCCCL_IGNORE_DEPRECATED_STREAM_REF_HEADER -Xcompiler=-Wno-error=sign-compare -Xcompiler=-Wno-error=parentheses"
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

vcpkg_cmake_config_fixup(PACKAGE_NAME cudf CONFIG_PATH lib/cmake/cudf
                         DO_NOT_DELETE_PARENT_CONFIG_PATH)
vcpkg_cmake_config_fixup(PACKAGE_NAME rtcx CONFIG_PATH lib/cmake/rtcx)

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

# cudf 26.06 exports the bare target name `nanoarrow_static` in its link
# interface, but the vcpkg nanoarrow port provides the namespaced imported
# target `nanoarrow::nanoarrow_static`. Without this, consumers get an
# unresolved `-lnanoarrow_static` and the link fails. Point it at the namespaced
# target (carries IMPORTED_LOCATION for libnanoarrow_static.a).
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/cudf/cudf-targets.cmake"
  "\$<LINK_ONLY:nanoarrow_static>" "\$<LINK_ONLY:nanoarrow::nanoarrow_static>")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
