vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

vcpkg_from_github(
  OUT_SOURCE_PATH
  SOURCE_PATH
  REPO
  rapidsai/cudf
  REF
  v${VERSION}
  SHA512
  61b77822b7fafdd1113ebb3eaf00010123b41052c5bc8ea53704ad21e1111d85ac44619afc9f3bb85e5aa1c1c3c29b8b5438b62906bb93a860b6e08181c70cf7
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_CMAKE_PATH
  REPO
  rapidsai/rapids-cmake
  REF
  v26.10.00
  SHA512
  c5c2be8672faae3990713a990b8e0d4e1082307c9dfd5857805bd543243daf0ec2c1fe41bbdb1736d5ea4e4e9fa261de44b2e8b885d812ce822814251d9f65d2
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RAPIDS_LOGGER_PATH
  REPO
  rapidsai/rapids-logger
  REF
  v0.3.0
  SHA512
  0952bcd7c6579b0cc613e848c1d695466942d3a09d3f98af4c8fed97d23f5e7ad86d73603bfc58b55a4f33461b4e567716d0a70957935fefb896907bdf1074d1
  HEAD_REF
  main)

vcpkg_from_github(
  OUT_SOURCE_PATH
  RTCX_PATH
  REPO
  rapidsai/librtcx
  REF
  a9f63f8cdd4b0b41a2d88a9f705576a61b4222ec
  SHA512
  e6f27dda72aa183be9ed636926e01fdd9740162f24f4ad5bcf2ce7a17d48cde0c554c24cbbd0ad351845c2253d4d82e2333bbbc5e57d346772ba64e52789a998
  HEAD_REF
  main)

# librtcx's embed helper uses std::variant without including its header.
vcpkg_replace_string("${RTCX_PATH}/include/rtcx/embed.hpp"
                     "#include <utility>\n#include <vector>"
                     "#include <utility>\n#include <variant>\n#include <vector>")

# Use the static CUDA redistributables supplied by the overlay ports.
vcpkg_replace_string(
  "${RTCX_PATH}/CMakeLists.txt"
  "CUDA::nvrtc_static CUDA::nvrtc_builtins_static" "nvrtc::nvrtc_static")
vcpkg_replace_string("${RTCX_PATH}/CMakeLists.txt" "CUDA::nvJitLink_static"
                     "nvjitlink::nvjitlink_static")
vcpkg_replace_string(
  "${RTCX_PATH}/CMakeLists.txt"
  "add_library(rtcx STATIC\n            src/hash.cpp"
  [[rapids_find_package(nvrtc REQUIRED CONFIG BUILD_EXPORT_SET rtcx-exports INSTALL_EXPORT_SET rtcx-exports)
rapids_find_package(nvjitlink REQUIRED CONFIG BUILD_EXPORT_SET rtcx-exports INSTALL_EXPORT_SET rtcx-exports)
add_library(rtcx STATIC
            src/hash.cpp]])

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
  8ed532dc0ba3ba664d1d41027269908ddcde04c8
  SHA512
  dade9af000bb490b053288312ab60e24ac1e1ba07bd4bfdde84beca98d19e20538d5b3dde285bc5feb7c8d208791aff018b8428d2b3e493721c352b6b83f127b
  HEAD_REF
  dev)

# Patch get_zstd.cmake to remove forced download - we provide zstd source via
# CPM variable
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_zstd.cmake"
  "set(CPM_DOWNLOAD_zstd ON)"
  "# CPM_DOWNLOAD_zstd removed - using CPM_zstd_SOURCE instead")

# cuDF only needs DLPack headers. Use the installed vcpkg package directly;
# its nested include path does not match rapids-cmake's generated module probe.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_dlpack.cmake"
  "find_and_configure_dlpack(\${CUDF_MIN_VERSION_dlpack} \${CUDF_EXCLUDE_DEPS_FROM_ALL})"
  "set(DLPACK_INCLUDE_DIR \"${CURRENT_INSTALLED_DIR}/include\")")

# The 26.10 AST header uses std::variant without including its header.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/include/cudf/ast/expressions.hpp"
  "#include <vector>"
  "#include <variant>\n#include <vector>")

# cuDF asks CPM to download xxhash unconditionally. Use the installed vcpkg
# package and its generated find module instead.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_xxhash.cmake"
  "set(CPM_DOWNLOAD_xxhash ON)"
  "# Use the vcpkg xxhash package")
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/cmake/thirdparty/get_xxhash.cmake"
  "if(xxhash_ADDED AND NOT TARGET xxhash)"
  "if(NOT TARGET xxhash)\n  add_library(xxhash INTERFACE IMPORTED)\n  set_target_properties(xxhash PROPERTIES INTERFACE_INCLUDE_DIRECTORIES \"${CURRENT_INSTALLED_DIR}/include\")\nendif()\nif(xxhash_ADDED AND NOT TARGET xxhash)")

# Preserve explicit static NVRTC/NVJitLink settings without forcing every
# cuDF dependency to be downloaded and rebuilt through CPM.
vcpkg_replace_string(
  "${SOURCE_PATH}/cpp/CMakeLists.txt"
  [[if(CUDF_BUILD_STATIC_DEPS STREQUAL "FORCE")
  set(RTCX_STATIC_LINK_NVRTC]]
  [[if(CUDF_BUILD_STATIC_DEPS STREQUAL "FORCE" OR RTCX_STATIC_LINK_NVRTC OR RTCX_STATIC_LINK_NVJITLINK)
  set(RTCX_STATIC_LINK_NVRTC]])

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
  -DCPM_LOCAL_PACKAGES_ONLY=ON
  -DCPM_rapids_logger_SOURCE=${RAPIDS_LOGGER_PATH}
  -DCPM_rtcx_SOURCE=${RTCX_PATH}
  -DRTCX_STATIC_LINK_NVRTC=ON
  -DRTCX_STATIC_LINK_NVJITLINK=ON
  -DCPM_bs_thread_pool_SOURCE=${BS_THREAD_POOL_PATH}
  -DCPM_zstd_SOURCE=${ZSTD_PATH}
  -DCPM_cuco_SOURCE=${CUCO_PATH}
  -DCMAKE_CUDA_ARCHITECTURES=RAPIDS
  -DBUILD_SHARED_LIBS=OFF
  -DCMAKE_CUDA_RUNTIME_LIBRARY=Static
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

# cuDF does not install cuco's CMake package; Sirius uses the separately
# installed header-only vcpkg port.

vcpkg_cmake_config_fixup(PACKAGE_NAME cudf CONFIG_PATH lib/cmake/cudf
                         DO_NOT_DELETE_PARENT_CONFIG_PATH)
vcpkg_cmake_config_fixup(PACKAGE_NAME rtcx CONFIG_PATH lib/cmake/rtcx)

# RTCX links zstd through its exported target, so load that target for every
# RTCX consumer rather than relying on cuDF having been found first.
vcpkg_replace_string(
  "${CURRENT_PACKAGES_DIR}/share/rtcx/rtcx-dependencies.cmake"
  "include(CMakeFindDependencyMacro)"
  "include(CMakeFindDependencyMacro)\nfind_dependency(zstd CONFIG)"
)

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
