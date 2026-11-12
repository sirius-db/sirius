vcpkg_check_linkage(ONLY_STATIC_LIBRARY)

get_filename_component(SOURCE_PATH "${CURRENT_PORT_DIR}/../../.." ABSOLUTE)

# vcpkg disables FetchContent downloads during configuration.
vcpkg_from_git(
  OUT_SOURCE_PATH CORROSION_SOURCE_PATH URL
  https://github.com/corrosion-rs/corrosion.git REF
  1499b14e4906a2890f5cee1547c8848db261753d)

# Cache the upstream commit and apply the host patches in a private checkout.
function(prepare_ducklake_source)
  find_package(Git REQUIRED)
  set(pin "417a9117f590e8141742ac0579d02f6a90162e37")
  if(DEFINED ENV{DUCKDB_DUCKLAKE_DIRECTORY})
    set(seed "$ENV{DUCKDB_DUCKLAKE_DIRECTORY}")
  else()
    set(seed "${DOWNLOADS}/ducklake-${pin}.git")
    file(MAKE_DIRECTORY "${DOWNLOADS}")
    file(
      LOCK "${seed}.lock"
      GUARD FUNCTION
      TIMEOUT 300)
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${seed}" rev-parse --verify HEAD
    OUTPUT_VARIABLE head
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
  if(NOT head STREQUAL pin)
    if(DEFINED ENV{DUCKDB_DUCKLAKE_DIRECTORY})
      message(FATAL_ERROR "DuckLake source must be at ${pin}")
    elseif(_VCPKG_NO_DOWNLOADS)
      message(FATAL_ERROR "Missing cached DuckLake commit ${pin} in ${seed}")
    endif()
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" init --bare --initial-branch=pinned "${seed}"
              COMMAND_ERROR_IS_FATAL ANY)
    execute_process(
      COMMAND
        "${GIT_EXECUTABLE}" -C "${seed}" fetch --depth=1 --no-tags
        https://github.com/duckdb/ducklake "${pin}:refs/heads/pinned"
        COMMAND_ERROR_IS_FATAL ANY)
  endif()

  set(source "${CURRENT_BUILDTREES_DIR}/src/ducklake-${TARGET_TRIPLET}")
  file(MAKE_DIRECTORY "${CURRENT_BUILDTREES_DIR}/src")
  file(
    LOCK "${source}.lock"
    GUARD PROCESS
    TIMEOUT 600)
  if(EXISTS "${source}" OR IS_SYMLINK "${source}")
    file(REAL_PATH "${source}" checkout_path)
    file(REAL_PATH "${seed}" seed_path)
    cmake_path(IS_PREFIX checkout_path "${seed_path}" NORMALIZE contains_seed)
    if(contains_seed)
      message(
        FATAL_ERROR "DuckLake source must be outside the port's build checkout")
    endif()
    file(REMOVE_RECURSE "${source}")
  endif()
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" clone --no-local --no-tags --depth=1 --
            "${seed}" "${source}" COMMAND_ERROR_IS_FATAL ANY)
  file(GLOB patches
       "${SOURCE_PATH}/duckdb/.github/patches/extensions/ducklake/*.patch")
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" apply ${patches}
                          COMMAND_ERROR_IS_FATAL ANY)
  set(ducklake_source
      "${source}"
      PARENT_SCOPE)
endfunction()
prepare_ducklake_source()

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
  "-DSIRIUS_DUCKLAKE_SOURCE_DIR=${ducklake_source}"
  "-DFETCHCONTENT_SOURCE_DIR_CORROSION=${CORROSION_SOURCE_PATH}"
  "-DCMAKE_CUDA_ARCHITECTURES=${VCPKG_CUDA_ARCHITECTURES}"
  ${sirius_launchers})
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME sirius CONFIG_PATH lib/cmake/sirius)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
     "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
file(REMOVE_RECURSE "${ducklake_source}")
