# Fetch upstream testcontainers-native at a pinned commit, apply the Sirius
# patch (third_party/testcontainers-native.patch — container-command support,
# error propagation from tc_container_run, and a double-append fix), and build
# the Go->C bridge as a static c-archive plus the thin C wrapper into a single
# `testcontainers_native` target.
#
# We deliberately do NOT build upstream's own CMake (it pulls in demos/modules
# and a fragile `-linkshared` Go build): SOURCE_SUBDIR points at a path with no
# CMakeLists.txt, so FetchContent only downloads + patches. Used by the [s3]
# integration tests (test/cpp/utils/s3_container.*). Requires a Go toolchain
# (provided by pixi) and network access on the first configure (to clone) and
# build (to fetch modules).

include(FetchContent)
find_package(Git REQUIRED)
find_program(GO_EXECUTABLE go REQUIRED)

FetchContent_Declare(
  testcontainers_native_src
  GIT_REPOSITORY https://github.com/testcontainers/testcontainers-native.git
  GIT_TAG d6929da7c953403eca5f2cf81008c961ab5d23fc
  PATCH_COMMAND
    "${GIT_EXECUTABLE}" apply --whitespace=nowarn
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/testcontainers-native.patch"
    # Download + patch only: this subdir has no CMakeLists.txt, so MakeAvailable
    # populates (and runs PATCH_COMMAND) without add_subdirectory()'ing
    # upstream.
    SOURCE_SUBDIR _sirius_build_only)
FetchContent_MakeAvailable(testcontainers_native_src)

set(TCN_SRC "${testcontainers_native_src_SOURCE_DIR}")
set(TCN_BRIDGE_DIR "${TCN_SRC}/testcontainers-bridge")
set(TCN_OUT "${CMAKE_BINARY_DIR}/testcontainers-native")
file(MAKE_DIRECTORY "${TCN_OUT}")

# Go `c-archive` derives the generated header name from the -o basename, and
# testcontainers-c/src/container.c includes <testcontainers-bridge.h>, so the
# archive MUST be named testcontainers-bridge.a (-> testcontainers-bridge.h).
set(TCN_BRIDGE_ARCHIVE "${TCN_OUT}/testcontainers-bridge.a")
set(TCN_BRIDGE_HEADER "${TCN_OUT}/testcontainers-bridge.h")

# Keep the Go module/build caches inside the build tree (no $HOME pollution).
# -mod=readonly: download into the cache but never mutate the fetched
# go.mod/sum. -modcacherw: write the module cache user-writable — Go's default
# read-only cache makes `rm -rf build` (make clean) fail on the .gopath files.
set(TCN_GO_ENV "GOPATH=${TCN_OUT}/.gopath" "GOCACHE=${TCN_OUT}/.gocache"
               "GOFLAGS=-mod=readonly -modcacherw" "CGO_ENABLED=1")

add_custom_command(
  OUTPUT "${TCN_BRIDGE_ARCHIVE}" "${TCN_BRIDGE_HEADER}"
  DEPENDS "${TCN_BRIDGE_DIR}/testcontainers-bridge.go"
          "${TCN_BRIDGE_DIR}/go.mod" "${TCN_BRIDGE_DIR}/go.sum"
  WORKING_DIRECTORY "${TCN_BRIDGE_DIR}"
  COMMAND
    ${CMAKE_COMMAND} -E env ${TCN_GO_ENV} "${GO_EXECUTABLE}" build
    -buildmode=c-archive -o "${TCN_BRIDGE_ARCHIVE}" testcontainers-bridge.go
  COMMENT "Building patched testcontainers Go->C bridge (c-archive)"
  VERBATIM)

add_custom_target(testcontainers-bridge-build DEPENDS "${TCN_BRIDGE_ARCHIVE}"
                                                      "${TCN_BRIDGE_HEADER}")

add_library(testcontainers_native STATIC
            "${TCN_SRC}/testcontainers-c/src/container.c")
add_dependencies(testcontainers_native testcontainers-bridge-build)

target_include_directories(
  testcontainers_native
  PUBLIC "${TCN_SRC}/testcontainers-c/include"
  PRIVATE "${TCN_OUT}") # for the generated testcontainers-bridge.h

# A Go c-archive carries the Go runtime and pulls in pthread/dl.
find_package(Threads REQUIRED)
target_link_libraries(
  testcontainers_native PUBLIC "${TCN_BRIDGE_ARCHIVE}" Threads::Threads
                               ${CMAKE_DL_LIBS})
