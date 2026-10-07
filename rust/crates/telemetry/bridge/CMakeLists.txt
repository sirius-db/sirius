include(FetchContent)
FetchContent_Declare(
  Corrosion
  GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
  GIT_TAG v0.6.1)
FetchContent_MakeAvailable(Corrosion)

# --- Import the Rust bridge crate (CXX bridges) ---
corrosion_import_crate(MANIFEST_PATH Cargo.toml CRATES telemetry-bridge)

# CXX-generated headers are copied to bridge/include/ by build.rs. This
# directory contains rust/cxx.h and the per-module .rs.h headers.
target_include_directories(
  telemetry_bridge
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# Corrosion's installer exports the archive location, but not its native links.
get_target_property(telemetry_native_libraries telemetry_bridge-static
                    INTERFACE_LINK_LIBRARIES)
if(VCPKG_BUILD)
  list(REMOVE_ITEM telemetry_native_libraries gcc_s)
endif()
if(telemetry_native_libraries)
  target_link_libraries(telemetry_bridge
                        INTERFACE ${telemetry_native_libraries})
endif()
