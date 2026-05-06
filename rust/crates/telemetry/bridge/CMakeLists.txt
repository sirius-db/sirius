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
target_include_directories(telemetry_bridge
                           INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
