# Compile the DuckDB substrait extension's Substrait->DuckDB reader directly
# into the sirius target so the FFI (src/sirius_ffi.cpp) can call
# `duckdb::SubstraitToDuckDB`. Only the from-substrait direction plus its
# bundled protobuf (renamespaced to `duckdb::google::protobuf`, no abseil) are
# compiled in — NOT the substrait SQL functions, to_substrait, or a second
# loadable extension — so there is no symbol collision with conda/system
# protobuf or a duplicate extension.
set(SIRIUS_SUBSTRAIT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/substrait")
file(GLOB_RECURSE SIRIUS_SUBSTRAIT_PROTOBUF_SOURCES
     "${SIRIUS_SUBSTRAIT_DIR}/third_party/google/protobuf/*.cc")
set(SIRIUS_SUBSTRAIT_SOURCES
    ${SIRIUS_SUBSTRAIT_DIR}/src/from_substrait.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/src/custom_extensions.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/src/custom_extensions_generated.cpp
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/algebra.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/extended_expression.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/plan.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/type.pb.cc
    ${SIRIUS_SUBSTRAIT_DIR}/third_party/substrait/substrait/extensions/extensions.pb.cc
    ${SIRIUS_SUBSTRAIT_PROTOBUF_SOURCES})
# The vendored/generated protobuf + substrait TUs are not warning-clean; never
# fail the build on them.
set_source_files_properties(${SIRIUS_SUBSTRAIT_SOURCES}
                            PROPERTIES COMPILE_OPTIONS "-w")
target_sources(sirius_objects PRIVATE ${SIRIUS_SUBSTRAIT_SOURCES})
