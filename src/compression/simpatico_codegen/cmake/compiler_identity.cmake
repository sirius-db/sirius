# Invoked at build time against the exact imported static compiler targets. A
# shared compiler must instead be identified at runtime.
include("${CMAKE_CURRENT_LIST_DIR}/jit_header_manifest.cmake")
set(identity "")
if(DEFINED NVRTC)
  jit_file_identity("${NVRTC}" nvrtc_hash)
  jit_file_identity("${BUILTINS}" builtins_hash)
  jit_file_identity("${PTX}" ptx_hash)
  set(identity
      "static-v2:xxh3-128:nvrtc:${nvrtc_hash}:builtins:${builtins_hash}:ptx:${ptx_hash}"
  )
endif()
file(
  WRITE "${OUT}"
  "// Generated compiler identity; installation paths are deliberately absent.\n"
  "#pragma once\nnamespace codegen::jit::detail {\n"
  "inline constexpr char kStaticCompilerIdentity[] = \"${identity}\";\n}\n")
