# Invoked at build time against the exact imported static compiler targets. A
# shared compiler must instead be identified at runtime.
set(identity "")
if(DEFINED NVRTC)
  file(SHA256 "${NVRTC}" nvrtc_hash)
  file(SHA256 "${BUILTINS}" builtins_hash)
  file(SHA256 "${PTX}" ptx_hash)
  set(identity
      "static-v1:nvrtc:${nvrtc_hash}:builtins:${builtins_hash}:ptx:${ptx_hash}")
endif()
file(
  WRITE "${OUT}"
  "// Generated compiler identity; installation paths are deliberately absent.\n"
  "#pragma once\nnamespace codegen::jit::detail {\n"
  "inline constexpr char kStaticCompilerIdentity[] = \"${identity}\";\n}\n")
