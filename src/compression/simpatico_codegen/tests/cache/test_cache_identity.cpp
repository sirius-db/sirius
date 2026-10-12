#include "jit/cache_identity.hpp"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

using namespace codegen::jit::detail;

static void require(bool condition, const char* message)
{
  if (!condition) throw std::runtime_error(message);
}

int main(int argc, char** argv)
{
  try {
    // Test-only bridge for comparing CMake's xxhsum output with the runtime
    // implementation, including files larger than its streaming buffer.
    if (argc == 3 && std::string_view(argv[1]) == "--hash-file") {
      Digest digest{};
      require(file_identity(argv[2], digest), "file hashing failed");
      std::puts(hex_digest(digest).c_str());
      return 0;
    }
    require(argc == 1, "unexpected arguments");
    require(hex_digest(content_identity({})) == "99aa06d3014798d86001c324468d497f",
            "XXH3-128 empty vector");
    require(hex_digest(content_identity("abc")) == "06b05ab6733a618578af5f94892f3950",
            "XXH3-128 canonical known vector");
    require(sizeof(Digest) == 16, "128-bit binary digest");
    std::array<std::string_view, 3> options{"-std=c++20", "-arch=sm_90", "-default-device"};
    RequestView request{"kernel", "entry", "codegen_jit.cu", options};
    const auto expected = request_identity(request);
    require(expected == request_identity(request), "request stability");
    require(DigestHash{}(expected) == DigestHash{}(request_identity(request)), "hash stability");
    auto changed   = request;
    changed.source = "kerneL";
    require(expected != request_identity(changed), "source bytes");
    changed       = request;
    changed.entry = "other";
    require(expected != request_identity(changed), "entry symbol");
    changed         = request;
    changed.program = "other.cu";
    require(expected != request_identity(changed), "program name");
    options[1] = "-arch=sm_80";
    require(expected != request_identity(request), "architecture option");
    options[1] = "-arch=sm_90";
    options[0] = "-std=c++17";
    require(expected != request_identity(request), "language option");
    options[0] = "-std=c++20";
    std::swap(options[0], options[1]);
    require(expected != request_identity(request), "option ordering");
    std::swap(options[0], options[1]);
    changed         = request;
    changed.options = std::span(options).first(2);
    require(expected != request_identity(changed), "option count");
    changed        = request;
    request.source = "a|b";
    request.entry  = "c";
    changed.source = "a";
    changed.entry  = "b|c";
    require(request_identity(request) != request_identity(changed), "field boundaries");
    request.source = std::string_view("a\0b", 3);
    changed.source = "a";
    changed.entry  = request.entry;
    require(request_identity(request) != request_identity(changed), "embedded NUL boundary");

    EnvironmentView environment{
      "embedded-libcudf", "project-bytes", "cccl-bytes", "compiler-build-1", 13030, 13030};
    const auto env = environment_identity(environment);
    for (int component = 0; component < 6; ++component) {
      auto variant = environment;
      switch (component) {
        case 0: variant.provider = "nvrtc-bundled"; break;
        case 1: variant.project_headers = "project-byteS"; break;
        case 2: variant.cccl_headers = "cccl-byteS"; break;
        case 3: variant.compiler = "compiler-build-2"; break;
        case 4: ++variant.cuda_runtime; break;
        case 5: ++variant.driver; break;
      }
      require(env != environment_identity(variant), "environment component omitted");
      require(CompilationIdentity{env, expected}.relative_path() !=
                CompilationIdentity{environment_identity(variant), expected}.relative_path(),
              "persistent path omitted environment");
    }
    require(CompilationIdentity{env, expected}.relative_path() ==
              "v2/" + hex_digest(env) + "/" + hex_digest(expected) + ".cubin",
            "versioned path");
    require(CompilationIdentity{env, expected}.relative_path() !=
              CompilationIdentity{env, request_identity(changed)}.relative_path(),
            "persistent path omitted request");
    std::puts("cache identity: OK");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
