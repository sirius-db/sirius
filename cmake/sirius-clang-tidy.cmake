if(SIRIUS_ENABLE_CLANG_TIDY)
  # Include after dependency setup so only subsequently created Sirius targets
  # inherit clang-tidy. Keep findings advisory while retaining compiler errors.
  find_program(SIRIUS_CLANG_TIDY clang-tidy REQUIRED)
  set(CMAKE_CXX_CLANG_TIDY "${SIRIUS_CLANG_TIDY};--warnings-as-errors=-*")
  set(CMAKE_C_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY}")
endif()
