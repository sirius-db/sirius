include_guard(GLOBAL)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
get_filename_component(_simpatico_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
add_library(simpatico_cache_identity STATIC
            "${_simpatico_root}/src/jit/cache_identity.cpp")
set_target_properties(simpatico_cache_identity
                      PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(simpatico_cache_identity PUBLIC cxx_std_20)
target_include_directories(simpatico_cache_identity
                           PUBLIC "${_simpatico_root}/src")
target_link_libraries(simpatico_cache_identity PRIVATE OpenSSL::Crypto)
