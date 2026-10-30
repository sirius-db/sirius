/* math_functions.hpp uses _NV_RSQRT_SPECIFIER in its host section. The macro is
   defined in math_functions.h, which clang's CUDA wrapper does not include first.
   On glibc >= 2.42 the specifier is noexcept; otherwise it is empty. */
#ifndef _NV_RSQRT_SPECIFIER
#if defined(__GNUC__) && !defined(__ANDROID__) && !defined(__QNX__) && !defined(__APPLE__)
#include <features.h>
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 42)
#define _NV_RSQRT_SPECIFIER noexcept(true)
#else
#define _NV_RSQRT_SPECIFIER
#endif
#else
#define _NV_RSQRT_SPECIFIER
#endif
#endif
