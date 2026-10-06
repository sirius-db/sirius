// GCC 15 + conda sysroot libc may lack __isoc23_* symbols referenced from
// nvcc-compiled TUs. Provide thin wrappers around the legacy entry points.
#include <stdlib.h>

unsigned long long __isoc23_strtoull(const char* nptr, char** endptr, int base)
{
  return strtoull(nptr, endptr, base);
}

long long __isoc23_strtoll(const char* nptr, char** endptr, int base)
{
  return strtoll(nptr, endptr, base);
}
