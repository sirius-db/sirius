/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#include <sirius_ffi.hpp>

int main()
{
  auto name = sirius::ffi::stream_view_name(7);
  return name && *name == "sirius_stream_7" ? 0 : 1;
}
