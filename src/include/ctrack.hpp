/*
 * Copyright 2026, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

// Minimal stub for the ctrack tracing library. Amin's Sirius IO / Prefetching
// framework expects a full ctrack implementation providing CTRACK_NAME(name)
// for scoped tracing; until that library is vendored upstream, define the
// macro as a no-op so the framework compiles. Replace this file with the real
// header (or a vcpkg / pixi port of ctrack) when one is available.
#define CTRACK_NAME(name) ((void)0)
