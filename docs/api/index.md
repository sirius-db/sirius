# Sirius C++ API

Sirius is a GPU-native analytics engine. This reference covers the public C++
headers under `include/sirius/`, used to embed Sirius and bind it from other
languages. Internal engine headers and third-party dependencies are outside its
scope.

## Getting started

- @ref sirius::ffi::Context "Context" owns an initialized engine and executes Substrait plans.
- @ref sirius::ffi::Fragment "Fragment" executes one fragment of a distributed query.
- @ref sirius::ffi "Factory functions" create contexts and fragments.
- @ref ffi.hpp "Public header" lists the complete embedding interface.

The API is under active development. These pages describe the headers on the
repository's default branch; they do not promise ABI stability.

## More documentation

- [Build and development guide](https://github.com/sirius-db/sirius/blob/dev/docs/DEVELOPMENT.md)
- [Engine architecture](https://github.com/sirius-db/sirius/tree/dev/docs/super-sirius)
- [Sirius on GitHub](https://github.com/sirius-db/sirius)
