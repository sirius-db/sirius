# Sirius C++ API

Sirius is a GPU-native analytics engine. This reference documents its public C++
API.

## Getting started

New fallible public C++ operations report errors through
`std::expected<T, sirius::Error>`. Allocation failures may throw.
The public API therefore requires C++23 and standard-library support for
`std::expected`.

- @ref sirius::Context "Engine context" owns an initialized engine. Only one active context per process is supported; callers must enforce this restriction.
- @ref sirius::ContextConfigBuilder "Configuration builder" assembles settings from defaults or YAML.
- @ref sirius::ContextConfig "Context configuration" holds an immutable, validated configuration.
- @ref sirius::Error "Errors" describe failures returned by the API.

The API is under active development. These pages describe the headers on the
repository's default branch; they do not promise ABI stability.

## More documentation

- [Build and development guide](https://github.com/sirius-db/sirius/blob/main/docs/DEVELOPMENT.md)
- [Engine architecture](https://github.com/sirius-db/sirius/tree/main/docs/super-sirius)
- [Sirius on GitHub](https://github.com/sirius-db/sirius)
