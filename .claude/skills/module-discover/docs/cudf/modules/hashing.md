# Hashing

**Status**: UNUSED
**Path**: `cudf/hashing/`

## Summary
GPU hash function implementations (MurmurHash3, MD5, SHA-256, etc.) for computing hash values of column data. Used internally by join and partitioning modules.

## Key APIs
- `cudf::hashing::murmurhash3_x86_32()` — MurmurHash3 32-bit
- `cudf::hashing::md5()` — MD5 hash
- `cudf::hashing::sha256()` — SHA-256 hash

## Potential Relevance
Not directly needed — Sirius uses `hash_join` and `hash_partition` which internally use hashing. Could be useful if implementing custom hash-based operations.
