---
name: module-discover
description: Discover and document a dependency library or submodule — analyzes all uses within the codebase, divides the library into logical modules, identifies which modules are used, and generates LLM-consumable API documentation for each module. Use when the user wants to understand a library dependency, map its modules, or generate API reference docs for a submodule.
---

# Library Module Discovery & Documentation

You are analyzing a dependency library or submodule used by this project. Your goal is to produce a structured, LLM-consumable module map and API reference that helps agents quickly understand what the library provides and how this project uses it.

## Before Starting

Ask the user:
1. **Which library/submodule?** (e.g., `cucascade`, `duckdb`, `cudf`, `rmm`, `spdlog`, or a path to any dependency)
2. **Where is the library source?** If it's a submodule, it's already in the repo. If it's an installed dependency, ask for the include path (e.g., `$CONDA_PREFIX/include/cudf` or `$LIBCUDF_ENV_PREFIX/include/rmm`).
3. **Output location?** Default: `.claude/skills/module-discover/docs/<library_name>/`

If the user gives just a name (e.g., "cudf"), try these locations in order:
- Submodule in repo root: `./<name>/`
- Conda prefix: `$CONDA_PREFIX/include/<name>/` or `$LIBCUDF_ENV_PREFIX/include/<name>/`
- System include: `/usr/include/<name>/` or `/usr/local/include/<name>/`

## Workflow

### Phase 1: Discover Our Usage

Find every place in **our codebase** (under `src/`, `test/`, `CMakeLists.txt`, `*.cmake`) that references the target library.

**Step 1a: Find includes**
```bash
# Find all #include directives referencing the library
grep -rn '#include.*<LIBRARY_NAME/' src/ test/ --include='*.hpp' --include='*.cpp' --include='*.cu' --include='*.cuh'
grep -rn '#include.*"LIBRARY_NAME/' src/ test/ --include='*.hpp' --include='*.cpp' --include='*.cu' --include='*.cuh'
```

**Step 1b: Find API calls**
Search for namespace-qualified calls and type references:
```bash
# For namespaced libraries (e.g., cudf::, rmm::, cucascade::)
grep -rn 'NAMESPACE::' src/ test/ --include='*.hpp' --include='*.cpp' --include='*.cu' --include='*.cuh'
```

**Step 1c: Find CMake references**
```bash
grep -rn 'LIBRARY_NAME' CMakeLists.txt third_party/*.cmake extension_config.cmake
```

**Step 1d: Compile usage inventory**
Create a list of:
- Every header we include from the library
- Every function/method we call
- Every type we use (classes, enums, typedefs)
- Every macro we reference

Group these by source file to understand usage patterns.

### Phase 2: Map the Library's Module Structure

Explore the library's source/headers to identify logical modules.

**Step 2a: Survey top-level structure**
```bash
# List top-level directories and key header files
ls -la LIBRARY_PATH/
ls -la LIBRARY_PATH/include/ 2>/dev/null
find LIBRARY_PATH/include/ -maxdepth 2 -type d 2>/dev/null
```

**Step 2b: Identify modules**
A "module" is a logical grouping of related functionality. Look for:
- Top-level subdirectories under `include/` (strongest signal)
- Namespace subdivisions (e.g., `cudf::io`, `cudf::strings`, `rmm::mr`)
- Separate header groups that serve a distinct purpose
- README or docs that describe the library's architecture

**Guidelines for module count:**
- Aim for 3-8 modules for most libraries
- A very large library (cudf, duckdb) may have 10-12
- A small library (spdlog, abseil component) may have 2-4
- Each module should represent a coherent unit of functionality
- Prefer fewer, broader modules over many granular ones

**Step 2c: Classify modules**
For each module, determine:
- **USED**: Our codebase includes headers from or calls APIs in this module
- **UNUSED**: No references found in our codebase

### Phase 3: Document Used Modules (Deep)

For each USED module, produce detailed API documentation.

**Step 3a: List all public headers**
```bash
find LIBRARY_PATH/include/MODULE_PATH -name '*.hpp' -o -name '*.h' | sort
```

**Step 3b: Extract API surface**
For each public header that we use (or that's closely related to what we use), extract:
- **Classes/Structs**: Name, brief purpose, key public methods with signatures
- **Free functions**: Signature and brief description
- **Enums**: Values and meaning
- **Type aliases**: What they resolve to
- **Constants/Macros**: Name and value/purpose

Focus on:
1. APIs we actually call (highest priority — include usage examples from our code)
2. APIs in the same headers we include (medium priority — we might need them)
3. Other public APIs in the module (lower priority — available but unused)

**Step 3c: Document usage patterns**
For each used API, find 1-2 representative call sites in our codebase showing how we use it. Include file path and line number.

### Phase 4: Document Unused Modules (Light)

For each UNUSED module, produce a brief summary:
- Module name and path
- 2-3 sentence description of what it provides
- Key classes/functions (names only, no signatures)
- Potential relevance to our project (if any)

### Phase 5: Generate Output

Write the documentation in the output directory with this structure:

```
docs/<library_name>/
  README.md           — Overview: library purpose, version, module map, usage summary
  modules/
    <module_name>.md  — Per-module documentation (deep for USED, light for UNUSED)
```

## Output Format

### README.md Template

```markdown
# <Library Name> — Module Reference

**Version**: <version or git commit>
**Location**: <path to library source>
**Namespace**: <primary namespace>

## Module Map

| Module | Status | Description | Key APIs Used |
|--------|--------|-------------|---------------|
| <name> | USED   | <one-line>  | <top 3 APIs>  |
| <name> | UNUSED | <one-line>  | —             |

## Our Usage Summary

We use <N> of <M> modules. Primary integration points:
- <bullet summary of how we use the library>

## Files That Reference This Library

| Source File | Modules Used | Key APIs |
|-------------|-------------|----------|
| <path>      | <modules>   | <APIs>   |
```

### Per-Module Documentation (USED — Deep)

```markdown
# <Module Name>

**Status**: USED
**Path**: <path within library>
**Headers we include**: <list>

## Summary
<2-3 sentences: what this module does and how we use it>

## API Reference

### <Class/Function Name>

**Header**: `<header_path>`
**Signature**:
\`\`\`cpp
<full signature>
\`\`\`

**Description**: <what it does, key parameters, return value>

**Our usage**:
- `<our_file.cpp>:<line>` — <brief context of how we call it>

### <Next API...>

## APIs Available but Not Used

These APIs exist in this module but are not currently called by our codebase:

| API | Header | Brief Description |
|-----|--------|-------------------|
| <name> | <header> | <one-line> |
```

### Per-Module Documentation (UNUSED — Light)

```markdown
# <Module Name>

**Status**: UNUSED
**Path**: <path within library>

## Summary
<2-3 sentences: what this module provides>

## Key APIs
- `<ClassName>` — <one-line>
- `<function_name>()` — <one-line>

## Potential Relevance
<1-2 sentences on whether/how this could be useful to our project, or "Not applicable to our use case">
```

## Important Guidelines

1. **Be thorough in Phase 1.** Missing a usage means misclassifying a module as UNUSED.
2. **Read headers, don't guess.** Extract actual signatures from the source code.
3. **Include line references.** Every usage example should include `file:line`.
4. **Keep unused module docs brief.** Don't spend time documenting APIs we don't use in detail.
5. **Use consistent formatting.** LLMs parse markdown tables and code blocks reliably.
6. **Note version-specific APIs.** If you notice the library version matters (e.g., API changed between versions), flag it.
7. **Parallelize where possible.** Phase 1 searches and Phase 3 header reads can be parallelized across modules.
8. **For very large libraries** (100+ headers in a module), focus deep documentation on the headers we actually include, and list the rest in a summary table.

## Updating Existing Documentation

If documentation already exists for a library in the output directory:
1. Read the existing README.md to understand what was previously documented
2. Re-run Phase 1 to detect any new or removed usages
3. Update only the modules/files that changed
4. Update the README.md module map and usage summary

## Example Invocations

```
User: "Document the cucascade submodule"
→ Library: cucascade, Path: ./cucascade/, Output: .claude/skills/module-discover/docs/cucascade/

User: "Map out how we use cudf"
→ Library: cudf, Path: $CONDA_PREFIX/include/cudf/ (or $LIBCUDF_ENV_PREFIX/include/cudf/), Output: .claude/skills/module-discover/docs/cudf/

User: "What parts of rmm do we use?"
→ Library: rmm, Path: $CONDA_PREFIX/include/rmm/ (or $LIBCUDF_ENV_PREFIX/include/rmm/), Output: .claude/skills/module-discover/docs/rmm/

User: "Analyze spdlog dependency"
→ Library: spdlog, Path: (find via FetchContent in CMake), Output: .claude/skills/module-discover/docs/spdlog/
```
