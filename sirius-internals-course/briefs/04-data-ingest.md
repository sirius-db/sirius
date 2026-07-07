# Module 4: Where Data Comes From (scans, splits & the IO framework)

**File to write:** `modules/04-data-ingest.html` — only a `<section class="module" id="module-4">…</section>` block. Even module → follow modules 02/05's convention: inline `style="background: var(--color-bg-warm)"` on the section (verify the token name against styles.css).

**AUDIENCE (course-wide override):** Senior systems engineer joining Sirius. No general CS/GPU explanations; tooltip Sirius-specific terms. Sharp-colleague tone. Also read `briefs/ui-upgrade-spec.md` and follow its diagram language — this module's hero diagram must use its zones/badges/hover rules.

### Teaching Arc
- **Metaphor:** A quarry feeding a crusher. Surveyors mark the seams (metadata walk → splits), the blasting schedule decides load sizes (split providers), and a conveyor with a buffer bin (IO framework: prefetch + cache) keeps the crusher (GPU decode) fed — the crusher must never wait on the quarry.
- **Opening hook:** "Module 3 hand-waved a whole subsystem: 'the scan executor produces batches.' Produced from *what*? Here's the machinery between a file path and a GPU-resident batch."
- **Key insight:** Scans are split-driven: the Scan Manager turns storage metadata into *splits* (units of scan work), providers push them through a `split_connector`, and a pluggable `gpu_ingestible` decodes each split to a cudf table — so "add a storage format" is a bounded problem, and the IO framework overlaps I/O with compute underneath all of it.
- **Why care (Alexander-specific):** Your streaming source (#836) is *an alternative data entry point* — its design explicitly mirrors this subsystem (`split_connector` → `exchange_channel`). Understand this module and Module 8's design decisions become obvious.

### Content beats (4-5 screens)
1. The pipeline from path to batch (HERO diagram, ui-upgrade-spec language, zoned):
   `Storage (parquet / .duckdb / s3:// / Iceberg)` [Storage/IO zone] → `datasource + IO framework (io_uring reactor, pinned prefetch cache, admission control)` [Storage/IO zone] → `Scan Manager: split provider → split_connector` [CPU zone] → `scan operator (gpu_ingestible decode)` [GPU zone] → `repository (idle batches)` [CPU zone edge]. Every node: `.term` hover with owner path; dirs linked (verify): `src/io/`, `src/scan_manager/`, `src/op/scan/`.
2. Splits & the connector (code↔English on the snippet below): a split = one schedulable unit of scan work (e.g. row-group batch / byte ranges). The connector's contract — close-then-drain, exception-carrying, `is_closed` = closed AND drained. Then the payoff callout: **"Memorize this contract. Your `exchange_channel` (#836) is its network twin — same EOS semantics, batches instead of splits."** (sets up Module 8).
3. Pluggable ingestion — expandable `<details>` datasource cards (4, per ui-upgrade-spec explorer pattern), each with PR receipts and verified file links:
   - **Parquet** — `src/op/scan/parquet_gpu_ingestible.cpp`; row-group pruning w/ filter pushdown (#363); multifile splits (#738).
   - **DuckDB-native GPU decode** — `src/op/scan/duckdb_native_gpu_ingestible.cpp`, `duckdb_native_decoder.cpp`; decodes `.duckdb` storage directly to cudf tables (#736 → #792).
   - **S3** — datasource foundation → SigV4 credentials → backend (#746 → #758 → #784), SQL surface (#805); routes `s3://` through the same scan manager.
   - **Iceberg** — `src/op/scan/iceberg_*.cpp`; V1/V2 scans with GPU-accelerated delete filters (#521).
   Unifying line: all four sit behind `gpu_ingestible` since the unification (#871) — one scan operator, pluggable decoders (`src/op/scan/gpu_ingestible.cpp`).
4. The IO framework underneath (cards with PR receipts): #675 (May 2026 milestone: io_uring reactor, pinned prefetching cache, buffer pool, admission control — a global-milestone-grade landing), #740 (turned on inside the scan manager — the point the stack started serving real scans), #997 (redesign: partial cache reads, cache-on-read, multi-GPU batch coalescing). Cache lineage: #340 (cache scan outputs) → #455 (skip file I/O entirely on cache hit). One sentence on why: scan throughput was gated on synchronous file I/O; the framework overlaps I/O with decode and gave S3 a place to plug in as "just another datasource".
5. Close + quiz. Hand-off: "Splits became batches sitting idle in repositories. Next: the rules those repositories play by — ports, barriers, and why data never travels by function return."

### Code Snippet (pre-extracted, use EXACTLY as-is)

File: src/include/scan_manager/split_connector.hpp (lines 60-79)
```cpp
  /// \brief Mark the connector as closed: no more splits will be pushed. Idempotent.
  ///        Wakes all waiting consumers.
  ///
  /// \param exception Optional exception captured by the producer. The first
  ///                  non-null exception passed across all close() calls is
  ///                  stored and rethrown by get_next_split() once the queue
  ///                  has been drained. Subsequent close() calls do not
  ///                  overwrite an already-stored exception.
  void close(std::exception_ptr const& exception = nullptr);

  /// \brief Pull the next split, blocking until one is available or the connector
  ///        is closed and drained.
  /// \return std::nullopt when closed and drained without error; the next split
  ///         otherwise.
  /// \throws The exception passed to close() (if any) once the queue is drained.
  std::optional<std::unique_ptr<op::operator_data>> get_next_split();

  /// \brief True iff close() has been called and the queue is drained.
  [[nodiscard]] bool is_closed() const;
```

### Interactive Elements
- [x] **Code↔English translation** — the snippet (translate the close-then-drain contract clause by clause; the exception-once semantics; why nullopt ≠ error).
- [x] **Expandable datasource cards** — the 4 `<details>` cards above (this satisfies "expandable operator diagrams" for this module).
- [x] **Hero zoned flow diagram** — per ui-upgrade-spec; hover ownership on every node.
- [x] **Quiz** — 3 questions, scenario style:
  1. "The same table is scanned twice; the second query does zero disk reads and starts decoding immediately. Which two mechanisms?" → scan-output caching (#340/#455) + the prefetching cache in the IO framework.
  2. "You're adding a new columnar file format. What do you implement, and what do you NOT touch?" → a `gpu_ingestible` decoder + a split provider; you don't touch the scan operator, task creator, or pipelines (that's the point of #871).
  3. "An S3 scan is slow and the GPU sits idle between decode bursts. Which layer owns fixing the overlap — the scan operator or the IO framework — and why?" → the IO framework (reactor + prefetcher own network/decode overlap; the operator just consumes splits).
- [x] **Glossary tooltips** — split, split provider, split_connector, gpu_ingestible, datasource, prefetch cache, admission control, io_uring, row group, scan manager, cache-on-read.

### Reference Files to Read
- `briefs/ui-upgrade-spec.md` (in this course directory) — MANDATORY, the diagram language.
- `/home/ubuntu/.claude/skills/codebase-to-course/references/interactive-elements.md` → "Code ↔ English Translation", "Multiple-Choice Quiz", "Callout Boxes", "Glossary Tooltips", "Pattern Cards"
- `/home/ubuntu/.claude/skills/codebase-to-course/references/design-system.md` → tokens
- `/home/ubuntu/.claude/skills/codebase-to-course/references/content-philosophy.md` → all (AUDIENCE override applies)
- `/home/ubuntu/.claude/skills/codebase-to-course/references/gotchas.md` → all

### Connections
- **Previous module:** Module 3 "Life of a Query" — ends by asking where the scan's splits came from; this module answers it.
- **Next module:** Module 5 "The Data Plane" — repositories, ports, barriers.
- **Tone/style notes:** teal accent; PR links `https://github.com/sirius-db/sirius/pull/<N>`; verify EVERY linked source path with ls before linking (ui-upgrade-spec rule).
